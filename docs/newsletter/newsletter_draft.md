# Unlocking 55% More Training Data Without New Annotations

*Semi-supervised self-training for brain tumor segmentation*

---

## TL;DR

- We leveraged 266 unlabeled MRI volumes alongside 484 labeled ones to improve brain tumor segmentation --- without any additional expert annotation.
- An EMA teacher-student loop with curriculum-based confidence thresholding progressively incorporates pseudo-labeled data across four training rounds.
- The technique is general-purpose: any 3D segmentation task with spare unlabeled data can apply this pipeline.

---

## The Challenge

Medical image segmentation depends on expert annotations --- and those annotations are expensive. A single brain MRI volume takes a trained neuroradiologist 30-60 minutes to segment voxel-by-voxel. At typical rates, labeling 500 volumes costs $15,000-30,000 and months of calendar time.

Meanwhile, unlabeled scans are everywhere. Every clinical MRI produces a detailed 3D volume of structural data, whether or not a radiologist draws segmentation boundaries on it. In our dataset alone (the Medical Segmentation Decathlon BraTS task), 266 test volumes sit unused --- that is 55% more data than what we train on.

The question: **Can we extract useful training signal from these unlabeled scans?**

```
    Labeled Data          Unlabeled Data
    ============          ==============
    484 volumes           266 volumes
    (with masks)          (scans only)
         |                      |
         v                      v
    [Currently used]      [Currently wasted]

    What if we could use ALL 750 volumes?
```

## Our Approach: Teacher-Student Self-Training

The core idea is elegantly simple: train a model, use it to label the unlabeled data, then retrain on everything. The challenge is doing this without the model reinforcing its own mistakes.

We use an **EMA (Exponential Moving Average) teacher-student** setup:

```
                        +-----------------+
                        |  Unlabeled MRI  |
                        |   (266 scans)   |
                        +--------+--------+
                                 |
                                 v
    +-------------------+   inference   +-------------------+
    |                   | ------------> |                   |
    |  TEACHER MODEL    |              |   Pseudo-Labels   |
    |  (EMA of student) |              | (filtered by      |
    |                   |              |  confidence)       |
    +--------+----------+              +---------+----------+
             ^                                   |
             |                                   v
        EMA update                    +-------------------+
        (each step)                   | Combined Dataset  |
             |                        | Labeled + Pseudo  |
             |                        +--------+----------+
             |                                 |
             |                          train on both
             |                                 |
             |                                 v
    +--------+----------+              +-------------------+
    |                   | <----------- |                   |
    |   STUDENT MODEL   |   gradient   |   Combined Loss   |
    |  (learns from     |   updates    | Sup + Pseudo +    |
    |   everything)     |              | Consistency        |
    +-------------------+              +-------------------+
```

**How it works in plain English:**

1. Train a baseline model on the labeled data.
2. Create a "teacher" --- a smoothed copy of the model that updates slowly via exponential moving average. Think of it as the running average of every version of the model during training.
3. The teacher generates predictions on unlabeled scans. We keep only the ones where the teacher is highly confident (>95% initially).
4. The student model trains on real labels AND the teacher's confident pseudo-labels, while a consistency loss encourages the student and teacher to agree.
5. Repeat for multiple rounds, gradually lowering the confidence bar.

The teacher's stability (it is a moving average, so it smooths out noise) makes it a better labeler than the student at any given point. And by using confidence filtering, we avoid the worst pseudo-labels.

## Key Innovation: Curriculum Thresholding

Not all pseudo-labels are created equal. And not all tumor classes are equally hard to segment.

**The curriculum approach:** Start strict, then gradually relax.

```
Confidence        Round 0     Round 1     Round 2     Round 3
Threshold     |
              |
    0.95  --- * (ET)
              |     \
    0.90  --- * (TC)  \
              |   \    \
    0.85  --- |    * -- *
              |         |  \
    0.80  --- |         |   * (ET)
              |         |
    0.75  --- |         * -- * (TC,WT)
              |
    0.70  --- |              * (WT)
              |
              +------|-------|-------|-------->  Round
                     0       1       2       3

    Volumes    ~150-200   ~200-230   ~240-260   ~260-266
    accepted    /266        /266       /266       /266
```

**Why per-class thresholds?** Brain tumors have three segmentation targets with very different difficulty levels:

- **Whole Tumor (WT):** Large, diffuse --- easy to detect, so we accept WT pseudo-labels more readily.
- **Tumor Core (TC):** Moderate difficulty --- standard thresholds.
- **Enhancing Tumor (ET):** Small, irregular, highly variable --- we demand near-certainty before trusting ET pseudo-labels.

This prevents the hardest class from being poisoned by noisy pseudo-labels early in training, while the easier classes quickly benefit from extra data.

## Results

| Metric | Baseline | Best | Improvement |
|--------|----------|------|-------------|
| Mean Dice | 0.8325 | 0.8346 (Round 1) | +0.25% |
| TC Dice | 0.8189 | 0.8205 (Round 1) | +0.19% |
| WT Dice | 0.8802 | 0.8802 | ±0% |
| ET Dice | 0.7985 | 0.8069 (Round 1) | **+1.05%** |
| Mean HD95 | 12.39 mm | 9.41 mm (Round 3) | **-24.0%** |

The headline: **Dice barely moved, but boundaries got 24% sharper.**

This was a surprise. We expected moderate Dice gains (2-4%) based on the literature. Instead, volume overlap improved only marginally (+0.25%), while boundary quality — measured by Hausdorff Distance at the 95th percentile — improved dramatically.

What does a 24% HD95 reduction actually mean? In clinical terms, the worst-case boundary error across the segmentation shrank from ~12.4 mm to ~9.4 mm. For radiotherapy planning, where margins of 1-2 cm are common, shaving 3 mm off the worst boundary error can meaningfully reduce irradiation of healthy brain tissue.

The ET class (enhancing tumor, the hardest target) showed the most improvement: +1.05% Dice and -26.7% HD95. This is the class where clinical accuracy matters most — the enhancing region guides surgical resection decisions — and it's where the additional unlabeled data provided the most value.

```
Per-class HD95 reduction (Baseline → Best):
  TC:  12.9 → 10.1 mm  (-21.6%)
  WT:  14.7 → 10.8 mm  (-26.2%)
  ET:   9.5 →  7.0 mm  (-26.7%)
```

![HD95 Boundary Improvement](../figures/hd95_boundary_improvement.png)
![Dice Progression](../figures/dice_progression.png)

## What We Learned

**1. Confidence filtering is non-negotiable.** In early experiments without threshold filtering, the model's performance actually degraded from self-training. Noisy pseudo-labels on ambiguous tumor boundaries overwhelmed the genuine signal. A conservative initial threshold (0.95) was essential to prevent this.

**2. Per-class thresholds matter for heterogeneous tasks.** Brain tumor segmentation has classes with vastly different difficulty levels. Treating ET the same as WT led to noisy ET pseudo-labels that dragged down performance on the hardest class. The per-class offset system (+0.05 for ET, -0.05 for WT) was a simple but effective fix.

**3. The EMA teacher is more than a convenience.** Compared to self-training where the same model generates and learns from pseudo-labels, the EMA teacher provided measurably more stable confidence estimates and better pseudo-label quality. The temporal smoothing acts as an implicit ensemble, reducing the impact of noisy gradient steps.

## Technical Details

The full pipeline is built on [MONAI](https://monai.io/) and [SwinUNETR](https://arxiv.org/abs/2201.01266):

| Component | Details |
|-----------|---------|
| Architecture | SwinUNETR (Swin Transformer encoder + CNN decoder) |
| Encoder | Pretrained on 5,050 CT volumes via self-supervised learning |
| Input | 4-channel MRI: T1, T1ce, T2, FLAIR |
| Output | 3-class segmentation: TC, WT, ET |
| Self-training | 4 rounds x 100 epochs, EMA decay=0.999 |
| Loss | DiceCE (supervised) + DiceCE (pseudo) + MSE (consistency) |
| Thresholds | Cosine schedule: 0.95 to 0.75, with per-class offsets |

For the full methodology, including mathematical formulations: [docs/method.md](../method.md)

## Reproduce It

```bash
git clone https://github.com/rgbussell/swinunetr_self_training.git
cd swinunetr_self_training
pip install -e ".[dev]"

# Run the self-training pipeline
st-train --config configs/self_training_config.yaml
```

Prerequisites: NVIDIA GPU (>=16 GB VRAM), CUDA 11.8+, Python 3.10+. Full setup instructions in the [README](../../README.md).

## What's Next

Based on the results and current literature, here are the most promising next steps ranked by expected impact vs effort:

1. **Boundary loss function.** Add Kervadec et al.'s boundary loss alongside Dice+CE. Since our biggest win was boundary refinement (HD95), directly optimizing for boundary quality should amplify this. Published results show up to 10% HD95 improvement — and it's a drop-in loss function, ~20 lines of code.

2. **BrainSegFounder pretrained weights.** A 2024 foundation model built *on the SwinUNETR architecture* — same model, much better initialization. Trained on 41,400 UK Biobank brain MRIs. Drop-in replacement for our pretrained encoder weights. Zero code changes, potentially significant gains.

3. **Connected component post-processing.** Remove small spurious predictions with per-class size thresholds. A 2024 BraTS paper showed +14.9% ranking metric improvement from post-processing alone — and it's pure inference-time CPU work with no retraining.

4. **Test-time augmentation.** Average predictions across 8 geometric transforms (3-axis flips). Standard in BraTS competitions. MONAI has built-in support. Consistently adds 0.5-1.5% Dice and 5-15% HD95 improvement.

5. **Architecture ensemble.** Train nnU-Net or MedNeXt alongside SwinUNETR and average predictions. The BraTS 2024 winning solution used equal-weight ensemble of nnU-Net + MedNeXt + SwinUNETR. More training cost, but reliable improvement.

See `docs/research-pathway.md` for the full research roadmap with paper references.

---

*For questions, collaboration, or to discuss applying this approach to your segmentation task, reach out via [GitHub](https://github.com/rgbussell/swinunetr_self_training).*
