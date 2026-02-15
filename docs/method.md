# Self-Training with Iterative Pseudo-Labeling for SwinUNETR Brain Tumor Segmentation

## Abstract

We present a self-training framework for improving 3D brain tumor segmentation by leveraging unlabeled MRI volumes through iterative pseudo-labeling with an Exponential Moving Average (EMA) teacher-student architecture. Starting from a SwinUNETR baseline trained on 484 labeled volumes from the Medical Segmentation Decathlon (MSD) Task01 dataset, we incorporate 266 previously unused test volumes (~55% more data) without any additional human annotation. Our approach combines curriculum-based confidence thresholding, strong data augmentation for consistency regularization, and a three-component loss function that balances supervised, pseudo-label, and consistency objectives across four iterative refinement rounds.

## 1. Introduction

Medical image segmentation is critical for clinical decision-making, treatment planning, and longitudinal monitoring of brain tumors. Deep learning models, particularly those based on the U-Net family, have achieved impressive performance on brain tumor segmentation benchmarks. However, these models are data-hungry: performance scales with the quantity and quality of labeled training data.

The annotation bottleneck is especially severe in medical imaging. Expert-quality voxel-wise segmentation of a single 3D MRI volume can take 30-60 minutes for a trained neuroradiologist, with inter-rater variability adding further complexity. For the BraTS challenge dataset, labeling 500 volumes requires roughly 250-500 hours of expert time, representing a cost of $15,000-30,000 at typical rates.

Meanwhile, unlabeled medical images are comparatively abundant. The MSD Task01 Brain Tumour dataset exemplifies this asymmetry: it provides 484 labeled training volumes alongside 266 unlabeled test volumes. The labeled subset is used for supervised training, but the unlabeled volumes --- representing a 55% increase in available data --- are typically discarded. This is wasteful. A patient who underwent an MRI scan produced useful structural information regardless of whether a radiologist drew segmentation contours on that scan.

Semi-supervised learning offers a principled path to exploit this unlabeled data. In this work, we adopt an iterative self-training approach where a teacher model generates pseudo-labels for unlabeled volumes, and a student model trains on the combination of real and pseudo-labeled data. Through curriculum thresholding that gradually relaxes confidence requirements across rounds, the pipeline progressively incorporates harder examples while mitigating the confirmation bias that plagues naive pseudo-labeling.

**Contributions.** We contribute:
1. A complete, reproducible self-training pipeline for 3D medical image segmentation built on top of the MONAI/SwinUNETR ecosystem.
2. An EMA teacher-student framework with curriculum-based per-class confidence thresholding.
3. A combined loss function integrating supervised, pseudo-label, and consistency regularization terms with configurable ramp-up schedules.
4. Comprehensive ablation-ready experimental design with per-round tracking and statistical evaluation.

## 2. Related Work

### 2.1 Semi-Supervised Learning Taxonomy

Semi-supervised learning methods fall into several broad categories:

**Pseudo-labeling / self-training.** The model generates labels for unlabeled data, then retrains on the combined labeled and pseudo-labeled set. Originally proposed by Lee (2013) and revisited as "Noisy Student" by Xie et al. (2020), this family is conceptually simple but prone to confirmation bias --- the model reinforces its own errors. Recent work addresses this through confidence filtering, curriculum scheduling, and temporal ensembling.

**Consistency regularization.** The model is encouraged to produce similar predictions for differently augmented versions of the same input. Methods include Pi-Model (Laine & Aila, 2017), Temporal Ensembling, and the Mean Teacher (Tarvainen & Valpola, 2017). The key insight is that a good model should be robust to realistic perturbations.

**Hybrid approaches.** Modern state-of-the-art methods combine pseudo-labeling with consistency regularization. MixMatch (Berthelot et al., 2019), FixMatch (Sohn et al., 2020), and FlexMatch (Zhang et al., 2021) unify these ideas. FixMatch, in particular, demonstrated that combining confidence-thresholded pseudo-labels with strong augmentation achieves remarkable performance with very few labels.

### 2.2 Mean Teacher

The Mean Teacher framework (Tarvainen & Valpola, 2017) maintains two copies of a model: a student trained via gradient descent and a teacher updated via exponential moving average (EMA) of the student's parameters. The teacher provides more stable predictions than the student at any given point during training because EMA acts as a temporal ensemble over past model states. This stability makes the teacher better suited for generating pseudo-labels and consistency targets.

Compared to pure self-training (where the model generates its own pseudo-labels and retrains), the EMA teacher approach:
- Provides smoother, more calibrated confidence estimates
- Reduces confirmation bias since the teacher lags behind the student
- Does not require a separate "teacher training" phase

### 2.3 Brain Tumor Segmentation

The BraTS challenge series (Menze et al., 2015; Bakas et al., 2018; Baid et al., 2021) has driven substantial progress in automated brain tumor segmentation. Modern approaches use 3D architectures processing multimodal MRI inputs (T1, T1ce, T2, FLAIR) to segment three overlapping tumor sub-regions:
- **Whole Tumor (WT)**: Union of all tumor tissues including edema
- **Tumor Core (TC)**: Union of necrotic core, enhancing tumor, and non-enhancing tumor
- **Enhancing Tumor (ET)**: The contrast-enhancing tumor region, the hardest to segment

SwinUNETR (Hatamizadeh et al., 2022) combines a 3D Swin Transformer encoder, pretrained via self-supervised learning on 5,050 CT volumes, with a CNN-based U-Net decoder. It achieved competitive results on BraTS 2021 and benefits from the Swin Transformer's efficient shifted-window attention mechanism for capturing long-range dependencies in volumetric data.

### 2.4 Semi-Supervised Medical Segmentation

Semi-supervised methods have been increasingly applied to medical image segmentation. Notable approaches include:

- **UA-MT** (Yu et al., 2019): Uncertainty-Aware Mean Teacher, which uses Monte Carlo dropout to estimate prediction uncertainty and weights the consistency loss accordingly.
- **CPS** (Chen et al., 2021): Cross Pseudo Supervision, where two networks with different initializations generate pseudo-labels for each other, reducing confirmation bias through diversity.
- **BCP** (Bai et al., 2023): Bidirectional Copy-Paste, which augments semi-supervised segmentation with copy-paste-based data augmentation strategies.
- **Switching Teachers** (Zeng et al., 2023): Periodically reinitializes the teacher to prevent the teacher-student pair from converging to the same erroneous predictions.

Our work follows the EMA teacher-student paradigm but adds curriculum-based per-class thresholding and iterative refinement rounds, specifically designed for the multi-class nature of brain tumor segmentation where class difficulty varies significantly.

## 3. Method

### 3.1 Problem Setup

Let D_L = {(x_i, y_i)}_{i=1}^{N_L} denote the labeled dataset of N_L = 484 paired MRI volumes and ground-truth segmentation masks, and D_U = {u_j}_{j=1}^{N_U} denote the unlabeled dataset of N_U = 266 MRI volumes without annotations.

Each input x_i is a 4-channel volume (T1, T1ce, T2, FLAIR) and each label y_i is a multi-channel binary mask with C = 3 channels corresponding to TC, WT, and ET. D_L is split into D_train (80%, ~387 volumes) and D_val (20%, ~97 volumes) for training and validation. D_val remains purely labeled throughout all experiments for fair comparison.

The goal is to learn a segmentation model f_theta that achieves higher Dice scores on D_val by leveraging D_U during training, compared to a baseline model trained only on D_train.

### 3.2 Baseline: SwinUNETR

The baseline model is a SwinUNETR with feature_size=48, taking 128x128x128 random crops from 4-channel MRI inputs and producing 3-channel segmentation logits. The Swin Transformer encoder is initialized from self-supervised pretrained weights (model_swinvit.pt), while the CNN decoder is randomly initialized. Training uses DiceCELoss, AdamW optimizer (lr=1e-4, weight_decay=1e-5), and a warmup-cosine learning rate schedule over 300 epochs.

The baseline establishes the fully-supervised performance ceiling for D_train alone. Any improvement from self-training must exceed this baseline on the same D_val.

### 3.3 EMA Teacher

We maintain two copies of the SwinUNETR model:
- **Student** f_s (parameterized by theta_s): Trained via gradient descent on the combined loss
- **Teacher** f_t (parameterized by theta_t): Updated via EMA of the student's parameters

After each training step, the teacher parameters are updated as:

    theta_t = alpha * theta_t + (1 - alpha) * theta_s

where alpha = 0.999 is the EMA decay rate. Higher values of alpha produce a more stable but slower-adapting teacher.

**Warmup phase.** For the first W = 100 training steps, the teacher is an exact copy of the student (equivalent to alpha = 0). This ensures the teacher starts from a reasonable initialization rather than gradually building up from potentially random early weights.

**Memory efficiency.** The teacher model requires no gradient computation, so its memory overhead is limited to storing the model parameters (~200 MB for SwinUNETR-48). The student uses gradient checkpointing to fit within GPU memory constraints.

### 3.4 Pseudo-Label Generation

At the beginning of each self-training round r, the current teacher generates pseudo-labels for all N_U unlabeled volumes:

1. For each unlabeled volume u_j, run sliding window inference through the teacher:
   p_j = sigma(f_t(u_j)), where sigma is the sigmoid function
2. Compute per-voxel, per-class confidence: c_j^k = max(p_j^k, 1 - p_j^k) for each class k
3. Apply per-class thresholds tau_r^k (from the curriculum scheduler) to create a binary acceptance mask:
   m_j^k = 1 if c_j^k >= tau_r^k, else 0
4. Accept the pseudo-label for volume j only if the fraction of confident voxels exceeds a minimum:
   accepted_j = 1 if (sum(m_j^k) / |m_j^k|) >= f_min for all k, where f_min = 0.3

Accepted pseudo-labels are saved as multi-channel NIfTI files alongside a manifest recording per-volume statistics (confidence distribution, acceptance rate, per-class coverage).

### 3.5 Curriculum Thresholding

Naive pseudo-labeling with a fixed confidence threshold faces a dilemma: too high a threshold accepts very few pseudo-labels (wasting data), while too low a threshold admits noisy labels that degrade performance. Curriculum thresholding resolves this by starting strict and gradually relaxing:

    tau(r) = tau_final + (tau_initial - tau_final) * 0.5 * (1 + cos(pi * r / (R - 1)))

where tau_initial = 0.95, tau_final = 0.75, r is the current round, and R = 4 is the total number of rounds.

**Per-class offsets.** Brain tumor sub-regions vary in segmentation difficulty. WT (whole tumor) is the easiest, while ET (enhancing tumor) is the hardest and most variable. We apply per-class offsets to the global threshold:
- TC: +0.00 (baseline)
- WT: -0.05 (accept more easily)
- ET: +0.05 (require higher confidence)

This means in round 0, the effective thresholds are: TC=0.95, WT=0.90, ET=1.00 (clamped). The ET threshold is intentionally very strict early on, reflecting the higher risk of noisy pseudo-labels for this challenging class.

### 3.6 Combined Loss

The total training loss combines three components:

    L_total = w_sup * L_sup + ramp(e) * w_pseudo * L_pseudo + ramp(e) * w_consist * L_consist

where:

**Supervised loss** L_sup is the standard DiceCE loss between student predictions and ground-truth labels on D_train:

    L_sup = DiceCE(f_s(x), y),  (x, y) in D_train

**Pseudo-label loss** L_pseudo is the DiceCE loss between student predictions and teacher-generated pseudo-labels on accepted unlabeled volumes, masked by the confidence mask m:

    L_pseudo = DiceCE(f_s(u) * m, p_teacher * m),  u in D_U^accepted

**Consistency loss** L_consist is the MSE between student and teacher logits on the same input (with the student receiving strongly augmented input and the teacher receiving weakly augmented input):

    L_consist = MSE(f_s(aug_strong(x)), f_t(aug_weak(x)).detach())

**Loss weights** are w_sup = 1.0, w_pseudo = 0.5, and w_consist = 0.1.

**Ramp-up schedule.** The pseudo-label and consistency losses are scaled by a linear ramp-up function:

    ramp(e) = min(e / E_ramp, 1.0)

where E_ramp = 20 epochs. This prevents the model from being overwhelmed by potentially noisy pseudo-labels at the start of each round, allowing it to first stabilize on the supervised signal.

### 3.7 Strong Augmentation

Following the FixMatch paradigm, the teacher sees weakly augmented inputs (standard normalization and cropping), while the student sees strongly augmented inputs during consistency training. Strong augmentations include:

- Gaussian noise (std=0.1)
- Random gamma correction (range [0.7, 1.5])
- Random brightness adjustment (range [0.7, 1.3])
- Random contrast adjustment (range [0.65, 1.5])
- Gaussian smoothing (sigma range [0.5, 1.15])

These augmentations are applied only to image intensities, not spatial geometry, ensuring that the teacher and student predictions are spatially aligned for the consistency loss.

The asymmetric augmentation strategy serves two purposes: (1) the teacher provides clean pseudo-labels from weakly augmented inputs, and (2) the student learns representations robust to intensity perturbations, improving generalization.

### 3.8 Iterative Refinement

The complete self-training pipeline executes R = 4 rounds:

```
For each round r in {0, 1, 2, 3}:
    1. Compute per-class thresholds tau_r from curriculum scheduler
    2. Generate pseudo-labels for all 266 unlabeled volumes using teacher
    3. Filter pseudo-labels by per-class confidence thresholds
    4. Build combined dataset: D_train + D_accepted_pseudo
    5. Train student for 100 epochs with combined loss L_total
    6. Update teacher via EMA after each training step
    7. Validate on D_val every 5 epochs
    8. Save round checkpoint (student, teacher, optimizer, metrics)
```

Each round benefits from an improved teacher (updated throughout the previous round's training), which generates higher-quality pseudo-labels. The curriculum scheduler simultaneously lowers confidence thresholds, allowing more unlabeled volumes to be incorporated. This creates a virtuous cycle: better pseudo-labels lead to a better student, which leads to a better teacher, which leads to even better pseudo-labels.

The expected progression across rounds:

| Round | Threshold (TC/WT/ET) | Expected Accepted Volumes | Expected Mean Dice |
|-------|---------------------|--------------------------|-------------------|
| Baseline | N/A | N/A | ~0.86 |
| 0 | 0.95/0.90/1.00 | ~150-200/266 | ~0.87 |
| 1 | 0.91/0.86/0.96 | ~200-230/266 | ~0.88 |
| 2 | 0.82/0.77/0.87 | ~240-260/266 | ~0.88-0.89 |
| 3 | 0.75/0.70/0.80 | ~260-266/266 | ~0.88-0.90 |

## 4. Experimental Setup

### 4.1 Dataset

We use the Medical Segmentation Decathlon (MSD) Task01 Brain Tumour dataset (Simpson et al., 2019), which contains:

| Split | Volumes | Labels | Usage |
|-------|---------|--------|-------|
| Training (D_L) | 484 | Yes | 80/20 train/val split |
| Test (D_U) | 266 | No | Pseudo-labeling target |
| **Total** | **750** | **484** | |

Each volume is a 4-channel 3D MRI with T1, T1-weighted contrast-enhanced (T1ce), T2, and FLAIR modalities, co-registered and skull-stripped. Segmentation labels encode:
- Label 0: Background
- Label 1: Necrotic/non-enhancing tumor core
- Label 2: Peritumoral edema
- Label 4: GD-enhancing tumor

These are converted to three overlapping binary channels: TC (labels 1+4), WT (labels 1+2+4), and ET (label 4).

### 4.2 Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GPU with >= 16 GB VRAM |
| RAM | 64 GB system memory |
| Storage | SSD with >= 100 GB free space |
| Framework | PyTorch 2.0+, MONAI 1.3+, CUDA 11.8+ |

### 4.3 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | | |
| Architecture | SwinUNETR | feature_size=48 |
| Input size | 128 x 128 x 128 | Random crop from full volume |
| Channels | 4 in, 3 out | T1/T1ce/T2/FLAIR to TC/WT/ET |
| Gradient checkpointing | Yes | Required for 16 GB GPUs |
| **Self-Training** | | |
| Rounds | 4 | |
| Epochs per round | 100 | |
| EMA decay | 0.999 | |
| EMA warmup steps | 100 | Direct copy during warmup |
| Initial threshold | 0.95 | |
| Final threshold | 0.75 | |
| Threshold schedule | Cosine | |
| Min confidence fraction | 0.30 | Per-volume acceptance criterion |
| **Loss** | | |
| Supervised weight | 1.0 | |
| Pseudo-label weight | 0.5 | |
| Consistency weight | 0.1 | |
| Ramp-up epochs | 20 | Linear ramp |
| **Optimization** | | |
| Optimizer | AdamW | |
| Learning rate | 5e-5 | Lower than baseline (fine-tuning) |
| Weight decay | 1e-5 | |
| Warmup epochs | 10 | Per-round warmup |
| Batch size | 1 | GPU memory constraint |
| Mixed precision | Yes | FP16 via torch.cuda.amp |
| Early stopping patience | 30 | On validation Dice |

### 4.4 Data Split and Evaluation Protocol

The 484 labeled volumes are split 80/20 into training (~387 volumes) and validation (~97 volumes) using a fixed random seed for reproducibility. The validation set D_val remains constant and uncontaminated across all rounds and experiments, enabling fair comparison between baseline and self-trained models.

Evaluation metrics:
- **Dice Similarity Coefficient (DSC)**: Overlap-based metric, reported per-class and as the mean across TC, WT, ET.
- **Hausdorff Distance 95 (HD95)**: Surface distance metric (in mm), more sensitive to outlier errors.
- **Sensitivity and Specificity**: Per-class detection rates.

Statistical significance is assessed via paired Wilcoxon signed-rank tests on per-subject Dice scores, comparing baseline to each self-training round.

## 5. Implementation Details

The pipeline is implemented in Python using PyTorch, MONAI, and the `swinunetr_seg` base package (which provides model creation, data transforms, metrics computation, and validation utilities). The self-training extension (`swinunetr_st`) adds EMA teacher management, curriculum scheduling, pseudo-label generation, combined loss computation, and the iterative training orchestration.

**Key design decisions:**
1. The self-training package imports from `swinunetr_seg` rather than duplicating code, demonstrating modular software engineering.
2. Pseudo-labels are saved as NIfTI files between rounds rather than held in memory, enabling inspection, debugging, and resumability.
3. The curriculum scheduler supports linear, cosine, and step schedules, configurable via YAML.
4. All hyperparameters are externalized in `configs/self_training_config.yaml` with validation at load time.
5. Per-round checkpoints store complete state (student, teacher, optimizer, scheduler, metrics) for full resumability.

**Code organization:**
```
src/swinunetr_st/
    data/           # Unlabeled dataset loading, pseudo-label dataset, strong augmentations
    models/         # EMA teacher wrapper, combined self-training loss
    training/       # Self-trainer loop, pseudo-label generator, curriculum scheduler
    analysis/       # Baseline vs self-training comparison, convergence analysis
    utils/          # Config loading and validation
    cli.py          # Entry points: st-train, st-pseudo-label, st-compare
```

## References

1. Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. R., & Xu, D. (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. *MICCAI BrainLes Workshop*, 272-284.

2. Simpson, A. L., Antonelli, M., Bakas, S., et al. (2019). A large annotated medical image dataset for the development and evaluation of segmentation algorithms. *arXiv preprint arXiv:1902.09063*.

3. Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised learning results. *NeurIPS*.

4. Sohn, K., Berthelot, D., Carlini, N., et al. (2020). FixMatch: Simplifying semi-supervised learning with consistency and confidence. *NeurIPS*.

5. Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML Workshop on Challenges in Representation Learning*.

6. Xie, Q., Luong, M.-T., Hovy, E., & Le, Q. V. (2020). Self-training with noisy student improves ImageNet classification. *CVPR*.

7. Zhang, B., Wang, Y., Hou, W., et al. (2021). FlexMatch: Boosting semi-supervised learning with curriculum pseudo labeling. *NeurIPS*.

8. Yu, L., Wang, S., Li, X., Fu, C.-W., & Heng, P.-A. (2019). Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation. *MICCAI*.

9. Chen, X., Yuan, Y., Zeng, G., & Wang, J. (2021). Semi-supervised semantic segmentation with cross pseudo supervision. *CVPR*.

10. Zeng, Z., et al. (2023). Switching Temporary Teachers for Semi-Supervised Semantic Segmentation. *NeurIPS*.

11. Bakas, S., et al. (2018). Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the BRATS challenge. *arXiv preprint arXiv:1811.02629*.

12. Tang, Y., Yang, D., Li, W., et al. (2022). Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis. *CVPR*.

13. MONAI Consortium (2024). MONAI: Medical Open Network for Artificial Intelligence. https://monai.io/
