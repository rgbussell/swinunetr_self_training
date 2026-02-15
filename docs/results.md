# Results and Analysis

This document presents the experimental results of the self-training pipeline compared to the fully-supervised baseline. All metrics are computed on the same held-out validation set (D_val, ~97 volumes) across all experiments.

## 1. Baseline Performance

Fully-supervised SwinUNETR trained on 387 labeled volumes for 300 epochs.

| Metric | TC | WT | ET | Mean |
|--------|------|------|------|------|
| Dice | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| HD95 (mm) | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Sensitivity | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Specificity | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

**Observations:**
- [PLACEHOLDER: Note which class performs best/worst and why]
- [PLACEHOLDER: Compare to published SwinUNETR results on BraTS]

## 2. Per-Round Self-Training Results

### 2.1 Summary Table

| Round | Accepted PLs | Mean Dice | Delta vs Baseline | TC Dice | WT Dice | ET Dice |
|-------|-------------|-----------|-------------------|---------|---------|---------|
| Baseline | N/A | [PLACEHOLDER] | --- | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 0 | [PLACEHOLDER]/266 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 1 | [PLACEHOLDER]/266 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 2 | [PLACEHOLDER]/266 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 3 | [PLACEHOLDER]/266 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

### 2.2 HD95 Progression

| Round | TC HD95 | WT HD95 | ET HD95 | Mean HD95 |
|-------|---------|---------|---------|-----------|
| Baseline | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 0 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 1 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 2 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 3 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

### 2.3 Convergence Behavior

[PLACEHOLDER: Description of training dynamics per round]
- Round 0: [PLACEHOLDER: How quickly did the model converge? Early stopping?]
- Round 1: [PLACEHOLDER]
- Round 2: [PLACEHOLDER]
- Round 3: [PLACEHOLDER]

## 3. Statistical Significance

Paired Wilcoxon signed-rank tests on per-subject Dice scores, comparing each round to the baseline.

| Comparison | Mean Dice Diff | p-value | Significant (p<0.05)? |
|------------|---------------|---------|----------------------|
| Round 0 vs Baseline | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 1 vs Baseline | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 2 vs Baseline | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| Round 3 vs Baseline | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

**Effect size (Cohen's d):** [PLACEHOLDER]

**Per-class significance:**

| Class | Best Round vs Baseline | p-value |
|-------|----------------------|---------|
| TC | [PLACEHOLDER] | [PLACEHOLDER] |
| WT | [PLACEHOLDER] | [PLACEHOLDER] |
| ET | [PLACEHOLDER] | [PLACEHOLDER] |

## 4. Per-Class Analysis

### 4.1 Whole Tumor (WT)

WT is typically the easiest class to segment due to its large, contiguous appearance. It includes all tumor tissue plus surrounding edema, making it the most forgiving class for boundary errors.

- Baseline Dice: [PLACEHOLDER]
- Best self-training Dice: [PLACEHOLDER] (Round [PLACEHOLDER])
- Improvement: [PLACEHOLDER]
- [PLACEHOLDER: Analysis of where improvement came from]

### 4.2 Tumor Core (TC)

TC encompasses the solid tumor mass without surrounding edema. Segmentation difficulty is moderate.

- Baseline Dice: [PLACEHOLDER]
- Best self-training Dice: [PLACEHOLDER] (Round [PLACEHOLDER])
- Improvement: [PLACEHOLDER]
- [PLACEHOLDER: Analysis]

### 4.3 Enhancing Tumor (ET)

ET is the most challenging class --- the contrast-enhancing region is often small, irregular, and heterogeneous. This class is expected to benefit most from additional data but is also most susceptible to noisy pseudo-labels.

- Baseline Dice: [PLACEHOLDER]
- Best self-training Dice: [PLACEHOLDER] (Round [PLACEHOLDER])
- Improvement: [PLACEHOLDER]
- [PLACEHOLDER: Discuss whether strict ET thresholds helped or limited improvement]

## 5. Pseudo-Label Quality

### 5.1 Acceptance Rates

| Round | Threshold (TC/WT/ET) | Volumes Accepted | Acceptance Rate | Mean Confidence |
|-------|---------------------|-----------------|----------------|-----------------|
| 0 | 0.95/0.90/1.00 | [PLACEHOLDER] | [PLACEHOLDER]% | [PLACEHOLDER] |
| 1 | 0.91/0.86/0.96 | [PLACEHOLDER] | [PLACEHOLDER]% | [PLACEHOLDER] |
| 2 | 0.82/0.77/0.87 | [PLACEHOLDER] | [PLACEHOLDER]% | [PLACEHOLDER] |
| 3 | 0.75/0.70/0.80 | [PLACEHOLDER] | [PLACEHOLDER]% | [PLACEHOLDER] |

### 5.2 Confidence Distribution

[PLACEHOLDER: Describe how the confidence distribution shifted across rounds]
- Were later-round pseudo-labels genuinely better, or just accepted under lower thresholds?
- Was there evidence of the teacher's confidence calibration improving over rounds?

## 6. Loss Component Analysis

### 6.1 Loss Breakdown Per Round

| Round | Final Supervised Loss | Final Pseudo Loss | Final Consistency Loss | Final Total Loss |
|-------|----------------------|-------------------|----------------------|-----------------|
| 0 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 1 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 2 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 3 | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

### 6.2 Ramp-Up Behavior

[PLACEHOLDER: Did the 20-epoch ramp-up prevent initial instability?]

## 7. Ablation Discussion

The following ablations would strengthen the experimental analysis. Mark with [DONE] or [TODO] as they are completed.

### 7.1 EMA Decay Rate
- [TODO] Compare alpha = {0.99, 0.999, 0.9999}
- Expected: 0.999 provides the best tradeoff between teacher stability and adaptability

### 7.2 Threshold Schedule
- [TODO] Compare linear vs cosine vs step schedules
- Expected: Cosine provides the smoothest transition

### 7.3 Loss Component Ablation
- [TODO] Self-training without consistency loss (w_consist = 0)
- [TODO] Self-training without pseudo-label loss (w_pseudo = 0, consistency only)
- [TODO] Self-training with equal weights (w_sup = w_pseudo = w_consist = 1.0)

### 7.4 Number of Rounds
- [TODO] Compare R = {1, 2, 4, 8}
- Expected: Diminishing returns after 3-4 rounds

### 7.5 Fixed vs Curriculum Thresholding
- [TODO] Self-training with fixed threshold = 0.85 (midpoint)
- Expected: Curriculum outperforms fixed by accepting more data in later rounds

## 8. Qualitative Results

### 8.1 Improvement Cases

[PLACEHOLDER: Include 2-3 example slices where self-training clearly improved segmentation]
- Case A: [Subject ID] --- [Description of improvement]
- Case B: [Subject ID] --- [Description of improvement]

### 8.2 Failure Cases

[PLACEHOLDER: Include 1-2 cases where self-training degraded or did not help]
- Case C: [Subject ID] --- [Description of issue]

## 9. Limitations

1. **Single dataset.** All experiments use MSD Task01 BraTS. Generalization to other datasets (BraTS 2021, internal clinical data) is not evaluated.

2. **No external test set.** The unlabeled volumes (D_U) are the MSD test set, so we cannot evaluate pseudo-label quality against ground-truth. Validation is done only on the held-out labeled split.

3. **Single architecture.** We only test SwinUNETR. The self-training framework should generalize to other architectures (nnU-Net, 3D U-Net) but this is not verified.

4. **Computational cost.** Four rounds of self-training require approximately 40-60 GPU-hours, significantly more than the ~20-30 hour baseline. The cost-benefit tradeoff depends on the application.

5. **No domain shift.** The labeled and unlabeled volumes come from the same dataset and were acquired with similar protocols. Performance under domain shift (e.g., using unlabeled data from a different institution) is unknown.

## 10. Conclusions

[PLACEHOLDER: Summary of key findings]

1. [PLACEHOLDER: Did self-training improve over baseline?]
2. [PLACEHOLDER: Which class benefited most?]
3. [PLACEHOLDER: Was curriculum thresholding effective?]
4. [PLACEHOLDER: Practical recommendations]
