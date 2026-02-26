# Research Pathway: Beyond Basic Self-Training

A prioritized roadmap for improving brain tumor segmentation beyond the current SwinUNETR + self-training baseline. All recommendations are evaluated against our hardware constraint (single NVIDIA RTX 5070 Ti, 16GB VRAM).

**Current performance** (our baseline → best self-trained):
- Mean Dice: 0.8325 → 0.8346 (+0.25%)
- Mean HD95: 12.39 → 9.41 mm (-24.0%)

---

## Tier 1: Highest Impact, Lowest Effort

These improvements require minimal code changes and no additional training. Do these first.

### 1.1 BrainSegFounder Pretrained Weights

**What**: Replace our pretrained SwinUNETR encoder weights with BrainSegFounder — a 3D foundation model specifically for brain MRI segmentation, built on the same SwinUNETR architecture.

**Why**: BrainSegFounder was pretrained on 41,400 UK Biobank brain MRIs using a two-stage approach: (1) anatomical structure learning, (2) disease-specific attribute learning. It surpasses previous BraTS winning solutions in some configurations. Since it's the same architecture, it's a direct drop-in weight replacement.

**Expected gain**: Significant (the paper reports improvements over standard pretrained SwinUNETR). The better initialization should particularly help ET class where features are harder to learn.

**Effort**: Very low — download weights, change the checkpoint path.

**Compute**: Zero additional training cost (just better initialization).

**Reference**: Ren et al., "BrainSegFounder: Towards 3D Foundation Models for Neuroimage Segmentation," Medical Image Analysis, 2024. [arXiv:2406.10395](https://arxiv.org/abs/2406.10395). GitHub: [lab-smile/BrainSegFounder](https://github.com/lab-smile/BrainSegFounder)

### 1.2 Boundary Loss Function

**What**: Add boundary loss (Kervadec et al.) to the existing Dice+CE loss. Boundary loss reformulates the surface distance as a differentiable regional integral over softmax outputs, enabling direct optimization of HD-like metrics.

**How**: Combined loss: `L = alpha * DiceCE + (1 - alpha) * BoundaryLoss`, with alpha annealed from 1.0 → 0.01 during training. Requires precomputing signed distance transform maps from ground truth masks.

**Expected gain**: Up to +8% Dice and -10% HD95 improvement over Dice loss alone (Kervadec et al.). Given our results already show strong boundary improvement from self-training, adding boundary loss should compound this effect.

**Effort**: Low — ~50 lines of code for the loss function + distance transform precomputation.

**Compute**: Negligible additional cost per training step.

**Reference**: Kervadec et al., "Boundary loss for highly unbalanced segmentation," MIDL 2019. [PDF](https://openreview.net/pdf?id=S1gTA5VggE)

**Also consider**: Celaya et al., "A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation," arXiv:2302.03868, 2024 — a more recent formulation with better numerical properties.

### 1.3 Connected Component Post-Processing

**What**: Remove small spurious predictions using connected component analysis with per-class size thresholds. Also use confusion-matrix-based label correction to fix systematic label swaps.

**Expected gain**: A 2024 BraTS paper reported **+14.9% improvement in the BraTS ranking metric** for BraTS-Africa and +0.9% for adult glioma from post-processing alone. This is an extraordinary return for zero-retraining effort.

**Effort**: Very low — SciPy's `ndimage.label` + size filtering. ~30 lines of code.

**Compute**: Negligible (pure CPU, inference-time only).

**Reference**: "Improving Pre-trained Adult Glioma Segmentation Models using only Post-processing Techniques," arXiv:2512.14937, 2024.

### 1.4 Test-Time Augmentation (TTA)

**What**: Apply geometric transforms (3-axis flips = 8 configurations) at inference, average predictions, then argmax.

**Expected gain**: +0.5-1.5% Dice, -5-15% HD95. Consistent across architectures. Standard in every BraTS competition submission.

**Effort**: Very low — MONAI has built-in TTA support.

**Compute**: 8x inference time (no extra training). Fits 16GB GPU easily (sequential inference).

**Reference**: Wang et al., "Automatic Brain Tumor Segmentation using Convolutional Neural Networks with Test-Time Augmentation," arXiv:1810.07884.

---

## Tier 2: Moderate Effort, Significant Impact

These require more implementation work but offer meaningful improvements. Do these after Tier 1.

### 2.1 Uncertainty-Filtered Pseudo-Labels (MC Dropout)

**What**: Replace current confidence thresholding with Monte Carlo Dropout uncertainty estimation. Run N forward passes with dropout enabled, compute voxel-wise entropy, and mask out high-uncertainty voxels from pseudo-labels.

**Why**: Current confidence is based on softmax probabilities, which are notoriously overconfident. MC Dropout provides calibrated uncertainty that better correlates with actual errors, especially near tumor boundaries.

**Expected gain**: 1-3% Dice improvement in self-training, -5-15% HD95 from better pseudo-label quality.

**Effort**: Medium — add dropout to decoder MLP layers, implement MC sampling loop, compute voxel-wise uncertainty maps.

**Compute**: 5-10x inference for pseudo-label generation (N forward passes per volume). Fits 16GB GPU.

**Reference**: Lambert et al., "An Empirical Study on MC Dropout-Based Uncertainty-Error Correlation in 2D Brain Tumor Segmentation," arXiv:2510.15541, 2025.

### 2.2 Curriculum Pseudo-Labeling (Per-Class Dynamic Thresholds)

**What**: Replace our fixed per-class threshold offsets with FlexMatch-style dynamic thresholds that adapt based on the model's learning status per class. Classes the model finds easier get higher thresholds; hard classes (ET) keep lower thresholds.

**Why**: Our current approach uses static offsets (ET=+0.05, WT=-0.05). Dynamic thresholds would automatically adapt as the model improves, potentially accepting more ET pseudo-labels in later rounds when the model is better at segmenting them.

**Expected gain**: 1-3% Dice, particularly on imbalanced classes (ET).

**Effort**: Low-Medium — threshold tracking logic, per-class running statistics.

**Compute**: Negligible.

**Reference**: Zhang et al., "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling," NeurIPS 2021.

### 2.3 nnU-Net + SwinUNETR Ensemble

**What**: Train an nnU-Net model on the same data and average its predictions with SwinUNETR at inference time.

**Why**: nnU-Net and SwinUNETR have complementary strengths — nnU-Net excels at local features and automatic configuration, while SwinUNETR captures long-range dependencies via self-attention. Their errors are partially decorrelated, making ensembles effective. The BraTS 2024 winning solution used equal-weight ensemble of nnU-Net (0.33) + MedNeXt (0.33) + SwinUNETR (0.34).

**Expected gain**: +1-2% Dice over either model alone. Ensembles are the single most reliable improvement in BraTS competitions.

**Effort**: Medium — install nnU-Net, configure dataset, train 5-fold cross-validation, implement inference averaging.

**Compute**: ~2x total training (nnU-Net trains separately). nnU-Net is efficient and fits 16GB GPU with automatic patch size adaptation.

**Reference**: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," Nature Methods, 2021. GitHub: [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

### 2.4 MedNeXt Architecture

**What**: Replace or supplement SwinUNETR with MedNeXt — a fully ConvNeXt-based 3D encoder-decoder that has dominated recent BraTS challenges.

**Why**: MedNeXt consistently outperforms SwinUNETR on BraTS benchmarks. The Small/Base variants fit 16GB GPU. It comes from the same DKFZ group as nnU-Net and integrates with their training pipeline.

**Performance**: Average DSC 0.896 on BraTS-2024 SSA.

**Effort**: Medium — new architecture integration, can reuse most of the training pipeline.

**Compute**: Similar to SwinUNETR.

**Reference**: Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation," MICCAI 2023. [arXiv:2303.09975](https://arxiv.org/abs/2303.09975). GitHub: [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)

---

## Tier 3: Higher Effort, Competitive-Level Results

These require significant implementation effort but could push results toward competition-winning levels.

### 3.1 Mamba-Based Architecture (nnMamba or SegMamba)

**What**: Replace SwinUNETR with a Mamba-based architecture that uses State Space Models instead of self-attention for long-range dependencies.

**Why**: Linear complexity (vs quadratic for transformers). More memory-efficient — nnMamba achieves SwinUNETR-level performance with only 15.55M parameters (vs ~62M for SwinUNETR). Potentially enables 128³ roi_size for self-training within 16GB VRAM.

**Performance**:
- SegMamba: Mean Dice 87.62 ± 0.16, HD95 4.73 ± 0.22 (slightly outperforms SwinUNETR)
- nnMamba: Average Dice 90.01, outperforming nnUNet, UNETR, and ViT variants

**Effort**: Medium-High — requires Mamba library, new architecture integration.

**Compute**: Same or less than SwinUNETR due to efficiency.

**References**:
- Xing et al., "SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation," MICCAI 2024. [arXiv:2401.13560](https://arxiv.org/abs/2401.13560)
- Gong et al., "nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model," [arXiv:2402.03526](https://arxiv.org/abs/2402.03526)

### 3.2 Contrastive Learning + Self-Training

**What**: Add pseudo-label-guided contrastive loss alongside segmentation loss. Encourages similar embeddings for voxels with the same pseudo-label while repelling different labels.

**Expected gain**: 2-5% Dice in semi-supervised settings with low labeled data fraction. Less impactful when labeled data is abundant (we have 484 labeled volumes).

**Effort**: Medium-High — projection head, memory bank or in-batch negatives, pseudo-label-aware sampling.

**Compute**: ~1.5x baseline. Feasible on 16GB with careful memory management.

**Reference**: Basak et al., "Pseudo-Label Guided Contrastive Learning for Semi-Supervised Medical Image Segmentation," CVPR 2023.

### 3.3 Synthetic Tumor Augmentation (GliGAN)

**What**: Train a GAN to generate synthetic tumors and insert them into healthy brain regions. Use these synthetic volumes as additional training data.

**Why**: This was the single largest contributor to the BraTS 2024 winning solution. It produced "the fewest false negatives among all submissions" — the model learned to never miss a tumor because it saw enormous variety during training.

**Expected gain**: Significant. Won BraTS 2024 Task 1 (GLI-post) and Task 3 (MEN-RT).

**Effort**: High — train GAN, implement synthetic tumor insertion pipeline, integrate with training loop.

**Compute**: High (train GAN + segmenter). Feasible on 16GB but slow.

**Reference**: Ferreira et al., "Improved Multi-Task Brain Tumour Segmentation with Synthetic Data Augmentation," arXiv:2411.04632, 2024. Code: [GitHub](https://github.com/shadowtwin41/brats_2023_2024_solutions)

### 3.4 Multi-Architecture Ensemble with 5-Fold CV

**What**: Train 3-4 different architectures (nnU-Net, MedNeXt, SwinUNETR, SegMamba), each with 5-fold cross-validation, and ensemble all predictions.

**Expected gain**: +2-4% Dice over any single model. This is the standard competition recipe.

**Effort**: High — multiple architecture integrations, 5x training per architecture.

**Compute**: 15-20x total training (3-4 architectures × 5 folds). Feasible on 16GB but requires days of GPU time.

---

## Recommended Implementation Order

Given our hardware (16GB GPU) and current results:

```
PHASE 1: Quick Wins (1-2 sessions)
├── 1. BrainSegFounder weights → retrain baseline
├── 2. Boundary loss → add to training
├── 3. Connected component post-processing → add to inference
└── 4. Test-time augmentation → add to inference

PHASE 2: Self-Training Refinement (2-3 sessions)
├── 5. MC Dropout uncertainty filtering → improve pseudo-labels
└── 6. FlexMatch dynamic thresholds → improve curriculum

PHASE 3: Architecture Expansion (3-5 sessions)
├── 7. Train nnU-Net on same data → ensemble with SwinUNETR
└── 8. Train MedNeXt-B → add to ensemble

PHASE 4: Competition-Level (5+ sessions)
├── 9. Mamba architecture → potential 128³ self-training
├── 10. Contrastive learning → semi-supervised improvement
└── 11. Synthetic tumor augmentation → data augmentation
```

**Expected cumulative improvement** (estimates based on published results):

| After Phase | Estimated Mean Dice | Estimated Mean HD95 |
|-------------|--------------------|--------------------|
| Current | 0.835 | 9.41 mm |
| Phase 1 | 0.855-0.870 | 7-8 mm |
| Phase 2 | 0.860-0.875 | 6-7 mm |
| Phase 3 (ensemble) | 0.880-0.895 | 4-5 mm |
| Phase 4 | 0.890-0.910 | 3-4 mm |

Note: These are rough estimates. Actual gains depend on data characteristics, hyperparameter tuning, and diminishing returns as we approach the ceiling.

---

## Key References

| Paper | Year | Key Contribution | arXiv/DOI |
|-------|------|------------------|-----------|
| BrainSegFounder | 2024 | SwinUNETR foundation model for brain MRI | arXiv:2406.10395 |
| Boundary Loss | 2019 | Differentiable surface distance optimization | MIDL 2019 |
| Generalized Surface Loss | 2024 | Better HD loss formulation | arXiv:2302.03868 |
| nnU-Net v2 | 2021 | Self-configuring segmentation pipeline | Nature Methods |
| MedNeXt | 2023 | ConvNeXt for medical segmentation | arXiv:2303.09975 |
| SegMamba | 2024 | Mamba for 3D medical segmentation | arXiv:2401.13560 |
| nnMamba | 2024 | Efficient Mamba+CNN hybrid | arXiv:2402.03526 |
| BraTS 2024 Winners | 2024 | Synthetic augmentation + ensemble | arXiv:2411.04632 |
| Post-processing Only | 2024 | CC analysis for BraTS | arXiv:2512.14937 |
| MC Dropout Uncertainty | 2025 | Uncertainty for brain tumor pseudo-labels | arXiv:2510.15541 |
| FlexMatch | 2021 | Curriculum pseudo-labeling | NeurIPS 2021 |
| Pseudo-Label Contrastive | 2023 | Contrastive + semi-supervised | CVPR 2023 |
| MedSAM | 2024 | Foundation model for medical segmentation | Nature Comms |
| SAM-Med3D | 2024 | 3D volumetric SAM variant | ECCV 2024 |
