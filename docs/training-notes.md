# Training Notes & Known Issues

## Training Workflow

The self-training pipeline requires a **fully trained baseline model** before self-training begins. The complete workflow is:

1. **Baseline fine-tuning** (in `vit_swinunetr_segmentation`):
   ```bash
   cd /path/to/vit_swinunetr_segmentation
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     swinunetr-train --config configs/train_config.yaml
   ```
   - Trains SwinUNETR on the 484 labeled MSD BraTS volumes for 300 epochs
   - Produces `checkpoints/best_model.pt` with full training state
   - Must converge (watch for early stopping or reaching max epochs) before proceeding

2. **Self-training** (in this project):
   ```bash
   cd /path/to/swinunetr_self_training
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     st-train --config configs/self_training_config.yaml
   ```
   - Uses the baseline checkpoint as starting weights for both student and teacher
   - Runs 4 rounds of pseudo-labeling + retraining

### Resuming Baseline Training

If baseline training is interrupted, resume from the last checkpoint:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  swinunetr-train --config configs/train_config.yaml \
  --resume checkpoints/best_model.pt
```

The `best_model.pt` checkpoint contains full training state (model weights, optimizer, scheduler, scaler, epoch number, best metric).

---

## Known Issues

### 1. CUDA OOM with 128^3 roi_size (Baseline Training)

**Symptom**: Training runs fine for many epochs then crashes with `torch.OutOfMemoryError` during a training or validation step.

**Cause**: PyTorch memory fragmentation over time. The 128^3 roi_size with SwinUNETR (feature_size=48) and AMP uses ~12-13 GB of the 16 GB available on an RTX 5070 Ti. Over many epochs, memory fragmentation can cause allocation failures even when total free memory appears sufficient.

**Fix**: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching training. This enables PyTorch's expandable memory segments, which reduce fragmentation.

**Note on roi_size reduction**: Reducing roi_size to 96^3 is an alternative but reduces spatial context by ~42% and changes the bottleneck feature map dimensions (3^3 vs 4^3). This can cost ~1-2% mean Dice, particularly on large structures like WT. Prefer the expandable_segments fix for baseline training.

### 2. CUDA OOM with Self-Training (Student + Teacher)

**Symptom**: Self-training OOMs immediately or early in training at 128^3 roi_size.

**Cause**: Self-training holds both student and teacher models in GPU memory simultaneously (~2x the memory of baseline training).

**Fix**: Reduce `data.roi_size` to `[96, 96, 96]` in `self_training_config.yaml`. The expandable_segments flag alone is insufficient — the dual-model memory footprint genuinely exceeds 16 GB at 128^3.

### 3. ET Class Dice = 0.0000 (Fixed Feb 2024)

**Symptom**: Enhancing Tumor (ET) class shows Dice=0.0000 and HD95=0.00 across all training epochs, while TC (~0.79) and WT (~0.83) appear to train normally.

**Root cause**: The self-training data pipeline was using MONAI's built-in `ConvertToMultiChannelBasedOnBratsClassesd`, which expects **BraTS 2018 labels** `{0, 1, 2, 4}`. The MSD Task01 dataset uses labels `{0, 1, 2, 3}` where ET is label 3 (not 4). Since label 4 doesn't exist in MSD data, the ET channel was always empty.

**Why it was insidious**: TC and WT metrics looked healthy because the transform still produced non-empty channels for those classes (labels 1 and 2 exist in both conventions). The error was invisible unless you specifically checked ET or understood both label conventions.

**Fix**: Replaced all uses of `ConvertToMultiChannelBasedOnBratsClassesd` with `ConvertMSDBratsClassesd` from the base package (`swinunetr_seg.data.transforms`), which correctly maps MSD labels:
- TC = label 2 | label 3
- WT = label 1 | label 2 | label 3
- ET = label 3

**Lesson**: When working with medical imaging datasets, always verify label conventions match between the dataset and the transform pipeline. Run a quick sanity check: load one sample, apply the transform, and verify every output channel has non-zero voxels.

### 4. AMP dtype mismatch in Self-Training Backward Pass

**Symptom**: `RuntimeError: Found dtype Float but expected Half` during backward pass.

**Status**: Open. Occurs during self-training (not baseline). Likely related to the interaction between AMP autocast, the EMA teacher (which may not be in the same dtype context), and the three-component loss function.

**Workaround**: Under investigation. May need explicit dtype management in the loss computation or teacher inference.

---

## Hardware Notes

**GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)

| Configuration | roi_size | Memory Usage | Status |
|---|---|---|---|
| Baseline training | 128^3 | ~12-13 GB | Works with `expandable_segments:True` |
| Baseline training | 128^3 | ~12-13 GB | OOMs without `expandable_segments` after ~35 epochs |
| Self-training (student + teacher) | 128^3 | >16 GB | OOMs immediately |
| Self-training (student + teacher) | 96^3 | ~12-14 GB | Works with `expandable_segments:True` |

Always launch with:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Monitoring Training

```bash
# Live monitoring
tail -f outputs/self_training.log

# Check GPU utilization
watch -n 1 nvidia-smi

# TensorBoard (if enabled)
tensorboard --logdir outputs/tb_logs
```

### Key metrics to watch at first validation:
- **ET Dice > 0**: Confirms the label transform fix is working
- **Mean Dice trending upward**: Normal training convergence
- **HD95 decreasing**: Spatial accuracy improving
- **Loss decreasing**: Optimization is working
