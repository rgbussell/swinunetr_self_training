# Enrichment Exercises: Debugging Silent Failures in Medical Image Segmentation

## Overview

| | |
|---|---|
| **Difficulty** | Intermediate |
| **Time** | 45-60 minutes |
| **Prerequisites** | PyTorch `state_dict`, transfer learning basics, Transformer architecture |
| **Skills practiced** | Debugging silent failures, reading framework source code, weight inspection |

## Motivation

Transfer learning is the dominant paradigm in medical image segmentation. You download a pretrained checkpoint, load it into your model, fine-tune on your dataset, and expect a significant performance boost over training from scratch.

But what happens when the loading step **appears to succeed** while **silently failing to load a large fraction of the weights**? The model trains, loss decreases, metrics improve — but performance plateaus well below what it should achieve. This is one of the most insidious bugs in deep learning because nothing crashes, no error is thrown, and the symptoms look like a normal (if disappointing) training run.

This exercise walks through a real debugging case from a SwinUNETR brain tumor segmentation project.

---

## Background: The Architecture

**SwinUNETR** (Hatamizadeh et al., 2022) combines a **Swin Transformer encoder** with a **CNN decoder** in a U-Net architecture for 3D medical image segmentation.

```
Input (4-ch MRI) --> [Swin Transformer Encoder] --> [CNN Decoder] --> Segmentation Map
                      ^^^^^^^^^^^^^^^^^^^^^^^^^
                      This part uses pretrained weights
```

The Swin Transformer encoder (`swinViT`) contains:
- **Patch embedding** layer (`patch_embed`)
- **4 transformer stages** (`layers1` through `layers4`)
- Each stage contains **transformer blocks** with:
  - Multi-head self-attention (`attn`)
  - **MLP layers** (`mlp`) with two linear projections

The pretrained checkpoint (`model_swinvit.pt`) contains weights for **only the encoder**, trained on a different dataset. We load these into `model.swinViT` and then fine-tune the full model on brain tumors.

---

## Part 1: The Symptom

You train SwinUNETR on the BraTS brain tumor dataset with pretrained encoder weights. The training log shows:

```
INFO - Pretrained encoder: loaded 134/126 keys from './pretrained/model_swinvit.pt'
WARNING - Missing keys in encoder: ['patch_embed.proj.weight',
  'layers1.0.blocks.0.mlp.linear1.weight', 'layers1.0.blocks.0.mlp.linear1.bias',
  'layers1.0.blocks.0.mlp.linear2.weight', 'layers1.0.blocks.0.mlp.linear2.bias',
  'layers1.0.blocks.1.mlp.linear1.weight', 'layers1.0.blocks.1.mlp.linear1.bias',
  ... (32 more keys) ...]
```

**Question 1:** The log says "loaded 134/126 keys" — more loaded than exist. What's wrong with this count? (Hint: look at how `loaded_keys` is calculated when `weight` dict has been preprocessed.)

**Question 2:** The `patch_embed.proj.weight` is missing. Given that the pretrained model was trained on single-channel images and our model uses 4-channel MRI input, why would this key be filtered out? Is this expected?

**Question 3:** There are 32 missing MLP keys (`linear1` and `linear2` across all transformer blocks). Is this expected for a transfer learning scenario? What information would you need to investigate further?

---

## Part 2: Investigation

Here is the weight loading function (simplified):

```python
def load_pretrained_encoder(model, weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        weight = checkpoint["state_dict"]
    else:
        weight = checkpoint

    # Strip "module." prefix from DataParallel
    weight = {k.replace("module.", ""): v for k, v in weight.items()}

    # Filter shape mismatches
    model_state = model.swinViT.state_dict()
    filtered_weight = {
        k: v for k, v in weight.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    # Load
    result = model.swinViT.load_state_dict(filtered_weight, strict=False)

    loaded_keys = len(weight) - len(result.unexpected_keys)  # BUG: wrong count
    logger.info("Loaded %d/%d keys", loaded_keys, len(model_state))

    if result.missing_keys:
        logger.warning("Missing keys: %s", result.missing_keys)
```

### Exercise: Inspect the checkpoint

Write a short script to load the checkpoint and examine its keys:

```python
import torch

checkpoint = torch.load("pretrained/model_swinvit.pt", map_location="cpu", weights_only=True)

# Strip module. prefix
weights = {k.replace("module.", ""): v for k, v in checkpoint.items()}

# Find all MLP-related keys
mlp_keys = sorted(k for k in weights if "mlp" in k)
print("Checkpoint MLP keys:")
for k in mlp_keys[:8]:
    print(f"  {k}: shape {weights[k].shape}")
```

**Question 4:** Run this (or predict the output). What naming convention does the checkpoint use for MLP layers?

Now inspect the model:

```python
from monai.networks.nets import SwinUNETR

model = SwinUNETR(in_channels=4, out_channels=3, feature_size=48)
model_mlp_keys = sorted(k for k in model.swinViT.state_dict() if "mlp" in k)
print("\nModel MLP keys:")
for k in model_mlp_keys[:8]:
    print(f"  {k}: shape {model.swinViT.state_dict()[k].shape}")
```

**Question 5:** Compare the two sets of keys. What is the naming mismatch? How many parameters are affected?

---

## Part 3: Root Cause Analysis

The mismatch stems from **version differences in naming conventions**:

| Component | Checkpoint (older convention) | MONAI Model (current convention) |
|-----------|------------------------------|----------------------------------|
| MLP expansion layer | `mlp.fc1.weight` | `mlp.linear1.weight` |
| MLP expansion bias | `mlp.fc1.bias` | `mlp.linear1.bias` |
| MLP contraction layer | `mlp.fc2.weight` | `mlp.linear2.weight` |
| MLP contraction bias | `mlp.fc2.bias` | `mlp.linear2.bias` |

This happens because:
1. The original Swin Transformer paper's reference implementation used `fc1`/`fc2`
2. MONAI's implementation adopted `linear1`/`linear2` (following PyTorch conventions)
3. The pretrained checkpoint was created with the original naming

**Question 6:** Why didn't `strict=False` in `load_state_dict` raise an error? What does `strict=False` actually do, and why is it both useful and dangerous?

**Question 7:** The shapes are identical between `fc1.weight` and `linear1.weight`. Why doesn't the shape-filtering step catch this? (Hint: the filter checks `k in model_state` — what does `k` equal at this point?)

---

## Part 4: Quantifying the Impact

The SwinUNETR-48 encoder has approximately **8.06 million parameters**:

| Component | Parameters | Pretrained? (Before Fix) | Pretrained? (After Fix) |
|-----------|-----------|-------------------------|------------------------|
| Patch embedding | ~9,264 | No (shape mismatch) | No (shape mismatch) |
| Attention layers (QKV, proj) | ~2.4M | Yes | Yes |
| Layer norms | ~50K | Yes | Yes |
| MLP fc1/linear1 layers | ~1.57M | **No (key mismatch)** | Yes |
| MLP fc2/linear2 layers | ~1.57M | **No (key mismatch)** | Yes |
| Downsample layers | ~400K | Yes | Yes |

**39% of encoder parameters** were training from random initialization instead of pretrained values.

**Question 8:** In a Transformer block, the MLP layers typically contain more parameters than the attention layers (for standard architectures with MLP ratio of 4x). Why would failing to load MLP weights have a larger impact than failing to load attention weights?

**Question 9:** If you were training with a frozen encoder for the first 50 epochs (a common strategy), what would happen to the MLP layers during this phase? Would they learn anything?

---

## Part 5: The Fix

Add key remapping between the prefix stripping and shape filtering steps:

```python
# Strip "module." prefix
weight = {k.replace("module.", ""): v for k, v in weight.items()}

# Remap fc1/fc2 keys to linear1/linear2 for MONAI compatibility
weight = {
    k.replace("mlp.fc1", "mlp.linear1").replace("mlp.fc2", "mlp.linear2"): v
    for k, v in weight.items()
}

# Filter shape mismatches
model_state = model.swinViT.state_dict()
filtered_weight = {
    k: v for k, v in weight.items()
    if k in model_state and v.shape == model_state[k].shape
}
```

After this fix: **125/126 keys loaded** (only `patch_embed.proj.weight` missing due to the expected channel-count shape mismatch).

**Question 10:** Write a more robust version of this function that automatically detects key mismatches and suggests remappings, rather than relying on hardcoded replacements. Consider: what if a future MONAI version renames other layers?

---

## Part 6: Defensive Programming Patterns

### Pattern 1: Always log what you loaded

```python
loaded = set(filtered_weight.keys()) - set(result.missing_keys)
not_loaded = set(model_state.keys()) - loaded
pct = len(loaded) / len(model_state) * 100
logger.info("Loaded %d/%d encoder keys (%.1f%%)", len(loaded), len(model_state), pct)
if pct < 90:
    logger.warning("Less than 90%% of encoder weights loaded — check key compatibility")
```

### Pattern 2: Compare key structures before loading

```python
checkpoint_keys = set(weight.keys())
model_keys = set(model_state.keys())

# Keys in checkpoint but not in model (after prefix stripping)
extra = checkpoint_keys - model_keys
# Keys in model but not in checkpoint
missing = model_keys - checkpoint_keys

if extra and missing:
    # Likely a naming mismatch — try to find systematic patterns
    for ek in sorted(extra)[:5]:
        for mk in sorted(missing)[:5]:
            if weight[ek].shape == model_state[mk].shape:
                logger.info("Possible key mapping: '%s' -> '%s' (same shape %s)", ek, mk, weight[ek].shape)
```

### Pattern 3: Assert minimum coverage

```python
MIN_PRETRAINED_COVERAGE = 0.8  # Expect at least 80% of encoder keys loaded

coverage = len(loaded_keys) / len(model_state)
if coverage < MIN_PRETRAINED_COVERAGE:
    raise ValueError(
        f"Only {coverage:.1%} of encoder keys loaded from pretrained weights. "
        f"Expected >= {MIN_PRETRAINED_COVERAGE:.0%}. Check for key naming mismatches."
    )
```

---

## Part 7: Broader Lessons

### Why this bug is common in medical imaging

1. **Pretrained weights come from research repos** with different naming conventions than production frameworks (MONAI, nnU-Net, etc.)
2. **`strict=False` is used everywhere** because some mismatch is expected (e.g., different input channels, different number of output classes)
3. **Models still train** — they just train slower and plateau lower, which looks like a hyperparameter issue
4. **Logs are noisy** — the WARNING about missing keys is easy to dismiss when you expect `patch_embed.proj.weight` to be missing

### The debugging checklist for pretrained weight loading

- [ ] Count keys: loaded vs total. Is the ratio > 90%?
- [ ] Check for systematic patterns in missing keys (all contain `mlp`, all from same layer, etc.)
- [ ] Compare shapes between extra checkpoint keys and missing model keys
- [ ] Log parameter counts, not just key counts (one key might be 1M params)
- [ ] When using `strict=False`, always inspect `missing_keys` and `unexpected_keys`
- [ ] Test with a simple forward pass and check that frozen pretrained layers produce non-zero, non-random outputs

### Connection to reproducibility

This bug also affects **reproducibility**. If two researchers use the same pretrained checkpoint with different framework versions, one might load 100% of keys while the other loads 61%. Their "baseline" results would differ significantly, and neither would know why.

---

## Solutions to Questions

<details>
<summary>Click to reveal solutions</summary>

**Q1:** `loaded_keys = len(weight) - len(result.unexpected_keys)` uses `weight` (the full checkpoint dict, 134 keys) instead of `filtered_weight` (what was actually loaded). The correct calculation should use `len(filtered_weight)`. The "134/126" number is meaningless — it's the checkpoint size minus unexpected keys, not actual matches.

**Q2:** The pretrained `patch_embed.proj.weight` has shape `[48, 1, 2, 2, 2]` (1 input channel) while the model expects `[48, 4, 2, 2, 2]` (4 MRI modalities). The shape filter correctly excludes it. This is expected and unavoidable — the patch embedding must learn to handle 4-channel input from scratch.

**Q3:** No, 32 missing MLP keys is NOT expected. In transfer learning, you expect to miss keys for layers that differ structurally (like patch_embed with different channels, or output heads with different classes). The MLP layers are architecturally identical and should transfer perfectly.

**Q4:** The checkpoint uses `mlp.fc1.weight`, `mlp.fc1.bias`, `mlp.fc2.weight`, `mlp.fc2.bias`.

**Q5:** The model uses `mlp.linear1.*` / `mlp.linear2.*`. The mismatch affects 32 keys totaling 3.14M parameters (1.57M for fc1/linear1, 1.57M for fc2/linear2).

**Q6:** `strict=False` allows missing keys (model has them, checkpoint doesn't) and unexpected keys (checkpoint has them, model doesn't) without raising errors. It's useful for partial loading (e.g., loading an encoder into a full encoder-decoder model) but dangerous because it silently ignores genuine mismatches.

**Q7:** At the filtering step, `k` is still `mlp.fc1.weight` (the checkpoint key). The check `k in model_state` fails because the model only has `mlp.linear1.weight`. The shape never gets compared because the key lookup fails first.

**Q8:** In a standard Transformer with MLP ratio 4x, the MLP contains 2 * (d * 4d) = 8d^2 parameters while attention contains 4 * d^2 = 4d^2 parameters (for QKV + projection). So MLP layers hold ~2/3 of each block's parameters. Missing MLP pretrained weights means most of each block is effectively random.

**Q9:** With a frozen encoder, the MLP layers would have `requires_grad=False`. They would remain at their random initialization for 50 epochs. When unfrozen at epoch 50, they would start from random values instead of pretrained values, requiring much more training to converge. This is worse than training from scratch because the attention layers (which ARE pretrained) are already adapted to work with specific MLP outputs.

**Q10:** See the "Pattern 2: Compare key structures" code above. A robust approach: (1) find all keys that exist in the checkpoint but not the model (and vice versa), (2) group by shape, (3) for keys with matching shapes, compute string edit distance to find systematic substitution patterns, (4) present candidates to the user or auto-apply if the pattern is unambiguous.

</details>

---

# Bonus Exercise: The Silent Label Convention Mismatch

## Overview

| | |
|---|---|
| **Difficulty** | Intermediate-Advanced |
| **Time** | 30-45 minutes |
| **Prerequisites** | BraTS dataset familiarity, MONAI transforms, multi-class segmentation |
| **Skills practiced** | Data validation, reading dataset metadata, debugging zero-metric classes |

## Motivation

You've fixed the pretrained weight loading and retrained your SwinUNETR model. After 184 epochs, TC Dice reaches 0.80 and WT Dice reaches 0.83 — but **ET Dice is stuck at exactly 0.000** across every single validation epoch. The model never produces a single correct ET voxel.

This is a different category of silent failure than the weight loading bug. The model trains, the loss decreases, two out of three classes learn well — but the third class is completely dead. No error is thrown. If you only looked at the mean Dice (0.54), you might assume the model is just underperforming. You'd need to check per-class metrics to notice that one class contributes nothing.

---

## Part 1: The Symptom

Training logs show validation metrics at select epochs:

```
Epoch   4: Dice TC=0.412 WT=0.152 ET=0.000 Mean=0.188
Epoch   9: Dice TC=0.652 WT=0.612 ET=0.000 Mean=0.421
Epoch  94: Dice TC=0.784 WT=0.821 ET=0.000 Mean=0.539
Epoch 134: Dice TC=0.797 WT=0.834 ET=0.000 Mean=0.544  (best)
Epoch 184: Early stopping. No improvement for 50 epochs.
```

TC and WT converge normally. ET is exactly 0.000 at every checkpoint — not 0.001, not fluctuating, but a hard zero from start to finish.

**Question 1:** What does a hard 0.000 Dice (never any non-zero value) tell you that a very small Dice (e.g. 0.002 fluctuating) would not? What category of bugs does this suggest?

**Question 2:** List three possible causes for a single class producing zero Dice while other classes train normally.

---

## Part 2: The Dataset

The project uses the **Medical Segmentation Decathlon (MSD) Task01_BrainTumour** dataset. The `dataset.json` file contains:

```json
{
  "labels": {
    "0": "background",
    "1": "edema",
    "2": "non-enhancing tumor",
    "3": "enhancing tumour"
  },
  "numTraining": 484,
  "numTest": 266
}
```

Meanwhile, MONAI provides a built-in transform called `ConvertToMultiChannelBasedOnBratsClassesd`. Here is its source code:

```python
class ConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core (NCR/NET),
    label 2 is the peritumoral edema (ED),
    label 4 is the GD-enhancing tumor (ET).
    """
    def __call__(self, img):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = [
            (img == 1) | (img == 4),              # TC: NCR/NET + ET
            (img == 1) | (img == 4) | (img == 2),  # WT: all tumor
            img == 4,                               # ET: enhancing only
        ]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) \
            else np.stack(result, axis=0)
```

**Question 3:** Read the `dataset.json` labels carefully and compare them to the MONAI transform. What label value does MSD use for enhancing tumor? What label value does the transform look for?

**Question 4:** What would `img == 4` evaluate to when applied to a label volume where the maximum label value is 3?

---

## Part 3: Two Datasets, Two Conventions

This bug exists because there are **two different BraTS label conventions** in wide use:

| Convention | Background | NCR/NET | Edema | Enhancing Tumor |
|-----------|-----------|---------|-------|-----------------|
| **BraTS 2018 Challenge** | 0 | 1 | 2 | **4** (label 3 skipped) |
| **MSD Task01** | 0 | 2 | 1 | **3** (contiguous) |

The BraTS 2018 convention skips label 3 for historical reasons (earlier BraTS challenges used it for a different structure that was later merged). The MSD repackaging of the same data renumbered labels to be contiguous (0, 1, 2, 3) and also **swapped the meaning of labels 1 and 2**.

MONAI's built-in transform was written for the BraTS 2018 convention. Most tutorials and papers reference this convention. But if you download from the MSD, you get the renumbered version.

**Question 5:** Beyond the ET issue (label 3 vs 4), the MSD convention also swaps labels 1 and 2 relative to BraTS 2018. In MSD, label 1 = edema and label 2 = non-enhancing tumor. In BraTS 2018, label 1 = non-enhancing tumor and label 2 = edema. Why didn't this swap cause an obvious failure in TC and WT Dice? (Hint: look at what the TC and WT channels combine.)

**Question 6:** Write a verification script that loads one label file and prints:
- All unique label values
- Voxel count per label value
- What the MONAI transform produces (channel sums)
- What a corrected transform would produce (channel sums)

```python
import nibabel as nib
import numpy as np

label_path = "./data/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz"
img = nib.load(label_path).get_fdata()

# Your code here — print unique values, counts per label,
# and compare MONAI's transform output vs corrected output
```

---

## Part 4: Why Didn't the Loss Catch This?

The training uses `DiceCELoss` with `sigmoid=True`:

```python
DiceCELoss(
    to_onehot_y=False,
    sigmoid=True,
    squared_pred=True,
    smooth_nr=0.0,
    smooth_dr=1e-6,
)
```

When the ET target channel is all zeros for every sample:

```
Dice_ET = 2 * |pred ∩ target| / (|pred| + |target| + 1e-6)
        = 2 * 0 / (|pred| + 0 + 1e-6)
        = 0
```

The gradient of this Dice loss with respect to the ET predictions pushes them toward zero — the model learns that the optimal strategy for ET is to predict nothing, because predicting anything yields a penalty from the denominator with no reward from the numerator.

**Question 7:** If you had class-weighted loss (e.g., `weight=[1.0, 1.0, 2.0]` giving ET double weight), would that fix the zero-target problem? Why or why not?

**Question 8:** The Cross-Entropy component of DiceCELoss also receives all-zero targets for ET. What does the CE loss learn from all-zero targets? Does it help or hurt?

---

## Part 5: The Fix

Replace MONAI's built-in transform with a custom one that handles MSD labels:

```python
class ConvertMSDBratsClassesd(MapTransform):
    """Convert MSD Task01 BraTS labels to multi-channel TC/WT/ET format.

    MSD label mapping:
        0 = background
        1 = edema (ED)
        2 = non-enhancing tumor (NCR/NET)
        3 = enhancing tumor (ET)

    Output channels:
        Channel 0 (TC): label 2 + label 3  (tumor core)
        Channel 1 (WT): label 1 + 2 + 3    (whole tumor)
        Channel 2 (ET): label 3             (enhancing only)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            result = [
                (img == 2) | (img == 3),               # TC
                (img == 1) | (img == 2) | (img == 3),   # WT
                img == 3,                                # ET
            ]
            if isinstance(img, torch.Tensor):
                d[key] = torch.stack(result, dim=0).float()
            else:
                d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
```

### Verification after fix

Results at epoch 4 — before and after:

| Class | Before Fix | After Fix |
|-------|-----------|-----------|
| TC | 0.412 | 0.335 |
| WT | 0.152 | 0.497 |
| ET | **0.000** | **0.367** |
| Mean | 0.188 | **0.400** |

**Question 9:** TC Dice dropped slightly after the fix (0.412 → 0.335 at epoch 4). Why? (Hint: what changed in the TC channel definition?)

**Question 10:** WT Dice improved dramatically (0.152 → 0.497). Why? (Hint: under the old transform with the wrong label mapping, what was WT actually measuring?)

---

## Part 6: Defensive Data Validation

### Pattern 1: Assert non-empty class targets

Add a validation step at the start of training that checks a few samples:

```python
def validate_label_channels(dataset, num_samples=5):
    """Check that all output label channels have non-zero voxels."""
    class_names = ["TC", "WT", "ET"]
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        label = sample["label"]
        for ch, name in enumerate(class_names):
            voxel_count = (label[ch] > 0).sum()
            if voxel_count == 0:
                logging.warning(
                    "Sample %d has zero voxels for class %s (channel %d). "
                    "Check label encoding.", i, name, ch
                )
```

### Pattern 2: Always check dataset metadata

```python
def check_label_convention(data_dir):
    """Detect whether labels use BraTS18 (0,1,2,4) or MSD (0,1,2,3) convention."""
    import nibabel as nib
    label_files = sorted(Path(data_dir).glob("labelsTr/*.nii.gz"))[:3]
    for lf in label_files:
        unique = np.unique(nib.load(lf).get_fdata().astype(int))
        if 4 in unique and 3 not in unique:
            return "brats18"    # Uses label 4 for ET, skips 3
        elif 3 in unique and 4 not in unique:
            return "msd"        # Uses contiguous labels 0-3
    raise ValueError(f"Unexpected label values: {unique}")
```

### Pattern 3: Log channel statistics during training

```python
# In the training loop, periodically log target channel sums
if step == 0 and epoch % 10 == 0:
    for ch, name in enumerate(["TC", "WT", "ET"]):
        ch_sum = labels[:, ch].sum().item()
        logger.info("Target channel %s: %.0f positive voxels in batch", name, ch_sum)
```

---

## Part 7: Broader Lessons

### The taxonomy of failures in medical image segmentation

| Failure Type | Symptom | Example |
|-------------|---------|---------|
| **Dead class** | One class Dice = 0.000 | Label convention mismatch (Exercise 2) |
| **Degraded pretrained weights** | All classes learn slowly, plateau low | Key naming mismatch (Exercise 1) |
| **Swapped classes** | Classes learn but semantics are wrong | Label 1 and 2 swapped between conventions |
| **Spatial misalignment** | All metrics poor, noisy | Wrong orientation (RAS vs LPS) or spacing |
| **Data leakage** | Validation looks too good | Same patient in train and val splits |
| **Phase-dependent OOM** | Crash only during validation | `sw_batch_size` too large for GPU (Exercise 3) |

### Why this bug is endemic in medical imaging

1. **No standard label encoding**: Different challenges (BraTS 2017/2018/2020/2023, MSD, FeTS) use different conventions for the same anatomy
2. **Convenience transforms hide assumptions**: MONAI's `ConvertToMultiChannelBasedOnBratsClassesd` is named generically but encodes a specific convention
3. **Multi-channel labels obscure the problem**: After conversion to TC/WT/ET channels, you can't easily see that the raw labels were wrong
4. **2 out of 3 classes working is deceptively reassuring**: If everything were broken, you'd investigate immediately. Partial success delays diagnosis.

### The golden rule

**Always validate your data pipeline end-to-end before training.** Load one sample, apply all transforms, visualize the input AND the label, and verify that what you see matches what you expect. Five minutes of visualization saves days of wasted GPU time.

---

## Solutions to Questions

<details>
<summary>Click to reveal solutions</summary>

**Q1:** A hard 0.000 means the model is producing **zero true positives for the entire validation set, at every epoch**. This rules out model capacity issues (which would show small, fluctuating values) and points to a data pipeline bug — either the targets are wrong, the class index mapping is wrong, or the post-processing is discarding correct predictions. When a metric is exactly zero with no variance, the problem is almost certainly deterministic (a code/data bug), not stochastic (a training issue).

**Q2:** Three possible causes: (1) Label encoding error — target channel is all zeros so there's nothing to learn. (2) Class index mismatch — model output channel 2 is being compared to the wrong target channel. (3) Post-processing error — e.g., argmax instead of sigmoid for multi-label, or wrong threshold that always zeros out a small class.

**Q3:** MSD uses label **3** for enhancing tumor. The MONAI transform checks for label **4**. Since no voxel in the MSD data has value 4, `img == 4` evaluates to all-False, producing an all-zero ET channel.

**Q4:** `img == 4` would produce an array of all `False` (or all zeros). No voxel matches because the maximum value is 3. The resulting ET target channel has zero positive voxels in every sample.

**Q5:** In the BraTS 2018 convention, TC = `(label==1) | (label==4)` where 1=NCR/NET and 4=ET. In MSD, label 1 = edema and label 2 = NCR/NET. So the old transform computed TC as `(edema) | (nothing)` = just edema. But edema is a large region that partially overlaps spatially with the true tumor core, so the model could still learn a reasonable (but semantically wrong) TC prediction. Similarly, WT = `(label==1) | (label==4) | (label==2)` = `edema | nothing | NCR/NET` — which happens to capture most of the tumor volume. The swap didn't cause a catastrophic failure because the combined regions still had substantial overlap with the correct answers, just with different semantic decompositions.

**Q6:**
```python
import nibabel as nib
import numpy as np

img = nib.load("./data/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz").get_fdata()

print("Unique labels:", np.unique(img).astype(int))
for v in range(5):
    count = np.sum(img == v)
    if count > 0:
        print(f"  Label {v}: {count:,} voxels")

# MONAI transform (BraTS 2018 convention)
tc_old = ((img == 1) | (img == 4)).sum()
wt_old = ((img == 1) | (img == 4) | (img == 2)).sum()
et_old = (img == 4).sum()
print(f"\nMONAI transform: TC={tc_old:,}  WT={wt_old:,}  ET={et_old:,}")

# Corrected transform (MSD convention)
tc_new = ((img == 2) | (img == 3)).sum()
wt_new = ((img == 1) | (img == 2) | (img == 3)).sum()
et_new = (img == 3).sum()
print(f"Fixed transform: TC={tc_new:,}  WT={wt_new:,}  ET={et_new:,}")
```

**Q7:** No. Class weighting multiplies the loss value but doesn't change the underlying problem. `2 * Dice_ET` is still `2 * 0 = 0`. The gradient is still zero with respect to the intersection term because the target is all zeros. Weighting helps when a class is rare (small but non-zero); it cannot help when a class is absent.

**Q8:** The CE component learns that the correct label for ET is 0 everywhere (since the target is all zeros). It pushes the ET logits negative (toward sigmoid output near 0). This actively **hurts** — the model is being trained to suppress ET predictions. It's not just failing to learn ET; it's actively learning to never predict ET.

**Q9:** The TC channel changed from `(label==1) | (label==4)` to `(label==2) | (label==3)`. Under the old (wrong) mapping, TC was capturing label 1 (which in MSD is edema — a larger region). Under the new (correct) mapping, TC captures the actual tumor core (labels 2+3), which is smaller and has sharper boundaries. At epoch 4, the model hasn't yet learned these finer features, so the smaller, more precise target yields lower initial Dice. This will improve rapidly as training progresses.

**Q10:** Under the old mapping, WT was `(label==1) | (label==4) | (label==2)` = `edema | nothing | NCR/NET` = labels 1+2. Under the correct mapping, WT is `labels 1+2+3` — which additionally includes the enhancing tumor (label 3). The old WT was missing a component of the whole tumor, making it an incomplete target. With the full target (all three tumor subregions), the model has a more coherent region to segment, leading to higher Dice even at early epochs.

</details>

---

---

# Bonus Exercise: CUDA Out-of-Memory During Validation Only

## Overview

| | |
|---|---|
| **Difficulty** | Beginner-Intermediate |
| **Time** | 20-30 minutes |
| **Prerequisites** | GPU memory model, sliding window inference, training vs inference memory |
| **Skills practiced** | Reading error messages, GPU memory profiling, understanding inference memory patterns |

## Motivation

Your model trains for 5 full epochs with no issues. Loss decreases, GPU utilization looks healthy. Then at the very start of epoch 5's validation, the process crashes:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.26 GiB.
GPU 0 has a total capacity of 15.45 GiB of which 4.93 GiB is free.
Including non-PyTorch memory, this process has 10.11 GiB memory in use.
Of the allocated memory 8.59 GiB is allocated by PyTorch, and
1.22 GiB is reserved by PyTorch but unallocated.
```

This is confusing. Training worked fine — why does validation use **more** memory than training?

---

## Part 1: The Symptom

The training pipeline:
- **Training step**: Forward pass on one 128x128x128 crop, backward pass, optimizer step
- **Validation step**: `sliding_window_inference` on full-resolution volumes (~240x240x155) with `sw_batch_size=4` and `overlap=0.5`

The crash happens at the first validation epoch (`val_interval=5`), inside `sliding_window_inference`:

```
File ".../validator.py", line 71, in validate
    logits = sliding_window_inference(
        images, roi_size=self.roi_size, sw_batch_size=self.sw_batch_size,
        predictor=model, overlap=self.overlap,
    )
```

**Question 1:** Training processes one 128x128x128 crop per batch. Validation processes a full ~240x240x155 volume using sliding windows. Even though validation uses `torch.no_grad()` (no activation storage for backpropagation), why might it still use more memory than training?

**Question 2:** The error says "Tried to allocate 5.26 GiB" with "4.93 GiB free." The total GPU has 15.45 GiB. Where is the other ~10.5 GiB being used? (Hint: what was happening right before validation started?)

---

## Part 2: Understanding Sliding Window Inference

`sliding_window_inference` is MONAI's standard approach for processing 3D medical images that are too large to fit in GPU memory at once. It works by:

1. Extracting overlapping patches (windows) of size `roi_size` from the full volume
2. Processing `sw_batch_size` patches **simultaneously** through the model
3. Aggregating predictions with Gaussian weighting in overlap regions

For a volume of size 240x240x155 with `roi_size=128x128x128` and `overlap=0.5`:

```
Number of windows per axis:
  x: ceil((240 - 128) / (128 * 0.5)) + 1 = ceil(112/64) + 1 = 3
  y: ceil((240 - 128) / (128 * 0.5)) + 1 = 3
  z: ceil((155 - 128) / (128 * 0.5)) + 1 = 2
Total windows: 3 * 3 * 2 = 18 patches per volume
```

With `sw_batch_size=4`, the model processes 4 patches at a time. Each patch has shape `[1, 4, 128, 128, 128]` (batch, channels, D, H, W), so each batch of 4 is `[4, 4, 128, 128, 128]`.

**Question 3:** During training, the model sees input shape `[1, 4, 128, 128, 128]` (batch_size=1, 4 channels, one crop). During validation with `sw_batch_size=4`, the effective input is `[4, 4, 128, 128, 128]`. How much more memory does the forward pass alone require? (Assume memory scales linearly with batch size for the forward pass.)

**Question 4:** The model uses gradient checkpointing (`use_checkpoint=True`) during training to reduce activation memory. Does gradient checkpointing help during validation? Why or why not?

---

## Part 3: Memory Budget Analysis

Let's work through the GPU memory budget on a 16GB (15.45 GiB usable) card:

**Permanent allocations:**
- Model weights (SwinUNETR-48): ~62M params x 4 bytes = ~248 MB (fp32) or ~124 MB (fp16 with AMP)
- OS/display server overhead: ~500 MB (Xorg, GNOME shell)

**Training step memory:**
- Input tensor `[1, 4, 128, 128, 128]`: 32 MB
- Activations for backprop: ~4-6 GB (with gradient checkpointing)
- Gradients: ~248 MB
- Optimizer state (AdamW: 2 momentum buffers): ~496 MB
- **Total: ~6-8 GB** — fits in 15 GiB

**Validation step memory (sw_batch_size=4):**
- Input tensor `[4, 4, 128, 128, 128]`: 128 MB
- Forward pass activations (no checkpointing benefit, 4x batch): ~8-10 GB
- Output accumulation buffer (full volume size): ~100 MB
- Count/weight map for overlap averaging: ~100 MB
- Training leftovers still in GPU cache: optimizer state, gradient buffers
- **Total: ~10-12 GB** — borderline on 15 GiB, depends on fragmentation

**Question 5:** PyTorch's CUDA memory allocator caches freed memory blocks for reuse rather than returning them to the OS immediately. After a training epoch, optimizer states, gradients, and activations are "freed" but the memory is still held in the cache. How does this interact with the validation memory request? What PyTorch function could help here?

**Question 6:** The error message mentions "1.22 GiB is reserved by PyTorch but unallocated." This is the memory fragmentation problem. The allocator has 1.22 GiB in its cache, but it's in fragments too small for the 5.26 GiB contiguous allocation needed. What environment variable does the error message suggest to help with this?

---

## Part 4: The Fix

Three changes, from most impactful to least:

### Fix 1: Reduce `sw_batch_size` from 4 to 2

```python
# Before
self.validator = Validator(roi_size=roi_size)  # default sw_batch_size=4

# After
self.validator = Validator(roi_size=roi_size, sw_batch_size=2)
```

This halves the peak memory during sliding window inference. Instead of processing 4 patches simultaneously, it processes 2. Validation takes ~2x longer per volume, but fits in memory.

### Fix 2: Clear GPU cache before validation

```python
# In the training loop, before calling validate:
if (epoch + 1) % self.val_interval == 0:
    torch.cuda.empty_cache()  # Return cached blocks to GPU
    metrics = self.validator.validate(...)
```

`torch.cuda.empty_cache()` forces PyTorch to release all cached memory blocks back to CUDA, reducing fragmentation and making large contiguous allocations more likely to succeed.

### Fix 3: Enable expandable segments

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py
```

This tells PyTorch's memory allocator to use expandable segments instead of fixed-size blocks, reducing fragmentation for workloads with varying allocation patterns (like switching between training and validation).

### Results after fix

GPU memory during validation: **10.2 GiB / 15.4 GiB** (stable, with ~5 GiB headroom).

**Question 7:** Why is reducing `sw_batch_size` from 4 to 2 preferable to reducing `roi_size` from 128 to 96? Consider the impact on both memory and segmentation quality.

**Question 8:** Another option would be to move validation to CPU. Why is this almost never done in practice for 3D medical image segmentation?

---

## Part 5: Why Training and Validation Have Different Memory Profiles

This is a fundamental concept that trips up many practitioners:

| Aspect | Training | Validation |
|--------|----------|------------|
| **Input size** | Fixed ROI crop (128^3) | Full volume (~240x240x155) via sliding window |
| **Effective batch** | `batch_size` (1) | `sw_batch_size` (4) patches at once |
| **Activations stored** | Yes (for backprop), reduced by checkpointing | Forward only, but no checkpointing benefit |
| **Gradients** | Allocated (~248 MB) | None (`torch.no_grad()`) |
| **Optimizer state** | Allocated (~496 MB for AdamW) | Still in memory from training |
| **Output buffers** | Just loss scalar | Full-volume prediction + weight map |

The counterintuitive result: even though validation has no backward pass, it can use **more peak memory** than training because:

1. `sw_batch_size > train_batch_size` processes more patches simultaneously
2. Gradient checkpointing only saves memory during backprop — it has no effect on inference
3. Cached memory from training isn't released until explicitly cleared

**Question 9:** A colleague suggests using `model.half()` (fp16) during validation to halve memory usage. Would this work with MONAI's `sliding_window_inference`? What risks does it introduce for segmentation accuracy?

**Question 10:** You're designing a training pipeline that needs to run on both 16GB and 8GB GPUs. Instead of hardcoding `sw_batch_size`, write a function that dynamically selects the largest safe `sw_batch_size` based on available GPU memory.

---

## Part 6: Defensive Patterns for GPU Memory

### Pattern 1: Pre-flight memory check

```python
def check_validation_memory(model, roi_size, sw_batch_size, device):
    """Estimate peak memory for validation and warn if tight."""
    # Dry run with one batch to measure actual usage
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(sw_batch_size, 4, *roi_size, device=device)
    with torch.no_grad():
        _ = model(dummy)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    free_needed = peak * 1.2  # 20% headroom
    if free_needed > total * 0.8:
        logging.warning(
            "Validation may OOM: estimated peak %.1f GiB on %.1f GiB GPU. "
            "Consider reducing sw_batch_size from %d.", peak, total, sw_batch_size
        )
    del dummy
    torch.cuda.empty_cache()
```

### Pattern 2: Automatic sw_batch_size selection

```python
def safe_sw_batch_size(model, roi_size, in_channels, device, headroom=0.2):
    """Find largest sw_batch_size that fits in GPU memory."""
    total = torch.cuda.get_device_properties(device).total_memory
    for bs in [4, 2, 1]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            dummy = torch.randn(bs, in_channels, *roi_size, device=device)
            with torch.no_grad():
                _ = model(dummy)
            peak = torch.cuda.max_memory_allocated(device)
            del dummy
            torch.cuda.empty_cache()
            if peak < total * (1 - headroom):
                logging.info("Selected sw_batch_size=%d (peak %.1f GiB)", bs, peak / 1024**3)
                return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    return 1  # Minimum fallback
```

### Pattern 3: Log GPU memory at phase transitions

```python
def log_gpu_memory(tag, device):
    alloc = torch.cuda.memory_allocated(device) / (1024**3)
    cached = torch.cuda.memory_reserved(device) / (1024**3)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    logging.info("[%s] GPU: %.1f GiB alloc, %.1f GiB cached, %.1f GiB peak", tag, alloc, cached, peak)

# Usage in training loop:
log_gpu_memory("pre-validation", self.device)
torch.cuda.empty_cache()
log_gpu_memory("post-cache-clear", self.device)
metrics = self.validator.validate(...)
log_gpu_memory("post-validation", self.device)
```

---

## Part 7: Broader Lessons

### The memory phases of a training pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Training   │───>│  Transition  │───>│  Validation  │
│              │    │              │    │              │
│ Activations  │    │ Cache still  │    │ SW inference │
│ + Gradients  │    │ holding old  │    │ with larger  │
│ + Optimizer  │    │ allocations  │    │ batch size   │
│              │    │              │    │              │
│ Peak: ~8 GB  │    │ Fragmented   │    │ Peak: ~12 GB │
└─────────────┘    └──────────────┘    └─────────────┘
                         │
                    torch.cuda.empty_cache()
                    reduces fragmentation here
```

### Common GPU memory pitfalls in medical imaging

| Pitfall | What happens | Fix |
|---------|-------------|-----|
| `sw_batch_size` too high | OOM during validation only | Reduce to 2 or 1 |
| No cache clear before validation | Fragmentation prevents large allocations | `torch.cuda.empty_cache()` |
| Large `overlap` ratio | More windows = more peak memory for accumulation | Reduce to 0.25 |
| Validation on training device | Competes with cached training state | Clear cache, or use separate GPU |
| Forgetting 3D vs 2D scaling | A 128^3 volume is 2 million voxels, not 16K | Always calculate memory budget for 3D |

### The 3D memory trap

Medical imaging practitioners coming from 2D computer vision often underestimate 3D memory requirements. A single 128x128x128 volume with 4 channels at fp32 is:

```
4 * 128 * 128 * 128 * 4 bytes = 33.5 MB
```

Compare to a 2D image at 256x256 with 3 channels:

```
3 * 256 * 256 * 4 bytes = 0.75 MB
```

That's a **45x difference** per sample. Every batch size increase, every extra intermediate tensor, every stored activation hits 45x harder in 3D. This is why 3D medical imaging models typically use batch_size=1 for training and sw_batch_size=1 or 2 for validation.

---

## Solutions to Questions

<details>
<summary>Click to reveal solutions</summary>

**Q1:** Validation uses more memory because `sliding_window_inference` processes `sw_batch_size=4` patches at once (effective batch size 4) vs training's `batch_size=1`. Even without storing activations for backprop, the forward pass through SwinUNETR for 4 simultaneous patches creates 4x the intermediate tensors. Additionally, validation must maintain accumulation buffers for the full output volume and the overlap weighting map.

**Q2:** The ~10.5 GiB is a combination of: (1) Model weights (~250 MB), (2) Optimizer state from training (AdamW stores 2 momentum buffers per parameter: ~500 MB), (3) PyTorch's cached memory blocks from the training epoch that just finished — freed logically but still held by the CUDA allocator for reuse, (4) OS/display overhead (~500 MB). The key insight is that PyTorch doesn't return freed GPU memory to CUDA by default — it keeps it in a cache for fast reallocation.

**Q3:** The forward pass processes 4x the data, so intermediate activations are ~4x larger. For SwinUNETR-48, a single 128^3 patch forward pass uses ~2-3 GB of intermediate memory; with 4 patches simultaneously, that's ~8-12 GB. This alone approaches the 15 GiB limit before accounting for anything else.

**Q4:** Gradient checkpointing does NOT help during validation. Checkpointing works by not storing intermediate activations during the forward pass, then recomputing them during the backward pass. Since validation uses `torch.no_grad()` (no backward pass), checkpointing has nothing to trade off against. In fact, checkpointing during inference would just add unnecessary recomputation overhead with no memory savings.

**Q5:** After training, the CUDA memory allocator holds freed blocks in a cache. When validation requests a large contiguous block (5.26 GiB), the cached blocks may not form a contiguous region large enough. `torch.cuda.empty_cache()` forces the allocator to release all cached blocks back to CUDA, defragmenting memory and making large contiguous allocations possible.

**Q6:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This tells PyTorch to use expandable memory segments instead of fixed-size blocks, which reduces fragmentation when allocation patterns change (like the transition from training to validation).

**Q7:** Reducing `roi_size` from 128 to 96 would save memory but at a significant quality cost: smaller patches mean less spatial context for the model, leading to more boundary artifacts and potentially worse segmentation of large, connected structures. It also changes the effective receptive field of the model compared to what the pretrained weights were trained on. Reducing `sw_batch_size` from 4 to 2 only affects inference speed (roughly 2x slower validation), not model quality — every voxel still gets the same prediction based on the same context window.

**Q8:** Moving validation to CPU would be extremely slow for 3D medical images. A single sliding window inference pass on a 240^3 volume with a SwinUNETR model takes ~1-2 seconds on GPU but would take 5-15 minutes on CPU. With ~97 validation volumes, that's 8-24 hours per validation vs 7 minutes on GPU. In a 300-epoch training run with validation every 5 epochs, you'd spend more time on CPU validation than on the entire GPU training.

**Q9:** `model.half()` would work with `sliding_window_inference` and roughly halve the activation memory. However, fp16 inference can cause numerical issues: (1) Softmax/sigmoid outputs may lose precision near 0 and 1, affecting small structures like ET. (2) Hausdorff distance computation on fp16 predictions can introduce rounding errors. (3) The Dice metric involves division, which is sensitive to precision near zero. A safer approach is to use `torch.autocast('cuda')` which keeps sensitive operations in fp32 while using fp16 for matrix multiplications.

**Q10:**
```python
def auto_sw_batch_size(model, roi_size, in_channels, device, target_utilization=0.75):
    """Select sw_batch_size based on available GPU memory."""
    total_mem = torch.cuda.get_device_properties(device).total_memory
    budget = total_mem * target_utilization

    # Estimate per-patch memory via a single forward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    dummy = torch.randn(1, in_channels, *roi_size, device=device)
    with torch.no_grad():
        _ = model(dummy)
    per_patch = torch.cuda.max_memory_allocated(device) - baseline
    del dummy
    torch.cuda.empty_cache()

    # Calculate max batch size with headroom
    current_alloc = torch.cuda.memory_allocated(device)
    available = budget - current_alloc
    max_bs = max(1, int(available / per_patch))
    return min(max_bs, 4)  # Cap at 4 — diminishing returns beyond that
```

</details>

---

## References

- Hatamizadeh, A., et al. (2022). "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." BrainLes Workshop, MICCAI.
- Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV.
- MONAI Project. SwinUNETR implementation. `monai.networks.nets.SwinUNETR`.
- MONAI Project. `sliding_window_inference` documentation and source.
- Simpson, A., et al. (2019). "A large annotated medical image dataset for the development and evaluation of segmentation algorithms." arXiv:1902.09063. (Medical Segmentation Decathlon)
- Bakas, S., et al. (2018). "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in a Multi-institutional Study." arXiv:1811.02629. (BraTS Challenge)
- PyTorch Documentation. "CUDA Memory Management." pytorch.org/docs/stable/notes/cuda.html
