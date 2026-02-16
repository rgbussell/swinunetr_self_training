"""Pseudo-labeled dataset loading and combined training datasets for self-training."""

from __future__ import annotations

import json
import logging
import os
import random

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)
from swinunetr_seg.data.transforms import ConvertMSDBratsClassesd

logger = logging.getLogger(__name__)


def _validate_path_containment(path: str, base_dir: str) -> str:
    """Resolve *path* and verify it stays within *base_dir* (CWE-22 prevention).

    Args:
        path: Relative or absolute path to validate.
        base_dir: Root directory that must contain *path*.

    Returns:
        Absolute, resolved path.

    Raises:
        ValueError: If the resolved path escapes *base_dir*.
    """
    full = os.path.realpath(os.path.join(base_dir, path))
    base = os.path.realpath(base_dir)
    if not full.startswith(base + os.sep) and full != base:
        raise ValueError(f"Path traversal detected: {path}")
    return full


class ConditionalBratsTransform(MapTransform):
    """Apply :class:`ConvertMSDBratsClassesd` only for non-pseudo labels.

    Pseudo-labels are already stored as 3-channel (TC, WT, ET) volumes and must
    **not** go through BraTS class conversion.  This transform inspects the
    ``"is_pseudo"`` metadata flag in each data dict and skips the conversion
    when the flag is ``True``.

    Args:
        keys: Key(s) of the label tensor(s) in the data dict.
        allow_missing_keys: If ``True``, don't raise when a key is absent.
    """

    def __init__(
        self,
        keys: str | list[str] = "label",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self._brats_convert = ConvertMSDBratsClassesd(keys=keys)

    def __call__(self, data: dict) -> dict:
        """Apply BraTS conversion conditionally based on ``is_pseudo`` flag."""
        if data.get("is_pseudo", False):
            logger.debug("Skipping BraTS conversion for pseudo-labeled sample")
            return data
        return self._brats_convert(data)


def get_pseudo_labeled_datalist(
    pseudo_label_dir: str,
    data_dir: str,
) -> list[dict]:
    """Read ``manifest.json`` from *pseudo_label_dir* and build a datalist.

    The manifest is expected to be a list of dicts, each with at least
    ``"image"`` (relative to *data_dir*), ``"pseudo_label"`` (relative to
    *pseudo_label_dir*), and ``"confidence"`` keys.

    Args:
        pseudo_label_dir: Directory containing pseudo-labels and ``manifest.json``.
        data_dir: Root dataset directory (images are relative to this).

    Returns:
        List of dicts with ``image``, ``label``, ``is_pseudo``, and ``confidence``.

    Raises:
        FileNotFoundError: If ``manifest.json`` does not exist.
        ValueError: On path-traversal or invalid manifest structure.
    """
    manifest_path = os.path.join(pseudo_label_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Pseudo-label manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if not isinstance(manifest, list):
        raise ValueError(
            f"manifest.json must be a JSON array, got {type(manifest).__name__}"
        )

    datalist: list[dict] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError(f"Each manifest entry must be a dict, got {type(entry).__name__}")
        for required_key in ("image", "pseudo_label", "confidence"):
            if required_key not in entry:
                raise ValueError(
                    f"Manifest entry missing required key '{required_key}': {entry!r}"
                )

        image_path = _validate_path_containment(entry["image"], data_dir)
        label_path = _validate_path_containment(entry["pseudo_label"], pseudo_label_dir)
        confidence = float(entry["confidence"])

        datalist.append(
            {
                "image": image_path,
                "label": label_path,
                "is_pseudo": True,
                "confidence": confidence,
            }
        )

    logger.info(
        "Loaded %d pseudo-labeled entries from %s", len(datalist), manifest_path
    )
    return datalist


def _get_labeled_datalist(data_dir: str, dataset_json: str = "dataset.json") -> list[dict]:
    """Parse MSD ``dataset.json`` training key for labeled data.

    Args:
        data_dir: Root directory of the MSD-format dataset.
        dataset_json: Filename of the JSON descriptor.

    Returns:
        List of ``{"image": abs_path, "label": abs_path}`` dicts.
    """
    json_path = os.path.join(data_dir, dataset_json)
    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "training" not in data:
        raise ValueError(f"Invalid dataset.json: missing 'training' key in {json_path}")

    datalist: list[dict] = []
    for entry in data["training"]:
        if "image" not in entry or "label" not in entry:
            raise ValueError("Invalid training entry: missing 'image' or 'label'")
        datalist.append(
            {
                "image": _validate_path_containment(entry["image"], data_dir),
                "label": _validate_path_containment(entry["label"], data_dir),
            }
        )
    return datalist


def _train_val_split(
    datalist: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split *datalist* into train and val sets deterministically.

    Args:
        datalist: Full labeled datalist.
        val_ratio: Fraction reserved for validation.
        seed: RNG seed for reproducibility.

    Returns:
        ``(train_list, val_list)`` tuple.
    """
    rng = random.Random(seed)
    indices = list(range(len(datalist)))
    rng.shuffle(indices)

    val_size = int(len(datalist) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_list = [datalist[i] for i in train_indices]
    val_list = [datalist[i] for i in val_indices]
    return train_list, val_list


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------


def get_labeled_train_transforms(config: dict) -> Compose:
    """Training transforms for labeled data (includes BraTS class conversion).

    Mirrors the base project ``get_train_transforms`` exactly.

    Args:
        config: Parsed YAML config dict with ``data.roi_size``.

    Returns:
        MONAI Compose transform pipeline.
    """
    roi_size = config["data"]["roi_size"]
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertMSDBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def get_pseudo_label_transforms(config: dict) -> Compose:
    """Training transforms for pseudo-labeled data (**skips** BraTS conversion).

    Pseudo-labels are already 3-channel (TC, WT, ET), so
    :class:`ConvertMSDBratsClassesd` must be omitted.

    Args:
        config: Parsed YAML config dict with ``data.roi_size``.

    Returns:
        MONAI Compose transform pipeline.
    """
    roi_size = config["data"]["roi_size"]
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # No ConvertMSDBratsClassesd -- labels are pre-converted
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def get_combined_train_transforms(config: dict) -> Compose:
    """Training transforms for a combined labeled + pseudo-labeled dataset.

    Uses :class:`ConditionalBratsTransform` to apply BraTS conversion only for
    real (non-pseudo) labels.

    Args:
        config: Parsed YAML config dict with ``data.roi_size``.

    Returns:
        MONAI Compose transform pipeline.
    """
    roi_size = config["data"]["roi_size"]
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConditionalBratsTransform(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def get_val_transforms(config: dict) -> Compose:
    """Validation transforms for labeled data (deterministic).

    Args:
        config: Parsed YAML config dict.

    Returns:
        MONAI Compose transform pipeline.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertMSDBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


# ---------------------------------------------------------------------------
# Combined dataset construction
# ---------------------------------------------------------------------------


def get_combined_dataset(
    config: dict,
    pseudo_label_dir: str,
    round_num: int,
) -> tuple[CacheDataset, CacheDataset]:
    """Build combined training dataset (labeled + pseudo-labeled) and validation dataset.

    The training set merges the labeled training split with pseudo-labeled data.
    The validation set contains **only** labeled data (no pseudo-labels) to
    ensure unbiased evaluation.

    Args:
        config: Parsed YAML config dict.
        pseudo_label_dir: Directory containing pseudo-labels and ``manifest.json``.
        round_num: Current self-training round (for logging).

    Returns:
        ``(train_dataset, val_dataset)`` tuple of MONAI CacheDatasets.
    """
    data_dir = config["paths"]["base_data_dir"]
    val_ratio = 1.0 - config["data"]["train_val_split"]
    cache_rate = config["data"].get("cache_rate", 0.0)
    num_workers = config["training"].get("num_workers", 4)

    # Labeled data
    labeled_list = _get_labeled_datalist(data_dir)
    labeled_train, labeled_val = _train_val_split(labeled_list, val_ratio=val_ratio)

    # Pseudo-labeled data
    pseudo_list = get_pseudo_labeled_datalist(pseudo_label_dir, data_dir)

    combined_train = labeled_train + pseudo_list
    logger.info(
        "Round %d: %d labeled train + %d pseudo-labeled = %d combined train, "
        "%d labeled val",
        round_num,
        len(labeled_train),
        len(pseudo_list),
        len(combined_train),
        len(labeled_val),
    )

    train_transforms = get_combined_train_transforms(config)
    val_transforms = get_val_transforms(config)

    train_ds = CacheDataset(
        data=combined_train,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=labeled_val,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    return train_ds, val_ds
