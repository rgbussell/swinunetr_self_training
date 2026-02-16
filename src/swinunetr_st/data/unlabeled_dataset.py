"""Unlabeled dataset loading for self-training pseudo-label generation."""

from __future__ import annotations

import json
import logging
import os

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

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


def get_unlabeled_datalist(
    data_dir: str,
    dataset_json: str = "dataset.json",
) -> list[dict]:
    """Parse MSD ``dataset.json`` *test* key to build an unlabeled datalist.

    The *test* key may contain plain strings (relative image paths) or dicts
    with an ``"image"`` key.  As a fallback when no ``dataset.json`` exists or
    the *test* key is missing/empty, the function scans the ``imagesTs/``
    subdirectory for NIfTI files.

    Args:
        data_dir: Root directory of the MSD-format dataset.
        dataset_json: Filename of the JSON descriptor inside *data_dir*.

    Returns:
        List of ``{"image": absolute_path}`` dicts.

    Raises:
        FileNotFoundError: If neither the JSON file nor ``imagesTs/`` provide
            any images.
        ValueError: On path-traversal or invalid JSON structure.
    """
    json_path = os.path.join(data_dir, dataset_json)
    datalist: list[dict] = []

    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid dataset JSON: expected mapping, got {type(data).__name__}"
            )

        test_entries = data.get("test", [])
        for entry in test_entries:
            if isinstance(entry, str):
                rel_path = entry
            elif isinstance(entry, dict) and "image" in entry:
                rel_path = entry["image"]
            else:
                raise ValueError(f"Unexpected test entry format: {entry!r}")

            abs_path = _validate_path_containment(rel_path, data_dir)
            datalist.append({"image": abs_path})

        if datalist:
            logger.info(
                "Loaded %d unlabeled entries from %s[test]", len(datalist), json_path
            )
            return datalist

        logger.warning("No entries in 'test' key of %s, falling back to imagesTs/", json_path)

    # Fallback: scan imagesTs/ directory
    images_ts_dir = os.path.join(data_dir, "imagesTs")
    if os.path.isdir(images_ts_dir):
        for fname in sorted(os.listdir(images_ts_dir)):
            if fname.endswith((".nii", ".nii.gz")):
                abs_path = _validate_path_containment(
                    os.path.join("imagesTs", fname), data_dir
                )
                datalist.append({"image": abs_path})

    if datalist:
        logger.info(
            "Loaded %d unlabeled entries by scanning imagesTs/", len(datalist)
        )
        return datalist

    raise FileNotFoundError(
        f"No unlabeled images found in {data_dir} "
        "(checked dataset.json[test] and imagesTs/)"
    )


def get_unlabeled_transforms(config: dict) -> Compose:
    """Minimal transforms for unlabeled images (no labels loaded).

    Args:
        config: Parsed YAML config dict.

    Returns:
        MONAI Compose transform pipeline.
    """
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


def get_unlabeled_dataset(config: dict, transforms: Compose | None = None) -> CacheDataset:
    """Build a :class:`CacheDataset` for unlabeled data.

    Args:
        config: Parsed YAML config dict with ``paths.base_data_dir`` and
            ``data.cache_rate``.
        transforms: Optional transform pipeline.  If ``None`` the default
            :func:`get_unlabeled_transforms` is used.

    Returns:
        MONAI CacheDataset of unlabeled images.
    """
    data_dir = config["paths"]["base_data_dir"]
    cache_rate = config["data"].get("cache_rate", 0.0)
    num_workers = config["training"].get("num_workers", 4)

    datalist = get_unlabeled_datalist(data_dir)
    if transforms is None:
        transforms = get_unlabeled_transforms(config)

    logger.info("Creating unlabeled CacheDataset with %d items", len(datalist))
    return CacheDataset(
        data=datalist,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
