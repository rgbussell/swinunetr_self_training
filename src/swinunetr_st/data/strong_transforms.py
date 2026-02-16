"""Strong and weak augmentation pipelines for consistency-based self-training."""

from __future__ import annotations

import logging

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)

logger = logging.getLogger(__name__)


def get_strong_transforms(config: dict) -> Compose:
    """Strong augmentation pipeline for unlabeled consistency training.

    Includes all base training augmentations **plus** additional noise,
    contrast, and smoothing perturbations.  Parameter defaults can be
    overridden via ``config["data"]["strong_augmentation"]``.

    Args:
        config: Parsed YAML config dict with ``data.roi_size`` and optional
            ``data.strong_augmentation`` sub-dict.

    Returns:
        MONAI Compose transform pipeline.
    """
    roi_size = config["data"]["roi_size"]
    strong_cfg = config["data"].get("strong_augmentation", {})

    noise_std = strong_cfg.get("gaussian_noise_std", 0.1)
    gamma_range = tuple(strong_cfg.get("gamma_range", [0.7, 1.5]))
    smooth_sigma = tuple(strong_cfg.get("gaussian_smooth_sigma", [0.5, 1.15]))

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            RandSpatialCropd(keys=["image"], roi_size=roi_size, random_size=False),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # --- Additional strong augmentations ---
            RandGaussianNoised(keys="image", std=noise_std, prob=0.5),
            RandAdjustContrastd(keys="image", gamma=gamma_range, prob=0.3),
            RandGaussianSmoothd(
                keys="image",
                sigma_x=smooth_sigma,
                sigma_y=smooth_sigma,
                sigma_z=smooth_sigma,
                prob=0.2,
            ),
        ]
    )


def get_weak_transforms(config: dict) -> Compose:
    """Weak augmentation pipeline for teacher-branch pseudo-label generation.

    Like validation transforms but **with** a random spatial crop for
    consistent input sizing.  No intensity augmentation is applied.

    Args:
        config: Parsed YAML config dict with ``data.roi_size``.

    Returns:
        MONAI Compose transform pipeline.
    """
    roi_size = config["data"]["roi_size"]

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            RandSpatialCropd(keys=["image"], roi_size=roi_size, random_size=False),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
