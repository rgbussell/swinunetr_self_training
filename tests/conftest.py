"""Shared fixtures for self-training tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config_path(project_root: Path) -> Path:
    """Return the path to the default config file."""
    return project_root / "configs" / "self_training_config.yaml"


@pytest.fixture
def device() -> torch.device:
    """Return cuda device if available, otherwise cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config(tmp_path: Path) -> dict:
    """Minimal config dict for fast testing."""
    return {
        "model": {
            "feature_size": 12,
            "img_size": [64, 64, 64],
            "in_channels": 4,
            "out_channels": 3,
            "use_checkpoint": False,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
        },
        "self_training": {
            "num_rounds": 2,
            "epochs_per_round": 2,
            "val_interval": 1,
            "ema_decay": 0.999,
            "ema_warmup_steps": 5,
            "initial_threshold": 0.95,
            "final_threshold": 0.75,
            "threshold_schedule": "cosine",
            "per_class_offsets": {"TC": 0.0, "WT": -0.05, "ET": 0.05},
            "supervised_weight": 1.0,
            "pseudo_label_weight": 0.5,
            "consistency_weight": 0.1,
            "ramp_up_epochs": 1,
            "min_confidence_fraction": 0.3,
            "max_pseudo_label_ratio": 0.5,
        },
        "training": {
            "lr": 5e-5,
            "weight_decay": 1e-5,
            "warmup_epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "amp": False,
            "gradient_accumulation_steps": 1,
            "early_stopping_patience": 10,
        },
        "data": {
            "roi_size": [64, 64, 64],
            "train_val_split": 0.8,
            "cache_rate": 0.0,
            "strong_augmentation": {
                "gaussian_noise_std": 0.1,
                "gamma_range": [0.7, 1.5],
                "brightness_range": [0.7, 1.3],
                "contrast_range": [0.65, 1.5],
                "gaussian_smooth_sigma": [0.5, 1.15],
            },
        },
        "paths": {
            "base_data_dir": str(tmp_path / "data"),
            "baseline_checkpoint": str(tmp_path / "baseline.pt"),
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "pseudo_label_dir": str(tmp_path / "pseudo_labels"),
            "results_dir": str(tmp_path / "results"),
        },
    }


@pytest.fixture
def dummy_volumes_3d() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (image, label) tensors for quick tests.

    Returns:
        Tuple of (image: (1, 4, 64, 64, 64), label: (1, 3, 64, 64, 64)).
    """
    image = torch.randn(1, 4, 64, 64, 64)
    label = torch.zeros(1, 3, 64, 64, 64)
    label[:, 0, 4:28, 4:28, 4:28] = 1.0  # TC
    label[:, 1, 2:30, 2:30, 2:30] = 1.0  # WT
    label[:, 2, 8:24, 8:24, 8:24] = 1.0  # ET
    return image, label


@pytest.fixture
def synthetic_nifti_dataset(tmp_path: Path) -> str:
    """Create a small synthetic MSD-format dataset with 4 NIfTI volumes.

    Returns:
        Path to data directory containing dataset.json, imagesTr/, labelsTr/, imagesTs/.
    """
    data_dir = str(tmp_path / "Task01_BrainTumour")
    images_tr = os.path.join(data_dir, "imagesTr")
    labels_tr = os.path.join(data_dir, "labelsTr")
    images_ts = os.path.join(data_dir, "imagesTs")
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)

    training = []
    for i in range(4):
        img_data = np.random.rand(32, 32, 32, 4).astype(np.float32)
        img_nii = nib.Nifti1Image(img_data, affine=np.eye(4))
        img_rel = os.path.join("imagesTr", f"BRATS_{i:03d}.nii.gz")
        nib.save(img_nii, os.path.join(data_dir, img_rel))

        lbl_data = np.zeros((32, 32, 32), dtype=np.int16)
        lbl_data[8:24, 8:24, 8:24] = 2  # Edema
        lbl_data[10:22, 10:22, 10:22] = 1  # NCR/NET
        lbl_data[12:20, 12:20, 12:20] = 4  # ET
        lbl_nii = nib.Nifti1Image(lbl_data, affine=np.eye(4))
        lbl_rel = os.path.join("labelsTr", f"BRATS_{i:03d}.nii.gz")
        nib.save(lbl_nii, os.path.join(data_dir, lbl_rel))

        training.append({"image": img_rel, "label": lbl_rel})

    # Test volumes (unlabeled)
    test_list = []
    for i in range(2):
        img_data = np.random.rand(32, 32, 32, 4).astype(np.float32)
        img_nii = nib.Nifti1Image(img_data, affine=np.eye(4))
        img_rel = os.path.join("imagesTs", f"BRATS_{i + 100:03d}.nii.gz")
        nib.save(img_nii, os.path.join(data_dir, img_rel))
        test_list.append(img_rel)

    dataset_json = {
        "name": "Task01_BrainTumour",
        "training": training,
        "test": test_list,
    }
    with open(os.path.join(data_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f)

    return data_dir
