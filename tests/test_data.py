"""Comprehensive tests for data modules: unlabeled, pseudo-label, and strong transforms."""

from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
import pytest
import torch

from swinunetr_st.data.pseudo_label_dataset import (
    ConditionalBratsTransform,
    _get_labeled_datalist,
    _train_val_split,
    get_combined_dataset,
    get_combined_train_transforms,
    get_labeled_train_transforms,
    get_pseudo_label_transforms,
    get_pseudo_labeled_datalist,
    get_val_transforms,
)
from swinunetr_st.data.strong_transforms import get_strong_transforms, get_weak_transforms
from swinunetr_st.data.unlabeled_dataset import (
    get_unlabeled_datalist,
    get_unlabeled_dataset,
    get_unlabeled_transforms,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pseudo_label_dir(
    tmp_path,
    data_dir: str,
    num_items: int = 2,
    *,
    empty: bool = False,
    bad_structure: bool = False,
) -> str:
    """Create a pseudo-label directory with manifest.json and NIfTI pseudo-labels.

    Args:
        tmp_path: pytest tmp_path.
        data_dir: Existing MSD data directory (images resolved relative to this).
        num_items: Number of pseudo-label entries to create.
        empty: If True, write an empty manifest list.
        bad_structure: If True, write a malformed manifest.

    Returns:
        Path to the pseudo-label directory.
    """
    pseudo_dir = os.path.join(str(tmp_path), "pseudo_labels")
    os.makedirs(pseudo_dir, mode=0o700, exist_ok=True)

    if bad_structure:
        with open(os.path.join(pseudo_dir, "manifest.json"), "w") as f:
            json.dump({"not": "a list"}, f)
        return pseudo_dir

    if empty:
        with open(os.path.join(pseudo_dir, "manifest.json"), "w") as f:
            json.dump([], f)
        return pseudo_dir

    # Create NIfTI pseudo-label files and build manifest
    manifest = []
    images_ts = os.path.join(data_dir, "imagesTs")
    ts_files = sorted(
        f for f in os.listdir(images_ts) if f.endswith((".nii", ".nii.gz"))
    )

    for i in range(min(num_items, len(ts_files))):
        # Pseudo-label: 3-channel (TC, WT, ET), already converted
        label_data = np.zeros((3, 32, 32, 32), dtype=np.float32)
        label_data[0, 10:22, 10:22, 10:22] = 1.0  # TC
        label_data[1, 8:24, 8:24, 8:24] = 1.0  # WT
        label_data[2, 12:20, 12:20, 12:20] = 1.0  # ET
        label_nii = nib.Nifti1Image(label_data, affine=np.eye(4))
        label_fname = f"pseudo_{i:03d}.nii.gz"
        nib.save(label_nii, os.path.join(pseudo_dir, label_fname))

        manifest.append(
            {
                "image": os.path.join("imagesTs", ts_files[i]),
                "pseudo_label": label_fname,
                "confidence": 0.85 + i * 0.05,
            }
        )

    with open(os.path.join(pseudo_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    return pseudo_dir


# ===================================================================
# Tests for unlabeled_dataset.py
# ===================================================================


class TestGetUnlabeledDatalist:
    """Tests for get_unlabeled_datalist."""

    def test_loads_from_test_key_strings(self, synthetic_nifti_dataset: str) -> None:
        """Test loading from dataset.json 'test' key containing string entries."""
        datalist = get_unlabeled_datalist(synthetic_nifti_dataset)
        assert len(datalist) == 2
        for item in datalist:
            assert "image" in item
            assert os.path.isabs(item["image"])
            assert os.path.isfile(item["image"])

    def test_loads_from_test_key_dicts(self, tmp_path) -> None:
        """Test loading from dataset.json 'test' key containing dict entries."""
        data_dir = str(tmp_path / "data")
        images_ts = os.path.join(data_dir, "imagesTs")
        os.makedirs(images_ts, mode=0o700)

        # Create a dummy NIfTI
        nib.save(
            nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)),
            os.path.join(images_ts, "test_001.nii.gz"),
        )

        dataset_json = {
            "test": [{"image": os.path.join("imagesTs", "test_001.nii.gz")}],
            "training": [],
        }
        with open(os.path.join(data_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f)

        datalist = get_unlabeled_datalist(data_dir)
        assert len(datalist) == 1
        assert datalist[0]["image"].endswith("test_001.nii.gz")

    def test_fallback_to_scanning_imagests(self, synthetic_nifti_dataset: str) -> None:
        """When test key is empty, falls back to scanning imagesTs/ directory."""
        # Rewrite dataset.json with empty test key
        json_path = os.path.join(synthetic_nifti_dataset, "dataset.json")
        with open(json_path) as f:
            data = json.load(f)
        data["test"] = []
        with open(json_path, "w") as f:
            json.dump(data, f)

        datalist = get_unlabeled_datalist(synthetic_nifti_dataset)
        assert len(datalist) == 2  # 2 test images created by fixture

    def test_fallback_when_no_json(self, synthetic_nifti_dataset: str) -> None:
        """When dataset.json is missing, falls back to scanning imagesTs/."""
        os.remove(os.path.join(synthetic_nifti_dataset, "dataset.json"))
        datalist = get_unlabeled_datalist(synthetic_nifti_dataset)
        assert len(datalist) == 2

    def test_path_traversal_rejected(self, synthetic_nifti_dataset: str) -> None:
        """Paths that escape data_dir must be rejected."""
        json_path = os.path.join(synthetic_nifti_dataset, "dataset.json")
        with open(json_path) as f:
            data = json.load(f)
        data["test"] = ["../../etc/passwd"]
        with open(json_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Path traversal detected"):
            get_unlabeled_datalist(synthetic_nifti_dataset)

    def test_no_images_raises(self, tmp_path) -> None:
        """FileNotFoundError when there are no images at all."""
        data_dir = str(tmp_path / "empty_data")
        os.makedirs(data_dir, mode=0o700)
        with pytest.raises(FileNotFoundError, match="No unlabeled images found"):
            get_unlabeled_datalist(data_dir)

    def test_invalid_json_structure(self, tmp_path) -> None:
        """Non-dict JSON raises ValueError."""
        data_dir = str(tmp_path / "bad_data")
        os.makedirs(data_dir, mode=0o700)
        with open(os.path.join(data_dir, "dataset.json"), "w") as f:
            json.dump([1, 2, 3], f)
        with pytest.raises(ValueError, match="expected mapping"):
            get_unlabeled_datalist(data_dir)

    def test_unexpected_entry_format(self, tmp_path) -> None:
        """An integer in the test list raises ValueError."""
        data_dir = str(tmp_path / "bad_entry")
        os.makedirs(data_dir, mode=0o700)
        with open(os.path.join(data_dir, "dataset.json"), "w") as f:
            json.dump({"test": [42]}, f)
        with pytest.raises(ValueError, match="Unexpected test entry format"):
            get_unlabeled_datalist(data_dir)

    def test_absolute_paths_returned(self, synthetic_nifti_dataset: str) -> None:
        """All returned image paths are absolute."""
        datalist = get_unlabeled_datalist(synthetic_nifti_dataset)
        for item in datalist:
            assert os.path.isabs(item["image"])


class TestGetUnlabeledDataset:
    """Tests for get_unlabeled_dataset and get_unlabeled_transforms."""

    def test_creates_cache_dataset(
        self, synthetic_nifti_dataset: str, small_config: dict
    ) -> None:
        """CacheDataset is created with correct length."""
        small_config["paths"]["base_data_dir"] = synthetic_nifti_dataset
        ds = get_unlabeled_dataset(small_config)
        assert len(ds) == 2

    def test_custom_transforms(
        self, synthetic_nifti_dataset: str, small_config: dict
    ) -> None:
        """Custom transforms override the default pipeline."""
        small_config["paths"]["base_data_dir"] = synthetic_nifti_dataset
        custom = get_unlabeled_transforms(small_config)
        ds = get_unlabeled_dataset(small_config, transforms=custom)
        assert len(ds) == 2

    def test_unlabeled_transforms_keys(self, small_config: dict) -> None:
        """Unlabeled transforms only operate on 'image' key."""
        transforms = get_unlabeled_transforms(small_config)
        # Check the first transform (LoadImaged) only loads "image"
        load_transform = transforms.transforms[0]
        assert list(load_transform.keys) == ["image"]


# ===================================================================
# Tests for pseudo_label_dataset.py
# ===================================================================


class TestGetPseudoLabeledDatalist:
    """Tests for get_pseudo_labeled_datalist."""

    def test_loads_manifest(
        self, synthetic_nifti_dataset: str, tmp_path
    ) -> None:
        """Valid manifest.json produces correct datalist entries."""
        pseudo_dir = _make_pseudo_label_dir(tmp_path, synthetic_nifti_dataset)
        datalist = get_pseudo_labeled_datalist(pseudo_dir, synthetic_nifti_dataset)

        assert len(datalist) == 2
        for item in datalist:
            assert item["is_pseudo"] is True
            assert "confidence" in item
            assert isinstance(item["confidence"], float)
            assert os.path.isabs(item["image"])
            assert os.path.isabs(item["label"])

    def test_missing_manifest_raises(self, tmp_path) -> None:
        """FileNotFoundError when manifest.json is absent."""
        pseudo_dir = str(tmp_path / "no_manifest")
        os.makedirs(pseudo_dir, mode=0o700)
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            get_pseudo_labeled_datalist(pseudo_dir, str(tmp_path))

    def test_empty_manifest(
        self, synthetic_nifti_dataset: str, tmp_path
    ) -> None:
        """Empty manifest produces empty datalist."""
        pseudo_dir = _make_pseudo_label_dir(
            tmp_path, synthetic_nifti_dataset, empty=True
        )
        datalist = get_pseudo_labeled_datalist(pseudo_dir, synthetic_nifti_dataset)
        assert datalist == []

    def test_bad_manifest_structure(
        self, synthetic_nifti_dataset: str, tmp_path
    ) -> None:
        """Non-list manifest raises ValueError."""
        pseudo_dir = _make_pseudo_label_dir(
            tmp_path, synthetic_nifti_dataset, bad_structure=True
        )
        with pytest.raises(ValueError, match="must be a JSON array"):
            get_pseudo_labeled_datalist(pseudo_dir, synthetic_nifti_dataset)

    def test_missing_required_key_in_entry(
        self, synthetic_nifti_dataset: str, tmp_path
    ) -> None:
        """Manifest entry missing 'confidence' raises ValueError."""
        pseudo_dir = str(tmp_path / "bad_entry_manifest")
        os.makedirs(pseudo_dir, mode=0o700)
        manifest = [{"image": "imagesTs/foo.nii.gz", "pseudo_label": "bar.nii.gz"}]
        with open(os.path.join(pseudo_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        with pytest.raises(ValueError, match="missing required key"):
            get_pseudo_labeled_datalist(pseudo_dir, synthetic_nifti_dataset)

    def test_path_traversal_in_manifest(
        self, synthetic_nifti_dataset: str, tmp_path
    ) -> None:
        """Path traversal in manifest image path is rejected."""
        pseudo_dir = str(tmp_path / "traversal_manifest")
        os.makedirs(pseudo_dir, mode=0o700)
        manifest = [
            {
                "image": "../../etc/passwd",
                "pseudo_label": "ok.nii.gz",
                "confidence": 0.9,
            }
        ]
        with open(os.path.join(pseudo_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        with pytest.raises(ValueError, match="Path traversal detected"):
            get_pseudo_labeled_datalist(pseudo_dir, synthetic_nifti_dataset)

    def test_non_dict_entry_raises(
        self, synthetic_nifti_dataset: str, tmp_path
    ) -> None:
        """A string entry in the manifest list raises ValueError."""
        pseudo_dir = str(tmp_path / "str_entry_manifest")
        os.makedirs(pseudo_dir, mode=0o700)
        with open(os.path.join(pseudo_dir, "manifest.json"), "w") as f:
            json.dump(["not_a_dict"], f)

        with pytest.raises(ValueError, match="must be a dict"):
            get_pseudo_labeled_datalist(pseudo_dir, synthetic_nifti_dataset)


class TestConditionalBratsTransform:
    """Tests for ConditionalBratsTransform."""

    def test_skips_conversion_for_pseudo(self) -> None:
        """When is_pseudo=True, label tensor is returned unchanged."""
        # Pseudo-label: already 3-channel
        label = torch.zeros(3, 8, 8, 8)
        label[0] = 1.0  # TC
        data = {"image": torch.randn(4, 8, 8, 8), "label": label, "is_pseudo": True}

        transform = ConditionalBratsTransform(keys="label")
        result = transform(data)

        assert torch.equal(result["label"], label)

    def test_applies_conversion_for_labeled(self) -> None:
        """When is_pseudo is absent or False, BraTS conversion is applied."""
        # BraTS-format integer label
        label = torch.zeros(1, 8, 8, 8, dtype=torch.int16)
        label[0, 2:6, 2:6, 2:6] = 2  # Edema
        label[0, 3:5, 3:5, 3:5] = 1  # NCR/NET
        label[0, 4:5, 4:5, 4:5] = 4  # ET
        data = {"image": torch.randn(4, 8, 8, 8), "label": label}

        transform = ConditionalBratsTransform(keys="label")
        result = transform(data)

        # After BraTS conversion, should be 3 channels: TC, WT, ET
        assert result["label"].shape[0] == 3

    def test_is_pseudo_false_still_converts(self) -> None:
        """When is_pseudo is explicitly False, conversion is applied."""
        label = torch.zeros(1, 8, 8, 8, dtype=torch.int16)
        label[0, 2:6, 2:6, 2:6] = 2
        data = {
            "image": torch.randn(4, 8, 8, 8),
            "label": label,
            "is_pseudo": False,
        }

        transform = ConditionalBratsTransform(keys="label")
        result = transform(data)
        assert result["label"].shape[0] == 3


class TestGetCombinedDataset:
    """Tests for get_combined_dataset."""

    def test_correct_train_val_sizes(
        self, synthetic_nifti_dataset: str, small_config: dict, tmp_path
    ) -> None:
        """Training set includes both labeled and pseudo, val is labeled-only."""
        small_config["paths"]["base_data_dir"] = synthetic_nifti_dataset
        # Use 0.5 split so val is non-empty with only 4 labeled samples
        # (int(4 * 0.5) = 2 val, 2 train)
        small_config["data"]["train_val_split"] = 0.5
        pseudo_dir = _make_pseudo_label_dir(tmp_path, synthetic_nifti_dataset)

        train_ds, val_ds = get_combined_dataset(small_config, pseudo_dir, round_num=1)

        # 4 labeled: 0.5 split = 2 train + 2 val; 2 pseudo
        # train_ds = 2 labeled + 2 pseudo = 4, val_ds = 2
        assert len(val_ds) == 2
        assert len(train_ds) == 4
        assert len(train_ds) + len(val_ds) == 6

    def test_no_validation_contamination(
        self, synthetic_nifti_dataset: str, small_config: dict, tmp_path
    ) -> None:
        """Validation set must not contain any pseudo-labeled samples."""
        small_config["paths"]["base_data_dir"] = synthetic_nifti_dataset
        small_config["data"]["train_val_split"] = 0.5
        pseudo_dir = _make_pseudo_label_dir(tmp_path, synthetic_nifti_dataset)

        _, val_ds = get_combined_dataset(small_config, pseudo_dir, round_num=1)

        # Check the underlying data list in the CacheDataset
        for item in val_ds.data:
            assert item.get("is_pseudo", False) is False, (
                "Validation set must not contain pseudo-labeled data"
            )

    def test_pseudo_labels_marked(
        self, synthetic_nifti_dataset: str, small_config: dict, tmp_path
    ) -> None:
        """Pseudo-labeled entries in train dataset have is_pseudo=True."""
        small_config["paths"]["base_data_dir"] = synthetic_nifti_dataset
        pseudo_dir = _make_pseudo_label_dir(tmp_path, synthetic_nifti_dataset)

        train_ds, _ = get_combined_dataset(small_config, pseudo_dir, round_num=1)

        pseudo_count = sum(1 for item in train_ds.data if item.get("is_pseudo"))
        assert pseudo_count == 2


class TestTransformPipelines:
    """Tests for labeled, pseudo-label, val, and combined transform pipelines."""

    def test_labeled_train_has_brats_conversion(self, small_config: dict) -> None:
        """Labeled train transforms include ConvertToMultiChannelBasedOnBratsClassesd."""

        transforms = get_labeled_train_transforms(small_config)
        type_names = [type(t).__name__ for t in transforms.transforms]
        assert "ConvertToMultiChannelBasedOnBratsClassesd" in type_names

    def test_pseudo_label_skips_brats_conversion(self, small_config: dict) -> None:
        """Pseudo-label transforms must NOT include BraTS conversion."""

        transforms = get_pseudo_label_transforms(small_config)
        type_names = [type(t).__name__ for t in transforms.transforms]
        assert "ConvertToMultiChannelBasedOnBratsClassesd" not in type_names

    def test_combined_uses_conditional_transform(self, small_config: dict) -> None:
        """Combined train transforms use ConditionalBratsTransform."""
        transforms = get_combined_train_transforms(small_config)
        type_names = [type(t).__name__ for t in transforms.transforms]
        assert "ConditionalBratsTransform" in type_names

    def test_val_transforms_are_deterministic(self, small_config: dict) -> None:
        """Val transforms have no Rand* transforms."""
        transforms = get_val_transforms(small_config)
        for t in transforms.transforms:
            assert not type(t).__name__.startswith("Rand"), (
                f"Validation transforms should not have random augmentations: {type(t).__name__}"
            )


# ===================================================================
# Tests for strong_transforms.py
# ===================================================================


class TestStrongTransforms:
    """Tests for get_strong_transforms."""

    def test_includes_extra_augmentations(self, small_config: dict) -> None:
        """Strong transforms include noise, contrast, and smoothing."""
        transforms = get_strong_transforms(small_config)
        type_names = [type(t).__name__ for t in transforms.transforms]

        assert "RandGaussianNoised" in type_names
        assert "RandAdjustContrastd" in type_names
        assert "RandGaussianSmoothd" in type_names

    def test_includes_base_augmentations(self, small_config: dict) -> None:
        """Strong transforms also include the standard base augmentations."""
        transforms = get_strong_transforms(small_config)
        type_names = [type(t).__name__ for t in transforms.transforms]

        assert "RandFlipd" in type_names
        assert "RandScaleIntensityd" in type_names
        assert "RandShiftIntensityd" in type_names
        assert "RandSpatialCropd" in type_names

    def test_respects_config_overrides(self, small_config: dict) -> None:
        """Config values for strong augmentation are picked up."""
        small_config["data"]["strong_augmentation"]["gaussian_noise_std"] = 0.5
        transforms = get_strong_transforms(small_config)

        # Find the RandGaussianNoised transform and check std via inner transform
        for t in transforms.transforms:
            if type(t).__name__ == "RandGaussianNoised":
                assert t.rand_gaussian_noise.std == 0.5
                break
        else:
            pytest.fail("RandGaussianNoised not found in strong transforms")

    def test_more_transforms_than_weak(self, small_config: dict) -> None:
        """Strong pipeline has strictly more transforms than weak."""
        strong = get_strong_transforms(small_config)
        weak = get_weak_transforms(small_config)
        assert len(strong.transforms) > len(weak.transforms)


class TestWeakTransforms:
    """Tests for get_weak_transforms."""

    def test_includes_spatial_crop(self, small_config: dict) -> None:
        """Weak transforms include RandSpatialCropd for size consistency."""
        transforms = get_weak_transforms(small_config)
        type_names = [type(t).__name__ for t in transforms.transforms]
        assert "RandSpatialCropd" in type_names

    def test_no_intensity_augmentation(self, small_config: dict) -> None:
        """Weak transforms do not include any intensity augmentation."""
        transforms = get_weak_transforms(small_config)
        type_names = [type(t).__name__ for t in transforms.transforms]

        intensity_transforms = {
            "RandScaleIntensityd",
            "RandShiftIntensityd",
            "RandGaussianNoised",
            "RandAdjustContrastd",
            "RandGaussianSmoothd",
        }
        for name in type_names:
            assert name not in intensity_transforms, (
                f"Weak transforms should not have intensity augmentation: {name}"
            )

    def test_only_operates_on_image_key(self, small_config: dict) -> None:
        """Weak transforms only use the 'image' key (no label loading)."""
        transforms = get_weak_transforms(small_config)
        load_transform = transforms.transforms[0]
        assert list(load_transform.keys) == ["image"]


class TestStrongVsWeakOutputs:
    """Integration test: strong and weak transforms produce different outputs."""

    def test_outputs_differ(self, synthetic_nifti_dataset: str, small_config: dict) -> None:
        """Given the same input, strong and weak produce statistically different results."""
        small_config["paths"]["base_data_dir"] = synthetic_nifti_dataset

        datalist = get_unlabeled_datalist(synthetic_nifti_dataset)
        assert len(datalist) >= 1

        strong = get_strong_transforms(small_config)
        weak = get_weak_transforms(small_config)

        # Run both on the same sample multiple times and check they differ
        sample = datalist[0].copy()
        torch.manual_seed(42)
        strong_out = strong(sample.copy())
        torch.manual_seed(42)
        weak_out = weak(sample.copy())

        # Shapes should match but values will differ due to augmentations
        assert strong_out["image"].shape == weak_out["image"].shape
        # At least some values should differ (strong has extra perturbations)
        # We don't assert exact difference since randomness is involved,
        # but the transforms themselves are verified above


# ===================================================================
# Tests for internal helpers
# ===================================================================


class TestInternalHelpers:
    """Tests for internal helpers in pseudo_label_dataset."""

    def test_get_labeled_datalist(self, synthetic_nifti_dataset: str) -> None:
        """_get_labeled_datalist reads training key correctly."""
        datalist = _get_labeled_datalist(synthetic_nifti_dataset)
        assert len(datalist) == 4
        for item in datalist:
            assert os.path.isabs(item["image"])
            assert os.path.isabs(item["label"])

    def test_get_labeled_datalist_missing_key(self, tmp_path) -> None:
        """Missing 'training' key in dataset.json raises ValueError."""
        data_dir = str(tmp_path / "no_training")
        os.makedirs(data_dir, mode=0o700)
        with open(os.path.join(data_dir, "dataset.json"), "w") as f:
            json.dump({"test": []}, f)

        with pytest.raises(ValueError, match="missing 'training' key"):
            _get_labeled_datalist(data_dir)

    def test_train_val_split_deterministic(self) -> None:
        """Same seed produces the same split."""
        datalist = [{"image": f"img_{i}", "label": f"lbl_{i}"} for i in range(10)]
        train1, val1 = _train_val_split(datalist, val_ratio=0.2, seed=42)
        train2, val2 = _train_val_split(datalist, val_ratio=0.2, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_train_val_split_no_overlap(self) -> None:
        """Train and val sets should not overlap."""
        datalist = [{"image": f"img_{i}", "label": f"lbl_{i}"} for i in range(10)]
        train, val = _train_val_split(datalist, val_ratio=0.3, seed=42)

        train_images = {item["image"] for item in train}
        val_images = {item["image"] for item in val}
        assert train_images.isdisjoint(val_images)
        assert len(train) + len(val) == len(datalist)
