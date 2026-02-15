"""Pseudo-label generation from a teacher model on unlabeled brain MRI volumes."""

from __future__ import annotations

import json
import logging
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, NormalizeIntensity, Orientation, Spacing

from swinunetr_st.data.unlabeled_dataset import get_unlabeled_datalist

logger = logging.getLogger(__name__)

CLASS_NAMES = ["TC", "WT", "ET"]


class PseudoLabeler:
    """Generates pseudo-labels from a teacher model on unlabeled volumes.

    For each unlabeled volume the model runs sliding-window inference,
    applies per-class confidence thresholds, and saves accepted predictions
    as soft (sigmoid probability) multi-channel NIfTI files.

    Args:
        model: A torch.nn.Module (e.g. SwinUNETR) for inference.
        config: Parsed YAML config dict.
        device: Torch device for inference.  Auto-detected when ``None``.
    """

    def __init__(self, model: nn.Module, config: dict, device: torch.device | None = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config
        self.roi_size: list[int] | tuple[int, ...] = config["data"]["roi_size"]
        self.min_confidence_fraction: float = config["self_training"].get("min_confidence_fraction", 0.3)
        self.use_amp: bool = config["training"].get("amp", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        datalist: list[dict],
        thresholds: dict[str, float],
        output_dir: str,
        round_num: int = 0,
    ) -> dict:
        """Run inference on *datalist* and save accepted pseudo-labels.

        Args:
            datalist: List of ``{"image": path}`` dicts from
                :func:`get_unlabeled_datalist`.
            thresholds: Per-class confidence thresholds, e.g.
                ``{"TC": 0.92, "WT": 0.87, "ET": 0.97}``.
            output_dir: Directory where pseudo-label NIfTI files and
                ``manifest.json`` are saved.
            round_num: Current self-training round (used for logging).

        Returns:
            Statistics dict with keys ``accepted``, ``rejected``, ``total``,
            ``mean_confidence``, and ``per_class_acceptance``.
        """
        os.makedirs(output_dir, mode=0o755, exist_ok=True)

        transform = self._build_transforms()

        accepted_count = 0
        rejected_count = 0
        confidence_sum = 0.0
        per_class_accepted = {name: 0 for name in CLASS_NAMES}
        per_class_total = {name: 0 for name in CLASS_NAMES}
        manifest: list[dict] = []

        total = len(datalist)
        logger.info("Round %d: generating pseudo-labels for %d volumes", round_num, total)

        for idx, entry in enumerate(datalist):
            image_path = entry["image"]
            probs, affine = self._infer_volume(image_path, transform)

            # Evaluate per-class acceptance
            class_accepted = {}
            class_confidences = {}
            for ci, name in enumerate(CLASS_NAMES):
                per_class_total[name] += 1
                channel = probs[ci]  # (D, H, W)
                thresh = thresholds[name]
                max_prob = float(channel.max())
                above_mask = channel >= thresh
                class_accepted[name] = max_prob >= thresh
                if class_accepted[name]:
                    class_confidences[name] = float(channel[above_mask].mean())
                else:
                    class_confidences[name] = max_prob

            # Volume-level acceptance: all classes must pass their threshold
            # AND overall fraction of confident voxels must meet minimum
            all_classes_pass = all(class_accepted.values())

            # Compute fraction of voxels above threshold across all classes
            total_above = 0
            total_voxels = 0
            for ci, name in enumerate(CLASS_NAMES):
                channel = probs[ci]
                thresh = thresholds[name]
                total_above += int((channel >= thresh).sum())
                total_voxels += channel.size

            confidence_fraction = total_above / max(total_voxels, 1)
            meets_fraction = confidence_fraction >= self.min_confidence_fraction

            if all_classes_pass and meets_fraction:
                # Accepted: save soft pseudo-label
                mean_conf = float(np.mean([class_confidences[n] for n in CLASS_NAMES]))
                filename = f"pseudo_round{round_num}_{idx:04d}.nii.gz"
                save_path = os.path.join(output_dir, filename)
                self._save_nifti(probs, affine, save_path)

                # Record relative image path for portability
                manifest.append({
                    "image": image_path,
                    "pseudo_label": filename,
                    "confidence": round(mean_conf, 6),
                })

                accepted_count += 1
                confidence_sum += mean_conf
                for name in CLASS_NAMES:
                    per_class_accepted[name] += 1

                logger.debug(
                    "  [%d/%d] ACCEPTED %s (confidence=%.4f)",
                    idx + 1, total, os.path.basename(image_path), mean_conf,
                )
            else:
                rejected_count += 1
                logger.debug(
                    "  [%d/%d] REJECTED %s (classes_pass=%s, frac=%.4f)",
                    idx + 1, total, os.path.basename(image_path),
                    all_classes_pass, confidence_fraction,
                )

        # Save manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved manifest with %d entries to %s", len(manifest), manifest_path)

        # Compute stats
        per_class_acceptance = {}
        for name in CLASS_NAMES:
            denom = per_class_total[name] if per_class_total[name] > 0 else 1
            per_class_acceptance[name] = per_class_accepted[name] / denom

        mean_confidence = confidence_sum / accepted_count if accepted_count > 0 else 0.0

        stats = {
            "accepted": accepted_count,
            "rejected": rejected_count,
            "total": total,
            "mean_confidence": round(mean_confidence, 6),
            "per_class_acceptance": per_class_acceptance,
        }

        logger.info(
            "Round %d pseudo-label stats: accepted=%d/%d (%.1f%%), mean_confidence=%.4f",
            round_num, accepted_count, total,
            100.0 * accepted_count / max(total, 1), mean_confidence,
        )
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_transforms(self) -> Compose:
        """Build the preprocessing pipeline for a single unlabeled volume.

        Returns:
            MONAI ``Compose`` of non-dictionary transforms for single images.
        """
        return Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ])

    @torch.no_grad()
    def _infer_volume(self, image_path: str, transform: Compose) -> tuple[np.ndarray, np.ndarray]:
        """Load, preprocess, and run inference on a single volume.

        Args:
            image_path: Absolute path to the NIfTI image.
            transform: Preprocessing transform pipeline.

        Returns:
            Tuple of (probs, affine) where *probs* is a float32 numpy array
            of shape ``(C, D, H, W)`` with sigmoid probabilities, and *affine*
            is the 4x4 affine matrix from the NIfTI header.
        """
        # Load the raw NIfTI to get the affine
        nii = nib.load(image_path)
        affine = nii.affine

        # Preprocess
        image_tensor = transform(image_path)
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.as_tensor(image_tensor)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1, C, D, H, W)

        # Sliding-window inference
        roi_size = list(self.roi_size)
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                output = sliding_window_inference(
                    image_tensor, roi_size, sw_batch_size=4, predictor=self.model, overlap=0.5,
                )
        else:
            output = sliding_window_inference(
                image_tensor, roi_size, sw_batch_size=4, predictor=self.model, overlap=0.5,
            )

        # Sigmoid to get probabilities
        probs = torch.sigmoid(output).squeeze(0).cpu().numpy().astype(np.float32)  # (C, D, H, W)
        return probs, affine

    @staticmethod
    def _save_nifti(data: np.ndarray, affine: np.ndarray, path: str) -> None:
        """Save a multi-channel float32 array as a NIfTI file.

        The data array has shape ``(C, D, H, W)`` and is transposed to
        ``(D, H, W, C)`` for NIfTI convention.

        Args:
            data: Float32 array of shape ``(C, D, H, W)``.
            affine: 4x4 affine matrix.
            path: Output file path.
        """
        # NIfTI convention: spatial dims first, channels last
        nii_data = np.transpose(data, (1, 2, 3, 0))
        img = nib.Nifti1Image(nii_data, affine)
        nib.save(img, path)


def generate_pseudo_labels(
    model: nn.Module,
    config: dict,
    round_num: int,
    thresholds: dict[str, float],
) -> dict:
    """Standalone convenience function for pseudo-label generation.

    Creates a :class:`PseudoLabeler`, loads the unlabeled datalist from the
    config, and runs generation.

    Args:
        model: Trained teacher model.
        config: Parsed YAML config dict.
        round_num: Current self-training round.
        thresholds: Per-class confidence thresholds from
            :class:`~swinunetr_st.training.curriculum.ThresholdScheduler`.

    Returns:
        Statistics dict (see :meth:`PseudoLabeler.generate`).
    """
    labeler = PseudoLabeler(model, config)
    data_dir = config["paths"]["base_data_dir"]
    datalist = get_unlabeled_datalist(data_dir)

    pseudo_label_dir = config["paths"]["pseudo_label_dir"]
    output_dir = os.path.join(pseudo_label_dir, f"round_{round_num}")

    return labeler.generate(datalist, thresholds, output_dir, round_num=round_num)
