"""Self-training loss functions combining supervised, pseudo-label, and consistency terms."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from monai.losses import DiceCELoss

logger = logging.getLogger(__name__)


class SelfTrainingLoss(nn.Module):
    """Combined loss for self-training with three components.

    Components:
        - Supervised loss: DiceCE between student logits and ground-truth labels.
        - Pseudo-label loss: DiceCE between student logits and pseudo-labels,
          optionally masked by confidence.
        - Consistency loss: MSE between student and teacher logits (detached).

    Args:
        supervised_weight: Weight for the supervised loss term.
        pseudo_label_weight: Weight for the pseudo-label loss term.
        consistency_weight: Weight for the consistency loss term.
    """

    def __init__(
        self,
        supervised_weight: float = 1.0,
        pseudo_label_weight: float = 0.5,
        consistency_weight: float = 0.1,
    ) -> None:
        super().__init__()

        self.supervised_weight = supervised_weight
        self.pseudo_label_weight = pseudo_label_weight
        self.consistency_weight = consistency_weight

        self.dice_ce = DiceCELoss(
            to_onehot_y=False,
            sigmoid=True,
            squared_pred=True,
            smooth_nr=0.0,
            smooth_dr=1e-6,
        )
        self.mse = nn.MSELoss()

        logger.info(
            "SelfTrainingLoss initialized (sup=%.2f, pseudo=%.2f, consist=%.2f)",
            supervised_weight,
            pseudo_label_weight,
            consistency_weight,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor | None = None,
        pseudo_labels: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
        confidence_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined self-training loss.

        Args:
            student_logits: Student model output logits, shape (B, C, ...).
            labels: Ground-truth segmentation labels, shape (B, C, ...). None if unsupervised.
            pseudo_labels: Pseudo-labels from teacher, shape (B, C, ...). None if not available.
            teacher_logits: Teacher model output logits, shape (B, C, ...). None if no consistency.
            confidence_mask: Binary mask for confident pseudo-label voxels, shape (B, 1, ...) or
                (B, C, ...). None means all voxels are used.

        Returns:
            Dictionary with keys "total", "supervised", "pseudo_label", "consistency",
            each containing a scalar tensor.
        """
        device = student_logits.device
        zero = torch.tensor(0.0, device=device)

        # Supervised loss
        supervised_loss = self.dice_ce(student_logits, labels) if labels is not None else zero

        # Pseudo-label loss
        if pseudo_labels is not None:
            if confidence_mask is not None:
                # Apply confidence mask: zero out unconfident voxels
                masked_logits = student_logits * confidence_mask
                masked_pseudo = pseudo_labels * confidence_mask
                pseudo_loss = self.dice_ce(masked_logits, masked_pseudo)
            else:
                pseudo_loss = self.dice_ce(student_logits, pseudo_labels)
        else:
            pseudo_loss = zero

        # Consistency loss (student vs detached teacher)
        consistency_loss = self.mse(student_logits, teacher_logits.detach()) if teacher_logits is not None else zero

        total = (
            self.supervised_weight * supervised_loss
            + self.pseudo_label_weight * pseudo_loss
            + self.consistency_weight * consistency_loss
        )

        return {
            "total": total,
            "supervised": supervised_loss,
            "pseudo_label": pseudo_loss,
            "consistency": consistency_loss,
        }


def apply_ramp_weight(loss_dict: dict[str, torch.Tensor], ramp_weight: float) -> dict[str, torch.Tensor]:
    """Scale pseudo-label and consistency losses by a ramp-up weight.

    The supervised loss is left unchanged. The total is recomputed.

    Args:
        loss_dict: Loss dictionary from SelfTrainingLoss.forward().
        ramp_weight: Ramp-up weight in [0.0, 1.0].

    Returns:
        New dictionary with ramped losses and recomputed total.
    """
    ramped_pseudo = loss_dict["pseudo_label"] * ramp_weight
    ramped_consistency = loss_dict["consistency"] * ramp_weight
    supervised = loss_dict["supervised"]

    total = supervised + ramped_pseudo + ramped_consistency

    return {
        "total": total,
        "supervised": supervised,
        "pseudo_label": ramped_pseudo,
        "consistency": ramped_consistency,
    }
