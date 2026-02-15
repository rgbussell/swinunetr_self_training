"""Tests for self-training loss functions."""

from __future__ import annotations

import pytest
import torch

from swinunetr_st.models.losses import SelfTrainingLoss, apply_ramp_weight


@pytest.fixture
def loss_fn() -> SelfTrainingLoss:
    """Create a default SelfTrainingLoss."""
    return SelfTrainingLoss(supervised_weight=1.0, pseudo_label_weight=0.5, consistency_weight=0.1)


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    """Create a small batch of test tensors (B=2, C=3, spatial=8x8x8)."""
    torch.manual_seed(0)
    student_logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    labels = torch.zeros(2, 3, 8, 8, 8)
    labels[:, 0, 2:6, 2:6, 2:6] = 1.0  # TC
    labels[:, 1, 1:7, 1:7, 1:7] = 1.0  # WT
    labels[:, 2, 3:5, 3:5, 3:5] = 1.0  # ET
    pseudo_labels = labels.clone()
    teacher_logits = torch.randn(2, 3, 8, 8, 8)
    confidence_mask = torch.ones(2, 1, 8, 8, 8)
    confidence_mask[:, :, 0:2, :, :] = 0.0  # mask out some voxels

    return {
        "student_logits": student_logits,
        "labels": labels,
        "pseudo_labels": pseudo_labels,
        "teacher_logits": teacher_logits,
        "confidence_mask": confidence_mask,
    }


class TestSelfTrainingLossSupervisedOnly:
    """Tests for supervised-only mode."""

    def test_supervised_only(self, loss_fn: SelfTrainingLoss, batch: dict) -> None:
        """With only labels, pseudo and consistency should be zero."""
        result = loss_fn(batch["student_logits"], labels=batch["labels"])
        assert result["supervised"].item() > 0, "Supervised loss should be positive"
        assert result["pseudo_label"].item() == 0.0, "Pseudo loss should be zero"
        assert result["consistency"].item() == 0.0, "Consistency loss should be zero"
        assert abs(result["total"].item() - result["supervised"].item()) < 1e-6

    def test_supervised_gradient_flows(self, loss_fn: SelfTrainingLoss, batch: dict) -> None:
        """Supervised loss should allow gradient flow to student logits."""
        result = loss_fn(batch["student_logits"], labels=batch["labels"])
        result["total"].backward()
        assert batch["student_logits"].grad is not None, "Gradients should flow to student logits"
        assert batch["student_logits"].grad.abs().sum() > 0, "Gradients should be non-zero"


class TestSelfTrainingLossPseudoOnly:
    """Tests for pseudo-label-only mode."""

    def test_pseudo_only(self, loss_fn: SelfTrainingLoss, batch: dict) -> None:
        """With only pseudo-labels, supervised and consistency should be zero."""
        result = loss_fn(batch["student_logits"], pseudo_labels=batch["pseudo_labels"])
        assert result["pseudo_label"].item() > 0, "Pseudo loss should be positive"
        assert result["supervised"].item() == 0.0, "Supervised loss should be zero"
        assert result["consistency"].item() == 0.0, "Consistency loss should be zero"

    def test_pseudo_with_confidence_mask(self, loss_fn: SelfTrainingLoss, batch: dict) -> None:
        """Confidence mask should change the pseudo-label loss value."""
        result_no_mask = loss_fn(batch["student_logits"], pseudo_labels=batch["pseudo_labels"])
        result_with_mask = loss_fn(
            batch["student_logits"],
            pseudo_labels=batch["pseudo_labels"],
            confidence_mask=batch["confidence_mask"],
        )
        # Masked loss should differ from unmasked
        assert result_with_mask["pseudo_label"].item() != pytest.approx(
            result_no_mask["pseudo_label"].item(), abs=1e-4
        ), "Confidence mask should change pseudo loss"


class TestSelfTrainingLossConsistencyOnly:
    """Tests for consistency-only mode."""

    def test_consistency_only(self, loss_fn: SelfTrainingLoss, batch: dict) -> None:
        """With only teacher logits, supervised and pseudo should be zero."""
        result = loss_fn(batch["student_logits"], teacher_logits=batch["teacher_logits"])
        assert result["consistency"].item() > 0, "Consistency loss should be positive"
        assert result["supervised"].item() == 0.0, "Supervised loss should be zero"
        assert result["pseudo_label"].item() == 0.0, "Pseudo loss should be zero"

    def test_teacher_logits_detached(self, loss_fn: SelfTrainingLoss) -> None:
        """Teacher logits should be detached (no gradient flow to teacher)."""
        student_logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
        teacher_logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)

        result = loss_fn(student_logits, teacher_logits=teacher_logits)
        result["total"].backward()

        # Student should get gradients
        assert student_logits.grad is not None, "Student should get gradients"
        # Teacher should NOT get gradients (detached in forward)
        assert teacher_logits.grad is None, "Teacher logits should be detached (no grad)"


class TestSelfTrainingLossAllComponents:
    """Tests with all three loss components active."""

    def test_all_three_components(self, loss_fn: SelfTrainingLoss, batch: dict) -> None:
        """All three losses should be non-zero and total should combine them."""
        result = loss_fn(
            batch["student_logits"],
            labels=batch["labels"],
            pseudo_labels=batch["pseudo_labels"],
            teacher_logits=batch["teacher_logits"],
        )
        assert result["supervised"].item() > 0
        assert result["pseudo_label"].item() > 0
        assert result["consistency"].item() > 0

        expected_total = (
            1.0 * result["supervised"].item()
            + 0.5 * result["pseudo_label"].item()
            + 0.1 * result["consistency"].item()
        )
        assert abs(result["total"].item() - expected_total) < 1e-4

    def test_custom_weights(self, batch: dict) -> None:
        """Custom weights should scale loss components correctly."""
        loss_fn = SelfTrainingLoss(supervised_weight=2.0, pseudo_label_weight=1.0, consistency_weight=0.5)
        result = loss_fn(
            batch["student_logits"],
            labels=batch["labels"],
            pseudo_labels=batch["pseudo_labels"],
            teacher_logits=batch["teacher_logits"],
        )

        expected_total = (
            2.0 * result["supervised"].item()
            + 1.0 * result["pseudo_label"].item()
            + 0.5 * result["consistency"].item()
        )
        assert abs(result["total"].item() - expected_total) < 1e-4

    def test_no_inputs_produces_zero_total(self, loss_fn: SelfTrainingLoss) -> None:
        """With no labels, pseudo-labels, or teacher logits, total should be zero."""
        student_logits = torch.randn(2, 3, 8, 8, 8)
        result = loss_fn(student_logits)
        assert result["total"].item() == 0.0

    def test_output_keys(self, loss_fn: SelfTrainingLoss, batch: dict) -> None:
        """Forward should return exactly the expected keys."""
        result = loss_fn(batch["student_logits"], labels=batch["labels"])
        assert set(result.keys()) == {"total", "supervised", "pseudo_label", "consistency"}

    def test_all_values_are_scalars(self, loss_fn: SelfTrainingLoss, batch: dict) -> None:
        """All loss values should be scalar tensors."""
        result = loss_fn(
            batch["student_logits"],
            labels=batch["labels"],
            pseudo_labels=batch["pseudo_labels"],
            teacher_logits=batch["teacher_logits"],
        )
        for key, val in result.items():
            assert isinstance(val, torch.Tensor), f"{key} should be a tensor"
            assert val.dim() == 0, f"{key} should be a scalar (0-dim), got {val.dim()}-dim"


class TestSelfTrainingLossGradientFlow:
    """Tests for gradient flow behavior."""

    def test_gradient_flows_through_all_components(self) -> None:
        """Gradients should flow from all active components to student logits."""
        loss_fn = SelfTrainingLoss(supervised_weight=1.0, pseudo_label_weight=0.5, consistency_weight=0.1)
        student_logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
        labels = torch.zeros(2, 3, 8, 8, 8)
        labels[:, 0, 2:6, 2:6, 2:6] = 1.0
        pseudo_labels = labels.clone()
        teacher_logits = torch.randn(2, 3, 8, 8, 8)

        result = loss_fn(
            student_logits,
            labels=labels,
            pseudo_labels=pseudo_labels,
            teacher_logits=teacher_logits,
        )
        result["total"].backward()
        assert student_logits.grad is not None
        assert student_logits.grad.abs().sum() > 0


class TestApplyRampWeight:
    """Tests for apply_ramp_weight function."""

    def test_ramp_zero_zeros_pseudo_and_consistency(self) -> None:
        """Ramp weight 0 should zero out pseudo and consistency."""
        loss_dict = {
            "total": torch.tensor(1.0),
            "supervised": torch.tensor(0.5),
            "pseudo_label": torch.tensor(0.3),
            "consistency": torch.tensor(0.2),
        }
        ramped = apply_ramp_weight(loss_dict, 0.0)
        assert ramped["pseudo_label"].item() == 0.0
        assert ramped["consistency"].item() == 0.0
        assert abs(ramped["supervised"].item() - 0.5) < 1e-6
        assert abs(ramped["total"].item() - 0.5) < 1e-6

    def test_ramp_one_no_change(self) -> None:
        """Ramp weight 1.0 should not change pseudo and consistency."""
        loss_dict = {
            "total": torch.tensor(1.0),
            "supervised": torch.tensor(0.5),
            "pseudo_label": torch.tensor(0.3),
            "consistency": torch.tensor(0.2),
        }
        ramped = apply_ramp_weight(loss_dict, 1.0)
        assert abs(ramped["pseudo_label"].item() - 0.3) < 1e-6
        assert abs(ramped["consistency"].item() - 0.2) < 1e-6
        expected_total = 0.5 + 0.3 + 0.2
        assert abs(ramped["total"].item() - expected_total) < 1e-5

    def test_ramp_half(self) -> None:
        """Ramp weight 0.5 should halve pseudo and consistency."""
        loss_dict = {
            "total": torch.tensor(1.0),
            "supervised": torch.tensor(0.4),
            "pseudo_label": torch.tensor(0.4),
            "consistency": torch.tensor(0.2),
        }
        ramped = apply_ramp_weight(loss_dict, 0.5)
        assert abs(ramped["pseudo_label"].item() - 0.2) < 1e-6
        assert abs(ramped["consistency"].item() - 0.1) < 1e-6
        expected_total = 0.4 + 0.2 + 0.1
        assert abs(ramped["total"].item() - expected_total) < 1e-5

    def test_ramp_preserves_grad(self) -> None:
        """Ramp weight should preserve gradient computation."""
        supervised = torch.tensor(0.5, requires_grad=True)
        pseudo = torch.tensor(0.3, requires_grad=True)
        consistency = torch.tensor(0.2, requires_grad=True)
        loss_dict = {
            "total": supervised + pseudo + consistency,
            "supervised": supervised,
            "pseudo_label": pseudo,
            "consistency": consistency,
        }
        ramped = apply_ramp_weight(loss_dict, 0.5)
        ramped["total"].backward()
        assert supervised.grad is not None
        assert pseudo.grad is not None
        assert consistency.grad is not None
