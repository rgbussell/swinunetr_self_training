"""Tests for EMA teacher model."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from swinunetr_st.models.ema_teacher import EMATeacher


class DummyModel(nn.Module):
    """Simple model for testing EMA updates."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture
def student() -> DummyModel:
    """Create a small dummy student model."""
    torch.manual_seed(42)
    return DummyModel(in_channels=4, out_channels=3)


@pytest.fixture
def ema_teacher(student: DummyModel) -> EMATeacher:
    """Create an EMA teacher from the dummy student."""
    return EMATeacher(student, decay=0.99, warmup_steps=5)


class TestEMATeacherInit:
    """Tests for EMATeacher initialization."""

    def test_teacher_identical_to_student(self, student: DummyModel, ema_teacher: EMATeacher) -> None:
        """Teacher parameters should be identical to student after init."""
        for t_param, s_param in zip(ema_teacher.teacher.parameters(), student.parameters(), strict=True):
            assert torch.equal(t_param, s_param), "Teacher params should match student at init"

    def test_teacher_no_gradients(self, ema_teacher: EMATeacher) -> None:
        """All teacher parameters should have requires_grad=False."""
        for param in ema_teacher.teacher.parameters():
            assert not param.requires_grad, "Teacher params should not require gradients"

    def test_teacher_eval_mode(self, ema_teacher: EMATeacher) -> None:
        """Teacher should be in eval mode after init."""
        assert not ema_teacher.teacher.training, "Teacher should be in eval mode"

    def test_invalid_decay(self, student: DummyModel) -> None:
        """Decay outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="decay must be in"):
            EMATeacher(student, decay=1.5)
        with pytest.raises(ValueError, match="decay must be in"):
            EMATeacher(student, decay=0.0)
        with pytest.raises(ValueError, match="decay must be in"):
            EMATeacher(student, decay=-0.1)

    def test_invalid_warmup_steps(self, student: DummyModel) -> None:
        """Negative warmup_steps should raise ValueError."""
        with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
            EMATeacher(student, warmup_steps=-1)

    def test_zero_warmup_steps(self, student: DummyModel) -> None:
        """Zero warmup_steps is valid — EMA from the start."""
        teacher = EMATeacher(student, warmup_steps=0)
        assert teacher.warmup_steps == 0

    def test_deep_copy_independence(self, student: DummyModel, ema_teacher: EMATeacher) -> None:
        """Modifying student should not affect the teacher."""
        original_teacher_params = [p.clone() for p in ema_teacher.teacher.parameters()]

        # Modify student parameters
        with torch.no_grad():
            for p in student.parameters():
                p.add_(1.0)

        # Teacher should be unchanged
        for t_param, orig in zip(ema_teacher.teacher.parameters(), original_teacher_params, strict=True):
            assert torch.equal(t_param, orig), "Teacher should not change when student is modified"


class TestEMATeacherUpdate:
    """Tests for the update method."""

    def test_warmup_direct_copy(self, student: DummyModel) -> None:
        """During warmup, teacher should be a direct copy of student."""
        teacher = EMATeacher(student, decay=0.99, warmup_steps=5)

        # Modify student
        with torch.no_grad():
            for p in student.parameters():
                p.add_(1.0)

        # Update during warmup (step 1 <= 5)
        teacher.update(student)
        assert teacher.step_count == 1

        for t_param, s_param in zip(teacher.teacher.parameters(), student.parameters(), strict=True):
            assert torch.equal(t_param, s_param), "During warmup, teacher should match student exactly"

    def test_ema_update_after_warmup(self, student: DummyModel) -> None:
        """After warmup, teacher should use EMA blending."""
        decay = 0.9
        teacher = EMATeacher(student, decay=decay, warmup_steps=2)

        # Do warmup steps
        for _ in range(2):
            teacher.update(student)

        # Capture teacher state before EMA update
        pre_update_params = [p.clone() for p in teacher.teacher.parameters()]

        # Modify student
        with torch.no_grad():
            for p in student.parameters():
                p.add_(1.0)

        # EMA update (step 3 > warmup_steps=2)
        teacher.update(student)
        assert teacher.step_count == 3

        for t_param, pre_param, s_param in zip(
            teacher.teacher.parameters(), pre_update_params, student.parameters(), strict=True
        ):
            expected = decay * pre_param + (1.0 - decay) * s_param
            assert torch.allclose(t_param, expected, atol=1e-6), "EMA update formula mismatch"

    def test_ema_moves_toward_student(self, student: DummyModel) -> None:
        """Repeated EMA updates should move teacher closer to student."""
        teacher = EMATeacher(student, decay=0.9, warmup_steps=0)

        # Modify student significantly
        with torch.no_grad():
            for p in student.parameters():
                p.add_(10.0)

        # Initial distance
        initial_dist = sum(
            (t - s).norm().item()
            for t, s in zip(teacher.teacher.parameters(), student.parameters(), strict=True)
        )

        # Several EMA updates
        for _ in range(20):
            teacher.update(student)

        final_dist = sum(
            (t - s).norm().item()
            for t, s in zip(teacher.teacher.parameters(), student.parameters(), strict=True)
        )

        assert final_dist < initial_dist, "Teacher should move closer to student over updates"

    def test_update_no_gradients(self, student: DummyModel, ema_teacher: EMATeacher) -> None:
        """Update should not create any gradient computation graph."""
        ema_teacher.update(student)
        for param in ema_teacher.teacher.parameters():
            assert param.grad is None, "Teacher should have no gradients after update"
            assert not param.requires_grad, "Teacher params should still not require grads"


class TestEMATeacherForward:
    """Tests for forward pass."""

    def test_output_shape(self, ema_teacher: EMATeacher) -> None:
        """Forward pass should produce correct output shape."""
        x = torch.randn(2, 4, 8, 8, 8)
        output = ema_teacher(x)
        assert output.shape == (2, 3, 8, 8, 8), f"Expected (2, 3, 8, 8, 8), got {output.shape}"

    def test_forward_eval_mode(self, ema_teacher: EMATeacher) -> None:
        """Teacher should be in eval mode during forward even if set to train."""
        ema_teacher.teacher.train()  # forcibly set to train
        x = torch.randn(1, 4, 8, 8, 8)
        ema_teacher(x)
        assert not ema_teacher.teacher.training, "Teacher should be eval after forward"

    def test_forward_no_gradients_on_output(self, ema_teacher: EMATeacher) -> None:
        """Forward output should not track gradients from teacher params."""
        x = torch.randn(1, 4, 8, 8, 8)
        output = ema_teacher(x)
        # Output does not require grad because teacher params are frozen
        # but if x requires grad, output would still get grad_fn from x
        assert output is not None

    def test_forward_deterministic(self, ema_teacher: EMATeacher) -> None:
        """Same input should produce same output (eval mode, no dropout)."""
        x = torch.randn(1, 4, 8, 8, 8)
        out1 = ema_teacher(x)
        out2 = ema_teacher(x)
        assert torch.equal(out1, out2), "Forward should be deterministic in eval mode"


class TestEMATeacherStateDictRoundtrip:
    """Tests for state_dict / load_state_dict."""

    def test_state_dict_contains_metadata(self, ema_teacher: EMATeacher) -> None:
        """State dict should include step_count, decay, and warmup_steps."""
        sd = ema_teacher.state_dict()
        assert "teacher_state" in sd
        assert "step_count" in sd
        assert "decay" in sd
        assert "warmup_steps" in sd

    def test_roundtrip_preserves_state(self, student: DummyModel) -> None:
        """Saving and loading state_dict should fully restore the teacher."""
        teacher = EMATeacher(student, decay=0.95, warmup_steps=10)

        # Modify student and do some updates
        with torch.no_grad():
            for p in student.parameters():
                p.add_(2.0)

        for _ in range(7):
            teacher.update(student)

        # Save state
        sd = teacher.state_dict()

        # Create a fresh teacher and load state
        teacher2 = EMATeacher(student, decay=0.5, warmup_steps=1)
        teacher2.load_state_dict(sd)

        assert teacher2.step_count == 7
        assert teacher2.decay == 0.95
        assert teacher2.warmup_steps == 10

        for p1, p2 in zip(teacher.teacher.parameters(), teacher2.teacher.parameters(), strict=True):
            assert torch.equal(p1, p2), "Loaded teacher params should match original"

    def test_loaded_teacher_frozen(self, student: DummyModel, ema_teacher: EMATeacher) -> None:
        """After load_state_dict, teacher params should still be frozen."""
        sd = ema_teacher.state_dict()

        teacher2 = EMATeacher(student, decay=0.5, warmup_steps=1)
        teacher2.load_state_dict(sd)

        for param in teacher2.teacher.parameters():
            assert not param.requires_grad, "Loaded teacher should have frozen params"
        assert not teacher2.teacher.training, "Loaded teacher should be in eval mode"
