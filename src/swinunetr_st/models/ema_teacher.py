"""Exponential Moving Average (EMA) teacher model for self-training."""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMATeacher(nn.Module):
    """Maintains an EMA copy of a student model for generating pseudo-labels.

    During warmup, teacher parameters are copied directly from the student.
    After warmup, parameters are updated via exponential moving average:
        teacher = decay * teacher + (1 - decay) * student

    Args:
        student_model: The student model to track.
        decay: EMA decay rate in (0, 1). Higher means slower updates.
        warmup_steps: Number of initial steps using direct copy instead of EMA.
        device: Device to place the teacher model on. If None, uses the student's device.
    """

    def __init__(
        self,
        student_model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 100,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Deep-copy the student and freeze all parameters
        self.teacher = copy.deepcopy(student_model)
        if device is not None:
            self.teacher = self.teacher.to(device)

        for param in self.teacher.parameters():
            param.requires_grad = False

        self.teacher.eval()
        logger.info("EMATeacher initialized (decay=%.4f, warmup_steps=%d)", decay, warmup_steps)

    @torch.no_grad()
    def update(self, student_model: nn.Module) -> None:
        """Update teacher parameters from the student.

        During warmup (step_count < warmup_steps), parameters are copied directly.
        After warmup, parameters are updated via EMA.

        Args:
            student_model: The current student model to update from.
        """
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Direct copy during warmup
            for t_param, s_param in zip(self.teacher.parameters(), student_model.parameters(), strict=True):
                t_param.data.copy_(s_param.data)
        else:
            # EMA update after warmup
            for t_param, s_param in zip(self.teacher.parameters(), student_model.parameters(), strict=True):
                t_param.data.mul_(self.decay).add_(s_param.data, alpha=1.0 - self.decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference through the teacher model in eval mode.

        Args:
            x: Input tensor.

        Returns:
            Teacher model output tensor.
        """
        self.teacher.eval()
        return self.teacher(x)

    def state_dict(self, *args, **kwargs) -> dict:
        """Return full state including EMA metadata.

        Returns:
            Dictionary with teacher model state, step_count, decay, and warmup_steps.
        """
        return {
            "teacher_state": self.teacher.state_dict(),
            "step_count": self.step_count,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Restore full state including EMA metadata.

        Args:
            state_dict: Dictionary from a previous state_dict() call.
            strict: Whether to strictly enforce matching keys in teacher model state.
        """
        self.teacher.load_state_dict(state_dict["teacher_state"], strict=strict)
        self.step_count = state_dict["step_count"]
        self.decay = state_dict["decay"]
        self.warmup_steps = state_dict["warmup_steps"]

        # Re-freeze parameters after loading
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.teacher.eval()
        logger.info(
            "EMATeacher state loaded (step_count=%d, decay=%.4f, warmup_steps=%d)",
            self.step_count,
            self.decay,
            self.warmup_steps,
        )
