"""Curriculum scheduling for self-training confidence thresholds and loss ramp-up."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

CLASS_NAMES = ["TC", "WT", "ET"]

_VALID_SCHEDULES = ("linear", "cosine", "step")


class ThresholdScheduler:
    """Schedules per-class confidence thresholds across self-training rounds.

    Thresholds decrease from initial to final over num_rounds, following
    a linear, cosine, or step schedule. Optional per-class offsets allow
    different classes to have shifted thresholds.

    Args:
        initial: Starting threshold (round 0).
        final: Ending threshold (last round).
        num_rounds: Total number of self-training rounds.
        schedule: Schedule type — "linear", "cosine", or "step".
        per_class_offsets: Optional dict mapping class names to offset values
            added to the global threshold (e.g., {"TC": 0.0, "WT": -0.05, "ET": 0.05}).

    Raises:
        ValueError: If schedule is not recognized or num_rounds < 1.
    """

    def __init__(
        self,
        initial: float = 0.95,
        final: float = 0.75,
        num_rounds: int = 4,
        schedule: str = "cosine",
        per_class_offsets: dict[str, float] | None = None,
    ) -> None:
        if schedule not in _VALID_SCHEDULES:
            raise ValueError(f"schedule must be one of {_VALID_SCHEDULES}, got '{schedule}'")
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")

        self.initial = initial
        self.final = final
        self.num_rounds = num_rounds
        self.schedule = schedule
        self.per_class_offsets = per_class_offsets or {name: 0.0 for name in CLASS_NAMES}

        logger.info(
            "ThresholdScheduler initialized (initial=%.3f, final=%.3f, rounds=%d, schedule=%s)",
            initial,
            final,
            num_rounds,
            schedule,
        )

    def _compute_global(self, round_num: int) -> float:
        """Compute the global (class-agnostic) threshold for a given round.

        Args:
            round_num: Current round number (0-indexed).

        Returns:
            Global threshold value for this round.
        """
        if self.num_rounds == 1:
            return self.initial

        # Clamp round_num to valid range
        t = min(max(round_num, 0), self.num_rounds - 1) / (self.num_rounds - 1)

        if self.schedule == "linear":
            return self.initial + (self.final - self.initial) * t

        if self.schedule == "cosine":
            # Cosine annealing: smooth transition from initial to final
            return self.final + (self.initial - self.final) * 0.5 * (1.0 + math.cos(math.pi * t))

        # Step schedule: stay at initial for first half, jump to final
        if t < 0.5:
            return self.initial
        return self.final

    def get_thresholds(self, round_num: int) -> dict[str, float]:
        """Get per-class thresholds for a given round.

        Args:
            round_num: Current round number (0-indexed).

        Returns:
            Dictionary mapping class names to threshold values, clamped to [0.5, 1.0].
        """
        global_threshold = self._compute_global(round_num)

        thresholds = {}
        for name in CLASS_NAMES:
            offset = self.per_class_offsets.get(name, 0.0)
            value = global_threshold + offset
            # Clamp to valid range
            thresholds[name] = min(max(value, 0.5), 1.0)

        return thresholds


def get_loss_ramp_weight(epoch: int, ramp_up_epochs: int) -> float:
    """Compute linear ramp-up weight for pseudo-label and consistency losses.

    Weight ramps linearly from 0.0 to 1.0 over ramp_up_epochs, then stays at 1.0.

    Args:
        epoch: Current epoch number (0-indexed).
        ramp_up_epochs: Number of epochs over which to ramp up.

    Returns:
        Ramp weight in [0.0, 1.0].
    """
    if ramp_up_epochs <= 0:
        return 1.0
    return min(max(epoch / ramp_up_epochs, 0.0), 1.0)
