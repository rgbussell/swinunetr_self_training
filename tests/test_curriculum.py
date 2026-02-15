"""Tests for curriculum scheduling: threshold schedules and loss ramp-up."""

from __future__ import annotations

import math

import pytest

from swinunetr_st.training.curriculum import CLASS_NAMES, ThresholdScheduler, get_loss_ramp_weight


class TestThresholdSchedulerInit:
    """Tests for ThresholdScheduler initialization."""

    def test_default_init(self) -> None:
        """Default scheduler should initialize without error."""
        scheduler = ThresholdScheduler()
        assert scheduler.initial == 0.95
        assert scheduler.final == 0.75
        assert scheduler.num_rounds == 4
        assert scheduler.schedule == "cosine"

    def test_invalid_schedule(self) -> None:
        """Invalid schedule type should raise ValueError."""
        with pytest.raises(ValueError, match="schedule must be one of"):
            ThresholdScheduler(schedule="exponential")

    def test_invalid_num_rounds(self) -> None:
        """num_rounds < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="num_rounds must be >= 1"):
            ThresholdScheduler(num_rounds=0)

    def test_per_class_offsets_default(self) -> None:
        """Without explicit offsets, all classes should have 0.0 offset."""
        scheduler = ThresholdScheduler()
        for name in CLASS_NAMES:
            assert scheduler.per_class_offsets[name] == 0.0

    def test_custom_per_class_offsets(self) -> None:
        """Custom offsets should be stored correctly."""
        offsets = {"TC": 0.0, "WT": -0.05, "ET": 0.05}
        scheduler = ThresholdScheduler(per_class_offsets=offsets)
        assert scheduler.per_class_offsets == offsets


class TestLinearSchedule:
    """Tests for the linear threshold schedule."""

    def test_round_zero_is_initial(self) -> None:
        """Round 0 should return the initial threshold."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=4, schedule="linear")
        global_thresh = scheduler._compute_global(0)
        assert abs(global_thresh - 0.95) < 1e-6

    def test_last_round_is_final(self) -> None:
        """Last round should return the final threshold."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=4, schedule="linear")
        global_thresh = scheduler._compute_global(3)
        assert abs(global_thresh - 0.75) < 1e-6

    def test_linear_midpoint(self) -> None:
        """Middle round should be halfway between initial and final."""
        scheduler = ThresholdScheduler(initial=1.0, final=0.5, num_rounds=5, schedule="linear")
        # Round 2 out of 5: t = 2/4 = 0.5
        global_thresh = scheduler._compute_global(2)
        assert abs(global_thresh - 0.75) < 1e-6

    def test_linear_monotonically_decreasing(self) -> None:
        """Linear schedule should be monotonically decreasing when initial > final."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=6, schedule="linear")
        values = [scheduler._compute_global(r) for r in range(6)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1], "Linear schedule should decrease monotonically"


class TestCosineSchedule:
    """Tests for the cosine threshold schedule."""

    def test_round_zero_is_initial(self) -> None:
        """Round 0 should return the initial threshold for cosine."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=4, schedule="cosine")
        global_thresh = scheduler._compute_global(0)
        assert abs(global_thresh - 0.95) < 1e-6

    def test_last_round_is_final(self) -> None:
        """Last round should return the final threshold for cosine."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=4, schedule="cosine")
        global_thresh = scheduler._compute_global(3)
        assert abs(global_thresh - 0.75) < 1e-6

    def test_cosine_midpoint(self) -> None:
        """Cosine midpoint should be the average of initial and final."""
        scheduler = ThresholdScheduler(initial=1.0, final=0.5, num_rounds=5, schedule="cosine")
        # At t=0.5: final + (initial - final) * 0.5 * (1 + cos(pi*0.5)) = 0.5 + 0.5*0.5*(1+0) = 0.75
        global_thresh = scheduler._compute_global(2)
        assert abs(global_thresh - 0.75) < 1e-6

    def test_cosine_monotonically_decreasing(self) -> None:
        """Cosine schedule should be monotonically decreasing when initial > final."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=10, schedule="cosine")
        values = [scheduler._compute_global(r) for r in range(10)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1] + 1e-10, "Cosine schedule should decrease monotonically"

    def test_cosine_formula(self) -> None:
        """Verify exact cosine formula at a specific point."""
        scheduler = ThresholdScheduler(initial=0.9, final=0.6, num_rounds=7, schedule="cosine")
        round_num = 3
        t = 3 / 6
        expected = 0.6 + (0.9 - 0.6) * 0.5 * (1.0 + math.cos(math.pi * t))
        actual = scheduler._compute_global(round_num)
        assert abs(actual - expected) < 1e-10


class TestStepSchedule:
    """Tests for the step threshold schedule."""

    def test_first_half_is_initial(self) -> None:
        """First half of rounds should use initial threshold."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=6, schedule="step")
        # Rounds 0, 1, 2 should be at initial (t < 0.5)
        for r in range(3):
            global_thresh = scheduler._compute_global(r)
            assert abs(global_thresh - 0.95) < 1e-6, f"Round {r} should be at initial"

    def test_second_half_is_final(self) -> None:
        """Second half of rounds should use final threshold."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=6, schedule="step")
        # Rounds 3, 4, 5 should be at final (t >= 0.5)
        for r in range(3, 6):
            global_thresh = scheduler._compute_global(r)
            assert abs(global_thresh - 0.75) < 1e-6, f"Round {r} should be at final"

    def test_single_round_step(self) -> None:
        """With 1 round, step should return initial."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=1, schedule="step")
        assert abs(scheduler._compute_global(0) - 0.95) < 1e-6


class TestGetThresholds:
    """Tests for get_thresholds with per-class offsets."""

    def test_all_classes_present(self) -> None:
        """Output should have all three class names."""
        scheduler = ThresholdScheduler(num_rounds=4)
        thresholds = scheduler.get_thresholds(0)
        assert set(thresholds.keys()) == set(CLASS_NAMES)

    def test_no_offsets_all_equal(self) -> None:
        """Without offsets, all classes should have the same threshold."""
        scheduler = ThresholdScheduler(initial=0.9, final=0.7, num_rounds=4, schedule="linear")
        thresholds = scheduler.get_thresholds(0)
        values = list(thresholds.values())
        assert all(abs(v - values[0]) < 1e-6 for v in values)

    def test_per_class_offsets_applied(self) -> None:
        """Per-class offsets should shift thresholds appropriately."""
        offsets = {"TC": 0.0, "WT": -0.05, "ET": 0.03}
        scheduler = ThresholdScheduler(
            initial=0.9, final=0.7, num_rounds=4, schedule="linear", per_class_offsets=offsets
        )
        thresholds = scheduler.get_thresholds(0)
        assert abs(thresholds["TC"] - 0.9) < 1e-6
        assert abs(thresholds["WT"] - 0.85) < 1e-6
        assert abs(thresholds["ET"] - 0.93) < 1e-6

    def test_clamping_upper_bound(self) -> None:
        """Thresholds should be clamped at 1.0."""
        offsets = {"TC": 0.0, "WT": 0.0, "ET": 0.2}
        scheduler = ThresholdScheduler(
            initial=0.95, final=0.75, num_rounds=4, schedule="linear", per_class_offsets=offsets
        )
        thresholds = scheduler.get_thresholds(0)
        assert thresholds["ET"] == 1.0, "ET threshold should be clamped to 1.0"

    def test_clamping_lower_bound(self) -> None:
        """Thresholds should be clamped at 0.5."""
        offsets = {"TC": 0.0, "WT": -0.3, "ET": 0.0}
        scheduler = ThresholdScheduler(
            initial=0.75, final=0.55, num_rounds=4, schedule="linear", per_class_offsets=offsets
        )
        thresholds = scheduler.get_thresholds(3)  # final = 0.55, WT offset = -0.3 -> 0.25 -> clamped to 0.5
        assert thresholds["WT"] == 0.5, "WT threshold should be clamped to 0.5"

    def test_negative_round_clamped(self) -> None:
        """Negative round number should be clamped to 0."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=4, schedule="linear")
        thresholds_neg = scheduler.get_thresholds(-1)
        thresholds_zero = scheduler.get_thresholds(0)
        assert thresholds_neg == thresholds_zero

    def test_round_beyond_max_clamped(self) -> None:
        """Round beyond num_rounds should be clamped to last round."""
        scheduler = ThresholdScheduler(initial=0.95, final=0.75, num_rounds=4, schedule="linear")
        thresholds_over = scheduler.get_thresholds(100)
        thresholds_last = scheduler.get_thresholds(3)
        assert thresholds_over == thresholds_last


class TestSingleRound:
    """Tests for single-round edge case."""

    def test_single_round_returns_initial(self) -> None:
        """With num_rounds=1, every round should return initial threshold."""
        for schedule in ("linear", "cosine", "step"):
            scheduler = ThresholdScheduler(initial=0.9, final=0.6, num_rounds=1, schedule=schedule)
            assert abs(scheduler._compute_global(0) - 0.9) < 1e-6, f"Failed for schedule={schedule}"


class TestGetLossRampWeight:
    """Tests for get_loss_ramp_weight function."""

    def test_epoch_zero_is_zero(self) -> None:
        """At epoch 0, ramp weight should be 0."""
        assert abs(get_loss_ramp_weight(0, 10)) < 1e-6

    def test_epoch_at_ramp_end_is_one(self) -> None:
        """At the ramp-up boundary, weight should be 1.0."""
        assert abs(get_loss_ramp_weight(10, 10) - 1.0) < 1e-6

    def test_epoch_beyond_ramp_is_one(self) -> None:
        """After ramp-up, weight should stay at 1.0."""
        assert abs(get_loss_ramp_weight(20, 10) - 1.0) < 1e-6

    def test_midpoint_ramp(self) -> None:
        """At midpoint, weight should be 0.5."""
        assert abs(get_loss_ramp_weight(5, 10) - 0.5) < 1e-6

    def test_zero_ramp_epochs(self) -> None:
        """With ramp_up_epochs=0, weight should always be 1.0."""
        assert abs(get_loss_ramp_weight(0, 0) - 1.0) < 1e-6
        assert abs(get_loss_ramp_weight(5, 0) - 1.0) < 1e-6

    def test_negative_ramp_epochs(self) -> None:
        """Negative ramp_up_epochs treated like 0: always returns 1.0."""
        assert abs(get_loss_ramp_weight(0, -5) - 1.0) < 1e-6

    def test_negative_epoch_clamped(self) -> None:
        """Negative epoch should be clamped to 0."""
        assert abs(get_loss_ramp_weight(-3, 10)) < 1e-6
