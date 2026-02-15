"""Tests for analysis modules: comparison and convergence."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from swinunetr_st.analysis.comparison import ResultsComparator
from swinunetr_st.analysis.convergence import ConvergenceAnalyzer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_metrics() -> dict:
    """Baseline model aggregate metrics."""
    return {
        "dice_tc": 0.84,
        "dice_wt": 0.91,
        "dice_et": 0.82,
        "dice_mean": 0.8567,
    }


@pytest.fixture
def round_metrics() -> list[dict]:
    """Four rounds of self-training metrics with improving scores."""
    return [
        {
            "dice_tc": 0.85,
            "dice_wt": 0.915,
            "dice_et": 0.83,
            "dice_mean": 0.865,
            "volumes_accepted": 12,
            "mean_confidence": 0.88,
            "supervised_loss": 0.35,
            "pseudo_label_loss": 0.42,
            "consistency_loss": 0.15,
            "epoch_history": [
                {
                    "dice_tc": 0.84,
                    "dice_wt": 0.91,
                    "dice_et": 0.82,
                    "dice_mean": 0.857,
                    "supervised_loss": 0.40,
                    "pseudo_label_loss": 0.50,
                    "consistency_loss": 0.20,
                },
                {
                    "dice_tc": 0.85,
                    "dice_wt": 0.915,
                    "dice_et": 0.83,
                    "dice_mean": 0.865,
                    "supervised_loss": 0.35,
                    "pseudo_label_loss": 0.42,
                    "consistency_loss": 0.15,
                },
            ],
        },
        {
            "dice_tc": 0.86,
            "dice_wt": 0.92,
            "dice_et": 0.845,
            "dice_mean": 0.875,
            "volumes_accepted": 18,
            "mean_confidence": 0.90,
            "supervised_loss": 0.30,
            "pseudo_label_loss": 0.38,
            "consistency_loss": 0.12,
            "epoch_history": [
                {
                    "dice_tc": 0.855,
                    "dice_wt": 0.917,
                    "dice_et": 0.835,
                    "dice_mean": 0.869,
                    "supervised_loss": 0.33,
                    "pseudo_label_loss": 0.40,
                    "consistency_loss": 0.14,
                },
                {
                    "dice_tc": 0.86,
                    "dice_wt": 0.92,
                    "dice_et": 0.845,
                    "dice_mean": 0.875,
                    "supervised_loss": 0.30,
                    "pseudo_label_loss": 0.38,
                    "consistency_loss": 0.12,
                },
            ],
        },
        {
            "dice_tc": 0.87,
            "dice_wt": 0.925,
            "dice_et": 0.855,
            "dice_mean": 0.8833,
            "volumes_accepted": 22,
            "mean_confidence": 0.92,
            "supervised_loss": 0.27,
            "pseudo_label_loss": 0.34,
            "consistency_loss": 0.10,
            "epoch_history": [
                {
                    "dice_tc": 0.865,
                    "dice_wt": 0.922,
                    "dice_et": 0.850,
                    "dice_mean": 0.879,
                    "supervised_loss": 0.29,
                    "pseudo_label_loss": 0.36,
                    "consistency_loss": 0.11,
                },
                {
                    "dice_tc": 0.87,
                    "dice_wt": 0.925,
                    "dice_et": 0.855,
                    "dice_mean": 0.8833,
                    "supervised_loss": 0.27,
                    "pseudo_label_loss": 0.34,
                    "consistency_loss": 0.10,
                },
            ],
        },
        {
            "dice_tc": 0.88,
            "dice_wt": 0.93,
            "dice_et": 0.865,
            "dice_mean": 0.8917,
            "volumes_accepted": 25,
            "mean_confidence": 0.93,
            "supervised_loss": 0.24,
            "pseudo_label_loss": 0.30,
            "consistency_loss": 0.08,
            "epoch_history": [
                {
                    "dice_tc": 0.875,
                    "dice_wt": 0.927,
                    "dice_et": 0.860,
                    "dice_mean": 0.887,
                    "supervised_loss": 0.26,
                    "pseudo_label_loss": 0.32,
                    "consistency_loss": 0.09,
                },
                {
                    "dice_tc": 0.88,
                    "dice_wt": 0.93,
                    "dice_et": 0.865,
                    "dice_mean": 0.8917,
                    "supervised_loss": 0.24,
                    "pseudo_label_loss": 0.30,
                    "consistency_loss": 0.08,
                },
            ],
        },
    ]


@pytest.fixture
def per_subject_scores() -> tuple[np.ndarray, np.ndarray]:
    """Arrays of 20 subjects with baseline and (improved) final scores."""
    rng = np.random.default_rng(42)
    baseline = rng.uniform(0.75, 0.92, size=20)
    # Final scores are baseline + small positive improvement + noise
    final = baseline + rng.uniform(0.01, 0.06, size=20)
    final = np.clip(final, 0.0, 1.0)
    return baseline, final


@pytest.fixture
def thresholds_per_round() -> list[dict]:
    """Per-class threshold schedule for 4 rounds."""
    return [
        {"TC": 0.95, "WT": 0.90, "ET": 1.00},
        {"TC": 0.90, "WT": 0.85, "ET": 0.95},
        {"TC": 0.85, "WT": 0.80, "ET": 0.90},
        {"TC": 0.80, "WT": 0.75, "ET": 0.85},
    ]


# ---------------------------------------------------------------------------
# ResultsComparator tests
# ---------------------------------------------------------------------------


class TestResultsComparator:
    """Tests for ResultsComparator."""

    def test_dice_comparison_returns_figure(self, baseline_metrics, round_metrics):
        """plot_dice_comparison returns a matplotlib Figure."""
        comp = ResultsComparator(baseline_metrics, round_metrics)
        fig = comp.plot_dice_comparison()
        assert isinstance(fig, Figure)

    def test_dice_comparison_saves_to_disk(self, baseline_metrics, round_metrics, tmp_path):
        """plot_dice_comparison saves PNG and PDF when output_path is given."""
        comp = ResultsComparator(baseline_metrics, round_metrics)
        out = str(tmp_path / "dice_comparison.png")
        fig = comp.plot_dice_comparison(output_path=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "dice_comparison.png").is_file()
        assert (tmp_path / "dice_comparison.pdf").is_file()

    def test_per_subject_scatter_returns_figure(self, baseline_metrics, round_metrics, per_subject_scores):
        """plot_per_subject_scatter returns a Figure with data."""
        baseline_scores, final_scores = per_subject_scores
        baseline_metrics["per_subject_dice"] = baseline_scores.tolist()
        round_metrics[-1]["per_subject_dice"] = final_scores.tolist()
        comp = ResultsComparator(baseline_metrics, round_metrics)
        fig = comp.plot_per_subject_scatter()
        assert isinstance(fig, Figure)

    def test_per_subject_scatter_empty(self, baseline_metrics, round_metrics):
        """Scatter plot handles missing per_subject_dice gracefully."""
        comp = ResultsComparator(baseline_metrics, round_metrics)
        fig = comp.plot_per_subject_scatter()
        assert isinstance(fig, Figure)

    def test_per_subject_scatter_saves_to_disk(
        self, baseline_metrics, round_metrics, per_subject_scores, tmp_path
    ):
        """plot_per_subject_scatter saves files when output_path given."""
        baseline_scores, final_scores = per_subject_scores
        baseline_metrics["per_subject_dice"] = baseline_scores.tolist()
        round_metrics[-1]["per_subject_dice"] = final_scores.tolist()
        comp = ResultsComparator(baseline_metrics, round_metrics)
        out = str(tmp_path / "scatter.png")
        fig = comp.plot_per_subject_scatter(output_path=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "scatter.png").is_file()
        assert (tmp_path / "scatter.pdf").is_file()

    def test_statistical_test_valid_p_value(self, per_subject_scores):
        """Wilcoxon test returns p_value in [0, 1]."""
        baseline_scores, final_scores = per_subject_scores
        comp = ResultsComparator()
        result = comp.statistical_significance_test(baseline_scores, final_scores)
        assert 0.0 <= result["p_value"] <= 1.0
        assert "statistic" in result
        assert "effect_size" in result
        assert result["n_subjects"] == 20

    def test_statistical_test_identical_scores(self):
        """Identical scores yield p_value = 1.0."""
        scores = np.ones(10) * 0.85
        comp = ResultsComparator()
        result = comp.statistical_significance_test(scores, scores)
        assert result["p_value"] == 1.0

    def test_statistical_test_mismatched_lengths(self):
        """Mismatched array lengths raise ValueError."""
        comp = ResultsComparator()
        with pytest.raises(ValueError, match="equal length"):
            comp.statistical_significance_test(np.ones(5), np.ones(10))

    def test_statistical_test_single_subject(self):
        """Single subject returns safe defaults."""
        comp = ResultsComparator()
        result = comp.statistical_significance_test(np.array([0.8]), np.array([0.9]))
        assert result["p_value"] == 1.0
        assert result["n_subjects"] == 1

    def test_improvement_table_valid_markdown(self, baseline_metrics, round_metrics):
        """generate_improvement_table returns a Markdown table string."""
        comp = ResultsComparator(baseline_metrics, round_metrics)
        table = comp.generate_improvement_table()
        assert isinstance(table, str)
        assert "| Metric" in table
        assert "| Dice TC" in table
        assert "|" in table
        # Check it has the right number of data rows (4 classes)
        data_lines = [line for line in table.strip().split("\n") if line.startswith("| Dice")]
        assert len(data_lines) == 4

    def test_improvement_table_no_rounds(self, baseline_metrics):
        """Table generation handles empty round_metrics."""
        comp = ResultsComparator(baseline_metrics, [])
        table = comp.generate_improvement_table()
        assert isinstance(table, str)
        assert "no round data" in table

    def test_load_from_files(self, tmp_path):
        """load_from_files reads a CSV and round JSON files."""
        import pandas as pd

        # Create baseline CSV
        csv_path = tmp_path / "baseline.csv"
        df = pd.DataFrame(
            {
                "subject": [f"sub_{i}" for i in range(5)],
                "dice_tc": [0.84, 0.83, 0.85, 0.84, 0.86],
                "dice_wt": [0.91, 0.90, 0.92, 0.91, 0.90],
                "dice_et": [0.82, 0.81, 0.83, 0.82, 0.84],
                "dice_mean": [0.8567, 0.8467, 0.8667, 0.8567, 0.8667],
            }
        )
        df.to_csv(csv_path, index=False)

        # Create round metric JSONs
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        import json

        for i in range(2):
            data = {"dice_tc": 0.85 + i * 0.01, "dice_wt": 0.92, "dice_et": 0.84, "dice_mean": 0.87}
            with open(results_dir / f"round_{i}_metrics.json", "w") as f:
                json.dump(data, f)

        comp = ResultsComparator()
        comp.load_from_files(str(csv_path), str(results_dir))
        assert "dice_tc" in comp.baseline_metrics
        assert len(comp.round_metrics) == 2

    def test_load_from_files_missing_csv(self, tmp_path):
        """Missing CSV raises FileNotFoundError."""
        comp = ResultsComparator()
        with pytest.raises(FileNotFoundError, match="Baseline CSV not found"):
            comp.load_from_files(str(tmp_path / "nonexistent.csv"), str(tmp_path))

    def test_load_from_files_missing_dir(self, tmp_path):
        """Missing results dir raises FileNotFoundError."""
        csv_path = tmp_path / "baseline.csv"
        csv_path.write_text("dice_tc\n0.84\n")
        comp = ResultsComparator()
        with pytest.raises(FileNotFoundError, match="Results directory not found"):
            comp.load_from_files(str(csv_path), str(tmp_path / "nonexistent_dir"))


# ---------------------------------------------------------------------------
# ConvergenceAnalyzer tests
# ---------------------------------------------------------------------------


class TestConvergenceAnalyzer:
    """Tests for ConvergenceAnalyzer."""

    def test_convergence_curves_returns_figure(self, round_metrics):
        """plot_convergence_curves returns a matplotlib Figure."""
        analyzer = ConvergenceAnalyzer(round_metrics=round_metrics)
        fig = analyzer.plot_convergence_curves()
        assert isinstance(fig, Figure)

    def test_convergence_curves_saves_to_disk(self, round_metrics, tmp_path):
        """plot_convergence_curves saves PNG and PDF."""
        analyzer = ConvergenceAnalyzer(round_metrics=round_metrics)
        out = str(tmp_path / "convergence.png")
        fig = analyzer.plot_convergence_curves(output_path=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "convergence.png").is_file()
        assert (tmp_path / "convergence.pdf").is_file()

    def test_loss_breakdown_returns_figure(self, round_metrics):
        """plot_loss_breakdown returns a matplotlib Figure."""
        analyzer = ConvergenceAnalyzer(round_metrics=round_metrics)
        fig = analyzer.plot_loss_breakdown()
        assert isinstance(fig, Figure)

    def test_loss_breakdown_saves_to_disk(self, round_metrics, tmp_path):
        """plot_loss_breakdown saves PNG and PDF."""
        analyzer = ConvergenceAnalyzer(round_metrics=round_metrics)
        out = str(tmp_path / "loss.png")
        fig = analyzer.plot_loss_breakdown(output_path=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "loss.png").is_file()
        assert (tmp_path / "loss.pdf").is_file()

    def test_pseudo_label_acceptance_returns_figure(self, round_metrics):
        """plot_pseudo_label_acceptance returns a matplotlib Figure."""
        analyzer = ConvergenceAnalyzer(round_metrics=round_metrics)
        fig = analyzer.plot_pseudo_label_acceptance()
        assert isinstance(fig, Figure)

    def test_pseudo_label_acceptance_saves_to_disk(self, round_metrics, tmp_path):
        """plot_pseudo_label_acceptance saves PNG and PDF."""
        analyzer = ConvergenceAnalyzer(round_metrics=round_metrics)
        out = str(tmp_path / "acceptance.png")
        fig = analyzer.plot_pseudo_label_acceptance(output_path=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "acceptance.png").is_file()
        assert (tmp_path / "acceptance.pdf").is_file()

    def test_threshold_schedule_returns_figure(self, thresholds_per_round):
        """plot_threshold_schedule returns a matplotlib Figure."""
        analyzer = ConvergenceAnalyzer(round_metrics=[])
        fig = analyzer.plot_threshold_schedule(thresholds_per_round)
        assert isinstance(fig, Figure)

    def test_threshold_schedule_saves_to_disk(self, thresholds_per_round, tmp_path):
        """plot_threshold_schedule saves PNG and PDF."""
        analyzer = ConvergenceAnalyzer(round_metrics=[])
        out = str(tmp_path / "thresholds.png")
        fig = analyzer.plot_threshold_schedule(thresholds_per_round, output_path=out)
        assert isinstance(fig, Figure)
        assert (tmp_path / "thresholds.png").is_file()
        assert (tmp_path / "thresholds.pdf").is_file()

    def test_threshold_schedule_empty(self):
        """Empty thresholds still returns a Figure."""
        analyzer = ConvergenceAnalyzer(round_metrics=[])
        fig = analyzer.plot_threshold_schedule([])
        assert isinstance(fig, Figure)

    def test_convergence_curves_single_round(self):
        """Single round produces a valid Figure without boundaries."""
        metrics = [
            {
                "dice_tc": 0.85,
                "dice_wt": 0.91,
                "dice_et": 0.83,
                "dice_mean": 0.863,
                "epoch_history": [
                    {"dice_tc": 0.85, "dice_wt": 0.91, "dice_et": 0.83, "dice_mean": 0.863}
                ],
            }
        ]
        analyzer = ConvergenceAnalyzer(round_metrics=metrics)
        fig = analyzer.plot_convergence_curves()
        assert isinstance(fig, Figure)

    def test_convergence_curves_no_epoch_history(self):
        """Rounds without epoch_history still produce a Figure."""
        metrics = [
            {"dice_tc": 0.85, "dice_wt": 0.91, "dice_et": 0.83, "dice_mean": 0.863},
            {"dice_tc": 0.87, "dice_wt": 0.92, "dice_et": 0.85, "dice_mean": 0.880},
        ]
        analyzer = ConvergenceAnalyzer(round_metrics=metrics)
        fig = analyzer.plot_convergence_curves()
        assert isinstance(fig, Figure)

    def test_empty_round_metrics(self):
        """Empty round metrics produce valid (empty) Figures."""
        analyzer = ConvergenceAnalyzer(round_metrics=[])
        fig_conv = analyzer.plot_convergence_curves()
        fig_loss = analyzer.plot_loss_breakdown()
        fig_accept = analyzer.plot_pseudo_label_acceptance()
        assert isinstance(fig_conv, Figure)
        assert isinstance(fig_loss, Figure)
        assert isinstance(fig_accept, Figure)

    def test_load_from_results_dir(self, tmp_path):
        """Initializing with results_dir loads JSON files."""
        import json

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        for i in range(3):
            data = {
                "dice_tc": 0.85 + i * 0.01,
                "dice_wt": 0.92,
                "dice_et": 0.84,
                "dice_mean": 0.87,
            }
            with open(results_dir / f"round_{i}_metrics.json", "w") as f:
                json.dump(data, f)

        analyzer = ConvergenceAnalyzer(results_dir=str(results_dir))
        assert len(analyzer.round_metrics) == 3

    def test_load_from_missing_dir(self):
        """Missing results dir raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Results directory not found"):
            ConvergenceAnalyzer(results_dir="/nonexistent/path")
