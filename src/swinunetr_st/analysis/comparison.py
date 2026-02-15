"""Comparison analysis between baseline and self-training results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

logger = logging.getLogger(__name__)

_CLASS_LABELS = ["TC", "WT", "ET", "Mean"]
_DICE_KEYS = ["dice_tc", "dice_wt", "dice_et", "dice_mean"]


class ResultsComparator:
    """Compare baseline and self-training segmentation results.

    Generates publication-quality figures and statistical tests for
    comparing baseline supervised performance against iterative
    self-training rounds.

    Args:
        baseline_metrics: Dict with keys like dice_tc, dice_wt, dice_et, dice_mean.
        round_metrics: List of dicts (one per self-training round) with the same keys.
    """

    def __init__(
        self,
        baseline_metrics: dict | None = None,
        round_metrics: list[dict] | None = None,
    ) -> None:
        self.baseline_metrics = baseline_metrics or {}
        self.round_metrics = round_metrics or []

    def load_from_files(self, baseline_csv: str, results_dir: str) -> None:
        """Load baseline per-subject CSV and per-round metrics JSON files.

        The baseline CSV is expected to have columns including the dice score
        keys (dice_tc, dice_wt, dice_et, dice_mean). Round metrics are loaded
        from JSON files named ``round_N_metrics.json`` found in *results_dir*.

        Args:
            baseline_csv: Path to a CSV with per-subject baseline metrics.
            results_dir: Directory containing round_N_metrics.json files.
        """
        baseline_path = Path(baseline_csv)
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline CSV not found: {baseline_path}")

        df = pd.read_csv(baseline_path)
        self.baseline_metrics = {key: float(df[key].mean()) for key in _DICE_KEYS if key in df.columns}
        logger.info("Loaded baseline metrics from %s (%d subjects)", baseline_path, len(df))

        results_path = Path(results_dir)
        if not results_path.is_dir():
            raise FileNotFoundError(f"Results directory not found: {results_path}")

        self.round_metrics = []
        round_idx = 0
        while True:
            json_path = results_path / f"round_{round_idx}_metrics.json"
            if not json_path.is_file():
                break
            with open(json_path) as f:
                self.round_metrics.append(json.load(f))
            round_idx += 1

        logger.info("Loaded %d round metric files from %s", len(self.round_metrics), results_path)

    def plot_dice_comparison(self, output_path: str | None = None) -> Figure:
        """Create a grouped bar chart comparing per-class Dice scores.

        Shows baseline vs each self-training round for TC, WT, ET, and Mean Dice.

        Args:
            output_path: If provided, save the figure to this path (PNG and PDF).

        Returns:
            Matplotlib Figure object.
        """
        sns.set_theme(style="whitegrid", palette="colorblind")

        groups = ["Baseline"] + [f"Round {i}" for i in range(len(self.round_metrics))]
        all_metrics = [self.baseline_metrics, *self.round_metrics]

        data_rows = []
        for group_name, metrics in zip(groups, all_metrics, strict=True):
            for label, key in zip(_CLASS_LABELS, _DICE_KEYS, strict=True):
                data_rows.append({"Group": group_name, "Class": label, "Dice": metrics.get(key, 0.0)})

        df = pd.DataFrame(data_rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="Class", y="Dice", hue="Group", ax=ax)
        ax.set_title("Dice Score Comparison: Baseline vs Self-Training Rounds")
        ax.set_ylabel("Dice Score")
        ax.set_xlabel("Tumor Class")
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="Stage", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

        if output_path is not None:
            self._save_figure(fig, output_path)

        return fig

    def plot_per_subject_scatter(self, output_path: str | None = None) -> Figure:
        """Create scatter plot of baseline vs self-trained Dice per subject.

        Requires that ``baseline_metrics`` and ``round_metrics`` each contain
        a ``per_subject_dice`` array (or list). Falls back to randomly sampling
        illustration data when these are not present.

        Args:
            output_path: If provided, save the figure to this path (PNG and PDF).

        Returns:
            Matplotlib Figure object.
        """
        sns.set_theme(style="whitegrid", palette="colorblind")

        baseline_scores = np.asarray(self.baseline_metrics.get("per_subject_dice", []))
        final_scores = np.asarray(
            self.round_metrics[-1].get("per_subject_dice", []) if self.round_metrics else []
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        if baseline_scores.size > 0 and final_scores.size > 0:
            n = min(len(baseline_scores), len(final_scores))
            ax.scatter(baseline_scores[:n], final_scores[:n], alpha=0.7, edgecolors="k", linewidths=0.5)
        else:
            logger.warning("No per-subject scores available; scatter plot will be empty.")

        # Diagonal reference line
        lims = [0.0, 1.0]
        ax.plot(lims, lims, "--", color="gray", linewidth=1, label="y = x")

        ax.set_title("Per-Subject Dice: Baseline vs Self-Trained")
        ax.set_xlabel("Baseline Dice")
        ax.set_ylabel("Self-Trained Dice")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()

        if output_path is not None:
            self._save_figure(fig, output_path)

        return fig

    def statistical_significance_test(
        self,
        baseline_scores: np.ndarray,
        final_scores: np.ndarray,
    ) -> dict:
        """Run a paired Wilcoxon signed-rank test between baseline and final scores.

        Args:
            baseline_scores: Array of per-subject Dice scores from baseline model.
            final_scores: Array of per-subject Dice scores from final self-trained model.

        Returns:
            Dict with keys: p_value, statistic, effect_size (Cohen's d), n_subjects.
        """
        baseline_scores = np.asarray(baseline_scores, dtype=np.float64)
        final_scores = np.asarray(final_scores, dtype=np.float64)

        if len(baseline_scores) != len(final_scores):
            raise ValueError(
                f"Score arrays must have equal length, got {len(baseline_scores)} vs {len(final_scores)}"
            )

        n_subjects = len(baseline_scores)
        if n_subjects < 2:
            logger.warning("Fewer than 2 subjects; statistical test may not be meaningful.")
            return {
                "p_value": 1.0,
                "statistic": 0.0,
                "effect_size": 0.0,
                "n_subjects": n_subjects,
            }

        diff = final_scores - baseline_scores

        # Cohen's d: mean difference / pooled std
        pooled_std = np.sqrt((np.var(baseline_scores, ddof=1) + np.var(final_scores, ddof=1)) / 2.0)
        effect_size = float(np.mean(diff) / pooled_std) if pooled_std > 0 else 0.0

        # Wilcoxon signed-rank test
        try:
            result = stats.wilcoxon(baseline_scores, final_scores)
            p_value = float(result.pvalue)
            statistic = float(result.statistic)
        except ValueError:
            # All differences are zero
            logger.warning("All differences are zero; Wilcoxon test not applicable.")
            p_value = 1.0
            statistic = 0.0

        return {
            "p_value": p_value,
            "statistic": statistic,
            "effect_size": effect_size,
            "n_subjects": n_subjects,
        }

    def generate_improvement_table(self) -> str:
        """Generate a Markdown table comparing baseline vs final round metrics.

        Returns:
            A formatted Markdown string with columns for each Dice metric,
            showing baseline, final, and delta values.
        """
        if not self.round_metrics:
            return (
                "| Metric | Baseline | Final | Delta |\n"
                "|--------|----------|-------|-------|\n"
                "| (no round data) | - | - | - |"
            )

        final = self.round_metrics[-1]

        lines = [
            "| Metric | Baseline | Final | Delta |",
            "|--------|----------|-------|-------|",
        ]

        for label, key in zip(_CLASS_LABELS, _DICE_KEYS, strict=True):
            b_val = self.baseline_metrics.get(key, 0.0)
            f_val = final.get(key, 0.0)
            delta = f_val - b_val
            sign = "+" if delta >= 0 else ""
            lines.append(f"| Dice {label} | {b_val:.4f} | {f_val:.4f} | {sign}{delta:.4f} |")

        return "\n".join(lines)

    @staticmethod
    def _save_figure(fig: Figure, output_path: str) -> None:
        """Save a figure as both PNG and PDF.

        Args:
            fig: Matplotlib figure to save.
            output_path: Base path (extension will be replaced for each format).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        png_path = path.with_suffix(".png")
        pdf_path = path.with_suffix(".pdf")

        fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
        fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
        logger.info("Saved figure to %s and %s", png_path, pdf_path)
