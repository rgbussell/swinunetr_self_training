"""Convergence analysis for self-training rounds."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class ConvergenceAnalyzer:
    """Analyze and visualize training convergence across self-training rounds.

    Generates publication-quality plots showing Dice progression, loss
    breakdowns, pseudo-label acceptance rates, and threshold schedules.

    Args:
        results_dir: Path to directory containing round metric JSON files.
        round_metrics: Pre-loaded list of per-round metric dicts. If provided,
            results_dir is ignored.
    """

    def __init__(
        self,
        results_dir: str | None = None,
        round_metrics: list[dict] | None = None,
    ) -> None:
        if round_metrics is not None:
            self.round_metrics = round_metrics
        elif results_dir is not None:
            self.round_metrics = self._load_round_metrics(results_dir)
        else:
            self.round_metrics = []

    @staticmethod
    def _load_round_metrics(results_dir: str) -> list[dict]:
        """Load round_N_metrics.json files from a directory.

        Args:
            results_dir: Directory to search for round metric files.

        Returns:
            List of metric dicts ordered by round index.
        """
        results_path = Path(results_dir)
        if not results_path.is_dir():
            raise FileNotFoundError(f"Results directory not found: {results_path}")

        metrics = []
        round_idx = 0
        while True:
            json_path = results_path / f"round_{round_idx}_metrics.json"
            if not json_path.is_file():
                break
            with open(json_path) as f:
                metrics.append(json.load(f))
            round_idx += 1

        logger.info("Loaded %d round metric files from %s", len(metrics), results_path)
        return metrics

    def plot_convergence_curves(self, output_path: str | None = None) -> Figure:
        """Plot Dice score over epochs across all self-training rounds.

        Round boundaries are marked with vertical dashed lines.

        Args:
            output_path: If provided, save the figure to this path (PNG and PDF).

        Returns:
            Matplotlib Figure object.
        """
        sns.set_theme(style="whitegrid", palette="colorblind")
        palette = sns.color_palette("colorblind", n_colors=4)

        fig, ax = plt.subplots(figsize=(10, 6))

        epoch_offset = 0
        class_keys = ["dice_tc", "dice_wt", "dice_et", "dice_mean"]
        class_labels = ["TC", "WT", "ET", "Mean"]

        # Accumulate epoch-level data across rounds
        all_epochs: dict[str, list[float]] = {k: [] for k in class_keys}
        all_x: list[float] = []
        round_boundaries: list[int] = []

        for round_idx, metrics in enumerate(self.round_metrics):
            epoch_history = metrics.get("epoch_history", [])
            if epoch_history:
                for ep_data in epoch_history:
                    for key in class_keys:
                        all_epochs[key].append(ep_data.get(key, 0.0))
                    all_x.append(epoch_offset)
                    epoch_offset += 1
            else:
                # Fallback: plot summary metric as single point per round
                for key in class_keys:
                    all_epochs[key].append(metrics.get(key, 0.0))
                all_x.append(epoch_offset)
                epoch_offset += 1

            if round_idx < len(self.round_metrics) - 1:
                round_boundaries.append(epoch_offset)

        for i, (key, label) in enumerate(zip(class_keys, class_labels, strict=True)):
            ax.plot(all_x, all_epochs[key], label=label, color=palette[i], linewidth=1.5)

        for boundary in round_boundaries:
            ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_title("Dice Score Convergence Across Self-Training Rounds")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice Score")
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="Class")
        fig.tight_layout()

        if output_path is not None:
            self._save_figure(fig, output_path)

        return fig

    def plot_loss_breakdown(self, output_path: str | None = None) -> Figure:
        """Plot supervised, pseudo-label, and consistency losses over epochs.

        Args:
            output_path: If provided, save the figure to this path (PNG and PDF).

        Returns:
            Matplotlib Figure object.
        """
        sns.set_theme(style="whitegrid", palette="colorblind")
        palette = sns.color_palette("colorblind", n_colors=3)

        fig, ax = plt.subplots(figsize=(10, 6))

        loss_keys = ["supervised_loss", "pseudo_label_loss", "consistency_loss"]
        loss_labels = ["Supervised", "Pseudo-Label", "Consistency"]

        all_losses: dict[str, list[float]] = {k: [] for k in loss_keys}
        all_x: list[float] = []
        epoch_offset = 0

        for metrics in self.round_metrics:
            epoch_history = metrics.get("epoch_history", [])

            if epoch_history:
                for ep_data in epoch_history:
                    for key in loss_keys:
                        all_losses[key].append(ep_data.get(key, 0.0))
                    all_x.append(epoch_offset)
                    epoch_offset += 1
            else:
                for key in loss_keys:
                    all_losses[key].append(metrics.get(key, 0.0))
                all_x.append(epoch_offset)
                epoch_offset += 1

        for i, (key, label) in enumerate(zip(loss_keys, loss_labels, strict=True)):
            ax.plot(all_x, all_losses[key], label=label, color=palette[i], linewidth=1.5)

        ax.set_title("Loss Breakdown Across Self-Training Rounds")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(title="Loss Component")
        fig.tight_layout()

        if output_path is not None:
            self._save_figure(fig, output_path)

        return fig

    def plot_pseudo_label_acceptance(self, output_path: str | None = None) -> Figure:
        """Plot pseudo-label acceptance counts and mean confidence per round.

        Bar chart shows volumes accepted; overlay line shows mean confidence.

        Args:
            output_path: If provided, save the figure to this path (PNG and PDF).

        Returns:
            Matplotlib Figure object.
        """
        sns.set_theme(style="whitegrid", palette="colorblind")
        palette = sns.color_palette("colorblind", n_colors=2)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        rounds = list(range(len(self.round_metrics)))
        volumes_accepted = [m.get("volumes_accepted", 0) for m in self.round_metrics]
        mean_confidence = [m.get("mean_confidence", 0.0) for m in self.round_metrics]

        ax1.bar(rounds, volumes_accepted, color=palette[0], alpha=0.7, label="Volumes Accepted")
        ax1.set_xlabel("Self-Training Round")
        ax1.set_ylabel("Volumes Accepted", color=palette[0])
        ax1.tick_params(axis="y", labelcolor=palette[0])

        ax2 = ax1.twinx()
        ax2.plot(rounds, mean_confidence, color=palette[1], marker="o", linewidth=2, label="Mean Confidence")
        ax2.set_ylabel("Mean Confidence", color=palette[1])
        ax2.tick_params(axis="y", labelcolor=palette[1])
        ax2.set_ylim(0.0, 1.0)

        ax1.set_title("Pseudo-Label Acceptance Across Rounds")
        ax1.set_xticks(rounds)
        ax1.set_xticklabels([f"R{i}" for i in rounds])

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        fig.tight_layout()

        if output_path is not None:
            self._save_figure(fig, output_path)

        return fig

    def plot_threshold_schedule(
        self,
        thresholds_per_round: list[dict],
        output_path: str | None = None,
    ) -> Figure:
        """Plot per-class threshold evolution across self-training rounds.

        Args:
            thresholds_per_round: List of dicts, each mapping class names
                (e.g., 'TC', 'WT', 'ET') to threshold values for that round.
            output_path: If provided, save the figure to this path (PNG and PDF).

        Returns:
            Matplotlib Figure object.
        """
        sns.set_theme(style="whitegrid", palette="colorblind")
        palette = sns.color_palette("colorblind", n_colors=4)

        fig, ax = plt.subplots(figsize=(10, 6))

        if not thresholds_per_round:
            ax.set_title("Threshold Schedule (No Data)")
            fig.tight_layout()
            return fig

        # Collect all class names across all rounds
        all_classes = sorted({cls for t in thresholds_per_round for cls in t})
        rounds = list(range(len(thresholds_per_round)))

        for i, cls in enumerate(all_classes):
            values = [t.get(cls, np.nan) for t in thresholds_per_round]
            color = palette[i % len(palette)]
            ax.plot(rounds, values, marker="o", label=cls, color=color, linewidth=2)

        ax.set_title("Per-Class Threshold Schedule Across Rounds")
        ax.set_xlabel("Self-Training Round")
        ax.set_ylabel("Confidence Threshold")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{i}" for i in rounds])
        ax.legend(title="Class")
        fig.tight_layout()

        if output_path is not None:
            self._save_figure(fig, output_path)

        return fig

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
