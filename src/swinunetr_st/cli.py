"""CLI entry points for self-training pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def self_train() -> None:
    """Entry point for the self-training pipeline (st-train).

    Runs iterative self-training: for each round, generates pseudo-labels
    from the EMA teacher on unlabeled data, then retrains the student on
    the combined labeled + pseudo-labeled dataset.
    """
    parser = argparse.ArgumentParser(description="Run SwinUNETR self-training pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to self_training_config.yaml")
    parser.add_argument("--resume-round", type=int, default=None, help="Resume from round N")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from swinunetr_st.training.self_trainer import SelfTrainer
    from swinunetr_st.utils.config import load_config, validate_config

    config = load_config(args.config)
    validate_config(config)
    logger.info("Starting self-training with config: %s", args.config)

    trainer = SelfTrainer(config)
    round_metrics = trainer.run(resume_round=args.resume_round)

    logger.info("Self-training complete. %d rounds finished.", len(round_metrics))
    for i, metrics in enumerate(round_metrics):
        logger.info(
            "  Round %d: Dice mean=%.4f, TC=%.4f, WT=%.4f, ET=%.4f",
            i,
            metrics.get("dice_mean", 0.0),
            metrics.get("dice_tc", 0.0),
            metrics.get("dice_wt", 0.0),
            metrics.get("dice_et", 0.0),
        )


def generate_pseudo_labels() -> None:
    """Entry point for standalone pseudo-label generation (st-pseudo-label).

    Loads a trained model checkpoint and generates confidence-filtered
    pseudo-labels on the unlabeled dataset.
    """
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--round", type=int, default=0, help="Round number (for threshold scheduling)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for pseudo-labels")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    import torch
    from swinunetr_seg.models.swinunetr import create_swinunetr

    from swinunetr_st.training.curriculum import ThresholdScheduler
    from swinunetr_st.training.pseudo_labeler import PseudoLabeler
    from swinunetr_st.utils.config import load_config, validate_config

    config = load_config(args.config)
    validate_config(config)

    # Load model
    model = create_swinunetr(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Get thresholds for the specified round
    st_cfg = config["self_training"]
    scheduler = ThresholdScheduler(
        initial=st_cfg["initial_threshold"],
        final=st_cfg["final_threshold"],
        num_rounds=st_cfg["num_rounds"],
        schedule=st_cfg["threshold_schedule"],
        per_class_offsets=st_cfg.get("per_class_offsets"),
    )
    thresholds = scheduler.get_thresholds(args.round)

    # Generate pseudo-labels
    output_dir = args.output_dir or str(Path(config["paths"]["pseudo_label_dir"]) / f"round_{args.round}")
    labeler = PseudoLabeler(model, config)
    stats = labeler.generate(
        datalist=None,  # will auto-load from config
        thresholds=thresholds,
        output_dir=output_dir,
        round_num=args.round,
    )

    logger.info("Pseudo-label generation complete: %d/%d accepted", stats["accepted"], stats["total"])


def compare_results() -> None:
    """Entry point for baseline vs self-training comparison (st-compare).

    Loads baseline and self-training metrics, generates comparison figures
    and statistical significance tests.
    """
    parser = argparse.ArgumentParser(description="Compare baseline vs self-training results")
    parser.add_argument("--baseline-csv", type=str, required=True, help="Path to baseline per-subject metrics CSV")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with round_N_metrics.json files")
    parser.add_argument("--output-dir", type=str, default="./results/figures", help="Output directory for figures")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    from swinunetr_st.analysis.comparison import ResultsComparator

    comparator = ResultsComparator()
    comparator.load_from_files(args.baseline_csv, args.results_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all comparison outputs
    comparator.plot_dice_comparison(str(output_dir / "dice_comparison"))
    comparator.plot_per_subject_scatter(str(output_dir / "per_subject_scatter"))
    table = comparator.generate_improvement_table()

    table_path = output_dir / "improvement_table.md"
    table_path.write_text(table, encoding="utf-8")
    logger.info("Saved improvement table to %s", table_path)

    # Print table to stdout
    sys.stdout.write("\n" + table + "\n\n")
    logger.info("Comparison complete. Figures saved to %s", output_dir)
