"""CLI entry points for self-training pipeline."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def self_train() -> None:
    """Entry point for the self-training pipeline (st-train)."""
    parser = argparse.ArgumentParser(description="Run SwinUNETR self-training pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to self_training_config.yaml")
    parser.add_argument("--resume-round", type=int, default=None, help="Resume from round N")
    args = parser.parse_args()
    logger.info("Starting self-training with config: %s", args.config)
    raise NotImplementedError("Self-training pipeline not yet implemented")


def generate_pseudo_labels() -> None:
    """Entry point for standalone pseudo-label generation (st-pseudo-label)."""
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--round", type=int, default=0, help="Round number for labeling")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for pseudo-labels")
    args = parser.parse_args()
    logger.info("Generating pseudo-labels from checkpoint: %s", args.checkpoint)
    raise NotImplementedError("Pseudo-label generation not yet implemented")


def compare_results() -> None:
    """Entry point for baseline vs self-training comparison (st-compare)."""
    parser = argparse.ArgumentParser(description="Compare baseline vs self-training results")
    parser.add_argument("--baseline-results", type=str, required=True, help="Path to baseline metrics CSV")
    parser.add_argument("--self-training-results", type=str, required=True, help="Path to self-training metrics CSV")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory for figures")
    args = parser.parse_args()
    logger.info("Comparing results: baseline=%s, self-training=%s", args.baseline_results, args.self_training_results)
    raise NotImplementedError("Results comparison not yet implemented")
