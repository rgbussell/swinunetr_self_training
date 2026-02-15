"""Configuration loading and validation for self-training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_REQUIRED_SECTIONS = ["model", "self_training", "training", "data", "paths"]

_REQUIRED_SELF_TRAINING_KEYS = [
    "num_rounds",
    "epochs_per_round",
    "ema_decay",
    "initial_threshold",
    "final_threshold",
    "supervised_weight",
    "pseudo_label_weight",
    "consistency_weight",
]

_REQUIRED_PATH_KEYS = [
    "base_data_dir",
    "baseline_checkpoint",
    "output_dir",
    "checkpoint_dir",
    "pseudo_label_dir",
]


def load_config(path: str | Path) -> dict:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(config).__name__}")

    logger.info("Loaded config from %s", path)
    return config


def validate_config(config: dict) -> None:
    """Validate that a config dict has all required keys and valid values.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If required keys are missing or values are invalid.
    """
    for section in _REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    st = config["self_training"]
    for key in _REQUIRED_SELF_TRAINING_KEYS:
        if key not in st:
            raise ValueError(f"Missing required self_training key: '{key}'")

    paths = config["paths"]
    for key in _REQUIRED_PATH_KEYS:
        if key not in paths:
            raise ValueError(f"Missing required paths key: '{key}'")

    # Validate ranges
    if not 0.0 < st["ema_decay"] < 1.0:
        raise ValueError(f"ema_decay must be in (0, 1), got {st['ema_decay']}")

    if not 0.0 <= st["final_threshold"] <= st["initial_threshold"] <= 1.0:
        raise ValueError(
            f"Thresholds must satisfy 0 <= final <= initial <= 1, "
            f"got final={st['final_threshold']}, initial={st['initial_threshold']}"
        )

    if st["num_rounds"] < 1:
        raise ValueError(f"num_rounds must be >= 1, got {st['num_rounds']}")

    schedule = st.get("threshold_schedule", "cosine")
    if schedule not in ("linear", "cosine", "step"):
        raise ValueError(f"threshold_schedule must be linear/cosine/step, got '{schedule}'")

    logger.info("Config validation passed")


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge override into base config.

    Args:
        base: Base configuration dictionary.
        override: Override values to merge in.

    Returns:
        Merged configuration dictionary (new dict, inputs unchanged).
    """
    merged = {}
    for key in set(base) | set(override):
        if key in override and key in base:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                merged[key] = merge_configs(base[key], override[key])
            else:
                merged[key] = override[key]
        elif key in override:
            merged[key] = override[key]
        else:
            merged[key] = base[key]
    return merged
