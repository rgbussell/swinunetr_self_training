"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
import yaml

from swinunetr_st.utils.config import load_config, merge_configs, validate_config


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_valid_config(self, config_path):
        """Loading the default config file succeeds."""
        config = load_config(config_path)
        assert isinstance(config, dict)
        assert "model" in config
        assert "self_training" in config

    def test_load_missing_file(self, tmp_path):
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """Loading invalid YAML raises an error."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(": : : invalid")
        with pytest.raises(yaml.YAMLError):
            load_config(bad_file)


class TestValidateConfig:
    """Tests for validate_config."""

    def test_valid_config(self, small_config):
        """Valid config passes validation."""
        validate_config(small_config)

    def test_missing_section(self, small_config):
        """Missing top-level section raises ValueError."""
        del small_config["self_training"]
        with pytest.raises(ValueError, match="Missing required config section"):
            validate_config(small_config)

    def test_missing_self_training_key(self, small_config):
        """Missing self_training key raises ValueError."""
        del small_config["self_training"]["ema_decay"]
        with pytest.raises(ValueError, match="Missing required self_training key"):
            validate_config(small_config)

    def test_invalid_ema_decay(self, small_config):
        """EMA decay outside (0, 1) raises ValueError."""
        small_config["self_training"]["ema_decay"] = 1.5
        with pytest.raises(ValueError, match="ema_decay"):
            validate_config(small_config)

    def test_invalid_thresholds(self, small_config):
        """Final > initial threshold raises ValueError."""
        small_config["self_training"]["final_threshold"] = 0.99
        small_config["self_training"]["initial_threshold"] = 0.50
        with pytest.raises(ValueError, match="Thresholds"):
            validate_config(small_config)

    def test_invalid_schedule(self, small_config):
        """Unknown schedule type raises ValueError."""
        small_config["self_training"]["threshold_schedule"] = "exponential"
        with pytest.raises(ValueError, match="threshold_schedule"):
            validate_config(small_config)


class TestMergeConfigs:
    """Tests for merge_configs."""

    def test_shallow_merge(self):
        """Override replaces base values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        """Nested dicts are merged recursively."""
        base = {"section": {"x": 1, "y": 2}}
        override = {"section": {"y": 3, "z": 4}}
        result = merge_configs(base, override)
        assert result == {"section": {"x": 1, "y": 3, "z": 4}}

    def test_no_mutation(self):
        """Original dicts are not modified."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        merge_configs(base, override)
        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}
