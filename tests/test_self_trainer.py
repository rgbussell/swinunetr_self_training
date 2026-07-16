"""Tests for the SelfTrainer orchestration module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from swinunetr_st.models.ema_teacher import EMATeacher
from swinunetr_st.models.losses import SelfTrainingLoss
from swinunetr_st.training.curriculum import ThresholdScheduler
from swinunetr_st.training.self_trainer import SelfTrainer


class DummyModel(nn.Module):
    """Minimal model that mimics SwinUNETR input/output shapes."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture
def baseline_checkpoint(tmp_path: Path) -> str:
    """Create a baseline checkpoint file for the dummy model."""
    model = DummyModel(in_channels=4, out_channels=3)
    checkpoint_path = str(tmp_path / "baseline.pt")
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def baseline_checkpoint_direct(tmp_path: Path) -> str:
    """Create a baseline checkpoint with direct state dict (no wrapper key)."""
    model = DummyModel(in_channels=4, out_channels=3)
    checkpoint_path = str(tmp_path / "baseline_direct.pt")
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


@pytest.fixture
def trainer_config(tmp_path: Path, baseline_checkpoint: str) -> dict:
    """Full configuration dictionary for SelfTrainer tests."""
    return {
        "model": {
            "type": "swinunetr",
            "feature_size": 12,
            "img_size": [64, 64, 64],
            "in_channels": 4,
            "out_channels": 3,
            "use_checkpoint": False,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
        },
        "self_training": {
            "num_rounds": 2,
            "epochs_per_round": 2,
            "val_interval": 1,
            "ema_decay": 0.999,
            "ema_warmup_steps": 5,
            "initial_threshold": 0.95,
            "final_threshold": 0.75,
            "threshold_schedule": "cosine",
            "per_class_offsets": {"TC": 0.0, "WT": -0.05, "ET": 0.05},
            "supervised_weight": 1.0,
            "pseudo_label_weight": 0.5,
            "consistency_weight": 0.1,
            "ramp_up_epochs": 1,
        },
        "training": {
            "lr": 5e-5,
            "weight_decay": 1e-5,
            "batch_size": 1,
            "num_workers": 0,
            "amp": False,
            "early_stopping_patience": 10,
        },
        "data": {
            "roi_size": [64, 64, 64],
            "train_val_split": 0.8,
            "cache_rate": 0.0,
        },
        "paths": {
            "base_data_dir": str(tmp_path / "data"),
            "baseline_checkpoint": baseline_checkpoint,
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "pseudo_label_dir": str(tmp_path / "pseudo_labels"),
            "results_dir": str(tmp_path / "results"),
        },
    }


@pytest.fixture
def mock_create_model():
    """Patch create_model to return a DummyModel."""
    with patch("swinunetr_st.training.self_trainer.create_model") as mock_fn:
        mock_fn.side_effect = lambda config: DummyModel(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
        )
        yield mock_fn


@pytest.fixture
def trainer(trainer_config: dict, mock_create_model) -> SelfTrainer:
    """Create a SelfTrainer instance with mocked model creation."""
    return SelfTrainer(trainer_config, device=torch.device("cpu"))


class TestSelfTrainerInit:
    """Tests for SelfTrainer initialization."""

    def test_init_creates_student_model(self, trainer: SelfTrainer) -> None:
        """Student model should be created and placed on the correct device."""
        assert trainer.student is not None
        assert isinstance(trainer.student, nn.Module)

    def test_init_creates_ema_teacher(self, trainer: SelfTrainer) -> None:
        """EMA teacher should be created from the student model."""
        assert trainer.teacher is not None
        assert isinstance(trainer.teacher, EMATeacher)
        assert trainer.teacher.decay == 0.999
        assert trainer.teacher.warmup_steps == 5

    def test_init_creates_loss_fn(self, trainer: SelfTrainer) -> None:
        """SelfTrainingLoss should be created with configured weights."""
        assert trainer.loss_fn is not None
        assert isinstance(trainer.loss_fn, SelfTrainingLoss)
        assert trainer.loss_fn.supervised_weight == 1.0
        assert trainer.loss_fn.pseudo_label_weight == 0.5
        assert trainer.loss_fn.consistency_weight == 0.1

    def test_init_creates_optimizer(self, trainer: SelfTrainer) -> None:
        """AdamW optimizer should be created with configured parameters."""
        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_init_creates_threshold_scheduler(self, trainer: SelfTrainer) -> None:
        """ThresholdScheduler should be created from config."""
        assert trainer.threshold_scheduler is not None
        assert isinstance(trainer.threshold_scheduler, ThresholdScheduler)

    def test_init_creates_tensorboard_writer(self, trainer: SelfTrainer) -> None:
        """TensorBoard SummaryWriter should be created."""
        assert trainer.writer is not None
        assert isinstance(trainer.writer, SummaryWriter)

    def test_init_round_tracking(self, trainer: SelfTrainer) -> None:
        """Round tracking should start at 0 with best_dice 0.0."""
        assert trainer.current_round == 0
        assert trainer.best_dice == 0.0

    def test_init_device_auto_detect(self, trainer_config: dict, mock_create_model) -> None:
        """When device is None, should auto-detect."""
        t = SelfTrainer(trainer_config, device=None)
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert t.device == expected_device

    def test_init_amp_disabled_by_default(self, trainer: SelfTrainer) -> None:
        """AMP should be disabled when config says amp=False."""
        assert not trainer.use_amp
        assert trainer.scaler is None

    def test_init_loads_baseline_checkpoint_model_state_dict(
        self, trainer_config: dict, mock_create_model
    ) -> None:
        """Should load checkpoint with model_state_dict key."""
        trainer = SelfTrainer(trainer_config, device=torch.device("cpu"))
        # If we got here without error, the checkpoint loaded successfully
        assert trainer.student is not None

    def test_init_loads_baseline_checkpoint_direct(
        self, trainer_config: dict, baseline_checkpoint_direct: str, mock_create_model
    ) -> None:
        """Should load checkpoint with direct state dict (no wrapper key)."""
        trainer_config["paths"]["baseline_checkpoint"] = baseline_checkpoint_direct
        trainer = SelfTrainer(trainer_config, device=torch.device("cpu"))
        assert trainer.student is not None

    def test_init_tb_log_dir_created(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """TensorBoard log directory should be created."""
        tb_log_dir = Path(trainer_config["paths"]["output_dir"]) / "tb_logs"
        assert tb_log_dir.exists()


class TestSelfTrainerCheckpoint:
    """Tests for checkpoint save/load round-trip."""

    def test_save_checkpoint_creates_file(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """Saving a checkpoint should create the expected file."""
        metrics = {"dice_tc": 0.8, "dice_wt": 0.85, "dice_et": 0.7, "dice_mean": 0.783, "hd95_mean": 5.0}
        trainer._save_checkpoint(0, metrics, is_best=False)

        checkpoint_path = Path(trainer_config["paths"]["checkpoint_dir"]) / "round_0_checkpoint.pt"
        assert checkpoint_path.exists()

    def test_save_checkpoint_best_creates_best_file(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """Saving as best should create both round checkpoint and best_self_trained.pt."""
        metrics = {"dice_tc": 0.8, "dice_wt": 0.85, "dice_et": 0.7, "dice_mean": 0.783, "hd95_mean": 5.0}
        trainer._save_checkpoint(0, metrics, is_best=True)

        checkpoint_dir = Path(trainer_config["paths"]["checkpoint_dir"])
        assert (checkpoint_dir / "round_0_checkpoint.pt").exists()
        assert (checkpoint_dir / "best_self_trained.pt").exists()

    def test_checkpoint_roundtrip(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """Checkpoint should preserve model, optimizer, and metadata on round-trip."""
        metrics = {"dice_tc": 0.8, "dice_wt": 0.85, "dice_et": 0.7, "dice_mean": 0.783, "hd95_mean": 5.0}
        trainer._save_checkpoint(1, metrics, is_best=False)

        checkpoint_path = Path(trainer_config["paths"]["checkpoint_dir"]) / "round_1_checkpoint.pt"
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        assert "model_state_dict" in loaded
        assert "ema_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
        assert "round_num" in loaded
        assert "metrics" in loaded
        assert loaded["round_num"] == 1
        assert loaded["metrics"]["dice_mean"] == pytest.approx(0.783)

    def test_checkpoint_scaler_none_when_amp_disabled(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """When AMP is disabled, scaler_state_dict should be None in checkpoint."""
        metrics = {"dice_mean": 0.5}
        trainer._save_checkpoint(0, metrics)

        checkpoint_path = Path(trainer_config["paths"]["checkpoint_dir"]) / "round_0_checkpoint.pt"
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert loaded["scaler_state_dict"] is None

    def test_checkpoint_model_state_loadable(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """Model state dict from checkpoint should be loadable into a new model."""
        metrics = {"dice_mean": 0.75}
        trainer._save_checkpoint(0, metrics)

        checkpoint_path = Path(trainer_config["paths"]["checkpoint_dir"]) / "round_0_checkpoint.pt"
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        new_model = DummyModel(in_channels=4, out_channels=3)
        new_model.load_state_dict(loaded["model_state_dict"])

        # Verify parameters match
        for p1, p2 in zip(trainer.student.parameters(), new_model.parameters(), strict=True):
            assert torch.equal(p1.cpu(), p2.cpu())


class TestSelfTrainerMetrics:
    """Tests for round metrics saving."""

    def test_save_round_metrics_creates_file(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """Saving round metrics should create the expected JSON file."""
        metrics = {"dice_tc": 0.8, "dice_wt": 0.85, "dice_et": 0.7, "dice_mean": 0.783, "hd95_mean": 5.0}
        pseudo_stats = {"num_generated": 100, "mean_confidence": 0.92}
        trainer._save_round_metrics(0, metrics, pseudo_stats)

        metrics_path = Path(trainer_config["paths"]["results_dir"]) / "round_0_metrics.json"
        assert metrics_path.exists()

    def test_save_round_metrics_content(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """Saved metrics JSON should contain all expected keys and values."""
        metrics = {
            "dice_tc": 0.80,
            "dice_wt": 0.85,
            "dice_et": 0.70,
            "dice_mean": 0.783,
            "hd95_tc": 4.0,
            "hd95_wt": 3.5,
            "hd95_et": 6.0,
            "hd95_mean": 4.5,
        }
        pseudo_stats = {"num_generated": 50, "mean_confidence": 0.88}
        trainer._save_round_metrics(2, metrics, pseudo_stats)

        metrics_path = Path(trainer_config["paths"]["results_dir"]) / "round_2_metrics.json"
        with open(metrics_path) as f:
            saved = json.load(f)

        assert saved["round_num"] == 2
        assert saved["dice_mean"] == pytest.approx(0.783)
        assert saved["hd95_mean"] == pytest.approx(4.5)
        assert saved["pseudo_stats"]["num_generated"] == 50
        assert saved["pseudo_stats"]["mean_confidence"] == pytest.approx(0.88)

    def test_save_round_metrics_multiple_rounds(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """Saving metrics for multiple rounds should create separate files."""
        for i in range(3):
            metrics = {"dice_mean": 0.7 + i * 0.05}
            trainer._save_round_metrics(i, metrics, {"num_generated": 10 * (i + 1)})

        results_dir = Path(trainer_config["paths"]["results_dir"])
        for i in range(3):
            assert (results_dir / f"round_{i}_metrics.json").exists()


class TestSelfTrainerTrainingLoop:
    """Tests for the training loop orchestration."""

    @staticmethod
    def _make_fake_batch(batch_size: int = 2, is_pseudo: bool = False) -> dict:
        """Create a fake batch dictionary mimicking DataLoader output."""
        batch = {
            "image": torch.randn(batch_size, 4, 16, 16, 16),
            "label": torch.zeros(batch_size, 3, 16, 16, 16),
        }
        if is_pseudo:
            batch["is_pseudo"] = torch.ones(batch_size, dtype=torch.bool)
        else:
            batch["is_pseudo"] = torch.zeros(batch_size, dtype=torch.bool)
        return batch

    @patch("swinunetr_st.training.self_trainer.DataLoader")
    @patch("swinunetr_st.training.self_trainer.get_combined_dataset")
    def test_train_round_calls_ema_update(
        self, mock_get_dataset, mock_dataloader_cls, trainer: SelfTrainer
    ) -> None:
        """EMA teacher update should be called for each batch during training."""
        fake_batch = self._make_fake_batch(batch_size=1)

        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_get_dataset.return_value = (mock_train_ds, mock_val_ds)

        # Mock DataLoader to yield fake batches when iterated (fresh iterator per epoch)
        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(side_effect=lambda: iter([fake_batch, fake_batch]))
        mock_loader.__len__ = MagicMock(return_value=2)
        mock_dataloader_cls.return_value = mock_loader

        initial_step_count = trainer.teacher.step_count

        with patch.object(trainer, "_validate", return_value={"dice_mean": 0.5}):
            trainer._train_round(0)

        # EMA teacher should have been updated at least once per batch
        assert trainer.teacher.step_count > initial_step_count

    @patch("swinunetr_st.training.self_trainer.DataLoader")
    @patch("swinunetr_st.training.self_trainer.get_combined_dataset")
    def test_train_round_optimizer_steps(
        self, mock_get_dataset, mock_dataloader_cls, trainer: SelfTrainer
    ) -> None:
        """Optimizer should step for each batch."""
        fake_batch = self._make_fake_batch(batch_size=1)

        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_get_dataset.return_value = (mock_train_ds, mock_val_ds)

        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(side_effect=lambda: iter([fake_batch]))
        mock_loader.__len__ = MagicMock(return_value=1)
        mock_dataloader_cls.return_value = mock_loader

        with (
            patch.object(trainer, "_validate", return_value={"dice_mean": 0.5}),
            patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as mock_step,
        ):
            trainer._train_round(0)

        # optimizer.step should be called at least once per batch per epoch
        assert mock_step.call_count >= 1

    @patch("swinunetr_st.training.self_trainer.DataLoader")
    @patch("swinunetr_st.training.self_trainer.get_combined_dataset")
    def test_early_stopping(self, mock_get_dataset, mock_dataloader_cls, trainer: SelfTrainer) -> None:
        """Training should stop early when no improvement is seen for patience epochs."""
        trainer.config["training"]["early_stopping_patience"] = 1
        trainer.config["self_training"]["epochs_per_round"] = 10
        trainer.config["self_training"]["val_interval"] = 1

        fake_batch = self._make_fake_batch(batch_size=1)

        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_get_dataset.return_value = (mock_train_ds, mock_val_ds)

        # DataLoader must return fresh iterators each time (one per epoch)
        def make_loader_iter():
            return iter([fake_batch])

        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(side_effect=make_loader_iter)
        mock_loader.__len__ = MagicMock(return_value=1)
        mock_dataloader_cls.return_value = mock_loader

        # Return decreasing dice to trigger early stopping
        val_call_count = 0
        dice_values = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

        def fake_validate(_round_num):
            nonlocal val_call_count
            idx = min(val_call_count, len(dice_values) - 1)
            val_call_count += 1
            return {"dice_mean": dice_values[idx]}

        with patch.object(trainer, "_validate", side_effect=fake_validate):
            trainer._train_round(0)

        # Should have stopped before reaching all 10 epochs
        # With patience=1 and val_interval=1: epoch 0 sets best=0.8, epoch 1 gives 0.7 -> stop
        assert val_call_count < 10


class TestSelfTrainerRun:
    """Tests for the full run() loop."""

    def test_run_returns_metrics_per_round(self, trainer: SelfTrainer) -> None:
        """run() should return a list of metrics dicts, one per round."""
        fake_metrics = {
            "dice_tc": 0.8,
            "dice_wt": 0.85,
            "dice_et": 0.7,
            "dice_mean": 0.783,
            "hd95_tc": 4.0,
            "hd95_wt": 3.5,
            "hd95_et": 6.0,
            "hd95_mean": 4.5,
        }
        fake_stats = {"num_generated": 10, "mean_confidence": 0.9}

        with (
            patch.object(trainer, "_generate_pseudo_labels", return_value=fake_stats),
            patch.object(trainer, "_train_round"),
            patch.object(trainer, "_validate", return_value=fake_metrics),
        ):
            results = trainer.run()

        assert len(results) == 2  # num_rounds = 2
        for m in results:
            assert "dice_mean" in m
            assert m["dice_mean"] == pytest.approx(0.783)

    def test_run_updates_best_dice(self, trainer: SelfTrainer) -> None:
        """run() should track the best dice across rounds."""
        fake_stats = {"num_generated": 10}

        round_metrics = [
            {"dice_mean": 0.7, "dice_tc": 0.7, "dice_wt": 0.7, "dice_et": 0.7, "hd95_mean": 5.0},
            {"dice_mean": 0.85, "dice_tc": 0.85, "dice_wt": 0.85, "dice_et": 0.85, "hd95_mean": 3.0},
        ]

        validate_call_count = 0

        def fake_validate(_round_num):
            nonlocal validate_call_count
            idx = min(validate_call_count, len(round_metrics) - 1)
            validate_call_count += 1
            return round_metrics[idx]

        with (
            patch.object(trainer, "_generate_pseudo_labels", return_value=fake_stats),
            patch.object(trainer, "_train_round"),
            patch.object(trainer, "_validate", side_effect=fake_validate),
        ):
            trainer.run()

        assert trainer.best_dice == pytest.approx(0.85)

    def test_run_saves_best_checkpoint(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """run() should save best_self_trained.pt when dice improves."""
        fake_stats = {"num_generated": 10}
        fake_metrics = {"dice_mean": 0.9, "dice_tc": 0.9, "dice_wt": 0.9, "dice_et": 0.9, "hd95_mean": 2.0}

        with (
            patch.object(trainer, "_generate_pseudo_labels", return_value=fake_stats),
            patch.object(trainer, "_train_round"),
            patch.object(trainer, "_validate", return_value=fake_metrics),
        ):
            trainer.run()

        best_path = Path(trainer_config["paths"]["checkpoint_dir"]) / "best_self_trained.pt"
        assert best_path.exists()

    def test_run_resume_from_round(self, trainer: SelfTrainer) -> None:
        """run(resume_round=1) should skip round 0."""
        fake_stats = {"num_generated": 10}
        fake_metrics = {"dice_mean": 0.8, "dice_tc": 0.8, "dice_wt": 0.8, "dice_et": 0.8, "hd95_mean": 4.0}

        with (
            patch.object(trainer, "_generate_pseudo_labels", return_value=fake_stats) as mock_gen,
            patch.object(trainer, "_train_round") as mock_train,
            patch.object(trainer, "_validate", return_value=fake_metrics),
        ):
            results = trainer.run(resume_round=1)

        # Only round 1 should be executed (num_rounds=2, so rounds 0,1; resume from 1 means only round 1)
        assert len(results) == 1
        assert mock_gen.call_count == 1
        assert mock_train.call_count == 1

    def test_run_saves_per_round_metrics(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """run() should save metrics JSON for each round."""
        fake_stats = {"num_generated": 10, "mean_confidence": 0.9}
        fake_metrics = {"dice_mean": 0.8, "dice_tc": 0.8, "dice_wt": 0.8, "dice_et": 0.8, "hd95_mean": 4.0}

        with (
            patch.object(trainer, "_generate_pseudo_labels", return_value=fake_stats),
            patch.object(trainer, "_train_round"),
            patch.object(trainer, "_validate", return_value=fake_metrics),
        ):
            trainer.run()

        results_dir = Path(trainer_config["paths"]["results_dir"])
        assert (results_dir / "round_0_metrics.json").exists()
        assert (results_dir / "round_1_metrics.json").exists()

    def test_run_logs_to_tensorboard(self, trainer: SelfTrainer) -> None:
        """run() should write metrics to TensorBoard."""
        fake_stats = {"num_generated": 10}
        fake_metrics = {"dice_mean": 0.8, "hd95_mean": 4.0}

        with (
            patch.object(trainer, "_generate_pseudo_labels", return_value=fake_stats),
            patch.object(trainer, "_train_round"),
            patch.object(trainer, "_validate", return_value=fake_metrics),
            patch.object(trainer.writer, "add_scalar") as mock_add_scalar,
        ):
            trainer.run()

        # Should have written metrics for each round
        assert mock_add_scalar.call_count > 0


class TestSelfTrainerGenerate:
    """Tests for pseudo-label generation."""

    def test_generate_pseudo_labels_creates_round_dir(self, trainer: SelfTrainer, trainer_config: dict) -> None:
        """Pseudo-label generation should create the round directory."""
        thresholds = {"TC": 0.9, "WT": 0.85, "ET": 0.95}

        with (
            patch("swinunetr_st.training.self_trainer.PseudoLabeler") as mock_labeler_cls,
            patch("swinunetr_st.training.self_trainer.get_unlabeled_datalist", return_value=[]),
        ):
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {"num_generated": 5}
            mock_labeler_cls.return_value = mock_instance

            trainer._generate_pseudo_labels(0, thresholds)

        pseudo_dir = Path(trainer_config["paths"]["pseudo_label_dir"]) / "round_0"
        assert pseudo_dir.exists()

    def test_generate_pseudo_labels_uses_teacher(self, trainer: SelfTrainer) -> None:
        """Pseudo-label generation should use the teacher model."""
        thresholds = {"TC": 0.9, "WT": 0.85, "ET": 0.95}

        with (
            patch("swinunetr_st.training.self_trainer.PseudoLabeler") as mock_labeler_cls,
            patch("swinunetr_st.training.self_trainer.get_unlabeled_datalist", return_value=[]),
        ):
            mock_instance = MagicMock()
            mock_instance.generate.return_value = {"num_generated": 5}
            mock_labeler_cls.return_value = mock_instance

            trainer._generate_pseudo_labels(0, thresholds)

        # Verify PseudoLabeler was called with the teacher model
        call_kwargs = mock_labeler_cls.call_args[1]
        assert call_kwargs["model"] is trainer.teacher


class TestSelfTrainerValidation:
    """Tests for validation."""

    def test_validate_returns_metrics_dict(self, trainer: SelfTrainer) -> None:
        """_validate should return a dict with expected metric keys."""
        fake_metrics = {
            "dice_tc": 0.8,
            "dice_wt": 0.85,
            "dice_et": 0.7,
            "dice_mean": 0.783,
            "hd95_tc": 4.0,
            "hd95_wt": 3.5,
            "hd95_et": 6.0,
            "hd95_mean": 4.5,
        }

        fake_batch = {
            "image": torch.randn(1, 4, 16, 16, 16),
            "label": torch.zeros(1, 3, 16, 16, 16),
        }

        with (
            patch("swinunetr_st.training.self_trainer._get_labeled_datalist") as mock_datalist,
            patch("swinunetr_st.training.self_trainer._train_val_split") as mock_split,
            patch("swinunetr_st.training.self_trainer.CacheDataset") as mock_cache_ds,
            patch("swinunetr_st.training.self_trainer.sliding_window_inference") as mock_swi,
            patch("swinunetr_st.training.self_trainer.MetricsTracker") as mock_tracker_cls,
        ):
            mock_datalist.return_value = [{"image": "a.nii", "label": "b.nii"}]
            mock_split.return_value = ([], [{"image": "a.nii", "label": "b.nii"}])
            mock_cache_ds.return_value = [fake_batch]
            mock_swi.return_value = torch.randn(1, 3, 16, 16, 16)

            mock_tracker_instance = MagicMock()
            mock_tracker_instance.compute.return_value = fake_metrics
            mock_tracker_cls.return_value = mock_tracker_instance

            metrics = trainer._validate(0)

        assert "dice_mean" in metrics
        assert "hd95_mean" in metrics
        assert metrics["dice_mean"] == pytest.approx(0.783)
