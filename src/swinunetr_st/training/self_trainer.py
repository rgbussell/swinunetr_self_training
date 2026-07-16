"""Self-training orchestration module for iterative pseudo-labeling with SwinUNETR."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from monai.data import CacheDataset
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose
from swinunetr_seg.models.factory import create_model
from swinunetr_seg.utils.metrics import MetricsTracker
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from swinunetr_st.data.pseudo_label_dataset import (
    _get_labeled_datalist,
    _train_val_split,
    get_combined_dataset,
    get_val_transforms,
)
from swinunetr_st.data.unlabeled_dataset import get_unlabeled_datalist
from swinunetr_st.models.ema_teacher import EMATeacher
from swinunetr_st.models.losses import SelfTrainingLoss, apply_ramp_weight
from swinunetr_st.training.curriculum import ThresholdScheduler, get_loss_ramp_weight
from swinunetr_st.training.pseudo_labeler import PseudoLabeler

logger = logging.getLogger(__name__)


class SelfTrainer:
    """Orchestrates iterative self-training across multiple rounds.

    Each round consists of:
        1. Generating pseudo-labels with the EMA teacher model.
        2. Training the student model on combined labeled + pseudo-labeled data.
        3. Validating the student on labeled-only data.
        4. Updating the best model checkpoint if validation dice improves.

    Args:
        config: Full configuration dictionary with model, self_training, training, data, and paths sections.
        device: Torch device to use. If None, auto-detects CUDA availability.
    """

    def __init__(self, config: dict, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("SelfTrainer using device: %s", self.device)

        # Create student model and load baseline checkpoint
        self.student = create_model(config).to(self.device)
        self._load_baseline_checkpoint(config["paths"]["baseline_checkpoint"])

        # Create EMA teacher from student
        st_cfg = config["self_training"]
        self.teacher = EMATeacher(
            self.student,
            decay=st_cfg["ema_decay"],
            warmup_steps=st_cfg.get("ema_warmup_steps", 100),
            device=self.device,
        )

        # Threshold scheduler
        self.threshold_scheduler = ThresholdScheduler(
            initial=st_cfg["initial_threshold"],
            final=st_cfg["final_threshold"],
            num_rounds=st_cfg["num_rounds"],
            schedule=st_cfg.get("threshold_schedule", "cosine"),
            per_class_offsets=st_cfg.get("per_class_offsets"),
        )

        # Loss function
        self.loss_fn = SelfTrainingLoss(
            supervised_weight=st_cfg["supervised_weight"],
            pseudo_label_weight=st_cfg["pseudo_label_weight"],
            consistency_weight=st_cfg["consistency_weight"],
        )

        # Optimizer
        train_cfg = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        # AMP scaler
        self.use_amp = train_cfg.get("amp", False)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Post-processing transforms for validation
        self.post_transforms = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # TensorBoard writer
        tb_log_dir = Path(config["paths"]["output_dir"]) / "tb_logs"
        os.makedirs(tb_log_dir, mode=0o700, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_log_dir))

        # Round tracking
        self.current_round = 0
        self.best_dice = 0.0

        logger.info("SelfTrainer initialized with %d rounds", st_cfg["num_rounds"])

    def _load_baseline_checkpoint(self, checkpoint_path: str) -> None:
        """Load baseline model weights from a checkpoint file.

        Handles both ``model_state_dict``-wrapped and direct state dict formats.

        Args:
            checkpoint_path: Path to the baseline checkpoint file.
        """
        logger.info("Loading baseline checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.student.load_state_dict(state_dict)
        logger.info("Baseline checkpoint loaded successfully")

    def run(self, resume_round: int | None = None) -> list[dict]:
        """Execute the full self-training loop across all rounds.

        Args:
            resume_round: Round number to resume from. If None, starts from round 0.

        Returns:
            List of per-round metrics dictionaries.
        """
        st_cfg = self.config["self_training"]
        num_rounds = st_cfg["num_rounds"]
        start_round = resume_round if resume_round is not None else 0
        all_metrics: list[dict] = []

        logger.info("Starting self-training: rounds %d to %d", start_round, num_rounds - 1)

        for round_num in range(start_round, num_rounds):
            self.current_round = round_num
            logger.info("=" * 80)
            logger.info("ROUND %d / %d", round_num, num_rounds - 1)
            logger.info("=" * 80)

            # Step 1: Get thresholds for this round
            thresholds = self.threshold_scheduler.get_thresholds(round_num)
            logger.info("Round %d thresholds: %s", round_num, thresholds)

            # Step 2: Generate pseudo-labels
            pseudo_stats = self._generate_pseudo_labels(round_num, thresholds)

            # Step 3: Train for this round
            self._train_round(round_num)

            # Step 4: Validate
            metrics = self._validate(round_num)

            # Step 5: Log metrics to TensorBoard
            global_step = round_num
            for key, value in metrics.items():
                self.writer.add_scalar(f"round/{key}", value, global_step)
            self.writer.flush()

            # Step 6: Save round checkpoint and metrics
            is_best = metrics["dice_mean"] > self.best_dice
            self._save_checkpoint(round_num, metrics, is_best=is_best)
            self._save_round_metrics(round_num, metrics, pseudo_stats)

            # Step 7: Update best dice
            if is_best:
                self.best_dice = metrics["dice_mean"]
                logger.info("New best dice_mean: %.4f (round %d)", self.best_dice, round_num)

            all_metrics.append(metrics)

        self.writer.close()
        logger.info("Self-training complete. Best dice_mean: %.4f", self.best_dice)
        return all_metrics

    def _generate_pseudo_labels(self, round_num: int, thresholds: dict[str, float]) -> dict:
        """Generate pseudo-labels using the EMA teacher model.

        Args:
            round_num: Current self-training round number.
            thresholds: Per-class confidence thresholds for pseudo-label filtering.

        Returns:
            Dictionary of pseudo-labeling statistics (e.g., num_generated, mean_confidence).
        """
        pseudo_dir = Path(self.config["paths"]["pseudo_label_dir"]) / f"round_{round_num}"
        os.makedirs(pseudo_dir, mode=0o700, exist_ok=True)

        labeler = PseudoLabeler(
            model=self.teacher,
            config=self.config,
            device=self.device,
        )

        data_dir = self.config["paths"]["base_data_dir"]
        datalist = get_unlabeled_datalist(data_dir)

        stats = labeler.generate(datalist, thresholds, str(pseudo_dir), round_num=round_num)
        logger.info("Round %d pseudo-label stats: %s", round_num, stats)
        return stats

    def _train_round(self, round_num: int) -> None:
        """Train the student model for one self-training round.

        Builds a combined dataset of labeled and pseudo-labeled data, then trains
        for ``epochs_per_round`` epochs with loss ramp-up and EMA teacher updates.

        Args:
            round_num: Current self-training round number.
        """
        st_cfg = self.config["self_training"]
        train_cfg = self.config["training"]
        epochs_per_round = st_cfg["epochs_per_round"]
        ramp_up_epochs = st_cfg.get("ramp_up_epochs", 5)
        val_interval = st_cfg.get("val_interval", 1)
        patience = train_cfg.get("early_stopping_patience", 10)

        pseudo_dir = str(Path(self.config["paths"]["pseudo_label_dir"]) / f"round_{round_num}")
        train_ds, _val_ds = get_combined_dataset(self.config, pseudo_dir, round_num)

        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True,
        )

        best_epoch_dice = 0.0
        epochs_without_improvement = 0

        for epoch in range(epochs_per_round):
            self.student.train()
            epoch_losses = {"total": 0.0, "supervised": 0.0, "pseudo_label": 0.0, "consistency": 0.0}
            num_batches = 0

            progress = tqdm(train_loader, desc=f"Round {round_num} Epoch {epoch}/{epochs_per_round - 1}")
            for batch_data in progress:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                is_pseudo = batch_data.get("is_pseudo", torch.zeros(images.shape[0], dtype=torch.bool))

                # Separate labeled from pseudo-labeled samples
                labeled_mask = ~is_pseudo
                pseudo_mask = is_pseudo

                # Forward passes and loss computation must share the same
                # autocast context to avoid dtype mismatches (float16 vs float32)
                # between student logits, teacher logits, and labels.
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        student_logits = self.student(images)

                        with torch.no_grad():
                            teacher_logits = self.teacher(images)

                        loss_dict = self.loss_fn(
                            student_logits,
                            labels=labels if labeled_mask.any() else None,
                            pseudo_labels=labels if pseudo_mask.any() else None,
                            teacher_logits=teacher_logits,
                        )

                        ramp_weight = get_loss_ramp_weight(epoch, ramp_up_epochs)
                        loss_dict = apply_ramp_weight(loss_dict, ramp_weight)
                else:
                    student_logits = self.student(images)

                    with torch.no_grad():
                        teacher_logits = self.teacher(images)

                    loss_dict = self.loss_fn(
                        student_logits,
                        labels=labels if labeled_mask.any() else None,
                        pseudo_labels=labels if pseudo_mask.any() else None,
                        teacher_logits=teacher_logits,
                    )

                    ramp_weight = get_loss_ramp_weight(epoch, ramp_up_epochs)
                    loss_dict = apply_ramp_weight(loss_dict, ramp_weight)

                # Backward pass
                self.optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss_dict["total"]).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict["total"].backward()
                    self.optimizer.step()

                # Update EMA teacher
                self.teacher.update(self.student)

                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += loss_dict[key].item()
                num_batches += 1

                progress.set_postfix(loss=f"{loss_dict['total'].item():.4f}", ramp=f"{ramp_weight:.2f}")

            # Average epoch losses
            if num_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches

            # Log epoch metrics to TensorBoard
            global_step = round_num * epochs_per_round + epoch
            for key, value in epoch_losses.items():
                self.writer.add_scalar(f"train/{key}", value, global_step)
            self.writer.flush()

            logger.info(
                "Round %d Epoch %d/%d — loss: %.4f (sup: %.4f, pseudo: %.4f, consist: %.4f)",
                round_num,
                epoch,
                epochs_per_round - 1,
                epoch_losses["total"],
                epoch_losses["supervised"],
                epoch_losses["pseudo_label"],
                epoch_losses["consistency"],
            )

            # Periodic validation and early stopping
            if (epoch + 1) % val_interval == 0:
                val_metrics = self._validate(round_num)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", value, global_step)
                self.writer.flush()

                if val_metrics["dice_mean"] > best_epoch_dice:
                    best_epoch_dice = val_metrics["dice_mean"]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += val_interval

                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch,
                        epochs_without_improvement,
                    )
                    break

    def _validate(self, round_num: int) -> dict:
        """Validate the student model on labeled-only data.

        Uses sliding window inference for full-volume evaluation.

        Args:
            round_num: Current self-training round number (for logging).

        Returns:
            Dictionary with dice_tc, dice_wt, dice_et, dice_mean, hd95_tc, hd95_wt, hd95_et, hd95_mean.
        """
        data_dir = self.config["paths"]["base_data_dir"]
        val_ratio = 1.0 - self.config["data"]["train_val_split"]
        labeled_list = _get_labeled_datalist(data_dir)
        _, val_list = _train_val_split(labeled_list, val_ratio=val_ratio)

        val_transforms = get_val_transforms(self.config)

        val_ds = CacheDataset(
            data=val_list,
            transform=val_transforms,
            cache_rate=self.config["data"].get("cache_rate", 0.0),
            num_workers=self.config["training"].get("num_workers", 0),
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        roi_size = self.config["data"]["roi_size"]
        tracker = MetricsTracker(num_classes=self.config["model"]["out_channels"])

        self.student.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                logits = sliding_window_inference(
                    images,
                    roi_size=roi_size,
                    sw_batch_size=4,
                    predictor=self.student,
                    overlap=0.5,
                )

                predictions = self.post_transforms(logits)
                tracker.update(predictions, labels)

        metrics = tracker.compute()
        tracker.reset()

        logger.info(
            "Round %d Validation — Dice: TC=%.4f WT=%.4f ET=%.4f Mean=%.4f | "
            "HD95: TC=%.2f WT=%.2f ET=%.2f Mean=%.2f",
            round_num,
            metrics["dice_tc"],
            metrics["dice_wt"],
            metrics["dice_et"],
            metrics["dice_mean"],
            metrics["hd95_tc"],
            metrics["hd95_wt"],
            metrics["hd95_et"],
            metrics["hd95_mean"],
        )

        return metrics

    def _save_checkpoint(self, round_num: int, metrics: dict, is_best: bool = False) -> None:
        """Save a training checkpoint for the current round.

        Args:
            round_num: Current self-training round number.
            metrics: Validation metrics dictionary for this round.
            is_best: If True, also saves the checkpoint as the best model.
        """
        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        os.makedirs(checkpoint_dir, mode=0o700, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.student.state_dict(),
            "ema_state_dict": self.teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "round_num": round_num,
            "metrics": metrics,
        }

        checkpoint_path = checkpoint_dir / f"round_{round_num}_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved checkpoint: %s", checkpoint_path)

        if is_best:
            best_path = checkpoint_dir / "best_self_trained.pt"
            torch.save(checkpoint, best_path)
            logger.info("Saved best model: %s", best_path)

    def _save_round_metrics(self, round_num: int, metrics: dict, pseudo_stats: dict) -> None:
        """Save round metrics and pseudo-label statistics to a JSON file.

        Args:
            round_num: Current self-training round number.
            metrics: Validation metrics dictionary.
            pseudo_stats: Pseudo-labeling statistics from the generation step.
        """
        results_dir = Path(self.config["paths"].get("results_dir", self.config["paths"]["output_dir"]))
        os.makedirs(results_dir, mode=0o700, exist_ok=True)

        round_metrics = {
            "round_num": round_num,
            **metrics,
            "pseudo_stats": pseudo_stats,
        }

        metrics_path = results_dir / f"round_{round_num}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(round_metrics, f, indent=2)
        logger.info("Saved round metrics: %s", metrics_path)
