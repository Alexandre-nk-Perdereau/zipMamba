"""Training callbacks for logging and checkpointing."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard logging callback."""

    def __init__(
        self,
        log_dir: str = "./logs",
        log_every_n_steps: int = 100,
        run_name: Optional[str] = None,
    ):
        """Initialize TensorBoardLogger.

        Args:
            log_dir: Base directory for TensorBoard logs.
            log_every_n_steps: Log frequency.
            run_name: Name for this run. If None, uses timestamp.
        """
        base_dir = Path(log_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create unique subdirectory for this run
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_every_n_steps = log_every_n_steps
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        self._step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar value.

        Args:
            tag: Metric name.
            value: Metric value.
            step: Global step (uses internal counter if None).
        """
        step = step if step is not None else self._step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple scalars under a main tag.

        Args:
            main_tag: Main tag for grouping.
            tag_scalar_dict: Dict of tag -> value.
            step: Global step.
        """
        step = step if step is not None else self._step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self, tag: str, values: torch.Tensor, step: Optional[int] = None
    ) -> None:
        """Log a histogram.

        Args:
            tag: Histogram name.
            values: Tensor of values.
            step: Global step.
        """
        step = step if step is not None else self._step
        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log text.

        Args:
            tag: Text tag.
            text: Text content.
            step: Global step.
        """
        step = step if step is not None else self._step
        self.writer.add_text(tag, text, step)

    def log_training_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        extra_scalars: Optional[dict[str, float]] = None,
    ) -> None:
        """Log training step metrics.

        Args:
            step: Current step.
            loss: Training loss.
            learning_rate: Current learning rate.
            grad_norm: Gradient norm (optional).
            extra_scalars: Additional scalars to log (optional).
        """
        self._step = step

        if step % self.log_every_n_steps == 0:
            self.log_scalar("train/loss", loss, step)
            self.log_scalar("train/learning_rate", learning_rate, step)
            if grad_norm is not None:
                self.log_scalar("train/grad_norm", grad_norm, step)
            if extra_scalars:
                for name, value in extra_scalars.items():
                    self.log_scalar(f"train/{name}", value, step)

    def log_validation(
        self,
        step: int,
        loss: float,
        wer: Optional[float] = None,
        cer: Optional[float] = None,
        extra_scalars: Optional[dict[str, float]] = None,
    ) -> None:
        """Log validation metrics.

        Args:
            step: Current step.
            loss: Validation loss.
            wer: Word Error Rate (optional).
            cer: Character Error Rate (optional).
            extra_scalars: Additional scalars to log (optional).
        """
        self.log_scalar("val/loss", loss, step)
        if wer is not None:
            self.log_scalar("val/wer", wer, step)
        if cer is not None:
            self.log_scalar("val/cer", cer, step)
        if extra_scalars:
            for name, value in extra_scalars.items():
                self.log_scalar(f"val/{name}", value, step)

    def log_audio_samples(
        self,
        step: int,
        audios: list[torch.Tensor],
        sample_rate: int,
        references: list[str],
        hypotheses: list[str],
    ) -> None:
        """Log audio samples with transcriptions.

        Args:
            step: Current step.
            audios: List of audio tensors (each 1D).
            sample_rate: Audio sample rate.
            references: Ground truth transcriptions.
            hypotheses: Model predictions.
        """
        for i, (audio, ref, hyp) in enumerate(zip(audios, references, hypotheses)):
            # Log audio
            # TensorBoard expects (1, L) or (2, L) for stereo
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            self.writer.add_audio(
                f"val_samples/{i}/audio",
                audio,
                step,
                sample_rate=sample_rate,
            )

            # Log transcriptions as text
            text_content = f"**Reference:** {ref}\n\n**Hypothesis:** {hyp}"
            self.writer.add_text(f"val_samples/{i}/transcription", text_content, step)

    def flush(self) -> None:
        """Flush pending logs."""
        self.writer.flush()

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()


class CheckpointCallback:
    """Checkpoint saving and loading callback.

    Keeps top-k best checkpoints based on validation metric,
    plus always saves the latest checkpoint for resuming.
    """

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        keep_top_k: int = 3,
        metric_name: str = "val_loss",
        mode: str = "min",
        run_name: Optional[str] = None,
    ):
        """Initialize CheckpointCallback.

        Args:
            save_dir: Base directory for checkpoints.
            keep_top_k: Number of best checkpoints to keep.
            metric_name: Name of metric to track (for logging).
            mode: "min" if lower is better, "max" if higher is better.
            run_name: Name for this run. If None, uses timestamp.
        """
        base_dir = Path(save_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create unique subdirectory for this run
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = base_dir / run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_top_k = keep_top_k
        self.metric_name = metric_name
        self.mode = mode

        # Track top-k checkpoints: list of (metric_value, step, path)
        self._top_checkpoints: list[tuple[float, int, Path]] = []

        # Load existing state if resuming
        self._load_checkpoint_state()

    def _load_checkpoint_state(self) -> None:
        """Load checkpoint tracking state from disk."""
        state_path = self.save_dir / "checkpoint_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
                self._top_checkpoints = [
                    (c["metric"], c["step"], Path(c["path"]))
                    for c in state.get("top_checkpoints", [])
                    if Path(c["path"]).exists()
                ]

    def _save_checkpoint_state(self) -> None:
        """Save checkpoint tracking state to disk."""
        state_path = self.save_dir / "checkpoint_state.json"
        state = {
            "metric_name": self.metric_name,
            "mode": self.mode,
            "top_checkpoints": [
                {"metric": m, "step": s, "path": str(p)}
                for m, s, p in self._top_checkpoints
            ],
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def save_last(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        extra_state: Optional[dict] = None,
    ) -> Path:
        """Save latest checkpoint (for resuming training).

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            scheduler: Scheduler state.
            step: Current training step.
            extra_state: Additional state to save.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.save_dir / "last.pt"

        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if extra_state is not None:
            state.update(extra_state)

        torch.save(state, checkpoint_path)
        return checkpoint_path

    def save_if_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metric_value: float,
        extra_state: Optional[dict] = None,
    ) -> Optional[Path]:
        """Save checkpoint if it's among the top-k best.

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            scheduler: Scheduler state.
            step: Current training step.
            metric_value: Value of the metric to compare.
            extra_state: Additional state to save.

        Returns:
            Path to saved checkpoint if saved, None otherwise.
        """
        # Check if this should be in top-k
        should_save = False

        if len(self._top_checkpoints) < self.keep_top_k:
            should_save = True
        else:
            # Compare with worst in top-k
            if self.mode == "min":
                worst_metric = max(c[0] for c in self._top_checkpoints)
                should_save = metric_value < worst_metric
            else:
                worst_metric = min(c[0] for c in self._top_checkpoints)
                should_save = metric_value > worst_metric

        if not should_save:
            return None

        # Save new checkpoint
        checkpoint_path = (
            self.save_dir / f"best_step_{step}_{self.metric_name}_{metric_value:.4f}.pt"
        )

        state = {
            "step": step,
            "metric_name": self.metric_name,
            "metric_value": metric_value,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if extra_state is not None:
            state.update(extra_state)

        torch.save(state, checkpoint_path)

        # Add to top-k
        self._top_checkpoints.append((metric_value, step, checkpoint_path))

        # Sort and remove worst if over limit
        if self.mode == "min":
            self._top_checkpoints.sort(key=lambda x: x[0])  # Best (lowest) first
        else:
            self._top_checkpoints.sort(key=lambda x: -x[0])  # Best (highest) first

        # Remove worst checkpoints
        while len(self._top_checkpoints) > self.keep_top_k:
            _, _, old_path = self._top_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

        # Save state
        self._save_checkpoint_state()

        return checkpoint_path

    def load(
        self,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
    ) -> dict:
        """Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
            model: Model to load state into.
            optimizer: Optimizer to load state into (optional).
            scheduler: Scheduler to load state into (optional).

        Returns:
            Checkpoint state dict.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        if not self._top_checkpoints:
            return None
        return self._top_checkpoints[0][2]

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint (for resuming)."""
        last_path = self.save_dir / "last.pt"
        return last_path if last_path.exists() else None

    def get_top_k_info(self) -> list[dict]:
        """Get info about top-k checkpoints."""
        return [
            {"metric": m, "step": s, "path": str(p)}
            for m, s, p in self._top_checkpoints
        ]
