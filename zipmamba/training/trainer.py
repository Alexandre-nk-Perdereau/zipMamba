"""Epoch-based training loop for ZipMamba ASR model."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from rich.console import Console
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    Task,
)
from rich.text import Text


class IterationsPerSecondColumn(ProgressColumn):
    """Custom column to display iterations per second."""

    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("-- it/s", style="cyan")
        return Text(f"{speed:.2f} it/s", style="cyan")


from ..model.encoder import EfficientASRModel
from ..data.tokenizer import Tokenizer
from ..utils.metrics import compute_wer, compute_cer
from .scheduler import create_scheduler
from .callbacks import TensorBoardLogger, CheckpointCallback


console = Console()


class Trainer:
    """Epoch-based training loop for ASR model."""

    def __init__(
        self,
        model: EfficientASRModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Tokenizer] = None,
        optimizer: str = "adamw",
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        gradient_clip: float = 0.5,
        mixed_precision: str = "bf16",
        log_dir: str = "./logs",
        log_every_n_steps: int = 100,
        run_name: Optional[str] = None,
        save_dir: str = "./checkpoints",
        keep_top_k_checkpoints: int = 3,
        checkpoint_metric: str = "val_wer",
        checkpoint_mode: str = "min",
        device: Optional[str] = None,
        resume_from: Optional[str] = None,
        sample_rate: int = 16000,
        n_audio_samples: int = 5,
        min_lr: float = 0.0,
        # Early stopping
        early_stopping_patience: Optional[int] = None,
        # Augmentation schedule
        aug_start_epoch: int = 5,
        aug_warmup_epochs: int = 5,
        # Callbacks for augmentation control
        set_augmentation_prob_fn: Optional[callable] = None,
        # Model config for checkpoint compatibility
        model_config: Optional[dict] = None,
    ):
        """Initialize Trainer.

        Args:
            model: ASR model to train.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader (optional).
            tokenizer: Tokenizer for decoding (optional, for WER eval).
            optimizer: Optimizer type ("adamw", "adam").
            learning_rate: Peak learning rate.
            weight_decay: Weight decay coefficient.
            warmup_epochs: Number of warmup epochs.
            max_epochs: Maximum training epochs.
            gradient_accumulation_steps: Gradient accumulation steps.
            gradient_clip: Gradient clipping norm.
            mixed_precision: Mixed precision type ("bf16", "fp16", "none").
            log_dir: TensorBoard log directory.
            log_every_n_steps: Logging frequency.
            run_name: Name for this run (for TensorBoard).
            save_dir: Checkpoint save directory.
            keep_top_k_checkpoints: Number of best checkpoints to keep.
            checkpoint_metric: Metric for selecting best checkpoints.
            checkpoint_mode: "min" if lower is better, "max" if higher is better.
            device: Device to train on (auto-detected if None).
            resume_from: Path to checkpoint to resume from.
            sample_rate: Audio sample rate for logging.
            n_audio_samples: Number of audio samples to log per validation.
            aug_start_epoch: Epoch to start applying augmentations.
            aug_warmup_epochs: Epochs to ramp up augmentation probability.
            set_augmentation_prob_fn: Callback to set augmentation probability.
            model_config: Model configuration dict for checkpoint compatibility.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip
        self.checkpoint_metric = checkpoint_metric
        self.sample_rate = sample_rate
        self.n_audio_samples = n_audio_samples
        self.log_every_n_steps = log_every_n_steps

        # Early stopping (uses same metric as checkpointing)
        self.early_stopping_patience = early_stopping_patience
        self._best_val_metric = float("inf")
        self._patience_counter = 0

        # Augmentation schedule
        self.aug_start_epoch = aug_start_epoch
        self.aug_warmup_epochs = aug_warmup_epochs
        self.set_augmentation_prob_fn = set_augmentation_prob_fn

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Optimizer
        if optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )

        # Compute steps per epoch for LR scheduler
        self.steps_per_epoch = (
            len(train_dataloader) + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps

        # Step-based scheduler (configured in epochs for convenience)
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            steps_per_epoch=self.steps_per_epoch,
            min_lr=min_lr,
        )

        # Mixed precision
        self.mixed_precision = mixed_precision
        if mixed_precision == "fp16":
            self.scaler = GradScaler("cuda")
            self.autocast_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.scaler = None
            self.autocast_dtype = torch.bfloat16
        else:
            self.scaler = None
            self.autocast_dtype = None

        # Generate run_name once if not provided
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Callbacks
        self.logger = TensorBoardLogger(
            log_dir=log_dir, log_every_n_steps=log_every_n_steps, run_name=run_name
        )
        self.checkpointer = CheckpointCallback(
            save_dir=save_dir,
            keep_top_k=keep_top_k_checkpoints,
            metric_name=checkpoint_metric,
            mode=checkpoint_mode,
            run_name=run_name,
            model_config=model_config,
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Resume from checkpoint
        if resume_from:
            self._resume_from_checkpoint(resume_from)

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        console.print(f"[bold blue]Resuming from {checkpoint_path}[/bold blue]")
        checkpoint = self.checkpointer.load(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
        )
        self.global_step = checkpoint.get("step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)

    def _get_augmentation_prob(self, epoch: int) -> float:
        """Calculate augmentation probability for current epoch.

        Returns 0.0 before aug_start_epoch, then ramps linearly to 1.0
        over aug_warmup_epochs.
        """
        if epoch < self.aug_start_epoch:
            return 0.0

        warmup_progress = (epoch - self.aug_start_epoch) / max(
            1, self.aug_warmup_epochs
        )
        return min(1.0, warmup_progress)

    def train(self) -> None:
        """Run epoch-based training loop."""
        console.print("[bold green]Starting training...[/bold green]")
        console.print(f"Device: {self.device}")
        console.print(f"Max epochs: {self.max_epochs}")
        console.print(f"Warmup epochs: {self.warmup_epochs}")
        console.print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        console.print(f"Augmentation starts at epoch {self.aug_start_epoch}")

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch + 1

            # Update augmentation probability
            aug_prob = self._get_augmentation_prob(self.current_epoch)
            if self.set_augmentation_prob_fn is not None:
                self.set_augmentation_prob_fn(aug_prob)

            console.print(
                f"\n[bold cyan]Epoch {self.current_epoch}/{self.max_epochs}[/bold cyan] "
                f"(LR: {self.optimizer.param_groups[0]['lr']:.6f}, Aug: {aug_prob:.0%})"
            )

            # Train one epoch
            train_metrics = self._train_epoch()

            # Validation
            val_metrics = None
            if self.val_dataloader is not None:
                val_metrics = self._validate()

                # Get metric for checkpointing
                if self.checkpoint_metric == "val_loss":
                    metric_value = val_metrics["loss"]
                elif self.checkpoint_metric == "val_wer":
                    metric_value = val_metrics.get("wer", val_metrics["loss"])
                else:
                    metric_value = val_metrics["loss"]

                # Save if among top-k best
                saved_path = self.checkpointer.save_if_best(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    metric_value=metric_value,
                    extra_state={"epoch": self.current_epoch},
                )

                if saved_path:
                    console.print(
                        f"[green]Saved new top-k checkpoint: {saved_path.name}[/green]"
                    )

            # Early stopping check
            if self.early_stopping_patience is not None and val_metrics is not None:
                if metric_value is not None:
                    if metric_value < self._best_val_metric:
                        self._best_val_metric = metric_value
                        self._patience_counter = 0
                    else:
                        self._patience_counter += 1
                        if self._patience_counter >= self.early_stopping_patience:
                            console.print(
                                f"[bold yellow]Early stopping triggered after "
                                f"{self.early_stopping_patience} epochs without "
                                f"{self.checkpoint_metric} improvement "
                                f"(best: {self._best_val_metric:.4f})[/bold yellow]"
                            )
                            # Save last checkpoint before stopping
                            self.checkpointer.save_last(
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                step=self.global_step,
                                extra_state={"epoch": self.current_epoch},
                            )
                            break

            # Always save last.pt for resuming
            self.checkpointer.save_last(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=self.global_step,
                extra_state={"epoch": self.current_epoch},
            )

            self.model.train()

        console.print("[bold green]Training complete![/bold green]")

        # Show top-k checkpoints
        top_k = self.checkpointer.get_top_k_info()
        if top_k:
            console.print("\n[bold]Top checkpoints:[/bold]")
            for i, info in enumerate(top_k, 1):
                console.print(
                    f"  {i}. Epoch {info.get('epoch', '?')}: "
                    f"{self.checkpoint_metric}={info['metric']:.4f}"
                )

        self.logger.close()

    def _train_epoch(self) -> dict:
        """Train for one epoch.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_blank_prob = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        num_train_batches = len(self.train_dataloader)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[yellow]loss: {task.fields[loss]:.4f}[/yellow]"),
            TextColumn("[magenta]blank: {task.fields[blank_prob]:.1%}[/magenta]"),
            IterationsPerSecondColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task(
                f"Epoch {self.current_epoch}",
                total=num_train_batches,
                loss=0.0,
                blank_prob=0.0,
            )

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move to device
                mel = batch["mel"].to(self.device)
                mel_lengths = batch["mel_lengths"].to(self.device)
                tokens = batch["tokens"].to(self.device)
                token_lengths = batch["token_lengths"].to(self.device)

                # Forward pass with mixed precision
                if self.autocast_dtype is not None:
                    with autocast("cuda", dtype=self.autocast_dtype):
                        loss, blank_prob, _ = self._compute_loss_with_blank_prob(
                            mel, mel_lengths, tokens, token_lengths
                        )
                else:
                    loss, blank_prob, _ = self._compute_loss_with_blank_prob(
                        mel, mel_lengths, tokens, token_lengths
                    )

                group_start = (
                    batch_idx // self.gradient_accumulation_steps
                ) * self.gradient_accumulation_steps
                current_accum_steps = min(
                    self.gradient_accumulation_steps,
                    num_train_batches - group_start,
                )
                raw_loss = loss.item()
                loss = loss / current_accum_steps

                accumulated_loss += loss.item()
                total_loss += raw_loss
                total_blank_prob += blank_prob
                num_batches += 1

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation step
                is_accum_boundary = (batch_idx + 1) % self.gradient_accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == num_train_batches
                if is_accum_boundary or is_last_batch:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    # Step scheduler (per optimizer step for smooth LR)
                    self.scheduler.step()

                    # Logging
                    if self.global_step % self.log_every_n_steps == 0:
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        avg_blank = total_blank_prob / max(1, num_batches)
                        self.logger.log_training_step(
                            step=self.global_step,
                            loss=accumulated_loss,
                            learning_rate=current_lr,
                            grad_norm=(
                                grad_norm.item()
                                if isinstance(grad_norm, torch.Tensor)
                                else grad_norm
                            ),
                            extra_scalars={"blank_prob": avg_blank},
                        )

                    self.global_step += 1

                # Update progress bar
                current_loss = (
                    accumulated_loss
                    if accumulated_loss > 0
                    else loss.item() * self.gradient_accumulation_steps
                )
                avg_blank = total_blank_prob / max(1, num_batches)
                progress.update(
                    task, advance=1, loss=current_loss, blank_prob=avg_blank
                )

                # Reset accumulated loss after logging
                if is_accum_boundary or is_last_batch:
                    accumulated_loss = 0.0

        avg_loss = total_loss / max(1, num_batches)
        avg_blank = total_blank_prob / max(1, num_batches)
        console.print(f"  Train loss: {avg_loss:.4f}, Blank prob: {avg_blank:.1%}")

        return {"loss": avg_loss, "blank_prob": avg_blank}

    def _compute_loss_with_blank_prob(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, float, dict]:
        """Compute CTC loss and average blank probability in a single forward pass.

        Returns:
            Tuple of (loss, blank_probability, loss_dict).
        """
        loss, loss_dict, logits = self.model.compute_loss(
            mel, mel_lengths, tokens, token_lengths
        )

        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1)
            blank_prob = log_probs[:, :, 0].exp().mean().item()

        return loss, blank_prob, loss_dict

    @torch.no_grad()
    def _validate(self) -> dict:
        """Run validation loop."""
        self.model.eval()

        total_loss = 0.0
        total_blank_prob = 0.0
        total_samples = 0
        all_hypotheses = []
        all_references = []

        # For audio logging
        logged_samples = 0
        sample_audio_paths = []
        sample_references = []
        sample_hypotheses = []

        for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
            mel = batch["mel"].to(self.device)
            mel_lengths = batch["mel_lengths"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            token_lengths = batch["token_lengths"].to(self.device)
            texts = batch["texts"]
            audio_paths = batch["audio_paths"]

            # Compute loss with blank prob
            if self.autocast_dtype is not None:
                with autocast("cuda", dtype=self.autocast_dtype):
                    loss, blank_prob, _ = self._compute_loss_with_blank_prob(
                        mel, mel_lengths, tokens, token_lengths
                    )
            else:
                loss, blank_prob, _ = self._compute_loss_with_blank_prob(
                    mel, mel_lengths, tokens, token_lengths
                )

            total_loss += loss.item() * mel.size(0)
            total_blank_prob += blank_prob * mel.size(0)
            total_samples += mel.size(0)

            # Decode for WER if tokenizer available
            if self.tokenizer is not None:
                decoded = self.model.decode_greedy(mel, mel_lengths)
                hypotheses = [self.tokenizer.decode(d) for d in decoded]
                all_hypotheses.extend(hypotheses)
                all_references.extend(texts)

                # Collect samples for audio logging
                for i in range(len(texts)):
                    if logged_samples < self.n_audio_samples:
                        sample_audio_paths.append(audio_paths[i])
                        sample_references.append(texts[i])
                        sample_hypotheses.append(hypotheses[i])
                        logged_samples += 1

        avg_loss = total_loss / max(1, total_samples)
        avg_blank = total_blank_prob / max(1, total_samples)

        metrics = {"loss": avg_loss, "blank_prob": avg_blank}

        # Compute WER/CER if we have hypotheses
        if all_hypotheses:
            metrics["wer"] = compute_wer(all_references, all_hypotheses)
            metrics["cer"] = compute_cer(all_references, all_hypotheses)

        # Log metrics
        self.logger.log_validation(
            step=self.global_step,
            loss=avg_loss,
            wer=metrics.get("wer"),
            cer=metrics.get("cer"),
            extra_scalars={"blank_prob": avg_blank},
        )

        # Log audio samples
        if sample_audio_paths:
            audios = []
            for path in sample_audio_paths:
                waveform, sr = torchaudio.load(path)
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, self.sample_rate
                    )
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                audios.append(waveform.squeeze(0))

            self.logger.log_audio_samples(
                step=self.global_step,
                audios=audios,
                sample_rate=self.sample_rate,
                references=sample_references,
                hypotheses=sample_hypotheses,
            )

        console.print(
            f"[bold]Val Epoch {self.current_epoch}:[/bold] "
            f"Loss={avg_loss:.4f}, Blank={avg_blank:.1%}"
            + (f", WER={metrics.get('wer', 0):.2%}" if "wer" in metrics else "")
            + (f", CER={metrics.get('cer', 0):.2%}" if "cer" in metrics else "")
        )

        return metrics
