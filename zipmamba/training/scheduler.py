"""Learning rate schedulers for training."""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupScheduler(LRScheduler):
    """Cosine annealing with linear warmup, step-based with epoch configuration.

    Learning rate schedule:
    - Linear warmup from 0 to max_lr over warmup_epochs
    - Cosine decay from max_lr to min_lr over remaining epochs

    Unlike epoch-only schedulers, this one is stepped per optimizer step,
    giving smooth LR progression within each epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        steps_per_epoch: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize CosineWarmupScheduler.

        Args:
            optimizer: PyTorch optimizer.
            warmup_epochs: Number of warmup epochs.
            max_epochs: Total number of training epochs.
            steps_per_epoch: Number of optimizer steps per epoch.
            min_lr: Minimum learning rate after decay.
            last_epoch: Last step number (for resuming).
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.min_lr = min_lr

        # Convert epochs to steps
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.max_steps = max_epochs * steps_per_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch  # In PyTorch scheduler, last_epoch tracks steps here

        if step < self.warmup_steps:
            # Linear warmup - start from small LR, not 0
            warmup_factor = (step + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(1.0, progress)

            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


# Legacy alias for backward compatibility
EpochCosineWarmupScheduler = CosineWarmupScheduler


class StepCosineWarmupScheduler(LRScheduler):
    """Cosine annealing with linear warmup, step-based (legacy).

    Learning rate schedule:
    - Linear warmup from 0 to max_lr over warmup_steps
    - Cosine decay from max_lr to min_lr over remaining steps
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(1.0, progress)

            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def create_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    steps_per_epoch: int,
    min_lr: float = 0.0,
) -> CosineWarmupScheduler:
    """Create a cosine warmup scheduler with step-level granularity.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Number of warmup epochs.
        max_epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps per epoch.
        min_lr: Minimum learning rate.

    Returns:
        CosineWarmupScheduler instance.
    """
    return CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        min_lr=min_lr,
    )


# Legacy alias
def create_epoch_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    steps_per_epoch: int,
    min_lr: float = 0.0,
) -> CosineWarmupScheduler:
    """Alias for create_scheduler."""
    return create_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        min_lr=min_lr,
    )
