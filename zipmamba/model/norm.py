"""BiasNorm: A simpler replacement for LayerNorm that retains length information.

Reference: "Zipformer: A faster and better encoder for ASR" (ICLR 2024)
"""

import torch
import torch.nn as nn


class BiasNorm(nn.Module):
    """BiasNorm: normalizes by RMS of (x - bias), scales by exp(gamma).

    Formula: BiasNorm(x) = (x / RMS[x - b]) * exp(γ)

    Where:
    - b is a learnable channel-wise bias
    - γ is a learnable scalar (log-scale)

    Compared to LayerNorm:
    - Retains vector length information through the bias term
    - Uses exp(γ) instead of γ to ensure always-positive scaling
    - Avoids mean subtraction, preserving more information
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize BiasNorm.

        Args:
            dim: Feature dimension (number of channels).
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BiasNorm.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape.
        """
        # Subtract learnable bias
        x_shifted = x - self.bias

        # Compute RMS (root mean square) along last dimension
        rms = torch.sqrt(torch.mean(x_shifted**2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale by exp(log_scale)
        return (x / rms) * torch.exp(self.log_scale)


class BasicNorm(nn.Module):
    """BasicNorm: simplified normalization without learnable parameters.

    Formula: BasicNorm(x) = x / RMS[x]

    Useful as a lightweight alternative when full BiasNorm isn't needed.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize BasicNorm.

        Args:
            dim: Feature dimension (unused, kept for API compatibility).
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BasicNorm.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of same shape.
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms


class DropPath(nn.Module):
    """DropPath (Stochastic Depth): randomly drop entire residual branches.

    During training, entire residual paths are dropped with probability `drop_prob`.
    At inference, outputs are scaled by (1 - drop_prob) for proper expectation.

    Reference: "Deep Networks with Stochastic Depth" (ECCV 2016)
    """

    def __init__(self, drop_prob: float = 0.0):
        """Initialize DropPath.

        Args:
            drop_prob: Probability of dropping the path (0.0 = no drop).
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DropPath.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with same shape, randomly zeroed during training.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = (random_tensor < keep_prob).float()

        return x * random_tensor / keep_prob
