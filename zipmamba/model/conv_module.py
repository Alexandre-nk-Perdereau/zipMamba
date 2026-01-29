"""Conformer-style depthwise separable convolution with SwooshR activation.

Uses SwooshR instead of SiLU and BiasNorm instead of LayerNorm.
Reference: "Zipformer: A faster and better encoder for ASR" (ICLR 2024)
"""

import torch
import torch.nn as nn

from .activations import SwooshR
from .norm import BiasNorm


class ConvModule(nn.Module):
    """Conformer conv: Pointwise → GLU → Depthwise → BiasNorm → SwooshR → Pointwise

    Uses BiasNorm instead of BatchNorm for more stable training with
    gradient accumulation and varying batch sizes. Uses SwooshR activation
    for better gradient flow with negative inputs.
    """

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.norm = BiasNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.bn = BiasNorm(dim)
        self.act = SwooshR()
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        x = self.norm(x).transpose(1, 2)  # (B, dim, T)
        x = self.glu(self.pw1(x))  # (B, dim, T)
        x = self.dw(x)  # (B, dim, T)
        x = x.transpose(1, 2)  # (B, T, dim) for BiasNorm
        x = self.bn(x)  # (B, T, dim)
        x = x.transpose(1, 2)  # (B, dim, T)
        x = self.pw2(self.act(x))  # (B, dim, T)
        return self.dropout(x.transpose(1, 2))  # (B, T, dim)
