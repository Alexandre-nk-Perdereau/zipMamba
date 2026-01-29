"""ConMamba block: Macaron-style FFN + BiMamba + Conv.

Uses BiasNorm instead of LayerNorm and Swoosh activations for better training.
Reference: "Zipformer: A faster and better encoder for ASR" (ICLR 2024)
"""

import torch
import torch.nn as nn

from .conv_module import ConvModule
from .feedforward import FeedForward
from .mamba import BiMambaBlock
from .norm import BiasNorm, DropPath


class ConMambaBlock(nn.Module):
    """Conformer-style block with Mamba instead of attention.

    Structure: FFN(0.5x) → BiMamba → Conv → FFN(0.5x) → BiasNorm

    Uses BiasNorm at the end of each block for better length information
    preservation and training stability.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        ff_expand: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_expand, dropout)
        self.bimamba = BiMambaBlock(dim, d_state, d_conv, mamba_expand, dropout)
        self.conv = ConvModule(dim, conv_kernel_size, dropout)
        self.ff2 = FeedForward(dim, ff_expand, dropout)
        self.final_norm = BiasNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(0.5 * self.ff1(x))
        x = x + self.drop_path(self.bimamba(x))
        x = x + self.drop_path(self.conv(x))
        x = x + self.drop_path(0.5 * self.ff2(x))
        return self.final_norm(x)
