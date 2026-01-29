"""Bidirectional Mamba (SSM) block. Requires mamba-ssm package.

Uses BiasNorm instead of LayerNorm for better length information preservation.
Reference: "Zipformer: A faster and better encoder for ASR" (ICLR 2024)
"""

import torch
import torch.nn as nn

from .norm import BiasNorm

try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError("mamba-ssm required. See README.md for installation.") from e


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba: forward + backward SSM, averaged. O(n) complexity.

    Uses BiasNorm instead of LayerNorm for better preservation of length
    information during normalization.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = BiasNorm(dim)
        self.mamba_forward = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_backward = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        y_fwd = self.mamba_forward(x)
        y_bwd = torch.flip(self.mamba_backward(torch.flip(x, [1])), [1])
        return self.dropout((y_fwd + y_bwd) / 2)
