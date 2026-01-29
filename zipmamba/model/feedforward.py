"""SwiGLU feed-forward module with SwooshL activation.

Uses SwooshL instead of SiLU for better gradient flow with negative inputs.
Reference: "Zipformer: A faster and better encoder for ASR" (ICLR 2024)
"""

import torch
import torch.nn as nn

from .activations import SwooshL
from .norm import BiasNorm


class FeedForward(nn.Module):
    """SwiGLU-style FFN with SwooshL: out = (SwooshL(W1·x) ⊙ W2·x) · W3

    SwooshL is used instead of SiLU because FFN modules with bypass connections
    tend to learn "normally-off" behavior (large negative bias in preceding layers).
    SwooshL handles this better than SiLU due to its non-vanishing slope for negative inputs.
    """

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expand
        self.norm = BiasNorm(dim)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.activation = SwooshL()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.dropout(self.w3(self.activation(self.w1(x)) * self.w2(x)))
