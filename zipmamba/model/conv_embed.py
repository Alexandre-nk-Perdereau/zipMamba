"""Conv2D frontend: mel (100Hz) → embedding (50Hz).

Uses SwooshR activation instead of ReLU for better gradient flow.
Reference: "Zipformer: A faster and better encoder for ASR" (ICLR 2024)
"""

import torch
import torch.nn as nn

from .activations import SwooshR
from .norm import BiasNorm


class ConvEmbed(nn.Module):
    """2D conv frontend. Downsamples time by 2x, freq by 8x, then projects.

    Uses SwooshR activation instead of ReLU for better gradient flow,
    especially for negative inputs.
    """

    def __init__(self, input_dim: int = 80, embed_dim: int = 192):
        super().__init__()
        self.act = SwooshR()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=(1, 2), padding=1)  # freq /2
        self.conv2 = nn.Conv2d(
            8, 32, kernel_size=3, stride=(2, 2), padding=1
        )  # time /2, freq /2
        self.conv3 = nn.Conv2d(
            32, 128, kernel_size=3, stride=(1, 2), padding=1
        )  # freq /2

        conv_out_freq = input_dim // 8
        self.proj = nn.Linear(128 * conv_out_freq, embed_dim)
        self.norm = BiasNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.act(self.conv1(x))  # (B, 8, T, F/2)
        x = self.act(self.conv2(x))  # (B, 32, T/2, F/4)
        x = self.act(self.conv3(x))  # (B, 128, T/2, F/8)
        x = x.permute(0, 2, 1, 3).reshape(batch, x.shape[2], -1)
        return self.norm(self.proj(x))

    def get_output_length(self, input_length: int) -> int:
        return (input_length + 1) // 2
