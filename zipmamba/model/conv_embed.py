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

    Args:
        input_dim: Number of mel frequency bins (default: 80)
        embed_dim: Output embedding dimension (default: 192)
        conv_channels: Tuple of (c1, c2, c3) channel sizes for the 3 conv layers.
                      If None, scales automatically based on embed_dim relative
                      to baseline of 128 (e.g., embed_dim=192 → scale=1.5x).
    """

    def __init__(
        self,
        input_dim: int = 80,
        embed_dim: int = 192,
        conv_channels: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.act = SwooshR()

        if conv_channels is not None:
            c1, c2, c3 = conv_channels
        else:
            scale = embed_dim / 128
            c1 = max(8, int(8 * scale))
            c2 = max(32, int(32 * scale))
            c3 = max(128, int(128 * scale))

        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, stride=(1, 2), padding=1)  # freq /2
        self.conv2 = nn.Conv2d(
            c1, c2, kernel_size=3, stride=(2, 2), padding=1
        )  # time /2, freq /2
        self.conv3 = nn.Conv2d(
            c2, c3, kernel_size=3, stride=(1, 2), padding=1
        )  # freq /2

        conv_out_freq = input_dim // 8
        self.proj = nn.Linear(c3 * conv_out_freq, embed_dim)
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
