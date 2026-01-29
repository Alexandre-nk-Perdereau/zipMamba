"""Temporal resampling modules for U-Net architecture."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    """Learnable weighted average over consecutive frames. (batch, time, dim) → (batch, time//factor, dim)"""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.weights = nn.Parameter(torch.ones(factor) / factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, dim = x.shape

        pad_len = (self.factor - time % self.factor) % self.factor
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        new_time = (time + pad_len) // self.factor
        x = x.reshape(batch, new_time, self.factor, dim)
        weights = F.softmax(self.weights, dim=0)
        return torch.einsum("btfd,f->btd", x, weights)

    def get_output_length(self, input_length: int) -> int:
        return (input_length + self.factor - 1) // self.factor


class Upsample(nn.Module):
    """Frame repetition. (batch, time, dim) → (batch, time*factor, dim)"""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, dim = x.shape
        x = x.unsqueeze(2).repeat(1, 1, self.factor, 1)
        return x.reshape(batch, time * self.factor, dim)

    def get_output_length(self, input_length: int) -> int:
        return input_length * self.factor


class Bypass(nn.Module):
    """Learnable interpolation: α * output + (1-α) * input, where α = sigmoid(scale)."""

    def __init__(self, dim_in: int, dim_out: Optional[int] = None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out if dim_out is not None else dim_in
        self.scale = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(dim_in, self.dim_out) if dim_in != self.dim_out else None

    def forward(
        self, stack_input: torch.Tensor, stack_output: torch.Tensor
    ) -> torch.Tensor:
        if self.proj is not None:
            stack_input = self.proj(stack_input)

        min_len = min(stack_input.shape[1], stack_output.shape[1])
        stack_input = stack_input[:, :min_len, :]
        stack_output = stack_output[:, :min_len, :]

        alpha = torch.sigmoid(self.scale)
        return alpha * stack_output + (1 - alpha) * stack_input
