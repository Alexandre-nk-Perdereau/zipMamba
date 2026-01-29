"""Zipformer activation functions: SwooshR and SwooshL.

These activations have non-vanishing slopes for negative inputs,
which helps gradient flow and prevents inputs from getting trapped
in zero-gradient regions.

Reference: "Zipformer: A faster and better encoder for ASR" (ICLR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwooshR(nn.Module):
    """SwooshR activation: used for convolution modules and Conv-Embed.

    Formula: SwooshR(x) = log(1 + exp(x - 1)) - 0.08*x - 0.313261687

    The offset 0.313261687 ensures the function passes through the origin.
    Unlike Swish, SwooshR has a non-vanishing slope for negative inputs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logaddexp(torch.zeros_like(x), x - 1) - 0.08 * x - 0.313261687


class SwooshL(nn.Module):
    """SwooshL activation: used for feed-forward and bypass modules.

    Formula: SwooshL(x) = log(1 + exp(x - 4)) - 0.08*x - 0.035

    SwooshL is a right-shifted version of SwooshR. The "L" suffix indicates
    the left zero-crossing is at x ≈ 0. Used for "normally-off" modules
    where preceding layers may learn large negative biases.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logaddexp(torch.zeros_like(x), x - 4) - 0.08 * x - 0.035


# Functional versions for flexibility
def swoosh_r(x: torch.Tensor) -> torch.Tensor:
    """Functional SwooshR activation."""
    return torch.logaddexp(torch.zeros_like(x), x - 1) - 0.08 * x - 0.313261687


def swoosh_l(x: torch.Tensor) -> torch.Tensor:
    """Functional SwooshL activation."""
    return torch.logaddexp(torch.zeros_like(x), x - 4) - 0.08 * x - 0.035
