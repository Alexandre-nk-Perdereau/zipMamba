"""U-Net ASR encoder with BiMamba blocks.

Uses BiasNorm and Swoosh activations for improved training stability and performance.
Reference: "Zipformer: A faster and better encoder for ASR" (ICLR 2024)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_embed import ConvEmbed
from .conmamba_block import ConMambaBlock
from .norm import BiasNorm
from .sampling import Bypass, Downsample, Upsample

# fmt: off
STACK_CONFIG_SMALL = [
    {"dim": 128, "blocks": 2, "downsample": 2},     # 50 → 25 Hz
    {"dim": 192, "blocks": 3, "downsample": 2},     # 25 → 12.5 Hz
    {"dim": 256, "blocks": 2, "downsample": 2},     # 12.5 → 6.25 Hz
    {"dim": 320, "blocks": 2, "downsample": None},  # bottleneck
    {"dim": 256, "blocks": 2, "upsample": 2},       # 6.25 → 12.5 Hz
    {"dim": 192, "blocks": 3, "upsample": 2},       # 12.5 → 25 Hz
]

STACK_CONFIG_MEDIUM = [
    {"dim": 192, "blocks": 2, "downsample": 2},
    {"dim": 256, "blocks": 4, "downsample": 2},
    {"dim": 384, "blocks": 3, "downsample": 2},
    {"dim": 512, "blocks": 2, "downsample": None},
    {"dim": 384, "blocks": 3, "upsample": 2},
    {"dim": 256, "blocks": 4, "upsample": 2},
]

STACK_CONFIG_LARGE = [
    {"dim": 256, "blocks": 2, "downsample": 2},
    {"dim": 384, "blocks": 4, "downsample": 2},
    {"dim": 512, "blocks": 4, "downsample": 2},
    {"dim": 768, "blocks": 3, "downsample": None},
    {"dim": 512, "blocks": 4, "upsample": 2},
    {"dim": 384, "blocks": 4, "upsample": 2},
]
# fmt: on


class EfficientASREncoder(nn.Module):
    """U-Net encoder: mel (100Hz) → ConvEmbed (50Hz) → encoder → bottleneck → decoder (25Hz)."""

    def __init__(
        self,
        input_dim: int = 80,
        stack_config: Optional[list[dict]] = None,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        ff_expand: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        conv_channels: Optional[tuple[int, int, int]] = None,
    ):
        super().__init__()

        if stack_config is None:
            stack_config = STACK_CONFIG_MEDIUM

        self.stack_config = stack_config
        self.input_dim = input_dim

        first_dim = stack_config[0]["dim"]
        self.conv_embed = ConvEmbed(input_dim, first_dim, conv_channels=conv_channels)

        self.stacks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.bypasses = nn.ModuleList()
        self.dim_projs = nn.ModuleList()

        prev_dim = first_dim
        n_stacks = len(stack_config)
        mid_point = n_stacks // 2

        total_blocks = sum(cfg["blocks"] for cfg in stack_config)
        block_idx = 0

        for cfg in stack_config:
            dim = cfg["dim"]

            if prev_dim != dim:
                self.dim_projs.append(nn.Linear(prev_dim, dim))
            else:
                self.dim_projs.append(nn.Identity())

            blocks = nn.ModuleList()
            for _ in range(cfg["blocks"]):
                block_drop_path = drop_path * block_idx / max(total_blocks - 1, 1)
                blocks.append(
                    ConMambaBlock(
                        dim=dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        mamba_expand=mamba_expand,
                        ff_expand=ff_expand,
                        conv_kernel_size=conv_kernel_size,
                        dropout=dropout,
                        drop_path=block_drop_path,
                    )
                )
                block_idx += 1
            self.stacks.append(blocks)

            if cfg.get("downsample"):
                self.downsamples.append(Downsample(cfg["downsample"]))
            else:
                self.downsamples.append(None)

            if cfg.get("upsample"):
                self.upsamples.append(Upsample(cfg["upsample"], dim=dim))
            else:
                self.upsamples.append(None)

            self.bypasses.append(Bypass(dim))
            prev_dim = dim

        # Skip projections (when encoder/decoder dims don't match)
        self.skip_projs = nn.ModuleDict()
        encoder_skip_dims = [stack_config[i]["dim"] for i in range(mid_point)]
        skip_idx = len(encoder_skip_dims) - 1

        for i in range(mid_point, n_stacks):
            if stack_config[i].get("upsample") and skip_idx >= 0:
                skip_dim = encoder_skip_dims[skip_idx]
                current_dim = stack_config[i]["dim"]
                if skip_dim != current_dim:
                    self.skip_projs[f"skip_{skip_dim}_to_{current_dim}"] = nn.Linear(
                        skip_dim, current_dim
                    )
                skip_idx -= 1

        # Handle unused encoder skip from first stack (highest resolution)
        n_decoder_upsamples = sum(
            1 for cfg in stack_config[mid_point:] if cfg.get("upsample")
        )
        self._n_unused_skips = mid_point - n_decoder_upsamples

        if self._n_unused_skips > 0:
            skip_dim = stack_config[0]["dim"]
            skip_ds = stack_config[0].get("downsample", 1)
            self.remaining_skip_ds = Downsample(skip_ds) if skip_ds > 1 else None
            self.remaining_skip_proj = (
                nn.Linear(skip_dim, stack_config[-1]["dim"])
                if skip_dim != stack_config[-1]["dim"]
                else None
            )
        else:
            self.remaining_skip_ds = None
            self.remaining_skip_proj = None

        self._output_dim = stack_config[-1]["dim"]
        self._n_stacks = n_stacks
        self._mid_point = mid_point

        # Final BiasNorm for stable CTC projection
        self.final_ln = BiasNorm(self._output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """Forward pass through U-Net encoder.

        Args:
            x: Input mel spectrogram (batch, time, freq).
            x_lengths: Input lengths (batch,).
            return_intermediates: If True, return intermediate outputs for aux CTC.

        Returns:
            Tuple of (output, output_lengths, intermediates).
            intermediates is a dict mapping stack indices to (output, lengths).
        """
        x = self.conv_embed(x)
        if x_lengths is not None:
            x_lengths = self.conv_embed.get_output_length(x_lengths)

        n_stacks = len(self.stacks)
        mid_point = n_stacks // 2
        skip_connections = []
        intermediates = {} if return_intermediates else None

        # Encoder path
        for i in range(mid_point):
            x = self.dim_projs[i](x)
            stack_input = x

            for block in self.stacks[i]:
                x = block(x)
            x = self.bypasses[i](stack_input, x)
            skip_connections.append(x)

            if return_intermediates and i == 1:
                intermediates[f"stack_{i}"] = (
                    x.clone(),
                    x_lengths.clone() if x_lengths is not None else None,
                )

            if self.downsamples[i] is not None:
                x = self.downsamples[i](x)
                if x_lengths is not None:
                    x_lengths = self.downsamples[i].get_output_length(x_lengths)

        # Decoder path
        for i in range(mid_point, n_stacks):
            x = self.dim_projs[i](x)

            if self.upsamples[i] is not None:
                x = self.upsamples[i](x)
                if x_lengths is not None:
                    x_lengths = self.upsamples[i].get_output_length(x_lengths)

                if skip_connections:
                    skip = skip_connections.pop()
                    if skip.shape[-1] != x.shape[-1]:
                        key = f"skip_{skip.shape[-1]}_to_{x.shape[-1]}"
                        skip = self.skip_projs[key](skip)
                    min_len = min(x.shape[1], skip.shape[1])
                    x = x[:, :min_len, :] + skip[:, :min_len, :]

            stack_input = x
            for block in self.stacks[i]:
                x = block(x)
            x = self.bypasses[i](stack_input, x)

            if return_intermediates and i == mid_point:
                intermediates[f"stack_{i}"] = (
                    x.clone(),
                    x_lengths.clone() if x_lengths is not None else None,
                )

        # Use remaining encoder skip (from first stack, highest resolution)
        if skip_connections and self._n_unused_skips > 0:
            skip = skip_connections[0]
            if self.remaining_skip_ds is not None:
                skip = self.remaining_skip_ds(skip)
            if self.remaining_skip_proj is not None:
                skip = self.remaining_skip_proj(skip)
            min_len = min(x.shape[1], skip.shape[1])
            x = x[:, :min_len, :] + skip[:, :min_len, :]

        # Final layer norm for stability
        x = self.final_ln(x)

        return x, x_lengths, intermediates


class EfficientASRModel(nn.Module):
    """ASR model: encoder + CTC head with optional intermediate CTC losses."""

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        stack_config: Optional[list[dict]] = None,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        ff_expand: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        use_intermediate_ctc: bool = False,
        intermediate_ctc_weight: float = 0.3,
        conv_channels: Optional[tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_intermediate_ctc = use_intermediate_ctc
        self.intermediate_ctc_weight = intermediate_ctc_weight

        if stack_config is None:
            stack_config = STACK_CONFIG_MEDIUM

        self.encoder = EfficientASREncoder(
            input_dim=input_dim,
            stack_config=stack_config,
            d_state=d_state,
            d_conv=d_conv,
            mamba_expand=mamba_expand,
            ff_expand=ff_expand,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            drop_path=drop_path,
            conv_channels=conv_channels,
        )
        self.ctc_proj = nn.Linear(self.encoder.output_dim, vocab_size)

        # Intermediate CTC heads for multi-scale supervision
        if use_intermediate_ctc:
            self.intermediate_ctc_heads = nn.ModuleDict()
            # Stack 1 (25Hz before downsample) - dim 256
            self.intermediate_ctc_heads["stack_1"] = nn.Linear(
                stack_config[1]["dim"], vocab_size
            )
            # Stack 3 (12.5Hz after first upsample) - dim 384
            mid_point = len(stack_config) // 2
            self.intermediate_ctc_heads[f"stack_{mid_point}"] = nn.Linear(
                stack_config[mid_point]["dim"], vocab_size
            )

            for head in self.intermediate_ctc_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                head.bias.data[0] = 6.0

        nn.init.zeros_(self.ctc_proj.weight)
        nn.init.zeros_(self.ctc_proj.bias)
        self.ctc_proj.bias.data[0] = 6.0

    def forward(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        encoded, out_lengths, _ = self.encoder(x, x_lengths, return_intermediates=False)
        return self.ctc_proj(encoded), out_lengths

    def forward_with_intermediates(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Forward pass returning intermediate outputs for aux CTC losses.

        Returns:
            Tuple of (logits, out_lengths, intermediate_logits).
            intermediate_logits maps stack names to (logits, lengths).
        """
        encoded, out_lengths, intermediates = self.encoder(
            x, x_lengths, return_intermediates=True
        )
        logits = self.ctc_proj(encoded)

        intermediate_logits = {}
        if intermediates and self.use_intermediate_ctc:
            for name, (hidden, lengths) in intermediates.items():
                if name in self.intermediate_ctc_heads:
                    inter_logits = self.intermediate_ctc_heads[name](hidden)
                    intermediate_logits[name] = (inter_logits, lengths)

        return logits, out_lengths, intermediate_logits

    def compute_loss(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Compute CTC loss with optional intermediate losses.

        Returns:
            Tuple of (total_loss, loss_dict with individual losses, logits).
        """
        if self.use_intermediate_ctc and self.training:
            logits, out_lengths, intermediate_logits = self.forward_with_intermediates(
                x, x_lengths
            )
        else:
            logits, out_lengths = self.forward(x, x_lengths)
            intermediate_logits = {}

        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        # Filter invalid samples (output too short for target)
        max_time = log_probs.shape[0]
        valid_mask = (out_lengths >= target_lengths) & (out_lengths <= max_time)
        if not valid_mask.all():
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                zero_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                return zero_loss, {"main": zero_loss}, logits
            log_probs = log_probs[:, valid_indices, :]
            out_lengths = out_lengths[valid_indices]
            targets_filtered = targets[valid_indices]
            target_lengths_filtered = target_lengths[valid_indices]
        else:
            targets_filtered = targets
            target_lengths_filtered = target_lengths

        main_loss = F.ctc_loss(
            log_probs,
            targets_filtered,
            out_lengths,
            target_lengths_filtered,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )

        loss_dict = {"main": main_loss}
        total_loss = main_loss

        # Compute intermediate CTC losses
        if intermediate_logits:
            for name, (inter_logits, inter_lengths) in intermediate_logits.items():
                inter_log_probs = F.log_softmax(inter_logits, dim=-1).transpose(0, 1)

                # Filter for this resolution
                if inter_lengths is not None:
                    max_inter_time = inter_log_probs.shape[0]
                    inter_valid = (inter_lengths >= target_lengths) & (
                        inter_lengths <= max_inter_time
                    )
                    if not inter_valid.all():
                        inter_valid_idx = inter_valid.nonzero(as_tuple=True)[0]
                        if len(inter_valid_idx) == 0:
                            continue
                        inter_log_probs = inter_log_probs[:, inter_valid_idx, :]
                        inter_lengths = inter_lengths[inter_valid_idx]
                        inter_targets = targets[inter_valid_idx]
                        inter_target_lengths = target_lengths[inter_valid_idx]
                    else:
                        inter_targets = targets
                        inter_target_lengths = target_lengths
                else:
                    inter_targets = targets
                    inter_target_lengths = target_lengths

                inter_loss = F.ctc_loss(
                    inter_log_probs,
                    inter_targets,
                    inter_lengths,
                    inter_target_lengths,
                    blank=0,
                    reduction="mean",
                    zero_infinity=True,
                )

                loss_dict[name] = inter_loss
                total_loss = total_loss + self.intermediate_ctc_weight * inter_loss

        return total_loss, loss_dict, logits

    @torch.no_grad()
    def decode_greedy(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> list[list[int]]:
        """Greedy CTC decoding: collapse repeats and remove blanks."""
        logits, out_lengths = self.forward(x, x_lengths)
        predictions = logits.argmax(dim=-1)

        decoded = []
        for i, pred in enumerate(predictions):
            length = out_lengths[i] if out_lengths is not None else len(pred)
            tokens = pred[:length].tolist()

            result = []
            prev = None
            for t in tokens:
                if t != 0 and t != prev:
                    result.append(t)
                prev = t
            decoded.append(result)

        return decoded
