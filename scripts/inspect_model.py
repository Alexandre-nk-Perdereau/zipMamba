import sys
sys.path.insert(0, ".")

import torch
from collections import defaultdict

from zipmamba.model.encoder import EfficientASRModel


def count_params_by_component(model):
    """Count parameters grouped by component type."""
    groups = defaultdict(int)
    details = {}

    for name, param in model.named_parameters():
        n = param.numel()
        details[name] = n

        if "conv_embed" in name:
            groups["conv_embed"] += n
        elif "ctc_proj" in name:
            groups["ctc_proj"] += n
        elif "intermediate_ctc" in name:
            groups["intermediate_ctc"] += n
        elif "final_ln" in name:
            groups["final_ln"] += n
        elif "dim_projs" in name:
            groups["dim_projs"] += n
        elif "skip_proj" in name or "remaining_skip" in name:
            groups["skip_projs"] += n
        elif "bypasses" in name:
            groups["bypasses"] += n
        elif "downsamples" in name:
            groups["downsamples"] += n
        elif "upsamples" in name:
            groups["upsamples"] += n
        elif "stacks" in name:
            parts = name.split(".")
            stack_idx = int(parts[2])

            if "ff1" in name or "ff2" in name:
                groups[f"stack_{stack_idx}_ffn"] += n
            elif "bimamba" in name:
                groups[f"stack_{stack_idx}_mamba"] += n
            elif "conv" in name:
                groups[f"stack_{stack_idx}_conv"] += n
            elif "final_norm" in name:
                groups[f"stack_{stack_idx}_norm"] += n
            else:
                groups[f"stack_{stack_idx}_other"] += n
        else:
            groups["other"] += n

    return groups, details


def analyze_config(name, stack_config, **kwargs):
    """Analyze a model config."""
    model = EfficientASRModel(
        vocab_size=5000,
        input_dim=80,
        stack_config=stack_config,
        d_state=kwargs.get("d_state", 16),
        d_conv=kwargs.get("d_conv", 4),
        mamba_expand=kwargs.get("mamba_expand", 2),
        ff_expand=kwargs.get("ff_expand", 4),
        conv_kernel_size=kwargs.get("conv_kernel_size", 31),
        dropout=kwargs.get("dropout", 0.1),
        drop_path=kwargs.get("drop_path", 0.1),
        use_intermediate_ctc=True,
        intermediate_ctc_weight=0.3,
    )

    total = sum(p.numel() for p in model.parameters())
    groups, _ = count_params_by_component(model)

    print(f"\n{'='*70}")
    print(f"  {name}: {total:,} params ({total/1e6:.1f}M)")
    print(f"{'='*70}")

    dims = [s['dim'] for s in stack_config]
    blocks = [s['blocks'] for s in stack_config]
    print(f"  Dims:   {dims}")
    print(f"  Blocks: {blocks}")
    print()

    for i, cfg in enumerate(stack_config):
        ffn = groups.get(f"stack_{i}_ffn", 0)
        mamba = groups.get(f"stack_{i}_mamba", 0)
        conv = groups.get(f"stack_{i}_conv", 0)
        norm = groups.get(f"stack_{i}_norm", 0)
        stack_total = ffn + mamba + conv + norm
        pct = 100 * stack_total / total
        print(f"  Stack {i} (dim={cfg['dim']:3d}, blocks={cfg['blocks']}): "
              f"{stack_total:>10,} ({pct:5.1f}%) "
              f"[FFN:{ffn:,} Mamba:{mamba:,} Conv:{conv:,}]")

    overhead_keys = ["conv_embed", "ctc_proj", "intermediate_ctc", "final_ln",
                     "dim_projs", "skip_projs", "bypasses", "downsamples", "upsamples"]
    print()
    for key in overhead_keys:
        v = groups.get(key, 0)
        if v > 0:
            print(f"  {key:20s}: {v:>10,} ({100*v/total:5.1f}%)")

    return total


def main():
    print("=" * 70)
    print("  ZipMamba Model Parameter Analysis")
    print("=" * 70)

    # Small (known good)
    small = [
        {"dim": 128, "blocks": 2, "downsample": 2},
        {"dim": 192, "blocks": 3, "downsample": 2},
        {"dim": 256, "blocks": 2, "downsample": 2},
        {"dim": 320, "blocks": 2, "downsample": None},
        {"dim": 256, "blocks": 2, "upsample": 2},
        {"dim": 192, "blocks": 3, "upsample": 2},
    ]
    small_total = analyze_config("SMALL (reference)", small)

    # Existing medium.yaml
    medium_old = [
        {"dim": 192, "blocks": 2, "downsample": 2},
        {"dim": 256, "blocks": 4, "downsample": 2},
        {"dim": 384, "blocks": 3, "downsample": 2},
        {"dim": 512, "blocks": 2, "downsample": None},
        {"dim": 384, "blocks": 3, "upsample": 2},
        {"dim": 256, "blocks": 4, "upsample": 2},
    ]
    medium_old_total = analyze_config("MEDIUM (existing yaml)", medium_old)

    # Candidate A: conservative scale ~70M - keep same block counts, just scale dims ~1.4x
    candidate_a = [
        {"dim": 192, "blocks": 2, "downsample": 2},
        {"dim": 256, "blocks": 3, "downsample": 2},
        {"dim": 384, "blocks": 2, "downsample": 2},
        {"dim": 448, "blocks": 2, "downsample": None},
        {"dim": 384, "blocks": 2, "upsample": 2},
        {"dim": 256, "blocks": 3, "upsample": 2},
    ]
    analyze_config("CANDIDATE A: conservative scale (same block structure)", candidate_a)

    # Candidate B: scale dims + add depth at mid-resolution
    candidate_b = [
        {"dim": 192, "blocks": 2, "downsample": 2},
        {"dim": 256, "blocks": 4, "downsample": 2},
        {"dim": 384, "blocks": 3, "downsample": 2},
        {"dim": 448, "blocks": 2, "downsample": None},
        {"dim": 384, "blocks": 3, "upsample": 2},
        {"dim": 256, "blocks": 4, "upsample": 2},
    ]
    analyze_config("CANDIDATE B: more depth at mid-res, smaller bottleneck", candidate_b)

    # Candidate C: same as existing but with d_state=32
    analyze_config("CANDIDATE C: existing medium + d_state=32", medium_old, d_state=32)

    # Candidate D: balanced - more blocks everywhere, moderate dims
    candidate_d = [
        {"dim": 192, "blocks": 2, "downsample": 2},
        {"dim": 256, "blocks": 4, "downsample": 2},
        {"dim": 384, "blocks": 3, "downsample": 2},
        {"dim": 512, "blocks": 3, "downsample": None},
        {"dim": 384, "blocks": 3, "upsample": 2},
        {"dim": 256, "blocks": 4, "upsample": 2},
    ]
    analyze_config("CANDIDATE D: existing + deeper bottleneck (3 blocks)", candidate_d)

    ckpt_path = "checkpoints/small_fr_11022026/best_step_326577_val_wer_0.1339.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"\n{'='*70}")
        print(f"  Small checkpoint analysis: {ckpt_path}")
        print(f"{'='*70}")

        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # Check bypass scales (alpha values): tells us how much each stack contributes
        print("\n  Bypass scales (sigmoid(scale) → how much stack output vs input):")
        for key in sorted(state.keys()):
            if "bypass" in key and "scale" in key:
                val = state[key].item()
                alpha = torch.sigmoid(torch.tensor(val)).item()
                print(f"    {key}: raw={val:.4f} → alpha={alpha:.4f}")

        print("\n  Weight norm per stack (higher = more utilized):")
        stack_norms = defaultdict(float)
        stack_counts = defaultdict(int)
        for key, val in state.items():
            if "stacks" in key:
                parts = key.split(".")
                stack_idx = parts[1] if parts[0] == "encoder" else parts[2]
                stack_norms[stack_idx] += val.float().norm().item()
                stack_counts[stack_idx] += 1

        for idx in sorted(stack_norms.keys()):
            avg = stack_norms[idx] / max(stack_counts[idx], 1)
            print(f"    Stack {idx}: total_norm={stack_norms[idx]:.2f}, "
                  f"n_params={stack_counts[idx]}, avg_norm={avg:.4f}")

        if "model_config" in ckpt:
            print(f"\n  Stored model config: {ckpt['model_config']}")
        if "epoch" in ckpt:
            print(f"\n  Trained for {ckpt['epoch']} epochs")
        if "best_metric" in ckpt:
            print(f"  Best metric: {ckpt['best_metric']}")

    except FileNotFoundError:
        print(f"\n  Checkpoint not found: {ckpt_path}")
    except Exception as e:
        print(f"\n  Error loading checkpoint: {e}")


if __name__ == "__main__":
    main()
