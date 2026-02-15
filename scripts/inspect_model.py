import sys
sys.path.insert(0, ".")

import torch
from collections import defaultdict


def load_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    return state, ckpt


def bypass_analysis(state, label):
    """Bypass alpha: how much each stack's output matters vs its input."""
    print(f"\n  Bypass scales ({label}):")
    print(f"  {'Stack':<8} {'raw':>8} {'alpha':>8}  {'interpretation'}")
    print(f"  {'-'*55}")
    for key in sorted(state.keys()):
        if "bypass" in key and "scale" in key:
            idx = key.split("bypasses.")[1].split(".")[0]
            val = state[key].item()
            alpha = torch.sigmoid(torch.tensor(val)).item()
            if alpha < 0.5:
                interp = "UNDERUSED (bypass dominates)"
            elif alpha < 0.7:
                interp = "moderate"
            elif alpha < 0.9:
                interp = "well used"
            else:
                interp = "critical (high reliance)"
            print(f"  {idx:<8} {val:>8.4f} {alpha:>8.4f}  {interp}")


def weight_norm_per_component(state, label):
    """Weight norms per stack and component type — shows where learning happened."""
    print(f"\n  Weight norms by stack & component ({label}):")

    stacks = defaultdict(lambda: defaultdict(list))
    for key, val in state.items():
        if "stacks" not in key:
            continue
        parts = key.split(".")
        stack_idx = int(parts[2])
        if "ff1" in key or "ff2" in key:
            comp = "ffn"
        elif "bimamba" in key:
            comp = "mamba"
        elif "conv" in key:
            comp = "conv"
        else:
            comp = "other"
        stacks[stack_idx][comp].append(val.float().norm().item())

    print(f"  {'Stack':<8} {'FFN':>10} {'Mamba':>10} {'Conv':>10} {'Total':>10}")
    print(f"  {'-'*50}")
    for idx in sorted(stacks.keys()):
        ffn = sum(stacks[idx].get("ffn", []))
        mamba = sum(stacks[idx].get("mamba", []))
        conv = sum(stacks[idx].get("conv", []))
        total = ffn + mamba + conv
        print(f"  {idx:<8} {ffn:>10.1f} {mamba:>10.1f} {conv:>10.1f} {total:>10.1f}")


def effective_rank(matrix, threshold=0.99):
    """Compute effective rank: number of singular values capturing `threshold` of total energy.
    Also returns the ratio effective_rank / min(rows, cols) as utilization %."""
    if matrix.dim() != 2:
        return None, None, None
    S = torch.linalg.svdvals(matrix.float())
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    eff_rank = (energy < threshold).sum().item() + 1
    max_rank = min(matrix.shape)
    utilization = eff_rank / max_rank
    return eff_rank, max_rank, utilization


def rank_analysis(state, label):
    """Effective rank of key weight matrices — low utilization = wasted capacity."""
    print(f"\n  Effective rank analysis ({label}, threshold=99% energy):")
    print(f"  {'Layer':<55} {'Shape':>15} {'EffRank':>8} {'MaxRank':>8} {'Util%':>7}")
    print(f"  {'-'*95}")

    stack_utils = defaultdict(list)

    targets = ["w1.weight", "w2.weight", "w3.weight",
               "out_proj.weight", "in_proj.weight",
               "mamba_forward.in_proj.weight", "mamba_backward.in_proj.weight",
               "mamba_forward.out_proj.weight", "mamba_backward.out_proj.weight"]

    for key in sorted(state.keys()):
        if not any(key.endswith(t) for t in targets):
            continue
        val = state[key]
        if val.dim() != 2:
            continue

        eff, maxr, util = effective_rank(val)
        if eff is None:
            continue

        shape_str = f"{val.shape[0]}x{val.shape[1]}"

        stack_idx = None
        if "stacks" in key:
            stack_idx = int(key.split("stacks.")[1].split(".")[0])
            stack_utils[stack_idx].append(util)

        short_key = key.replace("encoder.", "").replace(".weight", "")
        if len(short_key) > 53:
            short_key = "..." + short_key[-50:]

        flag = " <<<" if util < 0.5 else ""
        print(f"  {short_key:<55} {shape_str:>15} {eff:>8d} {maxr:>8d} {util:>6.1%}{flag}")

    print(f"\n  Average rank utilization per stack:")
    print(f"  {'Stack':<8} {'Avg Util%':>10} {'N matrices':>12}")
    print(f"  {'-'*32}")
    for idx in sorted(stack_utils.keys()):
        utils = stack_utils[idx]
        avg = sum(utils) / len(utils)
        flag = " <<< LOW" if avg < 0.6 else ""
        print(f"  {idx:<8} {avg:>9.1%} {len(utils):>12}{flag}")


def conv_embed_analysis(state, label):
    """Check ConvEmbed weight norms."""
    print(f"\n  ConvEmbed analysis ({label}):")
    for key in sorted(state.keys()):
        if "conv_embed" not in key:
            continue
        val = state[key]
        print(f"    {key}: shape={list(val.shape)}, norm={val.float().norm():.2f}")


def comparative_analysis(state_small, state_medium):
    """Side-by-side comparison of small vs medium trained weights."""
    print(f"\n{'='*70}")
    print(f"  COMPARATIVE ANALYSIS: Small vs Medium")
    print(f"{'='*70}")

    # Bypass comparison
    print(f"\n  Bypass alpha comparison:")
    print(f"  {'Stack':<8} {'Small α':>10} {'Medium α':>10} {'Delta':>8} {'Note'}")
    print(f"  {'-'*55}")

    small_alphas = {}
    medium_alphas = {}
    for key in sorted(state_small.keys()):
        if "bypass" in key and "scale" in key:
            idx = int(key.split("bypasses.")[1].split(".")[0])
            small_alphas[idx] = torch.sigmoid(torch.tensor(state_small[key].item())).item()
    for key in sorted(state_medium.keys()):
        if "bypass" in key and "scale" in key:
            idx = int(key.split("bypasses.")[1].split(".")[0])
            medium_alphas[idx] = torch.sigmoid(torch.tensor(state_medium[key].item())).item()

    for idx in sorted(set(small_alphas) | set(medium_alphas)):
        sa = small_alphas.get(idx, float("nan"))
        ma = medium_alphas.get(idx, float("nan"))
        delta = ma - sa
        note = ""
        if delta < -0.05:
            note = "REGRESSED (less useful with more params)"
        elif delta > 0.05:
            note = "improved"
        print(f"  {idx:<8} {sa:>10.4f} {ma:>10.4f} {delta:>+8.4f} {note}")

    print(f"\n  Weight norm density (norm / sqrt(n_params)) per stack:")
    print(f"  {'Stack':<8} {'Small':>10} {'Medium':>10} {'Ratio':>8}")
    print(f"  {'-'*38}")

    def get_stack_norm_density(st):
        norms = defaultdict(float)
        counts = defaultdict(int)
        for key, val in st.items():
            if "stacks" not in key:
                continue
            idx = int(key.split("stacks.")[1].split(".")[0])
            norms[idx] += val.float().norm().item() ** 2
            counts[idx] += val.numel()
        return {idx: (norms[idx] ** 0.5) / (counts[idx] ** 0.5) for idx in norms}

    sd_small = get_stack_norm_density(state_small)
    sd_medium = get_stack_norm_density(state_medium)

    for idx in sorted(set(sd_small) | set(sd_medium)):
        s = sd_small.get(idx, 0)
        m = sd_medium.get(idx, 0)
        ratio = m / s if s > 0 else float("nan")
        flag = " <<< DILUTED" if ratio < 0.8 else ""
        print(f"  {idx:<8} {s:>10.4f} {m:>10.4f} {ratio:>7.2f}x{flag}")


def main():
    small_ckpt_path = "checkpoints/small_fr_11022026/best_step_326577_val_wer_0.1339.pt"
    medium_ckpt_path = "checkpoints/20260212_001303/best_step_331310_val_wer_0.1219.pt"

    # --- SMALL ---
    print("=" * 70)
    print(f"  SMALL: {small_ckpt_path}")
    print("=" * 70)
    state_small, ckpt_small = load_state(small_ckpt_path)
    print(f"  Epoch: {ckpt_small.get('epoch', '?')}")

    bypass_analysis(state_small, "small")
    weight_norm_per_component(state_small, "small")
    rank_analysis(state_small, "small")
    conv_embed_analysis(state_small, "small")

    # --- MEDIUM ---
    print(f"\n{'='*70}")
    print(f"  MEDIUM: {medium_ckpt_path}")
    print("=" * 70)
    state_medium, ckpt_medium = load_state(medium_ckpt_path)
    print(f"  Epoch: {ckpt_medium.get('epoch', '?')}")

    bypass_analysis(state_medium, "medium")
    weight_norm_per_component(state_medium, "medium")
    rank_analysis(state_medium, "medium")
    conv_embed_analysis(state_medium, "medium")

    comparative_analysis(state_small, state_medium)


if __name__ == "__main__":
    main()
