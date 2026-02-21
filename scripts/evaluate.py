"""Evaluation script for ZipMamba ASR model."""

import argparse
import csv
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from zipmamba.utils.config import load_config, merge_configs
from zipmamba.model.encoder import EfficientASRModel
from zipmamba.data.audio import AudioProcessor
from zipmamba.data.tokenizer import Tokenizer
from zipmamba.data.dataset import create_dataloader
from zipmamba.utils.metrics import compute_wer, compute_cer, compute_wer_normalized, compute_cer_normalized

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ZipMamba ASR model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help="Path to config file(s).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to evaluate on.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions (optional).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("[bold]Loading configuration...[/bold]")
    configs = [load_config(cfg) for cfg in args.config]
    config = merge_configs(*configs)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    console.print(f"Device: {device}")

    audio_processor = AudioProcessor(
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
    )

    tokenizer = Tokenizer(model_path=config.data.tokenizer.model_path)
    console.print(f"Vocabulary size: {len(tokenizer)}")

    if args.split == "test":
        manifest_path = config.data.test_manifest
    else:
        manifest_path = config.data.dev_manifest

    dataloader = create_dataloader(
        manifest_path=manifest_path,
        clips_dir=config.data.clips_dir,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        audio_augmentation=None,
        spec_augmentation=None,
    )

    console.print(f"Evaluating on {len(dataloader.dataset)} samples")

    manifest_meta = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            manifest_meta[row["path"]] = {
                "down_votes": int(row.get("down_votes", 0)),
                "up_votes": int(row.get("up_votes", 0)),
            }

    console.print("[bold]Loading model...[/bold]")

    stack_config = None
    if hasattr(config.model, "stacks"):
        stack_config = [dict(s) for s in config.model.stacks]

    conv_channels_cfg = config.model.get("conv_channels", None)
    conv_channels = tuple(conv_channels_cfg) if conv_channels_cfg else None

    model = EfficientASRModel(
        vocab_size=len(tokenizer),
        input_dim=config.model.input_dim,
        stack_config=stack_config,
        d_state=config.model.d_state,
        d_conv=config.model.d_conv,
        mamba_expand=config.model.mamba_expand,
        ff_expand=config.model.ff_expand,
        conv_kernel_size=config.model.conv_kernel_size,
        dropout=config.model.dropout,
        use_intermediate_ctc=getattr(config.model, "use_intermediate_ctc", False),
        conv_channels=conv_channels,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Handle renamed keys from older checkpoints
    state_dict = checkpoint["model_state_dict"]
    key_mapping = {
        "encoder.remaining_skip_ds.weights": "encoder.remaining_skip_resamplers.0.weights",
        "encoder.remaining_skip_proj.weight": "encoder.remaining_skip_projs.0.weight",
        "encoder.remaining_skip_proj.bias": "encoder.remaining_skip_projs.0.bias",
    }
    for old_key, new_key in key_mapping.items():
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    console.print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    console.print("[bold]Running evaluation...[/bold]")

    all_references = []
    all_hypotheses = []
    all_predictions = []
    all_down_votes = []

    total_inference_time = 0.0
    total_audio_duration = 0.0
    hop_length = config.data.hop_length
    sample_rate = config.data.sample_rate

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            mel = batch["mel"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            texts = batch["texts"]
            audio_paths = batch["audio_paths"]

            if batch_idx == 0:
                _ = model.decode_greedy(mel, mel_lengths)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            decoded = model.decode_greedy(mel, mel_lengths)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            if batch_idx > 0:
                total_inference_time += end_time - start_time
                batch_audio_duration = (
                    mel_lengths.sum().item() * hop_length
                ) / sample_rate
                total_audio_duration += batch_audio_duration

            hypotheses = [tokenizer.decode(d) for d in decoded]

            all_references.extend(texts)
            all_hypotheses.extend(hypotheses)

            for ref, hyp, ap in zip(texts, hypotheses, audio_paths):
                filename = Path(ap).name
                meta = manifest_meta.get(filename, {})
                dv = meta.get("down_votes", 0)
                all_down_votes.append(dv)
                all_predictions.append({"reference": ref, "hypothesis": hyp, "down_votes": dv})

    wer = compute_wer(all_references, all_hypotheses)
    cer = compute_cer(all_references, all_hypotheses)
    wer_norm = compute_wer_normalized(all_references, all_hypotheses)
    cer_norm = compute_cer_normalized(all_references, all_hypotheses)

    rtfx = (
        total_audio_duration / total_inference_time if total_inference_time > 0 else 0.0
    )

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("WER", f"{wer * 100:.2f}%")
    table.add_row("CER", f"{cer * 100:.2f}%")
    table.add_row("WER (normalized)", f"{wer_norm * 100:.2f}%")
    table.add_row("CER (normalized)", f"{cer_norm * 100:.2f}%")
    table.add_row("RTFx", f"{rtfx:.0f}x")
    table.add_row("Samples", str(len(all_references)))
    table.add_row("Audio duration", f"{total_audio_duration:.1f}s")
    table.add_row("Inference time", f"{total_inference_time:.1f}s")

    console.print(table)

    groups = {
        "down_votes == 0": [i for i, dv in enumerate(all_down_votes) if dv == 0],
        "down_votes > 0": [i for i, dv in enumerate(all_down_votes) if dv > 0],
    }

    group_table = Table(title="Breakdown by down_votes")
    group_table.add_column("Group", style="cyan")
    group_table.add_column("N", style="white")
    group_table.add_column("WER", style="green")
    group_table.add_column("CER", style="green")
    group_table.add_column("WER norm.", style="green")
    group_table.add_column("CER norm.", style="green")

    for group_name, indices in groups.items():
        if not indices:
            continue
        g_refs = [all_references[i] for i in indices]
        g_hyps = [all_hypotheses[i] for i in indices]
        group_table.add_row(
            group_name,
            str(len(indices)),
            f"{compute_wer(g_refs, g_hyps) * 100:.2f}%",
            f"{compute_cer(g_refs, g_hyps) * 100:.2f}%",
            f"{compute_wer_normalized(g_refs, g_hyps) * 100:.2f}%",
            f"{compute_cer_normalized(g_refs, g_hyps) * 100:.2f}%",
        )

    console.print(group_table)

    console.print("\n[bold]Sample predictions:[/bold]")
    for i in range(min(5, len(all_predictions))):
        pred = all_predictions[i]
        console.print(f"\n[cyan]Reference:[/cyan] {pred['reference']}")
        console.print(f"[green]Hypothesis:[/green] {pred['hypothesis']}")

    if args.output:
        import json

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "metrics": {
                "wer": wer,
                "cer": cer,
                "wer_norm": wer_norm,
                "cer_norm": cer_norm,
                "rtfx": rtfx,
                "samples": len(all_references),
                "audio_duration_s": total_audio_duration,
                "inference_time_s": total_inference_time,
            },
            "predictions": all_predictions,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        console.print(f"\n[bold]Predictions saved to {output_path}[/bold]")


if __name__ == "__main__":
    main()
