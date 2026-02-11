"""Transcription script for ZipMamba ASR model."""

import argparse
from pathlib import Path

import torch
from rich.console import Console

from zipmamba.utils.config import load_config, merge_configs
from zipmamba.model.encoder import EfficientASRModel
from zipmamba.data.audio import AudioProcessor
from zipmamba.data.tokenizer import Tokenizer

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio with ZipMamba")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file to transcribe.",
    )
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
        "--device",
        type=str,
        default=None,
        help="Device to run on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    configs = [load_config(cfg) for cfg in args.config]
    config = merge_configs(*configs)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    audio_processor = AudioProcessor(
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
    )

    tokenizer = Tokenizer(model_path=config.data.tokenizer.model_path)

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
        drop_path=config.model.get("drop_path", 0.0),
        use_intermediate_ctc=config.model.get("use_intermediate_ctc", False),
        conv_channels=conv_channels,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        console.print(f"[red]Error: Audio file not found: {audio_path}[/red]")
        return

    console.print(f"[bold]Transcribing: {audio_path}[/bold]")

    mel = audio_processor.process_file(audio_path)
    mel = mel.unsqueeze(0).to(device)
    mel_lengths = torch.tensor([mel.shape[1]], device=device)

    with torch.no_grad():
        decoded = model.decode_greedy(mel, mel_lengths)
        transcription = tokenizer.decode(decoded[0])

    console.print(f"\n[green]Transcription:[/green] {transcription}")


if __name__ == "__main__":
    main()
