"""Benchmark dataloader to identify bottlenecks."""

import time
import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from zipmamba.utils.config import load_config, merge_configs
from zipmamba.data.audio import AudioProcessor
from zipmamba.data.tokenizer import Tokenizer
from zipmamba.data.augmentation import AudioAugmentation, SpecAugmentation
from zipmamba.data.dataset import create_dataloader, CommonVoiceDataset

console = Console()


def benchmark_single_sample(dataset, n_samples=100):
    """Benchmark single sample loading without DataLoader."""
    console.print(
        f"\n[bold]Benchmarking single sample loading ({n_samples} samples)...[/bold]"
    )

    times = []
    for i in range(n_samples):
        start = time.perf_counter()
        _ = dataset[i]
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if (i + 1) % 20 == 0:
            console.print(f"  Sample {i + 1}/{n_samples}: {elapsed * 1000:.1f}ms")

    avg_time = sum(times) / len(times)
    console.print(f"\n[green]Average per-sample time: {avg_time * 1000:.1f}ms[/green]")
    console.print(
        f"[green]Theoretical max throughput (1 worker): {1 / avg_time:.1f} samples/sec[/green]"
    )
    return avg_time


def benchmark_dataloader(dataloader, n_batches=50, warmup=5):
    """Benchmark DataLoader throughput."""
    console.print(
        f"\n[bold]Benchmarking DataLoader ({n_batches} batches, {warmup} warmup)...[/bold]"
    )

    # Warmup
    dataloader_iter = iter(dataloader)
    for _ in range(warmup):
        try:
            _ = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            _ = next(dataloader_iter)

    # Benchmark
    times = []
    for i in range(n_batches):
        start = time.perf_counter()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if (i + 1) % 10 == 0:
            console.print(f"  Batch {i + 1}/{n_batches}: {elapsed * 1000:.1f}ms")

    avg_time = sum(times) / len(times)
    batch_size = dataloader.batch_size
    samples_per_sec = batch_size / avg_time

    console.print(f"\n[green]Average batch time: {avg_time * 1000:.1f}ms[/green]")
    console.print(
        f"[green]Throughput: {samples_per_sec:.1f} samples/sec ({1 / avg_time:.2f} batches/sec)[/green]"
    )
    return avg_time, samples_per_sec


def benchmark_gpu_transfer(dataloader, device, n_batches=20):
    """Benchmark CPU->GPU transfer time."""
    console.print(f"\n[bold]Benchmarking GPU transfer ({n_batches} batches)...[/bold]")

    dataloader_iter = iter(dataloader)
    times = []

    for i in range(n_batches):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        torch.cuda.synchronize()
        start = time.perf_counter()
        mel = batch["mel"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    console.print(f"[green]Average GPU transfer time: {avg_time * 1000:.2f}ms[/green]")
    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark dataloader")
    parser.add_argument("--config", type=str, action="append", required=True)
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Override num_workers"
    )
    parser.add_argument(
        "--no_augmentation", action="store_true", help="Disable augmentation"
    )
    args = parser.parse_args()

    # Load config
    configs = [load_config(cfg) for cfg in args.config]
    config = merge_configs(*configs)

    # Initialize components
    audio_processor = AudioProcessor(
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
    )

    tokenizer_path = Path(config.data.tokenizer.model_path)
    tokenizer = Tokenizer(
        model_path=tokenizer_path if tokenizer_path.exists() else None,
        vocab_size=config.data.tokenizer.vocab_size,
    )

    # Augmentation
    audio_aug = None
    spec_aug = None

    if not args.no_augmentation:
        if config.data.augmentation.get("noise", {}).get("enabled", False):
            audio_aug = AudioAugmentation(
                noise_enabled=True,
                min_snr_db=config.data.augmentation.noise.get("min_snr_db", 5),
                max_snr_db=config.data.augmentation.noise.get("max_snr_db", 20),
                speed_perturb_enabled=config.data.augmentation.get(
                    "speed_perturb", {}
                ).get("enabled", False),
                speed_rates=config.data.augmentation.get("speed_perturb", {}).get(
                    "rates", [0.9, 1.0, 1.1]
                ),
                silence_enabled=config.data.augmentation.get("silence", {}).get(
                    "enabled", False
                ),
                sample_rate=config.data.sample_rate,
            )
        if config.data.augmentation.get("spec_augment", {}).get("enabled", False):
            spec_aug = SpecAugmentation(
                freq_mask_param=config.data.augmentation.spec_augment.get(
                    "freq_mask_param", 27
                ),
                time_mask_param=config.data.augmentation.spec_augment.get(
                    "time_mask_param", 100
                ),
            )

    # Create dataset directly for single-sample benchmark
    dataset = CommonVoiceDataset(
        manifest_path=config.data.train_manifest,
        clips_dir=config.data.clips_dir,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        audio_augmentation=audio_aug,
        spec_augmentation=spec_aug,
        max_audio_length=config.training.max_audio_length,
    )

    console.print(f"\n[bold cyan]Dataset size: {len(dataset)} samples[/bold cyan]")
    console.print(f"Augmentation: {'enabled' if audio_aug else 'disabled'}")

    # Single sample benchmark
    single_sample_time = benchmark_single_sample(dataset, n_samples=50)

    # Test different num_workers
    num_workers_list = [args.num_workers] if args.num_workers else [0, 2, 4, 8, 12]
    results = []

    for nw in num_workers_list:
        console.print(f"\n[bold yellow]Testing num_workers={nw}[/bold yellow]")

        from zipmamba.data.dataset import collate_fn
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=nw > 0,  # Test with persistent workers
            prefetch_factor=2 if nw > 0 else None,
        )

        batch_time, samples_per_sec = benchmark_dataloader(
            dataloader, n_batches=30, warmup=5
        )
        results.append((nw, batch_time, samples_per_sec))

    # Summary table
    table = Table(title="DataLoader Benchmark Results")
    table.add_column("num_workers", style="cyan")
    table.add_column("Batch Time (ms)", style="green")
    table.add_column("Samples/sec", style="green")
    table.add_column("Batches/sec", style="green")

    for nw, bt, sps in results:
        table.add_row(
            str(nw),
            f"{bt * 1000:.1f}",
            f"{sps:.1f}",
            f"{1 / bt:.2f}",
        )

    console.print("\n")
    console.print(table)

    # GPU transfer benchmark
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dataloader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        benchmark_gpu_transfer(dataloader, device)

    # Analysis
    console.print("\n[bold]Analysis:[/bold]")
    console.print(f"- Single sample processing: {single_sample_time * 1000:.1f}ms")
    console.print(f"- With batch_size={config.training.batch_size} and 4 workers:")
    console.print(f"  - Theoretical max: {4 / single_sample_time:.1f} samples/sec")
    console.print(
        f"  - Actual: {results[2][2] if len(results) > 2 else results[0][2]:.1f} samples/sec"
    )


if __name__ == "__main__":
    main()
