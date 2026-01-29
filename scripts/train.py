"""Training script for ZipMamba ASR model."""

import argparse
from pathlib import Path

from rich.console import Console

from zipmamba.utils.config import load_config, merge_configs
from zipmamba.model.encoder import EfficientASRModel
from zipmamba.data.audio import AudioProcessor
from zipmamba.data.tokenizer import Tokenizer
from zipmamba.data.augmentation import (
    AudioAugmentation,
    SpecAugmentation,
    AugmentationController,
)
from zipmamba.data.dataset import create_dataloader
from zipmamba.training.trainer import Trainer

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ZipMamba ASR model")
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help="Path to config file(s). Can be specified multiple times.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda, cpu).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (for TensorBoard). If not set, uses timestamp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print("[bold]Loading configuration...[/bold]")
    configs = [load_config(cfg) for cfg in args.config]
    config = merge_configs(*configs)

    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    console.print(f"Model: {config.model.get('name', 'zipmamba')}")
    console.print(f"Max epochs: {config.training.max_epochs}")
    console.print(f"Batch size: {config.training.batch_size}")

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

    if not tokenizer_path.exists():
        console.print("[bold yellow]Training tokenizer...[/bold yellow]")
        import csv

        texts = []
        with open(config.data.train_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("sentence"):
                    texts.append(row["sentence"])

        console.print(f"Training on {len(texts)} sentences")

        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.train(
            texts=texts,
            model_prefix=str(tokenizer_path).replace(".model", ""),
            vocab_size=config.data.tokenizer.vocab_size,
        )
        console.print(f"[bold green]Tokenizer saved to {tokenizer_path}[/bold green]")

    console.print(f"Vocabulary size: {len(tokenizer)}")

    audio_aug = None
    spec_aug = None

    if config.data.augmentation.get("noise", {}).get("enabled", False):
        audio_aug = AudioAugmentation(
            noise_enabled=True,
            min_snr_db=config.data.augmentation.noise.get("min_snr_db", 5),
            max_snr_db=config.data.augmentation.noise.get("max_snr_db", 20),
            speed_perturb_enabled=config.data.augmentation.get("speed_perturb", {}).get(
                "enabled", False
            ),
            speed_rates=config.data.augmentation.get("speed_perturb", {}).get(
                "rates", [0.9, 1.0, 1.1]
            ),
            silence_enabled=config.data.augmentation.get("silence", {}).get(
                "enabled", False
            ),
            silence_config={
                "min_silence_duration": config.data.augmentation.get("silence", {}).get(
                    "min_silence_duration", 0.5
                ),
                "max_silence_duration": config.data.augmentation.get("silence", {}).get(
                    "max_silence_duration", 2.0
                ),
                "silence_prob": config.data.augmentation.get("silence", {}).get(
                    "silence_prob", 0.3
                ),
            },
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
            n_freq_masks=config.data.augmentation.spec_augment.get("n_freq_masks", 2),
            n_time_masks=config.data.augmentation.spec_augment.get("n_time_masks", 10),
            time_warp_w=config.data.augmentation.spec_augment.get("time_warp_w", 80),
            time_warp_p=config.data.augmentation.spec_augment.get("time_warp_p", 0.5),
        )

    aug_controller = AugmentationController(audio_aug=audio_aug, spec_aug=spec_aug)

    console.print("[bold]Creating dataloaders...[/bold]")

    num_workers = config.training.get("num_workers", 8)
    train_dataloader = create_dataloader(
        manifest_path=config.data.train_manifest,
        clips_dir=config.data.clips_dir,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        audio_augmentation=audio_aug,
        spec_augmentation=spec_aug,
        max_audio_length=config.training.max_audio_length,
        persistent_workers=True,
        prefetch_factor=config.training.get("prefetch_factor", 2),
    )

    val_dataloader = None
    if hasattr(config.data, "dev_manifest") and config.data.dev_manifest:
        val_dataloader = create_dataloader(
            manifest_path=config.data.dev_manifest,
            clips_dir=config.data.clips_dir,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            audio_augmentation=None,
            spec_augmentation=None,
            max_audio_length=config.training.max_audio_length,
            persistent_workers=True,
            prefetch_factor=config.training.get("prefetch_factor", 2),
        )

    console.print(f"Train samples: {len(train_dataloader.dataset)}")
    if val_dataloader:
        console.print(f"Val samples: {len(val_dataloader.dataset)}")

    console.print("[bold]Building model...[/bold]")

    stack_config = None
    if hasattr(config.model, "stacks"):
        stack_config = [dict(s) for s in config.model.stacks]

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
        intermediate_ctc_weight=config.model.get("intermediate_ctc_weight", 0.3),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Total parameters: {total_params:,}")
    console.print(f"Trainable parameters: {trainable_params:,}")

    aug_config = config.training.get("augmentation", {})
    aug_start_epoch = aug_config.get("start_epoch", 5)
    aug_warmup_epochs = aug_config.get("warmup_epochs", 5)

    console.print(
        f"Augmentation: starts at epoch {aug_start_epoch}, warmup {aug_warmup_epochs} epochs"
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        optimizer=config.training.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_epochs=config.training.warmup_epochs,
        max_epochs=config.training.max_epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_clip=config.training.gradient_clip,
        mixed_precision=config.training.mixed_precision,
        log_dir=config.training.logging.log_dir,
        log_every_n_steps=config.training.logging.log_every_n_steps,
        run_name=args.run_name,
        save_dir=config.training.checkpointing.save_dir,
        keep_top_k_checkpoints=config.training.checkpointing.keep_top_k,
        checkpoint_metric=config.training.checkpointing.metric,
        checkpoint_mode=config.training.checkpointing.mode,
        device=args.device,
        resume_from=args.resume,
        sample_rate=config.data.sample_rate,
        n_audio_samples=config.training.logging.get("n_audio_samples", 5),
        aug_start_epoch=aug_start_epoch,
        aug_warmup_epochs=aug_warmup_epochs,
        set_augmentation_prob_fn=aug_controller.set_probability,
    )

    trainer.train()


if __name__ == "__main__":
    main()
