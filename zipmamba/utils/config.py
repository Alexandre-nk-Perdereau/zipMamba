"""Configuration loading and merging utilities using OmegaConf."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        DictConfig object with the configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return OmegaConf.load(config_path)


def merge_configs(*configs: DictConfig | dict | str | Path) -> DictConfig:
    """Merge multiple configurations.

    Later configs override earlier ones for conflicting keys.

    Args:
        *configs: Config objects, dicts, or paths to YAML files.

    Returns:
        Merged DictConfig.
    """
    loaded_configs = []
    for cfg in configs:
        if isinstance(cfg, (str, Path)):
            loaded_configs.append(load_config(cfg))
        elif isinstance(cfg, dict):
            loaded_configs.append(OmegaConf.create(cfg))
        else:
            loaded_configs.append(cfg)

    return OmegaConf.merge(*loaded_configs)


def config_to_dict(config: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to a regular dict."""
    return OmegaConf.to_container(config, resolve=True)


def validate_config(config: DictConfig, required_keys: list[str]) -> None:
    """Validate that required keys are present in config.

    Args:
        config: The configuration to validate.
        required_keys: List of dot-separated keys that must exist.

    Raises:
        ValueError: If a required key is missing.
    """
    for key in required_keys:
        parts = key.split(".")
        current = config
        for part in parts:
            if not OmegaConf.is_missing(current, part) and part in current:
                current = current[part]
            else:
                raise ValueError(f"Required config key missing: {key}")


def get_default_model_config() -> DictConfig:
    """Get default model configuration."""
    return OmegaConf.create(
        {
            "model": {
                "name": "zipmamba_medium",
                "input_dim": 80,
                "vocab_size": 5000,
                "d_state": 16,
                "d_conv": 4,
                "mamba_expand": 2,
                "ff_expand": 4,
                "conv_kernel_size": 31,
                "dropout": 0.1,
                "stacks": [
                    {"dim": 192, "blocks": 2, "downsample": 2},
                    {"dim": 256, "blocks": 4, "downsample": 2},
                    {"dim": 384, "blocks": 3, "downsample": 2},
                    {"dim": 512, "blocks": 2, "downsample": None},
                    {"dim": 384, "blocks": 3, "upsample": 2},
                    {"dim": 256, "blocks": 4, "upsample": None},
                ],
            }
        }
    )


def get_default_training_config() -> DictConfig:
    """Get default training configuration."""
    return OmegaConf.create(
        {
            "training": {
                "optimizer": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "warmup_steps": 10000,
                "max_steps": 500000,
                "batch_size": 32,
                "gradient_accumulation_steps": 4,
                "max_audio_length": 30.0,
                "mixed_precision": "bf16",
                "gradient_clip": 1.0,
                "logging": {
                    "backend": "tensorboard",
                    "log_dir": "./logs",
                    "log_every_n_steps": 100,
                },
                "checkpointing": {
                    "save_dir": "./checkpoints",
                    "save_every_n_steps": 5000,
                    "keep_last_n": 3,
                },
            }
        }
    )
