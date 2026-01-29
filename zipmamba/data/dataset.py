"""Dataset and DataLoader for Common Voice."""

import csv
import random
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .audio import AudioProcessor
from .tokenizer import Tokenizer
from .augmentation import AudioAugmentation, SpecAugmentation
from .text_normalizer import normalize_text


class CommonVoiceDataset(Dataset):
    """Dataset for Common Voice TSV format.

    Loads audio files, computes mel spectrograms, and tokenizes transcripts.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        clips_dir: Union[str, Path],
        audio_processor: AudioProcessor,
        tokenizer: Tokenizer,
        audio_augmentation: Optional[AudioAugmentation] = None,
        spec_augmentation: Optional[SpecAugmentation] = None,
        max_audio_length: Optional[float] = None,
        max_text_length: Optional[int] = None,
        downsample_factor: int = 8,
    ):
        """Initialize CommonVoiceDataset.

        Args:
            manifest_path: Path to TSV manifest file (train.tsv, dev.tsv, etc.).
            clips_dir: Path to clips directory containing audio files.
            audio_processor: AudioProcessor instance.
            tokenizer: Tokenizer instance.
            audio_augmentation: Audio augmentation (optional).
            spec_augmentation: SpecAugment (optional).
            max_audio_length: Maximum audio duration in seconds (optional).
            max_text_length: Maximum text length in characters (optional).
            downsample_factor: Total model downsampling factor for CTC check.
        """
        self.manifest_path = Path(manifest_path)
        self.clips_dir = Path(clips_dir)
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.audio_augmentation = audio_augmentation
        self.spec_augmentation = spec_augmentation
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.downsample_factor = downsample_factor

        # Load manifest
        self.samples = self._load_manifest()

    def _load_manifest(self) -> list[dict]:
        """Load and parse TSV manifest."""
        samples = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                audio_path = self.clips_dir / row["path"]
                sentence = row["sentence"].strip()

                # Skip if file doesn't exist
                if not audio_path.exists():
                    continue

                # Skip if text is empty
                if not sentence:
                    continue

                # Normalize text (remove unpronounced characters, normalize punctuation)
                sentence = normalize_text(sentence)

                # Skip if text is empty after normalization
                if not sentence:
                    continue

                # Skip if text is too long
                if self.max_text_length and len(sentence) > self.max_text_length:
                    continue

                samples.append(
                    {
                        "audio_path": audio_path,
                        "text": sentence,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            dict with keys:
                - mel: Mel spectrogram (time, n_mels)
                - mel_length: Number of mel frames
                - tokens: Token IDs (list)
                - token_length: Number of tokens
                - text: Original text
        """
        sample = self.samples[idx]

        # Load and process audio
        waveform, _ = self.audio_processor.load_audio(
            sample["audio_path"], max_duration=self.max_audio_length
        )

        # Apply audio augmentation (before mel)
        if self.audio_augmentation is not None:
            waveform = self.audio_augmentation(waveform)

        # Compute mel spectrogram
        mel = self.audio_processor.compute_mel(waveform)

        # Apply spec augmentation (after mel)
        if self.spec_augmentation is not None:
            mel = self.spec_augmentation(mel)

        # Tokenize text
        tokens = self.tokenizer.encode(sample["text"])

        # CTC validity check: output_frames must be >= token_length
        output_frames = mel.shape[0] // self.downsample_factor
        if output_frames < len(tokens):
            # Skip this sample and try another random one
            new_idx = random.randint(0, len(self.samples) - 1)
            while new_idx == idx:
                new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

        return {
            "mel": mel,
            "mel_length": mel.shape[0],
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "token_length": len(tokens),
            "text": sample["text"],
            "audio_path": str(sample["audio_path"]),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader.

    Pads mel spectrograms and token sequences to batch max length.

    Args:
        batch: List of samples from dataset.

    Returns:
        dict with padded tensors:
            - mel: (batch, max_time, n_mels)
            - mel_lengths: (batch,)
            - tokens: (batch, max_tokens)
            - token_lengths: (batch,)
            - texts: list of strings
    """
    mels = [s["mel"] for s in batch]
    mel_lengths = torch.tensor([s["mel_length"] for s in batch], dtype=torch.long)
    tokens = [s["tokens"] for s in batch]
    token_lengths = torch.tensor([s["token_length"] for s in batch], dtype=torch.long)
    texts = [s["text"] for s in batch]
    audio_paths = [s["audio_path"] for s in batch]

    # Pad mels: (batch, max_time, n_mels)
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=0.0)

    # Pad tokens: (batch, max_tokens)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

    return {
        "mel": mels_padded,
        "mel_lengths": mel_lengths,
        "tokens": tokens_padded,
        "token_lengths": token_lengths,
        "texts": texts,
        "audio_paths": audio_paths,
    }


def create_dataloader(
    manifest_path: Union[str, Path],
    clips_dir: Union[str, Path],
    audio_processor: AudioProcessor,
    tokenizer: Tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    audio_augmentation: Optional[AudioAugmentation] = None,
    spec_augmentation: Optional[SpecAugmentation] = None,
    max_audio_length: Optional[float] = None,
    max_text_length: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create a DataLoader for Common Voice dataset.

    Args:
        manifest_path: Path to TSV manifest.
        clips_dir: Path to clips directory.
        audio_processor: AudioProcessor instance.
        tokenizer: Tokenizer instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        audio_augmentation: Audio augmentation (optional).
        spec_augmentation: SpecAugment (optional).
        max_audio_length: Maximum audio duration in seconds.
        max_text_length: Maximum text length in characters.
        pin_memory: Whether to pin memory for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs (faster).
        prefetch_factor: Number of batches to prefetch per worker.

    Returns:
        DataLoader instance.
    """
    dataset = CommonVoiceDataset(
        manifest_path=manifest_path,
        clips_dir=clips_dir,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        audio_augmentation=audio_augmentation,
        spec_augmentation=spec_augmentation,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=shuffle,  # Drop last incomplete batch during training
        persistent_workers=num_workers > 0 and persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
