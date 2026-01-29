"""Audio processing utilities for mel spectrogram extraction."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


class AudioProcessor:
    """Audio loading and mel spectrogram extraction.

    Handles audio file loading, resampling, and mel spectrogram computation.
    Output is at 100Hz (10ms frame shift).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalize: bool = True,
    ):
        """Initialize AudioProcessor.

        Args:
            sample_rate: Target sample rate in Hz.
            n_mels: Number of mel frequency bins.
            n_fft: FFT size.
            hop_length: Hop length in samples (160 @ 16kHz = 10ms = 100Hz).
            win_length: Window length (defaults to n_fft).
            f_min: Minimum frequency for mel filterbank.
            f_max: Maximum frequency for mel filterbank (defaults to sample_rate/2).
            normalize: Whether to normalize mel spectrograms.
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.normalize = normalize

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            power=2.0,
        )

        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # Cache for resamplers (avoid creating per sample)
        self._resampler_cache: dict[int, T.Resample] = {}

    def load_audio(
        self, path: Union[str, Path], max_duration: Optional[float] = None
    ) -> tuple[torch.Tensor, int]:
        """Load audio file and resample if necessary.

        Args:
            path: Path to audio file.
            max_duration: Maximum duration in seconds (optional).

        Returns:
            waveform: Audio tensor (1, samples).
            sample_rate: Original sample rate.
        """
        waveform, sr = torchaudio.load(path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary (use cached resampler for efficiency)
        if sr != self.sample_rate:
            if sr not in self._resampler_cache:
                self._resampler_cache[sr] = T.Resample(sr, self.sample_rate)
            waveform = self._resampler_cache[sr](waveform)

        # Truncate if max_duration specified
        if max_duration is not None:
            max_samples = int(max_duration * self.sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

        return waveform, self.sample_rate

    def compute_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from waveform.

        Args:
            waveform: Audio tensor (1, samples) or (samples,).

        Returns:
            mel: Mel spectrogram (time, n_mels).
        """
        # Ensure 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Compute mel spectrogram: (1, n_mels, time)
        mel = self.mel_transform(waveform)

        # Convert to dB scale
        mel = self.amplitude_to_db(mel)

        # Remove batch dim and transpose: (time, n_mels)
        mel = mel.squeeze(0).transpose(0, 1)

        # Normalize
        if self.normalize:
            mel = self._normalize(mel)

        return mel

    def _normalize(self, mel: torch.Tensor) -> torch.Tensor:
        """Normalize mel spectrogram to zero mean and unit variance.

        Args:
            mel: Mel spectrogram (time, n_mels).

        Returns:
            Normalized mel spectrogram.
        """
        mean = mel.mean()
        std = mel.std()
        if std > 0:
            mel = (mel - mean) / std
        return mel

    def process_file(
        self, path: Union[str, Path], max_duration: Optional[float] = None
    ) -> torch.Tensor:
        """Load audio file and compute mel spectrogram.

        Args:
            path: Path to audio file.
            max_duration: Maximum duration in seconds.

        Returns:
            mel: Mel spectrogram (time, n_mels).
        """
        waveform, _ = self.load_audio(path, max_duration)
        mel = self.compute_mel(waveform)
        return mel

    def get_output_length(self, input_samples: int) -> int:
        """Calculate mel spectrogram length from audio samples."""
        return (input_samples - self.n_fft) // self.hop_length + 1

    def get_duration_frames(self, duration_seconds: float) -> int:
        """Calculate number of mel frames for a given duration."""
        samples = int(duration_seconds * self.sample_rate)
        return self.get_output_length(samples)
