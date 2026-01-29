"""Audio and spectrogram augmentation for ASR training."""

import random
from typing import Optional

import numpy as np
import torch

# Try to import audiomentations for advanced augmentations
try:
    import audiomentations as am

    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False


class SpecAugmentation:
    """SpecAugment: time warping, time and frequency masking on mel spectrograms.

    Reference: "SpecAugment: A Simple Data Augmentation Method for ASR"
    https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 10,
        time_warp_w: int = 80,
        time_warp_p: float = 0.5,
        p: float = 1.0,
    ):
        """Initialize SpecAugmentation.

        Args:
            freq_mask_param: Maximum frequency mask width.
            time_mask_param: Maximum time mask width.
            n_freq_masks: Number of frequency masks.
            n_time_masks: Number of time masks.
            time_warp_w: Time warp parameter W (max warp distance).
            time_warp_p: Probability of applying time warp.
            p: Base probability of applying augmentation.
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.time_warp_w = time_warp_w
        self.time_warp_p = time_warp_p
        self.p = p
        # Global probability multiplier (set by trainer)
        self._prob_multiplier = 1.0

    def set_prob_multiplier(self, multiplier: float) -> None:
        """Set the probability multiplier for augmentation scheduling.

        Args:
            multiplier: Value between 0.0 and 1.0 to scale augmentation probability.
        """
        self._prob_multiplier = max(0.0, min(1.0, multiplier))

    @property
    def effective_prob(self) -> float:
        """Get the effective probability after applying multiplier."""
        return self.p * self._prob_multiplier

    def _time_warp(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply time warping to a single spectrogram.

        Uses sparse image warping to distort the time axis non-linearly.

        Args:
            mel: Mel spectrogram (time, freq).

        Returns:
            Time-warped spectrogram.
        """
        time_len, freq_len = mel.shape
        W = self.time_warp_w

        if time_len < 2 * W + 1:
            return mel

        src_pt = random.randint(W, time_len - W - 1)

        warp_dist = random.randint(-W, W)
        if warp_dist == 0:
            return mel

        dst_pt = src_pt + warp_dist

        left_indices = torch.linspace(0, src_pt, dst_pt + 1, device=mel.device)
        right_indices = torch.linspace(
            src_pt, time_len - 1, time_len - dst_pt, device=mel.device
        )
        warp_indices = torch.cat([left_indices[:-1], right_indices])

        if len(warp_indices) != time_len:
            warp_indices = torch.nn.functional.interpolate(
                warp_indices.unsqueeze(0).unsqueeze(0),
                size=time_len,
                mode="linear",
                align_corners=True,
            ).squeeze()

        warp_indices = 2.0 * warp_indices / (time_len - 1) - 1.0

        freq_indices = torch.linspace(-1, 1, freq_len, device=mel.device)
        grid_y, grid_x = torch.meshgrid(warp_indices, freq_indices, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        mel_4d = mel.unsqueeze(0).unsqueeze(0)
        warped = torch.nn.functional.grid_sample(
            mel_4d, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        return warped.squeeze(0).squeeze(0)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment.

        Args:
            mel: Mel spectrogram (time, freq) or (batch, time, freq).

        Returns:
            Augmented mel spectrogram.
        """
        if random.random() > self.effective_prob:
            return mel

        # Handle batch dimension
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch, time, freq = mel.shape
        mel = mel.clone()

        for b in range(batch):
            # Time warping (applied first, before masking)
            if random.random() < self.time_warp_p * self._prob_multiplier:
                mel[b] = self._time_warp(mel[b])

            # Frequency masking
            for _ in range(self.n_freq_masks):
                f = random.randint(0, min(self.freq_mask_param, freq - 1))
                f0 = random.randint(0, max(1, freq - f) - 1)
                mel[b, :, f0 : f0 + f] = 0

            # Time masking
            for _ in range(self.n_time_masks):
                t = random.randint(0, min(self.time_mask_param, time // 2))
                t0 = random.randint(0, max(1, time - t) - 1)
                mel[b, t0 : t0 + t, :] = 0

        if squeeze_batch:
            mel = mel.squeeze(0)

        return mel


class SilenceAugmentation:
    """Insert silence segments to prevent hallucination on silence.

    Adds random silence at the beginning, end, or middle of audio
    to train the model to output blank tokens on silence.
    """

    def __init__(
        self,
        min_silence_duration: float = 0.5,
        max_silence_duration: float = 2.0,
        silence_prob: float = 0.3,
        sample_rate: int = 16000,
    ):
        """Initialize SilenceAugmentation.

        Args:
            min_silence_duration: Minimum silence duration in seconds.
            max_silence_duration: Maximum silence duration in seconds.
            silence_prob: Probability of inserting silence.
            sample_rate: Audio sample rate.
        """
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        self.silence_prob = silence_prob
        self.sample_rate = sample_rate
        self._prob_multiplier = 1.0

    def set_prob_multiplier(self, multiplier: float) -> None:
        """Set the probability multiplier for augmentation scheduling."""
        self._prob_multiplier = max(0.0, min(1.0, multiplier))

    @property
    def effective_prob(self) -> float:
        """Get the effective probability after applying multiplier."""
        return self.silence_prob * self._prob_multiplier

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply silence augmentation.

        Args:
            waveform: Audio waveform (samples,) or (1, samples).

        Returns:
            Augmented waveform.
        """
        if random.random() > self.effective_prob:
            return waveform

        # Handle dimensions
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_channel = True
        else:
            squeeze_channel = False

        # Generate silence duration
        duration = random.uniform(self.min_silence_duration, self.max_silence_duration)
        silence_samples = int(duration * self.sample_rate)
        silence = torch.zeros(1, silence_samples, device=waveform.device)

        # Choose where to insert: 0=start, 1=end, 2=middle
        position = random.randint(0, 2)

        if position == 0:
            # Insert at start
            waveform = torch.cat([silence, waveform], dim=1)
        elif position == 1:
            # Insert at end
            waveform = torch.cat([waveform, silence], dim=1)
        else:
            # Insert in middle
            mid = waveform.shape[1] // 2
            waveform = torch.cat([waveform[:, :mid], silence, waveform[:, mid:]], dim=1)

        if squeeze_channel:
            waveform = waveform.squeeze(0)

        return waveform


class AudioAugmentation:
    """Audio-level augmentations applied before mel extraction.

    Includes:
    - Gaussian noise injection
    - Speed perturbation
    - Silence insertion
    """

    def __init__(
        self,
        noise_enabled: bool = True,
        min_snr_db: float = 5.0,
        max_snr_db: float = 20.0,
        noise_prob: float = 0.5,
        speed_perturb_enabled: bool = True,
        speed_rates: Optional[list[float]] = None,
        silence_enabled: bool = True,
        silence_config: Optional[dict] = None,
        sample_rate: int = 16000,
    ):
        """Initialize AudioAugmentation.

        Args:
            noise_enabled: Whether to enable noise injection.
            min_snr_db: Minimum SNR in dB for noise.
            max_snr_db: Maximum SNR in dB for noise.
            noise_prob: Probability of adding noise.
            speed_perturb_enabled: Whether to enable speed perturbation.
            speed_rates: Speed perturbation rates (default: [0.9, 1.0, 1.1]).
            silence_enabled: Whether to enable silence insertion.
            silence_config: Config for SilenceAugmentation.
            sample_rate: Audio sample rate.
        """
        import torchaudio.transforms as T

        self.noise_enabled = noise_enabled
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.noise_prob = noise_prob
        self.speed_perturb_enabled = speed_perturb_enabled
        self.speed_rates = speed_rates or [0.9, 1.0, 1.1]
        self.silence_enabled = silence_enabled
        self.sample_rate = sample_rate
        self._prob_multiplier = 1.0

        # Pre-create resamplers for speed perturbation (avoid creating per sample)
        self._resamplers = {}
        if speed_perturb_enabled:
            for rate in self.speed_rates:
                if rate != 1.0:
                    new_freq = int(sample_rate * rate)
                    self._resamplers[rate] = (
                        T.Resample(sample_rate, new_freq),
                        T.Resample(new_freq, sample_rate),
                    )

        # Silence augmentation
        silence_config = silence_config or {}
        self.silence_aug = SilenceAugmentation(
            sample_rate=sample_rate, **silence_config
        )

        # Use audiomentations if available for better augmentations
        if AUDIOMENTATIONS_AVAILABLE and noise_enabled:
            self.noise_transform = am.AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.015,
                p=noise_prob,
            )
        else:
            self.noise_transform = None

    def set_prob_multiplier(self, multiplier: float) -> None:
        """Set the probability multiplier for all augmentations.

        Args:
            multiplier: Value between 0.0 and 1.0 to scale augmentation probability.
        """
        self._prob_multiplier = max(0.0, min(1.0, multiplier))
        self.silence_aug.set_prob_multiplier(multiplier)

    @property
    def effective_noise_prob(self) -> float:
        """Get the effective noise probability after applying multiplier."""
        return self.noise_prob * self._prob_multiplier

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentations.

        Args:
            waveform: Audio waveform (samples,) or (1, samples).

        Returns:
            Augmented waveform.
        """
        # Handle dimensions
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_channel = True
        else:
            squeeze_channel = False

        # Speed perturbation (always apply if enabled, uses multiplier for rate selection)
        if self.speed_perturb_enabled and self._prob_multiplier > 0:
            # When multiplier is low, prefer no speed change
            if random.random() < self._prob_multiplier:
                speed = random.choice(self.speed_rates)
            else:
                speed = 1.0
            if speed != 1.0:
                waveform = self._speed_perturb(waveform, speed)

        # Noise injection
        if self.noise_enabled:
            if self.noise_transform is not None and AUDIOMENTATIONS_AVAILABLE:
                # Use audiomentations with effective probability
                if random.random() < self.effective_noise_prob:
                    waveform_np = waveform.squeeze(0).numpy()
                    # Force apply since we already checked probability
                    transform = am.AddGaussianNoise(
                        min_amplitude=0.001, max_amplitude=0.015, p=1.0
                    )
                    waveform_np = transform(waveform_np, sample_rate=self.sample_rate)
                    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            elif random.random() < self.effective_noise_prob:
                # Fallback: add Gaussian noise
                waveform = self._add_gaussian_noise(waveform)

        # Silence insertion
        if self.silence_enabled:
            waveform = self.silence_aug(waveform)

        if squeeze_channel:
            waveform = waveform.squeeze(0)

        return waveform

    def _speed_perturb(self, waveform: torch.Tensor, speed: float) -> torch.Tensor:
        """Apply speed perturbation using resampling.

        Args:
            waveform: Audio waveform (1, samples).
            speed: Speed factor (>1 = faster, <1 = slower).

        Returns:
            Speed-perturbed waveform.
        """
        if speed == 1.0:
            return waveform

        # Use pre-created resamplers for efficiency
        resampler1, resampler2 = self._resamplers[speed]
        waveform = resampler1(waveform)
        waveform = resampler2(waveform)

        return waveform

    def _add_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR.

        Args:
            waveform: Audio waveform (1, samples).

        Returns:
            Noisy waveform.
        """
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)

        # Calculate signal power
        signal_power = waveform.pow(2).mean()

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)

        return waveform + noise


class AugmentationController:
    """Controller to manage augmentation probability across all augmenters.

    This class provides a centralized way to control augmentation probability,
    useful for implementing augmentation warmup during training.
    """

    def __init__(
        self,
        audio_aug: Optional[AudioAugmentation] = None,
        spec_aug: Optional[SpecAugmentation] = None,
    ):
        """Initialize AugmentationController.

        Args:
            audio_aug: AudioAugmentation instance to control.
            spec_aug: SpecAugmentation instance to control.
        """
        self.audio_aug = audio_aug
        self.spec_aug = spec_aug
        self._current_prob = 1.0

    def set_probability(self, prob: float) -> None:
        """Set the augmentation probability for all augmenters.

        Args:
            prob: Probability value between 0.0 and 1.0.
        """
        self._current_prob = max(0.0, min(1.0, prob))

        if self.audio_aug is not None:
            self.audio_aug.set_prob_multiplier(self._current_prob)

        if self.spec_aug is not None:
            self.spec_aug.set_prob_multiplier(self._current_prob)

    @property
    def current_probability(self) -> float:
        """Get the current augmentation probability."""
        return self._current_prob
