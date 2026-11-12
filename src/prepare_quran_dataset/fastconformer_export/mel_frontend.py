"""Static-shape log-mel graph that reproduces NeMo 2.1.0's ``FilterbankFeatures``.

This module has no NeMo import, so it can live in an exported graph and can be tested
without the NeMo stack.  What it has to match, for the configuration this project uses
(``normalize="NA"``, ``frame_splicing=1``, ``pad_to=0``, no dither)::

    y[0] = x[0];  y[n] = x[n] - 0.97 * x[n - 1]
    torch.stft(y, 512, hop_length=160, win_length=400, center=True, pad_mode="reflect",
               window=torch.hann_window(400, periodic=False))
    log(fb @ |X|^2 + 2**-24)

NeMo versions after 2.1.0 change this: the padding becomes zeros and the frame count
loses its ``+ 1``.  ``tests/test_fastconformer_export_mel.py`` compares against whichever
NeMo is installed, so an upgrade makes that test fail rather than silently changing the
features the model sees.

Each call is a pure function of a fixed-size raw window; see :class:`StreamingLogMel`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class MelGeometry:
    """Sizes shared by the mel graph and the streaming driver."""

    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    features: int = 80
    preemph: float = 0.97
    log_zero_guard: float = 2**-24
    frames_per_call: int = 52

    @classmethod
    def from_processor_kwargs(
        cls, processor_kwargs: dict[str, Any], frames_per_call: int
    ) -> "MelGeometry":
        """Build from ``config.processor_kwargs``, rejecting what the graph can't do."""
        unsupported = {
            "normalize": ("NA", processor_kwargs.get("normalize", "NA")),
            "window": ("hann", processor_kwargs.get("window", "hann")),
            "frame_splicing": (1, processor_kwargs.get("frame_splicing", 1)),
        }
        for name, (expected, got) in unsupported.items():
            if got != expected:
                raise ValueError(
                    f"The mel graph only supports {name}={expected!r}, got {got!r}."
                )
        sample_rate = processor_kwargs.get("sample_rate", 16000)
        return cls(
            sample_rate=sample_rate,
            n_fft=processor_kwargs.get("n_fft", 512),
            # Same conversion as NeMo's AudioToMelSpectrogramPreprocessor.
            win_length=int(processor_kwargs.get("window_size", 0.025) * sample_rate),
            hop_length=int(processor_kwargs.get("window_stride", 0.01) * sample_rate),
            features=processor_kwargs.get("features", 80),
            frames_per_call=frames_per_call,
        )

    @property
    def num_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def num_emphasized_samples(self) -> int:
        """Pre-emphasized samples covering ``frames_per_call`` windows."""
        return self.win_length + (self.frames_per_call - 1) * self.hop_length

    @property
    def num_raw_samples(self) -> int:
        """Raw samples per call: one extra, as the pre-emphasis needs a previous sample."""
        return self.num_emphasized_samples + 1


def mel_filterbank(geometry: MelGeometry) -> torch.Tensor:
    """Slaney filterbank of shape ``(features, num_bins)``, built with NeMo's exact call."""
    import librosa

    weights = librosa.filters.mel(
        sr=geometry.sample_rate,
        n_fft=geometry.n_fft,
        n_mels=geometry.features,
        fmin=0,
        fmax=geometry.sample_rate / 2,
        norm="slaney",
    )
    return torch.tensor(weights, dtype=torch.float32)


class _LogMelBase(nn.Module):
    """Pre-emphasis, reflect edges and the mel/log stage, around a pluggable STFT."""

    def __init__(self, geometry: MelGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.preemph = geometry.preemph
        self.log_zero_guard = geometry.log_zero_guard
        self.last_index = geometry.num_emphasized_samples - 1
        self.register_buffer(
            "positions",
            torch.arange(geometry.num_emphasized_samples, dtype=torch.int32),
            persistent=False,
        )
        # A FULLY_CONNECTED with a constant weight delegates better than a BATCH_MATMUL.
        self.filterbank = nn.Linear(geometry.num_bins, geometry.features, bias=False)
        self.filterbank.weight.data = mel_filterbank(geometry)

    def emphasized_with_reflect_edges(
        self,
        raw: torch.Tensor,
        first_index: torch.Tensor,
        num_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-emphasize, then mirror around the true signal edges.

        ``max(i, 2f - i)`` mirrors indices left of the signal's first sample ``f``, and
        ``min(i, 2e - i)`` mirrors those right of its last sample ``e``.  Away from the
        edges both are the identity, so the same graph serves every streaming step.
        The clamp only affects frames the driver discards: valid frames of a signal
        longer than ``n_fft // 2`` samples (NeMo's own minimum) never reflect twice.
        """
        emphasized = raw[:, 1:] - self.preemph * raw[:, :-1]
        indices = torch.maximum(self.positions, 2 * first_index - self.positions)
        last_valid = num_valid - 1
        indices = torch.minimum(indices, 2 * last_valid - indices)
        indices = torch.clamp(indices, min=0, max=self.last_index)
        return torch.index_select(emphasized, 1, indices)

    def log_mel(self, power: torch.Tensor) -> torch.Tensor:
        """``(B, num_bins, T)`` power spectrum to ``(B, features, T)`` log-mel."""
        mel = self.filterbank(power.transpose(1, 2))
        return torch.log(mel + self.log_zero_guard).transpose(1, 2)


class StreamingLogMel(_LogMelBase):
    """Raw PCM to log-mel with an STFT built from one plain convolution.

    The convolution holds the Hann window multiplied by cos/sin at each FFT bin, so the
    graph needs no FFT op and no complex tensors, and runs on any runtime.  Dropping the
    window's zero padding inside ``n_fft`` only changes the phase, not the magnitude, so
    frame ``f`` depends on just ``y[f * hop - win // 2 : f * hop + win // 2]``.

    Args:
        geometry: Sizes of the fixed input and output.

    Inputs:
        raw: Samples, ``(1, geometry.num_raw_samples)`` float32.
        first_index: ``(1,)`` int32.  Index into the pre-emphasized window of the
            signal's first sample, with zeros before it in ``raw``, which reproduces
            ``y[0] = x[0]``.  ``0`` once the signal start has passed.
        num_valid: ``(1,)`` int32.  Pre-emphasized samples before the signal ends, or
            the full window size while it hasn't ended.

    Returns:
        Log-mel of shape ``(1, features, frames_per_call)``.
    """

    def __init__(self, geometry: MelGeometry) -> None:
        super().__init__(geometry)
        win_length = geometry.win_length
        samples = torch.arange(win_length, dtype=torch.float64)
        bins = torch.arange(geometry.num_bins, dtype=torch.float64).unsqueeze(1)
        phase = 2 * math.pi * bins * samples / geometry.n_fft
        window = torch.hann_window(win_length, periodic=False, dtype=torch.float64)

        self.stft = nn.Conv1d(
            in_channels=1,
            out_channels=2 * geometry.num_bins,
            kernel_size=win_length,
            stride=geometry.hop_length,
            bias=False,
        )
        kernels = torch.cat([window * torch.cos(phase), -window * torch.sin(phase)])
        self.stft.weight.data = kernels.unsqueeze(1).to(torch.float32)

    def forward(
        self,
        raw: torch.Tensor,
        first_index: torch.Tensor,
        num_valid: torch.Tensor,
    ) -> torch.Tensor:
        emphasized = self.emphasized_with_reflect_edges(raw, first_index, num_valid)
        spectrum = self.stft(emphasized.unsqueeze(1))
        num_bins = self.geometry.num_bins
        real, imag = spectrum[:, :num_bins], spectrum[:, num_bins:]
        return self.log_mel(real * real + imag * imag)


class DirectStftLogMel(_LogMelBase):
    """The same graph using ``torch.stft``.

    Kept so the exporter can try the FFT route first and record what happens.  Same
    inputs, outputs and framing as :class:`StreamingLogMel`.
    """

    def __init__(self, geometry: MelGeometry) -> None:
        super().__init__(geometry)
        self.register_buffer(
            "window",
            torch.hann_window(geometry.win_length, periodic=False),
            persistent=False,
        )
        self.zero_padding = (geometry.n_fft - geometry.win_length) // 2

    def forward(
        self,
        raw: torch.Tensor,
        first_index: torch.Tensor,
        num_valid: torch.Tensor,
    ) -> torch.Tensor:
        emphasized = self.emphasized_with_reflect_edges(raw, first_index, num_valid)
        padded = nn.functional.pad(emphasized, (self.zero_padding, self.zero_padding))
        spectrum = torch.stft(
            padded,
            n_fft=self.geometry.n_fft,
            hop_length=self.geometry.hop_length,
            win_length=self.geometry.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        return self.log_mel(torch.view_as_real(spectrum).pow(2).sum(-1))
