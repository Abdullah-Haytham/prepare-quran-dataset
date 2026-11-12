"""The exported mel graph, driven by the real chunking, vs NeMo's whole-file features.

Plan gate 1 for the PyTorch module.  ``test_fastconformer_export_graphs.py`` repeats this
through the converted LiteRT and ONNX files.

The tolerance is ``MAX_LOG_MEL_ERROR`` rather than exact equality: a convolution and an FFT
round float32 differently, which only matters at the bottom of the log range.

Run with: ``uv run pytest tests/test_fastconformer_export_mel.py -v``
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")
pytest.importorskip("nemo")

from prepare_quran_dataset.fastconformer_export.mel_frontend import (  # noqa: E402
    DirectStftLogMel,
    MelGeometry,
    StreamingLogMel,
)
from prepare_quran_dataset.fastconformer_export.streaming_driver import (  # noqa: E402
    StreamingDriver,
    StreamingGeometry,
)
from prepare_quran_dataset.modeling_fastconformer_cache_aware.processor import (  # noqa: E402
    FastConformerMelProcessor,
)

MAX_LOG_MEL_ERROR = 1e-3
SAMPLE_RATE = 16000

# Lengths around every boundary of the framing and chunking.  With hop 160, L % 160 decides
# how the last frames reflect, and 49 frames (7680 samples) is the first chunk.
LENGTHS = (
    257,  # shortest signal NeMo accepts
    400,
    5000,  # ends inside the first chunk
    7679,
    7680,
    7681,
    7936,
    16000,
    48000,  # L % 160 == 0
    48037,  # L % 160 == 37
    48159,  # L % 160 == 159
)
SIGNALS = ("harmonic", "noise", "quiet", "silence_with_burst", "clipped")


def make_signal(kind: str, length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length) / SAMPLE_RATE
    if kind == "noise":
        signal = rng.normal(0, 0.1, length)
    elif kind == "harmonic":
        vibrato = 1 + 0.02 * np.sin(3 * t)
        signal = sum(
            0.2 / h * np.sin(2 * np.pi * 140 * h * t * vibrato) for h in range(1, 25)
        ) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    elif kind == "quiet":
        signal = rng.normal(0, 3e-4, length)
    elif kind == "silence_with_burst":
        signal = np.zeros(length)
        burst = min(4000, length // 3)
        signal[length // 3 : length // 3 + burst] = rng.normal(0, 0.3, burst)
    elif kind == "clipped":
        signal = np.clip(rng.normal(0, 2, length), -1, 1)
    else:
        raise ValueError(kind)
    return signal.astype(np.float32)


@pytest.fixture(scope="module")
def processor() -> FastConformerMelProcessor:
    return FastConformerMelProcessor(dither=0.0, pad_to=0)


@pytest.fixture(scope="module")
def geometry() -> StreamingGeometry:
    mel = MelGeometry(frames_per_call=52)
    return StreamingGeometry(
        levels=(),
        sample_rate=mel.sample_rate,
        n_fft=mel.n_fft,
        hop_length=mel.hop_length,
        win_length=mel.win_length,
        features=mel.features,
        num_raw_samples=mel.num_raw_samples,
        num_emphasized_samples=mel.num_emphasized_samples,
        first_chunk_frames=49,
        chunk_frames=52,
        pre_encode_cache_frames=5,
        sampling_frames=4,
        drop_extra_pre_encoded=2,
        encoder_frames=13,
        d_model=4,
        n_layers=1,
        last_channel_cache_size=78,
        last_time_cache_size=8,
    )


class RecordingBackend:
    """Runs the mel graph for real, and records what the pre-encode graphs would receive."""

    def __init__(self, geometry: StreamingGeometry, mel_module: torch.nn.Module) -> None:
        self.geometry = geometry
        self.mel_module = mel_module.eval()
        self.calls: list[tuple[str, np.ndarray, int]] = []

    @torch.no_grad()
    def run(self, graph, inputs):
        g = self.geometry
        if graph == "mel":
            args = {k: torch.from_numpy(v) for k, v in inputs.items()}
            return {"log_mel": self.mel_module(**args).numpy()}
        if graph.startswith("pre_encode"):
            self.calls.append(
                (graph, inputs["log_mel"], int(inputs["log_mel_length"][0]))
            )
            return {
                "features": np.zeros((1, g.d_model, g.encoder_frames), np.float32),
                "features_length": np.array([g.encoder_frames], np.int32),
            }
        length = inputs["cache_last_channel_length"] + g.encoder_frames
        return {
            "next_cache_last_channel": inputs["cache_last_channel"],
            "next_cache_last_time": inputs["cache_last_time"],
            "next_cache_last_channel_length": np.minimum(
                length, g.last_channel_cache_size
            ),
        }


def stream_frames(geometry, mel_module, wave, chunk_sizes=None):
    """Run the driver; returns the stitched frames and the raw per-step calls."""
    backend = RecordingBackend(geometry, mel_module)
    driver = StreamingDriver(backend, geometry)
    if chunk_sizes is None:
        driver.feed(wave)
    else:
        position = 0
        for size in chunk_sizes:
            driver.feed(wave[position : position + size])
            position += size
        driver.feed(wave[position:])
    driver.flush()

    stitched = []
    for index, (_, log_mel, length) in enumerate(backend.calls):
        overlap = 0 if index == 0 else geometry.pre_encode_cache_frames
        stitched.append(log_mel[0, :, overlap:length])
    return np.concatenate(stitched, axis=1), backend.calls


def whole_file_mel(processor, wave):
    signal = torch.from_numpy(wave).unsqueeze(0)
    mel, length = processor(input_signal=signal, length=torch.tensor([len(wave)]))
    assert mel.shape[2] == length.item()
    return mel[0].numpy()


@pytest.mark.parametrize("kind", SIGNALS)
@pytest.mark.parametrize("length", LENGTHS)
def test_streamed_frames_match_whole_file(processor, geometry, kind, length):
    wave = make_signal(kind, length)
    reference = whole_file_mel(processor, wave)
    frames, _ = stream_frames(geometry, StreamingLogMel(MelGeometry()), wave)

    # The driver drops a tail shorter than sampling_frames, as NeMo's iterator does, and
    # only that tail: anything else missing or extra is a chunking bug.
    missing = reference.shape[1] - frames.shape[1]
    assert 0 <= missing < geometry.sampling_frames, (reference.shape, frames.shape)

    error = np.abs(reference[:, : frames.shape[1]] - frames).max()
    assert error <= MAX_LOG_MEL_ERROR, f"max abs log-mel error {error:.2e}"


def test_chunk_sizes_do_not_change_the_result(geometry):
    """Mic-style arbitrary chunks must give what one big feed gives, to the bit."""
    wave = make_signal("harmonic", 48037)
    module = StreamingLogMel(MelGeometry())
    whole, _ = stream_frames(geometry, module, wave)
    rng = np.random.default_rng(1)
    chunked, _ = stream_frames(
        geometry, module, wave, chunk_sizes=rng.integers(160, 8000, size=8).tolist()
    )
    np.testing.assert_array_equal(whole, chunked)


def test_last_chunk_is_zero_padded_to_full_width(geometry):
    """Short tails must reach the encoder full-width, zero-padded, with their true length."""
    length = 7680 + 160 * 10  # first chunk, then 10 more frames
    wave = make_signal("noise", length)
    _, calls = stream_frames(geometry, StreamingLogMel(MelGeometry()), wave)
    graph, log_mel, valid = calls[-1]
    assert graph == "pre_encode_next"
    assert log_mel.shape[2] == 57
    assert valid < 57
    assert (log_mel[0, :, valid:] == 0).all()
    assert (log_mel[0, :, :valid] != 0).any()


@pytest.mark.parametrize("kind", SIGNALS)
def test_direct_stft_module_agrees_with_conv_stft(kind):
    """Attempt A in the exporter only tells us something if both modules compute the same."""
    mel = MelGeometry()
    raw = torch.from_numpy(make_signal(kind, mel.num_raw_samples)).unsqueeze(0)
    args = {
        "raw": raw,
        "first_index": torch.tensor([0], dtype=torch.int32),
        "num_valid": torch.tensor([mel.num_emphasized_samples], dtype=torch.int32),
    }
    with torch.no_grad():
        conv = StreamingLogMel(mel).eval()(**args)
        direct = DirectStftLogMel(mel).eval()(**args)
    assert (conv - direct).abs().max().item() <= MAX_LOG_MEL_ERROR
