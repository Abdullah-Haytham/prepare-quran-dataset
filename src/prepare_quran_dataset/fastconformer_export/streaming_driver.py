"""Chunk raw audio through the exported graphs, carrying the encoder cache between steps.

Numpy only, apart from the runtime each backend imports when it is constructed.  The
chunking is a port of ``FastConformerMicStreamer.feed`` / ``flush`` in
``modeling_fastconformer_cache_aware/streaming_mic.py``, and has to stay identical to it:

- step 0 consumes mel frames ``[0, cs0)``, and step ``s >= 1`` the next ``cs1``, with the
  last ``pc1`` frames of the previous step re-fed in front of them;
- a step runs mid-stream only when strictly more frames are available than it consumes;
- at the end, the last partial chunk is zero-padded along the mel-frame axis and given
  its true length, and a tail shorter than ``sf1`` new frames is dropped.

Like the reference, this returns every logit frame of the last chunk without trimming
to ``encoded_len``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

MANIFEST_NAME = "manifest.json"

MEL_INPUTS = ("raw", "first_index", "num_valid")
MEL_OUTPUTS = ("log_mel",)
PRE_ENCODE_INPUTS = ("log_mel", "log_mel_length")
PRE_ENCODE_OUTPUTS = ("features", "features_length")
CORE_INPUTS = (
    "features",
    "features_length",
    "cache_last_channel",
    "cache_last_time",
    "cache_last_channel_length",
)
CORE_STATE_OUTPUTS = (
    "encoded_length",
    "next_cache_last_channel",
    "next_cache_last_time",
    "next_cache_last_channel_length",
)


@dataclass(frozen=True)
class StreamingGeometry:
    """Everything the driver needs to know about a model; stored in the manifest."""

    levels: tuple[str, ...]
    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    features: int
    num_raw_samples: int
    num_emphasized_samples: int
    first_chunk_frames: int  # cs0
    chunk_frames: int  # cs1
    pre_encode_cache_frames: int  # pc1
    sampling_frames: int  # sf1
    drop_extra_pre_encoded: int
    encoder_frames: int
    d_model: int
    n_layers: int
    last_channel_cache_size: int
    last_time_cache_size: int

    @property
    def half_n_fft_samples(self) -> int:
        """Minimum signal length: the STFT can't reflect-pad anything shorter."""
        return self.n_fft // 2

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "StreamingGeometry":
        return cls(**{**data, "levels": tuple(data["levels"])})


class StreamingBackend(Protocol):
    """Runs one named graph on named numpy inputs, returning outputs by name."""

    def run(self, graph: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ...


class StreamingDriver:
    """Feed samples in any chunk sizes, call :meth:`flush`, then read :meth:`logits`."""

    def __init__(self, backend: StreamingBackend, geometry: StreamingGeometry) -> None:
        self.backend = backend
        self.geometry = geometry
        self.reset()

    def reset(self) -> None:
        g = self.geometry
        self.samples = np.zeros(0, dtype=np.float32)
        self.step = 0
        self.finished = False
        self.previous_log_mel: np.ndarray | None = None
        self.cache_last_channel = np.zeros(
            (g.n_layers, 1, g.last_channel_cache_size, g.d_model), dtype=np.float32
        )
        self.cache_last_time = np.zeros(
            (g.n_layers, 1, g.d_model, g.last_time_cache_size), dtype=np.float32
        )
        self.cache_last_channel_length = np.zeros(1, dtype=np.int32)
        self.level_logits: dict[str, list[np.ndarray]] = {lvl: [] for lvl in g.levels}

    # ------------------------------------------------------------------
    # Frame bookkeeping (same formulas as StreamingMelFeaturizer)
    # ------------------------------------------------------------------

    def _chunk_end(self, step: int) -> int:
        return self.geometry.first_chunk_frames + self.geometry.chunk_frames * step

    def _available_frames(self) -> int:
        half = self.geometry.half_n_fft_samples
        if len(self.samples) < half:
            return 0
        return (len(self.samples) - half) // self.geometry.hop_length + 1

    def _total_frames(self) -> int:
        return len(self.samples) // self.geometry.hop_length + 1

    # ------------------------------------------------------------------
    # Graph calls
    # ------------------------------------------------------------------

    def _new_log_mel(self, step: int) -> np.ndarray:
        """The ``chunk_frames`` frames starting at the global frame of ``step``.

        Every step uses the same fixed-size graph.  On step 0 that window starts before
        the signal (at frame ``first_chunk_frames - chunk_frames``); the caller discards
        those frames, and ``first_index`` tells the graph where the signal starts.
        """
        g = self.geometry
        start_frame = g.first_chunk_frames + g.chunk_frames * (step - 1)
        start = start_frame * g.hop_length - g.win_length // 2

        # raw[r] = samples[start + r - 1], with zeros outside the signal.
        raw = np.zeros((1, g.num_raw_samples), dtype=np.float32)
        begin, end = max(start - 1, 0), min(start - 1 + g.num_raw_samples, len(self.samples))
        if end > begin:
            raw[0, begin - (start - 1) : end - (start - 1)] = self.samples[begin:end]

        num_valid = g.num_emphasized_samples
        if self.finished:
            num_valid = min(num_valid, len(self.samples) - start)
        outputs = self.backend.run(
            "mel",
            {
                "raw": raw,
                "first_index": np.array([max(-start, 0)], dtype=np.int32),
                "num_valid": np.array([num_valid], dtype=np.int32),
            },
        )
        return outputs["log_mel"]

    def _run_step(self, log_mel: np.ndarray, length: int, graph: str) -> None:
        g = self.geometry
        pre = self.backend.run(
            graph,
            {
                "log_mel": log_mel,
                "log_mel_length": np.array([length], dtype=np.int32),
            },
        )

        # The model enforces this with a runtime check that can't be part of the graph.
        unfilled = g.last_channel_cache_size - int(self.cache_last_channel_length[0])
        if unfilled % g.encoder_frames != 0:
            raise ValueError(
                f"Cache offset {unfilled} is not a multiple of the chunk size "
                f"{g.encoder_frames}."
            )

        core = self.backend.run(
            "encoder_core",
            {
                "features": pre["features"],
                "features_length": pre["features_length"],
                "cache_last_channel": self.cache_last_channel,
                "cache_last_time": self.cache_last_time,
                "cache_last_channel_length": self.cache_last_channel_length,
            },
        )
        self.cache_last_channel = core["next_cache_last_channel"]
        self.cache_last_time = core["next_cache_last_time"]
        self.cache_last_channel_length = core["next_cache_last_channel_length"]
        for level in g.levels:
            self.level_logits[level].append(core[level][0])
        self.step += 1

    def _assemble(self, step: int, num_frames: int | None) -> np.ndarray:
        """Log-mel input of ``step``; frames at or past ``num_frames`` are zeroed."""
        g = self.geometry
        new = self._new_log_mel(step)
        if step == 0:
            log_mel = new[:, :, g.chunk_frames - g.first_chunk_frames :]
            first_frame = 0
        else:
            log_mel = np.concatenate(
                [self.previous_log_mel[:, :, -g.pre_encode_cache_frames :], new], axis=2
            )
            first_frame = self._chunk_end(step - 1) - g.pre_encode_cache_frames
        if num_frames is not None and first_frame + log_mel.shape[2] > num_frames:
            log_mel = log_mel.copy()
            log_mel[:, :, max(num_frames - first_frame, 0) :] = 0.0
        self.previous_log_mel = log_mel
        return log_mel

    def _run_regular_step(self) -> None:
        g = self.geometry
        step = self.step
        log_mel = self._assemble(step, None)
        if step == 0:
            self._run_step(log_mel, g.first_chunk_frames, "pre_encode_first")
        else:
            self._run_step(
                log_mel, g.chunk_frames + g.pre_encode_cache_frames, "pre_encode_next"
            )

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    def feed(self, samples: np.ndarray) -> None:
        """Append 16 kHz mono samples and run every step they fully cover."""
        if self.finished:
            raise RuntimeError("The stream has finished; call reset() first.")
        self.samples = np.concatenate(
            [self.samples, np.asarray(samples, dtype=np.float32).reshape(-1)]
        )
        while self._available_frames() > self._chunk_end(self.step):
            self._run_regular_step()

    def flush(self) -> None:
        """End of stream: run the remaining chunks, zero-padding the last one."""
        if self.finished:
            return
        self.finished = True
        if len(self.samples) <= self.geometry.half_n_fft_samples:
            return

        g = self.geometry
        total = self._total_frames()
        while total > self._chunk_end(self.step):
            self._run_regular_step()

        step = self.step
        if step == 0:
            self._run_step(self._assemble(0, total), total, "pre_encode_first")
            return
        new_frames = total - self._chunk_end(step - 1)
        if new_frames >= g.sampling_frames:
            self._run_step(
                self._assemble(step, total),
                new_frames + g.pre_encode_cache_frames,
                "pre_encode_next",
            )

    def logits(self) -> dict[str, np.ndarray]:
        """Per-level logits of shape ``(T, vocab_size)``, concatenated over all steps."""
        result = {}
        for level, chunks in self.level_logits.items():
            if chunks:
                result[level] = np.concatenate(chunks, axis=0)
            else:
                result[level] = np.zeros((0, 0), dtype=np.float32)
        return result


def stream_waveform(
    backend: StreamingBackend,
    geometry: StreamingGeometry,
    waveform: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run a whole waveform; per-level logits of shape ``(T, vocab_size)``."""
    driver = StreamingDriver(backend, geometry)
    driver.feed(waveform)
    driver.flush()
    return driver.logits()


# ----------------------------------------------------------------------
# Artifact loading and runtime backends
# ----------------------------------------------------------------------


def load_manifest(artifact_dir: str | Path) -> dict:
    with open(Path(artifact_dir) / MANIFEST_NAME, encoding="utf-8") as f:
        return json.load(f)


def _bundle_paths(artifact_dir: str | Path, fmt: str, precision: str) -> dict[str, Path]:
    manifest = load_manifest(artifact_dir)
    try:
        bundle = manifest["bundles"][fmt][precision]
    except KeyError as e:
        raise ValueError(
            f"{artifact_dir} has no {fmt}/{precision} bundle: "
            f"{ {f: list(b) for f, b in manifest['bundles'].items()} }"
        ) from e
    return {graph: Path(artifact_dir) / rel for graph, rel in bundle.items()}


class OnnxBackend:
    """ONNX Runtime on CPU."""

    def __init__(
        self,
        artifact_dir: str | Path,
        precision: str,
        num_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if num_threads is not None:
            options.intra_op_num_threads = num_threads
        self.sessions = {
            graph: ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
            for graph, path in _bundle_paths(artifact_dir, "onnx", precision).items()
        }

    def run(self, graph: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        session = self.sessions[graph]
        names = [o.name for o in session.get_outputs()]
        return dict(zip(names, session.run(names, inputs)))


class LiteRtBackend:
    """LiteRT interpreter with the default XNNPACK CPU delegate."""

    def __init__(
        self,
        artifact_dir: str | Path,
        precision: str,
        num_threads: int | None = None,
    ) -> None:
        from ai_edge_litert.interpreter import Interpreter

        manifest = load_manifest(artifact_dir)
        self.output_names = manifest["graphs"]
        self.runners = {}
        for graph, path in _bundle_paths(artifact_dir, "litert", precision).items():
            interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
            self.runners[graph] = interpreter.get_signature_runner()

    def run(self, graph: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self.runners[graph](**inputs)
        # LiteRT names outputs output_0, output_1, ... in the graph's return order.
        names = self.output_names[graph]["outputs"]
        return {name: outputs[f"output_{i}"] for i, name in enumerate(names)}
