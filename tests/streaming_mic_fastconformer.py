"""True live-microphone streaming demo for the cache-aware FastConformer model.

Unlike ``infer_fastconformer_streaming`` (which computes the whole file's mel
spectrogram up-front and slices it — a *simulation* of streaming), this module
computes mel frames incrementally from a growing raw-audio buffer while staying
bit-identical to the whole-file computation, and drives the model's cache-aware
``streaming_step`` chunk by chunk as audio arrives from a browser microphone
through a Gradio app.

Why incremental mel is tricky (and how it is handled here):

- The STFT is centered (``torch.stft(center=True, pad_mode="reflect")``) with
  ``n_fft=512`` / hop 160, so mel frame ``f`` covers samples
  ``[f*160 - 256, f*160 + 256)``. A frame may only be emitted mid-stream once
  its last sample exists — no reflection padding against a fake "end".
- Preemphasis special-cases the first sample of whatever signal it is given.
- Therefore each chunk's frames are computed over a window that starts
  ``MARGIN_FRAMES`` frames early; the (provably at most 2) frames contaminated
  by the window's left edge fall inside the discarded margin, and the interior
  frames match the whole-file mel exactly.
- Dither is forced to 0 (NeMo's own streaming does the same); otherwise the
  featurizer injects noise and results are non-deterministic.

Run the correctness gate on a machine with nemo/torch installed:

    python tests/streaming_mic_fastconformer.py --self-test assets/audio-sampels/test_sample.mp3

Launch the mic app:

    python tests/streaming_mic_fastconformer.py
"""

from __future__ import annotations

import argparse
import json
import time

import librosa
import numpy as np
import soxr
import torch
from huggingface_hub import hf_hub_download

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTC,
    FastConformerMelProcessor,
    infer_fastconformer_streaming,
)

SAMPLE_RATE = 16000
HOP = 160  # window_stride 10 ms
HALF_NFFT = 256  # n_fft // 2, the centered-STFT padding on each side
# Frames within 2 frames of a window's left edge are contaminated (reflect pad
# + preemphasis first-sample special case reach ceil(257/160) = 2 frames in).
MARGIN_FRAMES = 8

BLANK_ID = 0  # [PAD], also the CTC blank
EOS_ID = 1
SPECIAL_IDS = (BLANK_ID, EOS_ID)

DEFAULT_MODEL_ID = "obadx/muaalem-fastconformer-base-v1"


def ctc_collapse(ids: list[int], blank_id: int = BLANK_ID) -> list[int]:
    """Greedy CTC: collapse consecutive repeats, drop blanks and special ids."""
    out: list[int] = []
    prev = blank_id
    for t in ids:
        if t == blank_id:
            prev = blank_id
            continue
        if t == prev:
            continue
        out.append(t)
        prev = t
    return [t for t in out if t not in SPECIAL_IDS]


class StreamingMelFeaturizer:
    """Incremental log-mel extraction, bit-identical to whole-file mel.

    Frames ``[a, b)`` are computed by running the (dither-free) NeMo processor
    on a sample window that starts ``MARGIN_FRAMES`` frames before ``a`` and
    ends exactly at the last sample frame ``b-1`` needs, then slicing the
    margin off. Mid-stream, a frame is "available" only once sample
    ``f*160 + 255`` exists; at flush time the true signal end supplies the same
    right reflection padding the whole-file computation sees.
    """

    def __init__(self, processor_kwargs: dict, device: str | torch.device):
        self.device = torch.device(device)
        self.processor = FastConformerMelProcessor(
            **{**processor_kwargs, "dither": 0.0, "pad_to": 0}
        )
        self.processor.to(self.device)
        self.buf = np.zeros(0, dtype=np.float32)

    def append(self, samples: np.ndarray) -> None:
        self.buf = np.concatenate(
            [self.buf, np.asarray(samples, dtype=np.float32).reshape(-1)]
        )

    def available_frames(self) -> int:
        """Frames emittable now without any right-edge padding."""
        if len(self.buf) < HALF_NFFT:
            return 0
        return (len(self.buf) - HALF_NFFT) // HOP + 1

    def total_frames_final(self) -> int:
        """Whole-file frame count once the stream has ended (len//160 + 1)."""
        return len(self.buf) // HOP + 1

    def get_frames(self, a: int, b: int, final: bool = False) -> torch.Tensor:
        """Return mel frames [a, b) as (1, n_mels, b - a)."""
        w0_frame = max(0, a - MARGIN_FRAMES)
        w0 = w0_frame * HOP
        end = len(self.buf) if final else min(len(self.buf), (b - 1) * HOP + HALF_NFFT)
        window = torch.from_numpy(self.buf[w0:end]).unsqueeze(0).to(self.device)
        length = torch.tensor([end - w0], device=self.device)
        mel, _ = self.processor(input_signal=window, length=length)
        return mel[:, :, a - w0_frame : b - w0_frame]

    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)


class FastConformerMicStreamer:
    """Feeds arbitrary-sized 16 kHz sample chunks to the cache-aware model.

    Replicates the exact chunking of ``infer_fastconformer_streaming`` /
    NeMo's ``CacheAwareStreamingAudioBuffer``:

    - step 0 consumes mel frames ``[0, cs0)`` with ``drop_extra_pre_encoded=0``
    - step s>=1 consumes ``[cs0 + cs1*(s-1) - pc1, cs0 + cs1*s)`` (``pc1``
      re-fed overlap frames) with the config's ``drop_extra_pre_encoded``
    - a step runs mid-stream only when *strictly more* frames are available
      than it consumes: a stream ending exactly on a chunk boundary must run
      that chunk with ``keep_all_outputs=True``, which only ``flush`` can know
    - at flush, the final partial chunk is zero-padded on the mel-frame axis to
      the expected count, with the true valid length; a leftover of fewer than
      ``sampling_frames[1]`` new frames is dropped entirely (NeMo parity)
    """

    def __init__(
        self,
        model: FastConformerCacheAwareMultilevelCTC,
        vocab: dict[str, dict[str, int]],
        device: str | torch.device,
    ):
        self.model = model.eval()
        self.device = torch.device(device)

        model.setup_streaming_params()
        cfg = model.encoder.streaming_cfg
        cs = cfg.chunk_size
        self.cs0, self.cs1 = (cs[0], cs[1]) if isinstance(cs, (list, tuple)) else (cs, cs)
        pc = cfg.pre_encode_cache_size
        self.pc1 = pc[1] if isinstance(pc, (list, tuple)) else pc
        self.drop = cfg.drop_extra_pre_encoded
        if hasattr(model.encoder, "pre_encode") and hasattr(
            model.encoder.pre_encode, "get_sampling_frames"
        ):
            self.sf1 = model.encoder.pre_encode.get_sampling_frames()[1]
        else:
            self.sf1 = 1

        self.vocab = vocab
        self.id_to_token = {
            level: {idx: tok for tok, idx in level_vocab.items()}
            for level, level_vocab in vocab.items()
        }
        self.featurizer = StreamingMelFeaturizer(
            model.config.processor_kwargs, self.device
        )
        self.reset()

    def reset(self) -> None:
        self.cache = self.model.get_initial_cache(batch_size=1)
        self.step = 0
        self.level_ids: dict[str, list[int]] = {
            level: [] for level in self.model.level_to_lm_head
        }
        self.chunk_log: list[dict] = []
        self.featurizer.reset()
        self.finished = False

    def chunk_end(self, s: int) -> int:
        """Global mel-frame index one past step ``s``'s new frames."""
        return self.cs0 + self.cs1 * s

    @torch.no_grad()
    def _run_step(
        self, mel: torch.Tensor, length: int, keep_all: bool, drop: int
    ) -> None:
        t0 = time.perf_counter()
        out = self.model.streaming_step(
            processed_signal=mel,
            processed_length=torch.tensor([length], device=self.device),
            cache=self.cache,
            keep_all_outputs=keep_all,
            drop_extra_pre_encoded=drop,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.cache = out.cache

        new_ids: dict[str, list[int]] = {}
        for level, logits in out.logits.items():
            ids = logits[0].argmax(dim=-1).tolist()
            self.level_ids[level].extend(ids)
            new_ids[level] = ids

        ph_level = "phonemes" if "phonemes" in new_ids else next(iter(new_ids))
        ph_tokens = [
            self.id_to_token[ph_level].get(t, "?")
            for t in ctc_collapse(new_ids[ph_level])
        ]
        self.chunk_log.append(
            {
                "step": self.step,
                "frames_in": int(length),
                "frames_out": len(new_ids[ph_level]),
                "latency_ms": latency_ms,
                "phonemes": "".join(ph_tokens),
            }
        )
        self.step += 1

    def _run_regular_step(self, keep_all: bool, final: bool) -> None:
        s = self.step
        if s == 0:
            mel = self.featurizer.get_frames(0, self.cs0, final=final)
            self._run_step(mel, self.cs0, keep_all=keep_all, drop=0)
        else:
            a = self.chunk_end(s - 1) - self.pc1
            b = self.chunk_end(s)
            mel = self.featurizer.get_frames(a, b, final=final)
            self._run_step(mel, b - a, keep_all=keep_all, drop=self.drop)

    def feed(self, samples_16k: np.ndarray) -> None:
        """Append raw 16 kHz samples and run every step that is fully covered."""
        if self.finished:
            raise RuntimeError("Streamer is finished; call reset() first.")
        self.featurizer.append(samples_16k)
        while self.featurizer.available_frames() > self.chunk_end(self.step):
            self._run_regular_step(keep_all=False, final=False)

    def flush(self) -> None:
        """End of stream: run remaining chunks, final one with keep_all_outputs."""
        if self.finished:
            return
        self.finished = True
        # Too short for even one reflect-padded STFT window: nothing to emit
        # (the whole-file path would fail on such a signal as well).
        if len(self.featurizer.buf) <= HALF_NFFT:
            return

        T = self.featurizer.total_frames_final()
        while T > self.chunk_end(self.step):
            self._run_regular_step(keep_all=False, final=True)

        s = self.step
        if s == 0:
            mel = self.featurizer.get_frames(0, T, final=True)
            mel = torch.nn.functional.pad(mel, (0, self.cs0 - mel.size(-1)))
            self._run_step(mel, T, keep_all=True, drop=0)
        else:
            new = T - self.chunk_end(s - 1)
            if new >= self.sf1:
                a = self.chunk_end(s - 1) - self.pc1
                mel = self.featurizer.get_frames(a, T, final=True)
                expected = self.cs1 + self.pc1
                mel = torch.nn.functional.pad(mel, (0, expected - mel.size(-1)))
                self._run_step(mel, new + self.pc1, keep_all=True, drop=self.drop)
            # else 1 <= new < sf1: NeMo's iterator silently drops this tail.

    # ------------------------------------------------------------------
    # Decoding / rendering
    # ------------------------------------------------------------------

    def decode_ids(self, level: str) -> list[int]:
        return ctc_collapse(self.level_ids[level])

    def decode_tokens(self, level: str) -> list[str]:
        return [self.id_to_token[level].get(t, "?") for t in self.decode_ids(level)]

    def transcript(self) -> str:
        level = "phonemes" if "phonemes" in self.level_ids else next(iter(self.level_ids))
        return "".join(self.decode_tokens(level))

    def sifat_summary(self) -> str:
        """Last decoded token of every non-phoneme level, one line each."""
        lines = []
        for level in self.level_ids:
            if level == "phonemes":
                continue
            tokens = self.decode_tokens(level)
            lines.append(f"{level}: {tokens[-1] if tokens else '-'}")
        return "\n".join(lines)

    def seconds_fed(self) -> float:
        return len(self.featurizer.buf) / SAMPLE_RATE


# ----------------------------------------------------------------------
# Self test: true streaming vs the trusted file-based simulation
# ----------------------------------------------------------------------


def self_test(
    audio_path: str,
    model: FastConformerCacheAwareMultilevelCTC,
    device: str | torch.device,
    vocab: dict[str, dict[str, int]] | None = None,
    seed: int = 0,
) -> bool:
    """Compare the mic-streaming path against ``infer_fastconformer_streaming``.

    For base-v1 the streaming output is exactly identical to offline, so every
    per-level frame-argmax id must match exactly. Argmax ids (not logits) are
    compared because GPU kernel selection can shift logits by ~1e-6.
    """
    device = torch.device(device)
    model = model.eval()
    rng = np.random.default_rng(seed)
    wav, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    print(f"audio: {audio_path} ({len(wav) / SAMPLE_RATE:.2f}s)")

    # --- 1) mel exactness: incremental windowed frames vs whole-file mel ---
    proc_kwargs = model.config.processor_kwargs
    feat = StreamingMelFeaturizer(proc_kwargs, device)
    feat.append(wav)
    whole, _ = feat.processor(
        input_signal=torch.from_numpy(wav).unsqueeze(0).to(device),
        length=torch.tensor([len(wav)], device=device),
    )
    n_avail = feat.available_frames()
    max_diff, a = 0.0, 0
    while a < n_avail:
        b = min(n_avail, a + int(rng.integers(1, 200)))
        got = feat.get_frames(a, b)
        max_diff = max(max_diff, (got - whole[:, :, a:b]).abs().max().item())
        a = b
    T = feat.total_frames_final()
    if T > n_avail:  # tail frames that need the right reflection padding
        got = feat.get_frames(n_avail, T, final=True)
        max_diff = max(max_diff, (got - whole[:, :, n_avail:T]).abs().max().item())
    mel_ok = max_diff < 1e-5
    print(f"mel exactness: max |diff| = {max_diff:.3e} -> {'OK' if mel_ok else 'FAIL'}")

    # --- 2) reference: file-based streaming simulation (dither = 0) ---
    ref_processor = FastConformerMelProcessor(
        **{**proc_kwargs, "dither": 0.0, "pad_to": 0}
    )
    ref_processor.to(device)
    ref_logits = infer_fastconformer_streaming(
        [audio_path], device, torch.float32, model, ref_processor
    )
    ref_ids = {lvl: t[0].argmax(dim=-1).tolist() for lvl, t in ref_logits.items()}

    # --- 3) true streaming: random-sized chunks + flush ---
    if vocab is None:
        vocab = {
            lvl: {str(i): i for i in range(head.out_features)}
            for lvl, head in model.level_to_lm_head.items()
        }
    streamer = FastConformerMicStreamer(model, vocab, device)
    pos = 0
    while pos < len(wav):
        n = int(rng.integers(160, 8000))
        streamer.feed(wav[pos : pos + n])
        pos += n
    streamer.flush()

    # --- 4) compare per level ---
    all_ok = mel_ok
    for level, ref in ref_ids.items():
        got = streamer.level_ids[level]
        if len(got) != len(ref):
            print(f"{level}: FAIL length mismatch (stream {len(got)} vs ref {len(ref)})")
            all_ok = False
            continue
        n_match = sum(g == r for g, r in zip(got, ref))
        ratio = n_match / max(len(ref), 1)
        ok = n_match == len(ref)
        all_ok = all_ok and ok
        print(f"{level}: {n_match}/{len(ref)} frames match ({ratio:.4f}) -> "
              f"{'OK' if ok else 'FAIL'}")

    ph_level = "phonemes" if "phonemes" in ref_ids else next(iter(ref_ids))
    id_to_tok = streamer.id_to_token[ph_level]
    ref_text = "".join(id_to_tok.get(t, "?") for t in ctc_collapse(ref_ids[ph_level]))
    print(f"reference  phonemes: {ref_text}")
    print(f"streaming  phonemes: {streamer.transcript()}")
    print("SELF TEST PASSED" if all_ok else "SELF TEST FAILED")
    return all_ok


# ----------------------------------------------------------------------
# Gradio microphone app
# ----------------------------------------------------------------------


def build_demo(
    model: FastConformerCacheAwareMultilevelCTC,
    vocab: dict[str, dict[str, int]],
    device: str | torch.device,
):
    import gradio as gr

    state = {
        "streamer": FastConformerMicStreamer(model, vocab, device),
        "resampler": None,
        "native_sr": None,
    }

    def _to_float_mono(y: np.ndarray) -> np.ndarray:
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        else:
            y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y

    def _resample(sr: int, y: np.ndarray, last: bool = False) -> np.ndarray:
        if sr == SAMPLE_RATE and not last:
            return y
        if state["resampler"] is None or state["native_sr"] != sr:
            state["resampler"] = soxr.ResampleStream(
                sr, SAMPLE_RATE, 1, dtype="float32"
            )
            state["native_sr"] = sr
        return state["resampler"].resample_chunk(y, last=last)

    def _render():
        st = state["streamer"]
        transcript = st.transcript()
        sifat = st.sifat_summary()
        log_lines = [
            f"step {e['step']:03d} | in {e['frames_in']:3d} mel | "
            f"out {e['frames_out']:3d} | {e['latency_ms']:6.1f} ms | {e['phonemes']}"
            for e in st.chunk_log[-40:]
        ]
        avg_lat = (
            float(np.mean([e["latency_ms"] for e in st.chunk_log]))
            if st.chunk_log
            else 0.0
        )
        summary = (
            f"steps: {st.step} | audio fed: {st.seconds_fed():.1f}s | "
            f"avg step latency: {avg_lat:.0f} ms"
            + (" | FLUSHED" if st.finished else "")
        )
        return transcript, sifat, "\n".join(log_lines), summary

    def on_start():
        state["streamer"].reset()
        state["resampler"] = None
        state["native_sr"] = None
        return "", "", "", "recording..."

    def on_stream(audio_chunk):
        if audio_chunk is None:
            return _render()
        sr, y = audio_chunk
        y16k = _resample(sr, _to_float_mono(y))
        if len(y16k) and not state["streamer"].finished:
            state["streamer"].feed(y16k)
        return _render()

    def on_stop():
        st = state["streamer"]
        if not st.finished:
            if state["resampler"] is not None:
                tail = _resample(
                    state["native_sr"], np.zeros(0, dtype=np.float32), last=True
                )
                if len(tail):
                    st.feed(tail)
            st.flush()
        return _render()

    def on_reset():
        state["streamer"].reset()
        state["resampler"] = None
        state["native_sr"] = None
        return "", "", "", "reset OK"

    with gr.Blocks(title="Muaalem FastConformer live streaming") as demo:
        gr.Markdown(
            "## Muaalem FastConformer — live cache-aware streaming\n"
            "Speak into the mic; phonemes appear as chunks (~0.5 s of new audio "
            "each, ~0.5 s model lookahead) are processed. Stopping the "
            "recording flushes the final chunk."
        )
        audio_input = gr.Audio(
            sources=["microphone"], streaming=True, type="numpy", label="Microphone"
        )
        summary_box = gr.Textbox(label="Status", lines=1)
        transcript_box = gr.Textbox(label="Phonetic transcript", lines=4, rtl=True)
        sifat_box = gr.Textbox(label="Sifat (latest)", lines=10, rtl=True)
        log_box = gr.Textbox(label="Per-chunk log (last 40)", lines=15)
        reset_btn = gr.Button("Reset")

        outputs = [transcript_box, sifat_box, log_box, summary_box]
        audio_input.start_recording(fn=on_start, outputs=outputs)
        audio_input.stream(
            fn=on_stream, inputs=[audio_input], outputs=outputs, stream_every=0.5
        )
        audio_input.stop_recording(fn=on_stop, outputs=outputs)
        reset_btn.click(fn=on_reset, outputs=outputs)

    return demo


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def load_model_and_vocab(
    model_id: str, device: str | torch.device, token: str | None = None
):
    model = FastConformerCacheAwareMultilevelCTC.from_pretrained(model_id, token=token)
    model.to(device)
    model.eval()
    model.processor.to(device)  # processor is not an nn.Module; model.to() skips it
    model.setup_streaming_params()
    vocab_path = hf_hub_download(model_id, "vocab.json", token=token)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    return model, vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--self-test",
        metavar="AUDIO_PATH",
        default=None,
        help="Run the streaming-vs-simulation self test on a file and exit.",
    )
    parser.add_argument("--no-share", action="store_true")
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print(f"loading {args.model} on {device} ...")
    model, vocab = load_model_and_vocab(args.model, device)

    if args.self_test is not None:
        ok = self_test(args.self_test, model, device, vocab=vocab)
        raise SystemExit(0 if ok else 1)

    demo = build_demo(model, vocab, device)
    demo.launch(server_name="0.0.0.0", share=not args.no_share)


if __name__ == "__main__":
    main()
