# FastConformer export to LiteRT and ONNX

Converts the cache-aware FastConformer and its log-mel front-end into graphs that go from raw 16 kHz PCM to
per-level CTC logits, with no PyTorch or NeMo at runtime.

**Status:** the pipeline works end to end on random weights at full size. **Accuracy on the real checkpoint has not
been measured yet** (see [Still to run](#still-to-run)). On-device benchmarking comes later.

## Graphs

One streaming step is 13 encoder frames = 52 mel frames = **520 ms** of audio.

| Graph | Inputs | Outputs | Precision |
|---|---|---|---|
| `mel` | `raw` `[1,8561]` f32, `first_index` `[1]` i32, `num_valid` `[1]` i32 | `log_mel` `[1,80,52]` | fp32 only |
| `pre_encode_first` | `log_mel` `[1,80,49]`, `log_mel_length` `[1]` i32 | `features` `[1,512,13]`, `features_length` | fp32 only |
| `pre_encode_next` | `log_mel` `[1,80,57]`, `log_mel_length` `[1]` i32 | `features` `[1,512,13]`, `features_length` | fp32 only |
| `encoder_core` | `features`, `features_length`, `cache_last_channel` `[17,1,78,512]`, `cache_last_time` `[17,1,512,8]`, `cache_last_channel_length` `[1]` i32 | one logits tensor per level, `encoded_length`, and the three `next_cache_*` | fp32, int8 |

Why four graphs: step 0 and later steps need different shapes and frame drops. With static shapes, one graph would mean
two full copies of the encoder. The split also keeps the pre-encoder fp32: its weights aren't in the graph the quantizer sees.

"int8" means dynamic-range quantization: int8 weights and float activations, no calibration. The int8 bundle is the same fp32
`mel` and `pre_encode_*` files plus `encoder_core_int8`. Export fails if a graph that must stay fp32 contains int8 data.

### Driving the graphs

`streaming_driver.py` is the reference implementation: numpy only, and it uses the same chunking as `streaming_mic.py`.

1. Each `mel` call takes the raw window starting at `frame * 160 - 201`, zero-filled before the signal. The window advances 8320 samples per step.
2. Step 0 uses a window starting at frame -3, with `first_index=680`, and keeps the last 49 frames. Later steps set `first_index=0`.
3. `num_valid` is the full window size (8560) until the stream ends, then the number of samples left. This is how the
   graph reflects at the true signal end.
4. Later steps prepend the previous step's last 5 frames: 57 frames into `pre_encode_next`.
5. The last chunk is zero-padded to full width with its true length. A tail under 4 new frames is dropped.
6. Feed each `next_cache_*` output back in. Start from zeros, with `cache_last_channel_length=0`.

## How the mel graph was done

NeMo's STFT had to become something every runtime can execute.

- **Attempt A, `torch.stft` through `litert_torch`: fails.** `RuntimeError: Lowering not found: aten._fft_r2c.default`.
  The exporter retries this on each run and records the outcome in `manifest.json`.
- **Attempt B, STFT as a plain convolution: ships.** A `Conv1d` holding Hann × cos/sin kernels, then `re² + im²`, a filterbank
  and a log. Both reflect edges are done in the graph with integer min/max ops. 0.95 MB, no Flex ops.
- **Attempt C, ONNX fallback: not needed.** `mel.onnx` still exists because conformer-rs only reads ONNX.

Against real NeMo 2.1.0, the graph stays within the test limit of 1e-3 absolute log-mel error across 55 combinations of signal and
length. Float32 rounding means it's close rather than bit-identical.

**This depends on the NeMo version.** NeMo after 2.1.0 changes the STFT padding to `constant` and drops the `+ 1` from the frame count.
`tests/test_fastconformer_export_mel.py` compares against the installed NeMo, so an upgrade fails the test rather than silently
changing the features.

## Measured so far

All from a throwaway environment built from `uv.lock` (Python 3.14, torch 2.12, NeMo 2.1.0, litert-torch 0.9.1), on CPU.
**Weights are random**, so this shows the graphs are correct and how big they are, not how accurate.

Full size, 110.2 M parameters (17 layers, d_model 512, 11 levels):

| Artifact | LiteRT | ONNX |
|---|---|---|
| `mel` | 1.0 MB | 1.0 MB |
| `pre_encode_first`, `pre_encode_next` (each) | 11.3 MB | 11.3 MB |
| `encoder_core` fp32 | 418.3 MB | 432.4 MB |
| `encoder_core` int8 | 107.6 MB | 110.5 MB |

Output compared with the same split graphs in PyTorch, on 5.3 s of synthetic audio:

| Bundle | Max abs logit error | Argmax agreement |
|---|---|---|
| LiteRT fp32 | 3.2e-6 | 100% |
| ONNX fp32 | 2.1e-6 | 100% |
| LiteRT int8 | 5.8e-2 | 98.7% |
| ONNX int8 | 1.7e-1 | 98.5% |

Random weights say little about int8 accuracy; the PER gate decides that. Export peaked at about 4.5 GB of RAM.

**XNNPACK partitions:** `encoder_core` splits into **53** at fp32 and **69** at int8. The other graphs have 2 to 4. Each boundary
hands off between XNNPACK and the plain kernels, so this is the first thing to look at if phone latency disappoints.
`benchmark_model` will show which ops fall outside XNNPACK.

`int8` also ran without the `batch_matmul.cc` failure that blocked every wav2vec2-conformer in conformer-rs. If it does appear, the
exporter leaves `BATCH_MATMUL` in fp32 and records that in `manifest.json`.

## Gates

| # | Gate | Pass condition | Status |
|---|---|---|---|
| 1 | Mel vs real `FastConformerMelProcessor(dither=0)` | max abs error ≤ 1e-3, frame counts equal | Passed in PyTorch (55 cases), and via LiteRT and ONNX through gate 4 |
| 2 | Split graphs in PyTorch vs `infer_fastconformer_streaming` | logit error ≤ 1e-5 | Passed on a tiny model |
| 3 | A misaligned offset still raises outside export | raises `ValueError` | Passed |
| 4 | fp32 files vs PyTorch | logit error ≤ 1e-3, same argmax, no NaN | Passed on a tiny model with audio from `assets/`. **Real checkpoint pending** |
| 5 | PER on Test and QDAT, in one harness with dither off | fp32 Δ ≤ 2e-4; int8 QDAT Δ ≤ 0.005 | **Not run** |
| 6 | Graph report | no Flex ops | Passed |
| 7 | `mel` and `pre_encode_*` free of int8 | no int8 tensors or quantized ops | Passed, and enforced during export |

Gate 5 re-runs PyTorch rather than using `docs/streaming_analysis.md` (base-v1: Test 0.00211, QDAT 0.0572), because those numbers
were produced with dither on. If int8 misses its limit, bring the numbers back for a decision instead of shipping it.

## Still to run

Needs the real checkpoint and the `annotate` extra. The easiest route is the Colab notebook
[`notebooks/fastconformer_export_colab.ipynb`](../notebooks/fastconformer_export_colab.ipynb), which installs `uv.lock` under
Python 3.14 with `uv` and runs everything below. Its first cell lists what to set up: an HF token in the Colab secret `HF_TOKEN_MUALEM`, a CPU runtime and Drive.

The test set is about 33k utterances. At batch size 1, five runs of it would take days of CPU time, so the notebook scores a fixed random subset
(`--max-test-utterances`). That costs little rigor. fp32 equivalence is already settled by fidelity, since identical argmax means identical PER,
and the int8 gate is on QDAT (159 clips), which always runs in full.

Validation options:

- `--bundles litert/int8,onnx/fp32` checks only those bundles.
- Fidelity also runs the four graphs unconverted in PyTorch (`pytorch/split`). Each bundle's `vs_split` numbers are its
  conversion error alone; the `pytorch/split` row is what the conv-STFT mel changes relative to NeMo's.
- `--max-test-utterances N` scores a fixed random subset (seed 0) of the test set.
- PER runs are saved to `validation/per/<run>.json` and reused on a rerun with the same settings. `--no-resume` recomputes them.
- Running only one check writes `validation/report_<check>.json`; running both writes `validation/report.json`.

The same steps by hand:

```bash
uv sync --extra annotate --extra nemo --extra litert --extra onnx --extra test

# Tests. Paths must be explicit, because pyproject's testpaths doesn't include them.
uv run pytest tests/test_fastconformer_export_mel.py tests/test_fastconformer_export_graphs.py \
    tests/test_fastconformer_mask_golden.py tests/test_ctc_decoding.py tests/test_streaming_processor.py -v

uv run python -m prepare_quran_dataset.fastconformer_export.export \
    --model-id obadx/muaalem-fastconformer-base-v1 --output-dir exports/base-v1

# Gates 4 and 5, from the repository root. Stops non-zero if a gate fails; writes exports/base-v1/validation/report.json.
uv run python -m prepare_quran_dataset.fastconformer_export.validate \
    --model-id obadx/muaalem-fastconformer-base-v1 --artifacts exports/base-v1 --max-test-utterances 300
```

The `per` check has **never been executed**, since `train_streaming` needs packages from the `annotate` extra. Run it on its own with
`--checks per` if the combined run fails there. It evaluates at batch size 1 for every bundle, so expect it to take a while.

Only `constant_lookahead_delay=0` checkpoints can be exported; c2 and c5 are rejected with an error, because their last chunk's output
width differs.

## Later: on-device benchmark

Not started. Each item below is needed before numbers can be collected.

- **Harness:** the conformer-rs crates feed one constant rank-3 input. They need a streaming mode with several inputs, cache feedback, real audio,
  per-stage latency, RTF against 520 ms, peak RSS, and a check of decoded output against a Python golden.
- **Runtimes:** LiteRT through `android_aarch64_benchmark_model`, replaying captured mid-stream tensors. ONNX fp32 through `conformer-rs` (ort-tract) and
  `conformer-rten`. ONNX int8 through `conformer-rten`: its README lists ort-tract as unable to run quantized ops.
- **ONNX ops to confirm:** `encoder_core` uses `LayerNormalization`, which the conformer-rs README lists as unsupported in its tract backend. Current tract and rten
  sources both register it, but conformer-rs pins an older `ort-tract`, so this may need a version bump.
- **Device:** to be chosen.
- **Real-time risk:** a 97 M streaming conformer took 693 ms on one thread at fp32 on a Redmi 9 (conformer-rs README, 700 ms input). With a 520 ms budget,
  this model may need more threads or int8 there. That's an extrapolation, not a measurement.
