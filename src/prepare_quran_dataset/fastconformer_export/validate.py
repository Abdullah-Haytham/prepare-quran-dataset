#!/usr/bin/env python3
"""Check exported bundles against the PyTorch model, on the real checkpoint.

Two groups of checks, each writing into ``<artifacts>/validation/report.json``:

``fidelity``
    Per-utterance logits of every bundle vs ``infer_fastconformer_streaming``.

``per``
    Phoneme and average PER on the test set and QDAT, using ``train_streaming``'s own
    evaluation functions, so labels and metrics are exactly those of the training runs.
    The reference PyTorch model runs in this same harness with dither off, since the numbers
    in ``docs/streaming_analysis.md`` were produced with it on.

    Each run (the reference and every bundle) is saved to ``validation/per/<name>.json`` as
    soon as it finishes, and a rerun with the same settings reuses it.  So an interrupted run
    (a Colab disconnect, say) loses at most the run in progress.  ``--no-resume`` recomputes.

Exits non-zero if a gate fails.  Run from the repository root, as ``train_streaming`` reads
``./vocab_streaming``::

    uv run python -m prepare_quran_dataset.fastconformer_export.validate \\
        --model-id obadx/muaalem-fastconformer-base-v1 --artifacts exports/base-v1
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch

from .streaming_driver import (
    LiteRtBackend,
    OnnxBackend,
    StreamingGeometry,
    load_manifest,
    stream_waveform,
)

# Plan gates; see docs/fastconformer_export.md.
FP32_MAX_LOGIT_ERROR = 1e-3
FP32_MAX_PER_DELTA = 2e-4
INT8_MAX_QDAT_PER_DELTA = 5e-3

SAMPLE_RATE = 16000


def bundle_backends(
    artifact_dir: Path, manifest: dict, selected: Iterable[str] | None = None
) -> dict[str, object]:
    """One backend per bundle in the manifest, keyed ``format/precision``.

    ``selected`` restricts this to the named bundles; naming one the manifest lacks raises.
    """
    available = [
        (fmt, precision)
        for fmt, bundles in manifest["bundles"].items()
        for precision in bundles
    ]
    if selected is not None:
        selected = set(selected)
        unknown = selected - {f"{fmt}/{precision}" for fmt, precision in available}
        if unknown:
            raise ValueError(
                f"Unknown bundles {sorted(unknown)}; the manifest has "
                f"{[f'{fmt}/{precision}' for fmt, precision in available]}."
            )
        available = [b for b in available if f"{b[0]}/{b[1]}" in selected]
    classes = {"litert": LiteRtBackend, "onnx": OnnxBackend}
    return {
        f"{fmt}/{precision}": classes[fmt](artifact_dir, precision)
        for fmt, precision in available
    }


def _load_audio(source) -> np.ndarray:
    import librosa

    # Same call as MuaalemCacheAwareStreamingAudioBuffer.append_audio_file.
    wave, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    return wave


def _utterances(audio_dir: Path | None, num_qdat: int) -> Iterable[tuple[str, object]]:
    if audio_dir is not None:
        for path in sorted(audio_dir.iterdir()):
            if path.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac", ".m4a"):
                yield str(path), str(path)
    if num_qdat > 0:
        from datasets import Audio, load_dataset

        dataset = load_dataset("obadx/qdat_bench", split="train")
        dataset = dataset.cast_column("audio", Audio(decode=False))
        for index in range(min(num_qdat, len(dataset))):
            yield f"qdat/{index}", io.BytesIO(dataset[index]["audio"]["bytes"])


def check_fidelity(
    model,
    processor,
    backends: dict[str, object],
    geometry: StreamingGeometry,
    audio_dir: Path | None,
    num_qdat: int,
) -> dict:
    """Logit error and argmax agreement per bundle, against the PyTorch streaming path."""
    from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
        infer_fastconformer_streaming,
    )

    stats = {
        name: {"max_abs_error": 0.0, "frames": 0, "argmax_matches": 0, "non_finite": 0}
        for name in backends
    }
    count = 0
    for name, source in _utterances(audio_dir, num_qdat):
        wave = _load_audio(source)
        if isinstance(source, io.BytesIO):
            source.seek(0)
        reference = infer_fastconformer_streaming(
            [source], "cpu", torch.float32, model, processor
        )
        reference = {level: value[0].numpy() for level, value in reference.items()}
        count += 1
        for bundle, backend in backends.items():
            got = stream_waveform(backend, geometry, wave)
            s = stats[bundle]
            for level in geometry.levels:
                if got[level].shape != reference[level].shape:
                    raise RuntimeError(
                        f"{bundle} on {name}, level {level}: shape "
                        f"{got[level].shape} vs reference {reference[level].shape}."
                    )
                s["non_finite"] += int((~np.isfinite(got[level])).sum())
                s["max_abs_error"] = max(
                    s["max_abs_error"],
                    float(np.abs(got[level] - reference[level]).max()),
                )
                s["frames"] += got[level].shape[0]
                s["argmax_matches"] += int(
                    (got[level].argmax(-1) == reference[level].argmax(-1)).sum()
                )
        print(f"  fidelity {count}: {name}", flush=True)

    for name, s in stats.items():
        s["argmax_agreement"] = s["argmax_matches"] / max(s["frames"], 1)
        if name.endswith("/fp32"):
            s["passed"] = (
                s["max_abs_error"] <= FP32_MAX_LOGIT_ERROR
                and s["argmax_matches"] == s["frames"]
                and s["non_finite"] == 0
            )
        else:
            # int8 is judged on PER: small logit shifts are expected and fine.
            s["passed"] = s["non_finite"] == 0
    return {"utterances": count, "bundles": stats}


def _artifact_infer(backend, geometry: StreamingGeometry) -> Callable:
    """Same signature as ``infer_fastconformer_streaming``, running exported graphs."""

    def infer(audio_sources, device, dtype, model, processor, **_):
        if len(audio_sources) != 1:
            raise ValueError("Exported graphs are batch-size 1.")
        logits = stream_waveform(backend, geometry, _load_audio(audio_sources[0]))
        return {
            level: torch.from_numpy(np.array(value)).unsqueeze(0)
            for level, value in logits.items()
        }

    return infer


def _run_per(train_streaming, model, processor, testset, suffix, output_dir, tokenizer, vocab, config):
    device = torch.device("cpu")
    results = {}
    if testset is not None:
        metrics = train_streaming.run_streaming_testset_test(
            model=model,
            processor=processor,
            multi_level_tokenizer=tokenizer,
            testset=testset,
            output_dir=output_dir,
            vocab=vocab,
            device=device,
            dtype=torch.float32,
            batch_size=1,
            model_suffix=suffix,
            add_eos_token=config.add_eos_token,
        )
        results["test_per_phonemes"] = metrics.get("test_streaming__per_phonemes")
        results["test_average_per"] = metrics.get("test_streaming__average_per")
    metrics = train_streaming.run_streaming_qdat_bench_test(
        model=model,
        processor=processor,
        multi_level_tokenizer=tokenizer,
        output_dir=output_dir,
        vocab=vocab,
        device=device,
        dtype=torch.float32,
        batch_size=1,
        model_suffix=suffix,
    )
    speech = metrics.get("speech_metrics") or {}
    results["qdat_per_phonemes"] = speech.get("per_phonemes")
    results["qdat_average_per"] = speech.get("average_per")
    return results


def _subset_testset(testset, max_utterances: int | None):
    """A fixed random subset of the test split, so reruns score the same utterances."""
    if max_utterances is None or max_utterances >= len(testset["test"]):
        return testset
    testset["test"] = testset["test"].shuffle(seed=0).select(range(max_utterances))
    return testset


class PerCache:
    """One JSON file per PER run, valid only for the settings it was computed with."""

    def __init__(self, directory: Path, settings: dict, resume: bool) -> None:
        self.directory = directory
        self.settings = settings
        self.resume = resume

    def _path(self, name: str) -> Path:
        return self.directory / f"{name.replace('/', '_')}.json"

    def load(self, name: str) -> dict | None:
        path = self._path(name)
        if not self.resume or not path.exists():
            return None
        saved = json.loads(path.read_text())
        if saved.get("settings") != self.settings:
            return None
        return saved["result"]

    def save(self, name: str, result: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(name).write_text(
            json.dumps({"settings": self.settings, "result": result}, indent=2)
        )


def per_gates(per: dict[str, dict], bundles: Iterable[str]) -> dict[str, dict]:
    """Compare each bundle's PER with the reference under the plan's gates."""
    gates = {}
    reference = per["pytorch"]
    for bundle in bundles:
        deltas = {
            key: abs(per[bundle][key] - value)
            for key, value in reference.items()
            if value is not None and per[bundle].get(key) is not None
        }
        if bundle.endswith("/fp32"):
            passed = bool(deltas) and max(deltas.values()) <= FP32_MAX_PER_DELTA
        else:
            # Signed: only a worse int8 PER should fail the gate.
            qdat_delta = per[bundle]["qdat_per_phonemes"] - reference["qdat_per_phonemes"]
            passed = qdat_delta <= INT8_MAX_QDAT_PER_DELTA
        gates[bundle] = {"deltas": deltas, "passed": passed}
    return gates


def check_per(
    model,
    processor,
    backends: dict[str, object],
    geometry: StreamingGeometry,
    train_config_path: str,
    output_dir: Path,
    skip_testset: bool,
    max_test_utterances: int | None = None,
    resume: bool = True,
    model_id: str | None = None,
) -> dict:
    """PER of the reference and of each bundle, compared under the plan's gates."""
    # train_streaming is a script at the repository root, on pytest's pythonpath only.
    import sys

    sys.path.insert(0, os.getcwd())
    import train_streaming
    from datasets import load_dataset
    from quran_transcript import MoshafAttributes

    train_config = train_streaming.TrainConfig.from_yaml(train_config_path)
    tokenizer = train_streaming.MultiLevelTokenizer("./vocab_streaming")
    vocab, _ = train_streaming.load_vocab()

    use_testset = not skip_testset and train_config.test_moshaf_ids is not None
    cache = PerCache(
        output_dir / "per",
        {
            "model_id": model_id,
            "train_config": train_config_path,
            "testset": use_testset,
            "max_test_utterances": max_test_utterances if use_testset else None,
        },
        resume,
    )
    per: dict[str, dict] = {}
    for name in ("pytorch", *backends):
        cached = cache.load(name)
        if cached is not None:
            print(f"PER: {name} (reused from {cache.directory})", flush=True)
            per[name] = cached
    pending = [name for name in ("pytorch", *backends) if name not in per]

    testset = None
    if use_testset and pending:
        moshaf = load_dataset(
            "obadx/mualem-recitations-annotated", name="moshaf_metadata", split="train"
        )
        moshaf_dict = {example["id"]: example for example in moshaf}
        testset = train_streaming.prepare_dataset(
            train_config,
            processor,
            tokenizer,
            muaalem_ds=train_streaming.prepare_muaalem_dataset(
                train_config,
                {example["id"]: MoshafAttributes(**example) for example in moshaf},
                train_streaming.prepare_special_moshaf_ways(moshaf_dict),
                is_testset=True,
            ),
            is_testset=True,
        )
        testset = _subset_testset(testset, max_test_utterances)
        print(f"Test set: {len(testset['test'])} utterances", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_infer = train_streaming.infer_fastconformer_streaming
    try:
        for name in pending:
            print(f"PER: {name}", flush=True)
            if name == "pytorch":
                train_streaming.infer_fastconformer_streaming = reference_infer
                suffix = "reference"
            else:
                train_streaming.infer_fastconformer_streaming = _artifact_infer(
                    backends[name], geometry
                )
                suffix = name.replace("/", "_")
            per[name] = _run_per(
                train_streaming, model, processor, testset, suffix, output_dir,
                tokenizer, vocab, train_config,
            )
            cache.save(name, per[name])
    finally:
        train_streaming.infer_fastconformer_streaming = reference_infer

    return {
        "settings": cache.settings,
        "per": per,
        "gates": per_gates(per, backends),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default="obadx/muaalem-fastconformer-base-v1")
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument(
        "--checks", default="fidelity,per", help="Comma-separated: fidelity, per."
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("assets/audio-sampels"))
    parser.add_argument("--num-qdat-utterances", type=int, default=100)
    parser.add_argument(
        "--train-config",
        default="configs/train/streaming/train_config_fastconformer_base_v1.yml",
    )
    parser.add_argument("--skip-testset", action="store_true")
    parser.add_argument(
        "--max-test-utterances",
        type=int,
        default=None,
        help="Score a fixed random subset of the test set (the full set is ~33k utterances).",
    )
    parser.add_argument(
        "--bundles",
        default=None,
        help="Comma-separated bundles to check, e.g. litert/int8,onnx/fp32. Default: all.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute PER runs even if validation/per/ already holds them.",
    )
    args = parser.parse_args()

    from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
        FastConformerMelProcessor,
    )
    from prepare_quran_dataset.modeling_fastconformer_cache_aware.modeling_fastconformer_cache_aware_ctc import (
        FastConformerCacheAwareMultilevelCTC,
    )

    model = FastConformerCacheAwareMultilevelCTC.from_pretrained(
        args.model_id, token=os.environ.get("HF_TOKEN")
    )
    model = model.eval().float()
    model.setup_streaming_params()
    processor = FastConformerMelProcessor(
        **{**model.config.processor_kwargs, "dither": 0.0, "pad_to": 0}
    )

    manifest = load_manifest(args.artifacts)
    geometry = StreamingGeometry.from_json(manifest["geometry"])
    backends = bundle_backends(
        args.artifacts, manifest, args.bundles.split(",") if args.bundles else None
    )

    report: dict = {"model_id": args.model_id, "versions": manifest["versions"]}
    checks = args.checks.split(",")
    passed = True
    with torch.no_grad():
        if "fidelity" in checks:
            report["fidelity"] = check_fidelity(
                model, processor, backends, geometry, args.audio_dir, args.num_qdat_utterances
            )
            passed &= all(s["passed"] for s in report["fidelity"]["bundles"].values())
        if "per" in checks:
            report["per"] = check_per(
                model,
                processor,
                backends,
                geometry,
                args.train_config,
                args.artifacts / "validation",
                args.skip_testset,
                max_test_utterances=args.max_test_utterances,
                resume=not args.no_resume,
                model_id=args.model_id,
            )
            passed &= all(g["passed"] for g in report["per"]["gates"].values())
    report["passed"] = bool(passed)

    # Separate reports per check, so running them in separate steps keeps both.
    suffix = "" if set(checks) == {"fidelity", "per"} else "_" + "_".join(checks)
    report_path = args.artifacts / "validation" / f"report{suffix}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{'PASSED' if passed else 'FAILED'}: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
