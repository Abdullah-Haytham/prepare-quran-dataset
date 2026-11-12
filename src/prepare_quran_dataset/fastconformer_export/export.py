#!/usr/bin/env python3
"""Export a cache-aware FastConformer to LiteRT and ONNX, in fp32 and int8.

Writes four graphs per format, plus ``manifest.json``.  "int8" means dynamic-range
quantization (int8 weights, float activations) of ``encoder_core`` only; ``mel`` and
``pre_encode_*`` stay fp32 in both bundles, and the export fails if they don't.

Usage::

    uv run python -m prepare_quran_dataset.fastconformer_export.export \\
        --model-id obadx/muaalem-fastconformer-base-v1 --output-dir exports/base-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .mel_frontend import DirectStftLogMel, MelGeometry, StreamingLogMel
from .streaming_driver import MANIFEST_NAME
from .wrappers import GraphSpec, build_graph_specs

# Ops that must never appear in a graph that is meant to stay fp32.
_ONNX_QUANTIZED_OPS = {
    "ConvInteger",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "DynamicQuantizeMatMul",
    "MatMulInteger",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
}


def _sample_numpy(spec: GraphSpec) -> dict[str, np.ndarray]:
    return {name: tensor.numpy() for name, tensor in spec.inputs.items()}


def _check_outputs(name: str, outputs: list[np.ndarray]) -> None:
    for output in outputs:
        if np.issubdtype(output.dtype, np.floating) and not np.isfinite(output).all():
            raise RuntimeError(f"{name}: non-finite values in a sample-input run.")


# ----------------------------------------------------------------------
# LiteRT
# ----------------------------------------------------------------------


def _litert_interpreter(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    return Interpreter(model_path=str(path))


def litert_report(path: Path) -> dict:
    """Op histogram, tensor types and how many pieces XNNPACK splits the graph into.

    More partitions means more switching between XNNPACK and the plain kernels, which
    slows inference.  Which ops fell outside XNNPACK only shows in ``benchmark_model``'s log.
    """
    interpreter = _litert_interpreter(path)
    ops = Counter(op["op_name"] for op in interpreter._get_ops_details())
    tensor_types = Counter(
        np.dtype(t["dtype"]).name for t in interpreter.get_tensor_details()
    )
    interpreter.allocate_tensors()
    partitions = sum(
        op["op_name"] == "DELEGATE" for op in interpreter._get_ops_details()
    )
    return {
        "bytes": path.stat().st_size,
        "ops": dict(ops.most_common()),
        "flex_ops": sorted(op for op in ops if op.startswith("Flex")),
        "tensor_types": dict(tensor_types),
        "xnnpack_partitions": partitions,
    }


def _smoke_run_litert(path: Path, spec: GraphSpec) -> None:
    """Load and invoke, which is where int8 kernels reject unsupported tensor types."""
    runner = _litert_interpreter(path).get_signature_runner()
    outputs = runner(**_sample_numpy(spec))
    if len(outputs) != len(spec.output_names):
        raise RuntimeError(
            f"{path.name}: {len(outputs)} outputs, expected {len(spec.output_names)}."
        )
    _check_outputs(path.name, list(outputs.values()))


def convert_litert(spec: GraphSpec, path: Path) -> None:
    import litert_torch

    edge_model = litert_torch.convert(spec.module, sample_kwargs=spec.inputs)
    edge_model.export(str(path))
    _smoke_run_litert(path, spec)


def quantize_litert(spec: GraphSpec, fp32_path: Path, int8_path: Path) -> dict:
    """Dynamic-range int8 of a converted graph, with a fallback for ``BATCH_MATMUL``.

    Every int8 wav2vec2-conformer in conformer-rs failed in ``batch_matmul.cc`` with a
    mixed-precision error.  If that happens here, the graph is quantized again leaving
    ``BATCH_MATMUL`` in fp32, and the report records it.
    """
    from ai_edge_quantizer import algorithm_manager, qtyping, quantizer, recipe

    qt = quantizer.Quantizer(str(fp32_path))
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    qt.quantize().export_model(str(int8_path), overwrite=True)
    try:
        _smoke_run_litert(int8_path, spec)
        return {"recipe": "dynamic_wi8_afp32"}
    except Exception as e:  # noqa: BLE001 - the runtime raises a bare RuntimeError/ValueError
        first_error = f"{type(e).__name__}: {e}"

    qt = quantizer.Quantizer(str(fp32_path))
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.BATCH_MATMUL,
        algorithm_key=algorithm_manager.AlgorithmName.NO_QUANTIZE,
    )
    qt.quantize().export_model(str(int8_path), overwrite=True)
    _smoke_run_litert(int8_path, spec)
    return {
        "recipe": "dynamic_wi8_afp32 without BATCH_MATMUL",
        "first_attempt_error": first_error,
    }


def try_direct_stft(geometry: MelGeometry, work_dir: Path) -> dict:
    """Attempt A: can ``torch.stft`` itself go through ``litert_torch``?

    The outcome is recorded either way.  The conv STFT ships regardless, so a failure
    here is a finding rather than a problem.
    """
    module = DirectStftLogMel(geometry).eval()
    reference = StreamingLogMel(geometry).eval()
    generator = torch.Generator().manual_seed(0)
    raw = torch.randn(1, geometry.num_raw_samples, generator=generator) * 0.1
    args = {
        "raw": raw,
        "first_index": torch.tensor([0], dtype=torch.int32),
        "num_valid": torch.tensor([geometry.num_emphasized_samples], dtype=torch.int32),
    }
    path = work_dir / "mel_direct_stft.tflite"
    try:
        import litert_torch

        litert_torch.convert(module, sample_kwargs=args).export(str(path))
        report = {"converted": True, **litert_report(path)}
        outputs = _litert_interpreter(path).get_signature_runner()(
            **{name: value.numpy() for name, value in args.items()}
        )
        with torch.no_grad():
            expected = reference(**args).numpy()
        report["max_abs_diff_vs_conv_stft"] = float(
            np.abs(outputs["output_0"] - expected).max()
        )
        return report
    except Exception as e:  # noqa: BLE001 - recording any failure is the point
        return {
            "converted": False,
            "error": f"{type(e).__name__}: {e}"[:2000],
            "traceback_tail": traceback.format_exc()[-1500:],
        }
    finally:
        path.unlink(missing_ok=True)


def export_litert(specs: list[GraphSpec], out_dir: Path) -> tuple[dict, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    files: dict[str, str] = {}
    for spec in specs:
        fp32_name = f"{spec.name}_fp32.tflite" if spec.quantize else f"{spec.name}.tflite"
        print(f"[litert] converting {spec.name} ...", flush=True)
        convert_litert(spec, out_dir / fp32_name)
        report[spec.name] = litert_report(out_dir / fp32_name)
        files[spec.name] = fp32_name
        if not spec.quantize:
            _require_litert_fp32(out_dir / fp32_name, report[spec.name])
    return report, files


def _require_litert_fp32(path: Path, report: dict) -> None:
    """Gate 7: raise if a graph that must stay fp32 holds any int8 data."""
    quantized = {
        kind: count
        for kind, count in report["tensor_types"].items()
        if kind in ("int8", "uint8", "int4")
    }
    if quantized:
        raise RuntimeError(f"{path.name} must stay fp32 but has {quantized} tensors.")


# ----------------------------------------------------------------------
# ONNX
# ----------------------------------------------------------------------


def onnx_report(path: Path) -> dict:
    """Op histogram and initializer types."""
    import onnx

    model = onnx.load(str(path))
    ops = Counter(node.op_type for node in model.graph.node)
    initializer_types = Counter(
        onnx.TensorProto.DataType.Name(init.data_type)
        for init in model.graph.initializer
    )
    return {
        "bytes": path.stat().st_size,
        "ops": dict(ops.most_common()),
        "initializer_types": dict(initializer_types),
    }


def _require_onnx_fp32(path: Path, report: dict) -> None:
    """Gate 7: raise if a graph that must stay fp32 holds any quantized op or weight."""
    quantized_ops = sorted(set(report["ops"]) & _ONNX_QUANTIZED_OPS)
    int8_weights = {
        kind: count
        for kind, count in report["initializer_types"].items()
        if kind in ("INT8", "UINT8")
    }
    if quantized_ops or int8_weights:
        raise RuntimeError(
            f"{path.name} must stay fp32 but has ops {quantized_ops} "
            f"and initializers {int8_weights}."
        )


def _smoke_run_onnx(path: Path, spec: GraphSpec) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _check_outputs(path.name, session.run(None, _sample_numpy(spec)))


def convert_onnx(spec: GraphSpec, path: Path) -> None:
    import onnx

    torch.onnx.export(
        spec.module,
        args=tuple(spec.inputs.values()),
        f=str(path),
        input_names=list(spec.inputs),
        output_names=list(spec.output_names),
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(str(path))
    _smoke_run_onnx(path, spec)


def quantize_onnx(spec: GraphSpec, fp32_path: Path, int8_path: Path) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    settings = {
        "weight_type": QuantType.QInt8,
        "op_types_to_quantize": ["MatMul", "Conv"],
        "extra_options": {"MatMulConstBOnly": True},
    }
    quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path), **settings)
    _smoke_run_onnx(int8_path, spec)
    return {"recipe": "quantize_dynamic QInt8 MatMul,Conv MatMulConstBOnly"}


def export_onnx(specs: list[GraphSpec], out_dir: Path) -> tuple[dict, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    files: dict[str, str] = {}
    for spec in specs:
        fp32_name = f"{spec.name}_fp32.onnx" if spec.quantize else f"{spec.name}.onnx"
        print(f"[onnx] converting {spec.name} ...", flush=True)
        convert_onnx(spec, out_dir / fp32_name)
        report[spec.name] = onnx_report(out_dir / fp32_name)
        files[spec.name] = fp32_name
        if not spec.quantize:
            _require_onnx_fp32(out_dir / fp32_name, report[spec.name])
    return report, files


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version()}
    for package in (
        "torch",
        "nemo-toolkit",
        "transformers",
        "litert-torch",
        "ai-edge-litert",
        "ai-edge-quantizer",
        "onnx",
        "onnxruntime",
        "onnxscript",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def export_model(
    model,
    output_dir: str | Path,
    formats: tuple[str, ...] = ("litert", "onnx"),
    selected_levels: set[str] | None = None,
    model_id: str | None = None,
    attempt_direct_stft: bool = True,
) -> dict:
    """Export a loaded model and write ``manifest.json``.  Returns the manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, geometry = build_graph_specs(model, selected_levels=selected_levels)
    quantized_spec = next(spec for spec in specs if spec.quantize)

    manifest: dict = {
        "schema": 1,
        "model_id": model_id,
        "versions": _versions(),
        "geometry": geometry.to_json(),
        "graphs": {
            spec.name: {
                "inputs": {
                    name: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                    }
                    for name, tensor in spec.inputs.items()
                },
                "outputs": list(spec.output_names),
                "quantized_in_int8_bundle": spec.quantize,
            }
            for spec in specs
        },
        "bundles": {},
        "reports": {},
        "quantization": {},
        "files": {},
    }

    if "litert" in formats:
        if attempt_direct_stft:
            mel_geometry = specs[0].module.geometry
            print("[litert] attempt A: direct torch.stft ...", flush=True)
            manifest["mel_direct_stft_attempt"] = try_direct_stft(
                mel_geometry, output_dir
            )
            print(
                "[litert] attempt A converted:",
                manifest["mel_direct_stft_attempt"]["converted"],
                flush=True,
            )
        report, files = export_litert(specs, output_dir / "litert")
        int8_name = f"{quantized_spec.name}_int8.tflite"
        print(f"[litert] quantizing {quantized_spec.name} ...", flush=True)
        manifest["quantization"]["litert"] = quantize_litert(
            quantized_spec,
            output_dir / "litert" / files[quantized_spec.name],
            output_dir / "litert" / int8_name,
        )
        report[f"{quantized_spec.name}_int8"] = litert_report(
            output_dir / "litert" / int8_name
        )
        manifest["reports"]["litert"] = report
        manifest["bundles"]["litert"] = _bundles("litert", files, quantized_spec, int8_name)

    if "onnx" in formats:
        report, files = export_onnx(specs, output_dir / "onnx")
        int8_name = f"{quantized_spec.name}_int8.onnx"
        print(f"[onnx] quantizing {quantized_spec.name} ...", flush=True)
        manifest["quantization"]["onnx"] = quantize_onnx(
            quantized_spec,
            output_dir / "onnx" / files[quantized_spec.name],
            output_dir / "onnx" / int8_name,
        )
        report[f"{quantized_spec.name}_int8"] = onnx_report(
            output_dir / "onnx" / int8_name
        )
        manifest["reports"]["onnx"] = report
        manifest["bundles"]["onnx"] = _bundles("onnx", files, quantized_spec, int8_name)

    for fmt, bundles in manifest["bundles"].items():
        for bundle in bundles.values():
            for rel in bundle.values():
                if rel not in manifest["files"]:
                    path = output_dir / rel
                    manifest["files"][rel] = {
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }

    with open(output_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _bundles(fmt: str, files: dict, quantized_spec: GraphSpec, int8_name: str) -> dict:
    """The fp32 and int8 bundles, which point at the same fp32 ``mel`` and ``pre_encode_*``."""
    fp32 = {graph: f"{fmt}/{name}" for graph, name in files.items()}
    return {
        "fp32": fp32,
        "int8": {**fp32, quantized_spec.name: f"{fmt}/{int8_name}"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-id",
        default="obadx/muaalem-fastconformer-base-v1",
        help="Hugging Face model id or local checkpoint directory.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--formats",
        default="litert,onnx",
        help="Comma-separated subset of: litert, onnx.",
    )
    parser.add_argument(
        "--selected-levels",
        default=None,
        help="Comma-separated levels to keep (e.g. phonemes,ghonna). Default: all.",
    )
    parser.add_argument(
        "--skip-direct-stft-attempt",
        action="store_true",
        help="Don't try torch.stft through litert_torch first.",
    )
    args = parser.parse_args()

    from prepare_quran_dataset.modeling_fastconformer_cache_aware.modeling_fastconformer_cache_aware_ctc import (
        FastConformerCacheAwareMultilevelCTC,
    )

    model = FastConformerCacheAwareMultilevelCTC.from_pretrained(
        args.model_id, token=os.environ.get("HF_TOKEN")
    )
    manifest = export_model(
        model,
        args.output_dir,
        formats=tuple(args.formats.split(",")),
        selected_levels=set(args.selected_levels.split(","))
        if args.selected_levels
        else None,
        model_id=args.model_id,
        attempt_direct_stft=not args.skip_direct_stft_attempt,
    )

    print(f"\nWrote {Path(args.output_dir) / MANIFEST_NAME}")
    for rel, info in sorted(manifest["files"].items()):
        print(f"  {info['bytes'] / 1e6:9.1f} MB  {rel}")


if __name__ == "__main__":
    main()
