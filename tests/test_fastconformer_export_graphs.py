"""End-to-end export checks on a tiny, randomly initialised FastConformer.

It has the real streaming geometry (``att_context_size=[78, 12]``, 4x subsampling) but few
weights, so this exercises every conversion quickly and needs no checkpoint.  Random weights
say nothing about accuracy; ``fastconformer_export.validate`` covers that on the real model.

Plan gates covered: 1 (mel through the converted files), 2, 3, 4 (fp32 parity), 6 and 7.

Run with: ``uv run pytest tests/test_fastconformer_export_graphs.py -v``
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo")
sf = pytest.importorskip("soundfile")

from prepare_quran_dataset.fastconformer_export.export import export_model  # noqa: E402
from prepare_quran_dataset.fastconformer_export.streaming_driver import (  # noqa: E402
    MANIFEST_NAME,
    LiteRtBackend,
    OnnxBackend,
    StreamingGeometry,
    stream_waveform,
)
from prepare_quran_dataset.fastconformer_export.wrappers import (  # noqa: E402
    TorchBackend,
    build_graph_specs,
)
from prepare_quran_dataset.modeling_fastconformer_cache_aware import (  # noqa: E402
    FastConformerCacheAwareMultilevelCTC,
    FastConformerCacheAwareMultilevelCTCConfig,
    FastConformerMelProcessor,
    infer_fastconformer_streaming,
)

LEVELS = {"phonemes": 20, "hams_or_jahr": 4}
SAMPLE_RATE = 16000
# Exercises cache fill (0 -> 78 takes 6 steps), a full cache, and a partial tail.
WAVE_SECONDS = 5.3
WRAPPER_TOLERANCE = 1e-5
FP32_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = FastConformerCacheAwareMultilevelCTCConfig(
        level_to_vocab_size=LEVELS,
        level_to_loss_weight={"phonemes": 0.6},
        n_layers=2,
        d_model=64,
        n_heads=4,
        subsampling_conv_channels=32,
        att_context_size=[78, 12],
        fp32_pre_encode=True,
    )
    model = FastConformerCacheAwareMultilevelCTC(config).eval()
    model.setup_streaming_params()
    return model


@pytest.fixture(scope="module")
def wave():
    rng = np.random.default_rng(0)
    length = int(WAVE_SECONDS * SAMPLE_RATE)
    t = np.arange(length) / SAMPLE_RATE
    signal = 0.2 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 0.02, length)
    return signal.astype(np.float32)


@pytest.fixture(scope="module")
def reference(model, wave, tmp_path_factory):
    """The real streaming path: NeMo features through ``model.streaming_step``."""
    path = tmp_path_factory.mktemp("audio") / "wave.wav"
    sf.write(path, wave, SAMPLE_RATE, subtype="FLOAT")
    processor = FastConformerMelProcessor(
        **{**model.config.processor_kwargs, "dither": 0.0, "pad_to": 0}
    )
    logits = infer_fastconformer_streaming(
        [str(path)], "cpu", torch.float32, model, processor
    )
    return {level: value[0].numpy() for level, value in logits.items()}


@pytest.fixture(scope="module")
def artifacts(model, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("export")
    export_model(model, output_dir, model_id="tiny-random")
    return output_dir


def _geometry(artifacts) -> StreamingGeometry:
    manifest = json.loads((artifacts / MANIFEST_NAME).read_text())
    return StreamingGeometry.from_json(manifest["geometry"])


def _assert_close(got, reference, tolerance):
    assert set(got) == set(reference)
    for level in reference:
        assert got[level].shape == reference[level].shape, level
        error = np.abs(got[level] - reference[level]).max()
        assert error <= tolerance, f"{level}: max abs logit error {error:.2e}"
        np.testing.assert_array_equal(
            got[level].argmax(-1), reference[level].argmax(-1), err_msg=level
        )


def test_geometry_comes_from_the_model(artifacts):
    g = _geometry(artifacts)
    assert (g.first_chunk_frames, g.chunk_frames) == (49, 52)
    assert (g.pre_encode_cache_frames, g.drop_extra_pre_encoded) == (5, 2)
    assert (g.encoder_frames, g.sampling_frames) == (13, 4)
    assert (g.last_channel_cache_size, g.last_time_cache_size) == (78, 8)
    assert g.num_raw_samples == 8561


def test_wrappers_match_the_real_streaming_path(model, wave, reference):
    """Gate 2: the split graphs, in PyTorch, against ``infer_fastconformer_streaming``."""
    specs, geometry = build_graph_specs(model)
    got = stream_waveform(TorchBackend(specs), geometry, wave)
    _assert_close(got, reference, WRAPPER_TOLERANCE)


@pytest.mark.parametrize(
    "backend_class",
    [
        pytest.param(LiteRtBackend, id="litert"),
        pytest.param(OnnxBackend, id="onnx"),
    ],
)
def test_fp32_bundle_matches_the_real_streaming_path(
    artifacts, wave, reference, backend_class
):
    """Gates 1 and 4: from raw PCM, through all four converted fp32 graphs."""
    got = stream_waveform(backend_class(artifacts, "fp32"), _geometry(artifacts), wave)
    _assert_close(got, reference, FP32_TOLERANCE)


@pytest.mark.parametrize(
    "backend_class",
    [
        pytest.param(LiteRtBackend, id="litert"),
        pytest.param(OnnxBackend, id="onnx"),
    ],
)
def test_int8_bundle_runs_with_the_right_shapes(artifacts, wave, reference, backend_class):
    """Random weights say nothing about int8 accuracy, so only check that it runs."""
    got = stream_waveform(backend_class(artifacts, "int8"), _geometry(artifacts), wave)
    assert set(got) == set(reference)
    for level in reference:
        assert got[level].shape == reference[level].shape
        assert np.isfinite(got[level]).all()


def test_pre_encoder_and_mel_stay_fp32(artifacts):
    """Gate 7: only ``encoder_core`` differs between bundles, and it is really quantized."""
    manifest = json.loads((artifacts / MANIFEST_NAME).read_text())
    for fmt, bundles in manifest["bundles"].items():
        fp32, int8 = bundles["fp32"], bundles["int8"]
        assert set(fp32) == {"mel", "pre_encode_first", "pre_encode_next", "encoder_core"}
        for graph in ("mel", "pre_encode_first", "pre_encode_next"):
            assert fp32[graph] == int8[graph], (fmt, graph)
        assert fp32["encoder_core"] != int8["encoder_core"]

    reports = manifest["reports"]
    for graph in ("mel", "pre_encode_first", "pre_encode_next"):
        assert "int8" not in reports["litert"][graph]["tensor_types"]
        assert "INT8" not in reports["onnx"][graph]["initializer_types"]
        assert "ConvInteger" not in reports["onnx"][graph]["ops"]

    assert reports["litert"]["encoder_core_int8"]["tensor_types"].get("int8", 0) > 0
    assert (
        reports["litert"]["encoder_core_int8"]["bytes"]
        < reports["litert"]["encoder_core"]["bytes"]
    )
    onnx_int8 = reports["onnx"]["encoder_core_int8"]["ops"]
    assert "ConvInteger" in onnx_int8 or "MatMulInteger" in onnx_int8


def test_encoder_core_holds_no_pre_encode_weights(model):
    """What keeps the pre-encoder out of quantization: the graph doesn't contain it."""
    specs, _ = build_graph_specs(model)
    core = next(spec for spec in specs if spec.name == "encoder_core")
    names = [name for name, _ in core.module.named_parameters()]
    assert names
    assert not [name for name in names if "pre_encode" in name]


def test_no_flex_ops(artifacts):
    """Gate 6: a Flex op would need the TensorFlow runtime on the phone."""
    manifest = json.loads((artifacts / MANIFEST_NAME).read_text())
    for graph, report in manifest["reports"]["litert"].items():
        assert report["flex_ops"] == [], graph


def test_misaligned_offset_still_raises_outside_export(model):
    """Gate 3: the export guard must not weaken the runtime check."""
    encoder = model.encoder
    with pytest.raises(ValueError, match="multiple of `chunk_size`"):
        encoder._create_masks(
            att_context_size=[78, 12],
            padding_length=torch.tensor([91]),
            max_audio_length=91,
            offset=torch.tensor([5]),
            device="cpu",
        )


def test_max_audio_length_guard_still_raises(model):
    """The deliberate strict guard stays active: callers must feed full-width chunks."""
    with pytest.raises(ValueError, match="max_audio_length"):
        model.encoder._create_masks(
            att_context_size=[78, 12],
            padding_length=torch.tensor([80]),
            max_audio_length=80,
            offset=torch.tensor([0]),
            device="cpu",
        )


def test_constant_lookahead_is_rejected():
    """C>0 changes the last chunk's output width, which the static graphs can't serve."""
    config = FastConformerCacheAwareMultilevelCTCConfig(
        level_to_vocab_size=LEVELS,
        level_to_loss_weight={"phonemes": 0.6},
        n_layers=1,
        d_model=64,
        n_heads=4,
        subsampling_conv_channels=32,
        att_context_size=[78, 12, 5],
    )
    model = FastConformerCacheAwareMultilevelCTC(config).eval()
    with pytest.raises(ValueError, match="constant_lookahead_delay"):
        build_graph_specs(model)
