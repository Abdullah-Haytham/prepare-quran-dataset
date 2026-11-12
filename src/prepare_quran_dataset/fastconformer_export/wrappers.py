"""Static-shape ``nn.Module`` wrappers around the four graphs that get exported.

The model's own ``forward`` handles two chunk shapes and owns its cache objects, neither
of which a static graph can do, so inference is split at the pre-encode boundary:

- ``mel`` and ``pre_encode_*`` stay fp32 in every bundle, and only ``encoder_core`` gets
  quantized.  The split keeps pre-encode weights out of the graph the quantizer sees.
- ``encoder_core`` calls the real ``model.forward`` with a pass-through ``pre_encode``, so
  ``forward_internal`` remains the single source of truth for the encoder.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from prepare_quran_dataset.modeling_fastconformer_cache_aware.modeling_fastconformer_cache_aware_ctc import (
    FastConformerCache,
    FastConformerCacheAwareMultilevelCTC,
)

from .mel_frontend import MelGeometry, StreamingLogMel
from .streaming_driver import (
    CORE_INPUTS,
    CORE_STATE_OUTPUTS,
    MEL_INPUTS,
    MEL_OUTPUTS,
    PRE_ENCODE_INPUTS,
    PRE_ENCODE_OUTPUTS,
    StreamingGeometry,
)


class PassThroughPreEncode(nn.Module):
    """Stands in for ``encoder.pre_encode``: its input already went through the real one."""

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, lengths


class PreEncodeGraph(nn.Module):
    """Log-mel to subsampled features, dropping ``drop`` leading frames.

    Same as the pre-encode part of ``forward_internal``, including its length clamp.
    """

    def __init__(self, pre_encode: nn.Module, drop: int) -> None:
        super().__init__()
        self.pre_encode = pre_encode
        self.drop = drop

    def forward(
        self, log_mel: torch.Tensor, log_mel_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, length = self.pre_encode(
            x=log_mel.transpose(1, 2), lengths=log_mel_length
        )
        length = length.to(torch.int64)
        if self.drop > 0:
            features = features[:, self.drop :, :]
            length = (length - self.drop).clamp(min=0)
        return features.transpose(1, 2), length.to(torch.int32)


class EncoderCoreGraph(nn.Module):
    """Subsampled features and caches to per-level logits and the next caches."""

    def __init__(
        self, model: FastConformerCacheAwareMultilevelCTC, levels: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.model = model
        self.levels = levels

    def forward(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_length: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out = self.model(
            processed_signal=features,
            processed_length=features_length.to(torch.int64),
            cache=FastConformerCache(
                last_channel=cache_last_channel,
                last_time=cache_last_time,
                last_channel_len=cache_last_channel_length.to(torch.int64),
            ),
            # C=0: every chunk is exactly valid_out_len frames, so this has no effect.
            keep_all_outputs=False,
            # The real pre_encode ran in another graph, which already dropped its frames.
            drop_extra_pre_encoded=0,
            return_dict=True,
            selected_levels=set(self.levels),
        )
        return (
            *(out.logits[level] for level in self.levels),
            out.encoder_lengths.to(torch.int32),
            out.cache.last_channel,
            out.cache.last_time,
            out.cache.last_channel_len.to(torch.int32),
        )


@dataclass
class GraphSpec:
    """One graph to export, with sample inputs that fix its static shapes."""

    name: str
    module: nn.Module
    inputs: dict[str, torch.Tensor]
    output_names: tuple[str, ...]
    quantize: bool = False


def _pair(value) -> tuple[int, int]:
    return (value[0], value[1]) if isinstance(value, (list, tuple)) else (value, value)


def streaming_geometry(
    model: FastConformerCacheAwareMultilevelCTC,
    levels: tuple[str, ...],
    mel_geometry: MelGeometry,
) -> StreamingGeometry:
    """Read the geometry from the model, rejecting what the static graphs can't serve."""
    encoder = model.encoder
    context = list(encoder.att_context_size)
    if len(context) > 2 and context[2] > 0:
        raise ValueError(
            "Export supports constant_lookahead_delay=0 only, because with C>0 the last "
            f"chunk's output width differs. Got att_context_size={context}."
        )
    cfg = encoder.streaming_cfg
    first_chunk, chunk = _pair(cfg.chunk_size)
    _, pre_encode_cache = _pair(cfg.pre_encode_cache_size)
    _, sampling_frames = _pair(encoder.pre_encode.get_sampling_frames())
    if pre_encode_cache > chunk or chunk < first_chunk:
        raise ValueError(f"Unexpected chunking in streaming_cfg: {cfg}.")
    return StreamingGeometry(
        levels=levels,
        sample_rate=mel_geometry.sample_rate,
        n_fft=mel_geometry.n_fft,
        hop_length=mel_geometry.hop_length,
        win_length=mel_geometry.win_length,
        features=mel_geometry.features,
        num_raw_samples=mel_geometry.num_raw_samples,
        num_emphasized_samples=mel_geometry.num_emphasized_samples,
        first_chunk_frames=first_chunk,
        chunk_frames=chunk,
        pre_encode_cache_frames=pre_encode_cache,
        sampling_frames=sampling_frames,
        drop_extra_pre_encoded=cfg.drop_extra_pre_encoded,
        encoder_frames=cfg.valid_out_len,
        d_model=encoder.d_model,
        n_layers=len(encoder.layers),
        last_channel_cache_size=cfg.last_channel_cache_size,
        last_time_cache_size=encoder.conv_context_size[0],
    )


def _bake_position_table(encoder: nn.Module, input_length: int) -> None:
    """Rebuild ``pos_enc.pe`` at exactly the exported size, and check it.

    HF ``from_pretrained`` can leave this checkpoint-absent buffer as uninitialized
    memory.  The model checks at runtime, but that check can't be traced, so garbage here
    would become a constant in the graph without any error.
    """
    if hasattr(encoder.pos_enc, "pe"):
        del encoder.pos_enc.pe
    parameter = next(encoder.parameters())
    encoder.pos_enc.extend_pe(input_length, device=parameter.device, dtype=parameter.dtype)
    table = encoder.pos_enc.pe
    expected_rows = 2 * input_length - 1
    if table.size(1) != expected_rows:
        raise RuntimeError(
            f"Position table has {table.size(1)} rows, expected {expected_rows}."
        )
    centre = table[0, table.size(1) // 2]
    if not (bool((centre[0::2] == 0).all()) and bool((centre[1::2] == 1).all())):
        raise RuntimeError("Position table failed its centre-row sinusoid check.")


def build_graph_specs(
    model: FastConformerCacheAwareMultilevelCTC,
    selected_levels: set[str] | None = None,
    mel_module: nn.Module | None = None,
) -> tuple[list[GraphSpec], StreamingGeometry]:
    """Build the four graphs from a loaded model, which is left unmodified.

    Args:
        model: The model to export.
        selected_levels: Output levels to keep; ``None`` keeps all of them.
        mel_module: Replaces the default :class:`StreamingLogMel`.

    Returns:
        The specs, with ``mel`` and ``pre_encode_*`` first so that failures on the small
        graphs surface before the slow one, and the geometry for the manifest.
    """
    model = copy.deepcopy(model).eval().float()
    model.setup_streaming_params()
    # Only matters under autocast, of which export has none: everything is float32 anyway.
    # Clearing it keeps the autocast context manager out of the traced graph.
    model.encoder.fp32_pre_encode = False

    levels = tuple(
        level
        for level in model.level_to_lm_head
        if selected_levels is None or level in selected_levels
    )
    if selected_levels is not None and set(levels) != set(selected_levels):
        raise ValueError(
            f"Unknown levels {set(selected_levels) - set(levels)}; "
            f"available: {list(model.level_to_lm_head)}."
        )

    _, chunk = _pair(model.encoder.streaming_cfg.chunk_size)
    mel_geometry = MelGeometry.from_processor_kwargs(
        model.config.processor_kwargs, frames_per_call=chunk
    )
    geometry = streaming_geometry(model, levels, mel_geometry)

    first_pre_encode = PreEncodeGraph(model.encoder.pre_encode, drop=0).eval()
    next_pre_encode = PreEncodeGraph(
        model.encoder.pre_encode, drop=geometry.drop_extra_pre_encoded
    ).eval()

    core_model = copy.deepcopy(model).eval()
    core_model.encoder.pre_encode = PassThroughPreEncode()
    _bake_position_table(
        core_model.encoder, geometry.encoder_frames + geometry.last_channel_cache_size
    )

    int32_one = lambda value: torch.tensor([value], dtype=torch.int32)  # noqa: E731
    next_length = geometry.chunk_frames + geometry.pre_encode_cache_frames
    specs = [
        GraphSpec(
            name="mel",
            module=(mel_module or StreamingLogMel(mel_geometry)).eval(),
            inputs=dict(
                zip(
                    MEL_INPUTS,
                    (
                        torch.zeros(1, geometry.num_raw_samples),
                        int32_one(0),
                        int32_one(geometry.num_emphasized_samples),
                    ),
                )
            ),
            output_names=MEL_OUTPUTS,
        ),
        GraphSpec(
            name="pre_encode_first",
            module=first_pre_encode,
            inputs=dict(
                zip(
                    PRE_ENCODE_INPUTS,
                    (
                        torch.zeros(1, geometry.features, geometry.first_chunk_frames),
                        int32_one(geometry.first_chunk_frames),
                    ),
                )
            ),
            output_names=PRE_ENCODE_OUTPUTS,
        ),
        GraphSpec(
            name="pre_encode_next",
            module=next_pre_encode,
            inputs=dict(
                zip(
                    PRE_ENCODE_INPUTS,
                    (
                        torch.zeros(1, geometry.features, next_length),
                        int32_one(next_length),
                    ),
                )
            ),
            output_names=PRE_ENCODE_OUTPUTS,
        ),
        GraphSpec(
            name="encoder_core",
            module=EncoderCoreGraph(core_model, levels).eval(),
            inputs=dict(
                zip(
                    CORE_INPUTS,
                    (
                        torch.zeros(1, geometry.d_model, geometry.encoder_frames),
                        int32_one(geometry.encoder_frames),
                        torch.zeros(
                            geometry.n_layers,
                            1,
                            geometry.last_channel_cache_size,
                            geometry.d_model,
                        ),
                        torch.zeros(
                            geometry.n_layers,
                            1,
                            geometry.d_model,
                            geometry.last_time_cache_size,
                        ),
                        int32_one(geometry.last_channel_cache_size),
                    ),
                )
            ),
            output_names=(*levels, *CORE_STATE_OUTPUTS),
            quantize=True,
        ),
    ]
    return specs, geometry


class TorchBackend:
    """Runs the wrappers directly: the reference for the exported graphs."""

    def __init__(self, specs: list[GraphSpec]) -> None:
        self.specs = {spec.name: spec for spec in specs}

    @torch.no_grad()
    def run(self, graph: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        spec = self.specs[graph]
        args = [torch.from_numpy(np.asarray(inputs[name])) for name in spec.inputs]
        outputs = spec.module(*args)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        return {name: o.numpy() for name, o in zip(spec.output_names, outputs)}
