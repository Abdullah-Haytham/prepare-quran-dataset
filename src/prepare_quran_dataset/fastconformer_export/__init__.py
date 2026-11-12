"""Export the cache-aware FastConformer and its log-mel front-end to LiteRT and ONNX.

Inference is split into four graphs (see ``docs/fastconformer_export.md``)::

    raw PCM -> mel -> pre_encode_first / pre_encode_next -> encoder_core -> logits

``mel_frontend`` and ``streaming_driver`` don't import NeMo; the other modules do.  Import
from the submodules directly, so that this package stays importable without NeMo.
"""
