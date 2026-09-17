"""Minimal Gradio app for offline FastConformer multi-level CTC inference.

Runs the cache-aware FastConformer model from a local checkpoint and shows:
    1. The CTC-decoded phonemes (``phonemes`` level only).
    2. The raw argmax ids over time (before CTC decoding).
    3. The ratio of non-zero ids to the total id count (before CTC decoding).
"""

import gradio as gr
import torch
from librosa.core import load

from prepare_quran_dataset.modeling_fastconformer_cache_aware import (
    FastConformerCacheAwareMultilevelCTC,
)
from prepare_quran_dataset.modeling_fastconformer_cache_aware.vocab import (
    PAD_TOKEN_IDX,
)
from train_streaming import ctc_decode, load_vocab

CKPT_DIR = "results-fastconformer-base-v1/checkpoint-2180265"
VOCAB_PATH = "vocab_streaming/vocab.json"
SAMPLE_RATE = 16000

TYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PHONEME_ID_TO_TOKEN = {
    idx: token for token, idx in load_vocab(VOCAB_PATH)[0]["phonemes"].items()
}

_model = FastConformerCacheAwareMultilevelCTC.from_pretrained(CKPT_DIR)
_model.to(device=DEVICE)
_model.processor.to(DEVICE)
_model.eval()


def transcribe(audio_path: str) -> tuple[str, list[int], float]:
    wav, _ = load(audio_path, sr=SAMPLE_RATE, mono=True)

    wav_tensor = torch.tensor([wav], dtype=torch.float32, device=DEVICE)
    wav_lengths = torch.tensor([len(wav)], dtype=torch.long, device=DEVICE)

    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=TYPE):
        out = _model(raw_audio=wav_tensor, audio_length=wav_lengths)

    raw_ids = torch.argmax(out.logits["phonemes"].float(), dim=-1).squeeze(0)
    raw_ids_np = raw_ids.cpu().numpy()

    ratio = float((raw_ids_np != 0).sum()) / raw_ids_np.size

    decoded_ids = ctc_decode(raw_ids_np[None, ...], blank_id=PAD_TOKEN_IDX)[0]
    decoded_ids = [int(t) for t in decoded_ids if int(t) not in (PAD_TOKEN_IDX, 1)]
    phonemes = "".join(_PHONEME_ID_TO_TOKEN[t] for t in decoded_ids)

    return phonemes, [int(x) for x in raw_ids_np], ratio


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Audio"),
    outputs=[
        gr.Textbox(label="Phonemes (CTC decoded)"),
        gr.Textbox(label="Raw ids (before CTC decoding)"),
        gr.Number(label="Non-zero ratio (before CTC decoding)"),
    ],
    title="FastConformer Cache-Aware CTC",
    description=(
        "Offline inference with the checkpoint at "
        f"`{CKPT_DIR}` (dtype={TYPE}, device={DEVICE})."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
