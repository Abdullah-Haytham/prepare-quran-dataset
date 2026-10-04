"""Find where NaN loss / grad_norm comes from.

Loads the model, data and collator exactly like train.py and checks, step by step:
weights -> input features -> logits -> per-level CTC loss (fp32 and bf16) -> gradients.

Usage (on the GPU machine, while no training is running):
    uv run python lightning/debug_nan.py --config configs/train/offline/train_config_w2v2bert_384_lightning.yml
"""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from datasets import load_dataset
from quran_transcript import MoshafAttributes

import train as T
from prepare_quran_dataset.modeling.multi_level_tokenizer import MultiLevelTokenizer
from prepare_quran_dataset.modeling.vocab import PAD_TOKEN_IDX


def finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def per_level_losses(model, batch):
    """Each level's weighted CTC loss separately (forward sums over the given levels)."""
    out = {}
    for level, labels in batch["labels"].items():
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        inputs["labels"] = {level: labels}
        out[level] = model(**inputs).loss.item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--moshaf", default=None, help="moshaf id to sample from (default: first train id)")
    p.add_argument("--num-batches", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    T.register_model()
    cfg = T.TrainConfig.from_yaml(args.config)
    T.download_gdrive_files(cfg.gdrive_files)
    bs = args.batch_size or cfg.per_device_train_batch_size
    device = torch.device("cuda")

    with open("./vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    level_to_vocab_size = {l: len(v) for l, v in vocab.items()}
    processor, config, model = T.build_model_components(
        cfg, level_to_vocab_size, PAD_TOKEN_IDX, ignore_mismatched_sizes=cfg.ignore_mismatched_sizes
    )
    model.to(device).train()
    print("ctc_zero_infinity:", model.config.ctc_zero_infinity)
    print("loss weights:", model.config.level_to_loss_weight)

    # 1) weights
    bad = [n for n, w in model.named_parameters() if not finite(w)]
    print(f"\n[1] non-finite weights: {len(bad)}", bad[:10])

    # data: one moshaf, longest clips first (the hardest case for CTC lengths / memory)
    cfg.train_moshaf_ids = [args.moshaf or cfg.train_moshaf_ids[0]]
    ds = T.prepare_dataset(cfg, processor, None)["train"]
    order = np.argsort(-np.array(ds["duration_seconds"]))
    longest = ds.select(order[: bs * args.num_batches // 2])
    rand = ds.shuffle(seed=0).select(range(bs * args.num_batches // 2))

    meta = load_dataset("obadx/mualem-recitations-annotated", name="moshaf_metadata", split="train")
    T.moshaf_id_to_moshaf_dict = {ex["id"]: ex for ex in meta}  # used by prepare_special_moshaf_ways
    collator = T.DataCollatorCTCWithPadding(
        processor=processor,
        multi_level_tokenizer=MultiLevelTokenizer("./"),
        moshaf_id_to_moshaf_attr={ex["id"]: MoshafAttributes(**ex) for ex in meta},
        augment=T.Augment(augment_prob=cfg.augment_prob, seed=cfg.seed),
        special_moshaf_id_to_seg_to_moshaf_attr=T.prepare_special_moshaf_ways(T.moshaf_id_to_moshaf_dict),
        architecture=cfg.architecture,
    )

    for name, subset in [("longest", longest), ("random", rand)]:
        for b in range(0, len(subset), bs):
            rows = [subset[i] for i in range(b, min(b + bs, len(subset)))]
            batch = collator(rows)
            batch = {
                k: ({l: t.to(device) for l, t in v.items()} if isinstance(v, dict) else v.to(device))
                for k, v in batch.items()
            }
            feats = batch["input_features"]
            in_len = model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1))
            print(f"\n=== {name} batch {b // bs}: features {tuple(feats.shape)}, "
                  f"durations {min(r['duration_seconds'] for r in rows):.1f}-{max(r['duration_seconds'] for r in rows):.1f}s")
            # 2) inputs (augmentation can produce NaN audio)
            print("[2] features finite:", finite(feats))
            # 3) CTC feasibility: target must fit in the output frames
            for level, lab in batch["labels"].items():
                tgt = (lab >= 0).sum(-1)
                n_bad = int((tgt > in_len).sum())
                if n_bad:
                    print(f"[3] {level}: {n_bad} samples with target_len > frames "
                          f"(max target {int(tgt.max())}, min frames {int(in_len.min())}) -> inf CTC loss")

            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                out32 = model(**batch)
            print("[4] fp32 logits finite:", all(finite(v) for v in out32.logits.values()),
                  "| fp32 loss:", out32.loss.item())
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                print("[5] bf16 per-level loss:", {k: round(v, 3) for k, v in per_level_losses(model, batch).items()})

            # 6) backward in bf16 like training
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            bad_grads = [n for n, w in model.named_parameters() if w.grad is not None and not finite(w.grad)]
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            print(f"[6] bf16 loss {loss.item():.3f}, grad_norm {gnorm.item():.3f}, "
                  f"non-finite grads: {len(bad_grads)}", bad_grads[:5])


if __name__ == "__main__":
    main()
