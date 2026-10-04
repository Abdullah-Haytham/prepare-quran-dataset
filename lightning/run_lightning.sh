#!/usr/bin/env bash
# Train on a Lightning AI Studio.
#
# One-time setup in the Studio terminal:
#   git clone -b saudicomp-lightning https://github.com/abdullah-haytham/prepare-quran-dataset.git
#   cd prepare-quran-dataset
#   uv sync --all-extras
#   printf 'WANDB_API_KEY=...\nHUGGINGFACE_TOKEN=...\n' > .env
#
# Then (re-run the same command to resume from the last checkpoint):
#   bash lightning/run_lightning.sh [config] [extra train.py args]   # default adds --push-to-hub
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/train/offline/train_config_w2v2bert_384_lightning.yml}"
shift || true
EXTRA_ARGS=("$@")
[ ${#EXTRA_ARGS[@]} -eq 0 ] && EXTRA_ARGS=(--push-to-hub)

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
# batches have variable lengths: avoid allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
echo "CPUs: $(nproc), GPUs: $(nvidia-smi -L | wc -l)" | tee -a "$LOG_DIR/train.log"
df -h . | tee -a "$LOG_DIR/train.log"

# Download Drive files + all datasets once (parquet files deleted after each moshaf is
# prepared, so peak disk is ~the prepared data + one moshaf)
uv run python train.py --config "$CONFIG" --prepare-data-only --free-download-cache \
  2>&1 | tee -a "$LOG_DIR/train.log"

NUM_GPUS="$(nvidia-smi -L | wc -l)"
MULTI_GPU=()
[ "$NUM_GPUS" -gt 1 ] && MULTI_GPU=(--multi_gpu)

uv run accelerate launch "${MULTI_GPU[@]}" --num_processes "$NUM_GPUS" --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  train.py --config "$CONFIG" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"
