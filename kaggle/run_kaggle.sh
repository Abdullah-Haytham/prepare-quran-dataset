#!/usr/bin/env bash
# Train on Kaggle with 2x T4 GPUs (DDP via torchrun).
#
# Notebook settings: Accelerator = "GPU T4 x2", Internet = On.
# Add Kaggle Secrets: WANDB_API_KEY, HF_TOKEN
#
# Notebook cells:
#   !git clone -b saudicomp <repo-url> /kaggle/working/prepare-quran-dataset
#   %cd /kaggle/working/prepare-quran-dataset
#
#   import os
#   from kaggle_secrets import UserSecretsClient
#   secrets = UserSecretsClient()
#   for k in ["WANDB_API_KEY", "HUGGINGFACE_TOKEN"]:
#       os.environ[k] = secrets.get_secret(k)
#
#   !bash kaggle/run_kaggle.sh
#   # or with a config hosted on Google Drive / extra train.py args:
#   !bash kaggle/run_kaggle.sh "https://drive.google.com/file/d/<ID>/view" --push-to-hub
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/train/offline/train_config_w2v2_384_kaggle.yml}"
shift || true

pip install -q -r kaggle/requirements.txt

# pyproject requires python>=3.14, so use the sources directly instead of `pip install -e .`
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
# keep the (large) datasets cache off /kaggle/working (20GB limit)
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Launching on ${NUM_GPUS} GPU(s) with config: ${CONFIG}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" train.py --config "${CONFIG}" "$@"
