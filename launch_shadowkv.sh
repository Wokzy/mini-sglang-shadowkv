#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/mnt/LLM
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
export CUDA_HOME="$PWD/.venv/cuda"
export PATH="$CUDA_HOME/bin:$PWD/.venv/bin:$PATH"
export CPATH="$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name include | paste -sd:)${CPATH:+:$CPATH}"
exec .venv/bin/python -m minisgl \
    --model-path Qwen/Qwen3-0.6B \
    --additional-config-path additional_config.json \
    --memory-ratio 0.2 \
    --num-pages 10000000 \
    --page-size 1 \
    --max-seq-len-override 131072 \
    --max-prefill-length 131072 \
    --tp-size 1 \
    --graph 0 \
    --attention-backend fa,fa \
    --max-running-requests 2 \
    --cache-type naive \
    --host localhost \
    --port 6888
