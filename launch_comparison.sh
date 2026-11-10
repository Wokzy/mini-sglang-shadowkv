#!/usr/bin/env bash
# Matched serving settings; algorithm-specific options remain in separate JSONs.
# Usage: CUDA_VISIBLE_DEVICES=0 bash launch_comparison.sh arkvale [extra CLI args]
#        CUDA_VISIBLE_DEVICES=0 bash launch_comparison.sh shadowkv [extra CLI args]
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:?Usage: $0 arkvale|shadowkv [extra CLI args]}"
shift
case "$mode" in
    arkvale) backend=arkvale; config="${ARKVALE_CONFIG:-arkvale_config.json}" ;;
    shadowkv) backend=fa,fa; config="${SHADOWKV_CONFIG:-additional_config.json}" ;;
    *) echo "Unknown mode: $mode (expected arkvale or shadowkv)" >&2; exit 2 ;;
esac
max_seq_len="${MAX_SEQ_LEN:-4096}"
max_requests="${MAX_RUNNING_REQUESTS:-2}"
# Both backends support the same overlap setting; use 1 for a matched ablation.
export MINISGL_DISABLE_OVERLAP_SCHEDULING="${MINISGL_DISABLE_OVERLAP_SCHEDULING:-0}"
exec "${PYTHON:-python}" -m minisgl \
    --model-path "${MODEL:-Qwen/Qwen3-0.6B}" \
    --additional-config-path "$config" \
    --attention-backend "$backend" \
    --dtype bfloat16 \
    --cache-type naive --page-size 1 --graph 0 --tp-size 1 \
    --max-seq-len-override "$max_seq_len" --max-prefill-length "$max_seq_len" \
    --max-running-requests "$max_requests" \
    --num-pages "${NUM_PAGES:-$((max_seq_len * max_requests))}" \
    --memory-ratio "${MEMORY_RATIO:-0.5}" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-1919}" \
    "$@"
