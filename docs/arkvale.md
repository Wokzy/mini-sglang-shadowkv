# ArkVale: method and implementation notes

ArkVale bounds GPU KV-cache residency for long-context autoregressive inference.
It divides each request's KV history into pages. A pinned CPU backing store keeps
the complete history, while a per-request, per-layer GPU cache holds a limited
number of pages. At decode time, a query scores page summaries; the cache protects
configured sink and recent-window pages, retains resident hits, recalls selected
misses, and evicts pages only when space is needed. Attention reads all resident
pages. Prefill remains exact; decode is approximate when the history exceeds GPU
residency.

## Method

For each page, ArkVale stores key summaries formed from per-dimension extrema and
mean absolute radius. A query scores each page by the summed upper bound of its
query-key products. Top-scoring pages plus protected sink/window pages determine
the pages to retain or recall. Selection is shared across KV heads. Values and
keys for evicted pages remain in CPU backing, making them recallable.

This repository implements the method independently in Triton, using
[pku-liang/ArkVale](https://github.com/pku-liang/ArkVale) commit
`0c3b45ebe7ecc8c81ff4bc0f9c68c04a0866de88` as a behavioral reference; no upstream
source is vendored or upstream extension required. The inspected upstream checkout
did not contain a LICENSE file; do not assume permission to copy its source under
this repository's license. Digests use FP32 reductions and model-dtype storage;
page scores use FP32 accumulation.

## Serving integration

The existing Mini-SGLang scheduler admits pending prompts one at a time, gives
prefill priority, and dynamically batches all runnable decode requests. ArkVale
uses the same scheduler policy as the other attention backends. It requires full,
unchunked prefill, no prefix reuse, scheduler page size 1, tensor-parallel size 1,
FP16/BF16, and CUDA graphs disabled. Overlap scheduling is supported.

Request state is allocated in reusable scheduler slots. KV backing, bounded GPU
resident pages, digests and gathered-KV workspace are preallocated. GPU decode
uses batched Triton scoring, page selection and residency bookkeeping, followed
by batched FlashAttention. A transfer stream backs up new KV to mapped pinned
host memory and recalls selected pages without copying miss lists through the
CPU. Explicit stream dependencies order backup, recall, attention, completion
and slot reuse. Cross-layer predictive prefetch is not implemented.

Set `gpu_decode: false` in `arkvale_config.json` to use the retained synchronous
reference integration. GPU bookkeeping currently supports up to 32768 logical
pages per request. Equal page scores use deterministic lower-page-ID preference;
the reference PyTorch top-k tie ordering is unspecified. Performance depends on
model, context length, residency budget and GPU/host interconnect; the accelerated
path is not a claim of upstream ArkVale throughput parity.

Runtime configuration is in `arkvale_config.json`. Common server invocation:

```bash
python -m minisgl \
    --model-path Qwen/Qwen3-0.6B \
    --additional-config-path arkvale_config.json \
    --attention-backend arkvale \
    --cache-type naive --page-size 1 --graph 0 --tp-size 1 \
    --max-running-requests 2 \
    --max-seq-len-override 4096 --max-prefill-length 4096
```

`page_size` and `page_budget` control ArkVale's internal page geometry and
per-request/per-layer GPU residency. They are distinct from scheduler `--page-size`
and logical `--num-pages` capacity. `page_topk` includes protected pages;
`max_cpu_bytes` bounds the preallocated host backing. Keep memory headroom for
the model, activations and attention workspaces.
