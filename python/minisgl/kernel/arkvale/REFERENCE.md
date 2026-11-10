# ArkVale reference and scope

Reference: [pku-liang/ArkVale](https://github.com/pku-liang/ArkVale), commit
`0c3b45ebe7ecc8c81ff4bc0f9c68c04a0866de88` (inspected during integration).

This is an independent implementation of the published algorithm for
Mini-SGLang, not a vendored copy of the upstream package. No upstream CUDA,
FlashInfer submodule, or other third-party source is included. The inspected
upstream checkout did not contain a LICENSE file; do not assume permission to
copy its source under this repository's license.

Behavioral references:

- `source/arkvale/infer_state.py`, `_summarize_keys`: per-dimension center is
  `(max + min) / 2`; radius is mean absolute distance from that center. Store
  upper/lower endpoints, **not** ordinary min/max bounds or mean landmarks.
- `source/arkvale_cpp/src/estimate.cu`: sum `max(q * upper, q * lower)` over
  dimensions and average over query heads, with GQA query-to-KV-head mapping.
- `KvCache` and `InferState.select_topk/recall`: protected sink/window pages,
  query-selected pages, bounded resident KV and recall from CPU backing.

Adaptations/limitations:

- Triton kernels implement digests, batched scoring/selection, FIFO residency
  bookkeeping, mapped-host backup/recall, and compact resident-page gathering.
  CUDA SDPA performs exact prefill; the same FlashAttention KV-cache interface
  as ShadowKV performs batched decode. `gpu_decode=false` retains the previous
  synchronous Python/PyTorch-top-k reference path for ablations.
  Digest reductions and score accumulation use FP32. Digests retain model dtype.
- One page selection shared by all KV heads (`n_groups=1` upstream).
- `page_budget` is total resident pages per request per layer; `page_topk`
  includes protected pages. Cache hits and remaining resident pages are retained;
  decode attends **all resident pages**, like upstream's default
  `use_sparse_attn=False`. Unselected pages are evicted only to satisfy misses.
- A write-through pinned CPU mirror makes every generated token recallable.
  The accelerated path uses nonblocking prefill backup and GPU-driven mapped-host
  decode backup/recall on a transfer stream. Backup can overlap engine scoring;
  recall waits for selection, and attention waits for recall. No host round-trip
  for miss descriptors and no cross-layer predictive prefetch are implemented.
  This is not a claim of upstream throughput parity.
- Equal scores use lower-logical-page-first selection on the GPU. The reference's
  `torch.topk` tie ordering is unspecified, so tied selections can differ. FIFO
  insertion order, protected-first admission, retained hits, and attention over
  all resident pages are otherwise preserved. GPU bookkeeping supports at most
  32768 logical pages/request; use larger pages or the reference path beyond that.
- CPU space is preallocated at maximum sequence length per configured request
  slot/layer. GPU resident pages, digests, and batched gather workspaces are also
  preallocated. Completion/abort releases ownership, not storage; slot reuse
  resets lengths and page mappings. Startup checks cover CPU backing and GPU
  resident/digest/gather storage. Model activations, attention temporaries and
  prefill workspaces still need additional headroom.
- Overlap scheduling uses the same admission stream dependency as ShadowKV.
  Transfer work waits for its engine producers; the engine joins recall before
  updating/gathering resident pages. Release waits for GPU counter readiness,
  then changes host ownership without clearing buffers. Reused-slot writes
  follow preceding engine work, which includes those transfer joins. All buffer
  owners must remain alive until outstanding work completes (normal scheduler
  shutdown synchronizes the device).
- Scheduler pages stay at one token for accounting; ArkVale's internal page size
  is independent. `--num-pages` controls logical scheduler capacity only.
- Prefill is exact causal attention; only decode is approximated when sequences
  exceed the resident budget. No quantization, graph capture, prefix reuse,
  chunked prefill, TP, per-layer budgets, or cross-layer KV prefetch is implemented.

Tests: `tests/kernel/test_arkvale.py` checks digest/score references, GQA,
partial-page gather, full-budget equivalence, forced recall, bounded residency,
independent request/layer state and cleanup/reuse. The API smoke test is
`tests/arkvale/smoke_server.py`.
