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

- Triton kernels implement digests, scores, and compact resident-page gathering;
  PyTorch CUDA top-k performs selection and CUDA SDPA performs attention.
  Digest reductions and score accumulation use FP32. Digests retain model dtype.
- One page selection shared by all KV heads (`n_groups=1` upstream).
- `page_budget` is total resident pages per request per layer; `page_topk`
  includes protected pages. Cache hits and remaining resident pages are retained;
  decode attends **all resident pages**, like upstream's default
  `use_sparse_attn=False`. Unselected pages are evicted only to satisfy misses.
- A write-through pinned CPU mirror makes every previously generated token
  recallable. Transfers are synchronous on the engine stream, without upstream
  cross-layer prefetch or backup streams. This is a correctness-first integration,
  not a claim of upstream throughput parity.
- CPU space is reserved at maximum sequence length per live request/layer;
  buffers are allocated lazily and released on completion/abort. Startup checks
  cover worst-case CPU backing and GPU resident/digest storage. Model activation
  and prefill workspace memory still need additional headroom.
- Scheduler pages stay at one token for accounting; ArkVale's internal page size
  is independent. `--num-pages` controls logical scheduler capacity only.
- Prefill is exact causal attention; only decode is approximated when sequences
  exceed the resident budget. No quantization, graph capture, prefix reuse,
  chunked prefill, TP, per-layer budgets, or asynchronous overlap is implemented.

Tests: `tests/kernel/test_arkvale.py` checks digest/score references, GQA,
partial-page gather, full-budget equivalence, forced recall, bounded residency,
independent request/layer state and cleanup/reuse. The API smoke test is
`tests/arkvale/smoke_server.py`.
