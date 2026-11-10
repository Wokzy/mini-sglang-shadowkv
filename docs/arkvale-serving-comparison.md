# Matched ArkVale / ShadowKV serving

The serving/allocation alignment below remains in place. The current default
also enables GPU bookkeeping and asynchronous transfers; see
[the GPU-path report](arkvale-gpu-bookkeeping.md). Measurements at the end of
this document are historical results from the preceding synchronous integration,
now available with `arkvale_config.gpu_decode=false`.

## Launch

Run each mode separately with the same environment and workload:

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL=Qwen/Qwen3-0.6B
export MAX_SEQ_LEN=4096 MAX_RUNNING_REQUESTS=2
export MINISGL_DISABLE_OVERLAP_SCHEDULING=0
PYTHON=.venv/bin/python bash launch_comparison.sh arkvale
# Stop that server, then:
PYTHON=.venv/bin/python bash launch_comparison.sh shadowkv
```

Set up `CUDA_HOME`, `PATH`, and any CUDA include paths required by the existing
installation first. The launcher does not select a GPU or hard-code local paths.
It uses the same BF16 dtype, TP=1, graph=0, naive cache, prefill budget, logical
scheduler capacity, and overlap setting in both modes. Logical token capacity
defaults to `MAX_SEQ_LEN * MAX_RUNNING_REQUESTS`; `NUM_PAGES` overrides it.

Algorithm settings are separate: `ARKVALE_CONFIG` defaults to
`arkvale_config.json`; `SHADOWKV_CONFIG` defaults to `additional_config.json`.
Those JSONs are **not** accuracy-, GPU-memory-, or transfer-matched. In particular,
the ShadowKV default quantizes landmarks and uses a ratio sparse budget. Record
both JSONs along with prompts, arrival times, output lengths and sampling options
when reporting performance. Set overlap to `1` (disabled) in both modes for a
matched ablation. CUDA graphs remain unsupported in ArkVale.

## What is aligned

- Both use the existing single-request, prefill-first admission policy and dynamic
  decode batches. Neither mixes prefill and decode in a single batch.
- ArkVale backing, resident KV, digests and gathered decode workspaces are now
  allocated at pool initialization, indexed by scheduler `table_idx`, and reused.
  UID ownership maps prevent accidentally binding two requests to one live slot.
- Completion/abort discards live mappings and lengths, not the large allocations.
  `host_bytes` retains its meaning as *live* backing bytes; `allocated_host_bytes`
  reports the persistent CPU allocation.
- Both use the same FlashAttention KV-cache interface for batched decode, with
  explicit active-slot indices and per-request valid lengths. Unused workspace
  contents are excluded by those lengths.
- Both offloading backends apply the scheduler-to-engine stream dependency before
  admission, not just attention metadata preparation: admission also writes the
  token pool and must not race a prior forward's token write after slot reuse.
  A slot can be released on the host while a previous forward is
  finishing because release does not modify its buffers; subsequent slot writes
  run after the previous forward on the engine stream.
- Shared scheduler result handling uses committed host tokens for output-length
  termination. The speculative device cursor is one step ahead under overlap;
  using it truncated output by one token. Results already finished/aborted are
  ignored instead of being emitted or touching reused cache slots.

## Deliberate remaining differences

ArkVale still uses bounded resident pages, retains hits and recalls misses. Its
selection is shared across KV heads and decode attends **all resident pages**,
not just this step's top-k. ShadowKV gathers prefix/selected chunks/suffix anew
and selects by KV head. Equal token/page budgets do not imply equal accuracy or
traffic.

The retained ArkVale reference path (`gpu_decode=false`) uses Python per-request
bookkeeping, GPU top-k results transferred to the host, and synchronous copies.
The default GPU path removes those host round-trips and performs backup/recall on
a transfer stream. ShadowKV also has batch-aware selection/gather kernels and
optional mapped CPU backing/quantization. Neither this ArkVale path nor scheduler
overlap implements cross-layer predictive prefetch.

ArkVale uses SDPA for exact prefill, while ShadowKV uses its FlashAttention
wrapper. Both now use that wrapper for decode. On FA4 the existing shared wrapper
may gather cache slots into query order with `index_select`; its temporary
allocations and attention outputs are not persistent KV storage.

ArkVale allocates exactly the configured request slots; ShadowKV additionally
allocates one dummy slot for its graph-capable infrastructure. ArkVale's gathered
workspace is resident-budget-sized; ShadowKV's is maximum-sequence-sized.
ArkVale CPU backing is pinned in separate slot/layer allocations, whereas
ShadowKV registers its mapped backing in chunks. The residency bounds are kept
rather than allocating full GPU KV merely to equalize buffer shapes. Report
actual host/GPU allocations and traffic, not just `--memory-ratio`.

## Validation performed

No test files were added or modified. Temporary validation scripts and artifacts
were written outside the repository, under `/tmp/arkvale-serving-validation`.
Validation used a B200 and Qwen3-0.6B, with BF16 end-to-end serving.

- Existing tests: **46 passed, 1 deselected**, running:
  ```bash
  python -m pytest -q tests/kernel/test_arkvale.py tests/arkvale \
      tests/core/test_cache_allocate.py tests/misc/test_wildchat_traffic.py \
      -k 'not test_overlap_rejected'
  ```
  An unfiltered ArkVale run gave 37 passes and one failure: the unchanged legacy
  `test_overlap_rejected` requires the restriction intentionally removed here.
- Temporary numerical checks compared against the pre-change pool implementation:
  page mappings and packed KV matched exactly through forced eviction/recall.
  Batched attention matched per-request SDPA within BF16 tolerance (absolute
  0.015, relative 0.02); full-residency attention also matched full exact KV.
  Checks covered GQA, multiple layers, unequal lengths, partial pages, noncontiguous
  slots, reordered/shrinking batches and three rounds of slot reuse. Persistent
  CPU/GPU buffer pointers remained unchanged.
- Actual scheduler runs exercised five requests with different output lengths,
  repeated generation, staggered arrivals, abort with an outstanding forward,
  and immediate slot reuse. The final overlap checks deliberately delayed the
  engine stream before token-pool writes to exercise abort/reuse ordering.
  Cache integrity and slot ownership were checked on drain. ArkVale overlap/non-overlap greedy token IDs were identical, and the
  requested output lengths were returned exactly. ShadowKV passed the same
  overlap lifecycle checks.
- The matched ArkVale launcher passed the existing HTTP smoke client with overlap
  enabled: sequential/concurrent requests, natural EOS, streaming cancellation,
  and reuse after cancellation. Server logs confirmed nonzero score steps,
  evictions and recalls, and return to idle with cache integrity checks.

### Small serving sanity measurement (not a benchmark conclusion)

Five identical prompts per mode (roughly 60–110 prompt tokens), with output token
counts `[9, 17, 25, 13, 19]`, were run on the same GPU. Settings: BF16, max sequence
512, three live requests, logical capacity 1536, no graphs/prefix cache/chunking.
One warmup was excluded; the following are two subsequent elapsed times:

| Mode | Overlap | Elapsed seconds | CPU KV backing | End-of-run PyTorch GPU allocated / peak |
| --- | --- | --- | --- | --- |
| ArkVale | on | 6.316, 8.263 | 168 MiB | 1217.91 / 1220.69 MiB |
| ArkVale | off | 8.263, 8.164 | 168 MiB | 1217.91 / 1220.68 MiB |
| ShadowKV | on | 0.743, 0.715 | 224 MiB | 1201.10 / 1209.15 MiB |

ArkVale used page size 4, resident budget 8, top-k 6, one sink page and two window
pages. ShadowKV used chunk size 4, eight sparse chunks, no protected prefix/suffix,
no quantization and CPU offloading. These are **not accuracy-equivalent** settings.
GPU figures include model tensors and other PyTorch allocations; peak includes
warmup and lifecycle validation, not solely the timed requests. They exclude CUDA
context/library allocations. CPU figures describe KV storage, not process RSS.

ArkVale release counters reported 2,554 page loads per five-request run, equivalent
to 39.91 MiB of logical H2D KV payload including initial fills (BF16, 8 KV heads,
head dimension 128). This is derived payload, not a measured PCIe/NVLink counter.
ShadowKV transfer traffic was not instrumented in this sanity run.

The small sample and shared-machine conditions do not establish stable throughput
ratios. They do show that matching allocation and attention batching alone does
not eliminate ArkVale's remaining synchronous transfer and Python selection
costs; overlap is not necessarily a speedup for that path.
