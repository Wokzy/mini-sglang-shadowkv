# ArkVale GPU bookkeeping and asynchronous transfers

The default `arkvale_config.gpu_decode=true` enables the accelerated serving path.
Set it to `false` to retain the preceding synchronous integration for ablations.
The flag includes GPU prefill initialization as well as batched decode. CUDA
graphs remain disabled; scheduler overlap is independent of transfer-stream use.

## Implementation

- `python/minisgl/kernel/arkvale/batched.py`: batched Triton scoring, top-k
  selection, resident-slot assignment, mapped-host backup/recall, current-page
  updates/digests, and resident-page gathering.
- `python/minisgl/kernel/arkvale/gpu.py`: persistent GPU metadata and transfer
  stream dependencies. Uses the repository's existing
  `minisgl.shadowkv_kernels.map_to_gpu` helper; no new compiled extension is needed.
- `cache.py` / `backend.py`: select GPU versus reference execution while keeping
  the same preallocated backing, resident cache and batched FlashAttention.

Active batch positions map to persistent scheduler request slots. Each
request/layer stores physical-slot-to-logical-page mappings, 64-bit insertion
ages/counters, and statistics on the GPU. Miss descriptors, ordered gather slots,
and valid gathered lengths remain on the GPU. There is no per-request Python
selection/eviction loop and no `.tolist()`/`.item()` of selected pages in the
accelerated forward path. Scheduler request metadata and UID ownership remain
host-managed; sequence progression is checked once per batch.

Selection preserves the reference's protected-first admission and then
query-selected admission, both in ascending logical-page order. Free physical
slots are chosen before eviction; victims follow FIFO **insertion** order.
Hits do not change insertion age. Unselected resident pages remain until a miss
needs their slot, and attention still includes **all resident pages**.

**Tie behavior:** equal scores prefer lower logical page IDs on the GPU.
`torch.topk` in the reference has unspecified tie ordering, so equally scored
selections need not match that ordering. Non-tied numerical/reference checks and
the measured model workloads matched mappings, counters and generated outputs.

The scalar `pool.prefill(..., gpu=False)` / `pool.decode()` methods retain their
reference implementation for direct callers. GPU serving's authoritative
bookkeeping is in `pool.gpu`; `LayerCache`'s Python length/mapping fields are
reference-path state, not a synchronized mirror of GPU decode state. Do not mix
scalar decode with GPU-managed prefill/decode on a live request.

## Transfer dependencies and lifetime

Decode uses this dependency chain:

```text
engine:   produce Q/K/V ───────── score + select ────────┐
                 │                                      │
transfer:        └── backup new K/V to mapped CPU ────────┴── recall misses
                                                               │
engine:                                                        └── update current page
                                                                   digest / gather / attention
```

- The transfer stream waits for Q/K/V production. Its mapped-host backup kernel
  can overlap scoring and selection on the engine stream.
- Recall on that same transfer stream waits for selection. Thus it observes both
  the new backup token and the completed miss descriptors.
- The engine joins recall before current-page updates, gathering and attention.
- Prefill uses nonblocking pinned-CPU copies on the transfer stream, overlapping
  backup with digest/initial-selection work, followed by GPU-driven initial recall.
- Completion/abort reads statistics only after a GPU event. It releases host
  ownership without clearing buffers. Admission already waits for the previous
  engine forward, whose transfer work has been joined, so immediate slot reuse
  cannot overwrite a still-running transfer's storage or metadata.
- Normal scheduler shutdown synchronizes the device before destroying resources.
  Mapped aliases retain ownership of pinned CPU buffers for the pool lifetime.

Mapped-host recall is GPU memory traffic, **not** a promise of copy-engine DMA.
It consumes SM/interconnect resources. There is no cross-layer predictive prefetch
and no claim that current-query recall can be hidden behind the attention that
needs it. The improvement removes host round-trips and serial host-driven copies;
stream separation also permits backup/selection overlap where resources allow.

Workspace tails are explicitly zeroed. Stress validation showed that NaN padding
can propagate through some FlashAttention masked arithmetic (`0 * NaN`), despite
correct valid lengths. The GPU gather therefore never leaves a poisoned tail.

GPU bookkeeping currently supports up to **32768 logical pages per request**.
Larger page tables must use a larger internal page size or `gpu_decode=false`.
Large top-k values still increase the per-request GPU selection work; this is not
a claim of an optimal radix/top-k implementation for every configuration.

## Correctness validation

No test files were added or modified. Temporary checks and raw artifacts are in
`/tmp/arkvale-gpu-validation` on the validation machine.

- Existing regression group: 46 passing tests, with the unchanged legacy
  `test_overlap_rejected` deselected because overlap is now supported.
- Exact comparisons against the synchronous pool covered initial selection,
  two-phase decode selection, FIFO order, packed KV, CPU backing contents,
  completed-page digests, and recall/eviction/score counters.
- Stress cases covered FP16 and BF16; page sizes 1, 4, 32 and 128; capacities 7
  and 8 (including non-power-of-two capacity); head dimensions 64/128; GQA;
  multiple layers; partial pages; noncontiguous/reordered/shrinking batches;
  repeated slot reuse; deliberately NaN-poisoned gather workspaces; and a
  deterministic all-equal-score case. Additional synthetic checks exercised
  append/recall/score/gather with resident offsets beyond 2^31 elements, verifying
  64-bit addressing for multi-GiB slot/layer pools.
- Batched attention matched per-request SDPA within BF16/FP16 tolerance
  (absolute 0.015, relative 0.02). Full-residency checks matched exact full KV.
- Qwen3-0.6B greedy generation matched the pre-change integration and matched
  between overlap enabled/disabled. Staggered arrivals, abort and immediate slot
  reuse passed with deliberately delayed engine token-pool writes.
- The existing HTTP smoke client passed on the accelerated server: sequential
  and concurrent requests, natural EOS, streaming cancellation, and subsequent
  slot reuse. Logs confirmed evictions, recall, and normal return to idle.

## Paired performance measurement

**Baseline:** the preceding synchronous Python bookkeeping/copy integration,
which already had slot-based allocation and batched FlashAttention. This is not
an upstream ArkVale comparison and is not a ShadowKV speed comparison.

Both modes ran in the same initialized model/pool, switching only when all
requests were drained and the GPU synchronized. The reference bypassed
`pool.gpu`; the accelerated mode enabled it. GPU/CPU allocations, model, scheduler,
attention interface and prompts were otherwise shared. Each mode/case had one
excluded full warmup and three measured runs. Mode order was reversed each
repetition. All returned token IDs and page-load/eviction/score counters matched.

Settings: B200, Qwen3-0.6B, BF16, TP=1, graphs off, overlap on, naive cache,
maximum sequence length 4096, four request slots, logical token capacity 16384.
Internal page size 32, resident budget 32, top-k 16, sink/window pages 2/2.
Each request generated exactly **8 tokens**, greedily, ignoring EOS. Prompts were
fixed tokenized archive text, repeated/truncated to the stated lengths; each
request had a distinct final token. Each batch is admitted through the ordinary
single-request-prefill scheduler.

**Important qualification:** unrelated jobs were heavily using the same B200
(and other GPUs) during these measurements. Interleaving reduces drift but does
not eliminate contention. These are observed shared-machine speedups, not claims
about isolated-GPU throughput or speedups for all workloads. Short fixed-output
runs also do not establish long-running arrival-trace performance.

| Requests | Prompt tokens/request | Reference median | GPU median | End-to-end speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 9.978 s | 0.888 s | **11.24x** |
| 4 | 512 | 33.932 s | 2.491 s | **13.62x** |
| 4 | 2048 | 62.222 s | 2.589 s | **24.04x** |

Raw elapsed seconds (three repetitions):

- 1 x 512: reference `[10.2721, 9.9779, 9.8768]`; GPU `[0.8919, 0.8876, 0.8161]`.
- 4 x 512: reference `[33.2729, 34.0527, 33.9323]`; GPU `[2.3528, 2.5662, 2.4914]`.
- 4 x 2048: reference `[64.5444, 61.4233, 62.2217]`; GPU `[2.4544, 2.5888, 2.6304]`.

The 512-token cases fit within residency and exercised backup/gather without
score-based eviction. The 2048-token case exceeded residency: each measured run
recorded 3,972 recalls (including prefill), 388 evictions and 896 score steps,
identically for both paths. Those recalls represent 496.5 MiB of logical H2D KV
payload (BF16, 8 KV heads, head dimension 128, 32 tokens/page), not measured bus
traffic. Equal traffic/counters and token IDs rule out doing less model work as
the explanation for the measured speedup.

CUDA-event forward spans also decreased, especially decode, but include host
launch gaps and transfer-stream waits. They must not be interpreted as isolated
kernel execution times. Primary results above are synchronized wall-clock times,
including prefill, decode, scheduling, sampling and request cleanup.

### Reproduce

Use the same CUDA/toolchain setup as the server, preferably on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 MINISGL_DISABLE_OVERLAP_SCHEDULING=0 \
python benchmark/offline/bench_arkvale_gpu.py \
    --model Qwen/Qwen3-0.6B \
    --cases 1:512,4:512,4:2048 --output-tokens 8 --repeats 3 \
    --output /tmp/arkvale-paired.json
```

The benchmark emits raw times, forward-stream spans, counters, output/prompt
hashes and median speedups. It rejects differing outputs or traffic counters
instead of silently reporting a potentially invalid comparison. Test overlap
ablations by setting `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` for the whole run.
