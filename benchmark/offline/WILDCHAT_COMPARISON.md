# Qwen3-0.6B: full attention, YAKV and ArkVale

## Workload and method

Reference: [bench_wildchat.py on shadowkv_dev](https://github.com/Wokzy/mini-sglang-shadowkv/blob/shadowkv_dev/benchmark/offline/bench_wildchat.py).

- Model: `Qwen/Qwen3-0.6B`, snapshot `c1899de289a04d12100db370d81485cdf75e47ca`.
- Hardware/software: NVIDIA B200, PyTorch 2.9.1+cu128, Triton 3.5.1.
- Reuse the reference's first-user-turn selection, English/Chinese restriction,
  redaction/toxicity filtering and tokenizer chat template on the first
  `allenai/WildChat-4.8M` parquet shard.
- Select the **first 10 qualifying prompts of 2,048–4,096 tokens**, not the first
  10 unrestricted prompts (those were only 10–62 tokens). No padding or synthetic
  prompt repetition. Input lengths:
  `2520, 3665, 2952, 3908, 2987, 2784, 3021, 3029, 3722, 3729`.
  Total: 32,317 input tokens; mean: 3,231.7.
- Prompt token-ID SHA256:
  `2a192c6647af224f35130cb9bd5fbc79caa2d478dfac9889b8e85e9b171791d8`.
- **128 generated tokens per prompt**, `ignore_eos=True`, temperature 0.6, seed 0.
  This differs from the reference's 4,096-token maximum/natural EOS: fixed output
  work is necessary for a close traffic comparison. All measured runs returned
  exactly 1,280 output tokens and 10 nonempty outputs.
- Identical BF16 KV, TP=1, naive cache, scheduler page size 1, maximum sequence
  length/prefill budget 4,096 and maximum running requests 10. Disable CUDA graphs
  and overlap scheduling for every mode. Full attention and YAKV use FA; ArkVale
  uses its CUDA SDPA backend. These are current integration timings, not isolated
  comparisons using identical attention kernels.
- Each mode runs in its own fresh process. Warm all 10 prompts through all 128
  output steps, then measure three repetitions, resetting RNGs and counters.
  Timed regions include prefill and decode but exclude model loading, prompt
  preparation, initial warmup/JIT compilation and writing result files.

## Matched KV offload payload

ArkVale uses `page_size=32`, `page_budget=32`, `page_topk=32`, `sink_pages=2`,
`window_pages=2`. Resident capacity is 1,024 tokens per request/layer.

YAKV uses `chunk_size=1`, absolute `sparse_budget=54`, zero prefix/suffix budgets,
`min_seqlen_to_prune=0`, 2-bit landmarks, **unquantized BF16 KV**, and CPU
offloading. Prefix/suffix budgets are disabled for this byte-budget experiment;
this is not an accuracy-equivalent configuration to ArkVale's protected pages.

The budgets deliberately do **not** have equal selected-token counts. YAKV
re-gathers selected input tokens **plus every generated token** at every decode
step. ArkVale transfers only cache misses and retains resident hits. Its initial
resident fill is also counted.

The benchmark counts exact **logical KV payload**, not physical PCIe hardware
counters. It excludes metadata, transaction/cache-line amplification, GPU L2
caching of mapped host reads, driver staging, model loading and input/output
tensor transfers. No CUDA synchronization is added to the timed loop for byte
accounting. Full attention has no KV-offload payload, not zero total PCIe traffic.

For this model a token's BF16 K+V across all layers is
`2 * 28 * 8 * 128 * 2 = 114688 bytes`. With 128 output tokens there are 127
KV-appending decode steps per request. Thus YAKV's H2D payload is

```text
114688 * 10 * (127 * sparse_budget + sum(1..127))
```

ArkVale's observed page-load count gives an ideal YAKV sparse budget of
`53.9959505062`; **54 is the closest possible integer budget**. The script can
calculate this automatically with `--match-arkvale-result`.

## Results

Median-time repetition of three warmed runs; throughput is generated tokens per
second, including prefill time. GiB means 2^30 bytes.

| Mode | Median time (s) | Output tok/s | H2D KV GiB | D2H KV GiB | Total KV GiB |
|---|---:|---:|---:|---:|---:|
| full_attention | 0.8630 | 1483.15 | 0 | 0 | 0 |
| yakv | 2.2626 | 565.72 | 16.006775 | 3.587479 | 19.594254 |
| arkvale | 23.0299 | 55.58 | 16.006226 | 3.587479 | 19.593704 |

Individual timed repetitions:

- full_attention: `0.862921, 0.863737, 0.863030` seconds.
- yakv: `2.263422, 2.262604, 2.259458` seconds.
- arkvale: `23.608193, 23.029868, 22.937374` seconds.

The payloads were identical across repetitions within each mode. Between YAKV
and ArkVale:

- H2D differs by **589,824 bytes / 0.003432%**.
- D2H is **identical**, 3,852,025,856 bytes.
- Total KV offload differs by **0.002804%**.

ArkVale recorded 131,123 page loads, 122,163 evictions and 35,840 score steps
across layers per repetition. Current YAKV is approximately **10.18x faster**
than current ArkVale here at matched logical payload. ArkVale's many synchronous
per-page CPU/GPU copies are a likely major overhead; these measurements are not
a profiler-based attribution or a claim about optimized upstream ArkVale.
This is also **not a quality benchmark**: the two pruning policies need not
produce equivalent answers, and EOS is deliberately ignored.

The final comparison was run sequentially on a shared machine; all three modes
reported 177.71 GiB free GPU memory before model loading. Other users' workloads
are not controlled, so do not interpret these timings as isolated hardware peaks.
All processes exited normally, cache integrity checks passed, and ArkVale had no
remaining request states after each measured repetition.

## Reproduce

Activate the repository environment and configure the CUDA toolkit (`CUDA_HOME`,
`PATH`, and headers if using the checkout's `.venv/cuda` installation). Select the
GPU in the environment before running any command. Then:

```bash
export MINISGL_DISABLE_OVERLAP_SCHEDULING=1
MODEL=Qwen/Qwen3-0.6B
PROMPTS=/tmp/minisgl-wildchat-cache/prompts-long-10.json
OUT=logs/wildchat_comparison
mkdir -p "$OUT"

# Downloads/caches the first shard if necessary and prepares the shared corpus.
python benchmark/offline/bench_wildchat_compare.py \
    --model "$MODEL" --prompts-file "$PROMPTS" --prepare-only

for MODE in full_attention arkvale; do
    python benchmark/offline/bench_wildchat_compare.py \
        --mode "$MODE" --model "$MODEL" --prompts-file "$PROMPTS" \
        --repeats 3 --output "$OUT/$MODE.json" > "$OUT/$MODE.log" 2>&1
done
python benchmark/offline/bench_wildchat_compare.py \
    --mode yakv --model "$MODEL" --prompts-file "$PROMPTS" \
    --match-arkvale-result "$OUT/arkvale.json" \
    --repeats 3 --output "$OUT/yakv.json" > "$OUT/yakv.log" 2>&1
```

Actual run artifacts: `logs/wildchat_comparison/{full_attention,yakv,arkvale}.{json,log}`.
JSON contains individual timings, traffic accounting, configuration, generated
text and token IDs; no model server is needed or left running. Pilot artifacts
are retained separately in `logs/wildchat_comparison/pilot/`.
