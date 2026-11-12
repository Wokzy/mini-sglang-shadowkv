# MagicPIG CPU/GPU baseline

This backend integrates the authors' **native LSH and CPU sparse-attention kernels**, not a GPU masking emulation. It implements the algorithm from [MagicPIG](https://arxiv.org/abs/2410.16179) using the [official v0.2 sources](https://github.com/Infini-AI-Lab/MagicPIG/tree/ac9aa36c866330ca6ad2ce342a7848d7df6f49bb).

Prefill uses FlashInfer full causal attention. For sparse layers, prompt keys are centered; the middle of the prompt and LSH tables live in CPU RAM. The GPU retains sink tokens, the prompt suffix, and generated tokens. Decode hashes queries on GPU, retrieves tokens matching in at least two tables on CPU, and uses the authors' importance-corrected attention kernel. FlashInfer computes local attention on GPU and merges the two outputs using their base-2 log-sum-exp values. Dense layers stay entirely on GPU. Queries/hash codes use pinned staging buffers; local GPU attention can overlap CPU retrieval/computation.

## Build the optional CPU extensions

Use the **same Python environment and PyTorch installation as the server**. This does not replace PyTorch or install the authors' old CUDA/FlashInfer environment. The existing MiniSGL dependencies, including FlashInfer, must already be installed.

Requirements: Linux x86-64 CPU with AVX-512, C++ compiler compatible with the installed PyTorch, OpenMP, setuptools, and optionally ninja. `--bf16` additionally requires AVX512_BF16 and GCC >=11. By default the script detects AVX512_BF16 on the build host and prints the chosen mode; use `--bf16` explicitly when building for a compatible target on another host, or `--no-bf16` for the authors' AVX-512 conversion kernels while retaining BF16 KV storage.

From the repository root:

```bash
python tools/build_magicpig.py --build-dir "$PWD/work/magicpig-native" --threads 64 --bf16
export PYTHONPATH="$PWD/work/magicpig-native/lib:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/magicpig
```

Choose a **new build directory** when rebuilding. The active-slot adapter requires native ABI version 2: rebuild both extensions after updating from the original integration; older binaries are rejected. The script downloads revision `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb` and edits only its private copy. For an offline build, pass `--source /path/to/checkout-of-that-revision`. Choose `--threads` for the physical cores available **per TP rank**; set `OMP_NUM_THREADS` too, because upstream table construction also uses OpenMP's default team size. Bind CPU cores/NUMA memory consistently between benchmarks.

The build uses the installed PyTorch ABI instead of the upstream hardcoded old ABI. The following serving fixes are explicit in the build script: reset a slot's LSH buckets before refilling; give empty samples zero mass (`LSE=-inf`); clamp the cosine before `acos` and handle zero norms. The upstream probability stabilizer `+1e-4` is preserved. Extensions have separate `minisgl_magicpig_*` names so unrelated `lsh` packages cannot be loaded accidentally. Upstream Apache-2.0/third-party licenses remain in the downloaded source tree.

## Launch

```bash
python -m minisgl \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --attention-backend magicpig \
  --additional-config-path benchmark/configs/magicpig.json \
  --max-running-requests 1 \
  --max-seq-len-override 32768 \
  --max-prefill-length 32768 \
  --page-size 1 --cache-type naive --graph 0
```

`magicpig_config` selects this backend when `attention_backend=auto`; explicit conflicting/hybrid backends and enabled ShadowKV are rejected. Without a JSON config, `--attention-backend magicpig` uses the defaults in the example file.

Supported: BF16, head dimension 128, GQA, tensor parallelism with rank-local heads, changing decode batches, and reused request slots. Constraints: no prefix reuse, chunked prefill, hybrid backend, or CUDA graphs. Graph capture is disabled automatically. Set the prefill budget at least as large as the maximum sequence length. Other model-specific attention semantics (e.g. sliding windows) are not added by this integration.

Configuration:

| Field | Default | Meaning |
| --- | --- | --- |
| `hash_bits` | 10 | K; integer in 1..15 |
| `hash_tables` | 150 | L; at least 2 |
| `sink_tokens` | 4 | Exact prompt prefix |
| `local_tokens` | 64 | Exact prompt suffix, without overlap with the prefix |
| `generation_buffer` | 1024 | Maximum requested output length; generated KV stays on GPU |
| `dense_layers` | `[0,16,32,48,64]` | Exact GPU attention layers; IDs beyond the model depth are ignored |
| `seed` | 42 | Independent seed for random hash projections |
| `hash_chunk_size` | 1024 | Bound on prefill projection scratch size |

`dense_layers` follows the upstream defaults and should be recorded with results; it affects both quality and memory. K/L control expected sampling density, **not a fixed TopK budget**.

GPU KV capacity is explicit: each sparse layer reserves `(sink_tokens + local_tokens + generation_buffer)` tokens per request; each dense layer reserves `max_seq_len`. CPU KV/LSH capacity scales with `max_running_requests * max_seq_len`, and LSH tables can be larger than KV itself. Start with one request and size RAM before increasing concurrency. `--memory-ratio` does not budget these explicit CPU/GPU pools. `--num-pages` controls scheduler accounting only and defaults to `max_running_requests * max_seq_len`; it never allocates a second full GPU KV cache.

## Compare all three methods

Prepare `prompts.json` as a JSON array of the same text prompts, and your existing YAKV settings in `yakv.json`. Set explicit cache capacity for dense (`--num-pages`); otherwise its automatic allocation can reserve much more KV memory than the request needs. Run each command in a **fresh process** using identical model, batch size, input/output lengths and CPU affinity:

```bash
python benchmark/offline/compare_attention.py --model /path/to/model --method dense \
  --prompts prompts.json --output results/dense.json --batch-size 1 --num-pages 32768
python benchmark/offline/compare_attention.py --model /path/to/model --method yakv \
  --config yakv.json --prompts prompts.json --output results/yakv.json --batch-size 1 --num-pages 32768
python benchmark/offline/compare_attention.py --model /path/to/model --method magicpig \
  --config benchmark/configs/magicpig.json --prompts prompts.json \
  --output results/magicpig.json --batch-size 1 --num-pages 32768
```

The runner warms up, uses greedy fixed-length generation, saves generated texts/token IDs, and reports **end-to-end throughput including prefill** as the median of `--repeats` runs (default 3). Per-run durations are retained. Dense uses FlashInfer by default (`--dense-backend fi`); `--dense-backend fa` selects FlashAttention (on B200, also use `--page-size 128`; `--num-pages` then counts 128-token pages). All methods run without CUDA graphs. Record this when comparing with a graph-enabled deployment. It also reports PyTorch peak GPU allocation/reservation and the current process's lifetime peak RSS (including startup/warmup). Those are not total GPU usage or a cgroup/process-tree memory peak. For capacity comparisons, additionally measure total device memory and the cgroup/process tree in fresh runs, repeating each measurement. Assess quality with the same task scorer on the saved outputs; output agreement with dense decoding alone is not a quality benchmark.

This is a MiniSGL integration of the official CPU kernels, not a reproduction of the authors' entire serving scheduler. Prefill centering, hashing, GPU norms and transfers run on a separate CUDA stream; the host commits the previous layer’s native KV/LSH fill after subsequent GPU attention has been submitted. Native work stays on the calling thread to avoid creating competing OpenMP teams. At most one native layer fill is outstanding. The engine drains pending work before completing prefill, and before decode, slot reuse or shutdown; fill failures propagate to the caller. Full CPU/GPU overlap depends on the workload and is not guaranteed.

Native ABI version 2 accepts explicit active slot IDs. Retrieval and attention use packed active query/output rows and touch only the selected slots' KV/LSH tables, without copying the KV cache. The native attention formula and sampling probabilities remain unchanged; the active attention path uses dynamic scheduling over active heads. Inactive scratch rows are intentionally left untouched. GPU decode appends all requests with one fused kernel; hash signs are packed with an integer Triton kernel (including K > 11), and GPU CPU-result buffers are reused. FlashInfer planning receives CPU indptr/lengths, avoiding a GPU-to-CPU metadata round trip.

Prefill key norms now accumulate in FP32 on GPU. Reduction order differs from the old CPU FP32 norm, so tiny rounding changes can alter BF16 attention outputs and subsequent greedy tokens. Bitwise agreement with the original generation is not guaranteed; evaluate quality on the same tasks.

Persistent cache capacity still scales with the configured maximum batch and context. Query D2H synchronization and CPU attention remain on the decode critical path. Do not claim paper speedups without measurements on the target GPU/CPU.

Tests in `test_native.py` run on CPU and verify native selection, importance weighting, LSE, empty samples, GQA, and slot reuse. `test_gpu.py` exercises actual FlashInfer merging and KV lifecycle and is skipped without CUDA. Full-model timing is separate from these correctness tests.

### Decode latency profiling

Add `--profile-phases` to the comparison runner. After the uninstrumented end-to-end measurement it runs a separate pass, synchronizing the engine stream around each forward. The report includes mean/median/p95 forward latency for prefill and decode, and decode tokens/s for forward computation only. This includes CPU attention and GPU completion, but excludes scheduler preparation; it is not an inter-token latency measurement of a serving endpoint. Compare the same mode across methods. The separate profile does not alter the earlier end-to-end throughput or memory measurements.

The decode adapter reuses query-copy events and batch mappings. All batches use packed pinned input/output views; native kernels map those rows to physical request slots without scatter/gather copies. Results return to GPU asynchronously. The next query-copy event also orders reuse of the CPU result storage after the previous H2D copy.

### B200 environment and baseline

The B200 validation environment used Python 3.12, PyTorch 2.9.1+cu130, FlashInfer 0.5.3, Transformers 4.57.3, and the official CUDA 13 wheel of sgl-kernel 0.3.17.post1. The bundled FA4 implementation required compatible packages: nvidia-cutlass-dsl 4.3.3, quack-kernels 0.2.3, and nvidia-cudnn-frontend 1.13.0. Keep the venv `bin` directory on PATH so FlashInfer JIT can find ninja. These are tested environment versions, not a change to the project's global dependency requirements.

For YAKV on Blackwell, the FA backend uses FA4. It views the existing gathered KV buffers as 128-token pages and maps request slots through a page table, avoiding an extra KV copy. `max_seq_len` must be divisible by 128. The FA3 path on older GPUs is retained. `test_fa4.py` checks prefill and reordered decode slots against exact attention on B200.

Example CPU placement (adapt cores to the GPU's NUMA affinity):

```bash
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=ACTIVE \
  taskset -c 0-15 python benchmark/offline/compare_attention.py \
  --model /path/to/model --method magicpig \
  --config benchmark/configs/magicpig.json --prompts prompts.json \
  --output results/magicpig.json --batch-size 1 --max-seq-len 32768 \
  --num-pages 32768 --output-tokens 64 --repeats 3 --profile-phases
```

Use native extensions built with `--threads 16` for this example. OpenMP waiting policy can materially affect a CPU/GPU pipeline; record it along with affinity and compiled thread count. Compare real task inputs: LSH selection size depends on the queries and keys, so repeated synthetic text is only a diagnostic workload.

### Historical B200 diagnostic workload before active-slot/overlap optimizations (2026-11-09)

Qwen3-32B BF16, batch 1, 64 output tokens, 16 pinned Xeon 6960P cores, OpenMP ACTIVE, one warmup and three timed repetitions. Inputs repeat a synthetic sentence and are **not a quality benchmark or paper-speedup reproduction**. All methods disable CUDA graphs. End-to-end throughput includes prefill; decode forward is measured in a separate synchronized pass.

| Input tokens | Method | End-to-end tokens/s (median) | Decode forward mean, ms | PyTorch GPU peak allocated, GiB |
| ---: | --- | ---: | ---: | ---: |
| 7007 | Dense / FlashInfer | 47.10 | 14.58 | 64.39 |
| 7007 | YAKV / FA4 | 18.17 | 49.11 | 63.11 |
| 7007 | MagicPIG | 6.80 | 71.30 | 62.93 |
| 32007 | Dense / FlashInfer | 11.96 | 15.66 | 74.68 |
| 32007 | YAKV / FA4 | 5.49 | 147.47 | 70.25 |
| 32007 | MagicPIG | 1.79 | 163.16 | 67.60 |

MagicPIG used the example configuration unchanged and native BF16 extensions built with 16 threads. YAKV used chunk_size=1, prefix_budget=suffix_budget=0.00625, sparse_budget=0.015625, quantize_landmarks=true, quantize_kvcache=false, enable_offloading=true. These configurations are not matched by quality. Dense page capacity was explicitly 8192/32768 tokens. GPU peaks include weights and prefill scratch, not just KV. MagicPIG reduced GPU memory here but did not improve speed; its synchronous 32K prefill took approximately 26 seconds. YAKV's 32K timings varied between repetitions (10.45–12.38 seconds for full generation).
