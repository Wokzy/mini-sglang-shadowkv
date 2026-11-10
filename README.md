# ArkVale

[ArkVale](https://github.com/pku-liang/ArkVale) uses query-dependent key-page
scores to keep a bounded GPU KV cache and recall evicted pages from CPU.
This integration is separate from YAKV/ShadowKV. Its implementation and Triton
kernels live in [`python/minisgl/kernel/arkvale/`](python/minisgl/kernel/arkvale/).
See [ArkVale method and implementation notes](docs/arkvale.md) for the pinned
upstream reference and integration details.

## Install and launch

Use the repository's normal CUDA/PyTorch installation below (`pip install -e .
--no-build-isolation` in an existing environment). ArkVale's Triton kernels compile
on first use; no separate upstream ArkVale extension or FlashInfer fork is needed.
The accelerated path reuses this repository's `minisgl.shadowkv_kernels.map_to_gpu`
helper to map pinned CPU storage. Build the existing ShadowKV extension as part
of the normal installation, or set `gpu_decode=false` to use the synchronous
reference path without that helper. Ensure
`nvcc` is on `PATH` and `CUDA_HOME` points to your CUDA toolkit before launch;
Mini-SGLang's existing JIT kernels need the toolkit even when ArkVale uses Triton.

Supported runtime: FP16/BF16, tensor parallel size 1, scheduler page size
1, naive cache, no CUDA graphs, and no chunked prefill. Overlap scheduling is
supported, using the same shared-workspace stream ordering as ShadowKV.
Set maximum prefill length at least as large as maximum sequence length.
Unsupported combinations fail at startup. Model head dimensions up to 256 are
supported by the ArkVale kernels (the model's own RoPE constraints still apply).

```bash
export MINISGL_DISABLE_OVERLAP_SCHEDULING=0
python -m minisgl \
    --model Qwen/Qwen3-0.6B \
    --additional-config-path arkvale_config.json \
    --attention-backend arkvale \
    --cache-type naive --page-size 1 --graph 0 --tp-size 1 \
    --max-running-requests 2 \
    --max-seq-len-override 4096 --max-prefill-length 4096 \
    --memory-ratio 0.5 \
    --host 127.0.0.1 --port 1919
```

[`arkvale_config.json`](arkvale_config.json):

```json
{
  "arkvale_config": {
    "enabled": true,
    "gpu_decode": true,
    "page_size": 32,
    "page_budget": 32,
    "page_topk": 16,
    "sink_pages": 2,
    "window_pages": 2,
    "max_cpu_bytes": 8589934592
  }
}
```

- `gpu_decode`: enable GPU-resident batched scoring, selection, FIFO bookkeeping,
  and asynchronous backup/recall (default `true`, including GPU prefill setup).
  Set to `false` for the previous synchronous reference integration. The GPU
  path supports up to 32768 logical pages/request; increase `page_size` or use
  the reference path for larger page tables.
- `page_size`: internal ArkVale tokens per page, a power of two from 1 to 128.
  Independent of scheduler `--page-size`, which must remain 1.
- `page_budget`: maximum GPU-resident pages **per request, per layer**.
- `page_topk`: total pages requested per decode step, **including** protected
  sink/window pages. Must satisfy
  `sink_pages + window_pages < page_topk <= page_budget`.
  Other resident pages are retained until space is needed; attention uses all
  resident pages, matching upstream's default non-sparse-attention mode.
- `sink_pages` / `window_pages`: always retain the first / most recent pages.
  Sink may be zero; window must be at least two pages.
- `max_cpu_bytes`: total pinned CPU backing-store budget. Each configured request
  slot reserves `2 * layers * ceil(max_seq_len / page_size) * page_size * KV_heads *
  head_dim * dtype_bytes`. All slots are allocated at startup and reused on
  completion/abort; buffers remain allocated until the pool is destroyed.
  Startup checks require enough budget for all configured running requests.

GPU storage is bounded by resident KV pages, per-page GPU digests, and a reusable
batched gather workspace. Storage is indexed by scheduler request slot; decode
uses one batched FlashAttention call per layer.
`--memory-ratio` bounds this worst-case cache allocation against free memory
following model loading; leave extra memory for prefill and model activations.
`--num-pages`, if supplied, controls **logical scheduler token capacity**, not
ArkVale GPU residency; normally leave it unset. Do not enable ShadowKV in the
same JSON file. Prefill is exact; decode becomes approximate above `page_budget *
page_size` tokens. The accelerated path performs batched selection/bookkeeping
on the GPU and backup/recall on a transfer stream, without a CPU miss-list
round-trip. Selection is shared across KV heads; equal scores prefer the lower
logical page (the reference's `torch.topk` tie order is unspecified). It does not
implement cross-layer predictive prefetch, quantization, or per-layer budgets.
See [the method and implementation notes](docs/arkvale.md) for high-level details.

# YAKV

## Prerequisites
   - `flash_attn` installed
   - page_size == 1
   - no cuda graph
   - `--cache-type` naive
   - Model `head-dim` 128

## Installation:

Install from source:
```bash
pip install . --no-build-isolation
```

**Note** that the installation requires flash-attn. Building it from source can take up significant time and memory. Consider using pre-built wheels or images.
Also all prerequisites from mini-sglang are required, such as **NVIDIA CUDA Toolkit** and etc.

## Code
   -  YAKV kernels: `python/minisgl/kernel/shadowkv/`
   -  HIGGS kernels: `python/minisgl/kernel/higgs/`
   -  Implementation: `python/minisgl/attention/[shadowkv.py, fa.py]`

### Example:
```bash
python3 -m minisgl \
    --model-path Qwen3-30B-A3B-Instruct-2507 \
    --additional-config-path additional_config.json \
    --memory-ratio 0.2 \
    --num-pages 10000000 \
    --max-seq-len-override 131072 \
    --max-prefill-length 131072 \
    --tp-size 1 \
    --graph 0 \
    --attention-backend fa,fa \
    --max-running-requests 2 \
    --cache-type naive \
    --host "localhost" \
    --port 6888

```

additional_config.json
```json
{
   "shadowkv_config": {
      "enabled": true,
      "min_seqlen_to_prune": 512,
      "prefix_budget": 0.00625,
      "sparse_budget": 0.015625,
      "suffix_budget": 0.00625,
      "chunk_size": 1,
      "quantize_landmarks": true,
      "enable_offloading": true,
      "landmarks_dtype": "bf16",
      "kv_cache_dtype": "fp8"
   }
}
```

#### `shadowkv_config` options:
   - `min_seqlen_to_prune`: in fractional sparse-budget mode, requests won't be pruned to fewer than this many tokens. Make sure it does not clash with your ratio budgets (`min_seqlen_to_prune` must be less than or equal to `max_seq_len * total_budget`). Absolute sparse budgets take precedence over this setting.
   - `prefix_budget`: proportion of prefix input tokens that is always gathered
   - `sparse_budget`: selected sparse budget. A float is a proportion of request tokens (for example, `0.015625`); an integer is an absolute number of chunks selected for every request (for example, `2048`). If fewer chunks are available, all of them are selected.
   - `suffix_budget`: proportion of suffix input tokens that is always gathered
   - `chunk_size`: Landmarks chunk size (YAKV proposes chunk size 1)
   - `quantize_landmarks`: Enable 2-bit HIGGS qunatization of landmarks
   - `enable_offloading`: weather to offload KV-cache to CPU
   - `landmarks_dtype`: Allowed values `[bf16, fp8]`. If `quantize_landmarks == false` landmarks will be stored in selected dtype
   - `kv_cache_dtype`: Allowed values `[bf16, fp8]`




# Mini-SGLang

A **lightweight yet high-performance** inference framework for Large Language Models.

---

Mini-SGLang is a compact implementation of [SGLang](https://github.com/sgl-project/sglang), designed to demystify the complexities of modern LLM serving systems. With a compact codebase of **~5,000 lines of Python**, it serves as both a capable inference engine and a transparent reference for researchers and developers.

## ✨ Key Features

- **High Performance**: Achieves state-of-the-art throughput and latency with advanced optimizations.
- **Lightweight & Readable**: A clean, modular, and fully type-annotated codebase that is easy to understand and modify.
- **Advanced Optimizations**:
  - **Radix Cache**: Reuses KV cache for shared prefixes across requests.
  - **Chunked Prefill**: Reduces peak memory usage for long-context serving.
  - **Overlap Scheduling**: Hides CPU scheduling overhead with GPU computation.
  - **Tensor Parallelism**: Scales inference across multiple GPUs.
  - **Optimized Kernels**: Integrates **FlashAttention** and **FlashInfer** for maximum efficiency.
  - ...

## 🚀 Quick Start

> **⚠️ Platform Support**: Mini-SGLang currently supports **Linux only** (x86_64 and aarch64). Windows and macOS are not supported due to dependencies on Linux-specific CUDA kernels (`sgl-kernel`, `flashinfer`). We recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows or Docker for cross-platform compatibility.

### 1. Environment Setup

We recommend using `uv` for a fast and reliable installation (note that `uv` does not conflict with `conda`).

```bash
# Create a virtual environment (Python 3.10+ recommended)
uv venv --python=3.12
source .venv/bin/activate
```

**Prerequisites**: Mini-SGLang relies on CUDA kernels that are JIT-compiled. Ensure you have the **NVIDIA CUDA Toolkit** installed and that its version matches your driver's version. You can check your driver's CUDA capability with `nvidia-smi`.

### 2. Installation

Install Mini-SGLang directly from the source:

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 Installing on Windows (WSL2)</b></summary>

Since Mini-SGLang requires Linux-specific dependencies, Windows users should use WSL2:

1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install CUDA on WSL2**:
   - Follow [NVIDIA's WSL2 CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Ensure your Windows GPU drivers support WSL2

3. **Install Mini-SGLang in WSL2**:
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **Access from Windows**: The server will be accessible at `http://localhost:8000` from Windows browsers and applications.

</details>

<details>
<summary><b>🐳 Running with Docker</b></summary>

**Prerequisites**:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **Build the Docker image**:
   ```bash
   docker build -t minisgl .
   ```

2. **Run the server**:
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **Run in interactive shell mode**:
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **Using Docker Volumes for persistent caches** (recommended for faster subsequent startups):
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. Online Serving

Launch an OpenAI-compatible API server with a single command.

```bash
# Deploy Qwen/Qwen3-0.6B on a single GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy meta-llama/Llama-3.1-70B-Instruct on 4 GPUs with Tensor Parallelism, on port 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

Once the server is running, you can send requests using standard tools like `curl` or any OpenAI-compatible client.

### 4. Interactive Shell

Chat with your model directly in the terminal by adding the `--shell` flag.

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

You can also use `/reset` to clear the chat history.

## Benchmark

### Offline inference

Set `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` for an ablation study on overlap scheduling.

Test Configuration:

- Hardware: 1xH200 GPU.
- Model: Qwen3-0.6B, Qwen3-14B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100-1024 tokens
- Output Length: Randomly sampled between 100-1024 tokens

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### Online inference

See [benchmark_qwen.py](./benchmark/online/bench_qwen.py) for more details.

Test Configuration:

- Hardware: 4xH200 GPU, connected by NVLink.
- Model: Qwen3-32B
- Dataset: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl), replaying first 1000 requests.

Launch command:

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **Note**: If you encounter network issues when downloading models from HuggingFace, try using `--model-source modelscope` to download from ModelScope instead:
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 Learn More

- **[Detailed Features](./docs/features.md)**: Explore all available features and command-line arguments.
- **[System Architecture](./docs/structures.md)**: Dive deep into the design and data flow of Mini-SGLang.
