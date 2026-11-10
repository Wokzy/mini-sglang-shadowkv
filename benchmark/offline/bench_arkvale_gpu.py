"""Paired end-to-end ArkVale GPU vs synchronous-reference serving benchmark.

Both paths share one initialized model and slot allocation. Modes are switched
only after requests drain and the device synchronizes. Neither path uses CUDA
graphs; both use the same scheduler/attention settings. Run on an otherwise idle
GPU for meaningful dedicated-device throughput; report contention if unavoidable.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cases", default="1:512,4:512,4:2048", help="batch:prompt_tokens pairs")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--page-budget", type=int, default=32)
    parser.add_argument("--page-topk", type=int, default=16)
    parser.add_argument("--max-cpu-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        cases = [tuple(map(int, pair.split(":"))) for pair in args.cases.split(",")]
        assert all(len(pair) == 2 and min(pair) > 0 for pair in cases)
    except (ValueError, AssertionError):
        parser.error("--cases must contain positive batch:prompt_tokens pairs")
    if args.repeats < 1 or args.output_tokens < 2:
        parser.error("Require at least one repetition and two output tokens")
    if max(n for _, n in cases) + args.output_tokens > args.max_seq_len:
        parser.error("Prompts plus output budget must fit --max-seq-len")

    from minisgl.core import SamplingParams
    from minisgl.env import ENV
    from minisgl.kernel.arkvale import ArkValeConfig
    from minisgl.llm import LLM

    maximum_batch = max(bs for bs, _ in cases)
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        attention_backend="arkvale",
        cuda_graph_max_bs=0,
        page_size=1,
        cache_type="naive",
        max_seq_len_override=args.max_seq_len,
        max_extend_tokens=args.max_seq_len,
        max_running_req=maximum_batch,
        num_page_override=args.max_seq_len * maximum_batch,
        memory_ratio=0.5,
        arkvale_config=ArkValeConfig(
            gpu_decode=True,
            page_size=args.page_size,
            page_budget=args.page_budget,
            page_topk=args.page_topk,
            sink_pages=2,
            window_pages=2,
            max_cpu_bytes=args.max_cpu_bytes,
        ),
    )
    pool = llm.engine.ctx.kv_cache
    gpu = pool.gpu
    original_release = pool.release
    original_forward = llm._forward
    counters, events = [], []

    def release(uid):
        result = original_release(uid)
        counters.append(result)
        return result

    def forward(data):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_forward(data)
        end.record()
        events.append((data.batch.phase, start, end))
        return result

    pool.release = release
    llm._forward = forward
    base = llm.tokenizer.encode(
        "This archive describes rivers, mountains, cities and scientific discoveries. "
        "Compare the entries carefully. "
    )
    record = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "overlap": not bool(ENV.DISABLE_OVERLAP_SCHEDULING),
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "host_backing_bytes": pool.allocated_host_bytes,
        "cases": [],
    }
    try:
        for bs, length in cases:
            prompts = [
                (base * ((length + len(base) - 1) // len(base)))[: length - 1] + [100 + i]
                for i in range(bs)
            ]
            case = {"batch_size": bs, "prompt_tokens": length, "gpu": [], "reference": []}
            expected = None
            for rep in range(args.repeats + 1):
                modes = ("reference", "gpu") if rep % 2 == 0 else ("gpu", "reference")
                for mode in modes:
                    # Pool is drained and synchronized between modes; this keeps
                    # memory allocation/model/scheduler identical in the ablation.
                    pool.gpu = gpu if mode == "gpu" else None
                    counters.clear()
                    events.clear()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = llm.generate(
                        prompts,
                        SamplingParams(
                            temperature=0, ignore_eos=True, max_tokens=args.output_tokens
                        ),
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    llm.cache_manager.check_integrity()
                    assert not pool.states and not pool.uid_slots
                    tokens = [r["token_ids"] for r in result]
                    # LLM omits a final EOS from its returned IDs even with ignore_eos.
                    assert all(
                        args.output_tokens - 1 <= len(t) <= args.output_tokens for t in tokens
                    )
                    if expected is None:
                        expected = tokens
                    if tokens != expected:
                        raise RuntimeError(
                            "Generated IDs differ between runs/modes. Investigate numerics and "
                            "score ties before interpreting throughput (GPU tie order is defined; "
                            "torch.topk's is unspecified)."
                        )
                    measurement = {
                        "seconds": elapsed,
                        "returned_tokens": sum(map(len, tokens)),
                        "prefill_stream_ms": sum(
                            a.elapsed_time(b) for phase, a, b in events if phase == "prefill"
                        ),
                        "decode_stream_ms": sum(
                            a.elapsed_time(b) for phase, a, b in events if phase == "decode"
                        ),
                        "traffic": {k: sum(c[k] for c in counters) for k in counters[0]},
                    }
                    if rep:  # one complete warmup for each mode/case is excluded
                        case[mode].append(measurement)
                    print(
                        json.dumps({"case": [bs, length], "mode": mode, "rep": rep, **measurement}),
                        flush=True,
                    )
            for mode in ("gpu", "reference"):
                case[f"{mode}_median_seconds"] = statistics.median(m["seconds"] for m in case[mode])
            case["speedup"] = case["reference_median_seconds"] / case["gpu_median_seconds"]
            case["output_sha256"] = hashlib.sha256(json.dumps(expected).encode()).hexdigest()
            case["prompt_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
            if [m["traffic"] for m in case["gpu"]] != [m["traffic"] for m in case["reference"]]:
                raise RuntimeError("Recall/eviction counters differ between modes")
            record["cases"].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2))
            print(f"Batch {bs}, prompt {length}: {case['speedup']:.3f}x", flush=True)
    finally:
        pool.gpu = gpu
        pool.release = original_release
        llm._forward = original_forward
        llm.shutdown()


if __name__ == "__main__":
    main()
