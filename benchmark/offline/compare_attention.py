"""Run one method per fresh process on the same JSON list of text prompts.

Reports end-to-end generation throughput (including prefill), not decode-only throughput.
"""

import argparse
import json
import os
import platform
import resource
import statistics
import time
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["dense", "yakv", "magicpig"], required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dense-backend", choices=["fi", "fa"], default="fi")
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-pages", type=int)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-ratio", type=float, default=0.8)
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help="Run a separate synchronized pass for prefill/decode forward latency",
    )
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) for p in prompts):
        parser.error("--prompts must contain a nonempty JSON list of strings")
    if min(args.batch_size, args.max_seq_len, args.output_tokens, args.repeats) < 1:
        parser.error("batch-size, max-seq-len, output-tokens and repeats must be positive")
    config = json.loads(args.config.read_text()) if args.config else {}
    if args.method == "yakv" and not config.get("shadowkv_config", {}).get("enabled"):
        parser.error("yakv requires --config with shadowkv_config.enabled=true")
    if args.method != "magicpig" and "magicpig_config" in config:
        parser.error("magicpig_config is only allowed for --method magicpig")
    if args.method != "yakv" and config.get("shadowkv_config", {}).get("enabled"):
        parser.error("enabled shadowkv_config is only allowed for --method yakv")

    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        args.model,
        attention_backend=(
            "magicpig"
            if args.method == "magicpig"
            else args.dense_backend if args.method == "dense" else "fa"
        ),
        additional_config_path=str(args.config) if args.config else None,
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
        page_size=args.page_size,
        cache_type="naive",
        max_running_req=args.batch_size,
        max_seq_len_override=args.max_seq_len,
        max_extend_tokens=args.max_seq_len,
        memory_ratio=args.memory_ratio,
        num_page_override=args.num_pages,
    )
    try:
        tokenized = [llm.tokenizer.encode(prompt) for prompt in prompts]
        if any(len(ids) + args.output_tokens > args.max_seq_len for ids in tokenized):
            raise ValueError("A prompt + output exceeds --max-seq-len; prompts are not truncated")
        params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_tokens)
        llm.generate([tokenized[0]], params)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        generation_seconds = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            results = llm.generate(tokenized, params)
            torch.cuda.synchronize()
            generation_seconds.append(time.perf_counter() - start)
        elapsed = statistics.median(generation_seconds)
        tokens = sum(len(result["token_ids"]) for result in results)
        report = {
            "method": args.method,
            "attention_backend": (
                "magicpig"
                if args.method == "magicpig"
                else args.dense_backend if args.method == "dense" else "fa"
            ),
            "model": args.model,
            "config": config,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "generation_seconds_each_repeat": generation_seconds,
            "gpu": torch.cuda.get_device_name(),
            "hostname": platform.node(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "omp_places": os.environ.get("OMP_PLACES"),
            "page_size": args.page_size,
            "cuda_graphs": False,
            "num_pages": args.num_pages,
            "dependencies": {
                name: version(name) for name in ("flashinfer-python", "sgl-kernel", "transformers")
            },
            "max_seq_len": args.max_seq_len,
            "input_lengths": [len(ids) for ids in tokenized],
            "output_tokens": tokens,
            "generation_seconds_including_prefill": elapsed,
            "tokens_per_second_including_prefill": tokens / elapsed,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "process_peak_rss_kib_including_setup_and_warmup": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "torch_version": torch.__version__,
            "results": results,
        }
        if args.profile_phases:
            records = {"prefill": [], "decode": []}
            original_forward = llm.engine.forward_batch

            def timed_forward(batch, sampling_args):
                llm.engine.stream.synchronize()
                begin = time.perf_counter()
                result = original_forward(batch, sampling_args)
                llm.engine.stream.synchronize()
                records[batch.phase].append((time.perf_counter() - begin, batch.size))
                return result

            llm.engine.forward_batch = timed_forward
            try:
                llm.generate(tokenized, params)
            finally:
                llm.engine.forward_batch = original_forward
            profile = {}
            for phase, samples in records.items():
                if not samples:
                    continue
                seconds = [sample[0] for sample in samples]
                total = sum(seconds)
                profile[phase] = {
                    "forward_calls": len(samples),
                    "total_seconds": total,
                    "mean_ms": statistics.mean(seconds) * 1000,
                    "median_ms": statistics.median(seconds) * 1000,
                    "p95_ms": sorted(seconds)[max(0, (95 * len(seconds) + 99) // 100 - 1)] * 1000,
                }
                if phase == "decode":
                    profile[phase]["tokens_per_second_forward_only"] = (
                        sum(n for _, n in samples) / total
                    )
            report["separate_synchronized_forward_profile"] = profile
        if args.method == "magicpig":
            from dataclasses import asdict

            report["effective_magicpig_config"] = asdict(llm.engine.kv_cache.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"{args.method}: {tokens / elapsed:.2f} tokens/s including prefill; {args.output}")
    finally:
        llm.engine.shutdown()


if __name__ == "__main__":
    main()
