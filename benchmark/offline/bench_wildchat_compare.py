"""Comparable full-attention/YAKV/ArkVale offline WildChat runs.

Run each mode in a fresh process, on the same --prompts-file. KV traffic below
is exact *logical payload accounting*, not PCIe hardware counters: it excludes
metadata, transaction amplification, driver staging and model/input transfers.
ArkVale counts actual recalled pages; YAKV counts every gathered BF16 KV token,
including its growing generated-token tail. Neither counter synchronizes CUDA
inside the timed loop. See --help for workload/budget controls.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch


def prepare_prompts(args):
    import bench_wildchat as reference
    from transformers import AutoTokenizer

    if args.prompts_file.exists():
        prompts = json.loads(args.prompts_file.read_text())
    else:
        # Reuse precisely the reference script's language/safety/chat-template
        # filtering, while keeping its downloaded shard out of the source tree.
        reference.BASE_DIR = args.dataset_cache
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = []
        for ids in reference.iter_filtered_prompt_ids(tokenizer):
            if args.min_input_len <= len(ids) <= args.max_input_len:
                prompts.append(ids)
                if len(prompts) == args.num_prompts:
                    break
        args.prompts_file.parent.mkdir(parents=True, exist_ok=True)
        args.prompts_file.write_text(json.dumps(prompts))
    if len(prompts) != args.num_prompts:
        raise ValueError(f"Expected {args.num_prompts} prompts, got {len(prompts)}")
    if not all(args.min_input_len <= len(ids) <= args.max_input_len for ids in prompts):
        raise ValueError("Cached prompts do not satisfy the input length bounds")
    return prompts


class TrafficMeter:
    def __init__(self, llm, mode, sparse_tokens):
        self.mode = mode
        self.pool = llm.engine.ctx.kv_cache
        self.sparse_tokens = sparse_tokens
        # All modes here use BF16 KV and TP=1. Include both K and V.
        self.reset()
        backend = llm.engine.attn_backend
        original_prepare = backend.prepare_metadata
        if mode == "arkvale":
            original_release = self.pool.release

            def release(uid):
                stats = original_release(uid)
                self.h2d += stats["recalls"] * self.page_bytes
                self.recalls += stats["recalls"]
                self.evictions += stats["evictions"]
                self.score_steps += stats["score_steps"]
                return stats

            self.pool.release = release

        def prepare(batch):
            original_prepare(batch)
            for req in batch.padded_reqs:
                if not batch.is_prefill:
                    self.decode_steps += 1
                if self.mode == "full_attention":
                    continue
                self.d2h += req.extend_len * self.token_bytes_all_layers
                if self.mode == "yakv":
                    if batch.is_prefill:
                        self.input_lengths[req.uid] = req.device_len
                    else:
                        n = self.input_lengths[req.uid]
                        # This benchmark sets prefix/suffix=0 and chunk_size=1.
                        selected = min(n, self.sparse_tokens) + req.device_len - n
                        self.h2d += selected * self.token_bytes_all_layers

        backend.prepare_metadata = prepare

    def configure_model(self, model_config):
        self.token_bytes_all_layers = (
            2 * model_config.num_kv_heads * model_config.head_dim * 2 * model_config.num_layers
        )
        if self.mode == "arkvale":
            # release() reports recalls separately for each layer.
            self.page_bytes = (
                self.pool.config.page_size * self.token_bytes_all_layers // model_config.num_layers
            )

    def reset(self):
        self.h2d = self.d2h = 0
        self.recalls = self.evictions = self.score_steps = self.decode_steps = 0
        self.input_lengths = {}

    def result(self):
        return {
            "kv_h2d_bytes": self.h2d,
            "kv_d2h_bytes": self.d2h,
            "kv_total_bytes": self.h2d + self.d2h,
            "recalls": self.recalls,
            "evictions": self.evictions,
            "score_steps": self.score_steps,
            "request_decode_steps": self.decode_steps,
            "accounting": "logical KV payload; not physical PCIe counters",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["full_attention", "yakv", "arkvale"], default="full_attention"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--min-input-len", type=int, default=2048)
    parser.add_argument("--max-input-len", type=int, default=4096)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-running-requests", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--match-arkvale-result",
        type=Path,
        help="Choose nearest integer YAKV token budget to this ArkVale H2D payload",
    )
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/minisgl-wildchat-cache"))
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--yakv-sparse-tokens", type=int, default=64)
    parser.add_argument("--arkvale-page-size", type=int, default=32)
    parser.add_argument("--arkvale-page-budget", type=int, default=32)
    parser.add_argument("--arkvale-page-topk", type=int, default=32)
    args = parser.parse_args()
    prompts = prepare_prompts(args)
    digest = hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest()
    print(f"Prompt SHA256: {digest}; lengths: {[len(p) for p in prompts]}", flush=True)
    if args.prepare_only:
        return
    if args.output is None:
        parser.error("--output is required when running inference")
    if args.yakv_sparse_tokens < 1 or args.repeats < 1 or args.max_output_len < 2:
        parser.error("Require positive sparse budget/repeats and at least two output tokens")

    from minisgl.attention import ShadowKVConfig
    from minisgl.core import SamplingParams
    from minisgl.env import ENV
    from minisgl.kernel.arkvale import ArkValeConfig
    from minisgl.llm import LLM
    from minisgl.models import ModelConfig
    from transformers import AutoConfig

    if not ENV.DISABLE_OVERLAP_SCHEDULING:
        parser.error("All comparison modes require MINISGL_DISABLE_OVERLAP_SCHEDULING=1")
    hf_config = AutoConfig.from_pretrained(args.model)
    model_config = ModelConfig.from_hf(hf_config)
    maximum = math.ceil((max(map(len, prompts)) + args.max_output_len) / 128) * 128
    if args.match_arkvale_result:
        if args.mode != "yakv":
            parser.error("--match-arkvale-result is only valid for YAKV")
        reference = json.loads(args.match_arkvale_result.read_text())
        if (
            reference["mode"] != "arkvale"
            or reference["prompt_sha256"] != digest
            or reference["output_token_budget"] != len(prompts) * args.max_output_len
        ):
            parser.error("ArkVale calibration must use the same prompts and output length")
        token_bytes = (
            2 * model_config.num_kv_heads * model_config.head_dim * 2 * model_config.num_layers
        )
        steps = len(prompts) * (args.max_output_len - 1)
        # Generated tail lengths are 1..(output_len-1) for every request.
        tail_tokens = len(prompts) * args.max_output_len * (args.max_output_len - 1) // 2
        ideal = (reference["traffic"]["kv_h2d_bytes"] / token_bytes - tail_tokens) / steps
        args.yakv_sparse_tokens = max(1, min(min(map(len, prompts)), round(ideal)))
        print(
            f"Matched YAKV sparse tokens: ideal={ideal:.6f}, integer={args.yakv_sparse_tokens}",
            flush=True,
        )
    options = {}
    backend = "fa"
    if args.mode == "yakv":
        options["shadowkv_config"] = ShadowKVConfig(
            enabled=True,
            chunk_size=1,
            prefix_budget=0.0,
            suffix_budget=0.0,
            sparse_budget=args.yakv_sparse_tokens,
            min_seqlen_to_prune=0,
            quantize_landmarks=True,
            quantize_kvcache=False,
            enable_offloading=True,
        )
    elif args.mode == "arkvale":
        backend = "arkvale"
        options["arkvale_config"] = ArkValeConfig(
            page_size=args.arkvale_page_size,
            page_budget=args.arkvale_page_budget,
            page_topk=args.arkvale_page_topk,
            sink_pages=2,
            window_pages=2,
            max_cpu_bytes=16 * 1024**3,
        )
    # Identical non-overlapped, non-graph, naive scheduler settings in all modes.
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        attention_backend=backend,
        cuda_graph_max_bs=0,
        page_size=1,
        cache_type="naive",
        max_seq_len_override=maximum,
        max_extend_tokens=maximum,
        max_running_req=args.max_running_requests,
        num_page_override=maximum * args.max_running_requests,
        memory_ratio=0.5,
        **options,
    )
    meter = TrafficMeter(llm, args.mode, args.yakv_sparse_tokens)
    meter.configure_model(model_config)

    def generate():
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # Fixed output work makes inter-mode traffic/throughput comparable.
        params = [
            SamplingParams(
                temperature=args.temperature, ignore_eos=True, max_tokens=args.max_output_len
            )
            for _ in prompts
        ]
        return llm.generate(prompts, params)

    try:
        print(
            f"Warming all {len(prompts)} prompts through {args.max_output_len} generation steps",
            flush=True,
        )
        generate()
        torch.cuda.synchronize()
        measurements = []
        all_results = []
        for repeat in range(args.repeats):
            meter.reset()
            started = time.perf_counter()
            results = generate()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            output_lengths = [len(item["token_ids"]) for item in results]
            # LLM omits a final EOS token from token_ids even with ignore_eos=True.
            assert len(results) == len(prompts) and min(output_lengths) >= args.max_output_len - 1
            measurements.append(
                {
                    "elapsed_seconds": elapsed,
                    "output_tokens_per_second": sum(output_lengths) / elapsed,
                    "output_lengths": output_lengths,
                    "traffic": meter.result(),
                }
            )
            all_results.append(results)
            print(
                f"Repeat {repeat + 1}: {elapsed:.4f}s, {sum(output_lengths) / elapsed:.2f} output tok/s",
                flush=True,
            )
            if args.mode == "arkvale":
                assert not llm.engine.ctx.kv_cache.states, "Request cache leaked"
            llm.cache_manager.check_integrity()
        middle = sorted(range(args.repeats), key=lambda i: measurements[i]["elapsed_seconds"])[
            args.repeats // 2
        ]
        representative = measurements[middle]
        record = {
            "mode": args.mode,
            "model": args.model,
            "prompt_sha256": digest,
            "input_lengths": list(map(len, prompts)),
            "output_token_budget": len(prompts) * args.max_output_len,
            **representative,
            "measurements": measurements,
            "summary": "median-time repetition (upper median for even repeat counts)",
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "outputs": all_results[middle],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(
            json.dumps(
                {k: v for k, v in record.items() if k not in ("outputs", "settings")}, indent=2
            ),
            flush=True,
        )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
