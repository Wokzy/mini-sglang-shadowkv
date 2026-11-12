################################################################################
#
# Reference: https://github.com/ByteDance-Seed/ShadowKV/blob/main/models/kv_cache.py
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minisgl.distributed import get_tp_info
from minisgl.models.config import ModelConfig
from minisgl.utils import init_logger

import triton

from minisgl import og_shadowkv_kernels as kernels

from .batched_kernels import reduce_scores, selected_chunk_ids

logger = init_logger(__name__)


@dataclass
class OGShadowKVConfig:
    enabled: bool = False
    chunk_size: int = 8
    sparse_budget: int = 2048  # Tokens, not chunks.
    outliers_budget: int | None = None  # Chunks per KV head.
    local_chunks_count: int = 4  # Suffix chunks count
    rank: int = 160
    enable_offloading: bool = False
    max_out_len: int = 1024

    def __post_init__(self):
        if self.chunk_size <= 0 or self.sparse_budget < 0 or self.rank <= 0:
            raise ValueError("Invalid original ShadowKV chunk size, sparse budget, or rank")
        if self.sparse_budget % self.chunk_size:
            raise ValueError("Original ShadowKV sparse_budget must be divisible by chunk_size")
        if self.outliers_budget is None:
            self.outliers_budget = (self.sparse_budget // 1024) * 24
        if self.outliers_budget < 0:
            raise ValueError("Original ShadowKV outliers_budget must be nonnegative")
        if not isinstance(self.max_out_len, int) or self.max_out_len <= 0:
            raise ValueError("Original ShadowKV max_out_len must be a positive integer")

    @classmethod
    def from_dict(cls, config: dict):
        config = dict(config)
        if "keys_rank" in config:
            if "rank" in config:
                raise ValueError("Specify only one of rank and keys_rank")
            config["rank"] = config.pop("keys_rank")
        return cls(**config)


@dataclass
class _LayerCache:
    # K/V, landmarks, landmark IDs, and values are shared-pool views.
    # All persistent tensors are views of fixed-capacity slot pools.
    k: torch.Tensor
    v: torch.Tensor
    prompt_len: int
    sparse_start: int
    sparse_end: int
    u: torch.Tensor | None = None
    sv: torch.Tensor | None = None
    values: torch.Tensor | None = None
    landmarks: torch.Tensor | None = None
    landmark_ids: torch.Tensor | None = None
    positions: torch.Tensor | None = None


class ShadowKVCache_CPU:
    """Original ShadowKV with independent state for each scheduler request slot.

    Original scoring and cache-hit/value-gather kernels accept active slot
    indices. A Triton gather/GEMM/RoPE kernel reconstructs missing keys into the
    shared K pool. Unsupported configurations fail at initialization. Short
    prompts retain exact KV. Prefill is per request; decode is one batch per layer.
    """

    def __init__(
        self,
        config: ModelConfig,
        shadowkv_config: OGShadowKVConfig,
        batch_size=1,
        max_length=32 * 1024,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        if get_tp_info().size != 1:
            raise ValueError("Original ShadowKV currently requires TP size 1")
        self.config = config
        self.shadowkv_config = shadowkv_config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device)
        self.dtype = dtype
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_kv_heads
        self.num_attention_heads = config.num_qo_heads
        self.num_key_value_groups = config.num_qo_heads // config.num_kv_heads
        self.num_layers = config.num_layers
        self.chunk_size = shadowkv_config.chunk_size
        self.sparse_budget = shadowkv_config.sparse_budget
        self.select_sets = self.sparse_budget // self.chunk_size
        self.rank = shadowkv_config.rank
        self.outlier_chunk = shadowkv_config.outliers_budget
        self.local_chunk = shadowkv_config.local_chunks_count
        self.max_out_len = shadowkv_config.max_out_len
        if (
            dtype != torch.bfloat16
            or self.head_dim != 128
            or not 1 <= self.num_key_value_heads <= 16
            or self.chunk_size != 8
            or self.rank % 8
            or self.rank > self.num_key_value_heads * self.head_dim
            or self.select_sets not in (128, 256, 512, 1024)
            or self.outlier_chunk % 8
            or self.local_chunk < 0
            or config.num_qo_heads % config.num_kv_heads
            or max_length % 64
        ):
            raise ValueError(
                "Unsupported original ShadowKV shape: requires BF16, head_dim=128, "
                "1..16 KV heads, chunk_size=8, rank aligned to 8, "
                "128/256/512/1024 selected chunks, outliers aligned to 8, max_length aligned to 64"
            )
        # Rounding the context chunk count down to a multiple of eight leaves
        # up to seven extra full chunks and a partial chunk in the local suffix.
        # This bound also accommodates every short prompt using the dense path.
        prompt_capacity = (
            self.sparse_budget + (self.outlier_chunk + self.local_chunk + 8) * self.chunk_size
        )
        self.cache_capacity = min(max_length, prompt_capacity) + self.max_out_len + 1

        # +1 allows the scheduler's extra in-flight decode after completion.
        # FA3 may load padded V lanes even with cache_seqlens. Zero the pools
        # once so 0 * an uninitialized NaN cannot contaminate attention output.
        self.k_cache_buffer = torch.zeros(
            (
                self.num_layers,
                batch_size,
                self.num_key_value_heads,
                self.cache_capacity,
                self.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        self.v_cache_buffer = torch.zeros(
            (
                self.num_layers,
                batch_size,
                self.num_key_value_heads,
                self.cache_capacity,
                self.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        # Fixed strides for layer/slot/head; kernels receive valid lengths and
        # active slot indices, so no full-cache packing is needed at decode.
        self.max_chunks = max_length // self.chunk_size
        self.k_landmark = torch.empty(
            (self.num_layers, batch_size, self.num_key_value_heads, self.max_chunks, self.head_dim),
            device=self.device,
            dtype=dtype,
        )
        self.k_landmark_idx = torch.empty(
            (self.num_layers, batch_size, self.num_key_value_heads, self.max_chunks),
            device=self.device,
            dtype=torch.int64,
        )

        logger.info(
            "Allocated original ShadowKV landmarks pool: %.2f GiB on %s",
            self.k_landmark.numel() * self.k_landmark.element_size() / 2**30
            + self.k_landmark_idx.numel() * self.k_landmark_idx.element_size() / 2**30,
            self.k_landmark.device,
        )

        self.v_cache_cpu = torch.empty(
            (
                self.num_layers,
                batch_size,
                self.num_key_value_heads,
                self.max_chunks,
                self.chunk_size * self.head_dim,
            ),
            device="cpu" if shadowkv_config.enable_offloading else self.device,
            dtype=dtype,
            pin_memory=False,
        )

        if shadowkv_config.enable_offloading:
            kernels.pin_cpu_tensor(self.v_cache_cpu)

        self.svd_u = torch.empty(
            (self.num_layers, batch_size, self.max_length, self.rank),
            dtype=self.dtype,
            device=self.device,
        )

        self.svd_sv = torch.empty(
            (
                self.num_layers,
                batch_size,
                self.rank,
                self.head_dim * self.num_key_value_heads,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        # SV is the singular-value-weighted right factor (also called V).
        self.svd_v = self.svd_sv

        logger.info(
            "Allocated original ShadowKV value pool: %.2f GiB on %s",
            self.v_cache_cpu.numel() * self.v_cache_cpu.element_size() / 2**30,
            self.v_cache_cpu.device,
        )

        # Persistent cache state is layer/slot indexed. Scratch is shared by
        # layers and indexed by the current compact batch (all launches ordered
        # on the engine stream, with the value stream joined before returning).
        self.positions = torch.empty(
            (self.num_layers, batch_size, self.num_key_value_heads, self.select_sets),
            device=self.device,
            dtype=torch.int64,
        )
        self.landmark_lengths = torch.zeros(batch_size, device=self.device, dtype=torch.int32)
        self.sparse_starts = torch.zeros_like(self.landmark_lengths)
        b, h, g, n = (
            batch_size,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.max_chunks,
        )
        self.gemm = torch.empty((b, h, g, n), device=self.device, dtype=dtype)
        self.softmax = torch.empty_like(self.gemm)
        self.norm = torch.empty(
            (b * h, (n + 255) // 256, g), device=self.device, dtype=torch.float32
        )
        self.sums = torch.empty_like(self.norm)
        self.scores = torch.empty((b, h, n), device=self.device, dtype=dtype)
        self.top_values = torch.empty((b, h, self.select_sets), device=self.device, dtype=dtype)
        self.top_indices = torch.empty(
            (b, h, self.select_sets), device=self.device, dtype=torch.int64
        )
        self.selected = torch.empty_like(self.top_indices)
        self.offsets = torch.empty((b, h, self.select_sets), device=self.device, dtype=torch.int32)
        self.counts = torch.empty((b, h), device=self.device, dtype=torch.int32)
        self.signals = torch.zeros((b, h), device=self.device, dtype=torch.int32)
        self.temp = torch.empty(
            (b, h, self.sparse_budget, self.head_dim), device=self.device, dtype=dtype
        )
        self.slots: dict[int, dict[int, _LayerCache]] = {}
        self._pre_rope: torch.Tensor | None = None
        self._rope_cache: torch.Tensor | None = None
        self.copy_stream = torch.cuda.Stream(device=self.device)

    def capture_pre_rope(self, k: torch.Tensor, cos_sin_cache: torch.Tensor):
        # RoPE mutates q/k in place. Capture normalized keys before that happens.
        self._pre_rope = k.reshape(-1, self.num_key_value_heads, self.head_dim).clone()
        if self._rope_cache is None:
            self._rope_cache = cos_sin_cache.to(device=self.device, dtype=self.dtype)

    def take_pre_rope(self):
        k, self._pre_rope = self._pre_rope, None
        if k is None:
            raise RuntimeError("Original ShadowKV did not receive pre-RoPE keys")
        return k

    def reset_slot(self, slot: int):
        if not 0 <= slot < self.batch_size:
            raise ValueError(f"Invalid ShadowKV request slot {slot}")
        self.release_slot(slot)
        self.slots[slot] = {}

    def release_slot(self, slot: int):
        if slot in self.slots:
            # Retire mapped reads before this slot's value storage can be reused.
            self.copy_stream.synchronize()
            del self.slots[slot]

    def prefill(
        self,
        slot: int,
        layer: int,
        pre_k: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        output_len: int,
    ):
        """Initialize one layer for one request; tensors are [tokens, heads, dim]."""
        if not 0 < output_len <= self.max_out_len:
            raise ValueError(f"Original ShadowKV output_len must be in [1, {self.max_out_len}]")
        if not 0 < k.shape[0] <= self.max_length:
            raise ValueError("Original ShadowKV prompt exceeds max_length")
        if layer == 0:
            self.reset_slot(slot)
        n, h, d = k.shape
        c = self.chunk_size
        # The upstream kernels require chunk counts aligned to eight.
        chunks = max(0, n // c - self.local_chunk)
        chunks -= chunks % 8
        sparse = self.select_sets > 0 and chunks > self.select_sets + self.outlier_chunk
        if not sparse:
            state = self._make_layer_cache(slot, layer, n, n, n)
            state.k[0, :, :n].copy_(k.transpose(0, 1))
            state.v[0, :, :n].copy_(v.transpose(0, 1))
            self.landmark_lengths[slot] = 0
            self.sparse_starts[slot] = n
            self.slots[slot][layer] = state
            return

        local = n - chunks * c
        start = local + self.outlier_chunk * c
        end = start + self.sparse_budget
        state = self._make_layer_cache(slot, layer, n, start, end)
        keys = k.transpose(0, 1).contiguous().unsqueeze(0)
        values = v.transpose(0, 1).contiguous().unsqueeze(0)
        grouped = keys[:, :, : chunks * c].reshape(1, h, chunks, c, d)
        means = grouped.mean(dim=-2)
        scores = F.cosine_similarity(means.unsqueeze(-2), grouped, dim=-1).amin(-1)
        outliers = scores.topk(self.outlier_chunk, largest=False).indices
        all_ids = torch.arange(chunks, device=self.device).expand(1, h, -1)
        keep = torch.ones_like(all_ids, dtype=torch.bool).scatter_(-1, outliers, False)
        ids = all_ids[keep].reshape(1, h, chunks - self.outlier_chunk)
        num_landmarks = ids.shape[-1]
        state.landmark_ids = self.k_landmark_idx[layer, slot : slot + 1, :, :num_landmarks]
        state.landmark_ids.copy_(ids)
        state.landmarks = self.k_landmark[layer, slot : slot + 1, :, :num_landmarks]
        state.landmarks.copy_(means.gather(2, ids.unsqueeze(-1).expand(-1, -1, -1, d)))
        self.landmark_lengths[slot] = num_landmarks
        self.sparse_starts[slot] = start
        state.k[:, :, :local].copy_(keys[:, :, -local:])
        state.v[:, :, :local].copy_(values[:, :, -local:])
        outlier_tokens = self._token_ids(outliers)
        state.k[:, :, local:start].copy_(self._gather(keys, outlier_tokens))
        state.v[:, :, local:start].copy_(self._gather(values, outlier_tokens))

        # Decompose all heads together, as in the authors' implementation.
        matrix = pre_k.reshape(n, h * d).float()
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        if s.numel() < self.rank:
            raise ValueError("Original ShadowKV rank exceeds the prompt matrix rank dimension")
        state.u = self.svd_u[layer, slot : slot + 1, :n]
        state.u.copy_(u[:, : self.rank].to(self.dtype).unsqueeze(0))
        self.svd_sv[layer, slot].copy_((s[: self.rank, None] * vh[: self.rank]).to(self.dtype))
        # Preserve the useful [1, heads, dim, rank] reference view without copying.
        state.sv = self.svd_sv[layer, slot : slot + 1].view(1, self.rank, h, d).permute(0, 2, 3, 1)
        state.values = self.v_cache_cpu[layer, slot : slot + 1].view(1, h, self.max_chunks * c, d)[
            :, :, : chunks * c
        ]
        # Blocking D2H ensures the CPU cache is ready for mapped reads. The
        # pinned allocation is reused; only the valid prefix is written.
        state.values.copy_(values[:, :, : chunks * c])

        # Initialize exact selected keys. Subsequent hits preserve these keys;
        # misses use the low-rank reconstruction, matching the original cache.
        state.positions = self.positions[layer, slot : slot + 1]
        state.positions.copy_(self._select_prefill(state, q[-1:]))
        token_ids = self._token_ids(state.positions)
        state.k[:, :, start:end].copy_(self._gather(keys, token_ids))
        state.v[:, :, start:end].copy_(self._gather(values, token_ids))
        self.slots[slot][layer] = state

    def _make_layer_cache(self, slot, layer, prompt_len, start, end):
        if end + self.max_out_len + 1 > self.cache_capacity:
            raise ValueError("Original ShadowKV prompt exceeds the configured cache capacity")
        return _LayerCache(
            self.k_cache_buffer[layer, slot : slot + 1],
            self.v_cache_buffer[layer, slot : slot + 1],
            prompt_len,
            start,
            end,
        )

    def decode_lengths(self, sequence_lengths, slots):
        # Lengths are layer-independent and returned in active query order,
        # whereas the backing K/V storage remains in persistent slot order.
        result = []
        for slot, length in zip(slots, sequence_lengths):
            state = self.slots[slot][0]
            generated = length - state.prompt_len
            if not 1 <= generated <= self.max_out_len + 1:
                raise RuntimeError(
                    "Original ShadowKV decode exceeds max_out_len and overlap allowance"
                )
            result.append(state.sparse_end + generated)
        return result

    def _token_ids(self, ids):
        return (
            ids.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=ids.device)
        ).flatten(-2)

    def _gather(self, tensor, ids):
        return tensor.gather(2, ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

    def _select_prefill(self, state, q):
        q = q.reshape(1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
        logits = torch.matmul(q, state.landmarks.transpose(-1, -2)) * self.head_dim**-0.5
        scores = logits.softmax(-1, dtype=torch.float32).to(self.dtype).amax(dim=2)
        indices = scores.topk(self.select_sets, dim=-1).indices
        return state.landmark_ids.gather(-1, indices)

    def max_landmarks(self, slots):
        return max(
            (self.slots[s][0].landmarks.shape[-2] if self.slots[s][0].landmarks is not None else 0)
            for s in slots
        )

    def select_batch(self, layer, q, slots, max_landmarks):
        """Original fused scoring, GQA max and top-k, once for the active batch."""

        b, h, g = slots.numel(), self.num_key_value_heads, self.num_key_value_groups
        kernels.batch_gemm_softmax_indexed(
            q,
            self.k_landmark[layer],
            self.gemm,
            self.norm,
            self.sums,
            self.softmax,
            slots,
            self.landmark_lengths,
            h,
            g,
            max_landmarks,
        )
        reduce_scores[(b * h, triton.cdiv(max_landmarks, 256))](
            self.softmax,
            self.scores,
            slots,
            self.landmark_lengths,
            h,
            g,
            max_landmarks,
            self.max_chunks,
            triton.next_power_of_2(g),
        )
        torch.topk(
            self.scores[:b, :, :max_landmarks],
            self.select_sets,
            dim=-1,
            out=(self.top_values[:b], self.top_indices[:b]),
        )
        selected_chunk_ids[(b * h,)](
            self.top_indices,
            self.k_landmark_idx[layer],
            self.selected,
            slots,
            self.landmark_lengths,
            h,
            self.max_chunks,
            self.select_sets,
            self.select_sets,
        )
        return self.selected[:b]

    def decode_batch(self, layer, q, k, v, slots, lengths, max_landmarks):
        """One launch per stage for arbitrary active slots and prompt lengths.

        lengths are assembled KV lengths in query order (FA metadata). They
        already include the generated token; persistent cache rows use slot order.
        """
        import triton

        from minisgl import og_shadowkv_kernels as kernels

        from .batched_kernels import append_kv, reconstruct_keys

        b, h = slots.numel(), self.num_key_value_heads
        append_kv[(b,)](
            k,
            v,
            self.k_cache_buffer[layer],
            self.v_cache_buffer[layer],
            slots,
            lengths,
            h,
            self.cache_capacity,
            k.stride(0),
            v.stride(0),
            triton.next_power_of_2(h * 128),
        )
        if not max_landmarks:
            return
        selected = self.select_batch(layer, q, slots, max_landmarks)
        kernels.reorder_keys_indexed(
            self.positions[layer],
            selected,
            self.offsets,
            self.counts,
            slots,
            self.landmark_lengths,
            h,
            self.select_sets,
        )
        current = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current)
        with torch.cuda.stream(self.copy_stream):
            kernels.gather_copy_indexed(
                self.v_cache_cpu[layer],
                self.v_cache_buffer[layer],
                self.temp,
                self.offsets,
                self.counts,
                self.signals,
                slots,
                self.sparse_starts,
                self.landmark_lengths,
                h,
                self.max_chunks * 8 * 128,
                self.cache_capacity * 128,
                self.select_sets,
                False,
            )
        kernels.gather_copy_indexed(
            self.v_cache_cpu[layer],
            self.k_cache_buffer[layer],
            self.temp,
            self.offsets,
            self.counts,
            self.signals,
            slots,
            self.sparse_starts,
            self.landmark_lengths,
            h,
            self.max_chunks * 8 * 128,
            self.cache_capacity * 128,
            self.select_sets,
            True,
        )
        reconstruct_keys[(triton.cdiv(self.sparse_budget, 32), b * h)](
            self.svd_u[layer],
            self.svd_sv[layer],
            self._rope_cache,
            self.positions[layer],
            self.counts,
            slots,
            self.sparse_starts,
            self.landmark_lengths,
            self.k_cache_buffer[layer],
            h,
            self.max_length,
            self.rank,
            self.cache_capacity,
            self.select_sets,
            enable_fp_fusion=False,
        )
        current.wait_stream(self.copy_stream)

    def clear(self):
        self.copy_stream.synchronize()
        self.slots.clear()
        self._pre_rope = None
