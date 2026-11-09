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


@dataclass
class OGShadowKVConfig:
    enabled: bool = False
    chunk_size: int = 8
    sparse_budget: int = 2048  # Tokens, not chunks.
    outliers_budget: int | None = None  # Chunks per KV head.
    rank: int = 160
    enable_offloading: bool = False

    def __post_init__(self):
        if self.chunk_size <= 0 or self.sparse_budget < 0 or self.rank <= 0:
            raise ValueError("Invalid original ShadowKV chunk size, sparse budget, or rank")
        if self.sparse_budget % self.chunk_size:
            raise ValueError("Original ShadowKV sparse_budget must be divisible by chunk_size")
        if self.outliers_budget is None:
            self.outliers_budget = (self.sparse_budget // 1024) * 24
        if self.outliers_budget < 0:
            raise ValueError("Original ShadowKV outliers_budget must be nonnegative")

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
    # Every request owns its lengths, landmarks, SVD, and hit/miss state.
    # Kernel inputs remain contiguous batch-of-one views, even when scheduler
    # slots are noncontiguous or requests have different prompt lengths.
    k: torch.Tensor
    v: torch.Tensor
    prompt_len: int
    sparse_start: int
    sparse_end: int
    length: int
    u: torch.Tensor | None = None
    sv: torch.Tensor | None = None
    values: torch.Tensor | None = None
    landmarks: torch.Tensor | None = None
    landmark_ids: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    fast: bool = False
    workspace: tuple = ()


class ShadowKVCache_CPU:
    """Original ShadowKV with independent state for each scheduler request slot.

    The original fused scoring, cache-hit reuse, CPU value gathering and key
    reconstruction kernels are retained for their supported shapes. Other
    shapes use the same algorithm in PyTorch. Short prompts use exact KV.
    Only full prompt prefills and one-token decoding are supported.
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
        self.local_chunk = 4
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
            # CPU pinned values may still be read by a custom kernel, outside
            # PyTorch's allocator tracking. Retire those reads before releasing.
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
        if layer == 0:
            self.reset_slot(slot)
        n, h, d = k.shape
        c = self.chunk_size
        # The upstream kernels require chunk counts aligned to eight.
        chunks = max(0, n // c - self.local_chunk)
        chunks -= chunks % 8
        sparse = self.select_sets > 0 and chunks > self.select_sets + self.outlier_chunk
        if not sparse:
            state = self._allocate(n, n, n, output_len)
            state.k[0, :, :n].copy_(k.transpose(0, 1))
            state.v[0, :, :n].copy_(v.transpose(0, 1))
            self.slots[slot][layer] = state
            return

        local = n - chunks * c
        start = local + self.outlier_chunk * c
        end = start + self.sparse_budget
        state = self._allocate(n, start, end, output_len)
        keys = k.transpose(0, 1).contiguous().unsqueeze(0)
        values = v.transpose(0, 1).contiguous().unsqueeze(0)
        grouped = keys[:, :, : chunks * c].reshape(1, h, chunks, c, d)
        means = grouped.mean(dim=-2)
        scores = F.cosine_similarity(means.unsqueeze(-2), grouped, dim=-1).amin(-1)
        outliers = scores.topk(self.outlier_chunk, largest=False).indices
        all_ids = torch.arange(chunks, device=self.device).expand(1, h, -1)
        keep = torch.ones_like(all_ids, dtype=torch.bool).scatter_(-1, outliers, False)
        ids = all_ids[keep].reshape(1, h, chunks - self.outlier_chunk)
        state.landmark_ids = ids
        state.landmarks = means.gather(2, ids.unsqueeze(-1).expand(-1, -1, -1, d)).contiguous()
        state.k[:, :, :local].copy_(keys[:, :, -local:])
        state.v[:, :, :local].copy_(values[:, :, -local:])
        outlier_tokens = self._token_ids(outliers)
        state.k[:, :, local:start].copy_(self._gather(keys, outlier_tokens))
        state.v[:, :, local:start].copy_(self._gather(values, outlier_tokens))

        # Decompose all heads together, as in the authors' implementation.
        matrix = pre_k.reshape(n, h * d).float()
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(self.rank, s.numel())
        state.u = u[:, :rank].to(self.dtype).unsqueeze(0).contiguous()
        state.sv = (
            (s[:rank, None] * vh[:rank])
            .to(self.dtype)
            .reshape(rank, h, d)
            .permute(1, 2, 0)
            .unsqueeze(0)
            .contiguous()
        )
        if self.shadowkv_config.enable_offloading:
            state.values = torch.empty(
                (1, h, chunks * c, d), device="cpu", dtype=self.dtype, pin_memory=True
            )
            # Blocking D2H ensures the CPU cache is ready for mapped reads.
            state.values.copy_(values[:, :, : chunks * c])
        else:
            state.values = values[:, :, : chunks * c].contiguous()

        # Initialize exact selected keys. Subsequent hits preserve these keys;
        # misses use the low-rank reconstruction, matching the original cache.
        state.positions = self._select(state, q[-1:]).contiguous()
        token_ids = self._token_ids(state.positions)
        state.k[:, :, start:end].copy_(self._gather(keys, token_ids))
        state.v[:, :, start:end].copy_(self._gather(values, token_ids))
        state.fast = (
            self.dtype == torch.bfloat16
            and d == 128
            and h <= 16
            and c == 8
            and rank % 8 == 0
            and self.select_sets in (128, 256, 512, 1024)
            and ids.shape[-1] % 8 == 0
        )
        if state.fast:
            self._init_workspace(state)
        self.slots[slot][layer] = state

    def _allocate(self, prompt_len, start, end, output_len):
        # Allow one extra in-flight token from overlap scheduling. Unlike the
        # reference's fixed +128 buffer, this also supports long generations.
        capacity = end + output_len + 1
        shape = (1, self.num_key_value_heads, capacity, self.head_dim)
        return _LayerCache(
            torch.empty(shape, device=self.device, dtype=self.dtype),
            torch.empty(shape, device=self.device, dtype=self.dtype),
            prompt_len,
            start,
            end,
            end,
        )

    def _init_workspace(self, state):
        h, g, n = self.num_key_value_heads, self.num_key_value_groups, state.landmarks.shape[-2]

        def empty(shape, dtype=self.dtype):
            return torch.empty(shape, device=self.device, dtype=dtype)

        state.workspace = (
            empty((1, h, g, n)),
            empty((1, h, g, n)),
            empty((h, g, (n + 255) // 256), torch.float32),
            empty((h, g, (n + 255) // 256), torch.float32),
            empty((h * self.select_sets,), torch.int32),
            empty((h,), torch.int32),
            torch.zeros(h, device=self.device, dtype=torch.int32),
            empty((1, h, self.sparse_budget, self.head_dim)),
            empty((1, h, self.sparse_budget, self.head_dim)),
        )

    def _token_ids(self, ids):
        return (
            ids.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=ids.device)
        ).flatten(-2)

    def _gather(self, tensor, ids):
        return tensor.gather(2, ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

    def _select(self, state, q):
        q = q.reshape(1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
        if state.fast:
            from minisgl import og_shadowkv_kernels as kernels

            gemm, softmax, norm, sums, *_ = state.workspace
            kernels.batch_gemm_softmax(
                q.contiguous(),
                state.landmarks,
                gemm,
                norm,
                sums,
                softmax,
                self.num_key_value_heads,
                self.num_key_value_groups,
                state.landmarks.shape[-2],
                self.head_dim,
                self.head_dim**-0.5,
                0.0,
            )
            scores = softmax.amax(dim=2)
        else:
            logits = torch.matmul(q, state.landmarks.transpose(-1, -2)) * self.head_dim**-0.5
            scores = logits.softmax(-1, dtype=torch.float32).to(self.dtype).amax(dim=2)
        indices = scores.topk(self.select_sets, dim=-1).indices
        return state.landmark_ids.gather(-1, indices)

    def decode(
        self,
        slot: int,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sequence_len: int,
    ):
        state = self.slots[slot][layer]
        length = state.sparse_end + sequence_len - state.prompt_len
        if length > state.k.shape[-2] or length != state.length + 1:
            raise RuntimeError(f"Invalid ShadowKV decode length for slot {slot}: {length}")
        state.k[0, :, length - 1].copy_(k.reshape(self.num_key_value_heads, self.head_dim))
        state.v[0, :, length - 1].copy_(v.reshape(self.num_key_value_heads, self.head_dim))
        if state.landmarks is not None:
            selected = self._select(state, q)
            if state.fast:
                self._decode_fast(state, selected)
            else:
                self._decode_torch(state, selected)
        state.length = length
        # FlashAttention consumes [batch, tokens, heads, dim].
        return state.k[:, :, :length].transpose(1, 2), state.v[:, :, :length].transpose(1, 2)

    def _decode_fast(self, state, selected):
        from minisgl import og_shadowkv_kernels as kernels
        from .tensor_op import batch_gather_gemm_rotary_pos_emb_cuda

        _, _, _, _, offsets, counts, signals, temp, output = state.workspace
        h, d, budget = self.num_key_value_heads, self.head_dim, self.sparse_budget
        kernels.reorder_keys_and_compute_offsets(
            state.positions, selected.contiguous(), offsets, counts, 1, h, self.select_sets
        )
        current = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current)
        with torch.cuda.stream(self.copy_stream):
            kernels.gather_copy_with_offsets(
                state.values,
                state.v,
                temp,
                offsets,
                counts,
                signals,
                1,
                h,
                state.values.shape[-2] * d,
                budget * d,
                state.sparse_start * d,
                state.v.shape[-2] * d,
                self.select_sets,
            )
        kernels.gather_copy_d2d_with_offsets(
            state.k,
            offsets,
            counts,
            1,
            h,
            budget * d,
            state.sparse_start * d,
            state.k.shape[-2] * d,
            self.select_sets,
        )
        batch_gather_gemm_rotary_pos_emb_cuda(
            state.u,
            state.sv,
            self._rope_cache,
            state.positions,
            output,
            self.chunk_size,
            state.k,
            state.sparse_start,
            state.sparse_end,
            counts,
        )
        current.wait_stream(self.copy_stream)

    def _decode_torch(self, state, selected):
        # Retain cached (possibly exact-prefill) keys on hits, just like the
        # fused path. This fallback handles arbitrary budgets and head shapes.
        matches = selected.unsqueeze(-1) == state.positions.unsqueeze(-2)
        hits = matches.any(-1)
        old_ids = matches.to(torch.int32).argmax(-1)
        old_tokens = self._token_ids(old_ids)
        old_keys = self._gather(state.k[:, :, state.sparse_start : state.sparse_end], old_tokens)
        ids = self._token_ids(selected)
        u = state.u[:, None].expand(-1, self.num_key_value_heads, -1, -1)
        u = u.gather(2, ids.unsqueeze(-1).expand(-1, -1, -1, u.shape[-1]))
        keys = torch.matmul(u, state.sv.transpose(-1, -2))
        cs = self._rope_cache[ids].float()
        cos, sin = cs.chunk(2, -1)
        x, y = keys.float().chunk(2, -1)
        keys = torch.cat((x * cos - y * sin, y * cos + x * sin), -1).to(self.dtype)
        keys = torch.where(
            hits.repeat_interleave(self.chunk_size, -1).unsqueeze(-1), old_keys, keys
        )
        # Only selected values cross PCIe; never copy the entire host cache.
        value_ids = ids.cpu() if state.values.device.type == "cpu" else ids
        values = self._gather(state.values, value_ids).to(self.device)
        state.k[:, :, state.sparse_start : state.sparse_end].copy_(keys)
        state.v[:, :, state.sparse_start : state.sparse_end].copy_(values)
        state.positions = selected

    def clear(self):
        self.copy_stream.synchronize()
        self.slots.clear()
        self._pre_rope = None
