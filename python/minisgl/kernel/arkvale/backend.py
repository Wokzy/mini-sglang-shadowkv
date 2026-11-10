from dataclasses import dataclass

import torch
import torch.nn.functional as F
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata
from minisgl.core import get_global_ctx


@dataclass
class ArkValeMetadata(BaseAttnMetadata):
    # uid, new-token count, total sequence length
    requests: list[tuple[int, int, int]]
    last_indices: torch.Tensor
    cache_batch_idx: torch.Tensor
    sequence_lengths: torch.Tensor

    def get_last_indices(self, bs):
        return self.last_indices[:bs]


class ArkValeBackend(BaseAttnBackend):
    def __init__(self, config):
        self.pool = get_global_ctx().kv_cache
        self.scale = config.head_dim**-0.5

    def prepare_metadata(self, batch):
        reqs = batch.padded_reqs
        if batch.padded_size != batch.size:
            raise ValueError("ArkVale does not support graph padding")
        if batch.is_prefill and any(req.cached_len for req in reqs):
            raise ValueError("ArkVale does not support prefix reuse or chunked prefill")
        counts = [req.extend_len for req in reqs]
        if not batch.is_prefill and any(count != 1 for count in counts):
            raise ValueError("ArkVale decode requires one new token per request")
        if any(req.device_len > self.pool.max_seq_len for req in reqs):
            raise ValueError("ArkVale sequence exceeds configured maximum")
        if not batch.is_prefill:
            for req in reqs:
                previous = self.pool.request_lengths.get(req.uid)
                if previous is None or req.device_len != previous + 1:
                    raise RuntimeError("ArkVale request/cache sequence lengths disagree")
            for req in reqs:
                self.pool.request_lengths[req.uid] = req.device_len
        slots = [self.pool.bind(req.uid, getattr(req, "table_idx", None)) for req in reqs]
        batch.attn_metadata = ArkValeMetadata(
            [(req.uid, count, req.device_len) for req, count in zip(reqs, counts)],
            torch.tensor(counts, device=self.pool.device, dtype=torch.int32).cumsum(0) - 1,
            torch.tensor(slots, device=self.pool.device, dtype=torch.int32),
            torch.tensor(
                [req.device_len for req in reqs], device=self.pool.device, dtype=torch.int32
            ),
        )

    def forward(self, q, k, v, layer_id, batch):
        k = k.reshape(-1, self.pool.heads, self.pool.head_dim)
        v = v.reshape(-1, self.pool.heads, self.pool.head_dim)
        if not batch.is_prefill:
            # Keep page selection/recall semantics, but issue a single attention
            # call over the active slots, just like the ShadowKV FA backend.
            from minisgl.attention.fa import flash3_kvcache_impl

            if self.pool.gpu is not None:
                lengths = self.pool.gpu.decode(
                    q,
                    k,
                    v,
                    layer_id,
                    batch.attn_metadata.cache_batch_idx,
                    batch.attn_metadata.sequence_lengths,
                )
            else:
                lengths = []
                for i, (uid, count, length) in enumerate(batch.attn_metadata.requests):
                    packed = self.pool.decode(uid, layer_id, k[i], v[i], q[i], length)
                    lengths.append(packed.shape[1])
                lengths = torch.tensor(lengths, device=q.device, dtype=torch.int32)
            out = flash3_kvcache_impl(
                q.unsqueeze(1),
                self.pool.gathered_buffer[0],
                self.pool.gathered_buffer[1],
                cache_seqlens=lengths,
                cache_batch_idx=batch.attn_metadata.cache_batch_idx,
                softmax_scale=self.scale,
                causal=False,
            )
            return out.squeeze(1)

        outputs = []
        start = 0
        for uid, count, length in batch.attn_metadata.requests:
            qi, ki, vi = (
                q[start : start + count],
                k[start : start + count],
                v[start : start + count],
            )
            self.pool.prefill(uid, layer_id, ki, vi, qi[-1], gpu=True)
            keys, values = ki, vi
            # Exact causal prefill. Serving admits one prompt per forward;
            # retain multi-request support here for direct backend callers.
            out = F.scaled_dot_product_attention(
                qi.transpose(0, 1).unsqueeze(0),
                keys.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                is_causal=batch.is_prefill,
                scale=self.scale,
                enable_gqa=True,
            )
            outputs.append(out.squeeze(0).transpose(0, 1))
            start += count
        return torch.cat(outputs, dim=0)

    def init_capture_graph(self, max_seq_len, bs_list):
        raise ValueError("ArkVale requires --graph 0")

    def prepare_for_capture(self, batch):
        raise ValueError("ArkVale does not support CUDA graphs")

    def prepare_for_replay(self, batch):
        raise ValueError("ArkVale does not support CUDA graphs")
