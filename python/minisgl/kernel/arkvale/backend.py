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
        batch.attn_metadata = ArkValeMetadata(
            [(req.uid, count, req.device_len) for req, count in zip(reqs, counts)],
            torch.tensor(counts, device=self.pool.device, dtype=torch.int32).cumsum(0) - 1,
        )

    def forward(self, q, k, v, layer_id, batch):
        k = k.reshape(-1, self.pool.heads, self.pool.head_dim)
        v = v.reshape(-1, self.pool.heads, self.pool.head_dim)
        outputs = []
        start = 0
        for uid, count, length in batch.attn_metadata.requests:
            qi, ki, vi = (
                q[start : start + count],
                k[start : start + count],
                v[start : start + count],
            )
            if batch.is_prefill:
                self.pool.prefill(uid, layer_id, ki, vi, qi[-1])
                keys, values = ki, vi
            else:
                if count != 1:
                    raise ValueError("ArkVale decode requires one new token per request")
                packed = self.pool.decode(uid, layer_id, ki[0], vi[0], qi[0], length)
                keys, values = packed.unbind(0)
            # PyTorch dispatches SDPA to CUDA attention kernels. GQA is handled
            # without materializing repeated KV heads. Prefill is exact causal
            # attention; decode attends all resident pages (upstream default).
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
