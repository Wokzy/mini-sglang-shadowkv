from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

try:
    from flash_attn_interface import flash_attn_with_kvcache as flash3_kvcache_impl
except ImportError:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache as flash3_kvcache_impl


@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass(frozen=False)
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor
    cache_batch_idx: torch.Tensor
    og_max_landmarks: int = 0

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.version = 4 if is_sm100_supported() else 3

        self.shadowkv_enabled = ctx.shadowkv_enabled

    def flash3_kvcache_forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ):
        metadata: FAMetadata = batch.attn_metadata
        cache_batch_idx = metadata.cache_batch_idx

        match self.shadowkv_enabled:
            case "YAKV":
                self.kvcache.store_kv(
                    k, v, layer_id, is_prefill=batch.is_prefill, batch_indices=cache_batch_idx
                )

                if batch.is_prefill:
                    q_flash = q.view(
                        1, q.shape[0], self.kvcache.local_qo_heads, self.config.head_dim
                    )
                    batch_indices = [req.table_idx for req in batch.padded_reqs]
                    cache_batch_idx = torch.Tensor([0]).to(
                        device=self.kvcache.device, dtype=torch.int32, non_blocking=True
                    )
                    self.kvcache.compute_and_store_landmarks(k, layer_id, batch_indices)
                    k_cache = k.view(
                        1, k.shape[0], self.kvcache.local_kv_heads, self.kvcache.head_dim
                    )
                    v_cache = v.view(
                        1, v.shape[0], self.kvcache.local_kv_heads, self.kvcache.head_dim
                    )
                else:
                    q_flash = q.view(
                        metadata.cache_batch_idx.shape[0],
                        1,
                        self.kvcache.local_qo_heads,
                        self.config.head_dim,
                    )
                    k_cache, v_cache = self.kvcache.select_kv(
                        q,
                        layer_id,
                    )
            case "OGShadowKV":
                return self.og_shadowkv_forward(q, k, v, layer_id, batch)
            case _:
                raise RuntimeError

        return flash3_kvcache_impl(
            q_flash,
            k_cache,
            v_cache,
            cache_seqlens=metadata.cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            causal=True,
        )

    def og_shadowkv_forward(self, q, k, v, layer_id, batch):
        h, d = self.config.num_kv_heads, self.config.head_dim
        k, v = k.view(-1, h, d), v.view(-1, h, d)
        if batch.is_decode:
            metadata: FAMetadata = batch.attn_metadata
            self.kvcache.decode_batch(
                layer_id,
                q,
                k,
                v,
                metadata.cache_batch_idx,
                metadata.cache_seqlens,
                metadata.og_max_landmarks,
            )
            # Transpose is a view. The gather kernels keep their head-major
            # layout; FA accepts these strides without repacking the cache.
            return flash3_kvcache_impl(
                q.view(batch.size, 1, self.config.num_qo_heads, d),
                self.kvcache.k_cache_buffer[layer_id].transpose(1, 2),
                self.kvcache.v_cache_buffer[layer_id].transpose(1, 2),
                cache_batch_idx=metadata.cache_batch_idx,
                cache_seqlens=metadata.cache_seqlens,
                softmax_scale=self.scale,
                causal=True,
            ).squeeze(1)

        assert batch.size == 1, "1 prefill at a time only supported"

        pre_k = self.kvcache.take_pre_rope()
        req = batch.reqs[0]
        if req.cached_len:
            raise ValueError("Original ShadowKV requires a full prompt prefill")
        self.kvcache.prefill(
            req.table_idx,
            layer_id,
            pre_k,
            k,
            v,
            q,
            req.output_len,
        )

        return flash3_kvcache_impl(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            softmax_scale=self.scale,
            causal=True,
        ).squeeze(0)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)

        if self.shadowkv_enabled:
            return self.flash3_kvcache_forward(q, k, v, layer_id, batch)

        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)

        return _fa_sgl_impl(
            q=q,  # shape: (BS, num_qo_heads, HD)
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=(
                metadata.page_table
                if not (self.shadowkv_enabled and batch.is_prefill)
                else self.kvcache.imag_page_table[:1, : k.shape[0]]
            ),
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]

        og_max_landmarks = 0
        if self.shadowkv_enabled == "OGShadowKV" and batch.is_decode:
            if any(n != 1 for n in seqlens_q):
                raise ValueError("Original ShadowKV decode requires one token per request")
            og_max_landmarks = self.kvcache.max_landmarks([req.table_idx for req in reqs])
            seqlens_k = self.kvcache.decode_lengths(seqlens_k, [req.table_idx for req in reqs])

        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")

        batch_indices = [req.table_idx for req in reqs]
        if self.shadowkv_enabled == "YAKV":
            if batch.is_prefill:
                self.kvcache.prepare_shadowkv_metadata(seqlens_k, batch_indices)
            else:
                (
                    cu_seqlens_k,
                    max_seqlen_k,
                    cache_seqlens,
                    new_page_table,
                ) = self.kvcache.prepare_decode_metadata(seqlens_k, batch_indices)
                cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)
                cache_seqlens = cache_seqlens.to(device, non_blocking=True)

        batch.attn_metadata = FAMetadata(
            og_max_landmarks=og_max_landmarks,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
            cache_batch_idx=torch.Tensor(batch_indices).to(
                device=self.kvcache.device, dtype=torch.int32, non_blocking=True
            ),
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        batch_indices = [req.table_idx for req in batch.padded_reqs]
        self.kvcache.prepare_decode_metadata(capture.seq_lens[:bs], batch_indices)
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
            cache_batch_idx=torch.Tensor(batch_indices).to(
                device=self.kvcache.device, dtype=torch.int32, non_blocking=True
            ),
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    # try:
    #     from sgl_kernel.flash_attn import flash_attn_with_kvcache
    # except ImportError as e:
    #     raise ImportError(
    #         "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
    #         "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
    #     ) from e

    return flash3_kvcache_impl(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        # ver=3,  # TODO: support FA4 on blackwell
    )
