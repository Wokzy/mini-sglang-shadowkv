from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool

from minisgl.kernel import store_cache


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_batch_size: int = None
    ) -> None:
        tp_info = get_tp_info()
        self.local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self.head_dim = head_dim
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, self.local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        self.enable_fp8 = (dtype == torch.float8_e4m3fn)

        if self.enable_fp8:
            self.qmax = 448.0
            self.k_inv_batch_scales = torch.ones((num_layers, max_batch_size), dtype=torch.bfloat16, device=self.device)
            self.v_inv_batch_scales = torch.ones((num_layers, max_batch_size), dtype=torch.bfloat16, device=self.device)


        print(f"MHA POOL INIT: {num_pages=} {page_size=} {self._kv_buffer.shape}")

        self._dtype = dtype
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, self.local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def fp8_quantize(self, k, v, layer_idx: int, batch_indices) -> tuple[torch.Tensor, torch.Tensor]:
        if len(batch_indices) == 0:
            batch_idx = batch_indices[0]
            self.k_inv_batch_scales[layer_idx, batch_idx] = torch.amax(k) / self.qmax
            self.v_inv_batch_scales[layer_idx, batch_idx] = torch.amax(v) / self.qmax

            return ((k / self.k_inv_batch_scales[layer_idx, batch_idx]).to(torch.float8_e4m3fn),
                    (v / self.v_inv_batch_scales[layer_idx, batch_idx]).to(torch.float8_e4m3fn))

        qmaxes = torch.ones_like(k[0], device=k.device) * self.qmax
        k_res = []
        v_res = []
        for i, batch_idx in enumerate(batch_indices):
            k_res.append(torch.min(k[i] / self.k_inv_batch_scales[layer_idx, batch_idx], qmaxes).to(torch.float8_e4m3fn))
            v_res.append(torch.min(v[i] / self.k_inv_batch_scales[layer_idx, batch_idx], qmaxes).to(torch.float8_e4m3fn))

        return torch.stack(k_res, dim=0), torch.stack(v_res, dim=0)

    def fp8_dequantize(self, k, v, layer_idx: int, batch_indices) -> tuple[torch.Tensor, torch.Tensor]:
        if len(batch_indices) == 0:
            batch_idx = batch_indices[0]

            return (k.to(torch.bfloat16) * self.k_inv_batch_scales[layer_idx, batch_idx],
                    v.to(torch.bfloat16) * self.v_inv_batch_scales[layer_idx, batch_idx])

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int, batch_indices: list[int] | None,
    ) -> None:

        if self.enable_fp8:
            assert batch_indices is not None
            k, v = self.fp8_quantize(k, v, layer_id, batch_indices)

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
