"""MagicPIG CPU/GPU attention using the authors' native batched kernels."""

from dataclasses import dataclass

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.magicpig import NativeMagicPIG, hash_vectors, partition_prompt
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata

logger = init_logger(__name__)


class MagicPIGPool:
    """Request-slot storage; scheduler page IDs are accounting tokens, not GPU KV locations."""

    def __init__(self, config, model_config, max_batch_size, max_seq_len, device, dtype):
        if dtype != torch.bfloat16 or model_config.head_dim != 128:
            raise ValueError("MagicPIG native backend requires bfloat16 and head_dim=128")
        self.config = config
        self.device, self.dtype = device, dtype
        self.head_dim = model_config.head_dim
        self.qo_heads = div_even(model_config.num_qo_heads, get_tp_info().size)
        self.kv_heads = div_even(
            model_config.num_kv_heads, get_tp_info().size, allow_replicate=True
        )
        self.max_seq_len, self.slots = max_seq_len, max_batch_size
        self.dense_layers = set(config.dense_layers)
        sparse_layers = [i for i in range(model_config.num_layers) if i not in self.dense_layers]
        self.cpu_layer = {layer: i for i, layer in enumerate(sparse_layers)}
        self.native = (
            NativeMagicPIG(
                config,
                len(sparse_layers),
                self.qo_heads,
                self.kv_heads,
                self.head_dim,
                max_batch_size,
                max_seq_len,
            )
            if sparse_layers
            else None
        )
        self.cache = [
            torch.zeros(
                max_batch_size,
                2,
                self.kv_heads,
                max_seq_len if layer in self.dense_layers else config.gpu_capacity,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            for layer in range(model_config.num_layers)
        ]
        self.centers = torch.zeros(
            model_config.num_layers,
            max_batch_size,
            self.kv_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.projection = torch.randn(
            self.head_dim, config.hash_bits * config.hash_tables, generator=generator
        ).to(device=device, dtype=dtype)
        self.prompt_lengths = [0] * max_batch_size
        self.request_ids = [None] * max_batch_size
        # CPU memory is not covered by --memory-ratio. Report the dominant native allocations.
        n = (max_seq_len + 15) // 16 * 16
        cpu_bytes = (
            len(sparse_layers)
            * max_batch_size
            * self.kv_heads
            * (
                n * self.head_dim * 4
                + n * 4
                + config.hash_tables * (n + 2 * 2**config.hash_bits) * 4
            )
        )
        logger.info(
            f"MagicPIG native CPU KV + LSH storage: approximately {cpu_bytes / 2**30:.2f} GiB"
        )
        logger.info(
            f"MagicPIG GPU KV storage: {sum(t.numel() * t.element_size() for t in self.cache) / 2**30:.2f} GiB"
        )

    def fill(self, layer, slot, k, v):
        length = k.shape[0]
        if layer in self.dense_layers:
            self.cache[layer][slot, 0, :, :length].copy_(k.transpose(0, 1))
            self.cache[layer][slot, 1, :, :length].copy_(v.transpose(0, 1))
            return
        start, end = partition_prompt(length, self.config)
        offload_k = k[start:end].transpose(0, 1).contiguous()
        offload_v = v[start:end].transpose(0, 1).contiguous()
        center = (
            offload_k.float().mean(dim=1).to(k.dtype) if end > start else torch.zeros_like(k[0])
        )
        self.centers[layer, slot].copy_(center)
        local_k = torch.cat((k[:start], k[end:]), dim=0) - center
        local_v = torch.cat((v[:start], v[end:]), dim=0)
        self.cache[layer][slot, 0, :, : local_k.shape[0]].copy_(local_k.transpose(0, 1))
        self.cache[layer][slot, 1, :, : local_v.shape[0]].copy_(local_v.transpose(0, 1))
        offload_k = offload_k - center[:, None]
        # Keep only packed codes; bound the much larger projection scratch by chunk size.
        codes = torch.empty(
            self.kv_heads,
            self.config.hash_tables,
            end - start,
            device=self.device,
            dtype=torch.int16,
        )
        for begin in range(0, end - start, self.config.hash_chunk_size):
            stop = min(begin + self.config.hash_chunk_size, end - start)
            codes[:, :, begin:stop] = hash_vectors(
                offload_k[:, begin:stop],
                self.projection,
                self.config.hash_bits,
            ).transpose(1, 2)
        assert self.native is not None
        self.native.fill(self.cpu_layer[layer], slot, offload_k, offload_v, codes)


@dataclass
class MagicPIGMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor
    slots: list[int]
    sparse_lengths: list[int]

    def get_last_indices(self, bs):
        return self.last_indices[:bs]


class MagicPIGBackend(BaseAttnBackend):
    def __init__(self, model_config):
        import flashinfer

        self.fi = flashinfer
        self.last_forward_event = None
        self.pool = get_global_ctx().kv_cache
        if not isinstance(self.pool, MagicPIGPool):
            raise ValueError("MagicPIG must use its own KV pool; hybrid backends are unsupported")
        self.wrappers = []
        self.workspaces = []
        for _ in range(2):
            workspace = torch.empty(128 * 1024**2, device=self.pool.device, dtype=torch.uint8)
            self.workspaces.append(workspace)
            self.wrappers.append(
                flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    "HND",
                    use_tensor_cores=True,
                )
            )

    def prepare_metadata(self, batch: Batch):
        # The scheduler plans on a different stream. Do not overwrite FlashInfer's
        # shared device metadata while the preceding forward still consumes it.
        if self.last_forward_event is not None:
            torch.cuda.current_stream().wait_event(self.last_forward_event)
        pool = self.pool
        slots = [req.table_idx for req in batch.reqs]
        lengths = []
        offset, last = 0, []
        for req in batch.reqs:
            if not 0 <= req.table_idx < pool.slots:
                raise ValueError("MagicPIG request slot is outside allocated storage")
            if req.max_device_len > pool.max_seq_len:
                raise ValueError("MagicPIG prompt + generation exceeds max_seq_len")
            if batch.is_prefill:
                if type(req) is not Req or req.cached_len:
                    raise ValueError(
                        "MagicPIG requires unchunked prefill without prefix-cache hits"
                    )
                if req.output_len > pool.config.generation_buffer:
                    raise ValueError(
                        "Increase magicpig_config.generation_buffer for this output length"
                    )
                pool.prompt_lengths[req.table_idx] = req.device_len
                pool.request_ids[req.table_idx] = req.uid
            elif pool.request_ids[req.table_idx] != req.uid or req.extend_len != 1:
                raise ValueError("MagicPIG decode has no matching prefill or is not single-token")
            prompt_len = pool.prompt_lengths[req.table_idx]
            start, end = partition_prompt(prompt_len, pool.config)
            lengths.append(req.device_len - (end - start))
            offset += req.extend_len
            last.append(offset - 1)
        batch.attn_metadata = MagicPIGMetadata(
            torch.tensor(last, device=pool.device, dtype=torch.int32),
            slots,
            lengths,
        )
        if batch.is_decode:
            if pool.native is not None:
                pool.native.set_batch(slots)
            indices = torch.tensor(slots, device=pool.device, dtype=torch.int32)
            indptr = torch.arange(len(slots) + 1, device=pool.device, dtype=torch.int32)
            for wrapper, seq_lens, capacity in zip(
                self.wrappers,
                (lengths, [r.device_len for r in batch.reqs]),
                (pool.config.gpu_capacity, pool.max_seq_len),
            ):
                wrapper.plan(
                    indptr,
                    indices,
                    torch.tensor(seq_lens, device=pool.device, dtype=torch.int32),
                    pool.qo_heads,
                    pool.kv_heads,
                    pool.head_dim,
                    capacity,
                    pos_encoding_mode="NONE",
                    q_data_type=pool.dtype,
                    data_type=pool.dtype,
                )

    def forward(self, q, k, v, layer_id, batch):
        pool = self.pool
        metadata = batch.attn_metadata
        k = k.reshape(-1, pool.kv_heads, pool.head_dim)
        v = v.reshape_as(k)
        if batch.is_prefill:
            outputs, offset = [], 0
            for req in batch.reqs:
                stop = offset + req.extend_len
                qi, ki, vi = q[offset:stop], k[offset:stop], v[offset:stop]
                outputs.append(
                    self.fi.single_prefill_with_kv_cache(
                        qi,
                        ki,
                        vi,
                        causal=True,
                        kv_layout="NHD",
                    )
                )
                pool.fill(layer_id, req.table_idx, ki, vi)
                offset = stop
            return self._finish(torch.cat(outputs, dim=0))
        dense = layer_id in pool.dense_layers
        for i, req in enumerate(batch.reqs):
            position = req.device_len - 1 if dense else metadata.sparse_lengths[i] - 1
            key = k[i] if dense else k[i] - pool.centers[layer_id, req.table_idx]
            pool.cache[layer_id][req.table_idx, 0, :, position].copy_(key)
            pool.cache[layer_id][req.table_idx, 1, :, position].copy_(v[i])
        wrapper = self.wrappers[int(dense)]
        if dense:
            return self._finish(wrapper.run(q, pool.cache[layer_id]))
        codes = hash_vectors(q, pool.projection, pool.config.hash_bits)
        assert pool.native is not None
        pool.native.stage(q, codes)
        # CPU waits only for query/hash copies, allowing overlap with local GPU attention.
        gpu_output, gpu_lse = wrapper.run_return_lse(q, pool.cache[layer_id])
        assert pool.native is not None
        cpu_output, cpu_lse = pool.native.decode(pool.cpu_layer[layer_id], metadata.slots)
        output, _ = self.fi.merge_state(
            gpu_output,
            gpu_lse,
            cpu_output.to(pool.device, non_blocking=True),
            cpu_lse.to(pool.device, non_blocking=True),
        )
        return self._finish(output)

    def _finish(self, output):
        if self.last_forward_event is None:
            self.last_forward_event = torch.cuda.Event()
        self.last_forward_event.record()
        return output

    def init_capture_graph(self, max_seq_len, bs_list):
        raise RuntimeError("MagicPIG CPU/GPU backend does not support CUDA graphs")

    def prepare_for_capture(self, batch):
        raise RuntimeError("MagicPIG CPU/GPU backend does not support CUDA graphs")

    def prepare_for_replay(self, batch):
        raise RuntimeError("MagicPIG CPU/GPU backend does not support CUDA graphs")
