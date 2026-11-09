"""Configuration and CPU adapter for the official MagicPIG kernels."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MagicPIGConfig:
    hash_bits: int = 10
    hash_tables: int = 150
    sink_tokens: int = 4
    local_tokens: int = 64
    generation_buffer: int = 1024
    dense_layers: tuple[int, ...] = (0, 16, 32, 48, 64)
    seed: int = 42
    hash_chunk_size: int = 1024

    def __post_init__(self):
        for name in (
            "hash_bits",
            "hash_tables",
            "sink_tokens",
            "local_tokens",
            "generation_buffer",
            "seed",
            "hash_chunk_size",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"MagicPIG {name} must be an integer")
        if not 1 <= self.hash_bits <= 15:
            raise ValueError("MagicPIG hash_bits must be in [1, 15] (signed int16 hash codes)")
        if self.hash_tables < 2:
            raise ValueError("MagicPIG needs at least two hash tables")
        if min(self.sink_tokens, self.local_tokens, self.seed) < 0:
            raise ValueError("MagicPIG token counts and seed must be nonnegative")
        if min(self.generation_buffer, self.hash_chunk_size) < 1:
            raise ValueError("MagicPIG generation_buffer and hash_chunk_size must be positive")
        if any(type(layer) is not int or layer < 0 for layer in self.dense_layers):
            raise ValueError("MagicPIG dense_layers must contain nonnegative integer layer IDs")
        object.__setattr__(self, "dense_layers", tuple(self.dense_layers))

    @classmethod
    def from_dict(cls, config):
        return cls(**config)

    @property
    def gpu_capacity(self):
        return self.sink_tokens + self.local_tokens + self.generation_buffer


def partition_prompt(length: int, config: MagicPIGConfig) -> tuple[int, int]:
    """Return the CPU interval; prefix/suffix never overlap on short prompts."""
    start = min(length, config.sink_tokens)
    end = max(start, length - config.local_tokens)
    return start, end


def hash_vectors(vectors: torch.Tensor, projection: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack SimHash into int16, without FP16 integer rounding for K > 11."""
    signs = (vectors @ projection > 0).reshape(*vectors.shape[:-1], -1, bits)
    weights = 1 << torch.arange(bits, device=vectors.device, dtype=torch.int32)
    return (signs.to(torch.int32) * weights).sum(-1).to(torch.int16)


class NativeMagicPIG:
    """The upstream batched LSH/attention implementation, with guarded buffer shapes."""

    def __init__(self, config, layers, qo_heads, kv_heads, head_dim, slots, max_length, pin=True):
        try:
            import minisgl_magicpig_lsh as lsh
            import minisgl_magicpig_sparse as sparse
        except ImportError as exc:
            raise RuntimeError(
                "MagicPIG CPU extensions are missing. Run tools/build_magicpig.py with the "
                "same Python/PyTorch environment as MiniSGL (see docs/magicpig.md)."
            ) from exc
        if lsh.adapter_version != 1 or sparse.adapter_version != 1:
            raise RuntimeError("Rebuild MagicPIG CPU extensions with tools/build_magicpig.py")
        self.config = config
        self.slots, self.qo_heads = slots, qo_heads
        self.max_length = (max_length + 15) // 16 * 16
        self.lsh = lsh.LSH()
        self.lsh.alloc(
            config.hash_bits, config.hash_tables, layers, qo_heads, kv_heads, slots, self.max_length
        )
        self.attention = sparse.SparseAttentionServer()
        self.attention.alloc(layers, qo_heads, kv_heads, head_dim, slots, self.max_length)
        rows = slots * qo_heads
        self.codes = torch.zeros(rows, config.hash_tables, dtype=torch.int32, pin_memory=pin)
        self.query = torch.zeros(rows, head_dim, dtype=torch.bfloat16, pin_memory=pin)
        self.staged_query = torch.empty(self.query.shape, dtype=torch.bfloat16, pin_memory=pin)
        self.staged_codes = torch.empty(self.codes.shape, dtype=torch.int32, pin_memory=pin)
        self.copy_event = None
        self.batch_slots = None
        self.prefix_batch = False
        self.packed_output = torch.empty(
            slots, qo_heads, head_dim, dtype=torch.bfloat16, pin_memory=pin
        )
        self.packed_lse = torch.empty(slots, qo_heads, dtype=torch.float32, pin_memory=pin)
        self.indices = torch.zeros(rows, self.max_length, dtype=torch.int32)
        self.nnz = torch.zeros(rows, dtype=torch.int32)
        self.output = torch.zeros(rows, head_dim, dtype=torch.bfloat16, pin_memory=pin)
        self.stats = torch.zeros(2, rows, dtype=torch.float32, pin_memory=pin)

    def fill(self, layer, slot, keys, values, codes):
        # CPU copies are blocking: native extensions immediately dereference the pointers.
        keys = keys.to(device="cpu", dtype=torch.bfloat16).contiguous()
        values = values.to(device="cpu", dtype=torch.bfloat16).contiguous()
        norms = keys.float().norm(dim=-1).contiguous()
        self.attention.fill(layer, slot, keys, values, norms)
        sorted_codes, indices = codes.sort(dim=-1)
        self.lsh.fill(
            layer,
            slot,
            sorted_codes.cpu().contiguous(),
            indices.to(device="cpu", dtype=torch.int32).contiguous(),
        )

    def set_batch(self, slots):
        slots = tuple(slots)
        if slots == self.batch_slots:
            return
        self.batch_slots = slots
        self.prefix_batch = slots == tuple(range(len(slots)))
        self.slot_indices = torch.tensor(slots, dtype=torch.int64)
        self.inactive = torch.ones(self.slots, dtype=torch.bool)
        self.inactive[self.slot_indices] = False

    def stage(self, queries, codes):
        rows = queries.shape[0] * self.qo_heads
        target_query = self.query if self.prefix_batch else self.staged_query
        target_codes = self.codes if self.prefix_batch else self.staged_codes
        target_query[:rows].copy_(queries.reshape(rows, -1), non_blocking=True)
        target_codes[:rows].copy_(codes.reshape(rows, -1), non_blocking=True)
        if queries.is_cuda:
            if self.copy_event is None:
                self.copy_event = torch.cuda.Event()
            self.copy_event.record()
        else:
            self.copy_event = None

    def decode(self, layer, slots, queries=None, codes=None):
        """Return reusable CPU views, valid until the next decode; H2D must finish before reuse."""
        self.set_batch(slots)
        if queries is not None:
            self.stage(queries, codes)
        if self.copy_event is not None:
            self.copy_event.synchronize()
        count = len(slots)
        if not self.prefix_batch:
            # Preserve packed order for arbitrary scheduler slots, including reordered batches.
            self.query.view(self.slots, self.qo_heads, -1).index_copy_(
                0, self.slot_indices, self.staged_query.view(self.slots, self.qo_heads, -1)[:count]
            )
            self.codes.view(self.slots, self.qo_heads, -1).index_copy_(
                0, self.slot_indices, self.staged_codes.view(self.slots, self.qo_heads, -1)[:count]
            )
        self.lsh.batch_retrieve(layer, self.codes, self.indices, self.nnz)
        if count != self.slots:
            self.nnz.view(self.slots, self.qo_heads)[self.inactive] = 0
        self.attention.attention_wrapper(
            layer,
            self.config.hash_bits,
            self.config.hash_tables,
            self.output,
            self.stats,
            self.query,
            self.query.float().norm(dim=-1),
            self.indices,
            self.nnz,
        )
        # stats[1] is log2(sum(exp(corrected logits))), as expected by flashinfer.merge_state.
        output = self.output.view(self.slots, self.qo_heads, -1)
        lse = self.stats[1].view(self.slots, self.qo_heads)
        if self.prefix_batch:
            return output[:count], lse[:count]
        # Advanced indexing would allocate pageable memory and prevent async H2D.
        torch.index_select(output, 0, self.slot_indices, out=self.packed_output[:count])
        torch.index_select(lse, 0, self.slot_indices, out=self.packed_lse[:count])
        return self.packed_output[:count], self.packed_lse[:count]
