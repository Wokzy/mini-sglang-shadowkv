from dataclasses import dataclass, field

import torch

from .config import ArkValeConfig
from .kernels import estimate_scores, gather_pages, summarize_keys


@dataclass
class LayerCache:
    host: torch.Tensor
    resident: torch.Tensor
    digests: torch.Tensor
    # Scalar reference bookkeeping; GPU serving uses GPUDecode's tensors instead.
    length: int = 0
    mapping: dict[int, int] = field(default_factory=dict)
    recalls: int = 0
    evictions: int = 0
    score_steps: int = 0


class ArkValePool:
    """Write-through CPU backing store and bounded GPU cache per request/layer.

    Storage is preallocated by scheduler slot; UIDs only track live ownership.
    GPU serving uses a transfer stream with explicit engine dependencies. The
    scalar methods retain the synchronous reference path for direct callers.
    The scheduler orders admission after the preceding forward, as for ShadowKV.
    Logical scheduler token locations are not GPU KV addresses.
    """

    uses_shared_workspace = True

    def __init__(
        self,
        config: ArkValeConfig,
        model_config,
        max_seq_len,
        device,
        dtype,
        max_requests=1,
        memory_ratio=0.9,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim
        self.num_layers = model_config.num_layers
        self.max_seq_len = max_seq_len
        self.max_pages = (max_seq_len + config.page_size - 1) // config.page_size
        self.capacity = min(self.max_pages, config.page_budget)
        if config.gpu_decode and self.max_pages > 32768:
            raise ValueError(
                "ArkVale GPU bookkeeping supports at most 32768 logical pages per request; "
                "increase page_size or set gpu_decode=false for the reference path"
            )
        self.states: dict[tuple[int, int], LayerCache] = {}
        self.host_bytes = 0
        required_host = self.state_bytes * self.num_layers * max_requests
        if required_host > config.max_cpu_bytes:
            raise ValueError(
                f"ArkVale needs up to {required_host} CPU bytes for this request/sequence limit; "
                "increase max_cpu_bytes or reduce --max-running-requests/--max-seq-len-override"
            )
        # Reserve room for all admitted requests' resident pages and digests.
        unit = 2 * self.heads * self.head_dim * dtype.itemsize
        per_state_gpu = unit * (self.capacity * config.page_size + self.max_pages)
        # One batched gathered workspace is shared across layers. Include the
        # reusable score and page-index scratch storage in the admission check.
        required_gpu = (
            per_state_gpu * self.num_layers * max_requests
            + unit * self.capacity * config.page_size * max_requests
            + max_requests * (self.max_pages * 4 + self.capacity * 4)
        )
        if config.gpu_decode:
            required_gpu += max_requests * (
                self.num_layers * (self.capacity * 12 + 8 + 24 + 8) + self.capacity * 8 + 4
            )
        free, _ = torch.cuda.mem_get_info(device)
        if required_gpu > int(free * memory_ratio):
            raise ValueError(
                "ArkVale resident pages/digests/workspaces exceed the GPU memory ratio; "
                "reduce page_budget or max requests"
            )

        self.max_requests = max_requests
        self.uid_slots: dict[int, int] = {}
        # Scheduler-side sequence cursors, checked once per batch (not per layer).
        self.request_lengths: dict[int, int] = {}
        self.slot_uids: list[int | None] = [None] * max_requests
        self.allocated_host_bytes = required_host
        p, h, d = config.page_size, self.heads, self.head_dim
        # Separate pinned allocations avoid registering one enormous CPU range.
        # These buffers survive completion/abort and are overwritten on reuse.
        self.host_buffers = [
            [
                torch.empty(
                    (2, self.max_pages, p, h, d), device="cpu", dtype=dtype, pin_memory=True
                )
                for _ in range(self.num_layers)
            ]
            for _ in range(max_requests)
        ]
        self.resident_buffer = torch.empty(
            (max_requests, self.num_layers, 2, self.capacity, p, h, d),
            device=device,
            dtype=dtype,
        )
        self.digest_buffer = torch.empty(
            (max_requests, self.num_layers, self.max_pages, 2, h, d),
            device=device,
            dtype=dtype,
        )
        self.gathered_buffer = torch.zeros(
            (2, max_requests, self.capacity * p, h, d),
            device=device,
            dtype=dtype,
        )
        self.score_buffer = torch.empty(
            (max_requests, self.max_pages),
            device=device,
            dtype=torch.float32,
        )
        self.page_indices = torch.empty(
            (max_requests, self.capacity),
            device=device,
            dtype=torch.int32,
        )

        self.gpu = None
        if config.gpu_decode:
            from .gpu import GPUDecode

            self.gpu = GPUDecode(self)

    def bind(self, uid, slot=None):
        """Bind a live UID to its scheduler slot (automatic for direct callers)."""
        if uid in self.uid_slots:
            existing = self.uid_slots[uid]
            if slot is not None and slot != existing:
                raise RuntimeError("ArkVale request changed cache slot")
            return existing
        if slot is None:
            slot = next((i for i, owner in enumerate(self.slot_uids) if owner is None), None)
        if slot is None or not 0 <= slot < self.max_requests:
            raise RuntimeError("ArkVale has no available request slot")
        if self.slot_uids[slot] is not None:
            raise RuntimeError("ArkVale cache slot is still owned by another request")
        self.uid_slots[uid] = slot
        self.slot_uids[slot] = uid
        return slot

    @property
    def state_bytes(self):
        return (
            2
            * self.max_pages
            * self.config.page_size
            * self.heads
            * self.head_dim
            * self.dtype.itemsize
        )

    def create(self, uid, layer):
        key = (uid, layer)
        if key in self.states:
            raise RuntimeError(
                "ArkVale received a repeated prefill (prefix reuse/chunking unsupported)"
            )
        if self.host_bytes + self.state_bytes > self.config.max_cpu_bytes:
            raise RuntimeError(
                "ArkVale CPU backing budget exhausted; reduce max requests/sequence length or increase max_cpu_bytes"
            )
        slot = self.bind(uid)
        state = LayerCache(
            host=self.host_buffers[slot][layer],
            resident=self.resident_buffer[slot, layer],
            digests=self.digest_buffer[slot, layer],
        )
        self.states[key] = state
        self.host_bytes += self.state_bytes
        return state

    def release(self, uid):
        # Ownership only: never clear buffers that a submitted forward may still
        # read. Reinitialization writes are ordered on the engine stream.
        stats = {"recalls": 0, "evictions": 0, "score_steps": 0}
        keys = [key for key in self.states if key[0] == uid]
        gpu_stats = None
        if self.gpu is not None and keys:
            gpu_stats = self.gpu.statistics(self.uid_slots[uid], [key[1] for key in keys])
        for key in keys:
            state = self.states.pop(key)
            self.host_bytes -= self.state_bytes
            for name in stats:
                stats[name] += getattr(state, name)
        self.request_lengths.pop(uid, None)
        slot = self.uid_slots.pop(uid, None)
        if slot is not None:
            self.slot_uids[slot] = None
        return gpu_stats if gpu_stats is not None else stats

    def _protected(self, pages):
        return set(range(min(self.config.sink_pages, pages))) | set(
            range(max(0, pages - self.config.window_pages), pages)
        )

    def _ensure(self, state, desired):
        """Keep cache hits; only recall misses, evicting outside the requested set."""
        for page in sorted(desired):
            if page in state.mapping:
                continue
            used = set(state.mapping.values())
            free = next((slot for slot in range(self.capacity) if slot not in used), None)
            if free is None:
                victim = next((old for old in state.mapping if old not in desired), None)
                if victim is None:
                    raise RuntimeError("ArkVale requested more pages than the resident budget")
                free = state.mapping.pop(victim)
                state.evictions += 1
            state.resident[:, free].copy_(state.host[:, page], non_blocking=False)
            state.mapping[page] = free
            state.recalls += 1

    def _select(self, state, query, initial=False, slot=0):
        pages = (state.length + self.config.page_size - 1) // self.config.page_size
        if pages <= self.capacity:
            desired = set(range(pages))
        else:
            desired = self._protected(pages)
            scores = estimate_scores(
                query.contiguous(), state.digests[:pages], out=self.score_buffer[slot, :pages]
            )
            protected = torch.tensor(sorted(desired), device=self.device, dtype=torch.long)
            scores.index_fill_(0, protected, -float("inf"))
            requested = self.capacity if initial else self.config.page_topk
            count = min(requested - len(desired), pages - len(desired))
            desired.update(torch.topk(scores, count).indices.tolist())
            state.score_steps += 1
        self._ensure(state, desired)

    def prefill(self, uid, layer, keys, values, last_query, *, gpu=False):
        n = keys.shape[0]
        if n > self.max_seq_len:
            raise ValueError("ArkVale sequence exceeds configured maximum")
        state = self.create(uid, layer)
        self.request_lengths[uid] = n
        if gpu and self.gpu is not None:
            state.length = n
            self.gpu.prefill(uid, layer, state, keys, values, last_query)
            return
        # Blocking D2H makes the CPU mirror immediately safe to read for recall.
        state.host[0].view(-1, self.heads, self.head_dim)[:n].copy_(keys, non_blocking=False)
        state.host[1].view(-1, self.heads, self.head_dim)[:n].copy_(values, non_blocking=False)
        summarize_keys(keys.contiguous(), self.config.page_size, out=state.digests)
        state.length = n
        self._select(state, last_query, initial=True, slot=self.uid_slots[uid])
        if self.gpu is not None:
            self.gpu.initialize(uid, layer, state)

    def decode(self, uid, layer, key, value, query, expected_length):
        """Return a borrowed packed-KV view, overwritten on this slot's next decode."""
        state = self.states[(uid, layer)]
        if expected_length != state.length + 1 or expected_length > self.max_seq_len:
            raise RuntimeError("ArkVale request/cache sequence lengths disagree")
        self.request_lengths[uid] = expected_length
        p = self.config.page_size
        page, offset = divmod(state.length, p)
        state.host[0, page, offset].copy_(key, non_blocking=False)
        state.host[1, page, offset].copy_(value, non_blocking=False)
        state.length += 1
        # The new/current page must be resident before writing and scoring.
        self._ensure(state, self._protected(page + 1))
        slot = state.mapping[page]
        state.resident[0, slot, offset].copy_(key)
        state.resident[1, slot, offset].copy_(value)
        if offset == p - 1:
            summarize_keys(state.resident[0, slot], p, out=state.digests[page : page + 1])
        request_slot = self.uid_slots[uid]
        self._select(state, query, slot=request_slot)
        logical = sorted(state.mapping)
        slots = self.page_indices[request_slot, : len(logical)]
        slots.copy_(torch.tensor([state.mapping[i] for i in logical], dtype=torch.int32))
        # Current page is always the last logical page; all other pages are full.
        count = (len(logical) - 1) * p + offset + 1
        if self.gpu is not None:
            self.gpu.initialize(uid, layer, state)
        return gather_pages(
            state.resident,
            slots,
            count,
            out=self.gathered_buffer[:, request_slot, :count],
        )
