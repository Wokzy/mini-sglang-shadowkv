from dataclasses import dataclass, field

import torch

from .config import ArkValeConfig
from .kernels import estimate_scores, gather_pages, summarize_keys


@dataclass
class LayerCache:
    host: torch.Tensor
    resident: torch.Tensor
    digests: torch.Tensor
    length: int = 0
    mapping: dict[int, int] = field(default_factory=dict)
    recalls: int = 0
    evictions: int = 0
    score_steps: int = 0


class ArkValePool:
    """Write-through CPU backing store and bounded GPU cache per request/layer.

    Synchronous copies deliberately run on the current engine stream. No tensor
    is shared with an in-flight next batch; overlap/graphs are rejected up front.
    The scheduler's token locations are accounting only, not GPU KV addresses.
    """

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
        # One gathered decode workspace is live at a time.
        required_gpu = (
            per_state_gpu * self.num_layers * max_requests + unit * self.capacity * config.page_size
        )
        free, _ = torch.cuda.mem_get_info(device)
        if required_gpu > int(free * memory_ratio):
            raise ValueError(
                "ArkVale resident pages/digests exceed the GPU memory ratio; reduce page_budget or max requests"
            )

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
        p, h, d = self.config.page_size, self.heads, self.head_dim
        state = LayerCache(
            host=torch.empty(
                (2, self.max_pages, p, h, d), device="cpu", dtype=self.dtype, pin_memory=True
            ),
            resident=torch.empty((2, self.capacity, p, h, d), device=self.device, dtype=self.dtype),
            digests=torch.empty((self.max_pages, 2, h, d), device=self.device, dtype=self.dtype),
        )
        self.states[key] = state
        self.host_bytes += self.state_bytes
        return state

    def release(self, uid):
        stats = {"recalls": 0, "evictions": 0, "score_steps": 0}
        for key in [key for key in self.states if key[0] == uid]:
            state = self.states.pop(key)
            self.host_bytes -= self.state_bytes
            for name in stats:
                stats[name] += getattr(state, name)
        return stats

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

    def _select(self, state, query, initial=False):
        pages = (state.length + self.config.page_size - 1) // self.config.page_size
        if pages <= self.capacity:
            desired = set(range(pages))
        else:
            desired = self._protected(pages)
            scores = estimate_scores(query.contiguous(), state.digests[:pages])
            protected = torch.tensor(sorted(desired), device=self.device, dtype=torch.long)
            scores.index_fill_(0, protected, -float("inf"))
            requested = self.capacity if initial else self.config.page_topk
            count = min(requested - len(desired), pages - len(desired))
            desired.update(torch.topk(scores, count).indices.tolist())
            state.score_steps += 1
        self._ensure(state, desired)

    def prefill(self, uid, layer, keys, values, last_query):
        state = self.create(uid, layer)
        n = keys.shape[0]
        if n > self.max_seq_len:
            raise ValueError("ArkVale sequence exceeds configured maximum")
        # Blocking D2H makes the CPU mirror immediately safe to read for recall.
        state.host[0].view(-1, self.heads, self.head_dim)[:n].copy_(keys, non_blocking=False)
        state.host[1].view(-1, self.heads, self.head_dim)[:n].copy_(values, non_blocking=False)
        summarize_keys(keys.contiguous(), self.config.page_size, out=state.digests)
        state.length = n
        self._select(state, last_query, initial=True)

    def decode(self, uid, layer, key, value, query, expected_length):
        state = self.states[(uid, layer)]
        if expected_length != state.length + 1 or expected_length > self.max_seq_len:
            raise RuntimeError("ArkVale request/cache sequence lengths disagree")
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
        self._select(state, query)
        logical = sorted(state.mapping)
        slots = torch.tensor(
            [state.mapping[i] for i in logical], device=self.device, dtype=torch.int32
        )
        # Current page is always the last logical page; all other pages are full.
        count = (len(logical) - 1) * p + offset + 1
        return gather_pages(state.resident, slots, count)
