import math
import torch
from dataclasses import dataclass, fields

from minisgl.models.config import ModelConfig
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

logger = init_logger(__name__)


@dataclass(frozen=False)
class ShadowKVConfig:
    enabled: bool = False
    chunk_size: int = 8
    prefix_budget: float = 0.006125
    sparse_budget: float = 0.125
    suffix_budget: float = 0.06125
    total_budget: float = 0.0
    min_seqlen_to_prune: int = 512

    def __post_init__(self):
        self.total_budget = self.prefix_budget + self.sparse_budget + self.suffix_budget
        assert self.total_budget <= 1.0, "Total Budget cannot be greater than 100%"

        logger.info(f"INITTED shadow kv with {self}")

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)


class ShadowKVPool:
    def __init__(
        self,
        config: ShadowKVConfig,
        model_config: ModelConfig,
        max_batch_size: int,
        max_seq_len: int,
        device,
        dtype,
    ):
        if not config.enabled:
            raise RuntimeError("INITTING ShadowKV pool with shadowkv disabled is prohibited!")

        self.config = config
        self.model_config = model_config
        self.device = device
        self.dtype = dtype

        tp_info = get_tp_info()
        self.local_kv_heads = div_even(
            model_config.num_kv_heads, tp_info.size, allow_replicate=True
        )

        self.max_num_landmarks = max_seq_len // config.chunk_size

        self.landmarks_buffer = torch.empty(
            (
                model_config.num_layers,
                max_batch_size,
                self.local_kv_heads,
                self.max_num_landmarks,
                model_config.head_dim,
            ),
            device=device,
            dtype=dtype,
        )

        self.prefix_end_indices = torch.empty((max_batch_size,), dtype=torch.long, device="cpu")
        self.suffix_start_indices = torch.empty((max_batch_size,), dtype=torch.long, device="cpu")
        self.num_chunks_to_select = torch.empty((max_batch_size,), dtype=torch.long, device="cpu")
        self.batch_seqlens = torch.empty((max_batch_size,), dtype=torch.long, device="cpu")

    def landmarks(self, layer_idx: int) -> torch.Tensor:
        return self.landmarks_buffer[layer_idx]

    def compute_and_store_landmarks(
        self, key_states: torch.Tensor, layer_idx: int, batch_indices: list[int]
    ):
        assert len(batch_indices) == 1
        batch_index = batch_indices[0]

        if self.prefix_end_indices[batch_index] == self.suffix_start_indices[batch_index]:
            return

        key_states = key_states[self.prefix_end_indices[batch_index] : self.suffix_start_indices[batch_index]]
        SL = key_states.shape[0]
        # print(layer_idx, SL, SL % self.config.chunk_size)

        num_chunks = SL // self.config.chunk_size

        key_states = key_states.view(SL, self.local_kv_heads, self.model_config.head_dim).transpose(
            0, 1
        )
        key_states = key_states.view(
            self.local_kv_heads, num_chunks, self.config.chunk_size, self.model_config.head_dim
        )

        if layer_idx == 3:
            print(key_states.shape)
            print(self.landmarks_buffer[layer_idx, batch_index])

        new_landmarks = torch.mean(key_states, dim=2)
        self.landmarks_buffer[layer_idx, batch_index].index_copy_(
            dim=1, index=torch.arange(num_chunks, device=self.device), source=new_landmarks
        )

        if layer_idx == 3:
            print(self.landmarks_buffer[layer_idx, batch_index])

    def prepare_shadowkv_metadata(self, seqlens: list[int], batch_indices: list[int]):
        for SL, batch_index in zip(seqlens, batch_indices):

            if (SL * self.config.total_budget < self.config.min_seqlen_to_prune) or self.config.total_budget == 1.0:
                self.prefix_end_indices[batch_index] = 0
                self.suffix_start_indices[batch_index] = 0
            else:
                prefix_idx = math.floor(SL * self.config.prefix_budget)
                suffix_idx = math.floor(SL - SL * self.config.prefix_budget)

                sparse_gap = suffix_idx - prefix_idx
                suffix_idx -= (sparse_gap % self.config.chunk_size)
                sparse_gap = suffix_idx - prefix_idx

                self.num_chunks_to_select[batch_index] = math.ceil(min(SL * self.config.sparse_budget, sparse_gap)) // self.config.chunk_size
                self.prefix_end_indices[batch_index] = prefix_idx
                self.suffix_start_indices[batch_index] = suffix_idx

            print(f'{SL=}')

        print(f'{self.prefix_end_indices=} {self.suffix_start_indices=} {self.num_chunks_to_select=}')
