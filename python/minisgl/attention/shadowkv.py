import math
import torch
from dataclasses import dataclass, fields

from minisgl.models.config import ModelConfig
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from minisgl.kernel.shadowkv import shadowkv_score_landmarks_kernel_hd128

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
        self.model_config: ModelConfig = model_config
        self.device = device
        self.dtype = dtype

        tp_info = get_tp_info()
        self.local_kv_heads = div_even(
            model_config.num_kv_heads, tp_info.size, allow_replicate=True
        )

        self.max_num_landmarks = max_seq_len // config.chunk_size
        self.gqa_factor = self.model_config.num_qo_heads // self.model_config.num_kv_heads

        assert (self.model_config.num_qo_heads % self.model_config.num_kv_heads) == 0

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
        ).contiguous()

        self.prefix_end_indices = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()
        self.suffix_start_indices = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()
        self.num_chunks_to_select = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()
        self.total_num_chunks = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()
        self.batch_indices = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()

        self.selected_chunks = torch.empty(
            (max_batch_size, self.local_kv_heads, self.max_num_landmarks),
            dtype=torch.int64,
            device=self.device,
        )

        # self.cuda_total_num_chunks = None
        # self.cuda_prefix_end_indices = None
        # self.cuda_s

    def landmarks(self, layer_idx: int) -> torch.Tensor:
        return self.landmarks_buffer[layer_idx]

    def compute_and_store_landmarks(
        self, key_states: torch.Tensor, layer_idx: int, batch_indices: list[int]
    ):
        assert len(batch_indices) == 1, key_states.shape
        batch_index = batch_indices[0]

        if self.prefix_end_indices[batch_index] == self.suffix_start_indices[batch_index]:
            return

        key_states = key_states[
            self.prefix_end_indices[batch_index] : self.suffix_start_indices[batch_index]
        ]
        SL = key_states.shape[0]
        # print(layer_idx, SL, SL % self.config.chunk_size)

        num_chunks = self.total_num_chunks[batch_index]

        key_states = key_states.view(SL, self.local_kv_heads, self.model_config.head_dim).transpose(
            0, 1
        )

        if self.config.chunk_size > 1:
            key_states = key_states.view(
                self.local_kv_heads, num_chunks, self.config.chunk_size, self.model_config.head_dim
            )

            # if layer_idx == 3:
            #     print(key_states.shape)
            #     print(self.landmarks_buffer[layer_idx, batch_index])

            new_landmarks = torch.mean(key_states, dim=2)
        else:
            new_landmarks = key_states

        self.landmarks_buffer[layer_idx, batch_index].index_copy_(
            dim=1, index=torch.arange(num_chunks, device=self.device), source=new_landmarks
        )

        # if layer_idx == 3:
        #     print(self.landmarks_buffer[layer_idx, batch_index])

    def prepare_shadowkv_metadata(self, seqlens: list[int], batch_indices: list[int]):
        for SL, batch_index in zip(seqlens, batch_indices):
            if (
                SL * self.config.total_budget < self.config.min_seqlen_to_prune
            ) or self.config.total_budget == 1.0:
                self.prefix_end_indices[batch_index] = 0
                self.suffix_start_indices[batch_index] = 0
                self.total_num_chunks[batch_index] = 0
            else:
                prefix_idx = math.floor(SL * self.config.prefix_budget)
                suffix_idx = math.floor(SL - SL * self.config.prefix_budget)

                sparse_gap = suffix_idx - prefix_idx
                suffix_idx -= sparse_gap % self.config.chunk_size
                sparse_gap = suffix_idx - prefix_idx

                self.num_chunks_to_select[batch_index] = (
                    math.ceil(min(SL * self.config.sparse_budget, sparse_gap))
                    // self.config.chunk_size
                )
                self.prefix_end_indices[batch_index] = prefix_idx
                self.suffix_start_indices[batch_index] = suffix_idx
                self.total_num_chunks[batch_index] = sparse_gap // self.config.chunk_size

            print(f"{SL=}")

        print(
            f"{self.prefix_end_indices=} {self.suffix_start_indices=} {self.num_chunks_to_select=}"
        )

    def prepare_batch_indices(self, batch_indices: list[int]):
        for i, batch_idx in enumerate(batch_indices):
            self.batch_indices[i] = batch_idx

    def select_kv(self, query_states: torch.Tensor, layer_idx: int):
        BS, num_qo_heads, HD = query_states.shape
        query_states = query_states.view(BS, 1, num_qo_heads, HD).transpose(1, 2)
        query_states = query_states.view(BS, self.local_kv_heads, self.gqa_factor, 1, HD)

        mean_query_states = torch.mean(query_states, dim=2).contiguous()
        assert mean_query_states.shape == (BS, self.local_kv_heads, 1, HD)

        # SCORING
        assert self.model_config.head_dim == 128, "Supported head dims are: [128]"
        scores = shadowkv_score_landmarks_kernel_hd128(
            mean_query_states,
            self.landmarks_buffer[layer_idx],
            self.local_kv_heads,
            self.total_num_chunks,
            self.max_num_landmarks,
            self.batch_indices,
            device=self.device,
            dtype=self.dtype,
        )

        # TOPK
        for i in range(BS):
            batch_idx = self.batch_indices[i]
            num_selected_chunks = self.num_chunks_to_select[batch_idx]
            if num_selected_chunks == 0:
                continue

            self.selected_chunks[batch_idx].index_copy_(
                dim=1,
                index=torch.arange(0, num_selected_chunks, device=self.device),
                source=torch.topk(
                    scores[i][:, : self.total_num_chunks[batch_idx]],
                    k=num_selected_chunks,
                    sorted=False,
                ).indices,
            )
