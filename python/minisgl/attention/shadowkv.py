import math
import torch
from dataclasses import dataclass

from minisgl.models.config import ModelConfig
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from minisgl.kernel.shadowkv import (
    shadowkv_score_landmarks_kernel_hd128,
    shadowkv_gather_kv_cache_kernel_hd128,
)

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

        self.max_seq_len = max_seq_len

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

        assert self.model_config.head_dim == 128, "Supported head dims are: [128]"

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
        ).contiguous()
        self.selected_indices = torch.zeros(
            (max_batch_size, self.local_kv_heads, max_seq_len),
            dtype=torch.int64,
            device=self.device,
        ).contiguous()

        self.kv_buffer = torch.empty(
            (2, max_batch_size * max_seq_len, 1, self.local_kv_heads, self.model_config.head_dim),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self.seqlens = torch.zeros(
            (max_batch_size,),
            dtype=torch.int32,
            device=self.device,
        ).contiguous()

        self.num_tokens_to_gather = torch.zeros(
            (max_batch_size,),
            dtype=torch.int32,
            device=self.device,
        ).contiguous()

        self.imag_page_table = (
            torch.arange(
                0,
                max_seq_len,
                device=self.device,
            )
            .to(dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, max_seq_len)
        )

    def landmarks(self, layer_idx: int) -> torch.Tensor:
        return self.landmarks_buffer[layer_idx]

    def compute_and_store_landmarks(
        self, key_states: torch.Tensor, layer_idx: int, batch_indices: list[int]
    ):
        assert len(batch_indices) == 1, key_states.shape
        batch_index = batch_indices[0]
        num_chunks = self.total_num_chunks[batch_index]

        if num_chunks == 0:
            return

        key_states = key_states[
            self.prefix_end_indices[batch_index] : self.suffix_start_indices[batch_index]
        ]
        SL = key_states.shape[0]
        # print(layer_idx, SL, SL % self.config.chunk_size)

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
                suffix_idx = math.floor(SL - SL * self.config.suffix_budget)

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

            pruned_sl = int(
                self.prefix_end_indices[batch_index]
                + SL
                - self.suffix_start_indices[batch_index]
                + self.num_chunks_to_select[batch_index] * self.config.chunk_size
            )
            logger.info_rank0(
                f"req {batch_index} SL {SL} -> {pruned_sl} ({100 * pruned_sl / SL:.2f}%)"
            )

        # print(
        #     f"{self.prefix_end_indices=} {self.suffix_start_indices=} {self.num_chunks_to_select=}"
        # )

    def prepare_decode_metadata(self, seqlens: list[int], batch_indices: list[int]):
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        seqlens_k = []
        max_seqlen_k = 0

        for i, batch_idx in enumerate(batch_indices):
            self.batch_indices[i] = batch_idx
            self.seqlens[batch_idx] = seqlens[i]

            prefix_end = self.prefix_end_indices[batch_idx]
            num_selected_chunks = self.num_chunks_to_select[batch_idx]
            num_suffix_tokens = self.seqlens[batch_idx] - self.suffix_start_indices[batch_idx]

            pruned_sl = (
                prefix_end + num_selected_chunks * self.config.chunk_size + num_suffix_tokens
            )
            self.num_tokens_to_gather[batch_idx] = pruned_sl
            seqlens_k.append(pruned_sl)

        max_seqlen_k = max(seqlens_k)

        return (
            torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0),
            max_seqlen_k,
            torch.tensor(seqlens_k, **CPU_KWARGS),
            self.imag_page_table[: len(batch_indices), :max_seqlen_k],
        )

    def select_kv(
        self,
        query_states: torch.Tensor,
        page_table: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        BS, num_qo_heads, HD = query_states.shape
        query_states = query_states.view(BS, 1, num_qo_heads, HD).transpose(1, 2)
        query_states = query_states.view(BS, self.local_kv_heads, self.gqa_factor, 1, HD)

        mean_query_states = torch.mean(query_states, dim=2).contiguous()
        assert mean_query_states.shape == (BS, self.local_kv_heads, 1, HD)

        # SCORING
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

        # PREPARE INDICES
        for i in range(BS):
            batch_idx = self.batch_indices[i]
            prefix_end = self.prefix_end_indices[batch_idx]

            if prefix_end != 0:
                prefix_arange = torch.arange(0, prefix_end, device=self.device)
                self.selected_indices[batch_idx].index_copy_(
                    dim=1,
                    index=prefix_arange,
                    source=prefix_arange.unsqueeze(0).expand(self.local_kv_heads, prefix_end),
                )

            num_selected_chunks = self.num_chunks_to_select[batch_idx]

            if num_selected_chunks != 0:
                offsets = torch.arange(0, self.config.chunk_size, device=self.device)
                sparse_indices = (
                    self.selected_chunks[batch_idx][:, :num_selected_chunks]
                    * self.config.chunk_size
                    + prefix_end
                ).unsqueeze(-1) + offsets
                sparse_indices = sparse_indices.reshape(
                    self.local_kv_heads, num_selected_chunks * self.config.chunk_size
                )
                self.selected_indices[batch_idx].index_copy_(
                    dim=1,
                    index=torch.arange(
                        prefix_end,
                        prefix_end + num_selected_chunks * self.config.chunk_size,
                        device=self.device,
                    ),
                    source=sparse_indices,
                )

            index = prefix_end + num_selected_chunks * self.config.chunk_size
            num_suffix_tokens = self.seqlens[batch_idx] - self.suffix_start_indices[batch_idx]

            if num_suffix_tokens != 0:
                suffix_arange = torch.arange(index, index + num_suffix_tokens, device=self.device)
                suffix_index_arange = torch.arange(
                    self.suffix_start_indices[batch_idx],
                    self.seqlens[batch_idx],
                    device=self.device,
                )

                self.selected_indices[batch_idx].index_copy_(
                    dim=1,
                    index=suffix_arange,
                    source=suffix_index_arange.unsqueeze(0).expand(
                        self.local_kv_heads, num_suffix_tokens
                    ),
                )

            assert self.num_tokens_to_gather[batch_idx] == index + num_suffix_tokens

            # if layer_idx == 3:
            #     print(self.selected_indices[batch_idx, :, : index + num_suffix_tokens])

        # GATHER
        shadowkv_gather_kv_cache_kernel_hd128(
            page_table,
            k_cache,
            v_cache,
            self.kv_buffer[0],
            self.kv_buffer[1],
            self.local_kv_heads,
            self.selected_indices,
            self.num_tokens_to_gather,
            self.batch_indices,
            self.max_seq_len,
        )

        return self.kv_buffer[0], self.kv_buffer[1]
