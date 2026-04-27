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

        self.prefix_end_indices = [0] * max_batch_size
        self.suffix_start_indices = [0] * max_batch_size
        self.num_chunks_to_select = [0] * max_batch_size

        self.total_num_chunks = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()
        self.batch_indices = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()

        self._cpu_batch_indices = [0] * max_batch_size

        # RUNTIME BUFFERS:

        self.selected_chunks = torch.empty(
            (max_batch_size, self.local_kv_heads, self.max_num_landmarks),
            dtype=torch.int64,
            device=self.device,
        ).contiguous()
        self.selected_indices = (
            torch.zeros(  # prefix + sparse + suffix KV cache indices in page table
                (max_batch_size, self.local_kv_heads, max_seq_len),
                dtype=torch.int64,
                device=self.device,
            ).contiguous()
        )

        self._topk_values_buffer = torch.empty(
            (self.local_kv_heads, self.max_num_landmarks),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self._mean_query_states = torch.empty(
            (max_batch_size, self.local_kv_heads, 1, self.model_config.head_dim),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self._chunk_size_offset = torch.arange(0, self.config.chunk_size, device=self.device)
        self._arange_buffer = torch.empty((max_seq_len,), dtype=torch.int64, device=self.device)

        self.kv_buffer = torch.empty(
            (2, max_batch_size * max_seq_len, 1, self.local_kv_heads, self.model_config.head_dim),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self.scores = torch.empty(
            (max_batch_size, self.local_kv_heads, self.max_num_landmarks),
            dtype=dtype,
            device=device,
        ).contiguous()

        self.seqlens = [0] * max_batch_size

        self.num_tokens_to_gather = torch.zeros(
            (max_batch_size,),
            dtype=torch.int32,
            device=self.device,
        ).contiguous()

        self.imag_page_table = torch.stack(
            [
                torch.arange(i * max_seq_len, (i + 1) * max_seq_len, device=device)
                for i in range(max_batch_size)
            ],
            dim=0,
        ).to(dtype=torch.int32)

    def compute_and_store_landmarks(
        self, key_states: torch.Tensor, layer_idx: int, batch_indices: list[int]
    ):
        assert len(batch_indices) == 1

        batch_index = batch_indices[0]
        num_chunks = self.total_num_chunks[batch_index]

        if num_chunks == 0:
            return

        key_states_loc = key_states[
            self.prefix_end_indices[batch_index] : self.suffix_start_indices[batch_index]
        ]
        SL = key_states_loc.shape[0]
        # assert SL > 0, f"{key_states.shape} {batch_indices=} {self.prefix_end_indices[batch_index]} {self.suffix_start_indices[batch_index]}"
        # print(layer_idx, SL, SL % self.config.chunk_size)

        key_states_loc = key_states_loc.view(
            SL, self.local_kv_heads, self.model_config.head_dim
        ).transpose(0, 1)

        if self.config.chunk_size > 1:
            key_states_loc = key_states_loc.view(
                self.local_kv_heads, num_chunks, self.config.chunk_size, self.model_config.head_dim
            )

            new_landmarks = torch.mean(key_states_loc, dim=2)
        else:
            new_landmarks = key_states_loc

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
                self.num_chunks_to_select[batch_index] = 0
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
                f"req {batch_index} SL {SL} -> {pruned_sl} ({100 * pruned_sl / SL:.2f}%) {self.prefix_end_indices[batch_index]} {self.suffix_start_indices[batch_index]} {self.num_chunks_to_select[batch_index] * self.config.chunk_size}"
            )

            assert pruned_sl <= SL

            no_suffix_pruned_sl = int(
                self.prefix_end_indices[batch_index]
                + self.num_chunks_to_select[batch_index] * self.config.chunk_size
            )

            suffix_indices_arange = torch.arange(
                self.suffix_start_indices[batch_index], self.max_seq_len, device=self.device
            )

            if len(suffix_indices_arange) != 0:
                indices_arange = torch.arange(
                    no_suffix_pruned_sl,
                    no_suffix_pruned_sl + len(suffix_indices_arange),
                    device=self.device,
                )
                self.selected_indices[batch_index].index_copy_(
                    dim=1,
                    index=indices_arange,
                    source=suffix_indices_arange.unsqueeze(0).expand(
                        self.local_kv_heads, len(suffix_indices_arange)
                    ),
                )

            prefix_end = self.prefix_end_indices[batch_index]
            if prefix_end != 0:
                prefix_arange = torch.arange(0, prefix_end, device=self.device)
                self.selected_indices[batch_index].index_copy_(
                    dim=1,
                    index=prefix_arange,
                    source=prefix_arange.unsqueeze(0).expand(self.local_kv_heads, prefix_end),
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
            self._cpu_batch_indices[i] = batch_idx
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

        torch.mean(
            query_states.view(BS, self.local_kv_heads, self.gqa_factor, 1, HD),
            dim=2,
            out=self._mean_query_states[:BS],
        )
        # assert mean_query_states.is_contiguous()
        # assert mean_query_states.shape == (BS, self.local_kv_heads, 1, HD)

        # SCORING
        shadowkv_score_landmarks_kernel_hd128(
            self._mean_query_states[:BS],
            self.scores,
            self.landmarks_buffer[layer_idx],
            self.local_kv_heads,
            self.total_num_chunks,
            self.max_num_landmarks,
            self.batch_indices[:BS],
        )

        # TOPK
        for i in range(BS):
            batch_idx = self._cpu_batch_indices[i]
            num_selected_chunks = self.num_chunks_to_select[batch_idx]
            if num_selected_chunks == 0:
                continue

            torch.topk(
                self.scores[i, :, : self.total_num_chunks[batch_idx]],
                k=num_selected_chunks,
                sorted=False,
                out=(
                    self._topk_values_buffer[:, :num_selected_chunks],
                    self.selected_chunks[batch_idx, :, :num_selected_chunks],
                ),
            )

            # PREPARE SPARSE INDICES (PREFIX AND SUFFIX ALREADY PREPARED ON PREFILL)

            prefix_end = self.prefix_end_indices[batch_idx]

            sparse_indices = (
                self.selected_chunks[batch_idx, :, :num_selected_chunks] * self.config.chunk_size
                + prefix_end
            ).unsqueeze(-1) + self._chunk_size_offset
            sparse_indices = sparse_indices.reshape(
                self.local_kv_heads, num_selected_chunks * self.config.chunk_size
            )
            self.selected_indices[batch_idx].index_copy_(
                dim=1,
                index=torch.arange(
                    prefix_end,
                    prefix_end + num_selected_chunks * self.config.chunk_size,
                    device=self.device,
                    out=self._arange_buffer[: num_selected_chunks * self.config.chunk_size],
                ),
                source=sparse_indices,
            )

            index = prefix_end + num_selected_chunks * self.config.chunk_size
            num_suffix_tokens = self.seqlens[batch_idx] - self.suffix_start_indices[batch_idx]

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
            self.batch_indices[:BS],
            self.max_seq_len,
        )

        return self.kv_buffer[0], self.kv_buffer[1]
