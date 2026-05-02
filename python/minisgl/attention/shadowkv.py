import math
import torch
from dataclasses import dataclass

from minisgl.models.config import ModelConfig
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from minisgl.kernel.shadowkv import (
    shadowkv_score_landmarks_kernel_hd128,
)

from minisgl.kernel import store_cache

from minisgl.shadowkv_kernels import (
    fill_prefill_metadata,
    fill_decode_metadata,
    gather_kv_cache,
    map_to_gpu,
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

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        tp_info = get_tp_info()
        self.local_kv_heads = div_even(
            model_config.num_kv_heads, tp_info.size, allow_replicate=True
        )

        self.max_num_landmarks = max_seq_len // config.chunk_size
        self.gqa_factor = self.model_config.num_qo_heads // self.model_config.num_kv_heads
        self.head_dim = model_config.head_dim

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

        self.prefix_lens = torch.empty((max_batch_size,), dtype=torch.int32, device=self.device)
        self.infix_lens = torch.empty((max_batch_size,), dtype=torch.int32, device=self.device)
        self.pruned_infix_lens = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        )
        self.pruned_seq_lens = torch.empty((max_batch_size,), dtype=torch.int32, device=self.device)
        self.cu_pruned_seq_lens = torch.empty(
            (max_batch_size + 1,), dtype=torch.int32, device=self.device
        )
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
        self.seqlens = [0] * max_batch_size

        # RUNTIME BUFFERS:

        self.selected_chunks = torch.empty(
            (max_batch_size, self.local_kv_heads, self.max_num_landmarks),
            dtype=torch.int64,
            device=self.device,
        ).contiguous()

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

        self.full_kv_buffer = map_to_gpu(
            torch.zeros(
                (
                    2,
                    model_config.num_layers,
                    max_batch_size,
                    max_seq_len,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                # device=self.device,
                device="cpu",
                dtype=self.dtype,
            ).contiguous()
        )

        logger.info(
            f"ShadowkvPool: Allocated {(self.full_kv_buffer.numel() * self.full_kv_buffer.element_size()) / 2**30:.2f} GiB for KV cache"
        )

        self.kv_buffer = torch.empty(
            (2, max_batch_size, max_seq_len, self.local_kv_heads, self.model_config.head_dim),
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self.scores = torch.empty(
            (max_batch_size, self.local_kv_heads, self.max_num_landmarks),
            dtype=dtype,
            device=device,
        ).contiguous()

        self.imag_page_table = torch.stack(
            [
                torch.arange(i * max_seq_len, (i + 1) * max_seq_len, device=device)
                for i in range(max_batch_size)
            ],
            dim=0,
        ).to(dtype=torch.int32)

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, batch_indices: list[int], layer_idx: int):
        indices = None

        if len(batch_indices) == 1 and k.shape[0] > 1:
            indices = torch.arange(0, k.shape[0]) + (batch_indices[0] * self.max_seq_len)
            # if layer_idx == 0:
            # print('PREFILL cache')
            # print(batch_indices)
        else:
            indices = torch.tensor(
                [
                    batch_idx * self.max_seq_len + self.seqlens[batch_idx] - 1
                    for batch_idx in batch_indices
                ]
            )

        indices = indices.to(device=self.device, dtype=torch.int32)

        # if layer_idx == 0:
        #     print(indices)

        store_cache(
            k_cache=self.full_kv_buffer[0, layer_idx].view(
                self.max_batch_size * self.max_seq_len,
                self.local_kv_heads,
                self.model_config.head_dim,
            ),
            v_cache=self.full_kv_buffer[1, layer_idx].view(
                self.max_batch_size * self.max_seq_len,
                self.local_kv_heads,
                self.model_config.head_dim,
            ),
            indices=indices,
            k=k,
            v=v,
        )

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
        fill_prefill_metadata(
            self.prefix_lens,
            self.infix_lens,
            self.pruned_infix_lens,
            torch.Tensor(seqlens).to(device=self.device, dtype=torch.int32, non_blocking=True),
            torch.Tensor(batch_indices).to(
                device=self.device, dtype=torch.int32, non_blocking=True
            ),
            self.config.prefix_budget,
            self.config.sparse_budget,
            self.config.suffix_budget,
            self.config.min_seqlen_to_prune,
            self.config.chunk_size,
        )
        self.prefix_end_indices = self.prefix_lens.cpu().tolist()
        self.suffix_start_indices = (self.prefix_lens.cpu() + self.infix_lens.cpu()).tolist()
        torch.div(
            self.infix_lens,
            self.config.chunk_size,
            rounding_mode="floor",
            out=self.total_num_chunks,
        )
        self.num_chunks_to_select = (
            self.pruned_infix_lens.cpu() // self.config.chunk_size
        ).tolist()

    def prepare_decode_metadata(self, seqlens: list[int], batch_indices: list[int]):
        BS = len(batch_indices)

        self._cpu_batch_indices[:BS] = batch_indices
        self.batch_indices[:BS].copy_(
            torch.tensor(batch_indices, dtype=torch.int32), non_blocking=True
        )

        fill_decode_metadata(
            self.prefix_lens,
            self.infix_lens,
            self.pruned_infix_lens,
            torch.Tensor(seqlens).to(device=self.device, dtype=torch.int32, non_blocking=True),
            self.batch_indices[:BS],
            self.pruned_seq_lens,
            self.cu_pruned_seq_lens,
        )

        for i, batch_idx in enumerate(batch_indices):
            self.seqlens[batch_idx] = seqlens[i]

        return (
            self.cu_pruned_seq_lens[: BS + 1],
            max(seqlens),  # TODO: pruned estimate here
            self.pruned_seq_lens[:BS],
            self.imag_page_table[:BS, : max(seqlens)],
        )

    def select_kv(
        self,
        query_states: torch.Tensor,
        # page_table: torch.Tensor,
        # k_cache: torch.Tensor,
        # v_cache: torch.Tensor,
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
            # logger.info_rank0(f"num_selected_chunks: {num_selected_chunks}")
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

        gather_kv_cache(
            self.prefix_lens,
            self.infix_lens,
            self.pruned_infix_lens,
            self.batch_indices[:BS],
            self.pruned_seq_lens[:BS],
            self.cu_pruned_seq_lens[: BS + 1],
            self.selected_chunks,
            self.full_kv_buffer[0, layer_idx],
            self.full_kv_buffer[1, layer_idx],
            self.kv_buffer[0],
            self.kv_buffer[1],
            self.config.chunk_size,
        )

        return self.kv_buffer[0], self.kv_buffer[1]
