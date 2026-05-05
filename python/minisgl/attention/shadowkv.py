import math
import torch
from dataclasses import dataclass

from minisgl.models.config import ModelConfig
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from minisgl.kernel.shadowkv import (
    shadowkv_score_landmarks_kernel_hd128,
    shadowkv_score_landmarks_kernel_gqa_hd128,
)

from minisgl.kernel import store_cache

from minisgl.shadowkv_kernels import (
    fill_prefill_metadata,
    fill_decode_metadata,
    gather_kv_cache,
    map_to_gpu,
)

from minisgl.quantization.higgs import QuantizedTensor, HiggsQuantizerCUDA

logger = init_logger(__name__)

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}


@dataclass(frozen=False)
class ShadowKVConfig:
    enabled: bool = False
    chunk_size: int = 8
    prefix_budget: float = 0.006125
    sparse_budget: float = 0.125
    suffix_budget: float = 0.06125
    total_budget: float = 0.0
    min_seqlen_to_prune: int = 512
    quantize_landmarks: bool = False
    enable_offloading: bool = False
    gqa_mean_before_scoring: bool = True
    prune_generation: bool = False
    kv_cache_dtype: str | torch.dtype = "bf16"
    landmarks_dtype: str | torch.dtype = "bf16"

    def __post_init__(self):
        self.total_budget = self.prefix_budget + self.sparse_budget + self.suffix_budget
        assert self.total_budget <= 1.0, "Total Budget cannot be greater than 100%"

        assert isinstance(self.kv_cache_dtype, str)
        assert self.kv_cache_dtype in DTYPE_MAP

        self.kv_cache_dtype = DTYPE_MAP[self.kv_cache_dtype]
        self.landmarks_dtype = DTYPE_MAP[self.landmarks_dtype]

        assert not (self.quantize_landmarks and self.prune_generation)

        if self.enabled:
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

        self.gqa_factor = self.model_config.num_qo_heads // self.model_config.num_kv_heads

        tp_info = get_tp_info()
        self.local_kv_heads = div_even(
            model_config.num_kv_heads, tp_info.size, allow_replicate=True
        )
        self.local_qo_heads = self.local_kv_heads * self.gqa_factor

        self.max_num_landmarks = max_seq_len // config.chunk_size
        self.head_dim = model_config.head_dim

        assert (self.model_config.num_qo_heads % self.model_config.num_kv_heads) == 0

        if self.config.prune_generation:
            self.sl_to_prune_generaiton = (
                math.ceil(1 / self.config.sparse_budget) * self.config.chunk_size
            )

            logger.info(f"{self.sl_to_prune_generaiton=}")

        if not self.config.quantize_landmarks:
            self.landmarks_buffer = torch.empty(
                (
                    model_config.num_layers,
                    max_batch_size,
                    self.local_kv_heads,
                    self.max_num_landmarks,
                    model_config.head_dim,
                ),
                device=device,
                dtype=self.config.landmarks_dtype,
            ).contiguous()

            logger.info(
                f"ShadowkvPool: Allocated {(self.landmarks_buffer.numel() * self.landmarks_buffer.element_size()) / 2**30:.2f} GiB for Landmarks ({self.config.landmarks_dtype})"
            )
        else:
            self.edenn_d = 4
            assert (self.local_kv_heads * self.model_config.head_dim % self.edenn_d) == 0

            self.landmark_quantizer = HiggsQuantizerCUDA(
                self.local_kv_heads * self.model_config.head_dim,
                edenn_d=self.edenn_d,
                edenn_n=256,
                dtype=self.dtype,
                device=self.device,
            )

            self.quantized_landmarks_buffer = torch.zeros(
                (
                    self.model_config.num_layers,
                    max_batch_size,
                    self.max_num_landmarks,
                    self.local_kv_heads * self.model_config.head_dim // self.edenn_d,
                ),
                dtype=torch.uint8,
                device=self.device,
            ).contiguous()

            self.quantized_landmarks_scales = torch.zeros(
                (
                    self.model_config.num_layers,
                    max_batch_size,
                    self.max_num_landmarks,
                ),
                dtype=self.dtype,
                device=self.device,
            ).contiguous()

            self.landmarks_buffer = torch.empty(
                (
                    max_batch_size,
                    self.max_num_landmarks,
                    self.local_kv_heads * self.model_config.head_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            ).contiguous()

            logger.info(
                f"ShadowkvPool: Allocated {(self.quantized_landmarks_buffer.numel() * self.quantized_landmarks_buffer.element_size()) / 2**30:.2f} GiB for 2-bit Landmarks"
            )

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
        self.max_num_chunks_to_select = 0

        self.total_num_chunks = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()
        self.batch_indices = torch.empty(
            (max_batch_size,), dtype=torch.int32, device=self.device
        ).contiguous()

        self._cpu_batch_indices = [0] * max_batch_size

        self._cpu_total_num_chunks = [0] * max_batch_size
        self.req_to_prune_generation = [False] * max_batch_size
        self.suffix_lens = [0] * max_batch_size

        self.store_cache_indices = None

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

        if self.config.enable_offloading:
            self.full_kv_buffer = map_to_gpu(
                torch.empty(
                    (
                        2,
                        model_config.num_layers,
                        max_batch_size,
                        max_seq_len,
                        self.local_kv_heads,
                        self.head_dim,
                    ),
                    device="cpu",
                    dtype=self.config.kv_cache_dtype,
                ).contiguous()
            )

            logger.info(
                f"ShadowkvPool: Allocated {(self.full_kv_buffer.numel() * self.full_kv_buffer.element_size()) / 2**30:.2f} GiB for KV cache on CPU ({self.full_kv_buffer.dtype})"
            )
        else:
            self.full_kv_buffer = torch.empty(
                (
                    2,
                    model_config.num_layers,
                    max_batch_size,
                    max_seq_len,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                device=self.device,
                dtype=self.config.kv_cache_dtype,
            ).contiguous()

            logger.info(
                f"ShadowkvPool: Allocated {(self.full_kv_buffer.numel() * self.full_kv_buffer.element_size()) / 2**30:.2f} GiB for KV cache on GPU ({self.full_kv_buffer.dtype})"
            )

        self.kv_buffer = torch.zeros(
            (2, max_batch_size, max_seq_len, self.local_kv_heads, self.model_config.head_dim),
            dtype=self.config.kv_cache_dtype,
            device=self.device,
        ).contiguous()

        self.scores = torch.zeros(
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

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
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
            indices=self.store_cache_indices,
            k=k.to(dtype=self.config.kv_cache_dtype),
            v=v.to(dtype=self.config.kv_cache_dtype),
        )

    def compute_and_store_landmarks(
        self, key_states: torch.Tensor, layer_idx: int, batch_indices: list[int]
    ):
        assert len(batch_indices) == 1

        batch_index = batch_indices[0]
        num_chunks = int(self.total_num_chunks[batch_index])

        if num_chunks == 0:
            return

        # self.scores[batch_index].fill_(float("-inf"))

        key_states_loc = key_states[
            self.prefix_end_indices[batch_index] : self.suffix_start_indices[batch_index]
        ]
        SL = key_states_loc.shape[0]

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

        if not self.config.quantize_landmarks:
            self.landmarks_buffer[layer_idx, batch_index, :, :num_chunks].copy_(
                new_landmarks.to(dtype=self.config.landmarks_dtype)
            )
        else:
            new_landmarks = new_landmarks.transpose(0, 1)
            quantized_landmarks = self.landmark_quantizer.quantize(
                new_landmarks.reshape(
                    num_chunks, self.local_kv_heads * self.model_config.head_dim
                ).contiguous()
            )
            self.quantized_landmarks_buffer[layer_idx, batch_index].index_copy_(
                dim=0,
                index=torch.arange(num_chunks, device=self.device),
                source=quantized_landmarks.idx,
            )
            self.quantized_landmarks_scales[layer_idx, batch_index].index_copy_(
                dim=0,
                index=torch.arange(num_chunks, device=self.device),
                source=quantized_landmarks.scales,
            )

    def prepare_shadowkv_metadata(self, seqlens: list[int], batch_indices: list[int]):
        assert len(batch_indices) == 1
        batch_idx = batch_indices[0]

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
        self.suffix_lens[batch_idx] = seqlens[0] - self.suffix_start_indices[batch_idx]

        torch.div(
            self.infix_lens,
            self.config.chunk_size,
            rounding_mode="floor",
            out=self.total_num_chunks,
        )
        self._cpu_total_num_chunks = self.total_num_chunks.cpu().tolist()
        self.num_chunks_to_select = (
            self.pruned_infix_lens.cpu() // self.config.chunk_size
        ).tolist()

        self.max_num_chunks_to_select = max(self.num_chunks_to_select)

        self.store_cache_indices = (
            torch.arange(0, seqlens[0]) + (batch_indices[0] * self.max_seq_len)
        ).to(device=self.device, dtype=torch.int32, non_blocking=True)

    def prepare_decode_metadata(self, seqlens: list[int], batch_indices: list[int]):
        BS = len(batch_indices)

        self._cpu_batch_indices[:BS] = batch_indices
        self.batch_indices[:BS].copy_(
            torch.tensor(batch_indices, dtype=torch.int32), non_blocking=True
        )

        if self.config.prune_generation:
            for i, batch_idx in enumerate(batch_indices):
                if (
                    seqlens[i] - self.suffix_start_indices[batch_idx] - self.suffix_lens[batch_idx]
                ) >= self.sl_to_prune_generaiton:
                    # print("PRUNING GENERATION")

                    self.suffix_start_indices[batch_idx] += self.sl_to_prune_generaiton
                    self.infix_lens[batch_idx] += self.sl_to_prune_generaiton
                    self.pruned_infix_lens[batch_idx] += (
                        1 * self.config.chunk_size
                    )  # generation pruning happens when 1 chunk is added according to sparse budget
                    self.num_chunks_to_select[batch_idx] += 1
                    self.total_num_chunks[batch_idx] += (
                        self.sl_to_prune_generaiton // self.config.chunk_size
                    )
                    self._cpu_total_num_chunks[batch_idx] += (
                        self.sl_to_prune_generaiton // self.config.chunk_size
                    )
                    self.req_to_prune_generation[batch_idx] = True
                else:
                    self.req_to_prune_generation[batch_idx] = False

        fill_decode_metadata(
            self.prefix_lens,
            self.infix_lens,
            self.pruned_infix_lens,
            torch.Tensor(seqlens).to(device=self.device, dtype=torch.int32, non_blocking=True),
            self.batch_indices[:BS],
            self.pruned_seq_lens,
            self.cu_pruned_seq_lens,
        )

        # print(
        #     self.pruned_seq_lens,
        #     # self.pruned_infix_lens,
        #     # self.total_num_chunks,
        #     self._cpu_total_num_chunks,
        #     # self.num_chunks_to_select,
        #     self.suffix_start_indices,
        #     batch_indices
        # )

        # for i, batch_idx in enumerate(batch_indices):
        #     self.seqlens[batch_idx] = seqlens[i]

        self.store_cache_indices = torch.tensor(
            [
                batch_idx * self.max_seq_len + seqlens[i] - 1
                for i, batch_idx in enumerate(batch_indices)
            ]
        ).to(device=self.device, dtype=torch.int32, non_blocking=True)

        return (
            self.cu_pruned_seq_lens[: BS + 1],
            max(seqlens),  # TODO: pruned estimate here
            self.pruned_seq_lens[:BS],
            self.imag_page_table[:BS, : max(seqlens)],
        )

    def select_kv(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        BS, num_qo_heads, HD = query_states.shape

        if self.config.prune_generation:
            for i in range(BS):
                batch_idx = self._cpu_batch_indices[i]
                if not self.req_to_prune_generation[batch_idx]:
                    continue

                new_landmarks = torch.mean(
                    self.full_kv_buffer[
                        0,
                        layer_idx,
                        batch_idx,
                        self.suffix_start_indices[batch_idx]
                        - self.sl_to_prune_generaiton : self.suffix_start_indices[batch_idx],
                    ]
                    .transpose(0, 1)
                    .view(
                        self.local_kv_heads,
                        self.sl_to_prune_generaiton // self.config.chunk_size,
                        self.config.chunk_size,
                        self.head_dim,
                    ),
                    dim=2,
                )

                self.landmarks_buffer[
                    layer_idx,
                    batch_idx,
                    :,
                    self._cpu_total_num_chunks[batch_idx]
                    - new_landmarks.shape[1] : self._cpu_total_num_chunks[batch_idx],
                ].copy_(new_landmarks.to(self.config.landmarks_dtype))

        # SCORING
        if not self.config.quantize_landmarks:
            landmarks = self.landmarks_buffer[layer_idx]
        else:
            quant_tensor = QuantizedTensor(
                idx=self.quantized_landmarks_buffer[layer_idx].view(
                    self.max_batch_size * self.max_num_landmarks,
                    self.local_kv_heads * self.model_config.head_dim // self.edenn_d,
                ),
                scales=self.quantized_landmarks_scales[layer_idx].view(
                    self.max_batch_size * self.max_num_landmarks,
                ),
            )

            self.landmark_quantizer.full_dequantize(
                quant_tensor,
                self.landmarks_buffer,
            )

            landmarks = self.landmarks_buffer.view(
                self.max_batch_size,
                self.max_num_landmarks,
                self.local_kv_heads,
                self.model_config.head_dim,
            ).transpose(1, 2)

        if self.config.gqa_mean_before_scoring:
            torch.mean(
                query_states.view(BS, self.local_kv_heads, self.gqa_factor, 1, HD),
                dim=2,
                out=self._mean_query_states[:BS],
            )

            shadowkv_score_landmarks_kernel_hd128(
                self._mean_query_states[:BS],
                self.scores,
                landmarks,
                self.local_kv_heads,
                self.total_num_chunks,
                self.max_num_landmarks,
                self.batch_indices[:BS],
            )
        else:
            shadowkv_score_landmarks_kernel_gqa_hd128(
                query_states.view(BS, self.local_kv_heads, self.gqa_factor, HD),
                self.scores,
                landmarks,
                self.local_kv_heads,
                self.total_num_chunks,
                self.max_num_landmarks,
                self.batch_indices[:BS],
                GQA=self.gqa_factor,
            )

        # TOPK
        for i in range(BS):
            batch_idx = self._cpu_batch_indices[i]
            num_selected_chunks = self.num_chunks_to_select[batch_idx]
            # logger.info_rank0(f"num_selected_chunks: {num_selected_chunks}")
            if num_selected_chunks == 0:
                continue

            torch.topk(
                self.scores[batch_idx, :, : self.total_num_chunks[batch_idx]],
                k=num_selected_chunks,
                sorted=False,
                out=(
                    self._topk_values_buffer[:, :num_selected_chunks],
                    self.selected_chunks[batch_idx, :, :num_selected_chunks],
                ),
            )

        # torch.topk(
        #     self.scores,
        #     k=self.max_num_chunks_to_select,
        #     sorted=False,
        #     out=(
        #         self._topk_values_buffer[:, :, : self.max_num_chunks_to_select],
        #         self.selected_chunks[:, :, : self.max_num_chunks_to_select],
        #     ),
        # )

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
        # return self.full_kv_buffer[0, layer_idx], self.full_kv_buffer[1, layer_idx]
