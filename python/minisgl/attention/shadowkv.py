import torch
from dataclasses import dataclass, fields

from minisgl.models.config import ModelConfig
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ShadowKVConfig:
    enabled: bool = False
    chunk_size: int = 8
    prefix_budget: float = 0.006125
    sparse_budget: float = 0.125
    suffix_budget: float = 0.06125

    def __post_init__(self):
        total_budget = self.prefix_budget + self.sparse_budget + self.suffix_budget
        assert total_budget <= 1.0, "Total Budget cannot be greater than 100%"

        logger.info(f"INITTED shadow kv with {self}")

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)


@dataclass
class ShadowKVMetadata:
    prefix_end_indices: torch.LongTensor
    suffix_start_indices: torch.LongTensor
    num_chunks_to_select: torch.LongTensor


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

        tp_info = get_tp_info()
        local_kv_heads = div_even(model_config.num_kv_heads, tp_info.size, allow_replicate=True)

        self.config = config
        self.max_num_landmarks = max_seq_len // config.chunk_size

        self.landmarks_buffer = torch.empty(
            (
                model_config.num_layers,
                max_batch_size,
                self.max_num_landmarks,
                local_kv_heads,
                model_config.head_dim,
            ),
            device=device,
            dtype=dtype,
        )

    def landmarks(self, layer_idx: int) -> torch.Tensor:
        return self.landmarks_buffer[layer_idx]


def prepare_shadowkv_metadata(seqlens: list[int], batch_indices: list[int]):
    pass

