from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.magicpig import MagicPIGConfig
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

from minisgl.attention import ShadowKVConfig


@dataclass(frozen=False)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    additional_config_path: str | None = None
    shadowkv_config: ShadowKVConfig | None = None
    magicpig_config: MagicPIGConfig | None = None

    def __post_init__(self):
        if self.additional_config_path is not None:
            with open(self.additional_config_path, "r") as f:
                additional_config_dict = json.load(f)

            if (
                shadowkv_config_dict := additional_config_dict.get("shadowkv_config", None)
            ) is not None:
                self.shadowkv_config = ShadowKVConfig.from_dict(shadowkv_config_dict)

            if "magicpig_config" in additional_config_dict:
                self.magicpig_config = MagicPIGConfig.from_dict(
                    additional_config_dict["magicpig_config"]
                )

        if "magicpig" in self.attention_backend.split(",") or self.magicpig_config is not None:
            if self.attention_backend not in ("auto", "magicpig"):
                raise ValueError(
                    "MagicPIG requires --attention-backend magicpig (no hybrid backend)"
                )
            if self.shadowkv_config is not None and self.shadowkv_config.enabled:
                raise ValueError("MagicPIG and ShadowKV cannot be enabled together")
            if self.page_size != 1:
                raise ValueError("MagicPIG requires page_size=1")
            if self.dtype != torch.bfloat16:
                raise ValueError("MagicPIG requires --dtype bfloat16")
            self.magicpig_config = self.magicpig_config or MagicPIGConfig()
            self.attention_backend = "magicpig"
            self.cuda_graph_bs = []
            self.cuda_graph_max_bs = 0
            if getattr(self, "cache_type", "naive") != "naive":
                raise ValueError("MagicPIG requires --cache-type naive")

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
