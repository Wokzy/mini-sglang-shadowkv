import logging
import math
from dataclasses import dataclass, replace

import torch

logger = logging.getLogger(__name__)


@dataclass
class ArkValeConfig:
    enabled: bool = True
    # Disable for the synchronous Python reference / matched ablations.
    gpu_decode: bool = True
    page_size: int = 32
    # Integers are page counts; floats in (0, 1] are fractions of prompt pages.
    page_budget: int | float = 128
    # Total pages requested each step, including sink and window pages.
    # Remaining resident pages are retained and also attended (upstream's default).
    page_topk: int | float = 64
    sink_pages: int = 2
    window_pages: int = 2
    max_cpu_bytes: int = 8 * 1024**3

    def __post_init__(self):
        for name in ("enabled", "gpu_decode"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"arkvale.{name} must be a boolean")
        for name in (
            "page_size",
            "sink_pages",
            "window_pages",
            "max_cpu_bytes",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"arkvale.{name} must be an integer")
        if self.page_size < 1 or self.page_size > 128 or self.page_size & (self.page_size - 1):
            raise ValueError("arkvale.page_size must be a power of two in [1, 128]")
        if self.sink_pages < 0 or self.window_pages < 2:
            raise ValueError("ArkVale requires sink_pages >= 0 and window_pages >= 2")
        for name in ("page_budget", "page_topk"):
            value = getattr(self, name)
            if type(value) is int:
                if value <= 0:
                    raise ValueError(f"arkvale.{name} must be positive")
            elif type(value) is float:
                if not math.isfinite(value) or not 0 < value <= 1:
                    raise ValueError(f"arkvale.{name} ratio must be in (0, 1]")
            else:
                raise ValueError(f"arkvale.{name} must be an integer page count or float ratio")
        # Mixed units must first be resolved against each request's prompt length.
        if type(self.page_budget) is int and type(self.page_topk) is int:
            if not self.sink_pages + self.window_pages < self.page_topk <= self.page_budget:
                topk = max(self.page_topk, self.sink_pages + self.window_pages + 1)
                budget = max(self.page_budget, topk)
                logger.warning(
                    "ArkVale budgets page_budget=%s, page_topk=%s do not satisfy "
                    "sink_pages + window_pages < page_topk <= page_budget; "
                    "falling back to page_budget=%s, page_topk=%s",
                    self.page_budget, self.page_topk, budget, topk,
                )
                self.page_budget = budget
                self.page_topk = topk
        if self.max_cpu_bytes <= 0:
            raise ValueError("arkvale.max_cpu_bytes must be positive")

    def resolve(self, prompt_len: int) -> "ArkValeConfig":
        """Resolve ratios against this request's prompt; keep fixed during decode.

        Both ratios use ceil(prompt_len / page_size), not the resident budget.
        Round up to whole pages and expand undersized budgets for protected pages.
        The fallback keeps at least one selectable page beyond sink/window pages.
        """
        if type(prompt_len) is not int or prompt_len <= 0:
            raise ValueError("ArkVale prompt_len must be a positive integer")
        pages = (prompt_len + self.page_size - 1) // self.page_size
        counts = {
            name: math.ceil(value * pages) if type(value) is float else value
            for name in ("page_budget", "page_topk")
            for value in (getattr(self, name),)
        }
        return replace(self, **counts)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def validate_runtime(config):
    """Reject combinations whose scheduling/cache semantics are not implemented."""
    if config.shadowkv_config is not None and config.shadowkv_config.enabled:
        raise ValueError("ArkVale and ShadowKV are mutually exclusive")
    if config.attention_backend not in ("auto", "arkvale", "arkvale,arkvale"):
        raise ValueError("arkvale_config requires --attention-backend arkvale")
    config.attention_backend = "arkvale"
    if config.page_size != 1:
        raise ValueError(
            "ArkVale requires scheduler --page-size 1; configure ArkVale page_size in JSON"
        )
    if config.tp_info.size != 1:
        raise ValueError("ArkVale currently supports tensor parallel size 1 only")
    if config.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("ArkVale requires float16 or bfloat16")
    if config.cuda_graph_max_bs != 0 or config.cuda_graph_bs:
        raise ValueError("ArkVale requires --graph 0")
    if getattr(config, "cache_type", "naive") != "naive":
        raise ValueError("ArkVale requires --cache-type naive (no prefix reuse)")
    if getattr(config, "max_extend_tokens", config.max_seq_len) < config.max_seq_len:
        raise ValueError(
            "ArkVale requires --max-prefill-length >= maximum sequence length (no chunking)"
        )
    if config.model_config.head_dim > 256:
        raise ValueError("ArkVale supports head dimensions <= 256")
