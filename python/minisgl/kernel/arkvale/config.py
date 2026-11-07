from dataclasses import dataclass

import torch


@dataclass
class ArkValeConfig:
    enabled: bool = True
    page_size: int = 32
    page_budget: int = 128
    # Total pages requested each step, including sink and window pages.
    # Remaining resident pages are retained and also attended (upstream's default).
    page_topk: int = 64
    sink_pages: int = 2
    window_pages: int = 2
    max_cpu_bytes: int = 8 * 1024**3

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("arkvale.enabled must be a boolean")
        for name in (
            "page_size",
            "page_budget",
            "page_topk",
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
        if not self.sink_pages + self.window_pages < self.page_topk <= self.page_budget:
            raise ValueError(
                "ArkVale requires sink_pages + window_pages < page_topk <= page_budget"
            )
        if self.max_cpu_bytes <= 0:
            raise ValueError("arkvale.max_cpu_bytes must be positive")

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def validate_runtime(config):
    """Reject combinations whose scheduling/cache semantics are not implemented."""
    from minisgl.env import ENV

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
    if not ENV.DISABLE_OVERLAP_SCHEDULING:
        raise ValueError("ArkVale requires MINISGL_DISABLE_OVERLAP_SCHEDULING=1")
    if getattr(config, "cache_type", "naive") != "naive":
        raise ValueError("ArkVale requires --cache-type naive (no prefix reuse)")
    if getattr(config, "max_extend_tokens", config.max_seq_len) < config.max_seq_len:
        raise ValueError(
            "ArkVale requires --max-prefill-length >= maximum sequence length (no chunking)"
        )
    if config.model_config.head_dim > 256:
        raise ValueError("ArkVale supports head dimensions <= 256")
