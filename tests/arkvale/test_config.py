from types import SimpleNamespace

import pytest
import torch
from minisgl.env import ENV
from minisgl.kernel.arkvale import ArkValeConfig
from minisgl.kernel.arkvale.config import validate_runtime


def runtime(**changes):
    config = SimpleNamespace(
        shadowkv_config=None,
        attention_backend="auto",
        page_size=1,
        tp_info=SimpleNamespace(size=1),
        dtype=torch.bfloat16,
        cuda_graph_max_bs=0,
        cuda_graph_bs=None,
        cache_type="naive",
        max_extend_tokens=4096,
        max_seq_len=4096,
        model_config=SimpleNamespace(head_dim=128),
    )
    config.__dict__.update(changes)
    return config


@pytest.mark.parametrize(
    "changes",
    [
        {"shadowkv_config": SimpleNamespace(enabled=True)},
        {"attention_backend": "fa"},
        {"page_size": 32},
        {"tp_info": SimpleNamespace(size=2)},
        {"dtype": torch.float32},
        {"cuda_graph_max_bs": None},
        {"cuda_graph_bs": [1]},
        {"cache_type": "radix"},
        {"max_extend_tokens": 2048},
        {"model_config": SimpleNamespace(head_dim=512)},
    ],
)
def test_unsupported_runtime(changes, monkeypatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)
    with pytest.raises(ValueError):
        validate_runtime(runtime(**changes))


def test_overlap_rejected(monkeypatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    with pytest.raises(ValueError, match="OVERLAP"):
        validate_runtime(runtime())


def test_auto_backend(monkeypatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)
    config = runtime()
    validate_runtime(config)
    assert config.attention_backend == "arkvale"


def test_unknown_config_key():
    with pytest.raises(TypeError):
        ArkValeConfig.from_dict({"page_buget": 32})
