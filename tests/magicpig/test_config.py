"""Exercise the actual engine configuration without importing unrelated CUDA extensions."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.magicpig import MagicPIGConfig


@pytest.fixture
def engine_config(monkeypatch):
    # EngineConfig itself needs no CUDA. Isolate eager package-level imports from
    # the existing engine/distributed/ShadowKV stack, which require GPU libraries.
    distributed = ModuleType("minisgl.distributed")
    distributed.DistributedInfo = SimpleNamespace
    attention = ModuleType("minisgl.attention")
    attention.ShadowKVConfig = SimpleNamespace
    utils = ModuleType("minisgl.utils")
    utils.cached_load_hf_config = None
    monkeypatch.setitem(sys.modules, "minisgl.distributed", distributed)
    monkeypatch.setitem(sys.modules, "minisgl.attention", attention)
    monkeypatch.setitem(sys.modules, "minisgl.utils", utils)
    path = Path(__file__).resolve().parents[2] / "python/minisgl/engine/config.py"
    return runpy.run_path(str(path))["EngineConfig"]


def options():
    return {
        "model_path": "unused",
        "tp_info": SimpleNamespace(rank=0, size=1),
        "dtype": torch.bfloat16,
    }


def test_json_selects_magicpig_and_disables_graphs(engine_config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"magicpig_config": {"hash_bits": 9, "hash_tables": 200}}))
    config = engine_config(**options(), additional_config_path=str(path), cuda_graph_bs=[1, 2])
    assert config.attention_backend == "magicpig"
    assert config.magicpig_config.hash_bits == 9
    assert config.magicpig_config.hash_tables == 200
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0


def test_backend_name_uses_default_magicpig_config(engine_config):
    config = engine_config(**options(), attention_backend="magicpig")
    assert config.magicpig_config == MagicPIGConfig()


def test_dense_configuration_unchanged(engine_config):
    config = engine_config(**options(), attention_backend="fa", cuda_graph_bs=[1, 2])
    assert config.magicpig_config is None
    assert config.cuda_graph_bs == [1, 2]


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"attention_backend": "fa,magicpig"}, "no hybrid"),
        ({"attention_backend": "fi"}, "no hybrid"),
        ({"page_size": 16}, "page_size"),
        ({"dtype": torch.float16}, "bfloat16"),
        ({"shadowkv_config": SimpleNamespace(enabled=True)}, "cannot be enabled together"),
    ],
)
def test_reject_incompatible_configuration(engine_config, changes, match):
    kwargs = options() | {"magicpig_config": MagicPIGConfig()} | changes
    with pytest.raises(ValueError, match=match):
        engine_config(**kwargs)
