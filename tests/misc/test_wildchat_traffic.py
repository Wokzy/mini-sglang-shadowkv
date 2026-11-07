"""CPU-only checks for benchmark logical KV payload accounting."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "wildchat_compare",
    Path(__file__).resolve().parents[2] / "benchmark/offline/bench_wildchat_compare.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
TrafficMeter = _module.TrafficMeter


@pytest.mark.parametrize("mode", ["full_attention", "yakv", "arkvale"])
def test_payload_counts_and_reset(mode):
    pool = SimpleNamespace(
        config=SimpleNamespace(page_size=32),
        release=lambda uid: {"recalls": 3, "evictions": 1, "score_steps": 2},
    )
    backend = SimpleNamespace(prepare_metadata=lambda batch: None)
    llm = SimpleNamespace(
        engine=SimpleNamespace(ctx=SimpleNamespace(kv_cache=pool), attn_backend=backend)
    )
    meter = TrafficMeter(llm, mode, 54)
    meter.configure_model(SimpleNamespace(num_kv_heads=2, head_dim=64, num_layers=2))
    req = SimpleNamespace(uid=7, device_len=100, extend_len=100)
    batch = SimpleNamespace(is_prefill=True, padded_reqs=[req])
    backend.prepare_metadata(batch)
    batch.is_prefill = False
    req.device_len = 101
    req.extend_len = 1
    backend.prepare_metadata(batch)
    if mode == "arkvale":
        pool.release(7)
    unit = 2 * 2 * 64 * 2 * 2
    actual = meter.result()
    assert actual["request_decode_steps"] == 1
    if mode == "full_attention":
        assert actual["kv_total_bytes"] == 0
    else:
        assert actual["kv_d2h_bytes"] == 101 * unit
        expected = 55 * unit if mode == "yakv" else 3 * (32 * unit // 2)
        assert actual["kv_h2d_bytes"] == expected
    meter.reset()
    assert meter.result()["kv_total_bytes"] == 0
    assert meter.result()["request_decode_steps"] == 0
