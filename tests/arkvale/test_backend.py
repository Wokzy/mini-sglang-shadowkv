from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.kernel.arkvale import ArkValeConfig
from minisgl.kernel.arkvale import backend as impl
from minisgl.kernel.arkvale.cache import ArkValePool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backend_flattened_kv_and_mixed_request_lengths(monkeypatch):
    model = SimpleNamespace(num_kv_heads=2, head_dim=64, num_layers=1)
    pool = ArkValePool(
        ArkValeConfig(page_size=4, page_budget=32, page_topk=8),
        model,
        128,
        torch.device("cuda"),
        torch.bfloat16,
        max_requests=2,
    )
    monkeypatch.setattr(impl, "get_global_ctx", lambda: SimpleNamespace(kv_cache=pool))
    backend = impl.ArkValeBackend(model)
    q = torch.randn(16, 4, 64, device="cuda", dtype=torch.bfloat16)
    # Match AttentionLayer: flattened, potentially strided K/V from QKV split.
    storage = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
    k, v = storage[:, :128], storage[:, 128:256]
    reqs = [
        SimpleNamespace(uid=10, extend_len=7, device_len=7, cached_len=0),
        SimpleNamespace(uid=11, extend_len=9, device_len=9, cached_len=0),
    ]
    batch = SimpleNamespace(padded_reqs=reqs, padded_size=2, size=2, is_prefill=True)
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.get_last_indices(2).tolist() == [6, 15]
    actual = backend.forward(q, k, v, 0, batch)
    expected = []
    for start, end in [(0, 7), (7, 16)]:
        out = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1)[None],
            k[start:end].reshape(-1, 2, 64).transpose(0, 1)[None],
            v[start:end].reshape(-1, 2, 64).transpose(0, 1)[None],
            is_causal=True,
            enable_gqa=True,
        )
        expected.append(out[0].transpose(0, 1))
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
    batch.is_prefill = False
    for req in reqs:
        req.cached_len = req.device_len
        req.device_len += 1
        req.extend_len = 1
    backend.prepare_metadata(batch)
    actual = backend.forward(q[:2], k[:2], v[:2], 0, batch)
    assert actual.shape == (2, 4, 64) and torch.isfinite(actual).all()
    for uid in (10, 11):
        pool.release(uid)
    assert not pool.states and pool.host_bytes == 0
