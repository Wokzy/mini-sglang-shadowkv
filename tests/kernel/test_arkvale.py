from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.kernel.arkvale import ArkValeConfig
from minisgl.kernel.arkvale.cache import ArkValePool
from minisgl.kernel.arkvale.kernels import estimate_scores, gather_pages, summarize_keys

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page_size": 3},
        {"page_size": 0},
        {"page_budget": 3},
        {"window_pages": 1},
        {"sink_pages": -1},
        {"page_topk": 129},
        {"max_cpu_bytes": 0},
        {"page_size": True},
        {"enabled": "false"},
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        ArkValeConfig(**kwargs)


@gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "n,page,h,hq,d", [(33, 8, 2, 8, 64), (64, 32, 4, 4, 128), (7, 16, 1, 4, 96), (1, 1, 1, 1, 32)]
)
def test_digests_and_gqa_scores(dtype, n, page, h, hq, d):
    torch.manual_seed(17)
    keys = torch.randn(n, h, d, device="cuda", dtype=dtype)
    digest = summarize_keys(keys, page)
    refs = []
    for chunk in keys.float().split(page):
        center = (chunk.amax(0) + chunk.amin(0)) / 2
        radius = (chunk - center).abs().mean(0)
        refs.append(torch.stack((center + radius, center - radius)))
    expected = torch.stack(refs).to(dtype)
    torch.testing.assert_close(digest, expected, rtol=0, atol=0)
    query = torch.randn(hq, d, device="cuda", dtype=dtype)
    bounds = digest.float().repeat_interleave(hq // h, dim=2)
    scores = (
        torch.maximum(query.float() * bounds[:, 0], query.float() * bounds[:, 1]).sum(-1).mean(-1)
    )
    torch.testing.assert_close(estimate_scores(query, digest), scores, rtol=1e-5, atol=1e-5)


@gpu
@pytest.mark.parametrize("count", [1, 8, 13, 24])
def test_gather_partial_pages(count):
    kv = torch.randn(2, 5, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([4, 0, 3], device="cuda", dtype=torch.int32)
    actual = gather_pages(kv, slots, count)
    expected = kv[:, slots.long()].flatten(1, 2)[:, :count]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def make_pool(budget=6):
    cfg = ArkValeConfig(page_size=4, page_budget=budget, page_topk=5, sink_pages=1, window_pages=2)
    model = SimpleNamespace(num_kv_heads=2, head_dim=32, num_layers=2)
    return ArkValePool(cfg, model, 128, torch.device("cuda"), torch.bfloat16, max_requests=2)


@gpu
@pytest.mark.parametrize("budget", [6, 32])
def test_cache_recall_append_and_full_attention(budget):
    torch.manual_seed(9)
    pool = make_pool(budget)
    keys = torch.randn(91, 2, 32, device="cuda", dtype=torch.bfloat16)
    vals = torch.randn_like(keys)
    queries = torch.randn(91, 4, 32, device="cuda", dtype=torch.bfloat16)
    pool.prefill(100, 0, keys[:19], vals[:19], queries[18])
    for length in range(20, 92):
        kv = pool.decode(100, 0, keys[length - 1], vals[length - 1], queries[length - 1], length)
        state = pool.states[(100, 0)]
        assert len(state.mapping) <= budget
        pages = (length + 3) // 4
        assert pool._protected(pages) <= state.mapping.keys()
        tokens = [i for p in sorted(state.mapping) for i in range(p * 4, min((p + 1) * 4, length))]
        torch.testing.assert_close(kv[0], keys[tokens], rtol=0, atol=0)
        torch.testing.assert_close(kv[1], vals[tokens], rtol=0, atol=0)
        if budget == 32:
            query = queries[length - 1].view(1, 4, 1, 32)
            actual = F.scaled_dot_product_attention(
                query, kv[0].transpose(0, 1)[None], kv[1].transpose(0, 1)[None], enable_gqa=True
            )
            expected = F.scaled_dot_product_attention(
                query,
                keys[:length].transpose(0, 1)[None],
                vals[:length].transpose(0, 1)[None],
                enable_gqa=True,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if budget == 6:
        assert state.evictions > 0 and state.recalls > budget and state.score_steps > 0
        # Force a known nonresident page back and verify the backing-store data.
        missing = next(p for p in range(pages) if p not in state.mapping)
        pool._ensure(state, pool._protected(pages) | {missing})
        slot = state.mapping[missing]
        torch.testing.assert_close(
            state.resident[0, slot], keys[missing * 4 : (missing + 1) * 4], rtol=0, atol=0
        )
    stats = pool.release(100)
    assert not pool.states and pool.host_bytes == 0
    assert stats["recalls"] > 0
    # A reused request/slot must not retain any old mappings or sequence state.
    pool.prefill(100, 0, keys[:1], vals[:1], queries[0])
    assert pool.states[(100, 0)].length == 1
    assert pool.states[(100, 0)].mapping == {0: 0}
    pool.release(100)


@gpu
def test_independent_requests_layers_and_abort_cleanup():
    pool = make_pool()
    for uid, n in [(1, 7), (2, 31)]:
        for layer in range(2):
            k = torch.full((n, 2, 32), uid + layer, device="cuda", dtype=torch.bfloat16)
            pool.prefill(uid, layer, k, -k, torch.ones(4, 32, device="cuda", dtype=torch.bfloat16))
    assert len(pool.states) == 4
    pool.release(1)  # Same cleanup used for completion and abort.
    assert all(uid == 2 for uid, _ in pool.states)
    assert all(s.length == 31 for s in pool.states.values())
    pool.release(2)
    assert pool.host_bytes == 0
