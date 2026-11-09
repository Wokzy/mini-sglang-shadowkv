"""Run with the built official CPU extensions on PYTHONPATH (no CUDA required)."""

import math

import pytest
import torch
from minisgl.magicpig import MagicPIGConfig, NativeMagicPIG, hash_vectors, partition_prompt


@pytest.mark.parametrize(
    "length,expected", [(1, (1, 1)), (4, (4, 4)), (68, (4, 4)), (100, (4, 36))]
)
def test_short_prompt_partition(length, expected):
    assert partition_prompt(length, MagicPIGConfig()) == expected


@pytest.mark.parametrize(
    "options",
    [
        {"hash_bits": 0},
        {"hash_bits": 16},
        {"hash_bits": True},
        {"hash_tables": 1},
        {"sink_tokens": -1},
        {"generation_buffer": 0},
        {"dense_layers": [-1]},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        MagicPIGConfig(**options)


def test_hash_packing_preserves_all_fifteen_bits():
    vectors = torch.tensor([[1.0], [-1.0]])
    projection = torch.ones(1, 30)
    assert hash_vectors(vectors, projection, 15).tolist() == [[32767, 32767], [0, 0]]


@pytest.fixture
def native():
    pytest.importorskip("minisgl_magicpig_lsh")
    pytest.importorskip("minisgl_magicpig_sparse")
    return NativeMagicPIG(
        MagicPIGConfig(hash_bits=2, hash_tables=3),
        layers=1,
        qo_heads=4,
        kv_heads=2,
        head_dim=128,
        slots=2,
        max_length=64,
        pin=False,
    )


def fill(native, slot, length, code=0, value=None):
    g = torch.Generator().manual_seed(10 + slot)
    keys = torch.randn(2, length, 128, generator=g).bfloat16()
    values = torch.randn(2, length, 128, generator=g).bfloat16()
    if value is not None:
        values.fill_(value)
    codes = torch.full((2, 3, length), code, dtype=torch.int16)
    native.fill(0, slot, keys, values, codes)
    return keys, values


def test_requires_two_collisions_and_returns_each_token_once(native):
    keys, values = fill(native, 0, 4)
    codes = torch.tensor([[[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]]], dtype=torch.int16)
    native.fill(0, 0, keys, values, codes.expand(2, -1, -1).contiguous())
    native.lsh.batch_retrieve(0, native.codes, native.indices, native.nnz)
    assert native.nnz.tolist() == [2, 2, 2, 2, 0, 0, 0, 0]
    assert set(native.indices[0, :2].tolist()) == {0, 1}


def test_refill_clears_old_buckets_without_affecting_other_request(native):
    fill(native, 0, 48, code=0)
    fill(native, 1, 19, code=0)
    fill(native, 0, 3, code=1)
    native.lsh.batch_retrieve(0, native.codes, native.indices, native.nnz)
    assert native.nnz.tolist() == [0, 0, 0, 0, 19, 19, 19, 19]
    native.codes[:4].fill_(1)
    native.lsh.batch_retrieve(0, native.codes, native.indices, native.nnz)
    assert native.nnz.tolist() == [3, 3, 3, 3, 19, 19, 19, 19]


def test_empty_sample_contributes_zero_mass(native):
    fill(native, 0, 16, code=1)
    output, lse = native.decode(0, [0], torch.ones(1, 4, 128), torch.zeros(1, 4, 3))
    assert torch.isfinite(output).all()
    assert (output == 0).all()
    assert torch.isneginf(lse).all()


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33])
def test_native_estimator_matches_importance_weights_and_log2_lse(native, length):
    keys, values = fill(native, 0, length)
    query = torch.randn(1, 4, 128, generator=torch.Generator().manual_seed(22)).bfloat16()
    output, lse = native.decode(0, [0], query, torch.zeros(1, 4, 3))
    k = keys.repeat_interleave(2, dim=0).float()
    v = values.repeat_interleave(2, dim=0).float()
    q = query[0].float()
    dots = torch.einsum("hd,hnd->hn", q, k)
    cosine = dots / (q.norm(dim=-1)[:, None] * k.norm(dim=-1))
    p = (1 - cosine.clamp(-1, 1).acos() / math.pi) ** 2
    probability = 1 - (1 - p) ** 3 - 3 * p * (1 - p) ** 2
    # The official kernel adds 1e-4 to the sampling probability.
    scores = dots / math.sqrt(128) - (probability + 1e-4).log()
    expected = torch.einsum("hn,hnd->hd", scores.softmax(-1), v)
    torch.testing.assert_close(output[0].float(), expected, atol=0.015, rtol=0.03)
    torch.testing.assert_close(lse[0], scores.logsumexp(-1) / math.log(2), atol=0.025, rtol=0.01)


def test_reordered_batch_and_inactive_slots(native):
    fill(native, 0, 17, value=2)
    fill(native, 1, 33, value=-3)
    queries = torch.ones(2, 4, 128)
    codes = torch.zeros(2, 4, 3)
    output, _ = native.decode(0, [1, 0], queries, codes)
    torch.testing.assert_close(output[0].float(), torch.full((4, 128), -3.0))
    torch.testing.assert_close(output[1].float(), torch.full((4, 128), 2.0))
    output, _ = native.decode(0, [1], queries[:1], codes[:1])
    assert native.nnz[:4].tolist() == [0, 0, 0, 0]
    assert torch.isfinite(output).all()


def test_short_prompt_with_empty_cpu_cache(native):
    fill(native, 0, 0)
    output, lse = native.decode(0, [0], torch.ones(1, 4, 128), torch.zeros(1, 4, 3))
    assert torch.isfinite(output).all()
    assert torch.isneginf(lse).all()


@pytest.mark.parametrize("length", [0, 1, 17, 33])
def test_native_32_head_dispatch(length):
    pytest.importorskip("minisgl_magicpig_lsh")
    pytest.importorskip("minisgl_magicpig_sparse")
    native = NativeMagicPIG(
        MagicPIGConfig(hash_bits=2, hash_tables=3),
        layers=1,
        qo_heads=32,
        kv_heads=2,
        head_dim=128,
        slots=1,
        max_length=64,
        pin=False,
    )
    fill(native, 0, length, value=2)
    output, lse = native.decode(0, [0], torch.ones(1, 32, 128), torch.zeros(1, 32, 3))
    if length:
        torch.testing.assert_close(output.float(), torch.full((1, 32, 128), 2.0))
        assert torch.isfinite(lse).all()
    else:
        assert (output == 0).all()
        assert torch.isneginf(lse).all()


def test_direct_staging_then_empty_selection_does_not_reuse_old_output(native):
    fill(native, 0, 17, value=2)
    native.set_batch([0])
    native.stage(torch.ones(1, 4, 128), torch.zeros(1, 4, 3))
    output, _ = native.decode(0, [0])
    torch.testing.assert_close(output.float(), torch.full((1, 4, 128), 2.0))
    native.stage(torch.ones(1, 4, 128), torch.ones(1, 4, 3))
    output, lse = native.decode(0, [0])
    assert (output == 0).all()
    assert torch.isneginf(lse).all()
