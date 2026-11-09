"""Real CUDA/FlashInfer integration checks; no mocked attention kernels."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.fixture
def backend(monkeypatch, request):
    # Delay GPU-dependent project imports until after the skip check.
    import minisgl.core as core
    import minisgl.distributed.info as tp
    from minisgl.attention.magicpig import MagicPIGBackend, MagicPIGPool
    from minisgl.magicpig import MagicPIGConfig

    monkeypatch.setattr(tp, "_TP_INFO", tp.DistributedInfo(0, 1))
    model = SimpleNamespace(num_qo_heads=4, num_kv_heads=2, head_dim=128, num_layers=1)
    config = MagicPIGConfig(
        hash_bits=2,
        hash_tables=8,
        sink_tokens=2,
        local_tokens=3,
        generation_buffer=8,
        dense_layers=getattr(request, "param", ()),
    )
    pool = MagicPIGPool(config, model, 2, 64, torch.device("cuda"), torch.bfloat16)
    ctx = core.Context(page_size=1)
    ctx.kv_cache = pool
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return MagicPIGBackend(model)


def make_batch(slot, length, uid=0):
    from minisgl.core import Batch, Req, SamplingParams

    req = Req(torch.zeros(length, dtype=torch.int32), slot, 0, 4, uid, SamplingParams(), None)
    batch = Batch([req], "prefill")
    batch.padded_reqs = batch.reqs
    return batch


def tensors(length):
    generator = torch.Generator(device="cuda").manual_seed(91 + length)
    return (
        torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16, generator=generator),
        torch.randn(length, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator),
        torch.randn(length, 2, 128, device="cuda", dtype=torch.bfloat16, generator=generator),
    )


def exact(query, keys, values):
    keys = keys.repeat_interleave(2, dim=1).float()
    values = values.repeat_interleave(2, dim=1).float()
    scores = torch.einsum("bhd,nhd->bhn", query.float(), keys) / 128**0.5
    return torch.einsum("bhn,nhd->bhd", scores.softmax(-1), values), scores.logsumexp(-1)


@pytest.mark.parametrize("backend,length", [((), 3), ((0,), 37)], indirect=["backend"])
def test_exact_short_prompt_or_dense_layer(backend, length):
    batch = make_batch(1, length)
    q, k, v = tensors(length)
    backend.prepare_metadata(batch)
    prefill = backend.forward(q, k, v, 0, batch)
    reference, _ = exact(q[-1:], k, v)
    torch.testing.assert_close(prefill[-1:].float(), reference, atol=0.025, rtol=0.03)
    batch.reqs[0].complete_one()
    batch.phase = "decode"
    q1, k1, v1 = tensors(1)
    backend.prepare_metadata(batch)
    result = backend.forward(q1, k1, v1, 0, batch)
    reference, _ = exact(q1, torch.cat((k, k1)), torch.cat((v, v1)))
    torch.testing.assert_close(result.float(), reference, atol=0.025, rtol=0.03)


def test_sparse_cpu_gpu_merge_uses_same_center_and_log_base(backend):
    batch = make_batch(1, 37)
    q, k, v = tensors(37)
    backend.prepare_metadata(batch)
    backend.forward(q, k, v, 0, batch)
    batch.reqs[0].complete_one()
    batch.phase = "decode"
    q1, k1, v1 = tensors(1)
    backend.prepare_metadata(batch)
    result = backend.forward(q1, k1, v1, 0, batch)
    pool = backend.pool
    length = batch.attn_metadata.sparse_lengths[0]
    local_k = pool.cache[0][1, 0, :, :length].transpose(0, 1)
    local_v = pool.cache[0][1, 1, :, :length].transpose(0, 1)
    local_out, local_lse = exact(q1, local_k, local_v)
    native = pool.native
    cpu_out = native.output.view(2, 4, 128)[1:2].cuda().float()
    cpu_lse = native.stats[1].view(2, 4)[1:2].cuda() * torch.log(torch.tensor(2.0, device="cuda"))
    weights = torch.stack((local_lse, cpu_lse), dim=-1).softmax(-1)
    expected = weights[..., :1] * local_out + weights[..., 1:] * cpu_out
    torch.testing.assert_close(result.float(), expected, atol=0.025, rtol=0.03)
    assert native.nnz[:4].sum() == 0  # unused slot cannot contribute


def test_slot_reuse_and_generation_limit(backend):
    old = make_batch(0, 37, uid=10)
    backend.prepare_metadata(old)
    backend.forward(*tensors(37), 0, old)
    new = make_batch(0, 3, uid=11)
    backend.prepare_metadata(new)
    backend.forward(*tensors(3), 0, new)
    new.reqs[0].complete_one()
    new.phase = "decode"
    backend.prepare_metadata(new)
    backend.forward(*tensors(1), 0, new)
    assert backend.pool.native.nnz.sum() == 0
    invalid = make_batch(1, 3)
    invalid.reqs[0].output_len = 9
    with pytest.raises(ValueError, match="generation_buffer"):
        backend.prepare_metadata(invalid)


def test_decode_batch_reorders_and_shrinks(backend):
    from minisgl.core import Batch

    first = make_batch(0, 3, uid=20)
    q0, k0, v0 = tensors(3)
    backend.prepare_metadata(first)
    backend.forward(q0, k0, v0, 0, first)
    second = make_batch(1, 4, uid=21)
    q1, k1, v1 = tensors(4)
    backend.prepare_metadata(second)
    backend.forward(q1, k1, v1, 0, second)
    first.reqs[0].complete_one()
    second.reqs[0].complete_one()
    batch = Batch([second.reqs[0], first.reqs[0]], "decode")
    batch.padded_reqs = batch.reqs
    q, k, v = tensors(2)
    backend.prepare_metadata(batch)
    actual = backend.forward(q, k, v, 0, batch)
    expected0, _ = exact(q[:1], torch.cat((k1, k[:1])), torch.cat((v1, v[:1])))
    expected1, _ = exact(q[1:], torch.cat((k0, k[1:])), torch.cat((v0, v[1:])))
    torch.testing.assert_close(
        actual.float(), torch.cat((expected0, expected1)), atol=0.025, rtol=0.03
    )
    second.reqs[0].complete_one()
    batch = Batch(second.reqs, "decode")
    batch.padded_reqs = batch.reqs
    q_next, k_next, v_next = tensors(1)
    backend.prepare_metadata(batch)
    actual = backend.forward(q_next, k_next, v_next, 0, batch)
    expected, _ = exact(q_next, torch.cat((k1, k[:1], k_next)), torch.cat((v1, v[:1], v_next)))
    torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.03)
