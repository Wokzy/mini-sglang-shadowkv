"""B200 baseline checks: exercise FA4 with reordered YAKV request slots."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.mark.parametrize("prefill", [True, False])
def test_yakv_fa4_slot_addressing(prefill):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Blackwell required")
    from minisgl.attention.fa import FAMetadata, FlashAttentionBackend

    torch.manual_seed(19)
    device = "cuda"
    heads, kv_heads, dim, capacity = 4, 2, 128, 256
    keys = torch.randn(2, capacity, kv_heads, dim, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    slots = [1, 0]
    lengths = torch.tensor([117, 201], device=device, dtype=torch.int32)
    q = torch.randn(117 if prefill else 2, heads, dim, device=device, dtype=torch.bfloat16)
    metadata = FAMetadata(
        cu_seqlens_k=None,
        cu_seqlens_q=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
        cache_seqlens=lengths,
        max_seqlen_k=201,
        max_seqlen_q=1,
        page_table=None,
        cache_batch_idx=torch.tensor(slots, device=device, dtype=torch.int32),
        fa4_page_table=torch.tensor([[2, 3], [0, 1]], device=device, dtype=torch.int32),
    )
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.version = 4
    backend.config = SimpleNamespace(head_dim=dim)
    backend.kvcache = SimpleNamespace(
        device=device,
        local_qo_heads=heads,
        local_kv_heads=kv_heads,
        head_dim=dim,
        store_kv=lambda *a, **kw: None,
        compute_and_store_landmarks=lambda *a: None,
        select_kv=lambda *a: (keys, values),
    )
    batch = SimpleNamespace(
        is_prefill=prefill, attn_metadata=metadata, padded_reqs=[SimpleNamespace(table_idx=1)]
    )
    result = backend.flash3_kvcache_forward(q, keys[1, :117], values[1, :117], 0, batch)
    if prefill:
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None].float(),
            keys[1:2, :117].transpose(1, 2).repeat_interleave(2, dim=1).float(),
            values[1:2, :117].transpose(1, 2).repeat_interleave(2, dim=1).float(),
            is_causal=True,
        ).transpose(1, 2)
    else:
        reference = []
        for row, slot in enumerate(slots):
            length = int(lengths[row])
            reference.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[row : row + 1].transpose(0, 1)[None].float(),
                    keys[slot : slot + 1, :length]
                    .transpose(1, 2)
                    .repeat_interleave(2, dim=1)
                    .float(),
                    values[slot : slot + 1, :length]
                    .transpose(1, 2)
                    .repeat_interleave(2, dim=1)
                    .float(),
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        reference = torch.cat(reference)
    torch.testing.assert_close(result.float(), reference, atol=0.02, rtol=0.03)
