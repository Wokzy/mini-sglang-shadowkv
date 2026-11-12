# Attention backends

- CPU/GPU partial attention results must use the same key-centering offset and log-sum-exp convention. FlashInfer state merging expects base-2 LSE.
- Offloaded KV pools must replace the dense GPU pool, not allocate a second copy alongside it. Scheduler page IDs may be accounting-only in such pools.
- Native attention adapters need tests for empty selections, GQA, changing batch order, and reuse of request slots. Report CPU-only tests separately from actual GPU integration runs.
- With the tested Blackwell FA4 implementation, request-slot addressing uses 128-token page tables, not `cache_batch_idx`; graph replay must update captured table storage in place.
- Deferred CPU cache fills must retain transfer buffers until completion, and be drained before decode, request-slot reuse, and shutdown. Bound in-flight staging memory and propagate fill exceptions. Keep native OpenMP calls on one host thread to avoid competing thread teams.
- Packed active-row native APIs must distinguish request-slot IDs from query/output row indices; tests must check that inactive scratch rows are not touched.
