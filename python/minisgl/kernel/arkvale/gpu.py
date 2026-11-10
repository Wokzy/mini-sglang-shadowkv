"""Batched decode driver. Host code never reads page-selection results."""

import torch
import triton as tr

from . import batched


class GPUDecode:
    def __init__(self, pool):
        from minisgl.shadowkv_kernels import map_to_gpu

        self.pool = pool
        s, l, c = pool.max_requests, pool.num_layers, pool.capacity
        device = pool.device
        self.mapped_hosts = [[map_to_gpu(host) for host in row] for row in pool.host_buffers]
        self.host_ptrs = torch.tensor(
            [[host.data_ptr() for host in row] for row in self.mapped_hosts],
            device=device,
            dtype=torch.int64,
        )
        self.mapping = torch.full((s, l, c), -1, device=device, dtype=torch.int32)
        self.ages = torch.zeros((s, l, c), device=device, dtype=torch.int64)
        self.clock = torch.zeros((s, l), device=device, dtype=torch.int64)
        self.counts = torch.zeros((s, l, 3), device=device, dtype=torch.int64)
        self.misses = torch.empty((s, c), device=device, dtype=torch.int32)
        self.order = torch.empty_like(self.misses)
        self.lengths = torch.empty((s,), device=device, dtype=torch.int32)
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.last_use = torch.cuda.Event()
        self.last_use.record(torch.cuda.current_stream(device))

    def initialize(self, uid, layer, state):
        """Publish reference prefill/scalar state once, not on batched decode."""
        slot = self.pool.uid_slots[uid]
        mapping = [-1] * self.pool.capacity
        ages = [0] * self.pool.capacity
        for age, (page, physical) in enumerate(state.mapping.items()):
            mapping[physical] = page
            ages[physical] = age
        self.mapping[slot, layer].copy_(torch.tensor(mapping, dtype=torch.int32))
        self.ages[slot, layer].copy_(torch.tensor(ages, dtype=torch.int32))
        self.clock[slot, layer] = len(state.mapping)
        self.counts[slot, layer].copy_(
            torch.tensor(
                [state.recalls, state.evictions, state.score_steps],
                dtype=torch.int64,
            )
        )
        self.last_use.record(torch.cuda.current_stream(self.pool.device))

    def prefill(self, uid, layer, state, keys, values, query):
        """Asynchronous full-prompt backup and GPU initial resident selection."""
        from .kernels import summarize_keys

        pool = self.pool
        cfg = pool.config
        slot = pool.uid_slots[uid]
        n, h, d = keys.shape
        mp, cap, p = pool.max_pages, pool.capacity, cfg.page_size
        engine = torch.cuda.current_stream(pool.device)
        active = torch.tensor([slot], device=pool.device, dtype=torch.int32)
        lengths = torch.tensor([n], device=pool.device, dtype=torch.int32)
        self.mapping[slot, layer].fill_(-1)
        self.clock[slot, layer].zero_()
        self.counts[slot, layer].zero_()
        self.transfer_stream.wait_stream(engine)
        with torch.cuda.stream(self.transfer_stream):
            state.host[0].view(-1, h, d)[:n].copy_(keys, non_blocking=True)
            state.host[1].view(-1, h, d)[:n].copy_(values, non_blocking=True)
        summarize_keys(keys.contiguous(), p, out=state.digests)
        query = query.unsqueeze(0).contiguous()
        common = {"LAYER": layer, "L": pool.num_layers}
        batched.score_pages[(mp, 1)](
            query,
            pool.digest_buffer,
            active,
            lengths,
            pool.score_buffer,
            MP=mp,
            P=p,
            CAP=cap,
            H=h,
            HQ=query.shape[1],
            D=d,
            SINK=cfg.sink_pages,
            WINDOW=cfg.window_pages,
            QS0=query.stride(0),
            QS1=query.stride(1),
            BD=tr.next_power_of_2(d),
            BH=tr.next_power_of_2(query.shape[1]),
            **common,
        )
        batched.select_pages[(1,)](
            pool.score_buffer,
            active,
            lengths,
            self.mapping,
            self.ages,
            self.clock,
            self.misses,
            self.order,
            self.counts,
            self.lengths,
            MP=mp,
            P=p,
            CAP=cap,
            TOPK=cfg.page_topk,
            SINK=cfg.sink_pages,
            WINDOW=cfg.window_pages,
            BP=tr.next_power_of_2(mp),
            BC=tr.next_power_of_2(cap),
            INITIAL=True,
            **common,
        )
        self.transfer_stream.wait_stream(engine)
        with torch.cuda.stream(self.transfer_stream):
            batched.recall_pages[(1, cap, tr.cdiv(2 * p * h * d, 512))](
                self.host_ptrs,
                pool.resident_buffer,
                active,
                self.misses,
                lengths,
                MP=mp,
                P=p,
                CAP=cap,
                HD=h * d,
                BLOCK=512,
                **common,
            )
        engine.wait_stream(self.transfer_stream)
        self.last_use.record(engine)

    def statistics(self, slot, layers):
        # Completion/abort can run on the scheduler stream. Wait for the latest
        # engine metadata updates before reading counters; release is not a hot
        # decode synchronization. Buffer reuse itself remains engine-ordered.
        torch.cuda.current_stream(self.pool.device).wait_event(self.last_use)
        values = self.counts[slot, layers].sum(0).cpu().tolist()
        return dict(zip(("recalls", "evictions", "score_steps"), values))

    def decode(self, q, k, v, layer, active, lengths):
        pool = self.pool
        cfg = pool.config
        q = q if q.stride(-1) == 1 else q.contiguous()
        k = k if k.stride(-1) == 1 else k.contiguous()
        v = v if v.stride(-1) == 1 else v.contiguous()
        bs, hq, d = q.shape
        h = pool.heads
        hd = h * d
        p, cap, mp = cfg.page_size, pool.capacity, pool.max_pages
        common = {"LAYER": layer, "L": pool.num_layers}
        engine = torch.cuda.current_stream(pool.device)

        # Backup is asynchronous to the host and can overlap score/selection on
        # the engine stream. It only writes CPU backing, not residency metadata.
        self.transfer_stream.wait_stream(engine)
        with torch.cuda.stream(self.transfer_stream):
            batched.append_host[(bs, tr.cdiv(hd, 256))](
                k,
                v,
                self.host_ptrs,
                active,
                lengths,
                MAX_T=mp * p,
                HD=hd,
                D=d,
                KS0=k.stride(0),
                KS1=k.stride(1),
                VS0=v.stride(0),
                VS1=v.stride(1),
                BLOCK=256,
                **common,
            )

        batched.score_pages[(mp, bs)](
            q,
            pool.digest_buffer,
            active,
            lengths,
            pool.score_buffer,
            MP=mp,
            P=p,
            CAP=cap,
            H=h,
            HQ=hq,
            D=d,
            SINK=cfg.sink_pages,
            WINDOW=cfg.window_pages,
            QS0=q.stride(0),
            QS1=q.stride(1),
            BD=tr.next_power_of_2(d),
            BH=tr.next_power_of_2(hq),
            **common,
        )
        batched.select_pages[(bs,)](
            pool.score_buffer,
            active,
            lengths,
            self.mapping,
            self.ages,
            self.clock,
            self.misses,
            self.order,
            self.counts,
            self.lengths,
            MP=mp,
            P=p,
            CAP=cap,
            TOPK=cfg.page_topk,
            SINK=cfg.sink_pages,
            WINDOW=cfg.window_pages,
            BP=tr.next_power_of_2(mp),
            BC=tr.next_power_of_2(cap),
            **common,
        )
        # Recall must follow both backup and miss-list production. The engine
        # then waits before modifying/reading resident pages. No CPU wait and no
        # selected-page IDs or miss descriptors copied back to the host.
        self.transfer_stream.wait_stream(engine)
        with torch.cuda.stream(self.transfer_stream):
            batched.recall_pages[(bs, cap, tr.cdiv(2 * p * hd, 512))](
                self.host_ptrs,
                pool.resident_buffer,
                active,
                self.misses,
                lengths,
                MP=mp,
                P=p,
                CAP=cap,
                HD=hd,
                BLOCK=512,
                **common,
            )
        engine.wait_stream(self.transfer_stream)
        batched.update_current[(bs, h)](
            k,
            v,
            pool.resident_buffer,
            pool.digest_buffer,
            active,
            lengths,
            self.mapping,
            MP=mp,
            P=p,
            CAP=cap,
            H=h,
            D=d,
            KS0=k.stride(0),
            KS1=k.stride(1),
            VS0=v.stride(0),
            VS1=v.stride(1),
            BC=tr.next_power_of_2(cap),
            BD=tr.next_power_of_2(d),
            **common,
        )
        batched.gather_resident[(bs, 2, tr.cdiv(cap * p * hd, 512))](
            pool.resident_buffer,
            pool.gathered_buffer,
            active,
            self.order,
            self.lengths,
            SLOTS=pool.max_requests,
            CAP=cap,
            P=p,
            HD=hd,
            BLOCK=512,
            **common,
        )
        self.last_use.record(engine)
        return self.lengths[:bs]
