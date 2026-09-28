import os

import pytest
import torch

from recursionx import RXConfig, RecursionX
from recursionx.modules.engram import EngramMemory
from recursionx.modules.liquid import liquid_scan
from recursionx.modules.lora import (AdaptableLinear, add_skill_adapter, active_adapters,
                                     merge_skill, project_gradients)
from recursionx.modules.moe import FluidMoE
from recursionx.modules.neural_memory import NeuralMemory


def tiny_cfg(**kw):
    base = dict(vocab_size=48, d_model=32, n_heads=4, n_kv_heads=2, d_expert=48, n_experts=3,
                top_k=2, mem_dim=16, mem_chunk=4, engram_buckets=257, engram_dim=8, n_loops=2)
    base.update(kw)
    return RXConfig(**base)


def test_liquid_scan_matches_recurrence():
    torch.manual_seed(0)
    B, T, D = 2, 37, 5
    u = torch.randn(B, T, D)
    log_a = -torch.rand(B, T, D) * 3
    y, h = liquid_scan(u, log_a, chunk=8)
    ref = torch.zeros(B, D)
    outs = []
    for t in range(T):
        ref = log_a[:, t].exp() * ref + u[:, t]
        outs.append(ref)
    assert torch.allclose(y, torch.stack(outs, 1), atol=1e-5)
    assert torch.allclose(h, ref, atol=1e-5)


@pytest.mark.parametrize("kw", [{}, {"loop_sampling": "fixed", "n_loops": 3},
                                {"use_engram": False, "use_neural_memory": False},
                                {"mem_conv": 4}])
def test_model_is_causal(kw):
    torch.manual_seed(0)
    m = RecursionX(tiny_cfg(**kw)).eval()
    x = torch.randint(0, 48, (2, 21))
    y = x.clone()
    y[:, 13:] = torch.randint(0, 48, (2, 8))
    with torch.no_grad():
        a, b = m(x).logits, m(y).logits
    assert torch.allclose(a[:, :13], b[:, :13], atol=1e-5)
    assert not torch.allclose(a[:, 13:], b[:, 13:])


def test_loops_change_compute_and_backprop():
    torch.manual_seed(0)
    m = RecursionX(tiny_cfg(bptt_loops=1, loop_sampling="uniform", min_loops=1, max_loops=4))
    x = torch.randint(0, 48, (2, 9))
    out = m(x)
    out.logits.sum().backward()
    assert 1 <= out.loops <= 4
    m.eval()
    assert m(x, n_loops=4).loops == 4
    m.cfg.exit_tol = 1e9  # always converged -> exits after the 2nd iteration
    assert m(x).loops == 2


def test_projected_lora_merge_preserves_protected_inputs():
    torch.manual_seed(0)
    lin = AdaptableLinear(16, 8)
    old_inputs = torch.randn(200, 4) @ torch.randn(4, 16)  # old data lives in a 4-d subspace
    cov = old_inputs.t() @ old_inputs / 200
    added = lin.extend_protected(cov, threshold=0.999)
    assert added == 4
    U = lin.protected_basis
    assert torch.allclose(U.t() @ U, torch.eye(4), atol=1e-5)
    new_inputs = torch.randn(64, 16)
    ad = lin.add_adapter("s", rank=4, alpha=4, projected=True,
                         data_cov=new_inputs.t() @ new_inputs / 64)
    torch.nn.init.normal_(ad.B)
    before = lin(old_inputs).detach()
    lin.active = {"s": 1.0}
    with_adapter = lin(new_inputs).detach()
    lin.active = {}
    lin.merge_adapter("s")
    assert torch.allclose(lin(old_inputs), before, atol=1e-4)       # old behaviour intact
    assert torch.allclose(lin(new_inputs), with_adapter, atol=1e-4)  # merge is exact
    assert "s" not in lin.adapters


def test_gradient_projection_removes_protected_component():
    lin = AdaptableLinear(8, 3)
    basis = torch.linalg.qr(torch.randn(8, 3))[0]
    lin.protected_basis = basis
    lin.weight.grad = torch.randn(3, 8)
    project_gradients(lin)
    assert torch.allclose(lin.weight.grad @ basis, torch.zeros(3, 3), atol=1e-5)


def test_model_level_adapter_merge_equivalence():
    torch.manual_seed(0)
    m = RecursionX(tiny_cfg()).eval()
    add_skill_adapter(m, "s", rank=4, alpha=8)
    for p in m.parameters():
        if p.dim() == 2 and p.shape[1] == 4:  # B matrices
            torch.nn.init.normal_(p, std=0.05)
    x = torch.randint(0, 48, (2, 11))
    with torch.no_grad(), active_adapters(m, {"s": 1.0}):
        ref = m(x).logits
    merge_skill(m, "s")
    with torch.no_grad():
        assert torch.allclose(m(x).logits, ref, atol=1e-4)


def test_moe_growth_and_offload(tmp_path):
    torch.manual_seed(0)
    moe = FluidMoE(16, 24, n_experts=3, top_k=2).eval()
    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        ref = moe(x)
    store = moe.offload(str(tmp_path), max_resident=1)
    assert sum(p.numel() for p in moe.experts.parameters()) == 0
    with torch.no_grad():
        assert torch.allclose(moe(x), ref, atol=1e-5)
    assert store.loads >= 2 and len(store.resident) <= 1
    moe.load_all()
    idx = moe.grow_expert(0, router_direction=torch.zeros(16), bias=-1e4)
    assert idx == 3 and moe.n_experts == 4
    with torch.no_grad():  # a new expert that nobody routes to changes nothing
        assert torch.allclose(moe(x), ref, atol=1e-5)


def test_engram_disk_matches_memory_and_updates_sparsely(tmp_path):
    torch.manual_seed(0)
    mem = EngramMemory(16, orders=(2, 3), heads=2, buckets=101, dim=4)
    toks = torch.randint(0, 20, (2, 9))
    h = torch.randn(2, 9, 16)
    ref = mem(toks, h)
    path = os.path.join(tmp_path, "engram.memmap")
    mem.to_disk(path)
    mem.train()
    out = mem(toks, h)
    assert torch.allclose(out, ref, atol=1e-6)
    before = mem.disk.load_all().clone()
    out.sum().backward()
    n = mem.apply_sparse_grads(lr=0.1)
    after = mem.disk.load_all()
    changed = (before != after).any(-1)
    touched = torch.unique(mem.indices(toks))
    assert n == touched.numel()
    assert changed.sum() <= touched.numel() and changed[touched].any()
    assert not changed[[i for i in range(after.shape[0]) if i not in set(touched.tolist())]].any()


def test_neural_memory_state_carry_matches_full_sequence():
    torch.manual_seed(0)
    nm = NeuralMemory(16, mem_dim=8, chunk=4, bias="huber")
    with torch.no_grad():
        nm.hyper.bias.fill_(1.0)
    x = torch.randn(2, 16, 16)
    full, st = nm(x)
    a, s1 = nm(x[:, :8])
    b, s2 = nm(x[:, 8:], s1)
    assert torch.allclose(torch.cat([a, b], 1), full, atol=1e-5)
    assert torch.allclose(s2[0], st[0], atol=1e-5)


def test_neural_memory_learns_at_test_time():
    """With a high write rate the memory should recall a value written earlier."""
    torch.manual_seed(0)
    nm = NeuralMemory(8, mem_dim=8, chunk=1, max_lr=1.0)
    with torch.no_grad():
        for lin in (nm.q, nm.k, nm.v):
            lin.weight.copy_(torch.eye(8))
        nm.hyper.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))  # lr≈1, no momentum/decay
        nm.out.weight.copy_(torch.eye(8))
        nm.gate.bias.fill_(10.0)
        nm.gate.weight.zero_()
    key = torch.nn.functional.normalize(torch.randn(8), dim=0)
    x = torch.stack([key, torch.zeros(8), key]).unsqueeze(0)  # write key->key, then query key
    y, _ = nm(x)
    assert torch.nn.functional.cosine_similarity(y[0, 2], key, dim=0) > 0.9
