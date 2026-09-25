"""Fluid Mixture-of-Experts.

"Fluid" here means the expert set is not fixed:

* **growth** – :meth:`FluidMoE.grow_expert` clones an existing expert and adds a
  router row, so sleep consolidation can hand a new skill fresh, unprotected
  capacity instead of overwriting protected weights;
* **depth-aware routing** – the router sees the residual stream, which carries
  the loop-step embedding, so a token can use different experts on different
  iterations of the recurrent core;
* **offloading** – routed experts can be moved to disk and paged in on demand
  (LRU), the MoE analogue of keeping cold parameters on SSD.

Every expert is a SwiGLU MLP built from :class:`AdaptableLinear`, so projected
LoRA skill adapters can attach to experts too.
"""
from __future__ import annotations

import copy
import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import AdaptableLinear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, init_std: float = 0.02):
        super().__init__()
        self.gate = AdaptableLinear(d_model, d_hidden, init_std=init_std)
        self.up = AdaptableLinear(d_model, d_hidden, init_std=init_std)
        self.down = AdaptableLinear(d_hidden, d_model, init_std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ExpertStore:
    """LRU cache of experts paged in from disk."""

    def __init__(self, root: str, max_resident: int = 2):
        self.root = root
        self.max_resident = max_resident
        self.resident: "OrderedDict[str, nn.Module]" = OrderedDict()
        self.loads = 0
        os.makedirs(root, exist_ok=True)

    def fetch(self, stub: "OffloadedExpert") -> nn.Module:
        if stub.path in self.resident:
            self.resident.move_to_end(stub.path)
            return self.resident[stub.path]
        module = stub.materialize()
        module.eval()
        self.resident[stub.path] = module
        self.loads += 1
        while len(self.resident) > self.max_resident:
            self.resident.popitem(last=False)
        return module


class OffloadedExpert(nn.Module):
    """Placeholder that holds no parameters; the real expert lives on disk."""

    def __init__(self, path: str, factory, store: ExpertStore):
        super().__init__()
        self.path = path
        self.factory = factory
        self.store = store

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.store.fetch(self)(x)

    def materialize(self) -> nn.Module:
        m = self.factory()
        # only base weights are stored; adapters / protected bases stay with the
        # training copy (offloading is a serving-time optimisation)
        m.load_state_dict(torch.load(self.path, map_location="cpu", weights_only=True),
                          strict=False)
        return m


class FluidMoE(nn.Module):
    def __init__(self, d_model: int, d_expert: int, n_experts: int, top_k: int,
                 n_shared: int = 1, temperature: float = 1.0, init_std: float = 0.02):
        super().__init__()
        self.d_model, self.d_expert = d_model, d_expert
        self.top_k = top_k
        self.temperature = temperature
        self.init_std = init_std
        self.experts = nn.ModuleList([SwiGLU(d_model, d_expert, init_std) for _ in range(n_experts)])
        self.shared = nn.ModuleList([SwiGLU(d_model, d_expert, init_std) for _ in range(n_shared)])
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=init_std)
        self.router_bias = nn.Parameter(torch.zeros(n_experts))
        self.track_usage = False
        self.usage = torch.zeros(n_experts)
        self.last_aux = torch.zeros(())

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        N, E = flat.shape[0], self.n_experts
        logits = self.router(flat) / self.temperature + self.router_bias
        probs = logits.softmax(-1)
        k = min(self.top_k, E)
        topv, topi = probs.topk(k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        counts = torch.zeros(E, device=flat.device)
        for e in range(E):
            sel = topi == e
            rows = sel.any(-1).nonzero(as_tuple=True)[0]
            counts[e] = rows.numel()
            if rows.numel() == 0:
                continue
            w = (topv * sel).sum(-1)[rows].unsqueeze(-1)
            out.index_add_(0, rows, w * self.experts[e](flat[rows]))
        for s in self.shared:
            out = out + s(flat)
        # Switch-style load balancing loss
        frac = counts / max(N * k, 1)
        self.last_aux = E * (frac * probs.mean(0)).sum()
        if self.track_usage:
            self.usage = self.usage.to(counts.device)
            if self.usage.numel() != E:
                self.usage = torch.cat([self.usage, counts.new_zeros(E - self.usage.numel())])
            self.usage += counts.detach()
        return out.view(shape)

    # ------------------------------------------------------------------ growth
    @torch.no_grad()
    def grow_expert(self, src: int, router_direction: Optional[torch.Tensor] = None,
                    bias: float = 0.0, noise: float = 0.0) -> int:
        """Clone expert ``src`` into a new slot.  ``router_direction`` (d_model,)
        sets the new router row (e.g. the mean hidden state of the new skill),
        so the tokens that need the new capacity are sent there."""
        new = copy.deepcopy(self.experts[src])
        for n, m in new.named_modules():
            if hasattr(m, "adapters"):
                m.adapters = nn.ModuleDict()
                m.active = {}
                m.protected_basis = m.protected_basis[:, :0].clone()
        if noise > 0:
            for p in new.parameters():
                p.add_(noise * p.std() * torch.randn_like(p))
        self.experts.append(new)
        row = self.router.weight[src].clone() if router_direction is None else router_direction
        W = torch.cat([self.router.weight.data, row.view(1, -1).to(self.router.weight)], 0)
        self.router = nn.Linear(self.d_model, W.shape[0], bias=False).to(W.device)
        self.router.weight.data.copy_(W)
        self.router_bias = nn.Parameter(torch.cat([self.router_bias.data,
                                                   self.router_bias.new_tensor([bias])]))
        self.usage = torch.cat([self.usage.cpu(), torch.zeros(1)])
        return self.n_experts - 1

    # -------------------------------------------------------------- offloading
    def offload(self, root: str, max_resident: int = 2) -> ExpertStore:
        store = ExpertStore(root, max_resident)
        d, h, std = self.d_model, self.d_expert, self.init_std
        stubs = []
        for i, ex in enumerate(self.experts):
            if isinstance(ex, OffloadedExpert):
                stubs.append(ex)
                continue
            path = os.path.join(root, f"expert_{i}.pt")
            torch.save({k: v.cpu() for k, v in ex.state_dict().items()
                        if "adapters" not in k and "protected_basis" not in k}, path)
            stubs.append(OffloadedExpert(path, lambda: SwiGLU(d, h, std), store))
        self.experts = nn.ModuleList(stubs)
        return store

    def load_all(self) -> None:
        self.experts = nn.ModuleList([
            ex.materialize() if isinstance(ex, OffloadedExpert) else ex for ex in self.experts
        ])
