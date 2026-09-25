"""Titans-style neural long-term memory, parameterised along the MIRAS axes.

A matrix-valued memory ``M`` is *trained at test time*: for each chunk of
tokens it takes one gradient step on an associative loss ``ℓ(M k, v)``
("attentional bias" in MIRAS terms), with momentum ("surprise") and a
data-dependent forget gate ("retention gate").  Retrieval uses the memory
state from *before* the current chunk, so the layer is causal and the whole
thing is differentiable, which lets the outer model meta-learn how to write.

MIRAS axes exposed here:

* memory architecture – linear matrix memory (``M0`` is the learned init,
  i.e. persistent memory);
* attentional bias – ``"l2"`` (Titans / delta rule), ``"huber"`` (Yaad-like,
  robust to outlier values) or ``"l1"``;
* retention gate – per-chunk weight decay ``alpha``;
* memory learning algorithm – gradient descent with momentum ``eta`` and
  per-token learning rate ``theta``.

The final ``(M, S)`` state is returned so memory can persist across segments
of a long stream.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import AdaptableLinear

MemState = Tuple[torch.Tensor, torch.Tensor]


class NeuralMemory(nn.Module):
    def __init__(self, d_model: int, mem_dim: int = 64, chunk: int = 8, bias: str = "l2",
                 huber_delta: float = 1.0, max_lr: float = 1.0, max_decay: float = 0.2,
                 init_std: float = 0.02):
        super().__init__()
        assert bias in ("l2", "huber", "l1")
        self.dm, self.chunk, self.bias = mem_dim, chunk, bias
        self.delta, self.max_lr, self.max_decay = huber_delta, max_lr, max_decay
        self.q = AdaptableLinear(d_model, mem_dim, init_std=init_std)
        self.k = AdaptableLinear(d_model, mem_dim, init_std=init_std)
        self.v = AdaptableLinear(d_model, mem_dim, init_std=init_std)
        self.out = AdaptableLinear(mem_dim, d_model, init_std=init_std)
        self.gate = nn.Linear(d_model, d_model)
        self.hyper = nn.Linear(d_model, 3)  # theta (lr), eta (momentum), alpha (forget)
        nn.init.zeros_(self.hyper.weight)
        nn.init.constant_(self.hyper.bias, 0.0)
        self.M0 = nn.Parameter(torch.zeros(mem_dim, mem_dim))

    def _err(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e = pred - target  # d/dpred of 0.5||pred - v||^2
        if self.bias == "huber":
            e = e.clamp(-self.delta, self.delta)
        elif self.bias == "l1":
            e = torch.sign(e)
        return e

    def forward(self, x: torch.Tensor, state: Optional[MemState] = None
                ) -> Tuple[torch.Tensor, MemState]:
        B, T, _ = x.shape
        q = F.normalize(self.q(x), dim=-1)
        k = F.normalize(self.k(x), dim=-1)
        v = self.v(x)
        hp = self.hyper(x)
        theta = torch.sigmoid(hp[..., 0]) * self.max_lr          # (B, T)
        eta = torch.sigmoid(hp[..., 1])
        alpha = torch.sigmoid(hp[..., 2] - 2.0) * self.max_decay
        if state is None:
            M = self.M0.unsqueeze(0).expand(B, -1, -1)
            S = torch.zeros_like(M)
        else:
            M, S = state
        ys = []
        for s in range(0, T, self.chunk):
            sl = slice(s, s + self.chunk)
            qc, kc, vc = q[:, sl], k[:, sl], v[:, sl]
            ys.append(qc @ M)                                     # read before write
            err = self._err(kc @ M, vc) * theta[:, sl].unsqueeze(-1)
            grad = kc.transpose(1, 2) @ err / kc.shape[1]         # (B, dm, dm)
            e = eta[:, sl].mean(1).view(B, 1, 1)
            a = alpha[:, sl].mean(1).view(B, 1, 1)
            S = e * S - grad                                      # momentum / surprise
            M = (1 - a) * M + S                                   # retention gate
        y = torch.cat(ys, 1)
        return self.out(y) * torch.sigmoid(self.gate(x)), (M, S)
