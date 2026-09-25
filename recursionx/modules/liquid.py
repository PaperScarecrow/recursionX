"""Liquid mixer: LFM2-style double-gated short convolution + a liquid
time-constant (LTC/CfC-like) linear recurrence.

    (b, c, h) = in_proj(x)
    z  = causal_depthwise_conv(b ⊙ h)            # local mixing (LFM2)
    a  = exp(-softplus(dt(x)) · exp(A_log))       # input-dependent decay = 1/τ(x)
    s_t = a_t s_{t-1} + (1 - a_t) z_t             # liquid state, closed form
    y  = out_proj(c ⊙ (s + D ⊙ z))

The recurrence is the discretised solution of ``ds/dt = -(s - z)/τ(x)`` with
an input-dependent time constant – the core idea behind liquid time-constant
networks – computed with a chunked parallel scan so training is not a Python
loop over time.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import AdaptableLinear


def liquid_scan(u: torch.Tensor, log_a: torch.Tensor, h0: Optional[torch.Tensor] = None,
                chunk: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve ``h_t = exp(log_a_t) h_{t-1} + u_t`` for all t.  Shapes (B, T, D)."""
    B, T, D = u.shape
    h = u.new_zeros(B, D) if h0 is None else h0
    outs = []
    for s in range(0, T, chunk):
        la = log_a[:, s:s + chunk]
        uc = u[:, s:s + chunk]
        c = la.shape[1]
        L = la.cumsum(1)                                  # (B, c, D)
        diff = L.unsqueeze(2) - L.unsqueeze(1)            # (B, t, j, D) = L_t - L_j
        mask = torch.tril(torch.ones(c, c, dtype=torch.bool, device=u.device))
        diff = diff.masked_fill(~mask[None, :, :, None], float("-inf"))
        y = torch.einsum("btjd,bjd->btd", diff.exp(), uc) + L.exp() * h.unsqueeze(1)
        h = y[:, -1]
        outs.append(y)
    return torch.cat(outs, 1), h


class LiquidMixer(nn.Module):
    def __init__(self, d_model: int, conv_kernel: int = 4, scan_chunk: int = 16,
                 init_std: float = 0.02):
        super().__init__()
        self.d = d_model
        self.chunk = scan_chunk
        self.in_proj = AdaptableLinear(d_model, 3 * d_model, init_std=init_std)
        self.dt_proj = AdaptableLinear(d_model, d_model, init_std=init_std)
        self.out_proj = AdaptableLinear(d_model, d_model, init_std=init_std)
        self.conv = nn.Conv1d(d_model, d_model, conv_kernel, groups=d_model,
                              padding=conv_kernel - 1)
        # time constants spread over several scales (short to long memory)
        self.A_log = nn.Parameter(torch.linspace(-4.0, 0.5, d_model))
        self.dt_bias = nn.Parameter(torch.zeros(d_model))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        b, c, h = self.in_proj(x).chunk(3, dim=-1)
        z = self.conv((b * h).transpose(1, 2))[..., :T].transpose(1, 2)
        rate = F.softplus(self.dt_proj(x) + self.dt_bias) * self.A_log.exp()
        log_a = -rate
        a = log_a.exp()
        s, _ = liquid_scan((1 - a) * z, log_a, chunk=self.chunk)
        return self.out_proj(c * (s + self.D * z))
