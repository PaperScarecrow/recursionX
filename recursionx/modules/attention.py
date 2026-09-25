"""Grouped-query causal attention with RoPE and QK-norm."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import AdaptableLinear


def rope_cache(T: int, dim: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, init_std: float = 0.02):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h, self.kvh = n_heads, n_kv_heads
        self.hd = d_model // n_heads
        self.q = AdaptableLinear(d_model, n_heads * self.hd, init_std=init_std)
        self.k = AdaptableLinear(d_model, n_kv_heads * self.hd, init_std=init_std)
        self.v = AdaptableLinear(d_model, n_kv_heads * self.hd, init_std=init_std)
        self.o = AdaptableLinear(n_heads * self.hd, d_model, init_std=init_std)
        self.q_norm = nn.RMSNorm(self.hd)
        self.k_norm = nn.RMSNorm(self.hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_norm(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(B, T, self.kvh, self.hd)).transpose(1, 2)
        v = self.v(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        cos, sin = rope_cache(T, self.hd, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.kvh != self.h:
            rep = self.h // self.kvh
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, self.h * self.hd))
