"""Model-size presets for scaling Recursion-X beyond the CPU experiments.

Parameter counts are *unique* parameters (the looped core is weight-tied, so
compute per token ≈ prelude + R × core + coda).  Engram rows are reported
separately because they are meant to live in host RAM or on disk.  Use
:func:`describe` to print exact numbers for a preset; they are computed by
instantiating the model on the ``meta`` device (no memory is allocated).

The presets are starting points, not tuned recipes.  See
``docs/roadmap/05_scale_up.md``.
"""
from __future__ import annotations

from typing import Dict

import torch

from .config import RXConfig

PRESETS: Dict[str, dict] = {
    # the CPU research model used in experiments/
    "tiny": dict(d_model=128, n_heads=4, n_kv_heads=2, d_expert=192, n_experts=4, top_k=2,
                 prelude_layers=("liquid",), core_layers=("liquid", "attn"), coda_layers=("attn",),
                 n_loops=3, mem_dim=64, engram_buckets=4099, engram_dim=32),
    # first GPU model: a few hours on one consumer GPU (~30M-class)
    "small": dict(d_model=384, n_heads=6, n_kv_heads=2, d_expert=768, n_experts=8, top_k=2,
                  n_shared_experts=1, prelude_layers=("liquid",),
                  core_layers=("liquid", "liquid", "attn"), coda_layers=("attn",),
                  n_loops=4, max_loops=8, loop_sampling="uniform", min_loops=2, bptt_loops=4,
                  mem_dim=96, mem_chunk=16, mem_conv=4, engram_buckets=65521, engram_dim=64,
                  max_seq_len=1024),
    # ~100M-class unique parameters
    "base": dict(d_model=512, n_heads=8, n_kv_heads=4, d_expert=1024, n_experts=8, top_k=2,
                 n_shared_experts=1, prelude_layers=("liquid", "liquid"),
                 core_layers=("liquid", "liquid", "attn"), coda_layers=("liquid", "attn"),
                 n_loops=4, max_loops=8, loop_sampling="uniform", min_loops=2, bptt_loops=4,
                 mem_dim=128, mem_chunk=16, mem_conv=4, engram_buckets=262139, engram_dim=64,
                 max_seq_len=2048),
    # ~1B-class (MoE total)
    "1b": dict(d_model=1024, n_heads=16, n_kv_heads=4, d_expert=2048, n_experts=16, top_k=2,
               n_shared_experts=1, prelude_layers=("liquid", "attn"),
               core_layers=("liquid", "liquid", "attn", "liquid", "liquid", "attn"),
               coda_layers=("liquid", "attn"), n_loops=4, max_loops=8, loop_sampling="uniform",
               min_loops=2, bptt_loops=4, mem_dim=256, mem_chunk=64, mem_conv=4,
               engram_buckets=1048573, engram_dim=128, max_seq_len=4096),
    # ~4B-class (MoE total), the size originally proposed
    "4b": dict(d_model=1536, n_heads=16, n_kv_heads=4, d_expert=3072, n_experts=24, top_k=2,
               n_shared_experts=1, prelude_layers=("liquid", "attn"),
               core_layers=("liquid", "liquid", "attn", "liquid", "liquid", "attn"),
               coda_layers=("liquid", "attn"), n_loops=4, max_loops=8, loop_sampling="uniform",
               min_loops=2, bptt_loops=4, mem_dim=384, mem_chunk=64, mem_conv=4,
               engram_buckets=4194301, engram_dim=128, max_seq_len=4096),
}


def preset(name: str, vocab_size: int, **overrides) -> RXConfig:
    cfg = dict(PRESETS[name])
    cfg.update(vocab_size=vocab_size)
    cfg.update(overrides)
    return RXConfig(**cfg)


def count_params(cfg: RXConfig) -> Dict[str, int]:
    """Exact parameter counts via a meta-device instantiation (no memory)."""
    from .model import RecursionX
    c = RXConfig.from_dict({**cfg.to_dict(), "use_engram": False})
    with torch.device("meta"):
        m = RecursionX(c)
    total = sum(p.numel() for p in m.parameters())
    core = sum(p.numel() for p in m.core.parameters())
    experts = sum(p.numel() for n, p in m.named_parameters() if ".experts." in n)
    per_expert = experts // max(1, cfg.n_experts * (len(cfg.prelude_layers) + len(cfg.core_layers)
                                                     + len(cfg.coda_layers)))
    n_layers = len(cfg.prelude_layers) + len(cfg.core_layers) + len(cfg.coda_layers)
    active = total - experts + per_expert * cfg.top_k * n_layers
    engram = 0
    if cfg.use_engram:
        n_tables = len(cfg.engram_orders) * cfg.engram_heads
        engram = n_tables * cfg.engram_buckets * cfg.engram_dim + 2 * n_tables * cfg.engram_dim * cfg.d_model
    return {"unique_params": total, "core_params": core, "expert_params": experts,
            "active_params_per_token_per_pass": active, "engram_table_params": engram}


def describe(name: str, vocab_size: int = 50304) -> str:
    cfg = preset(name, vocab_size)
    c = count_params(cfg)
    fmt = lambda n: f"{n / 1e6:,.1f}M" if n < 1e9 else f"{n / 1e9:,.2f}B"
    eff = len(cfg.prelude_layers) + cfg.n_loops * len(cfg.core_layers) + len(cfg.coda_layers)
    return (f"{name}: unique {fmt(c['unique_params'])} (core {fmt(c['core_params'])}, experts "
            f"{fmt(c['expert_params'])}), active/pass {fmt(c['active_params_per_token_per_pass'])}, "
            f"engram table {fmt(c['engram_table_params'])}, effective depth {eff} blocks "
            f"at R={cfg.n_loops}")
