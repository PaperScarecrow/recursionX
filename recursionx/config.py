"""Configuration for Recursion-X models."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class RXConfig:
    # --- vocabulary / widths -------------------------------------------------
    vocab_size: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 2
    max_seq_len: int = 256

    # --- layer layout (LFM2-style liquid/attention hybrid) --------------------
    # Each entry is a mixer type: "liquid" (gated short-conv + liquid time-constant
    # recurrence) or "attn" (GQA softmax attention). Every block also has a
    # Fluid-MoE feed-forward.
    prelude_layers: tuple = ("liquid",)
    core_layers: tuple = ("liquid", "attn")  # weight-tied, looped
    coda_layers: tuple = ("attn",)

    # --- looped / recursive depth (Huginn / Ouro style) ----------------------
    n_loops: int = 3               # default number of core iterations
    min_loops: int = 1             # lower bound when sampling depth in training
    max_loops: int = 6             # upper bound when sampling depth in training
    loop_sampling: str = "fixed"   # "fixed" | "uniform"
    bptt_loops: int = 0            # backprop through last k loops only (0 = all)
    exit_tol: float = 0.0          # >0 enables convergence-based early exit at eval

    # --- liquid mixer --------------------------------------------------------
    conv_kernel: int = 4
    scan_chunk: int = 16

    # --- Fluid MoE -----------------------------------------------------------
    n_experts: int = 4
    top_k: int = 2
    n_shared_experts: int = 1
    d_expert: int = 256
    router_aux_coef: float = 0.01
    router_temperature: float = 1.0

    # --- Titans / MIRAS neural long-term memory ------------------------------
    use_neural_memory: bool = True
    mem_dim: int = 64
    mem_chunk: int = 8
    mem_bias: str = "l2"          # attentional bias: "l2" | "huber" | "l1"
    mem_huber_delta: float = 1.0
    mem_max_lr: float = 1.0
    mem_max_decay: float = 0.2
    mem_conv: int = 0              # Titans-style short conv on q/k/v (0 = off)

    # --- Engram: hashed n-gram lookup memory (host/disk offloadable) ----------
    use_engram: bool = True
    engram_orders: tuple = (2, 3)
    engram_heads: int = 2
    engram_buckets: int = 4099     # prime
    engram_dim: int = 32
    engram_storage: str = "memory"  # "memory" | "disk"
    engram_path: str | None = None

    # --- skill adapters (projected LoRA) -------------------------------------
    lora_rank: int = 8
    lora_alpha: float = 16.0

    # --- misc ----------------------------------------------------------------
    tie_embeddings: bool = True
    init_std: float = 0.02
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RXConfig":
        d = dict(d)
        for k in ("prelude_layers", "core_layers", "coda_layers", "engram_orders"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**d)
