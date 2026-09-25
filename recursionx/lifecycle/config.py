from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class LifecycleConfig:
    # ---- wake: projected-LoRA skill acquisition ----------------------------
    rank: int = 8
    alpha: float = 16.0
    projected: bool = True        # project adapters away from protected subspace
    data_init: bool = True        # init A from the skill's own activation PCA
    train_token_rows: bool = True # let new instruction tokens learn an embedding row
    wake_steps: int = 300
    wake_lr: float = 5e-3
    wake_batch: int = 64
    stats_batches: int = 4        # batches used for activation statistics

    # ---- knowledge (facts) -> Engram rows ----------------------------------
    fact_steps: int = 300
    fact_lr: float = 3e-2

    # ---- gate: is the skill good enough to be baked? -----------------------
    gate_min_acc: float = 0.5
    gate_min_gain: float = 0.2
    gate_max_anchor_drop: float = 1.0  # drop of anchors with adapter always on

    # ---- sleep: consolidation ---------------------------------------------
    sleep_pressure: int = 2       # accepted skills that trigger a sleep
    nrem_merge: bool = False      # fold adapters into the base before REM (see docs)
    rem_steps: int = 600          # distillation / rehearsal steps
    rem_lr: float = 1e-3
    rem_batch: int = 32           # per source (each new skill, and old-skill rehearsal)
    old_batch_per_skill: int = 16 # >0: stratified rehearsal, this many per old skill
    rem_params: str = "adaptable" # "adaptable" (AdaptableLinear weights) | "all"
    ce_weight: float = 1.0
    kl_weight: float = 1.0
    replay_per_skill: int = 64    # stored episodes per consolidated skill (hippocampus)
    dream_frac: float = 0.5       # fraction of rehearsal made of self-generated dreams
    dream_pool: int = 384
    dream_temperature: float = 1.0
    gpm_strength: float = 1.0     # gradient projection during REM (0 = off)
    gpm_threshold: float = 0.97
    max_protect_frac: float = 0.9

    # ---- growth ------------------------------------------------------------
    grow_experts: bool = False
    growth_acc_threshold: float = 0.9
    growth_extra_steps: int = 150

    # ---- serving ------------------------------------------------------------
    router_threshold: float = 0.9  # cosine similarity needed to route to an adapter

    def to_dict(self):
        return asdict(self)
