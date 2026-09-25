"""The gate that decides whether a freshly learned skill is good enough to be
sent to the sleeping hemisphere for consolidation."""
from __future__ import annotations

from typing import Dict, Tuple

from ..modules.lora import active_adapters
from ..train import EvalSet, evaluate
from .config import LifecycleConfig
from .skills import SkillRecord


class SkillGate:
    def __init__(self, cfg: LifecycleConfig):
        self.cfg = cfg

    def assess(self, model, record: SkillRecord, val: EvalSet,
               anchors: Dict[str, EvalSet]) -> Tuple[bool, Dict[str, float]]:
        c = self.cfg
        with active_adapters(model, {}):
            base = evaluate(model, val)["acc"]
            anchor_base = {k: evaluate(model, es)["acc"] for k, es in anchors.items()}
        with active_adapters(model, {record.name: 1.0}):
            skilled = evaluate(model, val)["acc"]
            # a single adapter left always-on == merged, so this previews the
            # interference that baking it into the base would cause
            anchor_on = {k: evaluate(model, es)["acc"] for k, es in anchors.items()}
        drops = [anchor_base[k] - anchor_on[k] for k in anchors]
        drop = max(drops) if drops else 0.0
        report = {"acc_base": base, "acc_skill": skilled, "gain": skilled - base,
                  "merge_preview_max_anchor_drop": drop}
        ok = (skilled >= c.gate_min_acc and skilled - base >= c.gate_min_gain
              and drop <= c.gate_max_anchor_drop)
        record.metrics.update(report)
        record.status = "accepted" if ok else "rejected"
        return ok, report
