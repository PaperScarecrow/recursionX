"""Dual-hemisphere orchestration (unihemispheric sleep, as in dolphins).

Two copies of the network take turns:

* the **awake** hemisphere serves requests and learns: new skills become
  projected LoRA adapters that are routed to immediately, new facts are written
  to the (shared) Engram tables;
* when enough accepted skills are pending (``sleep_pressure``), the **sleeping**
  hemisphere is synchronised to the awake base, consolidates the pending skills
  into its own base weights (merge → distil/rehearse/dream → protect) while the
  awake one keeps serving, and then the roles swap.  The previously awake
  hemisphere is *reset*: its adapters are dropped and it is re-synchronised to
  the new consolidated base.

``sleep(background=True)`` runs consolidation in a thread against a frozen
snapshot of the awake hemisphere, so serving continues during sleep.
"""
from __future__ import annotations

import copy
import random
import threading
from typing import Dict, List, Optional

import torch

from ..data.tasks import collate
from ..modules.lora import (collecting_stats, extend_protected_subspaces, gather_covariances,
                            list_skills, remove_skill)
from ..train import EvalSet, evaluate
from .config import LifecycleConfig
from .gate import SkillGate
from .skills import SkillRecord, SkillRouter, build_prototype
from .sleep import SleepConsolidator
from .wake import WakeLearner


def _clone_sharing_engram(model):
    """Deep copy that shares the Engram store (one hippocampus, two cortices)."""
    eng = model.engram
    model.engram = None
    try:
        twin = copy.deepcopy(model)
    finally:
        model.engram = eng
    twin.engram = eng
    return twin


class DualHemisphereBrain:
    def __init__(self, model, cfg: LifecycleConfig, seed: int = 0, input_weight: float = 0.1):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.hemispheres = [model, _clone_sharing_engram(model)]
        self.awake_idx = 0
        self.catalog: Dict[str, SkillRecord] = {}
        self.pending: List[SkillRecord] = []
        self.router = SkillRouter()
        self.waker = WakeLearner(cfg, seed)
        self.gate = SkillGate(cfg)
        self.sleeper = SleepConsolidator(cfg, seed, input_weight)
        self.history: List[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._sleep_result: Optional[dict] = None
        self.learned_while_asleep: List[SkillRecord] = []
        self.cycles = 0

    # ----------------------------------------------------------------- access
    @property
    def awake(self):
        return self.hemispheres[self.awake_idx]

    @property
    def asleep(self):
        return self.hemispheres[1 - self.awake_idx]

    @property
    def is_sleeping(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ birth
    def register_base_skills(self, records: List[SkillRecord], protect: bool = True) -> None:
        """Declare what the pre-trained base already knows, and protect it."""
        model = self.awake
        for r in records:
            r.status = "consolidated"
            if not r.episodes:
                r.episodes = r.sample(self.rng, self.cfg.replay_per_skill)
            self.catalog[r.name] = r
            self.router.register(r.name, build_prototype(model, r.episodes), None)
        if protect and (self.cfg.projected or self.cfg.gpm_strength > 0):
            model.eval()
            with torch.no_grad(), collecting_stats(model):
                for r in records:
                    for _ in range(2):
                        model(collate(r.sample(self.rng, 64))[0])
            extend_protected_subspaces(model, gather_covariances(model), self.cfg.gpm_threshold,
                                       self.cfg.max_protect_frac)
        self.hemispheres[1 - self.awake_idx] = _clone_sharing_engram(model)

    # ------------------------------------------------------------------- wake
    def ingest(self, record: SkillRecord, val: EvalSet, anchors: Optional[Dict[str, EvalSet]] = None,
               auto_sleep: bool = True, log_every: int = 0) -> dict:
        """Research -> learn -> gate.  Returns the gate report."""
        model = self.awake
        if record.kind == "fact":
            self.waker.learn_facts(model, record, log_every=log_every)
            acc = evaluate(model, val)["acc"]
            record.metrics["acc_skill"] = acc
            record.status = "consolidated"  # the Engram *is* long-term storage
            self.catalog[record.name] = record
            self.router.register(record.name, build_prototype(model, record.episodes), None)
            rep = {"skill": record.name, "kind": "fact", "acc": acc}
            self.history.append({"event": "fact", **rep})
            return rep
        self.waker.learn_skill(model, record, log_every=log_every)
        ok, rep = self.gate.assess(model, record, val, anchors or {})
        rep = {"skill": record.name, "accepted": ok, **rep}
        self.history.append({"event": "wake", **rep})
        if not ok:
            remove_skill(model, record.name)
            return rep
        self.catalog[record.name] = record
        self.router.register(record.name, build_prototype(model, record.episodes), record.name)
        if self.is_sleeping:
            self.learned_while_asleep.append(record)
        else:
            self.pending.append(record)
            if auto_sleep and len(self.pending) >= self.cfg.sleep_pressure:
                self.sleep()
        return rep

    # ------------------------------------------------------------------ sleep
    def _consolidate(self, snapshot, pending, old, val_sets, log_every):
        student = _clone_sharing_engram(snapshot)
        report = self.sleeper.consolidate(student, snapshot, pending, old, val_sets, log_every)
        self._sleep_result = {"student": student, "report": report, "pending": pending}

    def sleep(self, background: bool = False, val_sets: Optional[Dict[str, EvalSet]] = None,
              log_every: int = 0) -> Optional[dict]:
        if not self.pending:
            return None
        pending = list(self.pending)
        old = [r for r in self.catalog.values() if r.status == "consolidated"]
        # frozen teacher: the awake hemisphere as it is at bed-time
        snapshot = _clone_sharing_engram(self.awake)
        if background:
            self._thread = threading.Thread(target=self._consolidate,
                                            args=(snapshot, pending, old, val_sets, log_every))
            self._thread.start()
            return None
        self._consolidate(snapshot, pending, old, val_sets, log_every)
        return self.wake_up()

    def wake_up(self) -> dict:
        """Swap hemispheres once consolidation has finished."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        res = self._sleep_result
        assert res is not None, "no finished sleep to wake up from"
        self._sleep_result = None
        student = res["student"]
        # swap roles: the rested hemisphere wakes, the other is reset + synced
        self.hemispheres[1 - self.awake_idx] = student
        self.awake_idx = 1 - self.awake_idx
        self.hemispheres[1 - self.awake_idx] = _clone_sharing_engram(student)
        done = {r.name for r in res["pending"]}
        self.pending = [r for r in self.pending if r.name not in done]
        for name in list_skills(self.awake):
            remove_skill(self.awake, name)
        # base features moved: refresh every prototype on the new awake base
        for r in self.catalog.values():
            adapter = None if r.status == "consolidated" else r.name
            self.router.register(r.name, build_prototype(self.awake, r.episodes), adapter)
        self.cycles += 1
        report = dict(res["report"])
        report["cycle"] = self.cycles
        self.history.append({"event": "sleep", **report})
        # hand-over: skills learned while we slept were trained against the old
        # base; re-learn them on the new one (their episodes were kept)
        carry, self.learned_while_asleep = self.learned_while_asleep, []
        for r in carry:
            self.waker.learn_skill(self.awake, r)
            self.router.register(r.name, build_prototype(self.awake, r.episodes), r.name)
            self.pending.append(r)
        return report

    # ---------------------------------------------------------------- serving
    @torch.no_grad()
    def logits(self, inp: torch.Tensor) -> torch.Tensor:
        return self.router.logits(self.awake, inp)

    def evaluate(self, evalsets: Dict[str, EvalSet]) -> Dict[str, float]:
        m = self.awake
        return {k: evaluate(m, es, forward=self.logits)["acc"] for k, es in evalsets.items()}
