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
import zlib
from typing import Dict, List, Optional

import torch

from ..data.tasks import collate
from ..modules.lora import (collecting_stats, extend_protected_subspaces, gather_covariances,
                            list_skills, remove_skill)
from ..train import EvalSet, evaluate, sequence_accuracy
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
        # fast exact path: unique instruction token -> adapter (None = base).
        # Ambiguous/shared tokens fall back to the neural router (critic Q3).
        self.tok2adapter: Dict[int, Optional[str]] = {}
        self._tok_owner: Dict[int, str] = {}
        self._tok_ambiguous = set()
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
            self._ensure_probes(r)
            self.catalog[r.name] = r
            self.router.register(r.name, build_prototype(model, r.episodes), None)
            self._track_token(r, None)
        if protect and (self.cfg.projected or self.cfg.gpm_strength > 0):
            model.eval()
            with torch.no_grad(), collecting_stats(model):
                for r in records:
                    for _ in range(2):
                        model(collate(r.sample(self.rng, 64))[0])
            extend_protected_subspaces(model, gather_covariances(model), self.cfg.gpm_threshold,
                                       self.cfg.max_protect_frac)
        self.hemispheres[1 - self.awake_idx] = _clone_sharing_engram(model)

    def _ensure_probes(self, r: SkillRecord) -> None:
        if not r.probes and self.cfg.probes_per_skill:
            r.probes = r.sample(random.Random(zlib.crc32(r.name.encode()) + 7), self.cfg.probes_per_skill)

    def _track_token(self, record: SkillRecord, adapter: Optional[str]) -> None:
        """Map a skill's unique instruction token to its adapter.

        Only ``kind == "skill"`` records are tracked (facts share FACT).
        A token claimed by two skills becomes ambiguous and falls back
        to the neural router.
        """
        if record.kind != "skill":
            return
        tok = getattr(record.task, "task_token", None)
        if tok is None or tok in self._tok_ambiguous:
            return
        owner = self._tok_owner.get(tok)
        if owner is None:
            self._tok_owner[tok] = record.name
            self.tok2adapter[tok] = adapter
        elif owner != record.name:
            self._tok_ambiguous.add(tok)
            self.tok2adapter.pop(tok, None)
            self._tok_owner.pop(tok, None)
        else:
            self.tok2adapter[tok] = adapter

    # ------------------------------------------------------------------- wake
    def research_and_ingest(self, request, loop, anchors: Optional[Dict[str, EvalSet]] = None,
                            auto_sleep: bool = True, init_lora: Optional[dict] = None,
                            log_every: int = 0) -> dict:
        """Research a skill (gather + verify, see :mod:`recursionx.research`),
        then learn and gate it on its own held-out verified probes."""
        record, report = loop.run(request)
        if record is None:
            rep = {"skill": request.name, "accepted": False, "reason": "research failed",
                   "verified": report.n_verified, "candidates": report.n_candidates,
                   "reports": report.reports}
            self.history.append({"event": "research_failed", **rep})
            return rep
        val = EvalSet(record.task, seqs=record.probes)
        rep = self.ingest(record, val, anchors, auto_sleep, log_every, init_lora=init_lora)
        rep["provenance"] = {k: v for k, v in record.provenance.items() if k != "reports"}
        return rep

    def ingest(self, record: SkillRecord, val: EvalSet, anchors: Optional[Dict[str, EvalSet]] = None,
               auto_sleep: bool = True, log_every: int = 0, init_lora: Optional[dict] = None) -> dict:
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
        self.waker.learn_skill(model, record, log_every=log_every, init_lora=init_lora)
        self._ensure_probes(record)
        ok, rep = self.gate.assess(model, record, val, anchors or {})
        rep = {"skill": record.name, "accepted": ok, **rep}
        self.history.append({"event": "wake", **rep})
        if not ok:
            remove_skill(model, record.name)
            return rep
        self.catalog[record.name] = record
        self.router.register(record.name, build_prototype(model, record.episodes), record.name)
        self._track_token(record, record.name)
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
        audit = self.audit(snapshot, student, pending, old)
        report["audit"] = audit
        self._sleep_result = {"student": student, "report": report, "pending": pending,
                              "commit": audit["commit"]}

    def audit(self, before, after, pending, old) -> dict:
        """Retention check before the swap: every skill's held-out probes are
        run through the awake snapshot (as served, adapters routed) and through
        the consolidated student (base only).  The sleep is committed only if no
        protected skill drops by more than ``commit_max_drop`` and every new
        skill clears the gate.  Otherwise the sleep is rolled back: the awake
        hemisphere keeps serving with its adapters."""
        c = self.cfg
        served = lambda inp: self.hybrid_logits(before, inp)
        rows, syndrome = {}, []
        for r in old + pending:
            if not r.probes:
                continue
            b = sequence_accuracy(before, r.probes, forward=served)
            a = sequence_accuracy(after, r.probes)
            rows[r.name] = {"before": b, "after": a}
            if r in pending and a < c.gate_min_acc:
                syndrome.append(f"{r.name}: not consolidated ({a:.2f} < {c.gate_min_acc})")
            elif r not in pending and b - a > c.commit_max_drop:
                syndrome.append(f"{r.name}: regressed {b:.2f} -> {a:.2f}")
        commit = (not c.commit_check) or not syndrome
        return {"commit": commit, "syndrome": syndrome, "probes": rows}

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
        if not res.get("commit", True):
            # roll back: discard the student, keep serving the pending skills
            # from their adapters and retry them at the next sleep
            for r in res["pending"]:
                r.status = "accepted"
            report = dict(res["report"])
            report.update(cycle=self.cycles, committed=False)
            self.history.append({"event": "sleep_rolled_back", **report})
            self.pending = list(res["pending"]) + [r for r in self.pending if r not in res["pending"]]
            carry, self.learned_while_asleep = self.learned_while_asleep, []
            self.pending += carry
            return report
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
            self._track_token(r, adapter)
        self.cycles += 1
        report = dict(res["report"])
        report.update(cycle=self.cycles, committed=True)
        self.history.append({"event": "sleep", **report})
        # hand-over: skills learned while we slept were trained against the old
        # base; re-learn them on the new one (their episodes were kept)
        carry, self.learned_while_asleep = self.learned_while_asleep, []
        for r in carry:
            self.waker.learn_skill(self.awake, r)
            self.router.register(r.name, build_prototype(self.awake, r.episodes), r.name)
            self._track_token(r, r.name)
            self.pending.append(r)
        return report

    # ------------------------------------------------------- hybrid serving
    def hybrid_routes(self, model, inp: torch.Tensor) -> List[Optional[str]]:
        """Exact token path first, neural router as fallback.

        Sequences whose position-1 token uniquely identifies a known skill
        go straight to its adapter (or base).  Everything else — shared or
        unknown tokens — uses the ridge router.
        """
        dev = next(model.parameters()).device
        inp_d = inp.to(dev)
        toks = inp_d[:, 1].tolist() if inp_d.shape[1] > 1 else []
        routes: List[Optional[str]] = [None] * inp_d.shape[0]
        need_idx, need_inp = [], []
        for i, t in enumerate(toks):
            if int(t) in self.tok2adapter:
                routes[i] = self.tok2adapter[int(t)]
            else:
                need_idx.append(i)
        if need_idx:
            sub = inp_d[need_idx] if need_idx else inp_d[:0]
            fb = self.router.route(model, sub)
            for i, r in zip(need_idx, fb):
                routes[i] = r
        return routes

    def hybrid_logits(self, model, inp: torch.Tensor, **kw) -> torch.Tensor:
        """Routed forward using hybrid_routes (grouped by adapter)."""
        from ..modules.lora import active_adapters as _aa
        routes = self.hybrid_routes(model, inp)
        dev = next(model.parameters()).device
        inp_d = inp.to(dev)
        out = None
        for name in sorted({r for r in routes}, key=lambda r: (r is not None, r or "")):
            idx = torch.tensor([i for i, r in enumerate(routes) if r == name], device=inp_d.device)
            with _aa(model, {} if name is None else {name: 1.0}):
                lg = model(inp_d[idx], **kw).logits
            if out is None:
                out = lg.new_zeros(inp_d.shape[0], *lg.shape[1:])
            out[idx.to(lg.device)] = lg
        return out

    # ---------------------------------------------------------------- serving
    @torch.no_grad()
    def logits(self, inp: torch.Tensor) -> torch.Tensor:
        return self.hybrid_logits(self.awake, inp)

    def evaluate(self, evalsets: Dict[str, EvalSet]) -> Dict[str, float]:
        m = self.awake
        return {k: evaluate(m, es, forward=self.logits)["acc"] for k, es in evalsets.items()}
