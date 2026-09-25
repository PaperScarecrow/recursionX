"""Sleep phase: bake accepted skills into the base weights of the sleeping
hemisphere, then protect them.

1. **NREM – merge.**  Fold each accepted projected LoRA into its base matrix
   (``W += s·B·A·(I-UUᵀ)``).  Each merge alone is exact; merging several at once
   introduces cross-talk, which the next stage repairs.
2. **REM – multi-teacher distillation with rehearsal.**  The student (sleeping
   hemisphere) is trained so that
      * on each new skill it matches the awake hemisphere *with that skill's
        adapter* (plus the ground-truth episodes), and
      * on old skills it matches the awake hemisphere's *base* on replayed
        episodes (hippocampal buffer) and on *dreams* – inputs the model
        samples from its own input distribution, labelled by the teacher.
   Gradients are projected away from the protected input subspaces (GPM), so
   rehearsal is a second line of defence rather than the only one.
3. **Growth (optional).**  If a skill still does not fit, clone an expert in
   every MoE layer, point its router row at the skill's hidden states and
   give it a few more REM steps: new, unprotected capacity instead of
   overwriting protected weights.
4. **Protect.**  Extend every layer's protected subspace with the principal
   directions of the newly consolidated skills' activations.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import torch

from ..data.tasks import BOS, EOS, SEP, collate
from ..modules.lora import (active_adapters, adaptable_modules, collecting_stats,
                            extend_protected_subspaces, gather_covariances, list_skills,
                            merge_skill, project_gradients, protected_fraction, remove_skill)
from ..modules.moe import FluidMoE
from ..train import cosine_lr, evaluate, EvalSet, weighted_ce, weighted_kl
from .config import LifecycleConfig
from .skills import SkillRecord


class SleepConsolidator:
    def __init__(self, cfg: LifecycleConfig, seed: int = 0, input_weight: float = 0.1):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.input_weight = input_weight

    # ------------------------------------------------------------------ dreams
    @torch.no_grad()
    def dream(self, model, task_tokens: List[int], n: int, max_len: int = 26,
              batch: int = 128) -> List[List[int]]:
        """Generate rehearsal sequences from the model itself: sample an input
        after the instruction token (temperature ``dream_temperature``), then
        answer greedily.  No stored data is needed."""
        if not task_tokens or n <= 0:
            return []
        model.eval()
        out: List[List[int]] = []
        tau = self.cfg.dream_temperature
        while len(out) < n:
            b = min(batch, n - len(out))
            dev = next(model.parameters()).device
            cur = torch.tensor([[BOS, self.rng.choice(task_tokens)] for _ in range(b)], device=dev)
            in_answer = torch.zeros(b, dtype=torch.bool, device=dev)
            done = torch.zeros(b, dtype=torch.bool, device=dev)
            for _ in range(max_len - 2):
                logits = model(cur).logits[:, -1]
                sampled = torch.multinomial(torch.softmax(logits / tau, -1), 1).squeeze(-1)
                greedy = logits.argmax(-1)
                nxt = torch.where(in_answer, greedy, sampled)
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                cur = torch.cat([cur, nxt.unsqueeze(1)], 1)
                in_answer |= nxt == SEP
                done |= nxt == EOS
                if bool(done.all()):
                    break
            for row in cur.tolist():
                if EOS in row:
                    row = row[:row.index(EOS) + 1]
                    if row.count(SEP) == 1 and row.index(SEP) > 2:
                        out.append(row)
            if len(out) == 0 and len(cur) and not bool(done.any()):
                break  # model cannot dream yet; avoid an endless loop
        return out[:n]

    # ------------------------------------------------------------- consolidate
    def _rem_params(self, student) -> List[torch.nn.Parameter]:
        if self.cfg.rem_params == "all":
            return [p for n, p in student.named_parameters() if not n.startswith("engram")]
        params = [m.weight for _, m in adaptable_modules(student)]
        for moe in student.moe_layers():
            if getattr(moe, "_grown", False):
                params += [moe.router.weight, moe.router_bias]
        return params

    def _rem(self, student, teacher, new_records, old_records, dreams, steps, log_every=0):
        c = self.cfg
        params = self._rem_params(student)
        ids = {id(p) for p in params}
        for p in student.parameters():
            p.requires_grad_(id(p) in ids)
        opt = torch.optim.AdamW(params, lr=c.rem_lr, weight_decay=0.0, betas=(0.9, 0.98))
        replay = [e for r in old_records for e in r.episodes]
        dreams_by_token: Dict[int, List[List[int]]] = {}
        for d in dreams:
            dreams_by_token.setdefault(d[1], []).append(d)
        student.train()
        teacher.eval()
        hist = []
        t0 = time.time()
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, steps, c.rem_lr, warmup=10)
            loss = torch.zeros((), device=next(student.parameters()).device)
            sources = []
            for r in new_records:
                seqs = r.sample(self.rng, c.rem_batch)
                if r.episodes:
                    seqs[: c.rem_batch // 4] = self.rng.sample(r.episodes, min(len(r.episodes), c.rem_batch // 4))
                sources.append((seqs, {r.name: 1.0} if r.kind == "skill" else {}, True))
            if c.old_batch_per_skill > 0 and old_records:
                # stratified rehearsal: every old skill gets the same share
                seqs = []
                for r in old_records:
                    k = c.old_batch_per_skill
                    pool = dreams_by_token.get(getattr(r.task, "task_token", None), [])
                    n_dream = int(round(k * c.dream_frac)) if pool else 0
                    src = r.episodes or pool
                    seqs += [self.rng.choice(src) for _ in range(k - n_dream)] if src else []
                    seqs += [self.rng.choice(pool) for _ in range(n_dream)]
                if seqs:
                    sources.append((seqs, {}, True))
            elif replay or dreams:
                n_dream = int(round(c.rem_batch * c.dream_frac)) if dreams else 0
                n_rep = c.rem_batch - n_dream if replay else 0
                seqs = [self.rng.choice(replay) for _ in range(n_rep)]
                seqs += [self.rng.choice(dreams) for _ in range(n_dream)]
                sources.append((seqs, {}, True))
            for seqs, adapters, use_ce in sources:
                inp, tgt, w = collate(seqs, self.input_weight)
                with torch.no_grad(), active_adapters(teacher, adapters):
                    t_logits = teacher(inp).logits
                out = student(inp)
                l = c.kl_weight * weighted_kl(out.logits, t_logits, w) + out.aux_loss
                if use_ce and c.ce_weight:
                    l = l + c.ce_weight * weighted_ce(out.logits, tgt, w)
                loss = loss + l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if c.gpm_strength > 0:
                project_gradients(student, c.gpm_strength)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            hist.append(loss.item())
            if log_every and (step + 1) % log_every == 0:
                print(f"[rem] step {step + 1}/{steps} loss {sum(hist[-log_every:]) / log_every:.4f}"
                      f" ({time.time() - t0:.0f}s)", flush=True)
        for p in student.parameters():
            p.requires_grad_(True)
        return hist

    @torch.no_grad()
    def _grow(self, student, record: SkillRecord, old_records: List[SkillRecord]) -> int:
        """Add one expert per MoE layer, routed towards ``record``'s tokens."""
        moes = student.moe_layers()
        means: Dict[int, List[torch.Tensor]] = {id(m): [] for m in moes}
        hooks = [m.register_forward_hook(lambda mod, a, o: means[id(mod)].append(a[0].reshape(-1, a[0].shape[-1])))
                 for m in moes]
        student.eval()
        for m in moes:
            m.track_usage, m.usage = True, torch.zeros(m.n_experts)
        student(collate(record.sample(self.rng, 64))[0])
        new_feats = {k: torch.cat(v).mean(0) for k, v in means.items()}
        for k in means:
            means[k] = []
        old = [e for r in old_records for e in r.episodes[:16]]
        if old:
            student(collate(old)[0])
        for h in hooks:
            h.remove()
        grown = 0
        for m in moes:
            m.track_usage = False
            src = int(m.usage.cpu().argmax())
            mu_new = new_feats[id(m)]
            mu_old = torch.cat(means[id(m)]).mean(0) if old else torch.zeros_like(mu_new)
            direction = torch.nn.functional.normalize(mu_new - mu_old, dim=0) * m.router.weight.norm(dim=1).mean() * 4
            m.grow_expert(src, direction, bias=0.0, noise=0.01)
            m._grown = True
            grown += 1
        return grown

    def consolidate(self, student, teacher, new_records: List[SkillRecord],
                    old_records: List[SkillRecord], val_sets: Optional[Dict[str, EvalSet]] = None,
                    log_every: int = 0) -> Dict[str, object]:
        c = self.cfg
        report: Dict[str, object] = {"skills": [r.name for r in new_records]}
        t0 = time.time()
        # 1. NREM: fold adapters into the base
        for r in new_records:
            if r.kind == "skill" and c.nrem_merge and r.name in list_skills(student):
                merge_skill(student, r.name)
        for name in list_skills(student):
            remove_skill(student, name)
        # 2. REM: distill + rehearse (with dreams)
        old_skill_tokens = sorted({r.task.task_token for r in old_records if r.kind == "skill"})
        dreams = []
        if c.dream_frac > 0 and c.rem_steps > 0:
            with active_adapters(teacher, {}):
                dreams = self.dream(teacher, old_skill_tokens, c.dream_pool)
        report["n_dreams"] = len(dreams)
        if c.rem_steps > 0:
            hist = self._rem(student, teacher, new_records, old_records, dreams, c.rem_steps, log_every)
            report["rem_final_loss"] = sum(hist[-20:]) / min(20, len(hist))
        # 3. growth for skills that still do not fit
        if c.grow_experts and val_sets:
            weak = [r for r in new_records if r.kind == "skill"
                    and evaluate(student, val_sets[r.name])["acc"] < c.growth_acc_threshold]
            if weak:
                for r in weak:
                    self._grow(student, r, old_records)
                self._rem(student, teacher, new_records, old_records, dreams,
                          c.growth_extra_steps, log_every)
                report["grown_for"] = [r.name for r in weak]
        # 4. protect the newly consolidated knowledge
        if c.gpm_strength > 0 or c.projected:
            student.eval()
            with torch.no_grad(), collecting_stats(student):
                for r in new_records:
                    for _ in range(2):
                        student(collate(r.sample(self.rng, 64))[0])
            covs = gather_covariances(student)
            extend_protected_subspaces(student, covs, c.gpm_threshold, c.max_protect_frac)
        report["protected_fraction"] = protected_fraction(student)
        for r in new_records:
            r.status = "consolidated"
        report["seconds"] = time.time() - t0
        return report
