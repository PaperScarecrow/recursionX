"""Wake phase: acquire new skills as projected LoRA adapters, and new facts
as sparse writes to the Engram tables.  Base weights stay frozen."""
from __future__ import annotations

import random
from typing import Dict, Optional

import torch

from ..data.tasks import collate
from ..modules.lora import (adaptable_modules, add_skill_adapter, collecting_stats,
                            gather_covariances, remove_skill, set_active_adapters)
from ..train import freeze, train_loop, unfreeze
from .config import LifecycleConfig
from .skills import SkillRecord


class WakeLearner:
    def __init__(self, cfg: LifecycleConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)

    @torch.no_grad()
    def skill_covariances(self, model, record: SkillRecord) -> Dict[str, torch.Tensor]:
        model.eval()
        with collecting_stats(model):
            for _ in range(self.cfg.stats_batches):
                inp, _, _ = collate(record.sample(self.rng, self.cfg.wake_batch))
                model(inp)
        return gather_covariances(model)

    def learn_skill(self, model, record: SkillRecord, steps: Optional[int] = None,
                    log_every: int = 0, init_lora: Optional[dict] = None) -> Dict[str, float]:
        """Research -> reason -> write a projected LoRA for ``record``.

        ``init_lora`` (module name -> (A, B)), e.g. from a HyperLoRA, seeds the
        adapter before refinement; modules it does not cover keep the default
        (data-projected) initialisation."""
        c = self.cfg
        record.status = "learning"
        if not record.episodes:  # keep a small episodic buffer for rehearsal later
            record.episodes = record.sample(self.rng, c.replay_per_skill)
        remove_skill(model, record.name)
        covs = self.skill_covariances(model, record) if c.data_init else None
        params = add_skill_adapter(model, record.name, c.rank, c.alpha,
                                   projected=c.projected, covs=covs)
        if init_lora is not None:
            mods = dict(adaptable_modules(model))
            with torch.no_grad():
                for name, (A, B) in init_lora.items():
                    ad = mods[name].adapters[record.name]
                    if ad.A.shape == A.shape and ad.B.shape == B.shape:
                        ad.A.copy_(A)
                        ad.B.copy_(B)
        after = None
        if c.train_token_rows and record.new_tokens:
            params = params + [model.embed.weight]
            rows = torch.tensor(record.new_tokens)
            mask = torch.zeros(model.embed.weight.shape[0], 1)
            mask[rows] = 1.0

            def after():
                g = model.embed.weight.grad
                if g is not None:
                    g.mul_(mask.to(g))
        freeze(model, params)
        set_active_adapters(model, {record.name: 1.0})
        losses = train_loop(model, params, lambda n: record.sample(self.rng, n),
                            steps or c.wake_steps, lr=c.wake_lr, batch=c.wake_batch,
                            after_backward=after, log_every=log_every,
                            log_prefix=f"[wake:{record.name}] ")
        set_active_adapters(model, {})
        unfreeze(model)
        record.metrics["wake_final_loss"] = sum(losses[-20:]) / min(20, len(losses))
        return record.metrics

    def learn_facts(self, model, record: SkillRecord, steps: Optional[int] = None,
                    log_every: int = 0) -> Dict[str, float]:
        """Declarative knowledge: only the Engram rows addressed by the facts'
        n-grams are written (row-local, so nothing else can be disturbed)."""
        c = self.cfg
        assert model.engram is not None, "fact learning needs the Engram memory"
        record.status = "learning"
        eng = model.engram
        steps = steps or c.fact_steps
        if eng.disk is not None:
            model.eval()  # disk path does manual sparse SGD
            for p in model.parameters():
                p.requires_grad_(False)
            eng.train()
            from ..train import weighted_ce
            for _ in range(steps):
                inp, tgt, w = collate(record.sample(self.rng, c.wake_batch))
                loss = weighted_ce(model(inp).logits, tgt, w)
                loss.backward()
                eng.apply_sparse_grads(c.fact_lr)
            unfreeze(model)
        else:
            freeze(model, [eng.table])
            losses = train_loop(model, [eng.table], lambda n: record.sample(self.rng, n), steps,
                                lr=c.fact_lr, batch=c.wake_batch, log_every=log_every,
                                log_prefix=f"[facts:{record.name}] ", clip=0)
            unfreeze(model)
            record.metrics["wake_final_loss"] = sum(losses[-20:]) / min(20, len(losses))
        if not record.episodes:
            record.episodes = record.sample(self.rng, c.replay_per_skill)
        return record.metrics
