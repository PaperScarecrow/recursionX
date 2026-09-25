"""Skill records and the skill router (the "fluid" mixture of skill adapters)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..data.tasks import SEP, PAD, collate
from ..modules.lora import active_adapters


@dataclass
class SkillRecord:
    name: str
    task: object                       # anything with .sample(rng) -> seq
    kind: str = "skill"                # "skill" (procedural) | "fact" (declarative)
    status: str = "new"                # new -> learning -> accepted/rejected -> consolidated
    episodes: List[List[int]] = field(default_factory=list)   # replay buffer
    prototype: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    new_tokens: List[int] = field(default_factory=list)

    def sample(self, rng: random.Random, n: int) -> List[List[int]]:
        return [self.task.sample(rng) for _ in range(n)]


@torch.no_grad()
def prompt_features(model, inp: torch.Tensor) -> torch.Tensor:
    """Prompt embedding from the *base* prelude (no adapters): the mean state
    over the prompt (up to and including SEP) concatenated with the state at
    the end of the prompt, which in a causal model summarises it."""
    with active_adapters(model, {}):
        x = model.embed(inp)
        if model.engram is not None:
            x = x + model.engram(inp, x)
        for blk in model.prelude:
            x = blk(x)
    is_sep = (inp == SEP).int()
    before = (is_sep.cumsum(1) - is_sep) == 0   # positions up to the first SEP
    mask = (before & (inp != PAD)).unsqueeze(-1).float()
    mean = (x * mask).sum(1) / mask.sum(1).clamp_min(1)
    last_pos = (mask.squeeze(-1).sum(1) - 1).long().clamp_min(0)
    last = x[torch.arange(inp.shape[0]), last_pos]
    return torch.cat([mean, last], -1)


class SkillRouter:
    """Routes a prompt to the skill it belongs to.

    Every known skill (consolidated or pending) contributes exemplar features
    from its stored episodes; a ridge-regression classifier over those
    features picks the skill.  Prompts of pending skills run with that skill's
    adapter; everything else runs on the base weights.
    """

    def __init__(self, ridge: float = 1e-2):
        self.exemplars: Dict[str, torch.Tensor] = {}
        self.adapter_of: Dict[str, Optional[str]] = {}
        self.ridge = ridge
        self._fit = None

    def register(self, name: str, feats: torch.Tensor, adapter: Optional[str]) -> None:
        self.exemplars[name] = feats if feats.dim() == 2 else feats.unsqueeze(0)
        self.adapter_of[name] = adapter
        self._fit = None

    def set_adapter(self, name: str, adapter: Optional[str]) -> None:
        self.adapter_of[name] = adapter

    def _classifier(self):
        if self._fit is None:
            names = list(self.exemplars)
            X = torch.cat([self.exemplars[n] for n in names]).double()
            y = torch.cat([torch.full((self.exemplars[n].shape[0],), i) for i, n in enumerate(names)])
            mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
            Z = torch.cat([(X - mu) / sd, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
            Y = torch.nn.functional.one_hot(y, len(names)).double()
            A = Z.t() @ Z + self.ridge * X.shape[0] * torch.eye(Z.shape[1], dtype=X.dtype)
            W = torch.linalg.solve(A, Z.t() @ Y)
            self._fit = (names, mu, sd, W)
        return self._fit

    @torch.no_grad()
    def classify(self, model, inp: torch.Tensor) -> List[str]:
        names, mu, sd, W = self._classifier()
        X = prompt_features(model, inp).double()
        Z = torch.cat([(X - mu) / sd, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
        return [names[i] for i in (Z @ W).argmax(-1).tolist()]

    @torch.no_grad()
    def route(self, model, inp: torch.Tensor) -> List[Optional[str]]:
        if not self.exemplars:
            return [None] * inp.shape[0]
        return [self.adapter_of[n] for n in self.classify(model, inp)]

    def logits(self, model, inp: torch.Tensor, **kw) -> torch.Tensor:
        """Routed forward: each sequence runs with (at most) its own skill
        adapter.  The batch is split by route so that every module (including
        the token-level MoE dispatch) sees a homogeneous adapter set."""
        routes = self.route(model, inp)
        out = None
        for name in sorted({r for r in routes}, key=lambda r: (r is not None, r or "")):
            idx = torch.tensor([i for i, r in enumerate(routes) if r == name], device=inp.device)
            with active_adapters(model, {} if name is None else {name: 1.0}):
                lg = model(inp[idx], **kw).logits
            if out is None:
                out = lg.new_zeros(inp.shape[0], *lg.shape[1:])
            out[idx] = lg
        return out


def build_prototype(model, seqs: List[List[int]]) -> torch.Tensor:
    """Exemplar features (one row per episode) used to train the router."""
    inp, _, _ = collate(seqs)
    return prompt_features(model, inp)
