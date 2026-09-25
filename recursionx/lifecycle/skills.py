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
    """Mean prelude state over the prompt (tokens up to and including SEP),
    computed on the *base* weights (no adapters)."""
    with active_adapters(model, {}):
        x = model.embed(inp)
        if model.engram is not None:
            x = x + model.engram(inp, x)
        for blk in model.prelude:
            x = blk(x)
    is_sep = (inp == SEP).int()
    before = (is_sep.cumsum(1) - is_sep) == 0   # positions up to the first SEP
    mask = (before & (inp != PAD)).unsqueeze(-1).float()
    return (x * mask).sum(1) / mask.sum(1).clamp_min(1)


class SkillRouter:
    """Nearest-prototype routing between the base and pending skill adapters.

    Every known skill (consolidated or pending) has a prototype.  A prompt is
    routed to the adapter of its nearest prototype when that skill is still
    pending (lives in an adapter); otherwise it runs on the base weights.
    """

    def __init__(self):
        self.prototypes: Dict[str, torch.Tensor] = {}
        self.adapter_of: Dict[str, Optional[str]] = {}

    def register(self, name: str, proto: torch.Tensor, adapter: Optional[str]) -> None:
        self.prototypes[name] = proto
        self.adapter_of[name] = adapter

    def set_adapter(self, name: str, adapter: Optional[str]) -> None:
        self.adapter_of[name] = adapter

    def _centered(self):
        names = list(self.prototypes)
        P = torch.stack([self.prototypes[n] for n in names])
        mu = P.mean(0, keepdim=True)
        return names, torch.nn.functional.normalize(P - mu, dim=-1), mu

    @torch.no_grad()
    def route(self, model, inp: torch.Tensor) -> List[Optional[str]]:
        if not self.prototypes:
            return [None] * inp.shape[0]
        names, P, mu = self._centered()
        f = torch.nn.functional.normalize(prompt_features(model, inp) - mu, dim=-1)
        best = (f @ P.t()).argmax(-1)
        return [self.adapter_of[names[i]] for i in best.tolist()]

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
    inp, _, _ = collate(seqs)
    return prompt_features(model, inp).mean(0)
