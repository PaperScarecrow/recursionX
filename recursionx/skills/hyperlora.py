"""HyperLoRA: generate a skill adapter from a few demonstrations in one pass.

A small hypernetwork maps a *skill descriptor* (pooled base-model features of
K demonstration sequences, or any other embedding of a skill description) to
low-rank ``(A, B)`` factors for every targeted :class:`AdaptableLinear`, in
the spirit of Text-to-LoRA / Doc-to-LoRA.

* Modules are grouped by weight shape; each shape group has one output head,
  and each module gets a learned module embedding, so the parameter count
  stays independent of the number of layers that share a shape.
* The generated factors are installed as *functional* adapters
  (``AdaptableLinear.generated``), so the language-model loss backpropagates
  into the hypernetwork while the base stays frozen (meta-training).
* ``materialize`` turns a generated adapter into an ordinary trainable
  :class:`LoRAAdapter`, so the wake phase can start from the generated guess
  and refine it with gradient steps ("project, then refine").

Status: framework + tests + a small meta-training script
(``experiments/hyperlora.py``).  Whether it generalises to *unseen* skills
needs a much larger and more diverse skill family than the 20-skill
synthetic suite; see ``docs/roadmap/03_hyperlora.md``.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.tasks import PAD, collate
from ..modules.lora import AdaptableLinear, LoRAAdapter, active_adapters, adaptable_modules

LoRADict = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def default_targets(name: str) -> bool:
    """Mixers, memory and shared experts; routed experts and the loop
    injector are left to wake-time training."""
    return ".experts." not in name


@torch.no_grad()
def demo_features(model, seqs: Sequence[Sequence[int]]) -> torch.Tensor:
    """Skill descriptor from demonstrations: mean final hidden state over all
    tokens concatenated with the mean over answer tokens, averaged over demos."""
    inp, _, w = collate(list(seqs))
    with active_adapters(model, {}):
        out = model(inp, return_hidden=True)
    h = out.hidden.float()
    mask = (inp.to(h.device) != PAD).unsqueeze(-1).float()
    ans = (w.to(h.device) >= 1.0).unsqueeze(-1).float()
    mean_all = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
    mean_ans = (h * ans).sum(1) / ans.sum(1).clamp_min(1)
    return torch.cat([mean_all, mean_ans], -1).mean(0)


class HyperLoRA(nn.Module):
    def __init__(self, model, desc_dim: int, rank: int = 8, alpha: float = 16.0,
                 hidden: int = 256, target=default_targets):
        super().__init__()
        self.rank, self.scaling = rank, alpha / rank
        self.names: List[str] = []
        self.shapes: List[Tuple[int, int]] = []
        for name, m in adaptable_modules(model):
            if target(name):
                self.names.append(name)
                self.shapes.append((m.in_features, m.out_features))
        self.module_emb = nn.Embedding(len(self.names), hidden)
        nn.init.normal_(self.module_emb.weight, std=0.02)
        self.trunk = nn.Sequential(nn.LayerNorm(desc_dim), nn.Linear(desc_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden))
        self.heads = nn.ModuleDict()
        for fi, fo in sorted(set(self.shapes)):
            head = nn.Linear(hidden, (fi + fo) * rank)
            with torch.no_grad():
                # A part: ~ LoRA's default A scale; B part: exactly zero, so an
                # untrained generator produces ΔW = 0 (like LoRA's B = 0 init).
                # Without this, Adam inflates both factors at once and the
                # generated ΔW = BA explodes within ~100 steps.
                head.weight[: fi * rank].normal_(std=1.0 / (fi ** 0.5 * hidden ** 0.5))
                head.weight[fi * rank:].zero_()
                head.bias.zero_()
            self.heads[f"{fi}x{fo}"] = head

    def forward(self, desc: torch.Tensor) -> LoRADict:
        h = self.trunk(desc)
        out: LoRADict = {}
        idx = torch.arange(len(self.names), device=h.device)
        z = F.silu(h.unsqueeze(0) + self.module_emb(idx))           # (M, hidden)
        for i, (name, (fi, fo)) in enumerate(zip(self.names, self.shapes)):
            flat = self.heads[f"{fi}x{fo}"](z[i])
            A = flat[: fi * self.rank].view(self.rank, fi)
            B = flat[fi * self.rank:].view(fo, self.rank)
            out[name] = (A, B)
        return out

    # ----------------------------------------------------------- application
    def apply(self, model, lora: LoRADict) -> None:
        mods = dict(adaptable_modules(model))
        for name, (A, B) in lora.items():
            mods[name].generated = (A, B, self.scaling)

    @staticmethod
    def clear(model) -> None:
        for _, m in adaptable_modules(model):
            m.generated = None

    @torch.no_grad()
    def materialize(self, model, skill: str, lora: LoRADict) -> List[nn.Parameter]:
        """Install the generated factors as a regular trainable adapter
        (unprojected, so it reproduces the generated function exactly)."""
        mods = dict(adaptable_modules(model))
        params = []
        for name, (A, B) in lora.items():
            m = mods[name]
            ad = LoRAAdapter(m.in_features, m.out_features, self.rank, self.scaling * self.rank)
            ad.A.data.copy_(A)
            ad.B.data.copy_(B)
            ad.to(m.weight.device)
            m.adapters[skill] = ad
            params.extend(ad.parameters())
        return params


def meta_train(model, hyper: HyperLoRA, tasks: Sequence, steps: int, k_demos: int = 16,
               batch: int = 32, lr: float = 3e-4, log_every: int = 0, seed: int = 0) -> List[float]:
    """Meta-train the hypernetwork across a family of skills with the base frozen."""
    from ..train import weighted_ce
    rng = random.Random(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(hyper.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    losses = []
    for step in range(steps):
        task = rng.choice(list(tasks))
        desc = demo_features(model, [task.sample(rng) for _ in range(k_demos)])
        lora = hyper(desc)
        hyper.apply(model, lora)
        inp, tgt, w = collate([task.sample(rng) for _ in range(batch)])
        out = model(inp)
        loss = weighted_ce(out.logits, tgt, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hyper.parameters(), 1.0)
        opt.step()
        hyper.clear(model)
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"[hyperlora] step {step + 1}/{steps} loss {sum(losses[-log_every:]) / log_every:.4f}",
                  flush=True)
    for p in model.parameters():
        p.requires_grad_(True)
    return losses


@torch.no_grad()
def generate_for(model, hyper: HyperLoRA, demos: Sequence[Sequence[int]]) -> LoRADict:
    return {k: (A.detach(), B.detach()) for k, (A, B) in hyper(demo_features(model, demos)).items()}
