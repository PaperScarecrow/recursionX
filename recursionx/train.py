"""Training / evaluation utilities shared by the lifecycle and experiments."""
from __future__ import annotations

import math
import zlib
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .data.tasks import collate


def weighted_ce(logits: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    tgt, w = tgt.to(logits.device), w.to(logits.device)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
                         reduction="none")
    return (ce * w.reshape(-1)).sum() / w.sum().clamp_min(1e-6)


def weighted_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, w: torch.Tensor,
                T: float = 1.0) -> torch.Tensor:
    """KL(teacher || student) per position, weighted."""
    w = w.to(student_logits.device)
    s = F.log_softmax(student_logits.float() / T, -1)
    t = F.log_softmax(teacher_logits.to(student_logits.device).float() / T, -1)
    kl = (t.exp() * (t - s)).sum(-1)
    return (kl * w).sum() / w.sum().clamp_min(1e-6) * (T * T)


class EvalSet:
    """A fixed, pre-collated evaluation set for one task."""

    def __init__(self, task, n: int = 256, seed: int = 12345, seqs=None):
        if seqs is None:
            rng = random.Random(seed + zlib.crc32(task.name.encode()) % 10007)
            if hasattr(task, "all_examples"):
                seqs = task.all_examples()
            else:
                seqs = [task.sample(rng) for _ in range(n)]
        self.task = task
        self.inp, self.tgt, self.w = collate(seqs)
        self.out_mask = self.w >= 1.0


@torch.no_grad()
def evaluate(model, es: EvalSet, batch: int = 256, n_loops: Optional[int] = None,
             forward: Optional[Callable] = None) -> Dict[str, float]:
    was = model.training
    model.eval()
    correct_seq, correct_tok, n_seq, n_tok = 0, 0, 0, 0
    for s in range(0, es.inp.shape[0], batch):
        inp, tgt, m = es.inp[s:s + batch], es.tgt[s:s + batch], es.out_mask[s:s + batch]
        logits = forward(inp) if forward else model(inp, n_loops=n_loops).logits
        tgt, m = tgt.to(logits.device), m.to(logits.device)
        ok = (logits.argmax(-1) == tgt) | ~m
        correct_seq += ok.all(-1).sum().item()
        correct_tok += ((logits.argmax(-1) == tgt) & m).sum().item()
        n_seq += inp.shape[0]
        n_tok += m.sum().item()
    model.train(was)
    return {"acc": correct_seq / n_seq, "tok_acc": correct_tok / max(n_tok, 1)}


@torch.no_grad()
def sequence_accuracy(model, seqs: List[List[int]], forward: Optional[Callable] = None) -> float:
    """Exact-match accuracy on a list of raw sequences (e.g. retention probes)."""
    inp, tgt, w = collate(seqs)
    was = model.training
    model.eval()
    logits = forward(inp) if forward else model(inp).logits
    model.train(was)
    tgt, w = tgt.to(logits.device), w.to(logits.device)
    ok = (logits.argmax(-1) == tgt) | (w < 1.0)
    return ok.all(-1).float().mean().item()


def evaluate_all(model, evalsets: Dict[str, EvalSet], **kw) -> Dict[str, float]:
    return {k: evaluate(model, es, **kw)["acc"] for k, es in evalsets.items()}


def cosine_lr(step: int, total: int, base: float, warmup: int = 20, floor: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def train_loop(model, params: Sequence[torch.nn.Parameter], sampler: Callable[[int], List[List[int]]],
               steps: int, lr: float = 3e-3, batch: int = 64, input_weight: float = 0.0,
               weight_decay: float = 0.0, loss_fn: Optional[Callable] = None,
               after_backward: Optional[Callable] = None, log_every: int = 0,
               log_prefix: str = "", clip: float = 1.0,
               callback: Optional[Callable[[int], None]] = None) -> List[float]:
    """Generic loop.  ``loss_fn(model, inp, tgt, w) -> loss`` overrides plain CE."""
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    model.train()
    losses = []
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        inp, tgt, w = collate(sampler(batch), input_weight)
        if loss_fn is None:
            out = model(inp)
            loss = weighted_ce(out.logits, tgt, w) + out.aux_loss
        else:
            loss = loss_fn(model, inp, tgt, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if after_backward is not None:
            after_backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"{log_prefix}step {step + 1}/{steps} loss {avg:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            if callback is not None:
                callback(step + 1)
    return losses


def freeze(model, trainable: Iterable[torch.nn.Parameter] = ()) -> None:
    ids = {id(p) for p in trainable}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)


def unfreeze(model) -> None:
    for p in model.parameters():
        p.requires_grad_(True)
