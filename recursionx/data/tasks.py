"""Synthetic skill suite used to test continual skill acquisition.

Every example is ``[BOS, TASK_k, x_1..x_n, SEP, y_1..y_m, EOS, PAD...]`` where
``x`` is a random string over ``n_symbols`` symbols and ``y = f_k(x)``.
Tasks are chosen so that each is a distinct algorithmic *skill*; the task
token is the "instruction".  Teacher-forced exact match of every output token
equals greedy-decoding exact match, so evaluation is a single forward pass.

``FactTask`` is a *knowledge* task: a fixed random mapping from two-symbol
entity names to attribute symbols.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

PAD, BOS, SEP, EOS, FACT = 0, 1, 2, 3, 4
N_TASK_SLOTS = 24
TASK0 = 5
SYM0 = TASK0 + N_TASK_SLOTS  # 29


@dataclass
class Vocab:
    n_symbols: int = 16

    @property
    def size(self) -> int:
        return SYM0 + self.n_symbols

    def sym(self, i: int) -> int:
        return SYM0 + i


def _mirror_sum(x, V):
    n = len(x)
    return [(x[i] + x[n - 1 - i]) % V for i in range(n)]


def _cumsum(x, V):
    out, s = [], 0
    for v in x:
        s = (s + v) % V
        out.append(s)
    return out


def _swap_pairs(x, V):
    y = list(x)
    for i in range(0, len(y) - 1, 2):
        y[i], y[i + 1] = y[i + 1], y[i]
    return y


def _dedup(x, V):
    out = []
    for v in x:
        if not out or out[-1] != v:
            out.append(v)
    return out


SKILLS: Dict[str, Callable] = {
    "copy": lambda x, V: list(x),
    "reverse": lambda x, V: list(reversed(x)),
    "succ": lambda x, V: [(v + 1) % V for v in x],
    "pred": lambda x, V: [(v - 1) % V for v in x],
    "sort": lambda x, V: sorted(x),
    "max": lambda x, V: [max(x)],
    "min": lambda x, V: [min(x)],
    "rotl": lambda x, V: list(x[1:]) + list(x[:1]),
    "rotr": lambda x, V: list(x[-1:]) + list(x[:-1]),
    "swap_pairs": _swap_pairs,
    "interleave": lambda x, V: list(x[0::2]) + list(x[1::2]),
    "double": lambda x, V: [v for v in x for _ in range(2)],
    "cumsum": _cumsum,
    "mirror_sum": _mirror_sum,
    "dedup": _dedup,
    "first_last": lambda x, V: [x[0], x[-1]],
    "count_first": lambda x, V: [sum(1 for v in x if v == x[0])],
    "sort_desc": lambda x, V: sorted(x, reverse=True),
    "add_first": lambda x, V: [(v + x[0]) % V for v in x],
    "skip_first": lambda x, V: list(x[1:]),
}


class SkillTask:
    """An algorithmic skill bound to a task-token slot."""

    def __init__(self, name: str, slot: int, vocab: Vocab, min_len: int = 3, max_len: int = 8,
                 fn: Optional[Callable] = None, repeat_prob: float = 0.0):
        self.name, self.slot, self.vocab = name, slot, vocab
        self.fn = fn or SKILLS[name]
        self.min_len, self.max_len = min_len, max_len
        self.repeat_prob = repeat_prob
        self.task_token = TASK0 + slot

    def sample_io(self, rng: random.Random):
        V = self.vocab.n_symbols
        n = rng.randint(self.min_len, self.max_len)
        x = [rng.randrange(V) for _ in range(n)]
        if self.repeat_prob:
            for i in range(1, n):
                if rng.random() < self.repeat_prob:
                    x[i] = x[i - 1]
        return x, self.fn(x, V)

    def encode(self, x: Sequence[int], y: Sequence[int]) -> List[int]:
        s = self.vocab.sym
        return [BOS, self.task_token] + [s(v) for v in x] + [SEP] + [s(v) for v in y] + [EOS]

    def sample(self, rng: random.Random) -> List[int]:
        return self.encode(*self.sample_io(rng))

    def prompt_length(self, seq: Sequence[int]) -> int:
        return list(seq).index(SEP) + 1


class FactTask:
    """Knowledge: entity (two symbols) -> attribute (one symbol)."""

    def __init__(self, name: str, vocab: Vocab, n_facts: int = 64, seed: int = 0):
        self.name, self.vocab = name, vocab
        rng = random.Random(seed)
        V = vocab.n_symbols
        pairs = [(a, b) for a in range(V) for b in range(V)]
        rng.shuffle(pairs)
        self.facts = {p: rng.randrange(V) for p in pairs[:n_facts]}
        self.keys = list(self.facts)
        self.task_token = FACT

    def sample(self, rng: random.Random) -> List[int]:
        k = rng.choice(self.keys)
        s = self.vocab.sym
        return [BOS, FACT, s(k[0]), s(k[1]), SEP, s(self.facts[k]), EOS]

    def all_examples(self) -> List[List[int]]:
        s = self.vocab.sym
        return [[BOS, FACT, s(a), s(b), SEP, s(v), EOS] for (a, b), v in self.facts.items()]


def collate(seqs: List[List[int]], input_weight: float = 0.0, device="cpu"):
    """Pad and build next-token targets + per-token loss weights.

    Output tokens (after SEP, incl. EOS) get weight 1.  Input symbols (between
    the task token and SEP, incl. SEP) get ``input_weight`` – a small LM loss on
    inputs lets the model later *dream* plausible inputs for rehearsal.
    """
    T = max(len(s) for s in seqs)
    tok = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        tok[i, :len(s)] = torch.tensor(s)
    inp, tgt = tok[:, :-1], tok[:, 1:]
    w = torch.zeros_like(tgt, dtype=torch.float)
    for i, s in enumerate(seqs):
        sep = s.index(SEP)
        # target position j predicts token j+1
        w[i, sep:len(s) - 1] = 1.0
        if input_weight > 0:
            w[i, 1:sep] = input_weight
    return inp.to(device), tgt.to(device), w.to(device)


class Mixture:
    """Sample batches from several tasks with given weights."""

    def __init__(self, tasks: Sequence, weights: Optional[Sequence[float]] = None, seed: int = 0):
        self.tasks = list(tasks)
        self.weights = list(weights) if weights else [1.0] * len(self.tasks)
        self.rng = random.Random(seed)

    def sample_seqs(self, n: int) -> List[List[int]]:
        ts = self.rng.choices(self.tasks, weights=self.weights, k=n)
        return [t.sample(self.rng) for t in ts]


def make_suite(names: Sequence[str], vocab: Vocab, start_slot: int = 0, **kw) -> List[SkillTask]:
    return [SkillTask(n, start_slot + i, vocab, **kw) for i, n in enumerate(names)]
