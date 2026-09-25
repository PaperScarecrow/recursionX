"""The research loop: request -> gather -> verify -> split -> SkillRecord.

The resulting :class:`SkillRecord` carries

* ``task``: an :class:`EpisodeTask` that samples *only* verified examples,
* ``episodes``: a small replay buffer (for rehearsal during sleep),
* ``probes``: held-out verified examples never used for training (the
  retention registry used by the sleep audit),
* ``provenance``: which sources contributed, and every verifier report
  (including conflict syndromes).

``DualHemisphereBrain.research_and_ingest`` runs this loop and then the
usual wake → gate path.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..data.tasks import BOS, EOS, SEP, SYM0
from ..lifecycle.skills import SkillRecord
from .sources import Example, KnowledgeSource, ResearchResult, SkillRequest
from .verify import run_verifiers


def encode(task_token: int, x: Sequence[int], y: Sequence[int]) -> List[int]:
    return [BOS, task_token] + [SYM0 + v for v in x] + [SEP] + [SYM0 + v for v in y] + [EOS]


class EpisodeTask:
    """A skill known only through a finite pool of verified examples."""

    def __init__(self, name: str, task_token: int, examples: Sequence[Example]):
        self.name, self.task_token = name, task_token
        self.examples = list(examples)
        self.seqs = [encode(task_token, x, y) for x, y in self.examples]

    def sample(self, rng: random.Random) -> List[int]:
        return rng.choice(self.seqs)

    def sample_io(self, rng: random.Random):
        return rng.choice(self.examples)


@dataclass
class ResearchReport:
    request: SkillRequest
    n_candidates: int
    n_verified: int
    reports: list
    ok: bool


class ResearchLoop:
    def __init__(self, sources: Sequence[KnowledgeSource], verifiers: Sequence = (),
                 min_examples: int = 64, max_rounds: int = 3, probe_frac: float = 0.15,
                 episodes: int = 64, seed: int = 0):
        self.sources, self.verifiers = list(sources), list(verifiers)
        self.min_examples, self.max_rounds = min_examples, max_rounds
        self.probe_frac, self.episodes = probe_frac, episodes
        self.rng = random.Random(seed)

    def gather(self, request: SkillRequest) -> List[ResearchResult]:
        return [s.research(request, self.rng) for s in self.sources]

    def run(self, request: SkillRequest) -> tuple[Optional[SkillRecord], ResearchReport]:
        assert request.task_token is not None, "request needs an instruction token"
        results: List[ResearchResult] = []
        verified: List[Example] = []
        reports: list = []
        for _ in range(self.max_rounds):
            results += self.gather(request)
            verified, reports = run_verifiers(results, self.verifiers)
            if len(verified) >= self.min_examples:
                break
        n_cand = sum(len(r.examples) for r in results)
        ok = len(verified) >= self.min_examples
        report = ResearchReport(request, n_cand, len(verified), reports, ok)
        if not ok:
            return None, report
        self.rng.shuffle(verified)
        n_probe = max(1, int(len(verified) * self.probe_frac))
        probes, train = verified[:n_probe], verified[n_probe:]
        task = EpisodeTask(request.name, request.task_token, train)
        rec = SkillRecord(request.name, task, new_tokens=[request.task_token])
        rec.probes = [encode(request.task_token, x, y) for x, y in probes]
        rec.episodes = [self.rng.choice(task.seqs) for _ in range(min(self.episodes, len(task.seqs)))]
        rec.provenance = {"sources": sorted({r.source for r in results}),
                          "candidates": n_cand, "verified": len(verified),
                          "reports": reports, "description": request.description}
        return rec, report
