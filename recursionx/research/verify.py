"""Verifiers: decide which researched examples are safe to learn from.

Each verifier returns the accepted examples plus a report.  Verifiers are
chained; an example has to pass all of them.

* :class:`ExecutionVerifier` – re-computes the output with a trusted program
  (ground truth when one exists: code, maths, unit conversions, ...).
* :class:`ConsistencyVerifier` – the "syndrome" check: the same input seen
  with different outputs (across sources or within one) is a conflict.  All
  conflicting examples are rejected and reported, so a human or a stronger
  check can resolve them, instead of the model training on noise.
* :class:`AgreementVerifier` – keeps only examples on which at least ``k``
  independent sources agree.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

from .sources import Example, ResearchResult

Report = Dict[str, object]


def _key(x: Sequence[int]) -> Tuple[int, ...]:
    return tuple(x)


class ExecutionVerifier:
    name = "execution"

    def __init__(self, fn: Callable[[List[int], int], List[int]], n_symbols: int = 16):
        self.fn, self.n_symbols = fn, n_symbols

    def __call__(self, results: Sequence[ResearchResult]) -> Tuple[List[Example], Report]:
        ok, bad = [], 0
        for r in results:
            for x, y in r.examples:
                try:
                    good = list(self.fn(list(x), self.n_symbols)) == list(y)
                except Exception:
                    good = False
                if good:
                    ok.append((x, y))
                else:
                    bad += 1
        return ok, {"verifier": self.name, "accepted": len(ok), "rejected": bad}


class ConsistencyVerifier:
    name = "consistency"

    def __call__(self, results: Sequence[ResearchResult]) -> Tuple[List[Example], Report]:
        seen: Dict[Tuple[int, ...], set] = defaultdict(set)
        sources: Dict[Tuple[int, ...], set] = defaultdict(set)
        for r in results:
            for x, y in r.examples:
                seen[_key(x)].add(tuple(y))
                sources[_key(x)].add(r.source)
        conflicts = {k: v for k, v in seen.items() if len(v) > 1}
        ok = [(list(k), list(next(iter(v)))) for k, v in seen.items() if len(v) == 1]
        syndrome = [{"input": list(k), "outputs": [list(o) for o in v],
                     "sources": sorted(sources[k])} for k, v in list(conflicts.items())[:20]]
        return ok, {"verifier": self.name, "accepted": len(ok), "conflicts": len(conflicts),
                    "syndrome": syndrome}


class AgreementVerifier:
    name = "agreement"

    def __init__(self, k: int = 2):
        self.k = k

    def __call__(self, results: Sequence[ResearchResult]) -> Tuple[List[Example], Report]:
        votes: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], set] = defaultdict(set)
        for r in results:
            for x, y in r.examples:
                votes[(_key(x), tuple(y))].add(r.source)
        ok = [(list(x), list(y)) for (x, y), s in votes.items() if len(s) >= self.k]
        return ok, {"verifier": self.name, "accepted": len(ok), "min_sources": self.k}


def run_verifiers(results: Sequence[ResearchResult], verifiers: Sequence) -> Tuple[List[Example], List[Report]]:
    """Chain verifiers: each one sees only what the previous accepted."""
    reports: List[Report] = []
    current = list(results)
    accepted: List[Example] = [e for r in results for e in r.examples]
    for v in verifiers:
        accepted, rep = v(current)
        reports.append(rep)
        current = [ResearchResult(accepted, f"after:{v.name}")]
    # de-duplicate, keep order
    uniq, seen = [], set()
    for x, y in accepted:
        k = (_key(x), tuple(y))
        if k not in seen:
            seen.add(k)
            uniq.append((x, y))
    return uniq, reports
