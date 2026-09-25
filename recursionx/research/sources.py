"""Knowledge sources: where the wake phase "researches" a new skill.

A source takes a :class:`SkillRequest` and returns candidate examples plus
provenance.  Examples are ``(x, y)`` pairs over the model's symbol alphabet.
They are *candidates*: nothing is trained on until the verifiers in
:mod:`recursionx.research.verify` have accepted it.

Implemented here:

* :class:`OracleSource` – wraps a synthetic ``SkillTask`` (the benchmark setting).
* :class:`ProgramSource` – "research by writing a program": the skill is
  specified as a Python callable (e.g. produced by a code-writing LLM, or
  written by a human), and examples are produced by executing it on sampled
  inputs.  Execution-grounded data is the safest kind to train on.
* :class:`TeacherLLMSource` – asks a text-generating teacher (any callable
  ``prompt -> str``) for examples in JSON.  The Anthropic/OpenAI/local backend
  is intentionally *not* hard-wired; see ``docs/roadmap/04_research_loop.md``.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

Example = Tuple[List[int], List[int]]


@dataclass
class SkillRequest:
    name: str
    description: str = ""
    task_token: Optional[int] = None
    n_symbols: int = 16
    min_len: int = 3
    max_len: int = 8
    n_examples: int = 256
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ResearchResult:
    examples: List[Example]
    source: str
    provenance: Dict[str, object] = field(default_factory=dict)
    documents: List[str] = field(default_factory=list)


class KnowledgeSource(Protocol):
    name: str

    def research(self, request: SkillRequest, rng: random.Random) -> ResearchResult: ...


def _random_inputs(req: SkillRequest, rng: random.Random, n: int) -> List[List[int]]:
    return [[rng.randrange(req.n_symbols) for _ in range(rng.randint(req.min_len, req.max_len))]
            for _ in range(n)]


class OracleSource:
    """Ground-truth generator (benchmark / unit-test setting)."""

    name = "oracle"

    def __init__(self, task):
        self.task = task

    def research(self, request: SkillRequest, rng: random.Random) -> ResearchResult:
        ex = [self.task.sample_io(rng) for _ in range(request.n_examples)]
        return ResearchResult([(list(x), list(y)) for x, y in ex], self.name,
                              {"task": self.task.name})


class ProgramSource:
    """Examples produced by executing a program that implements the skill."""

    name = "program"

    def __init__(self, fn: Callable[[List[int], int], List[int]], code: str = ""):
        self.fn, self.code = fn, code

    def research(self, request: SkillRequest, rng: random.Random) -> ResearchResult:
        ex = []
        for x in _random_inputs(request, rng, request.n_examples):
            try:
                y = list(self.fn(list(x), request.n_symbols))
            except Exception as e:  # a broken program yields no data, not bad data
                continue
            ex.append((x, y))
        return ResearchResult(ex, self.name, {"code": self.code or getattr(self.fn, "__name__", "")})


class TeacherLLMSource:
    """Ask a teacher model for input/output examples.

    ``generate`` is any callable ``prompt -> completion text``.  The completion
    must contain a JSON list of ``{"input": [...], "output": [...]}`` objects.
    Teacher data is *unverified*: pair this source with an execution or
    consistency verifier before training on it."""

    name = "teacher_llm"

    PROMPT = (
        "You are generating training data for a small model.\n"
        "Skill: {name}\nDescription: {description}\n"
        "Symbols are integers in [0, {n_symbols}). Inputs have length {min_len}-{max_len}.\n"
        "Return ONLY a JSON list of {n} objects of the form "
        '{{"input": [ints], "output": [ints]}}.'
    )

    def __init__(self, generate: Callable[[str], str], batch: int = 32):
        self.generate, self.batch = generate, batch

    def research(self, request: SkillRequest, rng: random.Random) -> ResearchResult:
        prompt = self.PROMPT.format(name=request.name, description=request.description,
                                    n_symbols=request.n_symbols, min_len=request.min_len,
                                    max_len=request.max_len, n=self.batch)
        text = self.generate(prompt)
        return ResearchResult(parse_examples(text, request.n_symbols), self.name,
                              {"prompt": prompt}, documents=[text])


def parse_examples(text: str, n_symbols: int) -> List[Example]:
    """Extract ``[{"input": [...], "output": [...]}, ...]`` from free text."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        try:
            x, y = [int(v) for v in it["input"]], [int(v) for v in it["output"]]
        except (KeyError, TypeError, ValueError):
            continue
        if x and all(0 <= v < n_symbols for v in x + y):
            out.append((x, y))
    return out
