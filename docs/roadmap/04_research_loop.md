# 04 · Research loop: from "task oracle" to real knowledge acquisition

## Why
In the benchmark, "research" means sampling from a ground-truth generator.
A real system has to find, generate and **verify** its own training data
before anything touches weights.  That makes this the safety-critical part
of live learning.

## Built (tested)
`recursionx/research/`:
- `SkillRequest`: name, description, instruction token, alphabet and length
  constraints.
- Sources:
  - `OracleSource(task)`: benchmark setting.
  - `ProgramSource(fn, code)`: examples come from **executing a program**.
    This is the intended path for code, maths and anything checkable.
  - `TeacherLLMSource(generate)`: any `prompt -> str` callable returning
    JSON examples, parsed by `parse_examples` (robust to junk).  **Its data
    is unverified.**
- Verifiers, chained by `run_verifiers`:
  - `ExecutionVerifier(fn)`: recomputes outputs with a trusted program.
  - `ConsistencyVerifier`: the same input with different outputs is a
    conflict.  All conflicting examples are rejected, and a *syndrome* (the
    input, the competing outputs, the sources) is reported.
  - `AgreementVerifier(k)`: keeps examples at least `k` independent
    sources agree on.
- `ResearchLoop(sources, verifiers, min_examples, max_rounds, probe_frac)`
  gathers → verifies → splits into training pool / held-out **probes** /
  episodic buffer.  It returns a `SkillRecord` whose task is an
  `EpisodeTask` (it samples only verified examples) plus `provenance`.
- `DualHemisphereBrain.research_and_ingest(request, loop)` runs the loop,
  gates on the held-out probes, then goes through wake → gate as usual.
  Probes also feed the sleep audit.

## Placeholder: LLM teacher backends (not implemented here)
Implement `recursionx/research/teachers.py` with callables to pass to
`TeacherLLMSource`:

1. `AnthropicTeacher(model: str, max_tokens: int)`.  Use the official
   `anthropic` Python SDK (Messages API) and read the API key from the
   environment.  Before writing it, check the current model IDs and SDK
   usage in Anthropic's docs; the Claude Code `claude-api` skill can do this.
   Return the text of the first content block.  Add retries with backoff and
   a response cache keyed by prompt hash, so experiments are reproducible
   and cheap.
2. `LocalTeacher(endpoint)` for an OpenAI-compatible local server (llama.cpp,
   vLLM, Ollama), using plain `requests`.
3. **Program-writing teacher.**  Ask the LLM for a *Python function* that
   implements the skill, run it in a sandbox (subprocess with timeout, no
   network, restricted builtins), and feed it to `ProgramSource`.  Cross-check
   against `TeacherLLMSource` examples with `ConsistencyVerifier` and
   `AgreementVerifier(k=2)`.  Disagreement between "the program" and "the
   examples" is exactly the syndrome the verifier reports.
4. Tests must not hit the network: mock the callables.  Add one opt-in
   integration test gated by an env var (`RX_LIVE_TEACHER=1`).

## Further tasks
- **Documents → facts.**  A `DocumentSource` that turns retrieved text into
  fact examples, for `kind="fact"` records that go to the Engram wake path.
  It needs a text-capable base (see 05).
- **Trust policy** (the GPT "non-Newtonian learning rule"): add
  `LifecycleConfig.trust` weights per source type (program > agreement >
  single teacher).  Scale wake steps or learning rate by trust, and block
  consolidation of skills whose verified fraction is below a threshold.
- **Ledger.**  Persist `provenance` and the audit reports per skill as JSONL
  (`runs/ledger.jsonl`), so every consolidated skill can be traced to its
  sources, verifier reports and the sleep that baked it.

## Acceptance criteria
- A teacher backend with mocked tests.  A demo script
  (`experiments/research_demo.py`) learns one skill end to end from a
  program-writing teacher, with the ledger written.
- Documented behaviour when teacher data is wrong: a test injects 30%
  corrupted examples and shows the verifiers reject them and the gate still
  passes.
