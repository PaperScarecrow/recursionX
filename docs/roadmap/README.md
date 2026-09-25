# Roadmap and QA work orders

Each file below is a self-contained work order for a Claude Code instance, or
a human.  Each one lists what exists (built and tested), what is missing,
concrete tasks, and **acceptance criteria**.  Work them roughly in order;
`00` is always first.

| # | file | status |
|---|---|---|
| 00 | [QA checklist](00_QA_CHECKLIST.md) | do first: verify the repo end to end |
| 01 | [Multi-seed + long skill streams](01_multiseed_long_stream.md) | framework **built** (`run_matrix.py`, `--stream long`); runs pending |
| 02 | [Expert growth + router-gated adapters](02_growth_and_routed_experts.md) | growth **built + tested**, not benchmarked; routed-expert baking **placeholder** |
| 03 | [HyperLoRA (skill → adapter generator)](03_hyperlora.md) | framework **built + tested**; warm start 0.971 vs 0.786 default (blind control 0.928), 2 seeds |
| 04 | [Research loop](04_research_loop.md) | framework **built + tested** (sources, verifiers, loop, brain hook); LLM teacher backend **placeholder** |
| 05 | [Scale-up: GPU, real text](05_scale_up.md) | text pipeline, presets and GPU training script **built**; KV cache, fused MoE and FSDP **placeholders** |
| 06 | [Revision lane (facts that change)](06_revision_lane.md) | **placeholder**: design + tasks |
| 07 | [Titans memory + test-time depth](07_titans_and_depth.md) | inconclusive results so far; tuning tasks |

## How to hand these to Claude Code

1. Open the repo and let it read `CLAUDE.md`.
2. Prompt: *"Work through `docs/roadmap/00_QA_CHECKLIST.md`.  Report every
   failure with the command and output.  Do not change behaviour to make a
   check pass without explaining why."*
3. Then one work order at a time: *"Implement
   `docs/roadmap/0X_....md`.  Meet every acceptance criterion, add tests, and
   update the work order's status section and the README results when
   done."*

## Ground rules for every work order

- `pytest -q` stays green.  New features come with tests.
- No result goes into `README.md` without a JSON file under `results/` and
  the command that produced it.
- Negative or inconclusive results are recorded as such.
- Comparisons are fair: same learning budget, same episodic buffer, same base
  checkpoint, several seeds (at least 3 for any headline claim).
