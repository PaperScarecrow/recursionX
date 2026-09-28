# Recursion-X Lab — Handoff Report (2026-09-28 night)

Base: `recursionX-claude-recursion-x-transformer-tmbjqs.zip` (unpacked, untouched).
Machine: Ryzen 7600X 6C/12T · RTX 5060 8GB · 32GB RAM · torch 2.11+cu128.
Env: `~/htpcvenv` (+pytest, additive only). Lab: `~/recursionX-lab` (this repo).

## Scoreboard (seed 0, shared 4k-step GPU base unless noted)

| method | avg_all | note |
|---|---|---|
| finetune_replay | 0.775 (3 seeds ±0.000) | 2.8 min; add_first 0.07 |
| replay-3x (compute-matched) | 0.839 | 8.3 min; still −0.14 |
| rx (no audit) | 0.813 | bad sleep-2 committed |
| rx_audit | 0.932 s0 · 0.955±0.017 (3-seed) | rollbacks do the work |
| **rx_audit + hybrid router** | **0.984 / 0.983 (2-seed)** | +0.05 serving-only win |
| pressure-1 + REM1200 | 0.972 | 3 commits, then wall |

GPU ≈10x vs orig CPU (pretrain 100s vs 1019s; replay 170s vs 28min; RX 11–17min vs 90min).

## Breakthroughs (measured, JSON-backed)

1. **Hybrid token+neural router** (`recursionx/lifecycle/brain.py`): unique
   instruction token → adapter, ambiguous/shared → ridge fallback. Verified:
   pred 0.57→1.00, swap 0.45→1.00, sort_desc 0.73→0.98. Audit measures hybrid-served.
2. **Consolidation wall is structural, not cost**: sort_desc single-sleep fails at
   600 AND 1200 REM with identical syndrome; growth helps the student, old skills
   still regress. Bake-as-expert or adapters-forever remain.
3. **Protection innocent**: add_first on 3-commit base fails projected (0.000) AND
   plain (0.004). Repeated REM eats plasticity itself.
4. **Audit load-bearing**: 0.813→0.932 same students; 3-seed 0.955 vs 0.775 replay.

## Bugs fixed (lab only)

- Router CUDA device crash (`skills.py`); `--device` plumbing for experiments.
- `sleep_variants.VARIANTS` never set `nrem_merge=True` (merge rows ran nomerge);
  `rem300` rows run 600 steps (misnamed defaults).
- `--seed` never reaches Brain/samplers (`random.Random(0)` hardcoded) —
  replay s0/s1/s2 bit-identical. Threading the seed is queued work.

## Reproduce

```bash
/home/paperscarecrow/htpcvenv/bin/python -m pytest tests/ -q
cd experiments
/home/paperscarecrow/htpcvenv/bin/python pretrain_base.py --steps 3000 --threads 6 --device cuda
/home/paperscarecrow/htpcvenv/bin/python continual.py --methods rx_audit --seed 0 --threads 3 --device cuda
```

## Files that matter

- `runs/continual*/*.json` + `summary.md` + plots — all measured runs
- `runs/router_probe/` — ceiling, hybrid-verify, addfirst-wall probes
- `runs/sleep_variants/` + `logs/sv_*.log` — fixed merge/nomerge comparison
- `logs/LAB_LOG.md` — chronological lab journal
- `docs/NOTES_breakthrough.md` — whiteboard copy (comment on `~/recursionX_notes.md`)
- New scripts: `experiments/router_probe.py`, `tokid_probe.py`, `addfirst_probe.py`

## Queued (not run)

Proper seeded 3-seed + hybrid · shared-token pilot · 14-skill long stream ·
merge_preview-vs-merge_only norm puzzle · `--seed` threading fix.
