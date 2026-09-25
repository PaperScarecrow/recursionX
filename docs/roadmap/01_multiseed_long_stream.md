# 01 · Multiple seeds and longer skill streams

## Why
The headline comparison rests on 2 seeds and 6 new skills (3 sleep cycles).
Two questions are open: does the advantage hold, and how do forgetting,
rehearsal cost and protected-subspace growth behave over many cycles?

## Built
- `experiments/run_matrix.py`: runs a (method × seed) grid in parallel worker
  processes and then calls `report.py`.
- `continual.py --stream long`: 14 new skills
  (`common.NEW_SKILLS_LONG`, 7 sleeps).  Results go to
  `runs/continual_long/`.  Every JSON records its `base_skills` and
  `new_skills`, so `report.py` handles any stream.

## Tasks
1. **Default stream, 3+ seeds**:
   `python run_matrix.py --methods finetune_replay,rx_unprojected,rx,rx_audit --seeds 0,1,2 --skip-existing --parallel 2 --threads 2`
   (seeds 0/1 already exist for some methods).
2. **Long stream**:
   `python run_matrix.py --stream long --methods finetune_replay,rx,rx_audit --seeds 0,1,2 --parallel 2`.
   Expect about 3.5 CPU-hours per Recursion-X run; use a GPU box if possible.
3. **Per-cycle diagnostics.**  Extend `report.py` to plot, per sleep cycle:
   REM seconds, `protected_fraction`, number of dreams, audit
   commit/rollback, and max drop on old skills.  The data is already in
   each run's `history`.
4. **Rehearsal cost scaling.**  REM batch size grows linearly with the number
   of known skills (`old_batch_per_skill` × skills).  Add an option to cap
   total rehearsal per step, `max_old_batch`, sampling skills uniformly or
   by "risk" (largest recent audit drop).  Compare against the uncapped run
   on the long stream.
5. **Retry policy after rollback.**  Rolled-back skills are retried at the
   next sleep with the same settings.  Add a policy: retry with 2× REM
   steps; after 2 failures, mark the skill `adapter_resident` and stop
   retrying.  Test it.

## Acceptance criteria
- `results/continual/summary.md` reports ≥ 3 seeds for `finetune_replay`,
  `rx` and `rx_audit`, with mean ± std.
- `results/continual_long/summary.md` and plots exist for the long stream.
- README "Main result" is updated.  The statement "Recursion-X beats replay"
  is kept, qualified or removed based on the 3-seed numbers.
- The per-cycle diagnostic plot is committed.
