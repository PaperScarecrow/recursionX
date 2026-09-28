# 00 · QA checklist: verify the repository end to end

Run each step and record *pass/fail + evidence* (command, relevant output).
Timings are for a 4-core CPU; a GPU box will be faster for training steps.

## A. Install and unit tests
- [ ] `pip install -e .[dev]` succeeds on Python ≥ 3.10 with torch ≥ 2.4.
- [ ] `pytest -q` passes: 27 tests at the time of writing, about 3–5 minutes on CPU.
- [ ] `pytest -q tests/test_modules.py -k causal` passes.  Causality is the
      most important invariant.

## B. Reproduce the headline benchmark (≈ 3 h CPU)
- [ ] `cd experiments && python pretrain_base.py --steps 3000`.  The final
      eval line shows ≥ 0.95 on all six base skills (`runs/pretrain.log`).
- [ ] `python run_matrix.py --methods finetune_replay,rx --seeds 0 --parallel 2`.
- [ ] `python report.py`.  The `rx` avg_all is within ±0.05 of 0.943 (seed 0)
      and `finetune_replay` is within ±0.05 of 0.868.  CPU nondeterminism
      makes exact equality unlikely.  Record the deviation.

## C. Lifecycle behaviours (inspect logs / JSON)
- [ ] During wake, base-skill accuracy is unchanged (≥ 0.97) after each new
      skill.  Check the `matrix` rows before the first sleep.
- [ ] Each `sleep` history entry has `committed: true/false` and an `audit`
      block with per-skill `before/after` probe accuracies.
- [ ] `rx_audit` on seed 1 shows one rolled-back sleep with a syndrome
      naming the regressed skill (`results/continual/rx_audit_s1.json`,
      `history`).
- [ ] `DualHemisphereBrain.sleep(background=True)` keeps serving.  Covered by
      `tests/test_lifecycle.py::test_background_sleep_keeps_serving`.  Also try
      it manually with a larger model and time how long `brain.logits` takes
      during sleep.

## D. Frameworks
- [ ] `python pretrain_text.py prepare --data ../README.md ../docs --out ../runs/text/c.bin`
      and `python pretrain_text.py train --bin ../runs/text/c.bin --preset tiny --device cpu --steps 50 --batch 4 --seq-len 128 --eval-every 50`
      both run.  Loss decreases and a checkpoint is written.  `--resume`
      continues from it.
- [ ] On a GPU: the same with `--device cuda --dtype bf16 --preset small`.
      There are no dtype errors, and tok/s is reported.  Try `--compile`;
      if it fails, record the error and don't fix it by removing features.
- [ ] `python -c "from recursionx.presets import describe; [print(describe(n)) for n in ['tiny','small','base','1b','4b']]"`
      prints parameter counts.  Check that they match the table in
      `05_scale_up.md`.
- [ ] `python hyperlora.py --meta-steps 300 --wake-steps 50` runs to completion.

## E. Code review focus areas
- `recursionx/lifecycle/sleep.py::_rem`: check that the loss weighting across
  sources (new skills vs rehearsal) is what the docs claim.
- `recursionx/lifecycle/skills.py::SkillRouter`: the classifier is refit
  whenever a skill registers.  Check for stale fits after `wake_up`
  (prototypes are rebuilt there).
- `recursionx/modules/engram.py::DiskTable`: the gradients of rows gathered
  twice in one forward pass must accumulate correctly.  Unique indices are
  gathered once, so they should.
- Thread safety of background sleep: the sleeping thread uses its own
  snapshot, but `SkillRouter` is shared.  Check that nothing mutates it
  from the sleep thread.

## Report format
A markdown table: `step | pass/fail | evidence | notes`.  Put follow-up bugs
as a checklist at the bottom.
