# 07 · Titans memory and test-time depth: getting a real signal

## Status
Both tests are **inconclusive** so far (`results/architecture/`):
- **Associative recall**, liquid-only backbone, 8 key–value pairs, 3,000
  steps: 0.264 without Titans, 0.270 with it.  Chance is 0.125, so neither
  model learned the task.
- **Depth.**  Trained with R ∈ [1, 4] and evaluated at R = 1…8, accuracy was
  flat (0.98).  The base skills are too easy to need depth.  Extra loops
  beyond the trained range did not hurt.

## Tasks: Titans
1. **Sanity-check the write path.**  `tests/test_modules.py::test_neural_memory_learns_at_test_time`
   shows that the mechanism can recall a stored pair with hand-set weights.
   Next, check whether *training* discovers it: freeze everything except
   the memory's q/k/v/gates on a 2-pair recall task and plot accuracy.  If
   that fails, the issue is optimisation or initialisation, not capacity.
2. **Initialisation.**  `hyper.bias` = 0 gives lr ≈ 0.5·max_lr and momentum
   ≈ 0.5 from the start.  Try a strongly positive lr bias, near-zero
   momentum and near-zero decay at init.  Try `mem_chunk` = 1, 2, 4.  Use
   `mem_conv=4` (already implemented), which lets a value bind to its
   preceding key.
3. **Curriculum.**  Start with 2 pairs and grow to 8 and then 32.  Make the
   context longer than the liquid state can hold: with `A_log` spanning
   short time constants, a plain liquid model forgets after ~50 tokens, so
   use contexts of 256+ tokens.
4. **Streaming test.**  Split one long sequence into segments and carry the
   memory state (`memory_state=`).  Recall a pair from segment 1 in segment
   4, which attention cannot do because each segment is a separate forward
   pass.  This is where Titans should clearly win.
5. **MIRAS comparison.**  Compare `mem_bias` = l2 / huber / l1 on recall
   with noisy values (corrupt 10% of the values).

## Tasks: test-time depth
1. **Harder, compositional skills** whose required number of sequential
   steps varies: e.g. apply `succ` k times where k is given in the prompt,
   multi-hop pointer chasing, or cumulative sums over longer inputs.
   Train with R ∈ [1, 4]; evaluate at R = 1…12, split by k.
2. **Adaptive exit.**  Measure the loops actually used with `exit_tol`
   against the difficulty k.  The hope is that more loops get spent on
   harder inputs.  Log per-sequence exits (a batch-level exit needs to
   become per-sequence).
3. **Randomised-depth schedule.**  Compare uniform [1, 4] with Huginn's
   log-normal-Poisson and with fixed R = 4, at equal compute.

## Acceptance criteria
- A recall or streaming benchmark where Titans measurably beats the no-memory
  baseline across 3 seeds, or a documented negative result after tasks 1–3.
- An accuracy-vs-R plot per difficulty level on a compositional task.
