# 02 · Expert growth and router-gated adapters ("context-conditional capacity")

## Why
The main finding so far: skills that read *the same inputs* but compute
different functions cannot be separated by input-space projection.  They
need capacity that is conditional on context (the instruction).  Recursion-X
has two such mechanisms: the awake-time skill router, and MoE expert growth.
Growth is built but not benchmarked.

## Built (tested)
- `FluidMoE.grow_expert(src, router_direction, bias, noise)` clones an expert
  and adds a router row.
- `SleepConsolidator._grow`: after REM, if a new skill's accuracy is below
  `growth_acc_threshold`, one expert is added per MoE layer.  Its router row
  points along `mean_hidden(new skill) − mean_hidden(old skills)`, followed by
  `growth_extra_steps` of REM that also train the routers.
- Test: `tests/test_frameworks.py::test_sleep_expert_growth_adds_capacity`.
- Method `rx_grow` in `continual.py` (`grow_experts=True`).

## Tasks
1. **Benchmark growth**: `python run_matrix.py --methods rx,rx_grow --seeds 0,1,2`.
   Also try `growth_acc_threshold=1.01`, which always grows, to measure the
   capacity effect alone.  Report the final number of experts and the
   parameter overhead.
2. **Router-row initialisation study.**  The current direction heuristic
   (difference of means × 4 × mean row norm) is untested.  Alternatives:
   (a) a logistic-regression probe separating new-skill tokens from old
   ones, (b) copying the source row plus a learned bias, (c) zero init with
   a large negative bias that REM learns.  Measure how much traffic the new
   expert takes on old skills.  Target: ≈0.
3. **Router-gated adapters → expert baking** (placeholder, design below).
   - Add `RoutedAdapter`: a LoRA whose contribution is multiplied by a gate
     `g(x) = σ(w·x + b)` computed from the residual stream.  Train it during
     wake together with the LoRA, with a loss that pushes `g→0` on
     old-skill replay tokens (binary cross-entropy with labels new/old).
   - During sleep, bake it as a **new expert**: expert weights = source expert
     + the adapter delta restricted to that expert's matrices, and router row
     = `w`, bias = `b`.  For adapters on attention or liquid layers, keep
     the gated adapter resident, or distil it into the new expert with REM.
   - Acceptance: consolidating a routed adapter into an expert changes
     old-skill probe accuracy by ≤ 0.02 *before any REM*.  That is the
     property plain merging lacked (it gave 0.99 drops).
4. **Expert pruning / merging** (keeps growth bounded): experts with
   near-zero traffic for N cycles are removed, and pairs with highly
   similar outputs on probes are merged.  Add both and a test.

## Acceptance criteria
- 3-seed table `rx` vs `rx_grow` in `results/`, with expert counts.
- `RoutedAdapter` implemented with tests, including the ≤ 0.02 pre-REM drop
  criterion on the synthetic suite, or a documented negative result.
