# 03 · HyperLoRA: generate an adapter from demonstrations or a description

## Why
Wake learning currently needs hundreds of gradient steps per skill.  A
hypernetwork that maps "what the skill is" to an adapter in one forward pass
(Text-to-LoRA / Doc-to-LoRA style) would give an instant first draft.
Gradient steps then refine it ("project, then refine").

## Built (tested)
- `recursionx/skills/hyperlora.py`:
  - `demo_features(model, seqs)`: the skill descriptor.  It is the mean final
    hidden state over all tokens concatenated with the mean over answer
    tokens, averaged over K demos.
  - `HyperLoRA(model, desc_dim, rank, alpha, hidden)`: modules are grouped
    by shape, with one head per shape and a learned module embedding.  It
    targets all `AdaptableLinear` except routed experts.  At the tiny preset
    it has about 4.6M parameters.
  - `apply` / `clear` install functional adapters
    (`AdaptableLinear.generated`), and gradients flow into the hypernetwork.
  - `materialize` turns a generated adapter into a trainable `LoRAAdapter`,
    reproducing the function exactly.
  - `meta_train(model, hyper, tasks, steps, ...)` meta-trains across a
    family of skills with the base frozen.
- `WakeLearner.learn_skill(..., init_lora=...)` and
  `DualHemisphereBrain.ingest(..., init_lora=...)` start wake learning from
  a generated adapter.
- `experiments/hyperlora.py`: meta-trains on 14 skills (6 base + 8 extra)
  and tests on the 6 held-out continual skills, both zero-shot and as a warm
  start, with a *blind* control (one generated adapter from the mean
  descriptor, used for every skill).

## Results so far (2 seeds, `results/hyperlora/`)
| wake start (150 steps) | mean acc on held-out skills |
|---|---|
| default init | 0.786 |
| blind generated init | 0.928 |
| skill-specific generated init | 0.971 |

- Zero-shot is ~0 on held-out skills, and base skills are unaffected.
- Most of the gain is a *meta-learned initialisation* (the blind control).
  Skill conditioning adds +0.04, mostly on `add_first`.  Two seeds are not
  enough to call that increment significant: run ≥ 5.
- v1 exploded because the generator produced both LoRA factors from a
  non-zero init.  The fix was zero-initialising the B part of every head and
  using LR 3e-4.  The failed run is kept as `hyperlora_v1_unstable_s0.json`.

## Known limitations
- A family of 14 algorithmic skills is far too small to expect zero-shot
  generalisation to new functions.  The first experiment mainly checks that
  the machinery works and whether warm starts help at all.
- The descriptor comes from demos only.  A *text description* encoder
  (Text-to-LoRA proper) needs a base model that understands text, so it
  belongs after 05.

## Tasks
1. **Bigger skill family.**  Generate a procedural family of hundreds of
   skills (compositions of primitives such as `rev∘succ` or
   `sort∘add_first`) in `recursionx/data/tasks.py`.  Hold out whole
   compositions, not just instances.
2. **Reconstruction pre-training** (T2L recipe): collect adapters learned by
   the wake phase for many skills.  Train the hypernetwork to reconstruct
   them (MSE on ΔW = BA), then fine-tune end to end with `meta_train`.
   Compare the two.
3. **Projected output.**  Make generated A matrices respect the protected
   subspace (project `A` with each module's `U`).  Check the effect on
   interference when the generated adapter is always on.
4. **Warm-start curve.**  Accuracy vs wake steps (0, 25, 50, 100, 200) for
   default / blind / hyper init, on held-out skills, ≥ 5 seeds.  Also compare
   against a non-hypernetwork meta-learned init (e.g. Reptile on the LoRA
   parameters), since the blind control suggests that is most of the effect.
5. **Brain integration.**  Add `LifecycleConfig.hyper_init: bool`.  When a
   HyperLoRA is attached to the brain, `ingest` generates `init_lora`
   automatically from the record's episodes.  After each sleep, add the
   consolidated skills to the hypernetwork's meta-training set and run a few
   meta-steps: the generator improves with every skill the system learns.

## Acceptance criteria
- A warm-start curve plot with 3 seeds in `results/hyperlora/`.
- A clear statement, backed by data, of whether hyper-init reduces the wake
  steps needed to pass the gate.
- Tests for any new code.  `pytest -q` is green.
