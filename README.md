# Recursion-X

**A looped, liquid-hybrid, mixture-of-experts transformer that learns new
skills while awake and bakes them into its weights while one of its two
hemispheres sleeps.**

Recursion-X is a research prototype of a continually learning language-model
architecture.  It combines:

* **a Liquid-style hybrid backbone.**  Gated short convolutions with a liquid
  time-constant recurrence are interleaved with GQA attention (LFM2, LTC/CfC).
* **recursive / looped depth.**  A weight-tied core is iterated `R` times with
  input re-injection (Huginn, Ouro, universal transformers).  More loops can
  be spent at test time, and inference can stop early once the state
  converges.
* **Titans / MIRAS neural long-term memory.**  A matrix memory trained *at test
  time*, one gradient step per chunk, with momentum ("surprise") and a
  forget gate.  The attentional bias is configurable (`l2 | huber | l1`).
* **Engram-style hashed n-gram lookup tables.**  They depend only on token ids,
  so they can live in RAM or on disk (`numpy.memmap`) and be updated one row at
  a time.
* **a fluid mixture of experts.**  Shared + routed SwiGLU experts that can
  **grow** new experts during sleep and be **offloaded** to disk with LRU
  paging.
* **projected-LoRA skills.**  Each new skill is a LoRA adapter whose input is
  projected away from the subspace that consolidated knowledge occupies.  The
  adapter's `A` matrix is initialised from the new skill's own activation PCA,
  so the data is baked into the adapter before training starts.
* **dual-hemisphere wake/sleep consolidation**, as in dolphins, which sleep with
  one hemisphere at a time.  The *awake* hemisphere serves requests and learns
  adapters.  When enough skills pass the quality gate, the *sleeping*
  hemisphere syncs to the awake base and distils the skills into its own base
  weights: multi-teacher distillation with episodic replay and self-generated
  "dreams", gradient projection, then subspace protection.  The hemispheres
  then swap and the old one is reset.  Consolidation can run in a background
  thread while the awake side keeps serving.

See **[docs/DESIGN.md](docs/DESIGN.md)** for the full design and for where
each idea lives in the code.

```
recursionx/
  model.py                 prelude → looped core (weight-tied) → coda
  modules/liquid.py        LFM2-style gated conv + liquid time-constant scan
  modules/attention.py     GQA + RoPE + QK-norm
  modules/moe.py           Fluid MoE: shared/routed experts, growth, disk offload
  modules/neural_memory.py Titans/MIRAS test-time-trained memory
  modules/engram.py        hashed n-gram tables (memory or disk), sparse updates
  modules/lora.py          AdaptableLinear, projected LoRA, GPM subspaces
  lifecycle/wake.py        skill → projected LoRA, fact → Engram rows
  lifecycle/gate.py        "good enough to bake?"
  lifecycle/sleep.py       merge / REM distillation / dreams / growth / protect
  lifecycle/brain.py       DualHemisphereBrain: serve, learn, sleep, swap
  lifecycle/skills.py      skill records + learned skill router
experiments/               pretraining, continual benchmark, ablations
results/                   JSON results, plots, logs from the runs below
tests/                     unit + lifecycle tests
```

## Quick start

```bash
pip install -e .[dev]
pytest -q                                   # 18 tests, ~2 min on CPU

cd experiments
python pretrain_base.py --steps 3000        # base model on 6 base skills (~17 min CPU)
python continual.py --methods finetune,finetune_replay,lora_merge,rx,rx_unprojected
python report.py --plot
```

```python
from recursionx import RXConfig, RecursionX, LifecycleConfig, DualHemisphereBrain, SkillRecord

brain = DualHemisphereBrain(RecursionX(RXConfig()), LifecycleConfig())
brain.register_base_skills([...])            # what the pre-trained base knows (+ protect it)
brain.ingest(SkillRecord("new_skill", task), val_set)   # wake: learn, gate, serve via router
brain.sleep(background=True)                 # consolidate on the other hemisphere
logits = brain.logits(tokens)                # keeps serving meanwhile
brain.wake_up()                              # swap hemispheres
```

## Experiments

Everything here was run on a 4-core CPU with no GPU.  The model is therefore
tiny: 1.9 M backbone parameters plus a 0.5 M-parameter Engram table.  The
tasks are a synthetic *skill suite*: sequences are
`[BOS, TASK_k, x…, SEP, f_k(x)…, EOS]` over 16 symbols.  The task token is the
"instruction", and exact match counts every output token.  This is a
controlled test of **continual skill acquisition and forgetting**.  It is not
a claim about language-model quality at scale.

### Setup

* **Base model:** pre-trained 3,000 steps on six base skills (`copy, reverse,
  succ, sort, max, interleave`).  It reaches 0.98–1.00 exact match on each.
* **Continual phase:** six new skills arrive one at a time (`rotl, pred,
  swap_pairs, sort_desc, add_first, first_last`).  Every method gets the same
  600-step × 64-example learning budget per skill.  After each skill, all 12
  skills are evaluated through whatever the method would serve.
* **Recursion-X:** projected-LoRA wake learning, then the gate, then a sleep
  after every 2 accepted skills.  Sleep is 600 REM steps: multi-teacher
  distillation into the intact base, 16 rehearsal examples per old skill per
  step (half stored episodes, half dreams), and no gradient projection.  The
  episodic buffer is 64 examples per skill, the same buffer the replay
  baseline gets.

### Main result: 6 new skills learned sequentially

| method | seeds | avg_all | avg_base | avg_new | learn_acc | base_forgetting | bwt_new | CPU min |
|---|---|---|---|---|---|---|---|---|
| fine-tune (sequential) | 1 | 0.083 | 0.000 | 0.167 | 0.835 | 0.997 | −0.802 | 48 |
| LoRA per skill, merged at once | 1 | 0.083 | 0.000 | 0.167 | 0.854 | 0.997 | −0.825 | 29 |
| fine-tune + 50 % replay | 2 | 0.857 ± 0.011 | 0.938 ± 0.007 | 0.775 ± 0.015 | 0.881 ± 0.076 | 0.059 ± 0.007 | −0.127 ± 0.073 | 28 |
| Recursion-X + GPM in sleep | 1 | 0.870 | 0.941 | 0.799 | 0.983 | 0.055 | −0.220 | 91 |
| Recursion-X, plain LoRA | 2 | 0.906 ± 0.016 | 0.965 ± 0.010 | 0.846 ± 0.041 | 0.901 ± 0.070 | 0.031 ± 0.010 | −0.066 ± 0.035 | 81 |
| **Recursion-X** (projected LoRA, no GPM in sleep) | 2 | **0.922 ± 0.021** | 0.957 ± 0.003 | **0.886 ± 0.046** | 0.959 ± 0.035 | 0.039 ± 0.003 | −0.087 ± 0.013 | 90 |

(`results/continual/`; ± is the spread over seeds.  Raw per-step accuracy
matrices are in the JSON files.  The Recursion-X rows were run before the
sleep audit below existed, so every sleep was committed.)

*avg_all*: final mean accuracy over all 12 skills.  *learn_acc*: accuracy on
each new skill right after learning it.  *base_forgetting*: mean drop on the 6
pre-trained skills.  *bwt_new*: backward transfer on new skills (final minus
just-learned).

![continual](results/continual/continual.png)

Takeaways:

1. **Naive fine-tuning and merging LoRAs as you go are catastrophic.**  They
   end knowing only the last skill (0.08 average; base skills 0.00).
2. **The wake/sleep lifecycle works.**  While awake, Recursion-X has **zero
   interference**: new skills are served through their adapters by a learned
   router, and base skills stay at 0.98–1.00.  After each sleep the skills
   live in the base weights with no adapters left.  Recursion-X ends at
   **0.922** average accuracy against 0.857 for fine-tuning with replay, the
   standard strong baseline.  Both use the same 64-example episodic buffer
   per skill.  Recursion-X also forgets less of the base (0.039 vs 0.059).
   It uses about 3× the compute of the replay baseline, mostly in sleep.
3. **How to consolidate matters more than anything else.**  See the
   sleep-variant study below.
4. **Projection helps learning but hurts consolidation.**  Projected,
   data-initialised adapters learned new skills more reliably: 0.959 vs 0.901
   learn accuracy.  On one seed a plain LoRA failed to learn `add_first` at
   all (0.01) while the projected one reached 0.97.  But GPM gradient
   projection *during sleep* cost 0.05 in final accuracy.  See below.
5. **Remaining weakness: late skills.**  The last consolidated skills
   (`sort_desc`, `add_first`) are the least stable.  On seed 1 only 0.48 of
   `add_first`'s adapter accuracy (0.68) survived the final sleep.  That
   motivated the sleep audit below.

### Sleep-variant study (what makes consolidation work)

The same two freshly learned adapters (`rotl`, `pred`) were consolidated in
different ways (`experiments/sleep_variants.py`,
`results/sleep_variants/results.json`):

| sleep variant | worst old skill | rotl | pred |
|---|---|---|---|
| merge adapters only (no REM) | 0.00 | 0.00 | 0.00 |
| merge → REM 300 | 0.40 (interleave) | 0.89 | 1.00 |
| no merge → REM 300 | 0.93 | 0.41 | 1.00 |
| merge → REM 300, per-skill balanced rehearsal | 0.67 | 0.92 | 1.00 |
| no merge → REM 300, balanced | 0.93 | 0.64 | 1.00 |
| **no merge → REM 600, balanced** (default) | **0.93** | **1.00** | **1.00** |

(Awake hemisphere before sleep: old skills 0.98–1.00, rotl 0.68, pred 1.00.)

* **Merging even *projected* adapters directly into the base is destructive.**
  Two merged adapters destroyed each other *and* the base.  The merge-preview
  gate shows why: a single always-on adapter, which is mathematically the same
  as merging it, drops old skills by ~0.99.
* **Distilling into the intact base works best.**  Start from the pre-sleep
  base, then learn from each skill's adapter teacher plus balanced
  rehearsal (stored episodes + dreams).  The student even beat its teacher
  (rotl 0.68 → 1.00), because REM also sees ground-truth episodes.

### Where projection helps and where it hurts

Projected LoRA constrains the adapter to input directions that consolidated
knowledge does not use.  In this suite, a new skill is *a different function
of the same inputs*: rotl reads the same symbols and positions as copy.  The
only thing that separates skills is the instruction context.  Measured:

* At the default GPM threshold (0.97), only ~6–18 of 128 input dimensions per
  layer are protected, since a few dominant directions carry most of the
  energy.  The adapter's response to *old* inputs is as large as its response
  to new ones, so the projection protects little.
* Raising the threshold to 0.9999 protects 59% of the space and keeps old
  skills perfectly intact.  But the adapter can then no longer learn rotl
  (0.03).  This is the stability–plasticity trade-off in its purest form.
* Gradient projection during REM protects more as skills pile up.  By the
  third sleep it blocked `add_first` from consolidating (0.36).  With the same
  projected adapters but no GPM in sleep, it reached 0.96.
* As a *starting point* for a skill, though, projection plus data-projected
  initialisation beat plain LoRA.  It gave 0.959 vs 0.901 accuracy right
  after learning, with fewer failures to learn at all.

Separation between skills has to come from **context-conditional capacity**,
not input-space orthogonality.  In Recursion-X that means (a) the router,
awake, and (b) distillation with rehearsal, asleep.  The architectural version
of the same idea is **expert growth**: fresh experts the router sends only the
new skill's tokens to.  Growth is implemented (`grow_experts=True`) but not yet
benchmarked.  At scale, where layers are thousands of dimensions wide and
skills really do occupy different subspaces, projection may behave very
differently.  That has to be tested there.

### Sleep as a transaction (audit + rollback)

Because consolidation happens on the *other* hemisphere, a bad sleep can be
undone for free.  Every skill now keeps held-out **retention probes** that are
never trained on.  Before the hemispheres swap, both the awake snapshot (as
served, with adapters routed in) and the consolidated student run those
probes.  The sleep is **committed** only if no protected skill drops more than
`commit_max_drop` (0.15) and every new skill clears the gate.  Otherwise it is
**rolled back**: the student is discarded, the pending skills stay served by
their adapters, and they are retried at the next sleep.  The audit report
lists which skill regressed (`brain.history`).  This follows the
"update-as-transaction" idea: forgetting becomes a detected, attributable and
reversible event instead of a silent one.

**Measured (seed 1 rerun, `results/continual/rx_audit_s1.json`):**
* The first two sleeps passed the audit and were committed.
* The third sleep (`add_first`, `first_last`) was **rolled back**, with the
  syndrome `sort_desc: regressed 0.95 -> 0.72`.  The awake hemisphere kept
  serving both new skills from their adapters.
* Final served accuracy was **0.953** against 0.900 for the identical run
  without the audit.  `sort_desc` stayed at 0.96 instead of 0.65, and
  `add_first` at 0.68 instead of 0.48.
* The trade-off: two skills remain unconsolidated (still adapters) until a
  later sleep succeeds.

### Architecture ablation (base skills, trained from scratch, 1,000 steps)

| variant | mean exact match @1k steps |
|---|---|
| full Recursion-X | 0.701 |
| no Engram | 0.905 |
| no Titans memory | 0.962 |
| attention only (no liquid mixer) | 0.641 |
| no looping (R = 1) | 0.597 |

(`results/architecture/ablation_s0.json`, single seed.)  **Looping** (+0.10)
and the **liquid mixer** (+0.06) speed up learning.  **Engram** and the
**Titans memory** *slow* it on this benchmark, which is expected: inputs are
random symbol strings, so there are no recurring n-grams to look up and no
long-range structure to memorise.  Both components exist for real text and
long contexts, and this suite cannot show their value.

Two more architecture tests (`experiments/architecture.py`, 3,000 steps,
`results/architecture/`) did not produce positive results:

* **Test-time depth.**  The model was trained with 1–4 sampled loops and
  evaluated at R = 1…8.  Accuracy was flat (0.980 at R = 1, 0.984 at R = 2–8).
  The base skills are too easy to need extra depth once trained.  The one
  positive: running *beyond* the trained depth (R = 5–8) does not degrade
  anything, so extra test-time loops are safe.  Showing depth *scaling*
  needs harder, compositional tasks.
* **Titans memory on associative recall** (liquid-only backbone, 8 key–value
  pairs).  Accuracy was 0.264 without the memory and 0.270 with it (chance is
  0.125).  Neither model learned the task at this scale and budget, so this is
  inconclusive.  The memory still needs tuning (write-rate init, chunk size,
  deep MLP memory) before it can be judged.

## Status and roadmap

What exists and is tested:

* every architectural component, including causality tests, scan
  correctness, disk offload of experts and Engram rows, and Titans state
  carry-over across segments;
* the full wake → gate → sleep → audit → swap/rollback lifecycle, including
  background sleep while serving;
* a continual-learning benchmark with baselines, ablations and results.

Next steps, roughly in order of expected value:

1. **Multiple seeds and longer skill streams**, with more sleep cycles, to
   measure how rehearsal cost and forgetting scale over many cycles.
2. **Benchmark expert growth** as the structural answer to "same inputs,
   different function".  Also try routing-conditioned adapters: a LoRA
   gated by the router, which can then be baked as a new expert rather than a
   dense merge.
3. **Hypernetwork "projector"** that generates a LoRA directly from a
   skill's documentation or examples (Text-to-LoRA style).  That would let the
   wake phase produce a first adapter in one forward pass, then refine it.
4. **Real research loop**: replace the task oracle with a tool-using agent
   that searches, reads and writes its own training episodes, and let the
   gate verify skills against held-out checks.
5. **Scale up**: a real tokenizer and text data, a GPU, a ~100 M → 1 B →
   4 B-parameter progression, KV cache / incremental decoding, deep MLP
   Titans memory, and a large disk-resident Engram store.
