# Recursion-X — design

Recursion-X combines five lines of recent work into one network whose aim is
to *keep learning after deployment*:

| idea | source of inspiration | where it lives |
|---|---|---|
| liquid / attention hybrid backbone | Liquid AI LFM2 (gated short-conv + GQA), liquid time-constant & CfC networks | `modules/liquid.py`, `modules/attention.py` |
| looped / recursive depth | Huginn (depth-recurrent LM), Ouro (looped LMs), universal transformers, TRM/HRM | `model.py` (prelude → looped core → coda) |
| test-time-trained long-term memory | Google Titans, MIRAS (attentional bias / retention gate), Hope / nested learning | `modules/neural_memory.py` |
| huge, offloadable lookup memory | Engram-style hashed n-gram tables, MoE expert offloading | `modules/engram.py`, `modules/moe.py` |
| "fluid" mixture of experts | DeepSeek-style shared + routed experts, plus growth at runtime | `modules/moe.py` |
| projected LoRA skills | LoRA, O-LoRA / InfLoRA, Gradient Projection Memory (GPM), PiSSA/CorDA data-aware init | `modules/lora.py`, `lifecycle/wake.py` |
| dual-hemisphere sleep | unihemispheric sleep in dolphins; complementary learning systems (hippocampus → neocortex); pseudo-rehearsal ("dreams") | `lifecycle/brain.py`, `lifecycle/sleep.py` |

## 1. Backbone

```
tokens ─► embed ─(+ Engram)─► prelude ─(+ Titans memory)─► e
            h = e
            repeat R times:  h = Core( Inject([norm h ; e]) + loop_emb[r] )     (weight-tied)
         ─► coda ─► norm ─► tied LM head
```

* **Blocks** are `x + Mixer(norm x)` followed by `x + FluidMoE(norm x)`.  The
  mixer is either a *liquid mixer* or GQA attention with RoPE and QK-norm.  The
  default layout puts a liquid block in the prelude, `[liquid, attn]` in the
  looped core and one attention block in the coda.  That is the LFM2 idea:
  mostly cheap local/recurrent mixing with a few attention layers.
* **Liquid mixer** = LFM2 double-gated short convolution followed by a liquid
  time-constant recurrence `s_t = a_t s_{t-1} + (1-a_t) z_t`,
  `a_t = exp(-softplus(dt(x_t))·exp(A))`.  This is the exact discretisation of
  `ds/dt = -(s - z)/τ(x)`, so the time constant depends on the input.  It is
  computed with a chunked parallel scan.
* **Looped core.** The core is iterated `R` times with input re-injection and
  a loop-step embedding.  `R` can be sampled during training
  (`loop_sampling="uniform"`) and picked freely at inference.  Inference can
  also stop early once `h` stops changing (`exit_tol`), which spends
  test-time compute only where needed.  Truncated BPTT through the last
  `bptt_loops` iterations bounds training memory.

## 2. Memory hierarchy (a continuum of update frequencies)

Hope / nested learning treats a model as a stack of memories that update at
different rates.  Recursion-X makes that stack explicit:

| level | what | updated | by |
|---|---|---|---|
| 0 | attention KV / liquid state | every token | forward pass |
| 1 | **Titans neural memory** `M` | every chunk (8 tokens) at test time | one gradient step on `ℓ(Mk, v)` with momentum + forget gate |
| 2 | **Engram tables** | whenever a fact is ingested | sparse row-local SGD; rows can live on disk |
| 3 | **skill adapters** (projected LoRA) | during wake, per skill | Adam on adapter params only |
| 4 | **base weights** | during sleep only | merge → distillation / rehearsal → protection |

The Titans memory is differentiable end-to-end, so the outer model learns
*how* to write to it.  The MIRAS axes can be configured: attentional bias
`l2 | huber | l1`, a retention gate, and momentum.  Its state can be carried
across segments of a long stream.

Engram lookups depend only on token ids, so the rows a sequence needs are
known before the forward pass.  That is what makes RAM/disk residency
practical (`storage="disk"` uses a `numpy.memmap`).  MoE experts can also be
offloaded to disk and paged in through an LRU cache (`FluidMoE.offload`).

## 3. The wake / sleep lifecycle

```
            ┌───────────────────────── AWAKE hemisphere ─────────────────────────┐
 new info ─►│ research (KnowledgeSource) ─► fact?  ─► Engram rows (sparse write)  │
            │                          └─► skill? ─► projected LoRA (base frozen)│
            │                                          │                        │
            │                               gate: good enough? ── no ─► discard  │
            │                                          │ yes                     │
            │                     serve now via skill router (nearest prototype) │
            └──────────────────────────────────────────┼────────────────────────┘
                              sleep pressure reached   ▼
            ┌───────────────────────── SLEEPING hemisphere ──────────────────────┐
            │ sync to awake base → NREM: merge adapters → REM: multi-teacher     │
            │ distillation + replay + dreams, GPM-projected grads → (grow experts)│
            │ → extend protected subspaces                                        │
            └─────────────────────────────────────────────────────────────────────┘
                              swap roles; old awake hemisphere is reset + re-synced
```

### Projected LoRA (the "skill" format)

For every `AdaptableLinear` with base weight `W` and protected input basis
`U` (orthonormal columns spanning the inputs that consolidated knowledge
uses):

* `ΔW = s · B A (I − UUᵀ)`.  The adapter is blind to protected directions, so
  merging it into `W` leaves the layer's response to old inputs (almost)
  unchanged.  Merging is exact for the new skill.
* `A` starts as the top-`r` principal directions of the *new skill's own
  activations* inside the free subspace ("data-projected init").  The skill's
  data is baked into the adapter geometry before the first gradient step.
* `U` grows GPM-style after every sleep.  The fewest principal directions of
  the new skill's activations are added until `threshold` (default 0.97) of
  their energy is protected.  The basis is capped at `max_protect_frac` of the
  width.

### The gate

A skill goes to the sleeping hemisphere only if its adapter reaches
`gate_min_acc` and beats the base by `gate_min_gain`.  The gate also reports
a *merge preview*: anchor-skill accuracy with the adapter always on.  A
single always-on adapter is mathematically identical to merging it.

### Sleep

1. **Synchronise** the sleeping hemisphere to the awake base and snapshot the
   awake hemisphere as a frozen teacher, so serving can continue.
2. **NREM – merge** (optional, `nrem_merge`, off by default): fold each
   accepted adapter into the base.  Experiments showed that merging, even
   projected adapters, damages the base far more than distillation can
   repair.  The default therefore distils into the *intact* base.
3. **REM – consolidate.**  The student is distilled from several teachers:
   each new skill's teacher is *base + that skill's adapter*, together with
   its ground-truth episodes.  Old skills' teacher is the pre-sleep base,
   applied to stratified rehearsal: the same number of examples per old skill
   per step, half from a small episodic buffer and half **dreams**.  Dreams
   are inputs the model samples from its own input distribution after an old
   instruction token, answered greedily by the teacher.  GPM gradient
   projection is available (`gpm_strength`) but off by default, since it
   blocked late skills from consolidating.
4. **Grow** (optional): if a skill still misses `growth_acc_threshold`, clone
   an expert in each MoE layer.  The new router row points at the skill's
   hidden-state direction, which gives the skill fresh, unprotected capacity.
5. **Protect** the new knowledge by extending `U`.  Later wake adapters are
   then projected away from it.
6. **Audit** (sleep as a transaction): run every skill's held-out retention
   probes on the awake snapshot (as served) and on the student.  Commit only
   if no protected skill regresses by more than `commit_max_drop` and every
   new skill clears the gate.  Otherwise **roll back**: discard the student,
   keep serving the pending skills from their adapters, and record the
   syndrome.
7. **Swap** (on commit).  The rested hemisphere wakes up.  The other drops
   its adapters and is re-synchronised: the reset.  Skills learned *during*
   sleep are handed over by re-learning them on the new base from their
   stored episodes.

`DualHemisphereBrain.sleep(background=True)` runs consolidation in a thread,
so the awake hemisphere keeps answering requests.

## 4. What is deliberately simplified in this prototype

* Scale: the architecture is written to scale.  The experiments use a
  2.4 M-parameter model on CPU with an algorithmic skill suite, not a 4 B
  model on real text.
* Titans memory is a *linear* matrix memory with an analytic gradient.  Deep
  (MLP) memories need per-sequence functional gradients and are a
  straightforward extension.
* "Research" is an interface (`SkillRecord.task.sample`).  In the benchmark
  it is a data oracle; in a full system it would be a tool-using agent loop
  (search, read, verify) that produces training episodes.
* No KV cache or incremental decoding; generation recomputes the prefix.
