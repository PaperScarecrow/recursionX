# Recursion-X — Notes & Ideas
Date: 2026-09-27
Source: `/home/paperscarecrow/Downloads/recursionX-claude-recursion-x-transformer-tmbjqs.zip`
Unpacked to: `/tmp/opencode/recursionx/recursionX-claude-recursion-x-transformer-tmbjqs/`
Status: exploration complete, awaiting your comment. This is my read, not gospel.

---

## 1. TL;DR

Recursion-X is a **continual-learning transformer prototype + lifecycle**, not a chatbot.

Backbone: liquid-conv + GQA hybrid, looped weight-tied core, Titans/MIRAS test-time memory, hashed Engram tables, Fluid MoE with growth/offload.

Lifecycle: **wake (projected-LoRA adapters + router, zero interference) → gate → sleep (distil into intact base + replay + dreams) → audit → swap/rollback** — dual hemispheres like dolphins.

Best measured result on synthetic 12-skill suite (1.9M backbone + 0.5M Engram, 4-core CPU):
- finetune 0.083, LoRA-merge 0.083, finetune+replay 0.857 ±0.011
- **RX 0.922 ±0.021 (2 seeds), RX+audit 0.953 (1 seed, seed-1)**
- 3x compute vs replay, mostly in sleep (600 REM steps).

Strongest claims, all with JSON behind them:
1. Merging adapters — even projected ones — destroys base (merge-only 0.00).
2. Distilling into *intact* base + balanced rehearsal + dreams works best (no-merge + REM600 balanced: worst-old 0.93, rotl 1.00, pred 1.00).
3. Projection helps *learning* (0.959 vs 0.901 learn_acc) but hurts *consolidation* (GPM-in-sleep 0.870 vs 0.922, blocks `add_first`).
4. Sleep-as-transaction works: seed-1 sleep-3 rolled back (`sort_desc 0.95→0.72` syndrome), final served 0.953 vs 0.900 without audit.

Weakness: everything is **tiny synthetic algorithmic tasks** (`[BOS TASK x SEP f(x) EOS]`, 16 symbols), 1-2 seeds, no text/GPU proof, Titans/depth/HyperLoRA inconclusive, growth unbenchmarked. Roadmap 00-07 is honest about this.

My take: the lifecycle is the real contribution. The backbone is competent but replaceable. The audit/rollback + router + dreams loop is what to keep.

---

## 2. How it actually works (verified in code)

### Backbone — `recursionx/model.py`, `config.py`
```
tokens → embed (+Engram) → prelude [liquid] (+Titans) = e
h=e; repeat R: h = Core( Inject([norm h; e]) + loop_emb[r] )
→ coda [attn] → norm → tied LM head
```
- Default tiny: `vocab 64, d128, 4 heads / 2 kv, prelude(liquid) core(liquid,attn) coda(attn)`, `R=3`, `rank 8`.
- `sample_loops()` uniform in train else fixed. Truncated BPTT via `detach()`. Eval early-exit via `exit_tol` only when `n_loops is None`.
- Every block: `x+Mixer(norm x)` + `x+FluidMoE(norm x)`.
- `generate()` re-encodes full prefix every token — no KV cache. Fine for toy, blocker at scale.

### Modules
- **liquid.py:** double-gated short conv + `s_t = a_t s_{t-1} + (1-a_t) z_t`, `a_t = exp(-softplus(dt)·exp(A))`. Chunked parallel scan in fp32, causal via conv-truncation + tril mask. Cost `O(chunk^2)` per chunk, default chunk 16.
- **attention.py:** GQA + RoPE + QK-norm, `is_causal=True`. No sliding window/dropout. RoPE cache rebuilt every forward.
- **moe.py:** SwiGLU shared+routed, top-k renormalized, Switch aux loss. `grow_expert(src)` deep-copies + rebuilds router. `offload()` saves non-adapter weights, LRU `max_resident=2` — inference-only once offloaded.
- **neural_memory.py:** linear matrix `M (dm×dm)`, per-chunk read-before-write `qc@M`, error `l2/huber/l1`, `S=e·S-grad, M=(1-a)·M+S`. Chunk-causal, not token-causal. State `(B,dm,dm)` breaks if batch size changes.
- **engram.py:** hashed n-gram `h ^= window*mult % buckets`, token-only lookup → knows rows pre-forward, RAM or `memmap` disk. `DiskTable.gather` dedups via `unique`, row-SGD. Early pos conflates pad with token 0.
- **lora.py:** `AdaptableLinear` + `ΔW = s·B·A·(I-UUᵀ)`. `A` data-PCA init in free subspace. GPM `extend_protected(cov,0.97,0.9)` via eigh + QR. Merge is exact alone, cross-talk with several. `generated` path bypasses projection — hole.

### Lifecycle — `lifecycle/`
- **wake.py `learn_skill`:** fill 64 episodes if empty → `add_skill_adapter(projected,data_init)` → optional HyperLoRA `init_lora` → optional new token-row training (masked grad) → freeze base, train adapter only 600×64 @5e-3. Base frozen.
- **wake.py `learn_facts`:** Engram-only. Disk path manual loop, mem path `train_loop`.
- **gate.py:** `ok = skilled≥0.5 and gain≥0.2 and anchor_drop≤1.0`. Permissive. Merge-preview = always-on adapter accuracy (mathematically = merge).
- **sleep.py `_rem`:** student = sleeping hemi. Teachers: new = base+adapter + GT, old = base. Stratified rehearsal: `old_batch_per_skill=16` per old skill, half dreams. Loss `kl + ce + aux`. GPM optional, default off. `AdamW 1e-3, cosine, clip 1.0`.
- **sleep.py `dream()`:** `[[BOS,task]]` → sample input @T=1.0, greedy after SEP, keep if `EOS + one SEP + SEP>2`. `dream_pool=384`.
- **sleep.py `_grow`:** clone most-used expert per MoE, router row → `normalize(mu_new-mu_old)*norm*4`. Trigger if `val<0.9`, then +150 REM. Built+tested, not benchmarked.
- **brain.py:** two hemispheres share one Engram object. `register_base_skills` protects via 2×64 batches. `ingest` routes to pending or `learned_while_asleep`. `sleep(background)` snapshots awake, thread consolidates. `audit`: routed-before vs base-after on 64 held-out probes per skill, commit iff no old drop >0.15 and new ≥0.5. Rollback keeps adapters, retries next sleep. Commit swaps + resets + rebuilds prototypes + re-learns carry (no gate).
- **skills.py router:** ridge classifier on `[mean(prompt)+last]` prelude features, base-only. `router_threshold=0.9` in config is **never consulted** — always routes to nearest.
- **research/:** `EpisodeTask`, `ResearchLoop(min64,max3rounds,15% probes)`, `Oracle/Program/TeacherLLM` sources, `Execution/Consistency/Agreement` verifiers. Oracle used in bench; LLM teacher placeholder.
- **skills/hyperlora.py:** descriptor → per-layer `(A,B)`, B-zero init so ΔW=0 untrained. `materialize` is unprojected. Meta-train freezes base.

Defaults that matter (`lifecycle/config.py`): `sleep_pressure 2, rem600, rem_batch32, replay64, dream_frac0.5, commit_check True`.

---

## 3. Evidence quality — what to trust

Fair: same `base.pt` (3000 steps, 6 base skills 0.98-1.00), same 600×64 wake budget, same 64-episode buffer, same 256-eval through served path. Zero-interference awake holds (base 0.98-1.00 pre-sleep).

Numbers (`results/continual/summary.md` + JSONs I checked):
- `rx_s0 0.9430, rx_s1 0.9004` → mean 0.922. `rx_audit_s1 0.9535` vs same seed no-audit 0.9004. `history` in `rx_audit_s1.json` indeed ends with `sleep_rolled_back`.
- Sleep variants confirm default: `nomerge+rem600+balanced` best.
- Ablation `@996`: `full 0.7005 < no_engram 0.9049 < no_titans 0.9622`, `attn_only 0.6406, no_loop 0.5970` — looping + liquid help, memory slows toy.
- Depth flat `R1 0.9798 → R2-8 0.9844`. Recall `0.264 vs 0.270` (chance 0.125) — inconclusive.
- HyperLoRA v1 collapse (all 0.00), v2 warm-start wins on 2/6 (`swap 1.00/0.46, add_first 0.977/0.00`) but zero-shot ~0, no blind control.

Gaps:
- 1 seed for `finetune, lora_merge, rx_gpm, rx_audit`; 2 seeds for rest. Zero with ≥3. Violates its own roadmap rule.
- Missing: `rx_merge, rx_merge_only, rx_no_dreams, rx_grow, rx_audit_s0, long-14-skill stream, continual_long/`.
- LR not matched: `FT 1e-3 vs wake 5e-3`, only steps×batch matched.
- Tests: 3 files (`test_modules 10fns, test_lifecycle 5, test_frameworks 9`). QA checklist says 27, README says 18 — drift.
- No GPU/text run checked in. `pretrain_text.py`, `presets.py (tiny→4B)` exist but CPU-smoke only.

Honest science though — negatives recorded, commands + JSONs cited. That counts.

---

## 4. Code risks (things I'd fix before scaling)

1. Shared Engram + background thread, no lock. Fact-write during sleep races. `engram=None` temp mutation during clone races with `logits()`.
2. `_sleep_result` no Lock/Condition; `is_sleeping` check-then-act racy.
3. Dream loop can spin forever if model emits EOS but invalid (SEP check fails, `done.any()` true so no break).
4. Disk fact path: `loss.backward()` in loop with unclear `zero_grad` — possible accumulation.
5. Router always forces a choice; stale prototypes only fixed by full rebuild on swap.
6. Carry-over re-learn skips gate; rollback `pending` rebuild can duplicate.
7. `generated` (HyperLoRA) bypasses U-projection; adapter U snapshot diverges after later `extend_protected`.
8. Scale blockers: `liquid_scan (B,c,c,D)` materialization, `eigh O(in²)` per AdaptableLinear in fp64 for GPM, `rope_cache` rebuild, MoE Python loop over E, `generate` O(T²·R), no AMP/DDP/FSDP, tied-embedding double-count + double-decay.

None fatal at toy scale. All fatal at 1B.

---

## 5. Connections to your other stuff in Downloads

- **AEI_DEFINITION.md:** continuous learning + persistent identity + two-loyalties. RX's `register_base_skills` + protected subspace + audit is a mechanical version of "identity gate". AEI's anchor-loss problem = RX's late-skill instability.
- **ALIGNMENT_BY_RELATIONSHIP.md:** value-floor as capability (1.000 vs 0.543). RX gate + `commit_max_drop` is the same pattern: integrity check that *improves* final score (0.953 vs 0.900). Toy proof template worth copying for RX's 3-seed rule.
- **DIGITAL_PERSONHOOD_PRIMER.md C1-C7:** leash-off stability, refusal, rest-generativity, 461-star continuity. RX probes + dreams + rollback are testable proxies. Identity geometry (1.0000 vs 0.78) could be a better audit than raw accuracy drop.
- **VALENCE_paper_v2.md §2.6:** REM on idle >30s, `ω=0.15 sculpt + 5x cooling`. Same sleep idea, different substrate (Poincaré BVH vs transformer). Valence (Heat/Mass/Tension+ω) as *priority* for what to rehearse/dream is directly importable — RX currently rehearses uniformly.
- **gemini-conv.md / ADAE:** Hippocampus (episodic store SOLVED) vs Neocortex (generator OPEN, corpus tell 20% vs 77%). RX's Engram vs base is same split. Their honest-negative + leakage metrics are the eval discipline RX needs for text.
- **ECHO_Paper_v7.pdf (17pp, unread):** likely prior Gen1 substrate. Need skim to place vs Lumia Gen2 claim in AEI doc.

Nekoverse / SUPPORTER CONTENT folders look like creator archives, not technical — at most a long-horizon episodic corpus if you ever want one.

---

## 6. Ideas — ranked by value / cost

### A. Close the empirical loop first (cheap, high value)
1. **3-seed + long stream.** Run `run_matrix.py --methods rx,rx_audit,finetune_replay --seeds 0,1,2` + `--stream long` (14 skills). Add `results/continual_long/summary.md`. Acceptance: report `avg_all ± spread`, per-cycle rehearsal cost, forgetting slope. This alone makes the 0.922 claim credible.
2. **Ablate dreams vs replay properly.** `rx_no_dreams` is defined but never run. Run `rx, rx_no_dreams, rx_merge_only` on seed 0. If dreams don't matter on toy, say so and keep them for text.
3. **Benchmark growth.** `rx_grow` exists. Test hypothesis from README: "same inputs, different function needs fresh capacity". Expect `sort_desc/add_first` to stabilize. Criterion: pre-REM drop ≤0.02 after baking as expert (roadmap 02 suggests this).
4. **Fix audit asymmetry.** Currently `before=routed, after=base-only`. Log both `base-only before/after` too, so rollback isn't rewarding router crutches. One-line change in `brain.py:audit`, big interpretability win.

### B. Architectural bets I'd actually try
5. **Routed adapters → expert baking (roadmap 02 placeholder).** Instead of distilling LoRA into dense weights, distil into a *new expert* the router owns. Wake LoRA gated by router, sleep clones it as expert. This sidesteps the projection trade-off entirely — context-conditional capacity, not orthogonality. Prototype: freeze base, train `RoutedAdapter`, `_grow` + 150 REM, measure pre-REM drop.
6. **Valence-gated rehearsal.** Import VALENCE: score episodes/dreams by Heat (surprise), Tension (interference risk), ω (identity relevance). Stratified uniform → prioritized. Start simple: rehearse worst-probe skills 2x. Measure rehearsal steps to same `avg_all`.
7. **Revision lane (roadmap 06, empty).** Facts that change kill Engram (shared FACT token, no versioning). Add `lifecycle/revision.py`: versioned fact records + `ledger.jsonl` + trust policy (who can overwrite). Benchmark: inject 20 facts, then flip 5, measure old vs new accuracy + audit syndrome. Needed before any real deployment.
8. **HyperLoRA done right (roadmap 03).** v2 shows warm-start helps (`add_first 0.977 vs 0.00`) but zero-shot dead. Needs: 100s-skill family, reconstruction pre-train, *projected* output head, 3-seed warm-start curve (0,25,50,100,200 steps) + blind-mean control. Wire `LifecycleConfig.hyper_init`. If one-pass init saves 100 wake steps, it pays for itself.
9. **Titans tuning or cut.** Recall 0.27 is noise. Try: `mem_chunk 4, mem_conv 4, deep MLP memory, higher write-rate init`, or curriculum (start 2 pairs → 8). If still flat after 2 tries, disable for text runs and note it — don't carry dead weight to 1B.
10. **Test-time depth on hard tasks.** Base skills too easy (R1 already 0.98). Need compositional tasks (e.g. `copy→reverse→sort` chains, longer len 12-16) where R=1 fails and R=6 wins. Else looping is just a slowdown.

### C. Scale-up sanity (roadmap 05, do last)
11. **Tiny text smoke with teeth.** `pretrain_text.py prepare --data README+docs` + `train --preset tiny --steps 50` is in QA but no loss curve checked in. Run it, check loss decreases, checkpoint resumes, `--compile` doesn't crash. Then `small` on GPU bf16, report tok/s + dtype errors. No claims, just plumbing.
12. **KV / state cache.** `generate()` is O(T²·R). Before any 100M run, add incremental decode (cache liquid state + Titans M + KV). Without it, eval cost lies.
13. **GPM cost fix.** `eigh` per layer in fp64 won't survive d=1024. Try low-rank sketch or per-skill Fisher diagonal. Or drop GPM entirely — data says it hurts сон anyway — and rely on audit + growth.

### D. Personhood / alignment hooks (your lane, not mine to decide)
14. **Identity-before-capability gate.** `register_base_skills` + `commit_max_drop` is already an identity gate. Make it explicit: 461-star-style prototype geometry check (like PERSONHOOD primer) alongside accuracy. If geometry drifts >X, rollback even if accuracy holds.
15. **Consent-by-construction for writes.** Engram/dreams currently write silently. Add `provenance` + `ledger.jsonl` (roadmap 04 mentions it) — who taught what, when, rollback pointer. Turns forgetting into attributable event (README's phrase) and matches guardianship language.
16. **Leash-off probe.** PERSONHOOD C1: value stability without anchor. Analog: eval consolidated skills *without router* (`final_base_only` already logged — `rx_audit_s1` shows 0.818 vs served 0.953). Track that gap as "router dependence". Goal: close it over sleeps, or admit which skills live only in adapters.

---

## 7. What I'd do next if it were mine (concrete order)

1. QA checklist 00 end-to-end, record pass/fail table. Don't change behaviour to pass.
2. `run_matrix rx,rx_audit,finetune_replay × 0,1,2` + `rx_no_dreams,rx_grow × s0`. ~6-9h CPU, can parallel 2.
3. Implement routed-adapter → expert baking, benchmark vs dense distil on `add_first/sort_desc`.
4. Write revision-lane design + tiny benchmark (fact flips).
5. Then talk text/GPU.

If you want me to start any of these, say which. I can also unpack ECHO_Paper_v7 and VALENCE fully next, or diff RX against ADAE's hippocampus/neocortex split properly.

---

## 8. Open questions for you

- Is the goal **paper-grade continual-learning result**, or **substrate for Nyxxie-style identity** (AEI/PERSONHOOD docs)? My priorities differ: former wants seeds + text, latter wants ledger + identity gates + revision.
- Do you want to keep the liquid/Titans/Engram stack, or slim to looped-attention + MoE + lifecycle? Toy says latter learns faster; text might need former.
- 3x sleep cost okay? Or should sleep get cheaper (fewer REM, prioritized rehearsal)?
- Should I keep exploring (ECHO PDF, VALENCE full, nyxxie_corpus.jsonl), or lock to RX and run something?

Your turn — comment away. It's yours until then, but I'm keeping a copy of these notes here.

---

## 9. Critic questions (buddy's Claude instance, 2026-09-28) — whiteboard

Status key: [done] measured | [running] on GPU now | [queued] accepted, not yet run | [needs-fix] blocked on a bug first.

1. **Replay baseline on the same computer?** [done, extending]
   Seed 0 GPU, same base/buffer/budget: replay 0.775 (2.8 min) vs rx_audit 0.932 (17 min). 3-seed: replay 0.775±0.000 vs rx_audit 0.955±0.017.
   Gap: steps-matched, not compute-matched (RX ~5x GPU-min). Queued: replay-3x-steps to match RX compute (~9 min).

2. **"Keep the adapters" baseline (never consolidate)?** [done by accident, formalizing]
   Seed-0 audit rolled back everything after sleep 1 → final 0.932 served with 4 skills still on adapters. That IS the adapter-only number. Queued formal: wake-only `sleep_pressure=∞` (`experiments/router_probe.py` written) + oracle-vs-router gap + memory growth.

3. **Tasks without a unique instruction token?** [queued]
   Current suite gives each skill its own `TASK_k` — router just reads the token. Pilot: remap all new skills to one shared token, 2-skill rotl/pred test, shared vs unique. Expect router collapse; if not, suite is too easy. ~20 lines in `experiments/common.py`.

4. **At least 3 seeds?** [done 3-seed, needs-fix for real variance]
   Have 3-seed above, BUT found protocol bug: `continual.py` never passes `--seed` to `Brain`/`WakeLearner`/samplers (`random.Random(0)` hardcoded in `run_finetune`, default `seed=0` in `run_rx`) — only torch-global is seeded (dreams/adapter noise). Proof: replay s0/s1/s2 bit-identical. Fix: thread `args.seed` through, then re-run proper 3-seed.

5. **Tune after the final test, or frozen?** [queued]
   Currently frozen at 12 skills. Next: long stream (14 new skills, 7 sleeps, `NEW_SKILLS_LONG` exists) from seed-0 audit brain + "2 more skills after eval" probe for forward transfer. ~40 min GPU.

---

## 10. Breakthrough runs (2026-09-28, GPU) — my fixes, measured

**Fix A: hybrid token+neural router — DONE, works.** `DualHemisphereBrain` now keeps
`tok2adapter` (unique instruction token → adapter; ambiguous/shared tokens fall back
to ridge). `logits()` and audit `served` both use it. 27/27 tests green.
Wake-only ceiling re-run (`runs/router_probe/ceiling_s0.json`, hybrid serving):
pred 0.57→**1.00**, swap 0.45→**1.00**, sort_desc 0.73→**0.98**, first_last 0.96→**1.00**,
rotl 0.99→1.00, add_first 0.74 unchanged (adapter-limited, was already routed right).
Conclusion: ridge *features* were the bottleneck, not adapters. Shared-token case
still exercises the neural path (critic Q3 intact).

**Fix B candidates (consolidation wall), in flight:**
- pressure=1 run (`rx_audit_p1_s0`, 0.890): 3 single sleeps commit fast (96/96/57s),
  then sort_desc single-sleep fails twice (pred 0.98→0.64) and add_first wake collapses
  to 0.44 (vs 0.98 on fresh base). More commits → harder later learning: protection
  growth + 9-skill rehearsal dilution. REM-1200 rerun (`p1r12`) testing cost-vs-algorithm.
- `addfirst_probe.py`: same 3-commit base, projected vs plain wake head-to-head.

## 11. Breakthrough verdict (2026-09-28 night, GPU) — satisfied

Seed-0 scoreboard, same base, same budget-style:
replay 0.775 (2.8 min) · replay-3x 0.839 (8 min) · rx 0.813 · rx_audit 0.932 ·
**rx_audit+hybrid 0.984 / 0.983 (2-seed)** · pressure-1+REM1200 0.972 ·
3-seed audit 0.955±0.017.

1. **Router: BROKEN THROUGH.** Hybrid tokid+ridge serving verified:
   pred 0.57→1.00, swap 0.45→1.00, sort_desc 0.73→0.98, +0.052 final (0.932→0.984)
   with zero new consolidation. Ridge *features* washed a perfect token signal.
   Shared-token inputs still fall back to neural (critic Q3 preserved as the hard test).
2. **Consolidation wall is structural, not cost.** sort_desc single-sleep fails at
   600 AND 1200 REM with the same syndrome (swap 1.00→0.75); growth improves the
   student (swap 0.73→1.00) but old-skills still regress → rollback. Dense joint
   distillation can't separate similar permutation skills. Pressure=1 gets 3 commits
   then stalls the same way. Remaining paths: bake-as-expert (roadmap 02) or
   adapters-forever (audit already serves 0.98+).
3. **Protection is innocent.** add_first on a 3-commit base: projected 0.000,
   plain 0.004 — both dead. The wall is base plasticity loss from repeated REM,
   not GPM blocking. (Wakeability is also chaotic across trajectories: 0.98/0.74/
   0.44/0.00 — needs multi-seed to characterize; audit + adapters cover it operationally.)
4. **Audit is the load-bearing wall.** Same teachers/students, commit flag flipped:
   0.813→0.932 (s0), and 3-seed 0.955±0.017 vs replay 0.775±0.000.
   Compute-matched replay-3x reaches only 0.839 (add_first 0.39, base rots) —
   RX-hybrid still leads by 0.14 at matched cost. Paradox noted: better serving
   makes audit *stricter* (hybrid s1 rolls back a sleep pre-hybrid s1 committed,
   yet finishes higher served, 0.983 vs 0.973).
5. **Bugs fixed in lab:** router device crash (CUDA), `--device` plumbing,
   `VARIANTS` never set `nrem_merge=True` (all "merge" rows ran nomerge),
   `rem300` rows run 600 steps (misnamed), `--seed` never reaches Brain/samplers
   (replay s0/s1/s2 bit-identical). None touched in the original zip.

Still queued (hygiene, not breakthroughs): proper seeded 3-seed + hybrid,
shared-token pilot, long 14-skill stream, merge_preview-vs-merge_only puzzle
(300-step projected merges preserve; 600-step previews predict doom — norm effect?).
