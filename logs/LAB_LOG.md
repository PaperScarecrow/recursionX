# Lab log — Recursion-X on 7600X + RTX5060 8GB + 32GB
Lab dir: /home/paperscarecrow/recursionX-lab (copy of zip, original untouched)
Env: htpcvenv torch 2.11.0+cu128, pytest 9.1.1 installed (additive only)

## 2026-09-28
- Recon: 6C/12T 7600X (not 7700X, close), RTX5060 8GB, 30Gi RAM, 849G free. torch 2.11+cu128 sees GPU.
- 27/27 pytest pass in ~30s.
- Smoke: model forward OK on cuda. Wake OK. Sleep hit router device bug:
  `skills.py _classifier/classify`: torch.ones/eye/full default CPU vs CUDA exemplars.
  Fixed: device=X.device everywhere + store exemplars CPU + move mu/sd/W to X.device in classify.
  Retest: 27/27 pass, GPU wake->sleep->audit end-to-end OK.
- Added --device to common.pretrain, pretrain_base, continual, run_matrix, sleep_variants (default cpu, CPU repro unchanged).
- Pretrain seed0 3000 steps GPU: 100s (vs 1019s CPU orig). Final copy1.00 reverse1.00 succ1.00 sort0.92 max1.00 interleave1.00.
  Orig reported sort0.98. +1000 resume -> sort0.945. Using base_4k (4000 total) as continual base. Backups: base_3k_gpu.pt, base_4k.pt.
- Matrix launched: finetune_replay,rx,rx_audit,rx_no_dreams,rx_grow x seed0, parallel2 threads3 device cuda.
  Early: rx rotl gate 1.00, merge_preview 0.98, zero-interference holds. GPU 86%, 1.8GB.

## Matrix GPU seed0 — finished
Base: base_4k (sort 0.945). 5 methods, parallel2, threads3, cuda. Total wall ~55 min for 5 runs (vs ~7h CPU orig).
- finetune_replay 0.775 (170s): add_first 0.07 collapse, sort_desc 0.38. Weaker than orig 0.867 — base + GPU noise.
- rx (no audit) 0.813 (678s): sleep2 committed with syndrome (sort 0.91->0.75 etc), add_first rejected 0.25, final add_first 0.00.
- rx_audit 0.932 (1020s): sleep1 commit, sleep2/3/4 rollback. Served swap 0.66 low (router strain, see below).
- rx_grow 0.932 (953s): identical served to audit. Student probes better with growth (sleep2 swap 0.73->1.00, sort_desc 0.69->0.80) but still rollback on old regressions. Growth helps new, not old.
- rx_no_dreams 0.935 (1008s): all rollback, served similar. Dreams comparison moot when nothing commits — need a committing case to judge dreams.
Key new signals:
1. Audit gap 0.813->0.932 on seed0 (orig showed 0.900->0.953 on seed1). Transaction matters more than headline.
2. Router strain: swap_pairs served 0.93 (2 pending) -> 0.61/0.55/0.45 (4-5 pending). Ridge classifier confuses similar permutation tasks. Next: per-skill precision/recall + prototype geometry.
3. Sleep2/3/4 students all damage old (sort/pred/rotl -0.15..-0.25) while new sometimes fail (add_first 0.41/0.02). REM 600 not enough for 3-4 skills at once? Next: sleep_pressure 2 vs 3-4, more REM, prioritized rehearsal.
4. GPU 10x: pretrain 100s vs 1019s, replay 170s vs 1680s, RX 11-17min vs 90min. 8GB fits 2 concurrent jobs (2.2GB).
Files: runs/continual/*.json + summary.md + continual.png/matrices.png
Patches kept in lab only: skills.py router device fix, --device flags. Original zip untouched.

## Critic questions (buddy's Claude, 2026-09-28) — whiteboarded in recursionX_notes.md section 9
1. replay same-computer: done (0.775 vs 0.932 s0; 0.775±0 vs 0.955±0.017 3-seed); queued compute-matched replay-3x
2. keep-adapters: done by accident (s0 audit 0.932 on adapters); formal wake-only via router_probe.py queued
3. shared instruction token pilot: queued (~20 lines, rotl/pred)
4. 3 seeds: done but seeding bug found (replay bit-identical) — fix: thread args.seed to Brain/Waker/samplers, re-run
5. tune-after-final: queued long-stream 14 skills from s0 audit brain

## Breakthrough night results
- Hybrid router (tok2adapter + ridge fallback): 27/27 green. Ceiling verified pred .57->1, swap .45->1, sort_desc .73->.98. rx_audit+H seed0 0.984 (prev 0.932). Audit now measures hybrid-served.
- p1 (pressure 1): 3 single commits then wall (sort_desc fails 2x, add_first .44). Final .890.
- p1r12 (pressure1 + REM1200): 3 commits, sort_desc single STILL rolls back (same syndrome) -> structural not cost. Final served .972 (hybrid serving).
- addfirst probe: projected .000 plain .004 on 3-commit base -> protection innocent, plasticity loss real. Wakeability chaotic across trajectories (.98/.74/.44/.00).
- sleep variants fixed (nrem_merge=True restored): merge sort .84 vs nomerge .90; merge_only surprise preserves (.95/1/1 + rotl .96 pred .99) - 300-step norm hypothesis open.
- Seeding bug: --seed never reaches Brain/samplers; replay 3-seed bit-identical. Fix queued.
- Running: hybrid s1 + replay-3x s0.

## Final (night close)
- Hybrid 2-seed: s0 0.984, s1 0.983. swap 1.00 both, sort_desc 0.98 both.
- replay-3x s0: 0.839 (497s). +0.064 over 1x, still -0.14 vs hybrid.
- Paradox: hybrid serving raises audit before-scores -> stricter rollbacks (h1 rolls back s1-committed sleep) yet higher served finals.
- p1r12 final 0.972 (3 commits + hybrid serve).
- Stopping GPU campaign here. Lab state: runs/continual* + router_probe + sleep_variants logs. Code deltas: skills.py (device), brain.py (hybrid tok routing), continual.py (+p1/rem1200 methods), common.py/pretrain_base/run_matrix/sleep_variants (+device), sleep_variants VARIANTS fix.
