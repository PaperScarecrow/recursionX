"""Continual skill acquisition benchmark.

Start from the pre-trained base (6 base skills) and learn 6 new skills one at
a time.  After every new skill, evaluate exact-match accuracy on all 12 skills
through whatever the method would *serve*.  Methods:

  finetune         full fine-tuning on each new skill (classic catastrophic forgetting)
  finetune_replay  full fine-tuning with 50% experience replay (same episodic
                   buffer size as Recursion-X: 64 stored episodes per skill)
  lora_merge       plain LoRA per skill, merged into the base right away
  rx               Recursion-X: projected LoRA wake learning -> gate -> dual-
                   hemisphere sleep every 2 skills (REM: multi-teacher distillation
                   with stratified replay + dreams, GPM, subspace protection)
  rx_merge         rx, but adapters are merged into the base before REM (NREM merge)
  rx_merge_only    merge + protection only, no REM
  rx_no_dreams     rx with replay only (no self-generated dreams)
  rx_nogpm         rx with projected LoRA in wake but no gradient projection in REM
  rx_unprojected   rx with ordinary LoRA and no gradient projection
  rx_grow          rx with expert growth enabled during sleep
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import time

import torch

from common import (INPUT_WEIGHT, RUNS, base_tasks, eval_sets, load_base, new_tasks, save_json,
                    seed_all)
from recursionx import DualHemisphereBrain, LifecycleConfig, SkillRecord
from recursionx.data.tasks import collate
from recursionx.lifecycle.wake import WakeLearner
from recursionx.modules.lora import merge_skill
from recursionx.train import evaluate, evaluate_all, train_loop

WAKE_STEPS = 600
FT_LR = 1e-3


def lifecycle_cfg(**kw) -> LifecycleConfig:
    base = dict(wake_steps=WAKE_STEPS, sleep_pressure=2)
    base.update(kw)
    return LifecycleConfig(**base)


def run_finetune(model, replay: bool, log):
    rng = random.Random(0)
    btasks, ntasks = base_tasks(), new_tasks()
    evs = eval_sets(btasks + ntasks)
    buffer = {t.name: [t.sample(rng) for _ in range(64)] for t in btasks}
    matrix = []
    for t in ntasks:
        if replay:
            def sampler(n, t=t):
                k = n // 2
                old = [e for es in buffer.values() for e in es]
                return [t.sample(rng) for _ in range(n - k)] + [rng.choice(old) for _ in range(k)]
        else:
            sampler = lambda n, t=t: [t.sample(rng) for _ in range(n)]
        train_loop(model, list(model.parameters()), sampler, WAKE_STEPS, lr=FT_LR, batch=64,
                   input_weight=INPUT_WEIGHT)
        buffer[t.name] = [t.sample(rng) for _ in range(64)]
        accs = evaluate_all(model, evs)
        matrix.append({"after": t.name, **accs})
        log(t.name, accs)
    return matrix, {}


def run_lora_merge(model, log):
    btasks, ntasks = base_tasks(), new_tasks()
    evs = eval_sets(btasks + ntasks)
    cfg = lifecycle_cfg(projected=False, data_init=False)
    waker = WakeLearner(cfg)
    matrix = []
    for t in ntasks:
        rec = SkillRecord(t.name, t, new_tokens=[t.task_token])
        waker.learn_skill(model, rec)
        merge_skill(model, t.name)
        accs = evaluate_all(model, evs)
        matrix.append({"after": t.name, **accs})
        log(t.name, accs)
    return matrix, {}


def run_rx(model, log, **cfg_kw):
    btasks, ntasks = base_tasks(), new_tasks()
    evs = eval_sets(btasks + ntasks)
    cfg = lifecycle_cfg(**cfg_kw)
    brain = DualHemisphereBrain(model, cfg, input_weight=INPUT_WEIGHT)
    brain.register_base_skills([SkillRecord(t.name, t) for t in btasks])
    matrix = []
    for t in ntasks:
        rec = SkillRecord(t.name, t, new_tokens=[t.task_token])
        known = {n: evs[n] for n, r in brain.catalog.items() if r.status == "consolidated"}
        rep = brain.ingest(rec, evs[t.name], anchors=known, auto_sleep=False)
        print(f"  gate {t.name}: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in rep.items() if k != "skill"), flush=True)
        slept = None
        if len(brain.pending) >= cfg.sleep_pressure or t is ntasks[-1]:
            slept = brain.sleep(val_sets=evs, log_every=100)
            print(f"  sleep: {slept}", flush=True)
        accs = brain.evaluate(evs)
        matrix.append({"after": t.name, "slept": slept is not None, **accs})
        log(t.name, accs)
    extra = {"history": brain.history,
             "final_base_only": evaluate_all(brain.awake, evs),
             "n_experts": [m.n_experts for m in brain.awake.moe_layers()]}
    return matrix, extra


METHODS = {
    "finetune": lambda m, log: run_finetune(m, False, log),
    "finetune_replay": lambda m, log: run_finetune(m, True, log),
    "lora_merge": run_lora_merge,
    "rx": run_rx,
    "rx_merge": lambda m, log: run_rx(m, log, nrem_merge=True),
    "rx_merge_only": lambda m, log: run_rx(m, log, nrem_merge=True, rem_steps=0),
    "rx_no_dreams": lambda m, log: run_rx(m, log, dream_frac=0.0),
    "rx_nogpm": lambda m, log: run_rx(m, log, gpm_strength=0.0),
    "rx_unprojected": lambda m, log: run_rx(m, log, projected=False, data_init=False, gpm_strength=0.0),
    "rx_grow": lambda m, log: run_rx(m, log, grow_experts=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="finetune,finetune_replay,lora_merge,rx")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=os.path.join(RUNS, "continual"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    base = load_base(args.base)
    evs = eval_sets(base_tasks() + new_tasks())
    initial = evaluate_all(base, evs)
    print("base model:", {k: round(v, 3) for k, v in initial.items()}, flush=True)
    for name in args.methods.split(","):
        seed_all(args.seed)
        model = copy.deepcopy(base)
        t0 = time.time()
        print(f"=== {name} ===", flush=True)

        def log(skill, accs):
            print(f"  [{name}] after {skill} ({time.time() - t0:.0f}s): " +
                  " ".join(f"{k}={v:.2f}" for k, v in accs.items()), flush=True)

        matrix, extra = METHODS[name](model, log)
        save_json({"method": name, "seed": args.seed, "initial": initial, "matrix": matrix,
                   "seconds": time.time() - t0, **extra},
                  os.path.join(args.out, f"{name}_s{args.seed}.json"))


if __name__ == "__main__":
    main()
