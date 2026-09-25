"""HyperLoRA experiment: generate skill adapters from demonstrations.

1. Meta-train a HyperLoRA on a family of skills (base skills + the extra
   long-stream skills) with the pre-trained base frozen.
2. On *held-out* skills (the 6 continual-benchmark skills):
   a. zero-shot: accuracy with the generated adapter only;
   b. warm start: wake learning for a short budget starting from the
      generated adapter vs. the default data-projected initialisation.
"""
from __future__ import annotations

import argparse
import os
import random

import torch

import common
from common import RUNS, VOCAB, eval_sets, load_base, save_json, seed_all
from recursionx.data.tasks import make_suite
from recursionx.lifecycle.config import LifecycleConfig
from recursionx.lifecycle.skills import SkillRecord
from recursionx.lifecycle.wake import WakeLearner
from recursionx.modules.lora import active_adapters, remove_skill
from recursionx.skills.hyperlora import HyperLoRA, demo_features, generate_for, meta_train
from recursionx.train import evaluate

META_SKILLS = common.BASE_SKILLS + ["rotr", "double", "cumsum", "mirror_sum", "dedup",
                                    "count_first", "min", "skip_first"]
HELD_OUT = common.NEW_SKILLS


def zero_shot(model, hyper, tasks, evs, k_demos, rng):
    out = {}
    for t in tasks:
        lora = generate_for(model, hyper, [t.sample(rng) for _ in range(k_demos)])
        hyper.apply(model, lora)
        out[t.name] = evaluate(model, evs[t.name])["acc"]
        hyper.clear(model)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-steps", type=int, default=1500)
    ap.add_argument("--wake-steps", type=int, default=150)
    ap.add_argument("--k-demos", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    rng = random.Random(args.seed)
    model = load_base()
    meta = make_suite(META_SKILLS[:6], VOCAB, start_slot=0) + \
        make_suite(META_SKILLS[6:], VOCAB, start_slot=12)
    held = make_suite(HELD_OUT, VOCAB, start_slot=6)
    evs = eval_sets(meta + held, n=128)
    desc_dim = demo_features(model, [held[0].sample(rng)]).numel()
    hyper = HyperLoRA(model, desc_dim, rank=8, alpha=16.0, hidden=256)
    print(f"hypernetwork params: {sum(p.numel() for p in hyper.parameters()):,}", flush=True)
    res = {"zero_shot_before": zero_shot(model, hyper, held, evs, args.k_demos, rng)}
    meta_train(model, hyper, meta, args.meta_steps, k_demos=args.k_demos, log_every=100, seed=args.seed)
    res["zero_shot_meta_train"] = zero_shot(model, hyper, meta, evs, args.k_demos, rng)
    res["zero_shot_held_out"] = zero_shot(model, hyper, held, evs, args.k_demos, rng)
    print("zero-shot (meta-train skills):", res["zero_shot_meta_train"], flush=True)
    print("zero-shot (held-out skills):  ", res["zero_shot_held_out"], flush=True)
    os.makedirs(os.path.join(RUNS, "hyperlora"), exist_ok=True)
    torch.save(hyper.state_dict(), os.path.join(RUNS, "hyperlora", f"hyper_s{args.seed}.pt"))
    # control: a *skill-agnostic* generated adapter (mean descriptor over the
    # meta-training skills).  If it helps as much as hyper_init, the gain comes
    # from a meta-learned initialisation, not from reading the demonstrations.
    with torch.no_grad():
        mean_desc = torch.stack([demo_features(model, [t.sample(rng) for _ in range(args.k_demos)])
                                 for t in meta]).mean(0)
        blind = {k: (A.detach(), B.detach()) for k, (A, B) in hyper(mean_desc).items()}
    cfg = LifecycleConfig(wake_steps=args.wake_steps)
    res["warm_start"] = {}
    for t in held:
        row = {}
        for mode in ("default_init", "blind_init", "hyper_init"):
            emb = model.embed.weight.data.clone()  # wake trains the new token's row; undo between modes
            seed_all(args.seed)
            waker = WakeLearner(cfg, seed=args.seed)
            rec = SkillRecord(t.name, t, new_tokens=[t.task_token])
            init = {"default_init": None, "blind_init": blind,
                    "hyper_init": generate_for(model, hyper, [t.sample(rng) for _ in range(args.k_demos)])
                    if mode == "hyper_init" else None}[mode]
            waker.learn_skill(model, rec, init_lora=init)
            with active_adapters(model, {t.name: 1.0}):
                row[mode] = evaluate(model, evs[t.name])["acc"]
            remove_skill(model, t.name)
            model.embed.weight.data.copy_(emb)
        res["warm_start"][t.name] = row
        print(f"warm start {t.name}: {row}", flush=True)
    res["warm_start_mean"] = {m: sum(r[m] for r in res["warm_start"].values()) / len(held)
                              for m in ("default_init", "blind_init", "hyper_init")}
    print("mean:", res["warm_start_mean"], flush=True)
    save_json(res, os.path.join(RUNS, "hyperlora", f"hyperlora_s{args.seed}.json"))


if __name__ == "__main__":
    main()
