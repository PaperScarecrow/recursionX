"""Compare sleep-consolidation variants on the same pair of freshly learned
skills (rotl, pred).  The awake hemisphere's adapters are cached so every
variant consolidates exactly the same teachers."""
from __future__ import annotations

import argparse
import copy
import os

import torch

from common import INPUT_WEIGHT, RUNS, base_tasks, eval_sets, load_base, new_tasks, save_json, seed_all
from recursionx import DualHemisphereBrain, LifecycleConfig, SkillRecord
from recursionx.train import evaluate_all

CACHE = os.path.join(RUNS, "sleep_variants", "awake.pt")

VARIANTS = {
    "merge+rem300": dict(),
    "nomerge+rem300": dict(nrem_merge=False),
    "merge+rem300+balanced": dict(old_batch_per_skill=16),
    "nomerge+rem300+balanced": dict(nrem_merge=False, old_batch_per_skill=16),
    "nomerge+rem600+balanced": dict(nrem_merge=False, old_batch_per_skill=16, rem_steps=600),
    "merge_only": dict(rem_steps=0),
}


def awake_brain(cfg):
    bt, nt = base_tasks(), new_tasks()[:2]
    evs = eval_sets(bt + nt)
    seed_all(0)
    brain = DualHemisphereBrain(load_base(), cfg, input_weight=INPUT_WEIGHT)
    brain.register_base_skills([SkillRecord(t.name, t) for t in bt])
    recs = [SkillRecord(t.name, t, new_tokens=[t.task_token]) for t in nt]
    if os.path.exists(CACHE):
        ck = torch.load(CACHE, weights_only=False)
        from recursionx.modules.lora import add_skill_adapter
        for r in recs:
            add_skill_adapter(brain.awake, r.name, cfg.rank, cfg.alpha, projected=cfg.projected)
            r.episodes = ck["episodes"][r.name]
            r.status = "accepted"
        brain.awake.load_state_dict(ck["state"])
        from recursionx.lifecycle.skills import build_prototype
        for r in recs:
            brain.catalog[r.name] = r
            brain.router.register(r.name, build_prototype(brain.awake, r.episodes), r.name)
            brain.pending.append(r)
    else:
        for r in recs:
            brain.ingest(r, evs[r.name], auto_sleep=False)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        torch.save({"state": brain.awake.state_dict(),
                    "episodes": {r.name: r.episodes for r in recs}}, CACHE)
    return brain, evs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    results = {}
    for name in args.variants.split(","):
        cfg = LifecycleConfig(**VARIANTS[name])
        brain, evs = awake_brain(cfg)
        if name == args.variants.split(",")[0]:
            results["awake(served)"] = brain.evaluate(evs)
            print("awake served:", results["awake(served)"], flush=True)
        rep = brain.sleep(log_every=100)
        accs = brain.evaluate(evs)
        results[name] = {**accs, "sleep_seconds": rep["seconds"]}
        print(f"[{name}] " + " ".join(f"{k}={v:.2f}" for k, v in accs.items()), flush=True)
        save_json(results, os.path.join(RUNS, "sleep_variants", "results.json"))
