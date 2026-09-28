"""Shared setup for the Recursion-X experiments."""
from __future__ import annotations

import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recursionx import RecursionX, RXConfig  # noqa: E402
from recursionx.data.tasks import Mixture, Vocab, make_suite  # noqa: E402
from recursionx.train import EvalSet, evaluate_all, train_loop  # noqa: E402

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
VOCAB = Vocab(16)
BASE_SKILLS = ["copy", "reverse", "succ", "sort", "max", "interleave"]
NEW_SKILLS = ["rotl", "pred", "swap_pairs", "sort_desc", "add_first", "first_last"]
# longer stream: every remaining skill in the suite (14 new skills, 7 sleeps)
NEW_SKILLS_LONG = NEW_SKILLS + ["rotr", "double", "cumsum", "mirror_sum", "dedup", "count_first",
                                "min", "skip_first"]
STREAMS = {"default": NEW_SKILLS, "long": NEW_SKILLS_LONG}
STREAM = "default"  # set by experiment entry points
INPUT_WEIGHT = 0.1


def model_config(**kw) -> RXConfig:
    base = dict(vocab_size=VOCAB.size, d_model=128, n_heads=4, n_kv_heads=2, d_expert=192,
                n_experts=4, top_k=2, n_loops=3, mem_dim=64, engram_buckets=4099, engram_dim=32)
    base.update(kw)
    return RXConfig(**base)


def base_tasks():
    return make_suite(BASE_SKILLS, VOCAB, start_slot=0)


def new_skill_names():
    return STREAMS[STREAM]


def new_tasks():
    return make_suite(new_skill_names(), VOCAB, start_slot=len(BASE_SKILLS))


def eval_sets(tasks, n=256):
    return {t.name: EvalSet(t, n) for t in tasks}


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_base(path=None) -> RecursionX:
    path = path or os.path.join(RUNS, "base.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = RecursionX(RXConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["state"])
    return model


def pretrain(steps=3000, seed=0, out=None, lr=3e-3, batch=64, log_every=250, device="cpu", **cfg_kw):
    seed_all(seed)
    cfg = model_config(seed=seed, **cfg_kw)
    model = RecursionX(cfg).to(device)
    tasks = base_tasks()
    evs = eval_sets(tasks)
    mix = Mixture(tasks, seed=seed)
    print(f"params: {model.num_parameters():,} (w/o engram table "
          f"{model.num_parameters(exclude_engram=True):,})", flush=True)
    t0 = time.time()
    curve = []

    def log(step):
        accs = evaluate_all(model, evs)
        curve.append({"step": step, **accs})
        print(f"  eval @{step} ({time.time() - t0:.0f}s) " +
              " ".join(f"{k}={v:.2f}" for k, v in accs.items()), flush=True)

    train_loop(model, list(model.parameters()), mix.sample_seqs, steps, lr=lr, batch=batch,
               input_weight=INPUT_WEIGHT, log_every=log_every, callback=log)
    out = out or os.path.join(RUNS, "base.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"config": cfg.to_dict(), "state": model.state_dict(), "curve": curve}, out)
    return model, curve
