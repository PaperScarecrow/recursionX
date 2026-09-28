"""Architecture-level experiments (independent of the wake/sleep lifecycle).

  ablation  – train variants from scratch on the base suite with an equal step
              budget: full model / no Engram / no Titans memory / attention-only
              (no liquid) / no looping (R=1)
  depth     – train with randomised loop counts, then evaluate the *same*
              weights at R = 1..8 (test-time compute scaling)
  recall    – associative recall over a long context with a liquid-only
              backbone, with and without the Titans neural memory
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch

from common import INPUT_WEIGHT, RUNS, VOCAB, base_tasks, eval_sets, model_config, save_json, seed_all
from recursionx import RecursionX
from recursionx.data.tasks import BOS, EOS, SEP, Mixture
from recursionx.train import EvalSet, evaluate, evaluate_all, train_loop

VARIANTS = {
    "full": {},
    "no_engram": {"use_engram": False},
    "no_titans": {"use_neural_memory": False},
    "attn_only": {"prelude_layers": ("attn",), "core_layers": ("attn", "attn")},
    "no_loop": {"n_loops": 1},
}


def run_ablation(steps, variants, seed):
    tasks = base_tasks()
    evs = eval_sets(tasks)
    out = {}
    for name in variants:
        seed_all(seed)
        model = RecursionX(model_config(seed=seed, **VARIANTS[name]))
        curve = []
        t0 = time.time()

        def cb(step):
            accs = evaluate_all(model, evs)
            curve.append({"step": step, "avg": sum(accs.values()) / len(accs), **accs})
            print(f"  [{name}] @{step} avg={curve[-1]['avg']:.3f}", flush=True)

        train_loop(model, list(model.parameters()), Mixture(tasks, seed=seed).sample_seqs, steps,
                   lr=3e-3, batch=64, input_weight=INPUT_WEIGHT, log_every=max(steps // 6, 1),
                   callback=cb)
        out[name] = {"curve": curve, "params": model.num_parameters(exclude_engram=True),
                     "seconds": time.time() - t0}
        save_json(out, os.path.join(RUNS, "architecture", f"ablation_s{seed}.json"))
    return out


def run_depth(steps, seed):
    tasks = base_tasks()
    evs = eval_sets(tasks)
    seed_all(seed)
    model = RecursionX(model_config(seed=seed, loop_sampling="uniform", min_loops=1, max_loops=4,
                                    n_loops=4))
    train_loop(model, list(model.parameters()), Mixture(tasks, seed=seed).sample_seqs, steps,
               lr=3e-3, batch=64, input_weight=INPUT_WEIGHT, log_every=max(steps // 5, 1))
    res = {}
    for R in range(1, 9):
        accs = evaluate_all(model, evs, n_loops=R)
        res[R] = {"avg": sum(accs.values()) / len(accs), **accs}
        print(f"  loops={R}: avg={res[R]['avg']:.3f}", flush=True)
    model.cfg.exit_tol = 0.02
    model.eval()
    with torch.no_grad():
        used = [model(es.inp).loops for es in evs.values()]
    res["adaptive_exit_loops"] = used
    res["adaptive_exit_acc"] = evaluate_all(model, evs)
    save_json(res, os.path.join(RUNS, "architecture", f"depth_s{seed}.json"))
    return res


class RecallTask:
    """[BOS, T, k1 v1 ... kn vn, q, SEP] -> v(q).  Keys drawn from the first
    half of the symbols, values from the second half."""
    name = "recall"

    def __init__(self, n_pairs=12, V=VOCAB):
        self.n, self.V = n_pairs, V
        self.task_token = 5

    def sample(self, rng):
        half = self.V.n_symbols // 2
        keys = rng.sample(range(half), min(self.n, half))
        vals = [rng.randrange(half, self.V.n_symbols) for _ in keys]
        q = rng.randrange(len(keys))
        body = [t for k, v in zip(keys, vals) for t in (self.V.sym(k), self.V.sym(v))]
        return [BOS, self.task_token] + body + [self.V.sym(keys[q]), SEP, self.V.sym(vals[q]), EOS]


def run_recall(steps, seed):
    task = RecallTask(n_pairs=8)
    rng = random.Random(seed)
    es = EvalSet(task, 512)
    out = {}
    for name, kw in {"liquid_only": {"use_neural_memory": False},
                     "liquid_only+titans": {"use_neural_memory": True, "mem_conv": 4}}.items():
        seed_all(seed)
        model = RecursionX(model_config(seed=seed, prelude_layers=("liquid",),
                                        core_layers=("liquid",), coda_layers=("liquid",),
                                        use_engram=False, n_loops=1, mem_chunk=4, **kw))
        curve = []

        def cb(step):
            curve.append({"step": step, "acc": evaluate(model, es)["acc"]})
            print(f"  [{name}] @{step} acc={curve[-1]['acc']:.3f}", flush=True)

        train_loop(model, list(model.parameters()), lambda n: [task.sample(rng) for _ in range(n)],
                   steps, lr=3e-3, batch=64, log_every=max(steps // 6, 1), callback=cb)
        out[name] = curve
    save_json(out, os.path.join(RUNS, "architecture", f"recall_s{seed}.json"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ablation,depth,recall")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for e in args.exp.split(","):
        print(f"=== {e} ===", flush=True)
        if e == "ablation":
            run_ablation(args.steps, args.variants.split(","), args.seed)
        elif e == "depth":
            run_depth(args.steps, args.seed)
        elif e == "recall":
            run_recall(args.steps, args.seed)
