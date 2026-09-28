"""Ceiling probe: is the router or the adapter the bottleneck?

Wake-learns new skills (no sleep), then serves the same adapters three ways:
  ridge   = SkillRouter (current: mean+last prelude feats + ridge)
  tokid   = deterministic route on input[1] (the TASK_k token id)
  oracle  = forced-correct adapter
If tokid >> ridge, features are the problem (task signal exists, feats wash it).
If tokid ~= ridge, the skills genuinely overlap and we need capacity/routing reform.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import common
from common import base_tasks, new_tasks, eval_sets, load_base, save_json, seed_all, RUNS, INPUT_WEIGHT
from recursionx import LifecycleConfig, SkillRecord
from recursionx.lifecycle.brain import DualHemisphereBrain
from recursionx.modules.lora import active_adapters
from recursionx.train import evaluate


def serve_tokid(brain, token_to_adapter):
    model = brain.awake
    def forward(inp):
        dev = next(model.parameters()).device
        inp_d = inp.to(dev)
        routes = [token_to_adapter.get(int(t), None) for t in inp_d[:, 1].tolist()]
        out = None
        for name in sorted({r for r in routes}, key=lambda r: (r is not None, r or "")):
            idx = torch.tensor([i for i, r in enumerate(routes) if r == name], device=inp_d.device)
            with active_adapters(model, {} if name is None else {name: 1.0}):
                lg = model(inp_d[idx]).logits
            if out is None:
                out = lg.new_zeros(inp_d.shape[0], *lg.shape[1:])
            out[idx.to(lg.device)] = lg
        return out
    return forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wake-steps", type=int, default=600)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    base = load_base().to(args.device)
    bt, nt = base_tasks(), new_tasks()
    evs = eval_sets(bt + nt)
    cfg = LifecycleConfig(wake_steps=args.wake_steps)
    brain = DualHemisphereBrain(base, cfg, seed=args.seed, input_weight=INPUT_WEIGHT)
    brain.register_base_skills([SkillRecord(t.name, t) for t in bt])
    tok2ad = {}
    for t in nt:
        rec = SkillRecord(t.name, t, new_tokens=[t.task_token])
        rep = brain.ingest(rec, evs[t.name], anchors={}, auto_sleep=False)
        tok2ad[t.task_token] = t.name
        print(f"wake {t.name}: acc={rep.get('acc_skill',0):.3f}", flush=True)
    model = brain.awake
    tokid_fwd = serve_tokid(brain, tok2ad)
    rows = {}
    for t in bt + nt:
        es = evs[t.name]
        served = evaluate(model, es, forward=brain.logits)["acc"]
        with active_adapters(model, {}):
            base_acc = evaluate(model, es)["acc"]
        ad = brain.router.adapter_of.get(t.name)
        if ad:
            with active_adapters(model, {ad: 1.0}):
                oracle = evaluate(model, es)["acc"]
        else:
            oracle = base_acc
        tokid = evaluate(model, es, forward=tokid_fwd)["acc"]
        rows[t.name] = {"ridge": round(served, 3), "tokid": round(tokid, 3),
                        "oracle": round(oracle, 3), "base": round(base_acc, 3)}
        print(f"{t.name}: ridge={served:.2f} tokid={tokid:.2f} oracle={oracle:.2f}", flush=True)
    save_json(rows, os.path.join(RUNS, "router_probe", f"ceiling_s{args.seed}.json"))
    print("saved ceiling probe")


if __name__ == "__main__":
    main()
