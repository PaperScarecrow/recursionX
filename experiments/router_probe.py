"""Router diagnostic: separate routing error from adapter quality.

Wake-only run (no sleep): learn all 6 new skills as adapters, then for each
skill's eval set compare:
  served  = router-chosen adapter (what user gets)
  oracle  = forced-correct adapter (adapter quality ceiling)
  base    = no adapter (how much adapter helps)
routing_loss = oracle - served  (how much router misroutes)

Also logs full confusion matrix (true skill -> routed skill) and prototype
cosine geometry.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import common
from common import base_tasks, new_tasks, eval_sets, load_base, save_json, seed_all, RUNS, INPUT_WEIGHT
from recursionx import LifecycleConfig, SkillRecord
from recursionx.lifecycle.wake import WakeLearner
from recursionx.lifecycle.skills import build_prototype
from recursionx.lifecycle.brain import DualHemisphereBrain
from recursionx.modules.lora import active_adapters
from recursionx.train import evaluate

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
    for t in nt:
        rec = SkillRecord(t.name, t, new_tokens=[t.task_token])
        rep = brain.ingest(rec, evs[t.name], anchors={}, auto_sleep=False)
        print(f"wake {t.name}: accepted={rep.get('accepted')} acc={rep.get('acc_skill',0):.3f}", flush=True)
    # confusion + oracle gap
    model = brain.awake
    skills = [t.name for t in bt + nt]
    conf = {}
    gaps = {}
    for t in bt + nt:
        es = evs[t.name]
        served = evaluate(model, es, forward=brain.logits)["acc"]
        with active_adapters(model, {}):
            base_acc = evaluate(model, es)["acc"]
        adapter = brain.router.adapter_of.get(t.name)
        if adapter:
            with active_adapters(model, {adapter: 1.0}):
                oracle = evaluate(model, es)["acc"]
        else:
            oracle = base_acc
        gaps[t.name] = {"served": served, "oracle": oracle, "base": base_acc,
                        "routing_loss": oracle - served}
        # where do its prompts go?
        from collections import Counter
        routes = brain.router.route(model, es.inp[:256].to(next(model.parameters()).device))
        # true label is t.name; count routed adapter -> skill name reverse map
        inv = {v: k for k, v in brain.router.adapter_of.items()}
        # routes are adapter names/None; map to skill via inv (None -> base)
        cnt = Counter()
        for r in routes:
            cnt[inv.get(r, "base") if r else "base"] += 1
        conf[t.name] = dict(cnt)
        print(f"{t.name}: served={served:.2f} oracle={oracle:.2f} base={base_acc:.2f} routes={dict(cnt)}", flush=True)
    # prototype geometry
    import torch.nn.functional as F
    names = list(brain.router.exemplars)
    protos = torch.stack([brain.router.exemplars[n].float().mean(0) for n in names])
    protos = protos / protos.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sim = (protos @ protos.T).tolist()
    out = {"gaps": gaps, "confusion": conf, "names": names,
           "cosine": {names[i]: {names[j]: round(sim[i][j], 3) for j in range(len(names))} for i in range(len(names))}}
    save_json(out, os.path.join(RUNS, "router_probe", f"probe_s{args.seed}.json"))
    print("saved runs/router_probe/probe_s0.json" if args.seed == 0 else f"saved probe_s{args.seed}.json")

if __name__ == "__main__":
    main()
