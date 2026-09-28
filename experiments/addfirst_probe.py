"""Protection-wall probe: is projected wake what blocks add_first late?

Builds the p1 prefix (rotl, pred, swap each slept singly = 3 commits),
then wakes add_first twice from the same base: projected vs plain LoRA.
If plain >> projected, the GPM subspace (not capacity) is the wall.
"""
import sys, os, argparse, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import common
from common import base_tasks, new_tasks, eval_sets, load_base, seed_all, RUNS, INPUT_WEIGHT
from recursionx import LifecycleConfig, SkillRecord
from recursionx.lifecycle.brain import DualHemisphereBrain
from recursionx.modules.lora import protected_fraction

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    base = load_base().to(args.device)
    bt, nt = base_tasks(), new_tasks()
    byname = {t.name: t for t in bt + nt}
    evs = eval_sets(bt + nt)
    cfg = LifecycleConfig(sleep_pressure=1)
    brain = DualHemisphereBrain(base, cfg, seed=args.seed, input_weight=INPUT_WEIGHT)
    brain.register_base_skills([SkillRecord(t.name, t) for t in bt])
    for name in ["rotl", "pred", "swap_pairs"]:
        t = byname[name]
        rec = SkillRecord(t.name, t, new_tokens=[t.task_token])
        rep = brain.ingest(rec, evs[t.name], anchors={}, auto_sleep=False)
        print(f"wake {name}: {rep.get('acc_skill',0):.3f}", flush=True)
        srep = brain.sleep(val_sets=evs, log_every=0)
        print(f"sleep {name}: committed={srep.get('committed')} {srep.get('audit',{}).get('syndrome')}", flush=True)
    print(f"protected_fraction={protected_fraction(brain.awake):.3f}", flush=True)
    t = byname["add_first"]
    # projected attempt
    rec1 = SkillRecord(t.name, t, new_tokens=[t.task_token])
    rep1 = brain.ingest(rec1, evs[t.name], anchors={}, auto_sleep=False)
    print(f"add_first PROJECTED: accepted={rep1.get('accepted')} acc={rep1.get('acc_skill',0):.3f}", flush=True)
    # plain attempt from same base state (fresh adapter slot under new name to avoid clash)
    from recursionx.lifecycle.wake import WakeLearner
    from recursionx.train import evaluate
    import copy as _copy
    model_plain = copy.deepcopy(brain.awake)
    # strip the projected adapter just added so plain starts clean
    from recursionx.modules.lora import remove_skill
    remove_skill(model_plain, t.name)
    cfg_plain = LifecycleConfig(projected=False, data_init=False)
    waker = WakeLearner(cfg_plain, args.seed)
    rec2 = SkillRecord(t.name + "_plain", t, new_tokens=[t.task_token])
    # learn under a stand-in name then eval with it forced on
    rec2.name = t.name
    waker.learn_skill(model_plain, rec2)
    from recursionx.modules.lora import active_adapters as _aa
    with _aa(model_plain, {t.name: 1.0}):
        acc_plain = evaluate(model_plain, evs[t.name])["acc"]
    print(f"add_first PLAIN: acc={acc_plain:.3f}", flush=True)
    with open(os.path.join(RUNS, "router_probe", "addfirst_wall.txt"), "w") as f:
        f.write(f"projected={rep1.get('acc_skill',0):.3f} plain={acc_plain:.3f} "
                f"prot={protected_fraction(brain.awake):.3f}\n")

if __name__ == "__main__":
    main()
