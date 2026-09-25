import random

import torch

from recursionx import DualHemisphereBrain, LifecycleConfig, RecursionX, RXConfig, SkillRecord
from recursionx.data.tasks import FactTask, Mixture, SEP, Vocab, make_suite
from recursionx.modules.lora import list_skills
from recursionx.train import EvalSet, evaluate, train_loop

V = Vocab(8)


def make_brain(**lc):
    torch.manual_seed(0)
    cfg = RXConfig(vocab_size=V.size, d_model=32, n_heads=4, n_kv_heads=2, d_expert=48,
                   n_experts=3, mem_dim=16, engram_buckets=257, engram_dim=8, n_loops=2)
    model = RecursionX(cfg)
    base = make_suite(["copy", "reverse"], V, max_len=5)
    train_loop(model, list(model.parameters()), Mixture(base).sample_seqs, 60, batch=32,
               input_weight=0.1)
    opts = dict(rank=4, wake_steps=40, rem_steps=10, rem_batch=8, dream_pool=16,
                gate_min_acc=0.0, gate_min_gain=-1.0, replay_per_skill=16, sleep_pressure=2)
    opts.update(lc)
    lcfg = LifecycleConfig(**opts)
    brain = DualHemisphereBrain(model, lcfg)
    brain.register_base_skills([SkillRecord(t.name, t) for t in base])
    return brain, base


def test_wake_sleep_cycle_bakes_and_resets():
    brain, base = make_brain()
    new = make_suite(["succ", "rotl"], V, start_slot=5, max_len=5)
    anchors = {t.name: EvalSet(t, 32) for t in base}
    first_awake = brain.awake
    r0 = SkillRecord(new[0].name, new[0], new_tokens=[new[0].task_token])
    rep = brain.ingest(r0, EvalSet(new[0], 32), anchors)
    assert rep["accepted"] and list_skills(brain.awake) == ["succ"]
    # the awake hemisphere routes the new instruction to its adapter
    inp = EvalSet(new[0], 4).inp
    assert brain.router.route(brain.awake, inp) == ["succ"] * 4
    assert brain.router.route(brain.awake, EvalSet(base[0], 4).inp) == [None] * 4
    r1 = SkillRecord(new[1].name, new[1], new_tokens=[new[1].task_token])
    brain.ingest(r1, EvalSet(new[1], 32), anchors)  # sleep pressure reached -> sleeps
    assert brain.cycles == 1 and brain.awake is not first_awake
    assert list_skills(brain.awake) == [] and brain.pending == []
    assert all(brain.catalog[n].status == "consolidated" for n in ("succ", "rotl"))
    assert brain.router.route(brain.awake, inp) == [None] * 4  # served by the base now
    assert brain.history[-1]["event"] == "sleep"
    # both hemispheres share one Engram store
    assert brain.awake.engram is brain.asleep.engram


def test_background_sleep_keeps_serving():
    brain, base = make_brain(sleep_pressure=99)
    new = make_suite(["succ"], V, start_slot=5, max_len=5)
    val = EvalSet(new[0], 16)
    brain.ingest(SkillRecord("succ", new[0], new_tokens=[new[0].task_token]), val)
    brain.sleep(background=True)
    served = brain.logits(val.inp)  # the awake hemisphere answers while the other sleeps
    assert served.shape[0] == 16
    report = brain.wake_up()
    assert report["cycle"] == 1 and not brain.is_sleeping


def test_dreams_are_well_formed():
    brain, base = make_brain()
    dreams = brain.sleeper.dream(brain.awake, [t.task_token for t in base], 12)
    for d in dreams:
        assert d[1] in {t.task_token for t in base} and d.count(SEP) == 1


def test_fact_ingestion_only_touches_engram():
    brain, _ = make_brain()
    facts = FactTask("facts", V, n_facts=8)
    before = {n: p.detach().clone() for n, p in brain.awake.named_parameters()
              if not n.startswith("engram.table")}
    rec = SkillRecord("facts", facts, kind="fact")
    brain.waker.cfg.fact_steps = 30
    rep = brain.ingest(rec, EvalSet(facts))
    assert rep["kind"] == "fact"
    for n, p in brain.awake.named_parameters():
        if n in before:
            assert torch.equal(before[n], p), n
