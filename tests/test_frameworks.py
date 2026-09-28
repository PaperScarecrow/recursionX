"""Tests for the roadmap frameworks: HyperLoRA, research loop, text pipeline,
presets, expert growth."""
import os
import random

import torch

from recursionx import DualHemisphereBrain, LifecycleConfig, RecursionX, RXConfig, SkillRecord
from recursionx.data.tasks import SKILLS, Mixture, SkillTask, Vocab, make_suite
from recursionx.data.text import ByteTokenizer, TokenBin, build_bin
from recursionx.modules.lora import active_adapters, adaptable_modules
from recursionx.presets import count_params, preset
from recursionx.research import (ConsistencyVerifier, EpisodeTask, ExecutionVerifier, OracleSource,
                                 ProgramSource, ResearchLoop, ResearchResult, SkillRequest,
                                 TeacherLLMSource, parse_examples, run_verifiers)
from recursionx.skills.hyperlora import HyperLoRA, demo_features, meta_train
from recursionx.train import EvalSet, train_loop

V = Vocab(8)


def tiny_model(**kw):
    torch.manual_seed(0)
    base = dict(vocab_size=V.size, d_model=32, n_heads=4, n_kv_heads=2, d_expert=48, n_experts=3,
                mem_dim=16, engram_buckets=257, engram_dim=8, n_loops=2)
    base.update(kw)
    return RecursionX(RXConfig(**base))


# ------------------------------------------------------------------ HyperLoRA
def test_hyperlora_grads_and_materialize_equivalence():
    model = tiny_model().eval()
    task = make_suite(["succ"], V, max_len=5)[0]
    rng = random.Random(0)
    desc = demo_features(model, [task.sample(rng) for _ in range(4)])
    hyper = HyperLoRA(model, desc.numel(), rank=4, alpha=8, hidden=32)
    for h in hyper.heads.values():               # make the generated delta non-trivial
        torch.nn.init.normal_(h.weight, std=0.05)
    lora = hyper(desc)
    assert set(lora) == set(hyper.names) and all(".experts." not in n for n in lora)
    hyper.apply(model, lora)
    x = EvalSet(task, 4).inp
    out = model(x).logits
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in hyper.parameters())
    ref = out.detach()
    hyper.clear(model)
    hyper.materialize(model, "gen", {k: (A.detach(), B.detach()) for k, (A, B) in lora.items()})
    with torch.no_grad(), active_adapters(model, {"gen": 1.0}):
        assert torch.allclose(model(x).logits, ref, atol=1e-4)


def test_hyperlora_meta_train_reduces_loss():
    model = tiny_model()
    tasks = make_suite(["succ", "reverse"], V, max_len=5)
    desc_dim = demo_features(model, [tasks[0].sample(random.Random(0))]).numel()
    hyper = HyperLoRA(model, desc_dim, rank=4, alpha=8, hidden=32)
    before = [p.detach().clone() for p in model.parameters()]
    losses = meta_train(model, hyper, tasks, steps=30, k_demos=4, batch=16, lr=3e-3)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))  # base frozen


# -------------------------------------------------------------- research loop
def test_program_source_and_execution_verifier():
    fn = SKILLS["reverse"]
    req = SkillRequest("reverse", task_token=10, n_symbols=8, n_examples=40)
    good = ProgramSource(fn).research(req, random.Random(0))
    bad = ResearchResult([([1, 2, 3], [1, 2, 3])], "bad")   # wrong: not reversed
    ok, reports = run_verifiers([good, bad], [ExecutionVerifier(fn, 8)])
    assert reports[0]["rejected"] == 1 and all(y == x[::-1] for x, y in ok)


def test_consistency_verifier_reports_conflicts():
    a = ResearchResult([([1, 2], [2, 1]), ([3], [3])], "src_a")
    b = ResearchResult([([1, 2], [1, 2])], "src_b")        # disagrees on [1, 2]
    ok, rep = ConsistencyVerifier()([a, b])
    assert ok == [([3], [3])] and rep["conflicts"] == 1
    assert rep["syndrome"][0]["sources"] == ["src_a", "src_b"]


def test_teacher_source_parses_json():
    text = 'Sure! [{"input": [1, 2], "output": [2, 1]}, {"input": [9], "output": [1]}, {"x": 1}]'
    assert parse_examples(text, n_symbols=8) == [([1, 2], [2, 1])]  # 9 out of range, junk dropped
    src = TeacherLLMSource(lambda prompt: text)
    res = src.research(SkillRequest("rev", task_token=10, n_symbols=8), random.Random(0))
    assert res.examples == [([1, 2], [2, 1])] and "Skill: rev" in res.provenance["prompt"]


def test_research_loop_builds_record_and_brain_learns():
    torch.manual_seed(0)
    model = tiny_model()
    base = make_suite(["copy", "reverse"], V, max_len=5)
    train_loop(model, list(model.parameters()), Mixture(base).sample_seqs, 40, batch=32,
               input_weight=0.1)
    cfg = LifecycleConfig(rank=4, wake_steps=30, rem_steps=5, rem_batch=8, dream_pool=8,
                          gate_min_acc=0.0, gate_min_gain=-1.0, replay_per_skill=16,
                          sleep_pressure=99, probes_per_skill=16)
    brain = DualHemisphereBrain(model, cfg)
    brain.register_base_skills([SkillRecord(t.name, t) for t in base])
    fn = SKILLS["succ"]
    loop = ResearchLoop([ProgramSource(fn)], [ExecutionVerifier(fn, 8), ConsistencyVerifier()],
                        min_examples=32, seed=0)
    req = SkillRequest("succ", task_token=12, n_symbols=8, min_len=3, max_len=5, n_examples=64)
    rec, report = loop.run(req)
    assert report.ok and isinstance(rec.task, EpisodeTask) and rec.probes
    probe_set = {tuple(p) for p in rec.probes}
    assert not probe_set & {tuple(s) for s in rec.task.seqs}   # probes are held out
    rep = brain.research_and_ingest(req, loop)
    assert rep["accepted"] and rep["provenance"]["sources"] == ["program"]
    failing = ResearchLoop([ProgramSource(lambda x, n: 1 / 0)], [], min_examples=8)
    assert brain.research_and_ingest(SkillRequest("broken", task_token=13, n_symbols=8),
                                     failing)["reason"] == "research failed"


# ---------------------------------------------------------------- text / presets
def test_byte_tokenizer_and_token_bin(tmp_path):
    tok = ByteTokenizer()
    s = "héllo, Recursion-X ✓"
    assert tok.decode(tok.encode(s, bos=True, eos=True)) == s
    (tmp_path / "a.txt").write_text("the quick brown fox " * 50)
    (tmp_path / "b.jsonl").write_text('{"text": "jumps over the lazy dog"}\n' * 20)
    tr, va = build_bin([str(tmp_path)], str(tmp_path / "corpus.bin"), tok, val_frac=0.1)
    data = TokenBin(tr, seq_len=32)
    inp, tgt, w = data.batch(4)
    assert inp.shape == (4, 32) and torch.equal(inp[:, 1:], tgt[:, :-1]) and w.sum() > 0


def test_presets_count_on_meta_device():
    for name in ("tiny", "small"):
        c = count_params(preset(name, vocab_size=320))
        assert c["unique_params"] > c["core_params"] > 0
        assert c["active_params_per_token_per_pass"] < c["unique_params"]


# ------------------------------------------------------------------- growth
def test_sleep_expert_growth_adds_capacity():
    torch.manual_seed(0)
    model = tiny_model()
    base = make_suite(["copy", "reverse"], V, max_len=5)
    train_loop(model, list(model.parameters()), Mixture(base).sample_seqs, 40, batch=32,
               input_weight=0.1)
    cfg = LifecycleConfig(rank=4, wake_steps=20, rem_steps=5, rem_batch=8, dream_pool=8,
                          gate_min_acc=0.0, gate_min_gain=-1.0, replay_per_skill=16,
                          sleep_pressure=99, grow_experts=True, growth_acc_threshold=1.01,
                          growth_extra_steps=3, commit_check=False)
    brain = DualHemisphereBrain(model, cfg)
    brain.register_base_skills([SkillRecord(t.name, t) for t in base])
    new = make_suite(["succ"], V, start_slot=5, max_len=5)[0]
    val = EvalSet(new, 16)
    brain.ingest(SkillRecord("succ", new, new_tokens=[new.task_token]), val)
    n_before = [m.n_experts for m in brain.awake.moe_layers()]
    report = brain.sleep(val_sets={"succ": val, **{t.name: EvalSet(t, 16) for t in base}})
    assert report["grown_for"] == ["succ"]
    assert [m.n_experts for m in brain.awake.moe_layers()] == [n + 1 for n in n_before]
    assert brain.logits(val.inp).shape[0] == 16
