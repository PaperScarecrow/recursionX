"""Pre-train Recursion-X on real text (GPU-ready; also runs on CPU for smoke tests).

  # 1) tokenise a corpus once
  python pretrain_text.py prepare --data /path/to/texts --out ../runs/text/corpus.bin
  # 2) train
  python pretrain_text.py train --bin ../runs/text/corpus.bin --preset small \
      --device cuda --dtype bf16 --batch 32 --seq-len 1024 --grad-accum 4 --steps 20000

Features: preset configs, bf16/fp16 autocast, gradient accumulation, cosine
LR with warmup, periodic validation loss, sample generation, checkpoints and
resume, optional ``torch.compile``.  See ``docs/roadmap/05_scale_up.md`` for
what is still missing at scale (fused MoE kernels, KV cache, FSDP).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recursionx import RecursionX, RXConfig  # noqa: E402
from recursionx.data.text import TokenBin, build_bin, load_tokenizer  # noqa: E402
from recursionx.presets import count_params, preset  # noqa: E402
from recursionx.train import weighted_ce  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def lr_at(step, total, base, warmup, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def val_loss(model, data: TokenBin, batches: int, batch: int, device, ctx) -> float:
    model.eval()
    tot = 0.0
    for _ in range(batches):
        inp, tgt, w = data.batch(batch, device)
        with ctx():
            out = model(inp)
        tot += weighted_ce(out.logits, tgt, w).item()
    model.train()
    return tot / batches


def cmd_prepare(args):
    tok = load_tokenizer(args.tokenizer)
    tr, va = build_bin(args.data, args.out, tok, args.val_frac)
    meta = json.load(open(args.out + ".meta.json"))
    print(f"wrote {tr} ({meta['n_train']:,} tokens) and {va} ({meta['n_val']:,} tokens), "
          f"tokenizer={meta['tokenizer']} vocab={meta['vocab_size']}")


def cmd_train(args):
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    meta = json.load(open(args.bin + ".meta.json"))
    vocab = int(math.ceil(meta["vocab_size"] / 64) * 64)
    dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    ctx = (lambda: torch.autocast(device.type, dtype=dtype)) if dtype else contextlib.nullcontext
    os.makedirs(args.out, exist_ok=True)
    ck_path = os.path.join(args.out, "last.pt")
    start = 0
    if args.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = RXConfig.from_dict(ck["config"])
        start = ck["step"]
    else:
        overrides = json.loads(args.overrides) if args.overrides else {}
        cfg = preset(args.preset, vocab, max_seq_len=max(args.seq_len, 256), **overrides)
        ck = None
    print("config:", json.dumps({k: v for k, v in cfg.to_dict().items() if k != "extra"}))
    print("params:", {k: f"{v:,}" for k, v in count_params(cfg).items()}, flush=True)
    model = RecursionX(cfg).to(device)
    if ck is not None:
        model.load_state_dict(ck["state"])
    train = TokenBin(args.bin, args.seq_len, seed=args.seed)
    val = TokenBin(args.bin + ".val", args.seq_len, seed=args.seed + 1) if meta["n_val"] > args.seq_len + 1 else None
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "engram.table" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "engram.table" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    if ck is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    scaler = torch.amp.GradScaler(enabled=args.dtype == "fp16")
    fwd = torch.compile(model) if args.compile else model
    tok = load_tokenizer(meta["tokenizer"])
    model.train()
    t0, hist = time.time(), []
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.grad_accum):
            inp, tgt, w = train.batch(args.batch, device)
            with ctx():
                out = fwd(inp)
                loss = (weighted_ce(out.logits, tgt, w) + out.aux_loss) / args.grad_accum
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        hist.append(tot)
        if (step + 1) % args.log_every == 0:
            tps = args.batch * args.seq_len * args.grad_accum * args.log_every / (time.time() - t0)
            print(f"step {step + 1}/{args.steps} loss {sum(hist[-args.log_every:]) / args.log_every:.4f}"
                  f" lr {opt.param_groups[0]['lr']:.2e} tok/s {tps:,.0f}", flush=True)
            t0 = time.time()
        if val is not None and (step + 1) % args.eval_every == 0:
            vl = val_loss(model, val, args.eval_batches, args.batch, device, ctx)
            print(f"  val loss {vl:.4f} (ppl {math.exp(min(vl, 20)):.1f})", flush=True)
            prompt = torch.tensor([[tok.bos_id]], device=device)
            sample = model.generate(prompt, args.sample_tokens, temperature=0.8)
            print("  sample:", repr(tok.decode(sample[0, 1:].tolist())[:300]), flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save({"config": cfg.to_dict(), "state": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "tokenizer": meta["tokenizer"]}, ck_path)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="byte")
    p.add_argument("--val-frac", type=float, default=0.01)
    t = sub.add_parser("train")
    t.add_argument("--bin", required=True)
    t.add_argument("--preset", default="small")
    t.add_argument("--overrides", default="", help='JSON dict of RXConfig overrides')
    t.add_argument("--device", default="auto")
    t.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--steps", type=int, default=10000)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--sample-tokens", type=int, default=128)
    t.add_argument("--ckpt-every", type=int, default=1000)
    t.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "runs", "text"))
    t.add_argument("--resume", action="store_true")
    t.add_argument("--compile", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
