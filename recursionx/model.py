"""The Recursion-X language model.

    tokens ─► embed ─► (+ Engram n-gram memory) ─► prelude blocks
          ─► (+ Titans neural memory)  = e
          ─► h = e;  repeat R times:  h = core( inject(h, e) + loop_emb[r] )   ◄─ weight-tied
          ─► coda blocks ─► norm ─► tied LM head

* prelude / core / coda follows the "prelude-recurrent-coda" layout of
  depth-recurrent LMs (Huginn, Ouro); the core is weight-tied and iterated, with
  the prelude output re-injected at every step so the recurrence cannot
  drift away from the input.
* every block is ``x + mixer(norm x)`` then ``x + FluidMoE(norm x)`` where the
  mixer is a liquid mixer or attention (LFM2-style hybrid ratio).
* the number of loops can be sampled during training and chosen freely (or
  adaptively, by convergence of ``h``) at inference: test-time compute scaling.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RXConfig
from .modules.attention import Attention
from .modules.engram import EngramMemory
from .modules.liquid import LiquidMixer
from .modules.lora import AdaptableLinear
from .modules.moe import FluidMoE
from .modules.neural_memory import MemState, NeuralMemory


class Block(nn.Module):
    def __init__(self, kind: str, cfg: RXConfig):
        super().__init__()
        self.kind = kind
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        if kind == "liquid":
            self.mixer = LiquidMixer(cfg.d_model, cfg.conv_kernel, cfg.scan_chunk, cfg.init_std)
        elif kind == "attn":
            self.mixer = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.init_std)
        else:
            raise ValueError(kind)
        self.ffn = FluidMoE(cfg.d_model, cfg.d_expert, cfg.n_experts, cfg.top_k,
                            cfg.n_shared_experts, cfg.router_temperature, cfg.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


@dataclass
class RXOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor
    loops: int
    memory_state: Optional[MemState] = None
    hidden: Optional[torch.Tensor] = None


class RecursionX(nn.Module):
    def __init__(self, cfg: RXConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.embed.weight, std=cfg.init_std)
        self.engram = (EngramMemory(d, cfg.engram_orders, cfg.engram_heads, cfg.engram_buckets,
                                    cfg.engram_dim, cfg.engram_storage, cfg.engram_path,
                                    cfg.init_std, cfg.seed) if cfg.use_engram else None)
        self.prelude = nn.ModuleList([Block(k, cfg) for k in cfg.prelude_layers])
        self.memory = (NeuralMemory(d, cfg.mem_dim, cfg.mem_chunk, cfg.mem_bias,
                                    cfg.mem_huber_delta, cfg.mem_max_lr, cfg.mem_max_decay,
                                    cfg.init_std) if cfg.use_neural_memory else None)
        self.mem_norm = nn.RMSNorm(d)
        self.inject = AdaptableLinear(2 * d, d, init_std=cfg.init_std)
        self.inject_norm = nn.RMSNorm(d)
        self.loop_emb = nn.Embedding(max(cfg.max_loops, cfg.n_loops) + 1, d)
        nn.init.normal_(self.loop_emb.weight, std=cfg.init_std)
        self.core = nn.ModuleList([Block(k, cfg) for k in cfg.core_layers])
        self.coda = nn.ModuleList([Block(k, cfg) for k in cfg.coda_layers])
        self.norm = nn.RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, std=cfg.init_std)

    # ------------------------------------------------------------------ utils
    def moe_layers(self):
        return [m for m in self.modules() if isinstance(m, FluidMoE)]

    def num_parameters(self, exclude_engram: bool = False) -> int:
        n = 0
        for name, p in self.named_parameters():
            if exclude_engram and name.startswith("engram.table"):
                continue
            n += p.numel()
        return n

    def sample_loops(self) -> int:
        c = self.cfg
        if self.training and c.loop_sampling == "uniform":
            return random.randint(c.min_loops, c.max_loops)
        return c.n_loops

    # ---------------------------------------------------------------- forward
    def forward(self, tokens: torch.Tensor, n_loops: Optional[int] = None,
                memory_state: Optional[MemState] = None, return_hidden: bool = False) -> RXOutput:
        cfg = self.cfg
        x = self.embed(tokens)
        if self.engram is not None:
            x = x + self.engram(tokens, x)
        for blk in self.prelude:
            x = blk(x)
        new_state = None
        if self.memory is not None:
            m, new_state = self.memory(self.mem_norm(x), memory_state)
            x = x + m
        e = x
        R = n_loops if n_loops is not None else self.sample_loops()
        R = max(1, min(R, self.loop_emb.num_embeddings - 1))
        h = e
        used = 0
        prev = None
        grad_from = R - cfg.bptt_loops if (cfg.bptt_loops and self.training) else 0
        for r in range(R):
            if r < grad_from:
                h = h.detach()
            h = self.inject(torch.cat([self.inject_norm(h), e], -1)) + self.loop_emb.weight[r]
            for blk in self.core:
                h = blk(h)
            used = r + 1
            if cfg.exit_tol > 0 and not self.training and n_loops is None:
                if prev is not None:
                    rel = (h - prev).norm() / h.norm().clamp_min(1e-6)
                    if rel < cfg.exit_tol:
                        break
                prev = h
        for blk in self.coda:
            h = blk(h)
        hidden = self.norm(h)
        logits = self.lm_head(hidden)
        aux = torch.zeros((), device=x.device)
        for m in self.moe_layers():
            aux = aux + m.last_aux
            m.last_aux = torch.zeros(())  # don't keep graph references on the module
        return RXOutput(logits, cfg.router_aux_coef * aux, used, new_state,
                        hidden if return_hidden else None)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, temperature: float = 0.0,
                 stop_token: Optional[int] = None, n_loops: Optional[int] = None) -> torch.Tensor:
        """Simple (non-cached) sampling; fine for the small models used here."""
        out = prompt
        done = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
        for _ in range(max_new):
            logits = self(out, n_loops=n_loops).logits[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(-1)
            if stop_token is not None:
                nxt = torch.where(done, torch.full_like(nxt, stop_token), nxt)
                done |= nxt == stop_token
            out = torch.cat([out, nxt.unsqueeze(-1)], 1)
            if stop_token is not None and bool(done.all()):
                break
        return out
