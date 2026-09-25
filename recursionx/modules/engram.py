"""Engram: hashed n-gram lookup memory with host/disk offloadable tables.

For every position the last ``n`` tokens (for each order in ``orders``) are
hashed by several independent multiplicative-XOR hash heads into a large
embedding table.  The retrieved rows are fused into the residual stream
through a *context-aware gate* (dot product between the current hidden state
and a key projection of the retrieved memory), so the lookup is ignored when
it does not fit the context.

Lookups depend only on token ids, never on hidden states, so the rows needed
for a sequence are known before the forward pass and the table can live in
host RAM or on disk (``storage="disk"`` uses a ``numpy.memmap``) with only the
touched rows paged in.  Training the disk table uses sparse, row-local SGD
(:meth:`EngramMemory.apply_sparse_grads`), which is also what makes it a
low-interference place to write new facts during wake.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class DiskTable:
    """Row store backed by ``numpy.memmap``; gathers return autograd leaves."""

    def __init__(self, path: str, rows: int, dim: int, init_std: float = 0.02,
                 seed: int = 0, create: bool = True):
        self.path, self.rows, self.dim = path, rows, dim
        exists = os.path.exists(path)
        mode = "r+" if exists else "w+"
        self.mm = np.memmap(path, dtype=np.float32, mode=mode, shape=(rows, dim))
        if not exists and create:
            rng = np.random.default_rng(seed)
            self.mm[:] = rng.normal(0, init_std, size=(rows, dim)).astype(np.float32)
            self.mm.flush()
        self.pending: List[Tuple[np.ndarray, torch.Tensor]] = []
        self.rows_read = 0

    def gather(self, idx: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        uniq, inv = torch.unique(idx.cpu(), return_inverse=True)
        u = uniq.numpy()
        rows = torch.from_numpy(np.array(self.mm[u]))
        self.rows_read += len(u)
        if requires_grad:
            rows.requires_grad_(True)
            self.pending.append((u, rows))
        return rows[inv].to(idx.device)

    def sgd_step(self, lr: float) -> int:
        n = 0
        for u, rows in self.pending:
            if rows.grad is not None:
                self.mm[u] = (rows.detach() - lr * rows.grad).numpy()
                n += len(u)
        self.pending.clear()
        self.mm.flush()
        return n

    def load_all(self) -> torch.Tensor:
        return torch.from_numpy(np.array(self.mm))

    def write_all(self, t: torch.Tensor) -> None:
        self.mm[:] = t.detach().cpu().numpy().astype(np.float32)
        self.mm.flush()


class EngramMemory(nn.Module):
    def __init__(self, d_model: int, orders=(2, 3), heads: int = 2, buckets: int = 4099,
                 dim: int = 32, storage: str = "memory", path: Optional[str] = None,
                 init_std: float = 0.02, seed: int = 0):
        super().__init__()
        self.orders, self.heads, self.buckets, self.dim = tuple(orders), heads, buckets, dim
        self.n_tables = len(self.orders) * heads
        rows = self.n_tables * buckets
        g = torch.Generator().manual_seed(seed + 1234)
        maxn = max(self.orders)
        mult = torch.randint(1, 2 ** 30, (self.n_tables, maxn), generator=g) * 2 + 1
        self.register_buffer("mult", mult, persistent=True)
        self.register_buffer("offsets", torch.arange(self.n_tables) * buckets, persistent=True)
        self.storage = storage
        if storage == "memory":
            self.table = nn.Parameter(torch.randn(rows, dim, generator=g) * init_std)
            self.disk = None
        elif storage == "disk":
            assert path is not None, "disk storage needs engram_path"
            self.table = None
            self.disk = DiskTable(path, rows, dim, init_std, seed)
        else:
            raise ValueError(storage)
        width = self.n_tables * dim
        self.key = nn.Linear(width, d_model, bias=False)
        self.value = nn.Linear(width, d_model, bias=False)
        nn.init.normal_(self.key.weight, std=init_std)
        nn.init.normal_(self.value.weight, std=init_std)
        self.h_norm = nn.RMSNorm(d_model)
        self.k_norm = nn.RMSNorm(d_model)
        self.scale = d_model ** -0.5

    # --------------------------------------------------------------- hashing
    def indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, T, n_tables) table rows."""
        B, T = tokens.shape
        maxn = max(self.orders)
        padded = torch.cat([tokens.new_zeros(B, maxn - 1), tokens], 1).long() + 1
        # windows[..., i] = token at position t - i
        windows = torch.stack([padded[:, maxn - 1 - i: maxn - 1 - i + T] for i in range(maxn)], -1)
        out = []
        t = 0
        for n in self.orders:
            for _ in range(self.heads):
                h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
                for i in range(n):
                    h = h ^ (windows[..., i] * self.mult[t, i])
                out.append(h % self.buckets)
                t += 1
        return torch.stack(out, -1) + self.offsets

    def lookup(self, tokens: torch.Tensor) -> torch.Tensor:
        idx = self.indices(tokens)
        if self.disk is not None:
            rows = self.disk.gather(idx.reshape(-1), requires_grad=self.training and torch.is_grad_enabled())
            rows = rows.view(*idx.shape, self.dim)
        else:
            rows = self.table[idx]
        return rows.flatten(-2)

    def forward(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        e = self.lookup(tokens).to(h.dtype)
        k = self.key(e)
        gate = torch.sigmoid((self.h_norm(h) * self.k_norm(k)).sum(-1, keepdim=True) * self.scale)
        return gate * self.value(e)

    # ------------------------------------------------------------- storage
    def apply_sparse_grads(self, lr: float) -> int:
        return self.disk.sgd_step(lr) if self.disk is not None else 0

    def to_disk(self, path: str) -> None:
        """Move an in-memory table to a disk memmap (e.g. before serving)."""
        if self.disk is not None:
            return
        rows = self.table.shape[0]
        if os.path.exists(path):
            os.remove(path)
        self.disk = DiskTable(path, rows, self.dim, create=False)
        self.disk.write_all(self.table.data)
        self.table = None
        self.storage = "disk"

    def to_memory(self) -> None:
        if self.disk is None:
            return
        self.table = nn.Parameter(self.disk.load_all())
        self.disk = None
        self.storage = "memory"
