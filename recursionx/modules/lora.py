"""Adaptable linear layers and *projected* LoRA skill adapters.

Every weight matrix in Recursion-X that can learn new skills is an
:class:`AdaptableLinear`.  It holds

* the slow base weight ``W`` (only changed during *sleep* consolidation),
* a bank of named low-rank skill adapters (changed during *wake*),
* a *protected input subspace* ``U`` (orthonormal columns) that summarises the
  inputs this layer has already been consolidated on.

A **projected LoRA** computes ``ΔW = s · B A (I - U Uᵀ)``: it can only react to
input directions that the consolidated knowledge does *not* live in, so
merging it into ``W`` leaves the layer's response to old inputs (approximately)
unchanged.  This is the LoRA analogue of Gradient Projection Memory (GPM) /
InfLoRA.  In addition the adapter's ``A`` matrix is initialised from the top
principal directions of the *new skill's* activations inside that free
subspace ("data-projected init"), so the adapter starts out pointed at the
skill data before any gradient step is taken.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Weight = Union[float, torch.Tensor]


class LoRAAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float,
                 protected_basis: Optional[torch.Tensor] = None,
                 init_A: Optional[torch.Tensor] = None):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        if init_A is None:
            init_A = torch.randn(rank, in_features) / math.sqrt(in_features)
        self.A = nn.Parameter(init_A.clone().float())
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        # Snapshot of the protected subspace at creation time.  Later growth of
        # the layer's subspace must not silently change a trained adapter.
        if protected_basis is None:
            protected_basis = torch.zeros(in_features, 0)
        self.register_buffer("U", protected_basis.detach().clone())

    @property
    def projected(self) -> bool:
        return self.U.shape[1] > 0

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.projected:
            return x
        return x - (x @ self.U) @ self.U.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(self.project_input(x), self.A), self.B) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        A = self.A
        if self.projected:
            A = A - (A @ self.U) @ self.U.t()
        return self.scaling * self.B @ A


class AdaptableLinear(nn.Module):
    """``nn.Linear`` + named skill adapters + protected subspace + activation stats."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 init_std: Optional[float] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        std = init_std if init_std is not None else 1.0 / math.sqrt(in_features)
        nn.init.normal_(self.weight, std=std)
        self.adapters = nn.ModuleDict()
        self.register_buffer("protected_basis", torch.zeros(in_features, 0))
        # Mixing weights of active adapters: name -> float or (B, 1, 1) tensor.
        self.active: Dict[str, Weight] = {}
        # A *generated* low-rank delta (A, B, scale) supplied as plain tensors,
        # e.g. by a HyperLoRA; shared by the whole batch, differentiable w.r.t.
        # whatever produced it.
        self.generated: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None
        # Activation statistics (for GPM subspaces and data-projected init).
        self.collect_stats = False
        self._cov: Optional[torch.Tensor] = None
        self._n = 0

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.collect_stats:
            self._accumulate(x)
        for name, w in self.active.items():
            ad = self.adapters[name] if name in self.adapters else None
            if ad is None:
                continue
            y = y + w * ad(x)
        if self.generated is not None:
            A, B, scale = self.generated
            y = y + scale * F.linear(F.linear(x, A.to(x.dtype)), B.to(x.dtype))
        return y

    def _accumulate(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, self.in_features).double()
        if flat.shape[0] == 0:
            return
        if self._cov is None:
            self._cov = torch.zeros(self.in_features, self.in_features, dtype=torch.float64,
                                    device=x.device)
        self._cov += flat.t() @ flat
        self._n += flat.shape[0]

    def reset_stats(self) -> None:
        self._cov, self._n = None, 0

    def covariance(self) -> Optional[torch.Tensor]:
        if self._cov is None or self._n == 0:
            return None
        return (self._cov / self._n).float()

    # ----------------------------------------------------------------- adapters
    def free_projector(self) -> torch.Tensor:
        U = self.protected_basis
        eye = torch.eye(self.in_features, device=U.device)
        return eye - U @ U.t() if U.shape[1] else eye

    def add_adapter(self, name: str, rank: int, alpha: float, projected: bool = True,
                    data_cov: Optional[torch.Tensor] = None) -> LoRAAdapter:
        U = self.protected_basis if projected else None
        init_A = None
        if data_cov is not None:
            init_A = data_projected_init(data_cov, U, rank)
        ad = LoRAAdapter(self.in_features, self.out_features, rank, alpha, U, init_A)
        ad.to(self.weight.device)
        self.adapters[name] = ad
        return ad

    def remove_adapter(self, name: str) -> None:
        if name in self.adapters:
            del self.adapters[name]
        self.active.pop(name, None)

    @torch.no_grad()
    def merge_adapter(self, name: str, weight: float = 1.0, remove: bool = True) -> None:
        self.weight.add_(weight * self.adapters[name].delta_weight().to(self.weight.dtype))
        if remove:
            self.remove_adapter(name)

    # ------------------------------------------------------ protected subspace
    @torch.no_grad()
    def extend_protected(self, cov: torch.Tensor, threshold: float = 0.97,
                         max_frac: float = 0.9) -> int:
        """GPM-style subspace growth.  Adds the fewest principal directions of
        ``cov`` (outside the current basis) needed so that ``threshold`` of the
        activation energy lies inside the protected subspace.  Returns the
        number of added directions."""
        U = self.protected_basis
        total = torch.trace(cov).clamp_min(1e-12)
        P = self.free_projector()
        resid = P @ cov @ P
        covered = total - torch.trace(resid)
        evals, evecs = torch.linalg.eigh(resid)
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
        need = threshold * total - covered
        budget = int(max_frac * self.in_features) - U.shape[1]
        if need <= 0 or budget <= 0:
            return 0
        cum = torch.cumsum(evals, 0)
        m = int((cum < need).sum().item()) + 1
        m = min(m, budget)
        new = evecs[:, :m]
        # re-orthonormalise against the old basis for numerical safety
        if U.shape[1]:
            new = new - U @ (U.t() @ new)
        new, _ = torch.linalg.qr(new)
        self.protected_basis = torch.cat([U, new.to(U.dtype)], dim=1)
        return m


def data_projected_init(cov: torch.Tensor, U: Optional[torch.Tensor], rank: int) -> torch.Tensor:
    """Top-``rank`` principal directions of ``cov`` inside the free subspace."""
    d = cov.shape[0]
    if U is not None and U.shape[1]:
        P = torch.eye(d, device=cov.device) - U @ U.t()
        cov = P @ cov @ P
    evals, evecs = torch.linalg.eigh(cov.float())
    A = evecs.flip(1)[:, :rank].t().contiguous()  # (rank, d), orthonormal rows
    if evals.flip(0)[:rank].sum() <= 1e-10:  # nothing left: fall back to random
        A = torch.randn(rank, d) / math.sqrt(d)
        if U is not None and U.shape[1]:
            A = A - (A @ U) @ U.t()
    return A


# --------------------------------------------------------------------------- #
# model-level helpers
# --------------------------------------------------------------------------- #
def adaptable_modules(model: nn.Module) -> Iterator[Tuple[str, AdaptableLinear]]:
    for name, m in model.named_modules():
        if isinstance(m, AdaptableLinear):
            yield name, m


def add_skill_adapter(model: nn.Module, skill: str, rank: int, alpha: float,
                      projected: bool = True,
                      covs: Optional[Mapping[str, torch.Tensor]] = None) -> list:
    params = []
    for name, m in adaptable_modules(model):
        ad = m.add_adapter(skill, rank, alpha, projected=projected,
                           data_cov=None if covs is None else covs.get(name))
        params.extend(ad.parameters())
    return params


def adapter_parameters(model: nn.Module, skill: str) -> list:
    params = []
    for _, m in adaptable_modules(model):
        if skill in m.adapters:
            params.extend(m.adapters[skill].parameters())
    return params


def set_active_adapters(model: nn.Module, weights: Mapping[str, Weight]) -> None:
    for _, m in adaptable_modules(model):
        m.active = {k: v for k, v in weights.items() if k in m.adapters}


@contextmanager
def active_adapters(model: nn.Module, weights: Mapping[str, Weight]):
    old = {n: dict(m.active) for n, m in adaptable_modules(model)}
    set_active_adapters(model, weights)
    try:
        yield
    finally:
        for n, m in adaptable_modules(model):
            m.active = old[n]


def list_skills(model: nn.Module) -> list:
    names = []
    for _, m in adaptable_modules(model):
        for k in m.adapters.keys():
            if k not in names:
                names.append(k)
    return names


def merge_skill(model: nn.Module, skill: str, weight: float = 1.0) -> None:
    for _, m in adaptable_modules(model):
        if skill in m.adapters:
            m.merge_adapter(skill, weight)


def remove_skill(model: nn.Module, skill: str) -> None:
    for _, m in adaptable_modules(model):
        m.remove_adapter(skill)


@contextmanager
def collecting_stats(model: nn.Module):
    mods = [m for _, m in adaptable_modules(model)]
    for m in mods:
        m.reset_stats()
        m.collect_stats = True
    try:
        yield
    finally:
        for m in mods:
            m.collect_stats = False


def gather_covariances(model: nn.Module) -> Dict[str, torch.Tensor]:
    out = {}
    for name, m in adaptable_modules(model):
        c = m.covariance()
        if c is not None:
            out[name] = c
        m.reset_stats()
    return out


def extend_protected_subspaces(model: nn.Module, covs: Mapping[str, torch.Tensor],
                               threshold: float = 0.97, max_frac: float = 0.9) -> Dict[str, int]:
    added = {}
    for name, m in adaptable_modules(model):
        if name in covs:
            added[name] = m.extend_protected(covs[name], threshold, max_frac)
    return added


def protected_fraction(model: nn.Module) -> float:
    tot, used = 0, 0
    for _, m in adaptable_modules(model):
        tot += m.in_features
        used += m.protected_basis.shape[1]
    return used / max(tot, 1)


@torch.no_grad()
def project_gradients(model: nn.Module, strength: float = 1.0) -> None:
    """GPM: remove the gradient component that acts on protected input directions."""
    for _, m in adaptable_modules(model):
        g = m.weight.grad
        U = m.protected_basis
        if g is None or U.shape[1] == 0:
            continue
        g.sub_(strength * (g @ U) @ U.t())
