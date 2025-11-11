# src/bambon/distributions/distributions.py
from __future__ import annotations
import random
from dataclasses import dataclass, field
from ..utils.helpers import beta_sample, clamp, dirichlet_sample
from ..models.models import BaseModel
from typing import Dict, List, Optional, Set, Any, Tuple, Literal, Union  # add Tuple, Literal, Union

def _percentile(sorted_vals: List[float], q: float) -> float:
    q = clamp(q, 0.0, 1.0)
    if not sorted_vals:
        return 0.5
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


@dataclass
class NumericBeta(BaseModel):
    """Numeric belief with adaptive bounds (expands as new extremes are observed)."""
    L: float
    U: float
    alpha: float = 1.0
    beta: float = 1.0

    def _to_unit(self, x: float) -> float:
        if self.U == self.L:
            return 0.5
        return clamp((x - self.L) / (self.U - self.L), 0.0, 1.0)

    def _from_unit(self, z: float) -> float:
        return self.L + z * (self.U - self.L)

    def observe(self, x: float, lr: float = 1.0) -> None:
        span = (self.U - self.L) or 1.0
        if x < self.L:
            self.L = x - 0.1 * span
        if x > self.U:
            self.U = x + 0.1 * span
        z = self._to_unit(x)
        self.alpha += lr * z
        self.beta += lr * (1.0 - z)

    def sample(self, inertia_val: Optional[float] = None, inertia: float = 0.0) -> float:
        if inertia_val is not None and random.random() < inertia:
            return inertia_val
        z = beta_sample(self.alpha, self.beta)
        return self._from_unit(z)

@dataclass
class EnumDirichlet(BaseModel):
    """Categorical belief over string labels via a Dirichlet prior."""
    labels: Set[str] = field(default_factory=set)
    alpha: Dict[str, float] = field(default_factory=dict)


    def _obs_one(self, label: str, lr: float):
        if label not in self.labels:
            self.labels.add(label)
            self.alpha.setdefault(label, 1.0)
        self.alpha[label] = self.alpha.get(label, 1.0) + lr

    def observe(self, label: str | set, lr: float = 1.0) -> None:
        if isinstance(label, set):
            for item in label:
                self._obs_one(item, lr)
        else:
            self._obs_one(label, lr)


    def sample(self, inertia_val: Optional[str] = None, inertia: float = 0.0) -> str:
        if inertia_val is not None and random.random() < inertia:
            return inertia_val
        labs = sorted(self.labels) or ["__undef__"]
        vec = [self.alpha.get(l, 1.0) for l in labs]
        p = dirichlet_sample(vec)
        return labs[max(range(len(labs)), key=lambda i: p[i])]


@dataclass
class SetBelief(BaseModel):
    """Belief over sets of strings; independent Beta–Bernoulli per item. No unseen introduction."""
    alpha: Dict[str, float] = field(default_factory=dict)
    beta: Dict[str, float] = field(default_factory=dict)

    def ensure(self, item: str) -> None:
        self.alpha.setdefault(item, 1.0)
        self.beta.setdefault(item, 1.0)

    def observe(self, present_items: Set[str], lr: float = 1.0) -> None:
        seen = set(self.alpha.keys()) | set(self.beta.keys()) | set(present_items)
        for e in seen:
            self.ensure(e)
            if e in present_items:
                self.alpha[e] += lr
            else:
                self.beta[e] += lr

    def sample(
        self,
        prev_set: Set[str],
        tau: float = 0.5,
        inertia: float = 0.0,
        force_overlap_from: Optional[Set[str]] = None,
        min_overlap: int = 1,
        max_k: Optional[int] = None,
        intersection_first_with: Optional[Set[str]] = None,
        intersection_min_jaccard: float = 0.85,
    ) -> Set[str]:
        # Snap to intersection if close enough
        if intersection_first_with is not None:
            A, B = set(prev_set), set(intersection_first_with)
            j = (len(A & B) / len(A | B)) if (A or B) else 1.0
            if j >= intersection_min_jaccard and B:
                S = set(A & B)
                # ensure known before ranking
                for e in S:
                    self.ensure(e)
                if max_k and len(S) > max_k:
                    S = set(
                        sorted(
                            S,
                            key=lambda e: self.alpha[e] / (self.alpha[e] + self.beta[e]),
                            reverse=True,
                        )[:max_k]
                    )
                return S

        S: Set[str] = set()
        for e in prev_set:
            if random.random() < inertia:
                self.ensure(e)
                S.add(e)

        all_items = set(self.alpha.keys()) | set(self.beta.keys())
        if force_overlap_from:
            all_items |= set(force_overlap_from)
        all_items |= set(prev_set)
        for e in list(all_items):
            self.ensure(e)

        for e in all_items:
            p = beta_sample(self.alpha[e], self.beta[e])
            if p > tau:
                S.add(e)

        # Enforce minimum overlap with a reference set
        if force_overlap_from:
            need = max(0, min_overlap - len(S & force_overlap_from))
            if need > 0:
                ranked = sorted(
                    [e for e in force_overlap_from if e not in S],
                    key=lambda x: self.alpha.get(x, 1.0) / (self.alpha.get(x, 1.0) + self.beta.get(x, 1.0)),
                    reverse=True,
                )
                for e in ranked[:need]:
                    self.ensure(e)
                    S.add(e)

        if max_k and len(S) > max_k:
            S = set(
                sorted(
                    S,
                    key=lambda x: self.alpha[x] / (self.alpha[x] + self.beta[x]),
                    reverse=True,
                )[:max_k]
            )
        return S
