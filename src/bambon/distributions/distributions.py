# src/bambon/distributions/distributions.py
from __future__ import annotations
import random
from dataclasses import dataclass, field
from ..utils.helpers import beta_sample, clamp, dirichlet_sample
from ..models.models import BaseModel
from typing import Dict, List, Optional, Set, Any, Tuple, Literal, Union  # Tuple, Literal, Union

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

    # -------- ACCEPT --------
    def accept(
        self,
        x: float,
        *,
        objective: Literal["maximize", "minimize", "eq"] = "eq",
        eps: float = 0.0,
        risk: float = 0.05,
        use_credible_rule: bool = True,
        credible_mass: float = 0.90,
        mc: int = 3000,
    ) -> bool:
        """
        Accept a numeric proposal x (IN INTERNAL UNITS) if it's statistically good enough.

        - objective="eq" or use_credible_rule=True:
            accept if x lies inside a 'credible_mass' credible interval.
        - objective in {"maximize","minimize"} and use_credible_rule=False:
            accept if P(meaningfully better draw) <= risk, where "meaningfully better"
            is x+eps (maximize) or x-eps (minimize).
        """
        # Map x to [0,1]
        if self.U == self.L:
            z = 0.5
            span = 1.0
        else:
            z = self._to_unit(x)
            span = (self.U - self.L)

        # Credible interval rule
        if use_credible_rule or objective == "eq":
            samples = sorted(beta_sample(self.alpha, self.beta) for _ in range(mc))
            lo = _percentile(samples, (1 - credible_mass) / 2)
            hi = _percentile(samples, 1 - (1 - credible_mass) / 2)
            return lo <= z <= hi

        # Probability-of-improvement rule
        eps_unit = 0.0 if span == 0 else (eps / span)
        better = 0
        if objective == "maximize":
            thresh = min(1.0, z + eps_unit)
            for _ in range(mc):
                if beta_sample(self.alpha, self.beta) > thresh:
                    better += 1
        else:  # minimize
            thresh = max(0.0, z - eps_unit)
            for _ in range(mc):
                if beta_sample(self.alpha, self.beta) < thresh:
                    better += 1
        p_improve = better / max(1, mc)
        return p_improve <= risk


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

    # -------- ACCEPT --------
    def accept(
        self,
        label: Union[str, Set[str]],
        *,
        risk: float = 0.05,
        mc: int = 3000,
        min_margin: float = 0.0,
    ) -> Tuple[bool, float]:
        """
        If `label` is str: returns (accept?, p_best) with p_best = P(label is global argmax).
        If `label` is set[str]: returns (accept?, p_in_set) with p_in_set = P(global argmax ∈ set).
        """
        # Seed universe if empty
        if not self.labels:
            if isinstance(label, set):
                for l in label:
                    s = str(l)
                    self.labels.add(s)
                    self.alpha.setdefault(s, 1.0)
            else:
                s = str(label)
                self.labels.add(s)
                self.alpha.setdefault(s, 1.0)
            return True, 1.0

        # Single-label case
        if not isinstance(label, set):
            lab = str(label)
            if lab not in self.labels:
                self.labels.add(lab)
                self.alpha.setdefault(lab, 1.0)

            labs = sorted(self.labels)
            alphas = [self.alpha.get(l, 1.0) for l in labs]

            wins = 0
            margin_sum = 0.0
            for _ in range(mc):
                theta = dirichlet_sample(alphas)
                i = max(range(len(labs)), key=lambda k: theta[k])
                if labs[i] == lab:
                    wins += 1
                srt = sorted(theta, reverse=True)
                margin_sum += (srt[0] - srt[1]) if len(srt) > 1 else 1.0

            p_best = wins / max(1, mc)
            avg_margin = margin_sum / max(1, mc)
            return (p_best >= 1 - risk and avg_margin >= min_margin, p_best)

        # Set-of-labels case
        cand = {str(x) for x in label}
        for l in cand:
            if l not in self.labels:
                self.labels.add(l)
                self.alpha.setdefault(l, 1.0)

        labs = sorted(self.labels)
        alphas = [self.alpha.get(l, 1.0) for l in labs]
        cand_idx = {labs.index(l) for l in cand if l in labs}

        wins_in = 0
        for _ in range(mc):
            theta = dirichlet_sample(alphas)
            i = max(range(len(labs)), key=lambda k: theta[k])  # global argmax
            if i in cand_idx:
                wins_in += 1

        p_in_set = wins_in / max(1, mc)
        return (p_in_set >= 1 - risk, p_in_set)


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

    # -------- ACCEPT --------
    def accept(
        self,
        S: Set[str],
        *,
        tau_in: float = 0.6,
        tau_out: float = 0.6,
        require_all_in_confident: bool = True,
        require_all_out_confident: bool = False,
        jaccard_min: Optional[float] = None,
        mc: int = 2000,
    ) -> Tuple[bool, Dict[str, float], Optional[float]]:
        """
        Accept proposed set S if:
          1) each included item e has inclusion prob pi[e] >= tau_in
          2) (optional) each excluded item e has pi[e] <= 1 - tau_out
          3) (optional) expected Jaccard with a posterior sample >= jaccard_min

        Returns (accept?, {item: pi}, expected_jaccard|None).
        """
        universe = set(self.alpha) | set(self.beta) | set(S)
        for e in universe:
            self.ensure(e)

        pi: Dict[str, float] = {e: self.alpha[e] / (self.alpha[e] + self.beta[e]) for e in universe}

        if require_all_in_confident:
            for e in S:
                if pi[e] < tau_in:
                    return (False, pi, None)

        if require_all_out_confident:
            for e in (universe - S):
                if pi[e] > (1 - tau_out):
                    return (False, pi, None)

        if jaccard_min is not None:
            j_sum = 0.0
            for _ in range(mc):
                S_hat = {e for e in universe if random.random() < pi[e]}
                if not (S or S_hat):
                    j = 1.0
                else:
                    inter = len(S & S_hat)
                    union = len(S | S_hat)
                    j = inter / union if union else 1.0
                j_sum += j
            ej = j_sum / max(1, mc)
            if ej < jaccard_min:
                return (False, pi, ej)
            return (True, pi, ej)

        return (True, pi, None)
