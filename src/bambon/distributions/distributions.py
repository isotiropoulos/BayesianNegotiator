from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from ..utils.helpers import beta_sample, clamp, dirichlet_sample, _to_setish
from ..utils.distributions import _ctx_gain, _percentile, _unit_tail_prob, numeric_constraints_intersection
from ..models.models import BaseModel

from scipy.stats import beta
import numpy as np

# =====================================================================
#                     NUMERIC BETA (operator-aware)
# =====================================================================

@dataclass
class NumericBeta(BaseModel):
    """
    Numeric distribution over a scalar with adaptive bounds.

    You use it in three ways:

      - observe(): update belief from opponent offers
      - sample(): propose new numeric values given your operator
      - accept(): "Given my operator+value and belief, is their
                  operator+value OK for me?"
    """
    L: float
    U: float
    alpha: float = 1.0
    beta: float = 1.0

    # -------------------- internal mapping --------------------

    def _to_unit(self, x: float) -> float:
        if self.U == self.L:
            return 0.5
        return clamp((x - self.L) / (self.U - self.L), 0.0, 1.0)

    def _from_unit(self, z: float) -> float:
        return self.L + z * (self.U - self.L)

    # -------------------- learning from opp --------------------

    def observe(self, x: float, operator: str = "eq", lr: float = 1.0) -> None:
        span = (self.U - self.L) or 1.0

        # Expand bounds a bit to keep learning flexible
        if x < self.L:
            self.L = x - 0.1 * span
        if x > self.U:
            self.U = x + 0.1 * span

        z = self._to_unit(x)
        eps = 1e-9

        if operator == "eq":
            self.alpha += lr * z
            self.beta += lr * (1.0 - z)
            return

        if operator in ("gteq", "gt"):
            self.L = max(self.L, x + (eps if operator == "gt" else 0.0))
            self.alpha += lr * (1.0 - z) * 1.0
            self.beta += lr * z * 0.2
            return

        if operator in ("lteq", "lt"):
            self.U = min(self.U, x - (eps if operator == "lt" else 0.0))
            self.beta += lr * z * 1.0
            self.alpha += lr * (1.0 - z) * 0.2
            return

        if operator == "neq":
            if z <= 0.5:
                self.alpha += lr * (1.0 - z) * 0.7
                self.beta += lr * z * 0.1
            else:
                self.beta += lr * z * 0.7
                self.alpha += lr * (1.0 - z) * 0.1
            return

        # fallback ~ eq
        self.alpha += lr * z
        self.beta += lr * (1.0 - z)

    # -------------------- sampling proposals --------------------


    def _trunc_beta_unit(self, a, b, lo, hi):
        # CDF values at boundaries
        cdf_lo = beta.cdf(lo, a, b)
        cdf_hi = beta.cdf(hi, a, b)

        # uniform sample in truncated CDF region
        u = np.random.uniform(cdf_lo, cdf_hi)

        # invert CDF
        return beta.ppf(u, a, b)

    def sample(
        self,
        x: Optional[float] = None,
        operator: str = "eq",
        inertia_val: Optional[float] = None,
        inertia: float = 0.0,
    ) -> float:
        """
        Propose a numeric value given *your* operator.

        x:           anchor/threshold value in internal space (your constraint)
        operator:    'eq', 'neq', 'gteq', 'lteq', 'gt', 'lt'
        inertia_val: previous offer (internal) to stick to
        inertia:     probability to keep inertia_val
        """

        anchor = x
        if operator in ("eq", "neq") or anchor is None:
            z = beta_sample(self.alpha, self.beta)
            return self._from_unit(z)

        z_thr = self._to_unit(anchor)
        if operator in ("gteq", "gt"):
            z = self._trunc_beta_unit(self.alpha, self.beta, z_thr, 1.0)
            return self._from_unit(z)
        if operator in ("lteq", "lt"):
            z = self._trunc_beta_unit(self.alpha, self.beta, 0.0, z_thr)
            return self._from_unit(z)


        z = beta_sample(self.alpha, self.beta)
        return self._from_unit(z)

    # -------------------- belief-based acceptance --------------------

    @staticmethod
    def _effective_risk(
        base_risk: float,
        ctx_score: Optional[float],
        severity: Optional[float],
        max_risk: float = 0.9,
    ) -> float:
        """
        Adjust risk upwards (more willingness to accept) when
        global context is good and severity is low.
        """
        g = _ctx_gain(ctx_score, severity)
        return float(clamp(base_risk + g * (max_risk - base_risk), base_risk, max_risk))

    def _accept_single(
        self,
        x: float,
        operator: str = "eq",
        *,
        credible_mass: float = 0.90,
        risk: float = 0.2,
        mc: int = 3000,
        last_value: Optional[float] = None,
        sim_weight: float = 0.0,
        sim_scale: float = 0.25,
    ) -> bool:
        z = self._to_unit(x)

        draws = [beta_sample(self.alpha, self.beta) for _ in range(mc)]
        draws.sort()

        # similarity
        if last_value is not None:
            z_last = self._to_unit(last_value)
            dist = abs(z - z_last)
            sim_scale_eff = max(sim_scale, 1e-6)
            sim_conf = max(0.0, 1.0 - dist / sim_scale_eff)
        else:
            sim_conf = 0.5

        # base_conf
        if operator == "eq":
            lo = _percentile(draws, (1 - credible_mass) / 2)
            hi = _percentile(draws, 1 - (1 - credible_mass) / 2)
            base_conf = credible_mass if lo <= z <= hi else 0.0


        elif operator in ("gteq", "gt"):
            p_consistent = _unit_tail_prob(draws, z, side="ge")
            base_conf = p_consistent


        elif operator in ("lteq", "lt"):
            p_consistent = _unit_tail_prob(draws, z, side="le")
            base_conf = p_consistent

        elif operator == "neq":
            lo = _percentile(draws, (1 - credible_mass) / 2)
            hi = _percentile(draws, 1 - (1 - credible_mass) / 2)
            base_conf = 0.0 if lo <= z <= hi else 1.0

        else:
            lo = _percentile(draws, (1 - credible_mass) / 2)
            hi = _percentile(draws, 1 - (1 - credible_mass) / 2)
            base_conf = credible_mass if lo <= z <= hi else 0.0

        sim_weight = clamp(sim_weight, 0.0, 1.0)
        score = (1.0 - sim_weight) * base_conf + sim_weight * sim_conf
        return score >= 1.0 - risk

    def accept(
        self,
        my_op: str,
        my_val: float,
        opp_op: str,
        opp_val: float,
        *,
        last_value: Optional[float] = None,
        credible_mass: float = 0.90,
        risk: float = 0.2,
        mc: int = 3000,
        ctx_score: Optional[float] = None,
        severity: Optional[float] = None,
    ) -> bool:
        """
        Main high-level question:

            "Given my constraint (my_op, my_val) and my belief,
             is the incoming numeric constraint (opp_op, opp_val) OK for me?"

        Arguments:
          my_op, my_val:     your operator and value (in internal space)
          opp_op, opp_val:   opponent's operator and value (internal space)
          last_value:        your last offered numeric value (internal)
          ctx_score:         global context similarity for this action (0..1)
          severity:          global disagreement severity for this action (0..1)
        """

        # 1) semantic intersection
        compatible, interval = numeric_constraints_intersection(
            op_a=my_op,
            val_a=my_val,
            op_b=opp_op,
            val_b=opp_val,
        )
        if not compatible or interval is None:
            return False

        lo, hi = interval

        # 2) pick "worst acceptable" point for you within the intersection
        if my_op in ("gteq", "gt"):
            x_eff = lo  # smallest allowed still satisfies my gteq
        elif my_op in ("lteq", "lt"):
            x_eff = hi  # largest allowed still satisfies my lteq
        else:
            # eq / neq / others: just test at opp_val (or my_val)
            x_eff = opp_val

        eff_risk = self._effective_risk(risk, ctx_score, severity)

        return self._accept_single(
            x_eff,
            operator=my_op,
            credible_mass=credible_mass,
            risk=eff_risk,
            mc=mc,
            last_value=last_value,
            sim_weight=0.0,
        )


# =====================================================================
#                    ENUM DIRICHLET (operator-aware)
# =====================================================================

@dataclass
class EnumDirichlet(BaseModel):
    """
    Categorical belief over labels.

    Usage:

      - observe(label_or_set, opp_operator)
      - sample(operator=your_operator, ...)
      - accept(my_op, my_val, opp_op, opp_val)
    """
    labels: Set[str] = field(default_factory=set)
    alpha: Dict[str, float] = field(default_factory=dict)

    # -------------------- learning --------------------

    def _obs_one(self, label: str, lr: float) -> None:
        if label not in self.labels:
            self.labels.add(label)
            self.alpha.setdefault(label, 1.0)
        self.alpha[label] = self.alpha.get(label, 1.0) + lr

    def observe(self, x: str | Set[str], operator: str = "eq", lr: float = 1.0) -> None:

        if isinstance(x, set):
            S = {str(v) for v in x}

            if operator == "isAllOf":
                w = lr
                for item in S:
                    self._obs_one(item, w)
                return

            if operator == "isAnyOf":
                w = lr * 0.5
                for item in S:
                    self._obs_one(item, w)
                return

            if operator == "isNoneOf":
                if not self.labels:
                    for l in S:
                        self._obs_one(l, 0.0)
                    return

                for l in S:
                    self._obs_one(l, 0.0)

                others = [l for l in self.labels if l not in S]
                if not others:
                    return

                for l in others:
                    self.alpha[l] = self.alpha.get(l, 1.0) + lr * 0.2
                return

            for item in S:
                self._obs_one(item, lr)
            return

        # single label
        lab = str(x)

        if operator == "neq":
            if not self.labels:
                self._obs_one(lab, 0.0)
                return

            self._obs_one(lab, 0.0)
            for l in self.labels:
                if l != lab:
                    self.alpha[l] = self.alpha.get(l, 1.0) + lr * 0.2
            return

        self._obs_one(lab, lr)

    # -------------------- sampling --------------------

    def sample(
            self,
            x: Optional[Union[str, Set[str]]] = None,
            operator: str = "eq",
            inertia: float = 0.0,
            *,
            required: Optional[Set[str]] = None,
            max_k: Optional[int] = None,
    ) -> Union[str, Set[str]]:
        """
        Operator-aware sampling for enum / set-valued categorical issues.

        - eq: return a single label (high-prob)
        - isAnyOf: return a small set of strong labels (optionally including required)
        - isAllOf: must include required (or x if given as a set), fill up to k with strong labels
        - neq / isNoneOf: avoid forbidden labels (x), choose best outside
        """

        # inertia: keep previous candidate x
        if x is not None and random.random() < inertia:
            return x

        # current support
        labs = sorted(self.labels) or ["__undef__"]
        # ensure alpha exists
        for l in labs:
            self.alpha.setdefault(l, 1.0)

        # helper scores (posterior mean proxy)
        def score(l: str) -> float:
            a = self.alpha.get(l, 1.0)
            return a  # simple and monotone; you can also use a/(sum_alpha) but ranking is identical

        ranked = sorted(labs, key=score, reverse=True)

        # normalize required + forbidden
        req = {str(v) for v in (required or set())}

        forbidden: Set[str] = set()
        if operator in ("neq", "isNoneOf") and x is not None:
            if isinstance(x, (set, list, tuple)):
                forbidden = {str(v) for v in x}
            else:
                forbidden = {str(x)}

        # make sure unseen required/forbidden are known to the model
        for l in (req | forbidden):
            if l not in self.labels:
                self.labels.add(l)
            self.alpha.setdefault(l, 1.0)
            if l not in ranked:
                ranked.append(l)

        # choose size
        k = max_k or min(3, len(ranked))
        k = max(1, k)

        # -------------------------
        # eq: pick a single label
        # -------------------------
        if operator == "eq":
            # if required is a singleton, honor it
            if len(req) == 1:
                return next(iter(req))

            # otherwise pick highest-scoring (or sample from Dirichlet if you want stochasticity)
            return ranked[0]

        # ------------------------------------
        # neq / isNoneOf: pick outside forbidden
        # ------------------------------------
        if operator in ("neq", "isNoneOf"):
            # required must be respected even if it conflicts (your caller should avoid conflicting constraints)
            allowed_ranked = [l for l in ranked if l not in forbidden] or ranked

            # if returning a single value is acceptable for your pipeline, do it for neq
            if operator == "neq":
                # pick best allowed single label
                # if required exists, prefer it (as long as not forbidden)
                for l in allowed_ranked:
                    if (not req) or (l in req):
                        return l
                return allowed_ranked[0]

            # isNoneOf: return a set that excludes forbidden, but includes required
            out: Set[str] = set(req)
            for l in allowed_ranked:
                if l in out:
                    continue
                out.add(l)
                if len(out) >= k:
                    break

            # if required already fills k, that's fine
            return out

        # ------------------------------------
        # isAllOf: must include required (and/or x if x is a set)
        # ------------------------------------
        if operator == "isAllOf":
            base = set(req)
            if x is not None and isinstance(x, (set, list, tuple)):
                base |= {str(v) for v in x}

            # fill up with best labels (avoid duplicates)
            out: Set[str] = set(base)
            for l in ranked:
                if l not in out:
                    out.add(l)
                if len(out) >= k:
                    break
            return out

        # ------------------------------------
        # isAnyOf (and fallback): propose a small strong set + required
        # ------------------------------------
        if operator == "isAnyOf":
            out: Set[str] = set(req)
            for l in ranked:
                if l not in out:
                    out.add(l)
                if len(out) >= k:
                    break
            return out

        # fallback
        out: Set[str] = set(req)
        for l in ranked:
            if l not in out:
                out.add(l)
            if len(out) >= k:
                break
        return out

    # -------------------- acceptance of a given label/set --------------------

    @staticmethod
    def _effective_risk(
        base_risk: float,
        ctx_score: Optional[float],
        severity: Optional[float],
        max_risk: float = 0.9,
    ) -> float:
        g = _ctx_gain(ctx_score, severity)
        return float(clamp(base_risk + g * (max_risk - base_risk), base_risk, max_risk))

    def _accept_core(
        self,
        x: Union[str, Set[str]],
        operator: str = "eq",
        *,
        risk: float = 0.2,
        min_margin: float = 0.0,
        mc: int = 3000,
    ) -> Tuple[bool, float]:

        if not self.labels:
            S_seed = _to_setish(x)
            for l in S_seed:
                s = str(l)
                self.labels.add(s)
                self.alpha.setdefault(s, 1.0)
            return True, 1.0

        labs = sorted(self.labels)
        alphas = [self.alpha.get(l, 1.0) for l in labs]

        # eq(label)
        if operator == "eq" and isinstance(x, str):
            lab = str(x)
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

        S = _to_setish(x)

        # isAnyOf(S)
        if operator == "isAnyOf":
            cand_idx = set()
            for l in S:
                if l not in self.labels:
                    self.labels.add(l)
                    self.alpha.setdefault(l, 1.0)
            labs = sorted(self.labels)
            alphas = [self.alpha.get(l, 1.0) for l in labs]
            for l in S:
                if l in labs:
                    cand_idx.add(labs.index(l))

            wins_in = 0
            for _ in range(mc):
                theta = dirichlet_sample(alphas)
                i = max(range(len(labs)), key=lambda k: theta[k])
                if i in cand_idx:
                    wins_in += 1
            p_in = wins_in / max(1, mc)
            return (p_in >= 1 - risk, p_in)

        # isAllOf(S)
        if operator == "isAllOf":
            if not S:
                return True, 1.0
            per = []
            for lab in S:
                if lab not in self.labels:
                    self.labels.add(lab)
                    self.alpha.setdefault(lab, 1.0)
            labs = sorted(self.labels)
            alphas = [self.alpha.get(l, 1.0) for l in labs]
            for lab in S:
                wins = 0
                li = labs.index(lab)
                for _ in range(mc):
                    theta = dirichlet_sample(alphas)
                    i = max(range(len(labs)), key=lambda k: theta[k])
                    if i == li:
                        wins += 1
                per_lab = wins / max(1, mc)
                per.append(per_lab)
            score = min(per) if per else 1.0
            return (score >= 1 - risk, score)

        # neq / isNoneOf: P(argmax not in S)
        if operator in ("neq", "isNoneOf"):
            cand_idx = set()
            for l in S:
                if l not in self.labels:
                    self.labels.add(l)
                    self.alpha.setdefault(l, 1.0)
            labs = sorted(self.labels)
            alphas = [self.alpha.get(l, 1.0) for l in labs]
            for l in S:
                if l in labs:
                    cand_idx.add(labs.index(l))

            wins_in = 0
            for _ in range(mc):
                theta = dirichlet_sample(alphas)
                i = max(range(len(labs)), key=lambda k: theta[k])
                if i in cand_idx:
                    wins_in += 1
            p_in = wins_in / max(1, mc)
            p_out = 1.0 - p_in
            return (p_out >= 1 - risk, p_out)

        # fallback -> isAnyOf
        return self._accept_core(_to_setish(x), operator="isAnyOf", risk=risk, mc=mc)

    # -------------------- high-level: accept incoming constraint --------------------

    def accept(
        self,
        my_op: str,
        my_val: Union[str, Set[str]],
        opp_op: str,
        opp_val: Union[str, Set[str]],
        *,

        risk: float = 0.2,
        mc: int = 3000,
        ctx_score: Optional[float] = None,
        severity: Optional[float] = None,
    ) -> bool:
        """
        "Given my categorical constraint (my_op, my_val) and my belief,
         is their constraint (opp_op, opp_val) OK for me?"

        my_op, my_val:   your operator and labels (single or set)
        opp_op, opp_val: opponent operator and labels
        ctx_score:       global context similarity (0..1)
        severity:        global disagreement severity (0..1)
        """

        eff_risk = self._effective_risk(risk, ctx_score, severity)

        # We ask: P(argmax ∈ shared) high enough?
        ok, score = self._accept_core(
            opp_val, #shared,
            operator= opp_op, #"isAnyOf",
            risk=eff_risk,
            mc=mc,
        )
        return ok


# =====================================================================
#                    SET BELIEF (operator-aware)
# =====================================================================

@dataclass
class SetBelief(BaseModel):
    """
    Independent Beta–Bernoulli per item.

    Used for things like language/industry/purpose where the *actual value*
    is a set of strings.

    Usage:

      - observe(S, opp_op)
      - sample(x=prev_set, operator=my_op, ...)
      - accept(my_op, my_S, opp_op, opp_S)
    """
    alpha: Dict[str, float] = field(default_factory=dict)
    beta: Dict[str, float] = field(default_factory=dict)

    # -------------------- utilities --------------------

    def ensure(self, item: str) -> None:
        self.alpha.setdefault(item, 1.0)
        self.beta.setdefault(item, 1.0)

    # -------------------- learning --------------------

    def observe(self, x: Set[str] | str, operator: str = "isAllOf", lr: float = 1.0) -> None:
        S = _to_setish(x)

        seen = set(self.alpha.keys()) | set(self.beta.keys()) | set(S)
        for e in seen:
            self.ensure(e)

        if operator == "isAnyOf":
            for e in seen:
                if e in S:
                    self.alpha[e] += lr * 0.7
                else:
                    self.beta[e] += lr * 0.3

        elif operator in ("isNoneOf", "neq"):
            for e in seen:
                if e in S:
                    self.beta[e] += lr
                else:
                    self.alpha[e] += lr * 0.3

        else:  # isAllOf and eq
            for e in seen:
                if e in S:
                    self.alpha[e] += lr
                else:
                    self.beta[e] += lr * 0.1

    # -------------------- sampling --------------------

    def sample(
            self,
            x: Optional[Set[str]] = None,
            operator: str = "isAllOf",
            tau: List[float, float] = [0.5, 0.6],
            inertia: float = 0.0,
            *,
            required: Optional[Dict[str,Set[str]]] = None,
            max_k: Optional[int] = None,
    ) -> Set[str]:
        prev_set = x or set()
        req = set() if required else None
        forbid = set() if required else None

        # --- universe + ensure ---
        universe = set(self.alpha) | set(self.beta) | set(prev_set) # | set(req)
        if required:
            for value in required.values():
                universe |= value
        for e in universe:
            self.ensure(e)

        pi: Dict[str, float] = {e: self.alpha[e] / (self.alpha[e] + self.beta[e]) for e in universe}

        if required:
            for key, val in required.items():
                if key in ['eq', 'isAllOf']:
                    req |= val
                elif key == 'isAnyOf':
                    req |= set([i for i in val if pi[i] >= tau[0]])
                elif key in ['neq', 'isNoneOf']:
                    req -= val
                    forbid |= val
                else:
                    pass


        # --- inertia seed ---
        S: Set[str] = set()
        if inertia > 0.0 and prev_set:
            for e in prev_set:
                if random.random() < inertia:
                    S.add(e)

        S |= req
        S -= forbid

        # helper: enforce max_k by pi sorting
        def cap_by_pi_desc(T: Set[str]) -> Set[str] | None:
            if max_k and len(T) > max_k:
                return set(sorted(T, key=lambda u: pi[u], reverse=True)[:max_k])
            return T if len(T)>0 else None

        def cap_by_pi_asc(T: Set[str]) -> Set[str] | None:
            if max_k and len(T) > max_k:
                return set(sorted(T, key=lambda u: pi[u])[:max_k])
            return T if len(T)>0 else None

        # --- operator-aware construction ---
        if operator == "eq":
            # propose exactly one highly probable item
            candidates = universe - forbid if forbid else universe
            if not candidates:
                return set()
            best = max(candidates, key=lambda e: pi[e])
            return {best}

        if operator == "neq":
            # forbid exactly one unlikely item (safe-to-forbid)
            candidates = forbid if forbid else universe  # don't forbid required items
            worst = min(candidates, key=lambda e: pi[e]) if candidates else None
            return {worst} if worst is not None else set()

        if operator == "isAnyOf":
            # propose a small set of likely items; ensure non-empty
            # Start with inertia-retained items, then add those above tau
            for e in universe - forbid if forbid else universe:
                if pi[e] > tau[0]:
                    S.add(e)

            # cap and ensure non-empty
            S = cap_by_pi_desc(S)
            if not S and universe - forbid if forbid else universe:
                S = {max(universe, key=lambda e: pi[e])}
            return S

        if operator == "isAllOf":
            # propose all items that are sufficiently likely + required
            for e in universe  - forbid if forbid else universe:
                if pi[e] > tau[0]:
                    S.add(e)
            S = cap_by_pi_desc(S)
            if not S and universe - forbid if forbid else universe:
                S = {max(universe, key=lambda e: pi[e])}
            return S

        if operator == "isNoneOf":
            # propose items that are unlikely (safe to forbid), exclude required items
            # threshold uses tau_out: forbid e if pi[e] < 1 - tau_out
            for e in (forbid if forbid else universe):
                if pi[e] < (1.0 - tau[1]):
                    S.add(e)

            # if empty, forbid a small number of least-likely items (but never required)
            if not S:
                candidates = list(forbid if forbid else universe)
                if candidates:
                    # forbid one worst item (or more if you want; keep it minimal)
                    S = {min(candidates, key=lambda e: pi[e])}

            # cap by ascending pi (keep the least-likely items when trimming)
            S = cap_by_pi_asc(S)
            if not S and universe - forbid if forbid else universe:
                S = {min(universe, key=lambda e: pi[e])}
            return S

        # fallback: treat unknown operators like isAllOf
        for e in universe - forbid if forbid else universe:
            if pi[e] > tau[0]:
                S.add(e)

        return cap_by_pi_desc(S)

    # -------------------- low-level acceptance of a concrete set --------------------

    @staticmethod
    def _effective_thresholds(
        tau_in: float,
        tau_out: float,
        jaccard_min: Optional[float],
        ctx_score: Optional[float],
        severity: Optional[float],
    ) -> Tuple[float, float, Optional[float]]:
        """
        Relax thresholds when context is good and severity low.

        - tau_in: lowered (easier to include items)
        - tau_out: slightly lowered (less strict about excluding)
        - jaccard_min: lowered (accept slightly worse overlap)
        """
        g = _ctx_gain(ctx_score, severity)
        if g <= 0.0:
            return tau_in, tau_out, jaccard_min

        # in-confidence: from, say, 0.6 → as low as ~0.3 when g=1
        tau_in_eff = float(clamp(tau_in - 0.3 * g, 0.3, tau_in))

        # out-confidence: from 0.6 → 0.4
        tau_out_eff = float(clamp(tau_out - 0.2 * g, 0.4, tau_out))

        if jaccard_min is not None:
            jacc_eff = float(clamp(jaccard_min - 0.3 * g, 0.0, jaccard_min))
        else:
            jacc_eff = None

        return tau_in_eff, tau_out_eff, jacc_eff

    @staticmethod
    def _effective_risk(
        base_risk: float,
        ctx_score: Optional[float],
        severity: Optional[float],
        max_risk: float = 0.9,
    ) -> float:
        g = _ctx_gain(ctx_score, severity)
        return float(clamp(base_risk + g * (max_risk - base_risk), base_risk, max_risk))

    # -------------------- acceptance primitives --------------------

    @staticmethod
    def _safe_log(x: float, eps: float = 1e-12) -> float:
        return math.log(max(eps, min(1.0, x)))

    def _posterior_means(self, universe: Set[str]) -> Dict[str, float]:
        for e in universe:
            self.ensure(e)
        return {e: self.alpha[e] / (self.alpha[e] + self.beta[e]) for e in universe}

    def _hard_gate(
        self,
        *,
        op: str,
        S: Set[str],
        universe: Set[str],
        pi: Dict[str, float],
        tau_in: float,
        tau_out: float,
        require_all_out_confident: bool,
    ) -> bool:
        """
        Deterministic safety/feasibility gate using tau_in/tau_out.
        Mirrors your previous logic, but separated out.
        """
        if op == "isAnyOf":
            # at least one item in S must be high-confidence includable
            return True if not S else any(pi[e] >= tau_in for e in S)

        if op in ("isNoneOf", "neq"):
            # all forbidden items must be confidently excludable
            for e in S:
                if pi[e] > (1.0 - tau_out):
                    return False
            return True

        # eq / isAllOf (and fallback)
        # all proposed items must be confidently includable
        for e in S:
            if pi[e] < tau_in:
                return False

        # optional "out-confidence" (mainly for my_op == eq)
        if require_all_out_confident:
            for e in (universe - S):
                if pi[e] > (1.0 - tau_out):
                    return False

        return True

    def _conf_op(self, *, op: str, S: Set[str], universe: Set[str], pi: Dict[str, float]) -> float:
        """
        Operator-dependent confidence in [0,1] based on independence.
        Uses log-space for numerical stability.
        """
        if not S:
            return 1.0

        if op == "isAnyOf":
            # 1 - Π(1 - pi_e)
            s = 0.0
            for e in S:
                s += self._safe_log(1.0 - pi[e])
            return float(1.0 - math.exp(s))

        if op in ("isNoneOf", "neq"):
            # Π(1 - pi_e)
            s = 0.0
            for e in S:
                s += self._safe_log(1.0 - pi[e])
            return float(math.exp(s))

        # isAllOf / eq:
        # Π(pi_e)
        s = 0.0
        for e in S:
            s += self._safe_log(pi[e])
        return float(math.exp(s))

    def _expected_jaccard_mc(
        self,
        *,
        target: Set[str],
        universe: Set[str],
        pi: Dict[str, float],
        mc: int,
    ) -> float:
        """
        E[J(target, S_hat)] where S_hat ~ Π_e Bernoulli(pi_e)
        """
        j_sum = 0.0
        for _ in range(mc):
            S_hat = {e for e in universe if random.random() < pi[e]}
            if not (target or S_hat):
                j = 1.0
            else:
                inter = len(target & S_hat)
                union = len(target | S_hat)
                j = inter / union if union else 1.0
            j_sum += j
        return j_sum / max(1, mc)

    # -------------------- final accept --------------------

    def accept(
        self,
        my_op: str,
        my_val: Union[str, Set[str]],
        opp_op: str,
        opp_val: Union[str, Set[str]],
        *,
        tau_in: float = 0.6,
        tau_out: float = 0.6,
        jaccard_min: Optional[float] = None,
        mc: int = 2000,
        risk: float = 0.2,
        ctx_score: Optional[float] = None,
        severity: Optional[float] = None,
    ) -> bool:
        """
        HardGate(tau_in,tau_out)  AND  Conf(opponent_offer) >= 1 - rho_eff
        with optional global-overlap (expected Jaccard) check.
        """

        SB = _to_setish(opp_val)

        # universe (you can expand this if you want to include my_val items too)
        universe = set(SB) | set(self.alpha) | set(self.beta)
        # note: including my_val helps keep "out-confidence" meaningful early

        pi = self._posterior_means(universe)

        # adapt thresholds (your existing mechanism)
        eff_tau_in, eff_tau_out, eff_jacc = self._effective_thresholds(
            tau_in, tau_out, jaccard_min, ctx_score, severity
        )

        # compute effective risk (same style as enum/numeric)
        rho_eff = self._effective_risk(risk, ctx_score, severity)

        # optional out-confidence only when *I* am eq (same as your current design)
        # (0) optional global set-consistency check (Jaccard)
        if eff_jacc is not None:
            target = (universe - SB) if opp_op in ("isNoneOf", "neq") else SB
            ej = self._expected_jaccard_mc(target=target, universe=universe, pi=pi, mc=mc)
            if ej < eff_jacc:
                return False

        # (1) probabilistic stopping rule on the SAME incoming (op,val)
        conf = self._conf_op(op=opp_op, S=SB, universe=universe, pi=pi)

        return conf >= 1.0 - rho_eff