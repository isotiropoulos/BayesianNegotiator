"""ODRL-semantic utility and hard-constraint evaluation.

This module turns a settled ODRL agreement into a utility value *from the
perspective of one party*, using ODRL deontic semantics:

  * A party cares about **prohibitions more than obligations, and obligations
    more than permissions**. The default rule weights encode this ranking
    (0.55 / 0.30 / 0.15) and are configurable via :class:`UtilityConfig`.

  * Each negotiated issue produces a **satisfaction score in [0,1]**:
      - numeric / temporal / duration issues use a configurable *quadratic*
        penalty on the normalised distance from the party's preferred value;
      - string / set issues are normalised to sets and scored with a **weighted
        overlap** that rewards *preservation* of the preferred values more than
        *compactness* of the agreement (so ``{"JSON"}`` vs ``{"JSON","XML"}``
        scores high but not perfect). Textual values may optionally be compared
        with sentence-embedding similarity (Sentence-Transformers,
        ``all-MiniLM-L6-v2``); exact matching is the default.

  * Hard constraints define the feasible region. Instead of a binary
    feasible/infeasible test we compute a **continuous violation score in [0,1]**
    per constraint (numeric distance outside the feasible interval, missing
    mandatory set elements, or semantic mismatch), aggregate them into a weighted
    overall violation ``V``, and turn it into a multiplicative penalty
    ``H = exp(-lambda * V)`` (default ``lambda = 10``) so even moderate
    violations sharply reduce utility.

The final utility is::

    U = (sum_r w_r * sat_r) * exp(-lambda * V)

where ``sat_r`` is the mean issue-satisfaction of rule type ``r`` (permission /
obligation / prohibition) and ``w_r`` the (renormalised) rule weight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.bambon.utils.helpers import (
    is_iso_date, iso_to_days, is_iso_duration, duration_to_seconds, satisfies,
)

RULE_TYPES = ("permission", "obligation", "prohibition")
_INCLUSION_OPS = ("eq", "isAnyOf", "isAllOf")
_EXCLUSION_OPS = ("neq", "isNoneOf")
_NUM_OPS = ("lt", "lteq", "gt", "gteq", "eq", "neq")


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass
class UtilityConfig:
    """All knobs of the ODRL utility. Defaults encode the deontic ranking."""
    # Rule-type importance: prohibition > obligation > permission.
    rule_weights: Dict[str, float] = field(default_factory=lambda: {
        "prohibition": 0.55, "obligation": 0.30, "permission": 0.15})
    # Numeric satisfaction: 1 - (dist/scale)**power.
    numeric_penalty_power: float = 2.0
    # If True, beating the preferred value in the operator's favourable direction
    # (e.g. provider prefers `lteq 80`, agreement is 70) scores a perfect 1.0;
    # only unfavourable deviations are penalised. If False, pure distance is used.
    numeric_directional: bool = True
    # Set satisfaction: reward preservation of preferred values over compactness.
    set_preservation_weight: float = 0.7
    set_compactness_weight: float = 0.3
    # Hard-constraint penalty H = exp(-hard_lambda * V).
    hard_lambda: float = 10.0
    # Importance of a violation by rule type (same deontic ranking by default).
    violation_rule_weights: Dict[str, float] = field(default_factory=lambda: {
        "prohibition": 0.55, "obligation": 0.30, "permission": 0.15})
    # Optional semantic similarity for textual set/string values.
    use_semantic: bool = False
    semantic_model_name: str = "all-MiniLM-L6-v2"
    # Below this cosine similarity two labels are treated as unrelated.
    semantic_min_similarity: float = 0.0


# ----------------------------------------------------------------------
# value helpers
# ----------------------------------------------------------------------
def _strip(s: Any) -> Any:
    return s.replace("odrl:", "") if isinstance(s, str) else s


def _to_num(v: Any) -> Optional[float]:
    """Number / ISO date / ISO duration -> float, else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            if is_iso_date(v):
                return float(iso_to_days(v))
            if is_iso_duration(v):
                return float(duration_to_seconds(v))
            return float(v)
        except Exception:
            return None
    return None


def _as_set(v: Any) -> set:
    if isinstance(v, (list, tuple, set)):
        return {str(x) for x in v}
    return {str(v)}


def _scale(operand: str, ranges: Dict[str, Tuple[float, float]], ref: float) -> float:
    """Normalisation scale for a numeric operand (its negotiation span)."""
    if operand in ranges:
        lo, hi = ranges[operand]
        span = float(hi) - float(lo)
        if span > 0:
            return span
    return max(abs(ref), 1.0)


def _find(offer: Optional[Dict], rule_type: str, action: Any, operand: str) -> Any:
    """First agreed rightOperand for (rule_type, action, operand), else None."""
    if not offer:
        return None
    for rule in offer.get(rule_type, []) or []:
        if action is not None and rule.get("action") != action:
            continue
        for c in rule.get("constraint", []) or []:
            if _strip(c["leftOperand"]) == operand:
                return c["rightOperand"]
    return None


def _find_clause(offer: Optional[Dict], rule_type: str, action: Any,
                 operand: str) -> Tuple[Optional[str], Any]:
    """First agreed ``(operator, rightOperand)`` for the operand, else ``(None, None)``."""
    if not offer:
        return None, None
    for rule in offer.get(rule_type, []) or []:
        if action is not None and rule.get("action") != action:
            continue
        for c in rule.get("constraint", []) or []:
            if _strip(c["leftOperand"]) == operand:
                return _strip(c["operator"]), c["rightOperand"]
    return None, None


# ----------------------------------------------------------------------
# optional semantic matching (Sentence-Transformers)
# ----------------------------------------------------------------------
class _Semantic:
    _model = None
    _cache: Dict[str, Any] = {}
    _warned = False

    @classmethod
    def available(cls, name: str) -> bool:
        if cls._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            cls._model = SentenceTransformer(name)
            return True
        except Exception:
            if not cls._warned:
                print("[odrl_utility] sentence-transformers unavailable; "
                      "falling back to exact matching for textual values.")
                cls._warned = True
            return False

    @classmethod
    def embed(cls, text: str):
        if text not in cls._cache:
            import numpy as np  # local import; numpy is a project dep
            v = cls._model.encode([text])[0]
            n = float(np.linalg.norm(v)) or 1.0
            cls._cache[text] = np.asarray(v) / n
        return cls._cache[text]

    @classmethod
    def sim(cls, a: str, b: str) -> float:
        if a == b:
            return 1.0
        import numpy as np
        return float(np.dot(cls.embed(a), cls.embed(b)))


def _best_sim(x: str, pool: set, cfg: UtilityConfig) -> float:
    if x in pool:
        return 1.0
    if not pool:
        return 0.0
    if cfg.use_semantic and _Semantic.available(cfg.semantic_model_name):
        return max((max(0.0, _Semantic.sim(x, y)) for y in pool), default=0.0)
    return 0.0


def _avg_best_sim(src: set, dst: set, cfg: UtilityConfig) -> float:
    if not src:
        return 1.0
    sims = [_best_sim(x, dst, cfg) for x in src]
    sims = [s if s >= cfg.semantic_min_similarity else 0.0 for s in sims]
    return sum(sims) / len(sims)


# ----------------------------------------------------------------------
# per-issue satisfaction
# ----------------------------------------------------------------------
def numeric_satisfaction(preferred: float, agreed: float, operator: str,
                         scale: float, cfg: UtilityConfig) -> float:
    """Satisfaction in [0,1] for a numeric issue (configurable quadratic penalty)."""
    if scale <= 0:
        scale = max(abs(preferred), 1.0)
    op = _strip(operator)
    if cfg.numeric_directional and op in ("lteq", "lt", "gteq", "gt"):
        favourable = ((op in ("lteq", "lt") and agreed <= preferred) or
                      (op in ("gteq", "gt") and agreed >= preferred))
        if favourable:
            return 1.0
    dist = abs(agreed - preferred) / scale
    return max(0.0, 1.0 - dist ** cfg.numeric_penalty_power)


def set_satisfaction(preferred: set, agreed: set, cfg: UtilityConfig) -> float:
    """Weighted-overlap satisfaction that rewards preservation over compactness.

    preservation = how much of the *preferred* set is kept by the agreement;
    compactness  = how little *extra* the agreement adds beyond preferred.
    Example: preferred {JSON}, agreed {JSON, XML} -> preservation 1.0,
    compactness 0.5 -> 0.7*1.0 + 0.3*0.5 = 0.85 (high but not perfect).
    """
    if not preferred and not agreed:
        return 1.0
    if not preferred:
        return 1.0
    if not agreed:
        return 0.0
    if cfg.use_semantic and _Semantic.available(cfg.semantic_model_name):
        preservation = _avg_best_sim(preferred, agreed, cfg)
        compactness = _avg_best_sim(agreed, preferred, cfg)
    else:
        inter = len(preferred & agreed)
        preservation = inter / len(preferred)
        compactness = inter / len(agreed)
    wp, wc = cfg.set_preservation_weight, cfg.set_compactness_weight
    total = (wp + wc) or 1.0
    return (wp * preservation + wc * compactness) / total


def issue_satisfaction(operator: str, preferred: Any, agreed: Any,
                       operand: str, ranges: Dict[str, Tuple[float, float]],
                       cfg: UtilityConfig) -> float:
    if agreed is None:                       # operand dropped from the agreement
        return 0.0
    pn, an = _to_num(preferred), _to_num(agreed)
    if pn is not None and an is not None:
        return numeric_satisfaction(pn, an, operator, _scale(operand, ranges, pn), cfg)
    P, A = _as_set(preferred), _as_set(agreed)
    if _strip(operator) in _EXCLUSION_OPS:   # prefer to AVOID these labels
        if not A:
            return 1.0
        overlap = (len(P & A) / len(A) if not (cfg.use_semantic and
                   _Semantic.available(cfg.semantic_model_name))
                   else _avg_best_sim(A, P, cfg))
        return max(0.0, 1.0 - overlap)
    return set_satisfaction(P, A, cfg)


# ----------------------------------------------------------------------
# per-issue violation (continuous, in [0,1])
# ----------------------------------------------------------------------
def _is_feasible(hard_op: str, hard_value: Any, agreed_op: Optional[str],
                 agreed: Any) -> Optional[bool]:
    """Authoritative feasibility test — the *exact* overlap semantics BAMBON
    enforces internally (``helpers.satisfies``). Returns True/None when the agreed
    clause does NOT violate the hard rule, False when it genuinely does.

    Using the same test guarantees the continuous violation score is a strict
    refinement of BAMBON's hard-feasibility guarantee: it can never flag a
    "violation" on a clause BAMBON considers feasible (e.g. being *granted a
    superset* of an ``isAnyOf`` permission is a benefit, not a violation).
    """
    hn, an = _to_num(hard_value), _to_num(agreed)
    if hn is not None and an is not None:
        # Numeric/temporal/duration: agreed is a settled point.
        return satisfies(hard_op if hard_op in _NUM_OPS else "eq", hn, "eq", an)
    return satisfies(hard_op, hard_value, agreed_op or "eq", agreed)


def issue_violation(operator: str, hard_value: Any, agreed_op: Optional[str],
                    agreed: Any, operand: str,
                    ranges: Dict[str, Tuple[float, float]], cfg: UtilityConfig) -> float:
    """Continuous violation score in [0,1] of one hard constraint.

    The score is gated by the authoritative ``satisfies`` overlap test so it is
    ``0`` whenever the agreed clause is feasible (identical to BAMBON's own
    compliance guard) and only grows with the *magnitude* of a genuine breach.
    """
    if agreed is None:                       # non-negotiable limit was dropped
        return 1.0
    op = _strip(operator)
    # Feasibility gate: no genuine violation -> no penalty.
    feasible = _is_feasible(op, hard_value, agreed_op, agreed)
    if feasible is None or feasible:
        return 0.0
    # Genuine violation -> grade its magnitude.
    hn, an = _to_num(hard_value), _to_num(agreed)
    if hn is not None and an is not None:
        scale = _scale(operand, ranges, hn)
        if op in ("lteq", "lt"):
            return min(1.0, (an - hn) / scale)
        if op in ("gteq", "gt"):
            return min(1.0, (hn - an) / scale)
        if op == "eq":
            return min(1.0, abs(an - hn) / scale)
        if op == "neq":
            return 1.0
        return 1.0
    H, A = _as_set(hard_value), _as_set(agreed)
    if op in ("isAllOf", "eq", "isAnyOf"):   # required/mandatory elements missing
        if cfg.use_semantic and _Semantic.available(cfg.semantic_model_name):
            return 1.0 - _avg_best_sim(H, A, cfg)
        return (len(H - A) / len(H)) if H else 1.0
    if op in ("isNoneOf", "neq"):            # forbidden labels present
        if not A:
            return 0.0
        return len(A & H) / len(A)
    return 1.0


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------
def _weighted_combine(per_rule: Dict[str, float], weights: Dict[str, float]) -> float:
    if not per_rule:
        return 0.0
    total = sum(weights.get(rt, 0.0) for rt in per_rule)
    if total <= 0:
        return sum(per_rule.values()) / len(per_rule)
    return sum(weights.get(rt, 0.0) * v for rt, v in per_rule.items()) / total


def per_rule_satisfaction(agreement: Optional[Dict], preferred: Dict,
                          ranges: Dict[str, Tuple[float, float]],
                          cfg: UtilityConfig) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for rt in RULE_TYPES:
        sats: List[float] = []
        for rule in preferred.get(rt, []) or []:
            action = rule.get("action")
            for c in rule.get("constraint", []) or []:
                op = _strip(c["leftOperand"])
                sats.append(issue_satisfaction(
                    c["operator"], c["rightOperand"],
                    _find(agreement, rt, action, op), op, ranges, cfg))
        if sats:
            out[rt] = sum(sats) / len(sats)
    return out


def per_rule_violation(agreement: Optional[Dict], hard_policy: Optional[Dict],
                       ranges: Dict[str, Tuple[float, float]],
                       cfg: UtilityConfig) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not hard_policy:
        return out
    for rt in RULE_TYPES:
        viols: List[float] = []
        for rule in hard_policy.get(rt, []) or []:
            action = rule.get("action")
            for c in rule.get("constraint", []) or []:
                op = _strip(c["leftOperand"])
                a_op, a_val = _find_clause(agreement, rt, action, op)
                viols.append(issue_violation(
                    c["operator"], c["rightOperand"], a_op, a_val, op, ranges, cfg))
        if viols:
            out[rt] = sum(viols) / len(viols)
    return out


def odrl_utility(agreement: Optional[Dict], preferred: Dict,
                 hard_policy: Optional[Dict] = None,
                 ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 config: Optional[UtilityConfig] = None) -> Dict[str, Any]:
    """Full ODRL utility breakdown for one party.

    Returns a dict with the final ``utility`` plus its components: the weighted
    ``satisfaction`` (before penalty), the aggregate ``violation`` ``V``, the
    hard-constraint penalty ``hard_penalty`` ``H``, and per-rule-type tables.
    """
    cfg = config or UtilityConfig()
    ranges = ranges or {}
    if agreement is None:
        return {"utility": 0.0, "satisfaction": 0.0, "violation": 1.0,
                "hard_penalty": math.exp(-cfg.hard_lambda),
                "per_rule_satisfaction": {}, "per_rule_violation": {}, "agreed": False}

    sat_rule = per_rule_satisfaction(agreement, preferred, ranges, cfg)
    viol_rule = per_rule_violation(agreement, hard_policy, ranges, cfg)
    satisfaction = _weighted_combine(sat_rule, cfg.rule_weights)
    V = _weighted_combine(viol_rule, cfg.violation_rule_weights) if viol_rule else 0.0
    H = math.exp(-cfg.hard_lambda * V)
    return {
        "utility": satisfaction * H,
        "satisfaction": satisfaction,
        "violation": V,
        "hard_penalty": H,
        "per_rule_satisfaction": sat_rule,
        "per_rule_violation": viol_rule,
        "agreed": True,
    }


def odrl_utility_value(agreement: Optional[Dict], preferred: Dict,
                       hard_policy: Optional[Dict] = None,
                       ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                       config: Optional[UtilityConfig] = None) -> float:
    """Convenience scalar wrapper around :func:`odrl_utility`."""
    return odrl_utility(agreement, preferred, hard_policy, ranges, config)["utility"]
