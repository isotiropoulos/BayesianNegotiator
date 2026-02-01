from .helpers import clamp, _to_setish
from typing import Tuple, Optional
from ..distributions.distributions import numeric_constraints_intersection
from ..models.models import Issue


def _intervals_overlap(a: Tuple[float, float, bool, bool],
                       b: Tuple[float, float, bool, bool]) -> bool:
    (al, ah, acl, ach) = a
    (bl, bh, bcl, bch) = b
    lo = max(al, bl)
    hi = min(ah, bh)
    if lo < hi:
        return True
    if lo == hi:
        # boundary touching counts only if both include the boundary
        left_ok = (lo != al) or acl
        right_ok = (hi != bh) or bch
        return left_ok and right_ok
    return False

def _LU(issue: Issue) -> Tuple[Optional[float], Optional[float]]:
    L = getattr(issue.model, "L", None)
    U = getattr(issue.model, "U", None)
    L = float(L) if isinstance(L, (int, float)) else None
    U = float(U) if isinstance(U, (int, float)) else None
    return L, U

def _interval(op: str, v: float) -> Tuple[float, float, bool, bool]:
    if op == "eq":
        return (v, v, True, True)
    if op == "lteq":
        return (float("-inf"), v, False, True)
    if op == "lt":
        return (float("-inf"), v, False, False)
    if op == "gteq":
        return (v, float("+inf"), True, False)
    if op == "gt":
        return (v, float("+inf"), False, False)
    # unknown operator: unconstrained
    return (float("-inf"), float("+inf"), False, False)

# -------------------- per-issue similarity --------------------

def _set_similarity(issue: Issue, with_init) -> float:
    """Similarity in [0,1] for set/enum kinds using both operators."""
    Bset = _to_setish(issue.oppLastValue)
    Bop = issue.oppOperator
    if with_init:
        Aset = _to_setish(issue.initValue)
        Aop = issue.initOperator
    else:
        Aset = _to_setish(issue.lastValue)
        Aop = issue.operator

    if not (Aset or Bset):
        return 1.0

    # Base Jaccard
    j = len(Aset & Bset) / len(Aset | Bset)

    # Fulfilment helpers
    def frac_allof(superset: set, required: set) -> float:
        """How much of 'required' is contained by 'superset' (|∩|/|required|)."""
        return (len(superset & required) / max(1, len(required)))

    def sat(opX: str, X: set, opY: str, Y: set) -> float:
        """Satisfaction of (X opX ·) against the counterpart Y/opY."""
        # normalize a representative counterpart set for eq vs set
        if opY == "eq":
            # if counterpart has multiple, treat as exact set; if singleton, same
            Yeff = Y
        else:
            Yeff = Y

        if opX == "isAllOf":    # want Y ⊆ X
            return frac_allof(X, Yeff)
        if opX == "isAnyOf":    # any overlap is enough
            return 1.0 if (X & Yeff) else j
        if opX == "isNoneOf":   # must be disjoint
            return 1.0 if not (X & Yeff) else max(0.0, 1.0 - j)
        if opX == "eq":
            # exact match if sets equal; else allow singleton-in-set as soft match
            if X == Yeff:
                return 1.0
            if len(X) == 1 and (next(iter(X)) in Yeff):
                return 0.8
            return j * 0.5
        if opX == "neq":
            if X == Yeff:
                return 0.0
            if len(X) == 1 and (next(iter(X)) not in Yeff):
                return 1.0
            return max(0.0, 1.0 - j)
        # default: symmetric similarity
        return j

    # combine: base Jaccard plus directional fulfilments
    sat_A = sat(Aop, Aset, Bop, Bset)
    sat_B = sat(Bop, Bset, Aop, Aset)
    sim = 0.25 * j + 0.375 * sat_A + 0.375 * sat_B
    return float(clamp(sim, 0.0, 1.0))

def _num_similarity(issue: Issue, with_init) -> float:
    """
    Soft similarity in [0,1] for numeric issues.

    - 0.0 if constraints are semantically incompatible
    - Higher when operators/values are compatible and close
    - Inequalities are treated more forgivingly than strict eq
    """
    if with_init:
        ai = issue.to_internal(issue.initValue)
    else:
        ai = issue.to_internal(issue.lastValue)
    bi = issue.to_internal(issue.oppLastValue)
    if ai is None or bi is None:
        return 0.0

    try:
        ai = float(ai)
        bi = float(bi)
    except Exception:
        return 0.0


    A = issue.operator or "eq"    # one of: eq, neq, gt, gteq, lt, lteq
    if with_init:
        A = issue.initOperator
    B = issue.oppOperator or "eq"

    # 1) semantic compatibility using your existing logic
    compatible, interval = numeric_constraints_intersection(
        op_a=A,
        val_a=ai,
        op_b=B,
        val_b=bi,
    )
    if not compatible or interval is None:
        return 0.0

    # 2) define a numeric domain for scaling distances
    L, U = _LU(issue)
    if (L is None) or (U is None) or not (U > L):
        base_min = min(ai, bi)
        base_max = max(ai, bi)
        half_span = max(1.0, abs(base_max - base_min))
        L = base_min - half_span
        U = base_max + half_span

    L = float(L)
    U = float(U)
    dom_span = max(1e-9, U - L)

    # 3) distance-based component (0 if very far, 1 if identical)
    dist = abs(ai - bi) / dom_span
    dist_sim = max(0.0, 1.0 - dist)

    # 4) raw similarity:
    #    - eq/eq only happens when ai == bi (otherwise incompatible above) → dist_sim=1
    #    - inequalities: more forgiving
    if A == "eq" and B == "eq":
        sim_raw = dist_sim  # will be 1.0 in practice
    else:
        # inequalities / mixed: keep ≥0.5 even when far but compatible
        sim_raw = 0.5 + 0.5 * dist_sim  # ∈ [0.5,1]

    # 5) final: any compatible pair gets at least ~0.3
    sim = 0.3 + 0.7 * sim_raw  # ∈ [0.3,1] for compatible, 0 for incompatible
    return float(clamp(sim, 0.0, 1.0))



def _issue_similarity(issue: Issue, with_init=False) -> float:
    if issue.kind == "num":
        return _num_similarity(issue, with_init)
    # treat enums as sets (singletons)
    return _set_similarity(issue, with_init)
