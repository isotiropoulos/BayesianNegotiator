from typing import List, Any, Tuple, Optional
from statistics import median
from .helpers import clip
from ..models.models import Issue


def _as_set(x: Any) -> set:
    if x is None: return set()
    if isinstance(x, set): return x
    if isinstance(x, (list, tuple)): return set(x)
    return {x}

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

def _num_similarity(issue: Issue) -> float:
    """Similarity in [0,1] for numeric/time kinds using both operators and interval overlap."""
    ai = issue.to_internal(issue.lastValue)
    bi = issue.to_internal(issue.oppLastValue)
    if ai is None or bi is None:
        return 0.0

    try:
        ai = float(ai)
        bi = float(bi)
    except Exception:
        # If conversion failed, we cannot do numeric semantics
        return 0.0

    A = issue.operator
    B = issue.oppOperator

    # Build intervals and check feasibility
    IA = _interval(A, ai)
    IB = _interval(B, bi)

    # Normalization span
    L, U = _LU(issue)
    if (L is not None) and (U is not None) and (U > L):
        span = max(1e-6, U - L)
    else:
        span = max(1.0, abs(ai), abs(bi))

    # If intervals don't overlap, no assignment satisfies both → similarity 0
    if not _intervals_overlap(IA, IB):
        return 0.0

    # If both are "eq", they overlap only if equal
    if A in ("eq", "==") and B in ("eq", "=="):
        return 1.0 if ai == bi else 0.0

    # If one is "eq", check if that point satisfies the other's half-space
    if A in ("eq", "=="):
        # ai must be within IB
        return 1.0  # we already know intervals overlap
    if B in ("eq", "=="):
        # bi must be within IA
        return 1.0  # already overlapping

    # Both half-spaces and they overlap → fully compatible.
    # Optionally add a tiny soft penalty for widely separated bounds.
    # Here we keep it simple and treat as fully compatible.
    return 1.0


def _set_similarity(issue: Issue) -> float:
    """Similarity in [0,1] for set/enum kinds using both operators."""
    Aset = _as_set(issue.lastValue)
    Bset = _as_set(issue.oppLastValue)
    Aop = issue.operator
    Bop = issue.oppOperator

    if not (Aset or Bset):
        return 1.0

    # Base Jaccard
    j = len(Aset & Bset) / len(Aset | Bset)

    # Fulfilment helpers
    def frac_allof(superset: set, required: set) -> float:
        """How much of 'required' is contained by 'superset' (|∩|/|required|)."""
        return (len(superset & required) / max(1, len(required)))

    def frac_partof(subset: set, container: set) -> float:
        """How much of 'subset' is present in 'container' (|∩|/|subset|)."""
        return (len(subset & container) / max(1, len(subset)))

    def sat(opX: str, X: set, opY: str, Y: set) -> float:
        """Satisfaction of (X opX ·) against the counterpart Y/opY."""
        # normalize a representative counterpart set for eq vs set
        if opY == "eq":
            # if counterpart has multiple, treat as exact set; if singleton, same
            Yeff = Y
        else:
            Yeff = Y

        if opX == "isallof":    # want Y ⊆ X
            return frac_allof(X, Yeff)
        if opX == "haspart":    # same as isAllOf wrt counterpart
            return frac_allof(X, Yeff)
        if opX == "ispartof":   # want X ⊆ Y
            return frac_partof(X, Yeff)
        if opX == "isanyof":    # any overlap is enough
            return 1.0 if (X & Yeff) else j
        if opX == "isnoneof":   # must be disjoint
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
    return float(clip(sim, 0.0, 1.0))


def _issue_similarity(issue: Issue) -> float:
    if issue.kind == "num":
        return _num_similarity(issue)
    # treat enums as sets (singletons)
    return _set_similarity(issue)
