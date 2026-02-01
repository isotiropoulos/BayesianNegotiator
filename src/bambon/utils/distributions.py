from __future__ import annotations
from typing import List, Optional, Tuple, Literal
from ..utils.helpers import clamp


def _percentile(sorted_vals: List[float], q: float) -> float:
    q = clamp(q, 0.0, 1.0)
    if not sorted_vals:
        return 0.5
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


def _unit_tail_prob(samples_unit: List[float], z: float, *, side: Literal["ge", "le"]) -> float:
    if side == "ge":
        return sum(1 for s in samples_unit if s >= z) / max(1, len(samples_unit))
    return sum(1 for s in samples_unit if s <= z) / max(1, len(samples_unit))


def _ctx_gain(ctx_score: Optional[float], severity: Optional[float]) -> float:
    """
    How 'globally good' the offer is, in [0, 1].

      - high when context high & severity low
      - low when context low or severity high
    """
    if ctx_score is None or severity is None:
        return 0.0
    c = clamp(ctx_score, 0.0, 1.0)
    s = clamp(severity, 0.0, 1.0)
    return clamp(c * (1.0 - s), 0.0, 1.0)


# ---------------------------------------------------------------------
# Numeric: deterministic compatibility of two one-sided constraints
# ---------------------------------------------------------------------

def numeric_constraints_intersection(
    *,
    op_a: str,
    val_a: float,
    op_b: str,
    val_b: float,
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """
    Treat constraints as half-intervals:
      - gteq/gt   -> [x, +inf) / (x, +inf)
      - lteq/lt   -> (-inf, x] / (-inf, x)
      - eq        -> {x}
      - neq       -> R \\ {x}  (we treat this as not shrinking the interval)

    Returns:
      (compatible?, (lo, hi) in original numeric space or None)
    """
    # direct contradiction: x == a AND x != a
    if op_a == "eq" and op_b == "neq" and val_a == val_b:
        return False, None
    if op_b == "eq" and op_a == "neq" and val_a == val_b:
        return False, None

    lo = -float("inf")
    hi = float("inf")

    # op_a
    if op_a in ("gteq", "gt"):
        lo = max(lo, val_a)
    elif op_a in ("lteq", "lt"):
        hi = min(hi, val_a)
    elif op_a == "eq":
        lo = max(lo, val_a)
        hi = min(hi, val_a)
    # 'neq' does not shrink interval

    # op_b
    if op_b in ("gteq", "gt"):
        lo = max(lo, val_b)
    elif op_b in ("lteq", "lt"):
        hi = min(hi, val_b)
    elif op_b == "eq":
        lo = max(lo, val_b)
        hi = min(hi, val_b)
    # 'neq' does not shrink interval

    if lo <= hi:
        return True, (lo, hi)
    return False, None

