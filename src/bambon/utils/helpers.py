from __future__ import annotations
from collections.abc import Iterable
import random
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, List, Tuple
from ..models.expressions import BASE_DATE, _ISO_DURATION_RE, _ISO_DATETIME_RE
from ..models.models import Issue

def _to_setish(x):
    if isinstance(x, (set, list, tuple)): return set(x)
    return {x}

def _is_iterable(value):
    """Return True if value is an iterable (list, tuple, set, etc.) but not a string or bytes."""
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))

def format_operator_roperand(operator, right_operand):
    if _is_iterable(right_operand):
        right_operand = list(right_operand)
        if len(right_operand) > 1 and operator == 'eq':
            operator = 'isAllOf'
        elif len(right_operand) == 1 and operator == 'isAllOf':
            operator = 'eq'
            right_operand = right_operand[0]
        # elif len(right_operand) == 0:
        #     print(operator, right_operand)
        #     raise
        else:

            pass
    return operator, right_operand

def satisfies(my_op, my_val, their_op, their_val) -> bool | None:
    from .context import _intervals_overlap, _interval
    if isinstance(their_val,(set, list, tuple)) and len(their_val)==1:
        their_val = list(their_val)[0]

    A, B = _to_setish(my_val), _to_setish(their_val)

    # ---------- ENUM/SET logic ----------
    if my_op == "eq":
        if their_op == "eq":        return their_val == my_val
        if their_op == "neq":       return their_val != my_val
        if their_op == "isAnyOf":   return my_val in B
        if their_op == "isAllOf":   return A.issubset(B)
        if their_op == "isNoneOf":  return my_val not in B
        return False

    if my_op == "neq":
        if their_op == "eq":        return their_val != my_val
        if their_op == "neq":       return True
        if their_op == "isAnyOf":   return my_val not in B
        if their_op == "isAllOf":   return my_val not in B
        if their_op == "isNoneOf":  return True
        return False

    if my_op == "isAnyOf":
        if their_op == "eq":        return their_val in A
        if their_op == "neq":       return len(A - B) > 0
        if their_op == "isAnyOf":   return len(A & B) > 0
        if their_op == "isAllOf":   return B.issubset(A)
        if their_op == "isNoneOf":  return len(A & B) == 0
        return False

    if my_op == "isAllOf":
        if their_op == "eq":        return A.issubset(B)
        if their_op == "neq":       return their_val not in A
        if their_op == "isAnyOf":   return A.issubset(B)
        if their_op == "isAllOf":   return A.issubset(B)
        if their_op == "isNoneOf":  return len(A & B) == 0 and len(A) == 0
        return False

    if my_op == "isNoneOf":
        if their_op == "eq":        return their_val not in A
        if their_op == "neq":       return True  # they exclude one value; I exclude A — overlap exists
        if their_op in ("isAnyOf","isAllOf"): return len(A & B) == 0
        if their_op == "isNoneOf":  return True
        return False

    # ---------- NUM/TIME logic (numbers already) ----------
    if my_op in ("lt","lteq","gt","gteq","eq","neq"):
        # handle 'neq' by splitting into two rays
        if my_op == "neq" or their_op == "neq":
            if my_op == "neq" and their_op == "eq":
                return float(their_val) != float(my_val)
            if my_op == "eq" and their_op == "neq":
                return float(my_val) != float(their_val)
            if my_op == "neq" and their_op == "neq":
                return True  # anything except two points still overlaps on R

            # general split cases
            IA1 = _interval("lt", float(my_val)) if my_op == "neq" else _interval(my_op, float(my_val))
            IA2 = _interval("gt", float(my_val)) if my_op == "neq" else None
            IB1 = _interval("lt", float(their_val)) if their_op == "neq" else _interval(their_op, float(their_val))
            IB2 = _interval("gt", float(their_val)) if their_op == "neq" else None

            if my_op == "neq" and their_op != "neq":
                return _intervals_overlap(IA1, IB1) or _intervals_overlap(IA2, IB1)
            if my_op != "neq" and their_op == "neq":
                return _intervals_overlap(IA1, IB1) or _intervals_overlap(IA1, IB2)
            return False  # shouldn't reach

        # regular inequalities/equality
        IA = _interval(my_op,   float(my_val))
        IB = _interval(their_op,float(their_val))
        return _intervals_overlap(IA, IB)

    return None


class HardRule:
    def __init__(self,key,val):
        self.operator = strip_key(key)
        self.value = val
        self.func = partial(satisfies, self.operator, val)


    def __call__(self, their_operator, their_value):
        # print(self.key, self.val, their_value)
        return self.func(their_operator, their_value)
        # return self.func(their_value)


# ---------------------- small math ---------------------- #
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------- dates ---------------------- #
def is_iso_date(s: str) -> bool:
    return bool(_ISO_DATETIME_RE.match(s))

def iso_to_days(s: str | None) -> int:
    if not s:
        return None
    d = datetime.fromisoformat(s).date()
    return (d - BASE_DATE).days


def days_to_iso(days: int) -> str:
    return (BASE_DATE + timedelta(days=int(days))).isoformat()

# ---------------------- durations (ISO-8601) ---------------------- #
def is_iso_duration(s: str) -> bool:
    return bool(_ISO_DURATION_RE.match(s))

def duration_to_seconds(s: str | None) -> float:
    """ Parse ISO-8601 duration PnYnMnWnDTnHnMnS into seconds. """
    if not s:
        return None

    m = _ISO_DURATION_RE.match(s)
    if not m:
        raise ValueError(f"Not an ISO-8601 duration: {s}")
    years = int(m.group("Y") or 0)
    months = int(m.group("Mo") or 0)
    weeks = int(m.group("W") or 0)
    days = int(m.group("D") or 0)
    hours = int(m.group("H") or 0)
    minutes = int(m.group("Mi") or 0)
    seconds = float(m.group("S") or 0.0)
    total_days = years * 365 + months * 30 + weeks * 7 + days
    total_seconds = (total_days * 86400 + hours * 3600 + minutes * 60 + seconds)
    return float(total_seconds)

def seconds_to_duration(total_seconds: float) -> str:
    """ Convert seconds back to a normalized ISO-8601 duration using D/H/M/S only. """
    if total_seconds < 0:
        raise ValueError("Duration cannot be negative")
    secs = float(total_seconds)
    days = int(secs // 86400)
    secs -= days * 86400
    hours = int(secs // 3600)
    secs -= hours * 3600
    minutes = int(secs // 60)
    secs -= minutes * 60
    parts: List[str] = []
    if days:
        parts.append(f"{days}D")
    time_parts: List[str] = []
    if hours:
        time_parts.append(f"{hours}H")
    if minutes:
        time_parts.append(f"{minutes}M")
    if not parts and not time_parts and secs == 0:
        return "PT0S"
    if secs:
        if abs(secs - round(secs)) < 1e-9:
            time_parts.append(f"{int(round(secs))}S")
        else:
            time_parts.append(f"{secs:.3f}".rstrip("0").rstrip(".") + "S")
    if time_parts:
        return "P" + "".join(parts) + "T" + "".join(time_parts)
    else:
        return "P" + "".join(parts)


# ---------------------- Bayesian primitives ---------------------- #
def beta_sample(alpha: float, beta: float) -> float:
    x = random.gammavariate(max(alpha, 1e-9), 1.0)
    y = random.gammavariate(max(beta, 1e-9), 1.0)
    s = x + y
    return x / s if s > 0 else 0.5

def dirichlet_sample(alphas: List[float]) -> List[float]:
    xs = [random.gammavariate(max(a, 1e-9), 1.0) for a in alphas]
    s = sum(xs) or 1.0
    return [x / s for x in xs]


def strip_key(key:str):
    key = key.split('/')[-1]
    return key.split(':')[-1]

def is_iterable(x):
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

def create_issue(left_operand, operator, value, num_bounds:Tuple = None) -> Issue:
    from ..distributions.distributions import EnumDirichlet, NumericBeta, SetBelief

    if isinstance(value, (int, float)):
        if num_bounds:
            L, U = num_bounds
            if L and not U:
                U = value + (value - L)
            if U and not L:
                L = value + (U - value)
            m = NumericBeta(L=L, U=U)
        else:
            m = NumericBeta(L=float(value) - 0.2*float(value), U=float(value) + 0.2*float(value))

        current = Issue(
            kind='num',
            operand=left_operand,
            operator=operator,
            oppOperator=None,
            lastValue=value,
            initOperator=operator,
            initValue=value,
            oppLastValue=None,
            model=m,
            to_internal=lambda x: float(x),
            to_external=lambda x: float(x),
            hard_rule=None
        )
        return current

    if isinstance(value, str):
        if is_iso_duration(value):
            base = float(duration_to_seconds(value))
            if num_bounds:
                L, U = num_bounds
                L = duration_to_seconds(L)
                U = duration_to_seconds(U)
                if L and not U:
                    U = base + (base - L)
                if U and not L:
                    L = base + (U - base)
                m = NumericBeta(L=L, U=U)
            else:
                m = NumericBeta(L=base - 0.2*base, U=base + 0.2*base)
            current = Issue(
                kind='num',
                operand=left_operand,
                operator=operator,
                oppOperator=None,
                lastValue=value,
                initOperator=operator,
                initValue=value,
                oppLastValue=None,
                model=m,
                to_internal=lambda s: float(duration_to_seconds(s)),
                to_external=lambda secs: seconds_to_duration(float(secs)),
                hard_rule=None
            )
            return current

        if is_iso_date(value):
            base = float(iso_to_days(value))
            if num_bounds:
                L, U = num_bounds
                L = iso_to_days(L)
                U = iso_to_days(U)
                if L and not U:
                    U = base + (base - L)
                if U and not L:
                    L = base + (U - base)
                m = NumericBeta(L=L, U=U)
            else:
                m = NumericBeta(L=base - 15, U=base + 15)
            current = Issue(
                kind='num',
                operand=left_operand,
                operator=operator,
                oppOperator=None,
                lastValue=value,
                initOperator=operator,
                initValue=value,
                oppLastValue=None,
                model=m,
                to_internal=lambda s: float(iso_to_days(s)),
                to_external=lambda d: days_to_iso(int(round(d))),
                hard_rule=None
            )
            return current

        m = EnumDirichlet(labels=set())
        m.observe(value, lr=0.0)
        current = Issue(
            kind='enum',
            operand=left_operand,
            operator=operator,
            oppOperator=None,
            lastValue=value,
            initOperator=operator,
            initValue=value,
            oppLastValue=None,
            model=m,
            to_internal=lambda s: set(s) if is_iterable(s) else str(s),
            to_external=lambda s: list(s) if is_iterable(s) else str(s),
            hard_rule=None
        )
        return current

    if isinstance(value, list):  # and all(isinstance(x, str) for x in value):
        m = SetBelief()
        m.observe(set(value), lr=0.0)
        current = Issue(
            kind='set',
            operand=left_operand,
            operator=operator,
            oppOperator=None,
            lastValue=value,
            initOperator=operator,
            initValue=value,
            oppLastValue=None,
            model=m,
            to_internal=lambda xs: set(xs) if ( isinstance(xs, list) or isinstance(xs, tuple)) else {xs},
            to_external=lambda S: sorted(list(S)),
            hard_rule=None
        )
        return current

    m = EnumDirichlet(labels=set())
    m.observe(str(value), lr=0.0)
    current = Issue(
        kind='enum',
        operand=left_operand,
        operator=operator,
        oppOperator=None,
        lastValue=value,
        initOperator=operator,
        initValue=value,
        oppLastValue=None,
        model=m,
        to_internal=lambda s: set(s) if is_iterable(s) else str(s),
        to_external=lambda s: list(s) if is_iterable(s) else str(s),
        hard_rule=None
    )
    return current



def update_inertia(
    *,
    ctx: float,                 # c(t) in [0,1]
    sev: float | None = None,   # s(t) in [0,1] (optional)
    iota_min: float = 0.05,
    iota_max: float = 0.65,
    ctx_switch: float = 0.70,   # when ctx >= this, we consider "near agreement"
    use_severity: bool = True,  # if True, also reduce inertia when severity is low
) -> float:
    """
    Inertia schedule for ENUM + SET issues.

    - High inertia when context is low (exploration phases persist).
    - Low inertia when context is high (fast convergence).
    - Optionally also lowers inertia when severity is low (agreement is near).

    Returns iota in [iota_min, iota_max].
    """
    c = clamp(ctx, 0.0, 1.0)

    # Base: iota = iota_max*(1 - c), then clipped to [iota_min, iota_max]
    iota = iota_max * (1.0 - c)
    iota = clamp(iota, iota_min, iota_max)

    # If we are near agreement (high context), force minimal inertia
    if c >= ctx_switch:
        iota = iota_min

    # Optional: incorporate severity (when severity is low, reduce inertia further)
    if use_severity and sev is not None:
        s = clamp(sev, 0.0, 1.0)
        # g = c*(1-s): when high => negotiation is "globally good"
        g = clamp(c * (1.0 - s), 0.0, 1.0)
        # linearly push inertia down as g increases
        iota = clamp(iota * (1.0 - 0.8 * g), iota_min, iota_max)

    return iota