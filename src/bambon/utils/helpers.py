# src/bambon/utils/helpers.py
from __future__ import annotations
from collections.abc import Iterable

import random
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, Dict, List, Tuple

from ..models.expressions import BASE_DATE, _ISO_DATE_RE, _ISO_DURATION_RE, _ISO_DATETIME_RE
from ..models.models import IssueKey, Issue

def _to_setish(x):
    if isinstance(x, (set, list, tuple)): return set(x)
    return {x}
# , {'mine': my_val, 'op': my_op, 'their':their_val}

# def satisfies(my_op, my_val, their_op, their_val) -> (bool, Any):
#     A, B = _to_setish(my_val), _to_setish(their_val)
#
#     # ENUM/SET logic
#     if my_op == "eq":
#         # they satisfy iff they allow my exact value
#         if their_op == "eq": return their_val == my_val, None
#         # if their_op == "isAnyOf":return my_val in B, None
#         # if their_op == "isAllOf":return A.issubset(B), None
#         # if their_op == "isNoneOf":return my_val not in B, None
#         return False, None
#
#     if my_op == "isAnyOf":
#         if their_op == "eq":     return their_val in A
#         if their_op == "isAnyOf":return len(A & B) > 0
#         if their_op == "isAllOf":return B.issubset(A)   # they require a subset of what I allow
#         if their_op == "isNoneOf":return len(A & B) == 0
#         return False
#
#     if my_op == "isAllOf":
#         if their_op == "eq":     return A.issubset(B)
#         if their_op == "isAnyOf":return A.issubset(B)   # they must at least allow all I require
#         if their_op == "isAllOf":return A.issubset(B)
#         if their_op == "isNoneOf":return len(A & B) == 0 and len(A) == 0
#         return False
#
#     if my_op == "isNoneOf":
#         if their_op == "eq":     return their_val not in A
#         if their_op in ("isAnyOf","isAllOf"): return len(A & B) == 0
#         return False
#
#     # NUM/TIME logic via intervals (use your existing helpers)
#     if my_op in ("lt","lteq","gt","gteq","eq"):
#         IA = _interval(my_op,   float(_to_num(my_val)))
#         IB = _interval(their_op,float(_to_num(their_val)))
#         # require at least overlap; if you want "they entail me", require inclusion instead:
#         return _intervals_overlap(IA, IB)
#
#     return False
#
#















class HardRule:
    def __init__(self,key,val):
        self.operator = strip_key(key)
        self.value = val
        self.func = partial(LAMBDA_OPS.get(self.operator, __no_hard_constraint__), val)
    def __call__(self, their_value):
        # print(self.key, self.val, their_value)
        return self.func(their_value)

LAMBDA_OPS = {
    "eq":   lambda mine, their: (
        # accept if their single value equals mine
        their == mine
        # OR if their is a set/list that CONTAINS mine (e.g. their isAnyOf)
        or (mine in _to_setish(their))
    ),

    "neq":  lambda mine, their: (
        their != mine
        # if their is a set, ensure it does not contain mine
        and (mine not in _to_setish(their))
    ),

    "lt":   lambda mine, their: float(their) <  float(mine),
    "lteq": lambda mine, their: float(their) <= float(mine),
    "gt":   lambda mine, their: float(their) >  float(mine),
    "gteq": lambda mine, their: float(their) >= float(mine),

    # I allow ANY of 'mine'; their value(s) must intersect mine
    "isAnyOf": lambda mine, their: (
        len(_to_setish(mine) & _to_setish(their)) > 0
        # and len(_to_setish(their) - _to_setish(mine)) == 0
    ),

    # I require ALL of 'mine'; their must include mine (subset the right way)
    "isAllOf": lambda mine, their: _to_setish(mine).issubset(_to_setish(their)),

    # I forbid all of 'mine'; their must be disjoint from mine
    "isNoneOf": lambda mine, their: len(_to_setish(mine) & _to_setish(their)) == 0,
}

def __no_hard_constraint__(mine, their):
    return None


# ---------------------- small math ---------------------- #
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _interp(lo: float, hi: float, t: float) -> float:
    """Linear interpolation for t in [0, 1]."""
    t = clamp(t, 0.0, 1.0)
    return lo + (hi - lo) * t

# ---------------------- dates ---------------------- #
def is_iso_date(s: str) -> bool:
    return bool(_ISO_DATETIME_RE.match(s))

def iso_to_days(s: str) -> int:
    d = datetime.fromisoformat(s).date()
    return (d - BASE_DATE).days

def is_iso_datetime(s: str) -> bool:
    return bool(_ISO_DATETIME_RE.match(s))

def iso_datetime_to_xsd_date(s: str) -> str:
    """ Parse an ISO-8601 datetime (Z or ±HH:MM) and return xsd:date (YYYY-MM-DD) in UTC. """
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.date().isoformat()

def days_to_iso(days: int) -> str:
    return (BASE_DATE + timedelta(days=int(days))).isoformat()

# ---------------------- durations (ISO-8601) ---------------------- #
def is_iso_duration(s: str) -> bool:
    return bool(_ISO_DURATION_RE.match(s))

def duration_to_seconds(s: str) -> float:
    """ Parse ISO-8601 duration PnYnMnWnDTnHnMnS into seconds. """
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

# ---------------------- offer (ODRL) helpers ---------------------- #
def parse_offer_generic(offer: Dict[str, Any]) -> Dict[IssueKey, Any]:
    """ Parse an ODRL offer into {(leftOperand, operator, index): rightOperand}.
        Normalizations:
        - odrl:dateTime odrl:isWithin [start,end] -> emit two constraints (gteq/lteq)
        - any odrl:dateTime datetime strings -> coerced to xsd:date (YYYY-MM-DD)
    """
    permissions = offer.get("permission") or []
    if not permissions:
        return {}

    constraints = permissions[0].get("constraint") or []
    groups: Dict[Tuple[str, str], int] = {}
    out: Dict[IssueKey, Any] = {}

    def _add(lo: str, op: str, ro: Any):
        k0 = (lo, op)
        idx = groups.get(k0, 0)
        groups[k0] = idx + 1
        out[(lo, op, None)] = ro

    for c in constraints:
        lo = c["leftOperand"]
        op = c["operator"]
        ro = c["rightOperand"]
        if lo == "odrl:dateTime":
            if op == "odrl:isWithin" and isinstance(ro, list) and len(ro) == 2:
                a, b = ro[0], ro[1]
                a = iso_datetime_to_xsd_date(a) if isinstance(a, str) and is_iso_datetime(a) else (a if is_iso_date(a) else a)
                b = iso_datetime_to_xsd_date(b) if isinstance(b, str) and is_iso_datetime(b) else (b if is_iso_date(b) else b)
                _add(lo, "odrl:gteq", a)
                _add(lo, "odrl:lteq", b)
                continue
            if isinstance(ro, str) and is_iso_datetime(ro):
                ro = iso_datetime_to_xsd_date(ro)
            elif isinstance(ro, list):
                ro = [iso_datetime_to_xsd_date(x) if isinstance(x, str) and is_iso_datetime(x) else x for x in ro]
        _add(lo, op, ro)
    return out

def build_offer_from_issues(issues: Dict[IssueKey, Any]) -> Dict[str, Any]:
    """ Build an ODRL offer from issue dict. """
    items = sorted(issues.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    constraint = []
    for (lo, op, idx), ro in items:
        if isinstance(ro, set):
            ro = sorted(ro)
        constraint.append({"leftOperand": lo, "operator": op, "rightOperand": ro})
    return {
        "type": "odrl:Offer",
        "permission": [
            {
                "target": "urn:asset:demo-123",
                "action": "odrl:use",
                "constraint": constraint,
            }
        ],
    }

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

# ---------------------- numeric-like detection ---------------------- #
def is_numeric_like(sample: Any) -> bool:
    """Number, ISO date, or ISO duration."""
    if isinstance(sample, (int, float)):
        return True
    if isinstance(sample, str) and (is_iso_date(sample) or is_iso_duration(sample)):
        return True
    return False


def strip_key(key:str):
    key = key.split('/')[-1]
    return key.split(':')[-1]

def is_iterable(x):
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

def create_issue(left_operand, operator, value) -> Issue:
    from ..distributions.distributions import EnumDirichlet, NumericBeta, SetBelief

    if isinstance(value, (int, float)):
        m = NumericBeta(L=float(value) - 0.2*float(value), U=float(value) + 0.2*float(value))
        current = Issue(
            kind='num',
            operand=left_operand,
            operator=operator,
            oppOperator=None,
            lastValue=value,
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
            m = NumericBeta(L=base - 0.2*base, U=base + 0.2*base)
            current = Issue(
                kind='num',
                operand=left_operand,
                operator=operator,
                oppOperator=None,
                lastValue=value,
                oppLastValue=None,
                model=m,
                to_internal=lambda s: float(duration_to_seconds(s)),
                to_external=lambda secs: seconds_to_duration(float(secs)),
                hard_rule=None
            )
            return current

        if is_iso_date(value):
            base = float(iso_to_days(value))
            m = NumericBeta(L=base - 15, U=base + 15)
            current = Issue(
                kind='num',
                operand=left_operand,
                operator=operator,
                oppOperator=None,
                lastValue=value,
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
        oppLastValue=None,
        model=m,
        to_internal=lambda s: set(s) if is_iterable(s) else str(s),
        to_external=lambda s: list(s) if is_iterable(s) else str(s),
        hard_rule=None
    )
    return current

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

