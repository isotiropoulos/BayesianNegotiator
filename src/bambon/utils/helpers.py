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

def _as_number(v):
    """A numeric view of an operand, or ``None`` when it has no place on the real line.

    The inequality block at the bottom of :func:`satisfies` is only meaningful
    for scalars, but it used to call ``float()`` on whatever it was handed. ODRL
    documents in the wild carry a categorical clause and a numeric one on the
    SAME leftOperand -- the AgentODRL-derived corpora have ``purpose eq '...'``
    sitting beside ``purpose lt 2`` -- so a numeric hard rule really does get
    compared against a list of labels, and ``float(['a','b'])`` raised TypeError
    while ``float('P5DT3H')`` raised ValueError, taking the caller down with it.

    :func:`satisfies` is documented to return ``None`` for incomparable operand
    pairs, and ``_clause_ok`` in the scorer already reads ``None`` as "cannot be
    shown to be a violation". This helper is what lets it keep that promise.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if is_iso_date(v):
            return float(iso_to_days(v))
        if is_iso_duration(v):
            return float(duration_to_seconds(v))
        try:
            return float(v)
        except ValueError:
            return None
    return None


def satisfies(my_op, my_val, their_op, their_val) -> bool | None:
    from .context import _intervals_overlap, _interval
    # Normalise a singleton collection on BOTH sides. The `eq`/`neq` branches
    # below compare `my_val` as a bare label, but a categorical issue's
    # `to_internal` wraps every right operand in a set -- so an `eq` hard rule
    # arrived here as `{'education'}` and `their_val == my_val` was
    # `'education' == {'education'}` -> False. The agent then rejected the very
    # term that satisfies its own rule and could never close: every categorical
    # `eq` hard rule deadlocked the negotiation. Only `their_val` used to be
    # normalised, which is why `isAllOf` (which reads the set-ified `A`) worked
    # while `eq`, `neq`, `isAnyOf` and `isNoneOf` did not.
    if isinstance(my_val,(set, list, tuple)) and len(my_val)==1:
        my_val = list(my_val)[0]
    if isinstance(their_val,(set, list, tuple)) and len(their_val)==1:
        their_val = list(their_val)[0]

    A, B = _to_setish(my_val), _to_setish(their_val)

    # ---------- ENUM/SET logic ----------
    # Every test below is written on the set-ified operands A and B. For a scalar
    # operand A == {x}, so `x in B` is `A.issubset(B)`, `x == y` is `A == B` and
    # `x not in B` is `A.isdisjoint(B)`: the results are identical to the earlier
    # scalar formulations. Writing them this way also makes the block total --
    # `their_val in A` used to raise `TypeError: unhashable type: 'list'` whenever
    # an `eq`/`neq` clause carried a multi-label right operand, which an external
    # ODRL document may well do even though this project's own emitters normalise
    # such a clause to `isAllOf`.
    if my_op == "eq":
        if their_op == "eq":        return A == B
        if their_op == "neq":       return A != B
        if their_op == "isAnyOf":   return A.issubset(B)
        if their_op == "isAllOf":   return A.issubset(B)
        if their_op == "isNoneOf":  return A.isdisjoint(B)
        return False

    if my_op == "neq":
        if their_op == "eq":        return A != B
        if their_op == "neq":       return True
        if their_op == "isAnyOf":   return A.isdisjoint(B)
        if their_op == "isAllOf":   return A.isdisjoint(B)
        if their_op == "isNoneOf":  return True
        return False

    if my_op == "isAnyOf":
        if their_op == "eq":        return B.issubset(A)
        if their_op == "neq":       return len(A - B) > 0
        if their_op == "isAnyOf":   return len(A & B) > 0
        if their_op == "isAllOf":   return B.issubset(A)
        if their_op == "isNoneOf":  return len(A & B) == 0
        return False

    if my_op == "isAllOf":
        if their_op == "eq":        return A.issubset(B)
        if their_op == "neq":       return B.isdisjoint(A)
        if their_op == "isAnyOf":   return A.issubset(B)
        if their_op == "isAllOf":   return A.issubset(B)
        if their_op == "isNoneOf":  return len(A & B) == 0 and len(A) == 0
        return False

    if my_op == "isNoneOf":
        if their_op == "eq":        return B.isdisjoint(A)
        if their_op == "neq":       return True  # they exclude one value; I exclude A — overlap exists
        if their_op in ("isAnyOf","isAllOf"): return len(A & B) == 0
        if their_op == "isNoneOf":  return True
        return False

    # ---------- NUM/TIME logic (numbers already) ----------
    if my_op in ("lt","lteq","gt","gteq","eq","neq"):
        # Both operands have to land on the real line before any interval
        # arithmetic is attempted. When one of them is categorical the two
        # clauses constrain different domains and no overlap test is meaningful,
        # so say so with None rather than raising out of `float()`.
        _a, _b = _as_number(my_val), _as_number(their_val)
        if _a is None or _b is None:
            return None
        my_val, their_val = _a, _b

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
    # An opponent may send a value already in internal units (days since
    # BASE_DATE) instead of an ISO string — accept it as-is rather than crashing.
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return s
    if not s:
        return None
    text = s.strip()
    # ``datetime.fromisoformat`` only learned to parse a trailing ``Z``
    # (UTC designator) in Python 3.11. ODRL/JSON-LD dates routinely carry
    # it, so normalise to an explicit offset to stay correct on 3.8-3.10.
    if text and text[-1] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    d = datetime.fromisoformat(text).date()
    return (d - BASE_DATE).days


def days_to_iso(days: int) -> str:
    return (BASE_DATE + timedelta(days=int(days))).isoformat()

# ---------------------- durations (ISO-8601) ---------------------- #
def is_iso_duration(s: str) -> bool:
    return bool(_ISO_DURATION_RE.match(s))

def duration_to_seconds(s: str | None) -> float:
    """ Parse ISO-8601 duration PnYnMnWnDTnHnMnS into seconds. """
    # Accept a value already in internal units (seconds) instead of an ISO string.
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
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

    # A JSON boolean is an ``int`` subclass in Python, so this test has to come
    # BEFORE the numeric branch below: otherwise ``recipient eq true`` is built
    # as the number 1.0 with ``to_internal=float``, and a hard constraint on the
    # same operand written set-style (``recipient isAnyOf ["True"]``) crashes in
    # ``float(["True"])``. A boolean is categorical everywhere else in the code
    # base -- ``is_numeric_value`` and ``negmas_negotiator._to_num`` both reject
    # it -- and the corpora serialise it as the string "True"/"False" inside
    # list-valued operands, so normalise to that spelling and let it fall
    # through to the enum branch, whose ``to_internal`` already accepts both a
    # bare label and a list of them.
    if isinstance(value, bool):
        value = str(value)

    if isinstance(value, (int, float)):
        if num_bounds:
            L, U = num_bounds
            if L and not U:
                U = value + (value - L)
            if U and not L:
                # Mirror the known bound across the anchor (symmetric with the
                # `L and not U` case above). The previous `value + (U - value)`
                # collapsed to exactly U, giving a zero-length ruler whenever only
                # an upper hard bound was supplied (the provider-side asymmetry).
                # Floor at 0: these numeric operands (percentage, count, ...) are
                # non-negative, so the mirrored end is clamped to the domain min.
                L = max(0.0, value - (U - value))
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
                    # Mirror the known bound across the anchor (symmetric with the
                    # `L and not U` case). The previous `base + (U - base)` collapsed
                    # to exactly U, giving a zero-length ruler whenever only an upper
                    # hard bound was supplied (the provider-side asymmetry).
                    # Floor at 0: durations/dates are non-negative in internal units,
                    # so the mirrored end is clamped to the domain min (this also
                    # prevents seconds_to_duration() from receiving a negative value).
                    L = max(0.0, base - (U - base))
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
                # Durations are non-negative; the belief bound L can be driven
                # below 0 mid-negotiation (NumericBeta.observe lowers L when an
                # opponent offers a small value), which would make a sampled value
                # negative and crash seconds_to_duration. Clamp at the domain floor.
                to_external=lambda secs: seconds_to_duration(max(0.0, float(secs))),
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
                    # Mirror the known bound across the anchor (symmetric with the
                    # `L and not U` case). The previous `base + (U - base)` collapsed
                    # to exactly U, giving a zero-length ruler whenever only an upper
                    # hard bound was supplied (the provider-side asymmetry).
                    # Floor at 0: durations/dates are non-negative in internal units,
                    # so the mirrored end is clamped to the domain min (this also
                    # prevents seconds_to_duration() from receiving a negative value).
                    L = max(0.0, base - (U - base))
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
                # Dates map to non-negative day counts; clamp at the domain floor
                # for the same reason as durations (belief L can drift below 0).
                to_external=lambda d: days_to_iso(int(round(max(0.0, d)))),
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



INCLUSION_OPS = ("eq", "isAnyOf", "isAllOf")
EXCLUSION_OPS = ("neq", "isNoneOf")


def is_numeric_value(v) -> bool:
    """True if a rightOperand should be negotiated as a numeric/temporal scalar."""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str) and (is_iso_duration(v) or is_iso_date(v)):
        return True
    return False


def numeric_family(operator: str) -> str:
    """Group numeric operators so opponent bounds pair with my bounds robustly."""
    op = strip_key(operator)
    if op in ("gt", "gteq"):
        return "lower"
    if op in ("lt", "lteq"):
        return "upper"
    if op == "neq":
        return "neq"
    return "eq"


def _values_as_set(v):
    if isinstance(v, (list, tuple, set)):
        return {x for x in v}
    return {v}


def split_categorical_clauses(clauses):
    """Split same-operand categorical clauses into (allowed, forbidden, incl_op).

    ``clauses`` is a list of dicts with (already stripped) keys
    ``operator``/``rightOperand``. Inclusion clauses (eq/isAnyOf/isAllOf) feed the
    allowed pool; exclusion clauses (neq/isNoneOf) feed the forbidden set.
    """
    allowed: list = []
    forbidden: set = set()
    has_isallof = False
    eq_values: set = set()
    for c in clauses:
        op = strip_key(c["operator"])
        vals = _values_as_set(c["rightOperand"])
        if op in EXCLUSION_OPS:
            forbidden |= vals
        else:
            allowed.extend(vals)
            if op == "isAllOf":
                has_isallof = True
            if op == "eq":
                eq_values |= vals

    allowed_unique = sorted({str(a) for a in allowed})
    forbidden = {str(f) for f in forbidden}

    if allowed_unique:
        if has_isallof:
            incl_op = "isAllOf"
        elif eq_values and len(allowed_unique) == 1:
            incl_op = "eq"
        else:
            incl_op = "isAnyOf"
    else:
        incl_op = "isNoneOf"

    return allowed_unique, forbidden, incl_op


def create_merged_categorical_issue(left_operand, clauses, hard_constraints=None) -> Issue:
    """Build ONE issue representing all categorical clauses on the same operand."""
    allowed, forbidden, incl_op = split_categorical_clauses(clauses)

    primary_value = allowed if allowed else sorted(forbidden)
    issue = create_issue(left_operand, incl_op, list(primary_value))

    issue.merged = True
    issue.forbidden = set(forbidden)
    issue.opp_forbidden = set()

    # register forbidden labels with the belief so they are part of the universe
    if forbidden:
        issue.model.observe(set(forbidden), "isNoneOf", lr=0.0)

    if hard_constraints:
        issue.hard_rule = [
            HardRule(item["operator"], issue.to_internal(item["rightOperand"]))
            for item in hard_constraints
            if strip_key(item["leftOperand"]) == left_operand
        ]
    return issue


def has_numeric_range(clauses) -> bool:
    """True if same-operand numeric clauses contain BOTH a lower and an upper bound."""
    has_lo = any(strip_key(c["operator"]) in ("gt", "gteq") for c in clauses)
    has_hi = any(strip_key(c["operator"]) in ("lt", "lteq") for c in clauses)
    return has_lo and has_hi


def create_merged_numeric_issue(left_operand, clauses, hard_constraints=None) -> Issue:
    """Build ONE interval issue from same-operand numeric clauses (lower + upper)."""
    from ..distributions.distributions import NumericBeta

    lower_vals, upper_vals = [], []
    lower_op, upper_op = "gteq", "lteq"
    for c in clauses:
        op = strip_key(c["operator"])
        v = c["rightOperand"]
        if op in ("gt", "gteq"):
            lower_vals.append(v)
            if op == "gt":
                lower_op = "gt"
        elif op in ("lt", "lteq"):
            upper_vals.append(v)
            if op == "lt":
                upper_op = "lt"

    rep = (lower_vals or upper_vals)[0]
    base = create_issue(left_operand, lower_op, rep)
    to_internal, to_external = base.to_internal, base.to_external

    lo_i = max(to_internal(v) for v in lower_vals) if lower_vals else None
    hi_i = min(to_internal(v) for v in upper_vals) if upper_vals else None

    if lo_i is not None and hi_i is not None:
        span = max(abs(hi_i - lo_i), 1.0)
        L, U = lo_i - 0.5 * span, hi_i + 0.5 * span
    elif lo_i is not None:
        span = max(abs(lo_i), 1.0)
        L, U = lo_i - 0.5 * span, lo_i + 1.0 * span
    else:
        span = max(abs(hi_i), 1.0)
        L, U = hi_i - 1.0 * span, hi_i + 0.5 * span

    base.kind = "numrange"
    base.merged = True
    base.operator = "range"
    base.model = NumericBeta(L=L, U=U)
    base.model_hi = NumericBeta(L=L, U=U)
    base.lower_op = lower_op if lower_vals else None
    base.upper_op = upper_op if upper_vals else None
    base.lastValue = [
        to_external(lo_i) if lo_i is not None else None,
        to_external(hi_i) if hi_i is not None else None,
    ]
    base.initValue = list(base.lastValue)
    base.oppLastValue = None

    if hard_constraints:
        base.hard_rule = [
            HardRule(item["operator"], to_internal(item["rightOperand"]))
            for item in hard_constraints
            if strip_key(item["leftOperand"]) == left_operand
        ]
    return base


def merged_issue_clauses(incl_op, allowed_value, forbidden):
    """Render a merged categorical issue back into ODRL (operator, value) clauses."""
    out = []
    allowed_set = _values_as_set(allowed_value) if allowed_value is not None else set()
    if incl_op in INCLUSION_OPS and allowed_set:
        op, val = format_operator_roperand(incl_op, sorted(allowed_set))
        out.append((op, val))
    if forbidden:
        op, val = format_operator_roperand("isNoneOf", sorted(forbidden))
        out.append((op, val))
    if not out and allowed_set:
        op, val = format_operator_roperand("isNoneOf", sorted(allowed_set))
        out.append((op, val))
    return out


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