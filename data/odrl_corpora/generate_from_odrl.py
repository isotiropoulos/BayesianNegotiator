#!/usr/bin/env python
"""Build the five BAMBON ZOPA corpora from a corpus of real ODRL Offers.

The seeded generator in ``data/generator/generate_scenarios.py`` synthesises
every document from an operand catalogue. This script does the same job from
the other end: it takes offers produced by AgentODRL (``offers_negotiation.json``)
and treats each one as the *provider's aspiration*, then derives the five
remaining documents of a scenario around it.

For every input offer and every family it writes one scenario directory holding
the same six files as the published corpora::

    provider.json        the input offer, unchanged (the provider's opening A_p)
    hard_provider.json   the provider's non-negotiable policy H_p
    consumer.json        the consumer's opening A_c, opposed on every issue
    hard_consumer.json   the consumer's non-negotiable policy H_c
    witness.json         a point agreement satisfying H_p and H_c
    meta.json            scenario geometry, provenance and verification result

Geometry follows Table 1 of the paper. Each ordered issue is laid out as
``a_p <= h_c <= h_p <= a_c`` in the frame where the provider prefers the small
value, so each party's ideal lies outside the other's lawful range while the
lawful ranges still overlap on ``[h_c, h_p]``:

    family            opening gap   ZOPA / gap   split
    far_wide             0.80         0.25-1.00   50/50
    far_narrow           0.80         0.10-0.25   50/50
    close_narrow         0.20         0.10-0.25   50/50
    far_narrow_asym      0.80         0.10-0.25   20/80
    full_infeasible      0.80         empty        --

The four feasible families share a per-case seed, so their categorical halves
are identical and only the numeric geometry differs. ``full_infeasible``
crosses the two hard bounds on exactly one ordered issue, which is then the
sole entry in ``blocking_operands``.

Usage::

    python data/odrl_corpora/generate_from_odrl.py \\
        --input ../AgentODRL/ODRL_V3/result/offers_negotiation.json

    python data/odrl_corpora/generate_from_odrl.py -i offers.json \\
        --out data/generated_agentodrl --families far_wide,far_narrow --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from datetime import date as _date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- reuse the BAMBON helpers -------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.bambon.utils.helpers import (           # noqa: E402
    days_to_iso, iso_to_days, duration_to_seconds, seconds_to_duration,
)
from src.bambon.models.expressions import BASE_DATE      # noqa: E402

# The internal axis for a date is days since BASE_DATE, and `days_to_iso` builds
# a real `datetime.date`, so the axis is bounded by what that type can hold.
# Corpora use 9999-12-31 as a 'no expiry' sentinel, which sits exactly on the
# upper bound: any geometry laid out above it raises OverflowError instead of
# producing a scenario, so the bound is carried as the operand's domain.
_DATE_MIN_DAYS = float((_date.min - BASE_DATE).days)
_DATE_MAX_DAYS = float((_date.max - BASE_DATE).days)

try:                                             # optional cross-check
    from src.bambon.baselines.veto import feasibility_report   # noqa: E402
except Exception:                                # pragma: no cover
    feasibility_report = None


# ======================================================================
# 1. families
# ======================================================================
NUM, DATE, DUR, CAT = "num", "date", "duration", "cat"

FAMILIES: Dict[str, Dict[str, Any]] = {
    "far_wide":        dict(window=(0.25, 1.00), opening=1.00, skew=0.5,
                            feasible=True),
    "far_narrow":      dict(window=(0.10, 0.25), opening=1.00, skew=0.5,
                            feasible=True),
    "close_narrow":    dict(window=(0.10, 0.25), opening=0.25, skew=0.5,
                            feasible=True),
    "far_narrow_asym": dict(window=(0.10, 0.25), opening=1.00,
                            skew_choices=(0.20, 0.80), feasible=True),
    "full_infeasible": dict(window=1.00, opening=1.00, skew=0.5,
                            feasible=False),
}

ALL_OPERATORS = {"eq", "gt", "gteq", "isAllOf", "isAnyOf", "isNoneOf",
                 "lt", "lteq", "neq"}
SET_OPERATORS = {"isAnyOf", "isAllOf", "isNoneOf"}
ORDER_OPERATORS = {"lt", "lteq", "gt", "gteq"}
RULE_TYPES = ("permission", "prohibition", "obligation")

# Strict / exclusion operators for the hard policies, as in the seeded generator.
HARD_NUM_OPS = {"low": ["lteq", "lt"], "high": ["gteq", "gt"]}
HARD_CAT_OPS = ["isAnyOf", "eq", "isAllOf", "isNoneOf", "neq"]

# Preference direction for operands the seeded generator knows about. Anything
# absent is inferred from the operator the provider actually used.
DIRECTION = {
    "count": "low", "percentage": "low", "unitOfCount": "low",
    "elapsedTime": "low", "meteredTime": "low", "samplingRate": "low",
    "absoluteTemporalPosition": "low", "absoluteSize": "low",
    "absolutePosition": "low", "relativeSize": "low", "relativePosition": "low",
    "payAmount": "high", "frequency": "high", "spatialResolution": "high",
    "resolution": "high", "confidenceLevel": "high", "delayPeriod": "high",
    "timeInterval": "high", "relativeTemporalPosition": "high",
    "version": "high",
}

# Label pools for known categorical operands, so a one-label offer still has a
# universe to disagree over.
CATALOGUE_LABELS = {
    "purpose": ["research", "education", "commercial", "marketing",
                "journalism", "statistics", "audit"],
    "recipient": ["internal-only", "consortium", "affiliate", "third-party",
                  "public", "regulator"],
    "fileFormat": ["CSV", "PARQUET", "JSON", "XML", "RDF", "XLSX"],
    "industry": ["health", "finance", "energy", "mobility", "public",
                 "retail", "defence"],
    "language": ["en", "el", "de", "fr", "es"],
    "spatial": ["EU", "EEA", "CH", "UK", "US", "global"],
    "virtualLocation": ["on-prem", "eu-cloud", "us-cloud", "hybrid",
                        "air-gapped", "edge"],
    "systemDevice": ["hsm", "tee", "vm", "container", "bare-metal"],
    "deliveryChannel": ["sftp", "https", "kafka", "s3", "courier"],
    "media": ["print", "web", "broadcast", "internal-wiki"],
    "event": ["breach", "audit", "renewal", "termination"],
    "product": ["raw", "derived", "aggregate", "report"],
    "system": ["linux", "windows", "macos", "rtos"],
}

# Operands whose value domain is bounded regardless of what the offer says.
HARD_DOMAINS = {"percentage": (0.0, 100.0), "samplingRate": (0.0, 1.0),
                "confidenceLevel": (0.0, 1.0)}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DUR_RE = re.compile(r"^-?P(?=[\dT])")

# Days since 0001-01-01 for 2026-01-01, matching the seeded generator's _DAY0.
_DAY0 = 739616

# Catalogue used to pad a thin offer up to a usable issue count. These are the
# operands and ranges of the seeded generator, so a padded issue is
# indistinguishable in kind from one in the published corpora.
PAD_OPERANDS: List[Dict[str, Any]] = [
    dict(operand="count", kind=NUM, lo=100, hi=2_000_000),
    dict(operand="percentage", kind=NUM, lo=1, hi=100),
    dict(operand="payAmount", kind=NUM, lo=50, hi=50_000),
    dict(operand="unitOfCount", kind=NUM, lo=1, hi=10_000),
    dict(operand="spatialResolution", kind=NUM, lo=1, hi=5000),
    dict(operand="resolution", kind=NUM, lo=1, hi=600),
    dict(operand="version", kind=NUM, lo=1, hi=20),
    dict(operand="samplingRate", kind=NUM, lo=0.001, hi=1.0, float_=True),
    dict(operand="confidenceLevel", kind=NUM, lo=0.5, hi=0.999, float_=True),
    dict(operand="absoluteTemporalPosition", kind=DATE,
         lo=_DAY0 + 30, hi=_DAY0 + 1500),
    dict(operand="relativeTemporalPosition", kind=DATE,
         lo=_DAY0 + 10, hi=_DAY0 + 900),
    dict(operand="elapsedTime", kind=DUR, lo=3600, hi=365 * 86400),
    dict(operand="delayPeriod", kind=DUR, lo=3600, hi=90 * 86400),
    dict(operand="meteredTime", kind=DUR, lo=60, hi=30 * 86400),
    dict(operand="timeInterval", kind=DUR, lo=3600, hi=180 * 86400),
]

PAD_CATEGORICAL: List[str] = [
    "purpose", "recipient", "fileFormat", "industry", "language", "spatial",
    "virtualLocation", "systemDevice", "deliveryChannel", "media", "event",
]


# ======================================================================
# 2. value typing, parsing and rendering
# ======================================================================
def _parses(kind: str, value: Any) -> bool:
    """Can the BAMBON helpers actually read this literal in that kind?

    A generated corpus contains literals that look temporal and are not:
    'P36H' (the T separator missing), '2025-13-01', 'PT1H30' . Trusting the
    shape alone turns those into an exception in the middle of a 1837-offer
    run, so the shape is confirmed by parsing before the kind is assigned.
    """
    try:
        to_internal(kind, value)
        return True
    except Exception:
        return False


def classify(operand: str, operator: str, value: Any) -> str:
    if operator in SET_OPERATORS or isinstance(value, list):
        return CAT
    if isinstance(value, bool):
        return CAT
    if isinstance(value, (int, float)):
        return NUM
    if isinstance(value, str):
        if _DUR_RE.match(value):
            return DUR if _parses(DUR, value) else CAT
        if _DATE_RE.match(value):
            return DATE if _parses(DATE, value) else CAT
        try:
            float(value)
            return NUM
        except ValueError:
            return CAT
    return CAT


def to_internal(kind: str, value: Any) -> float:
    """Map a rendered right operand onto the common scalar axis."""
    if kind == DATE:
        return float(iso_to_days(str(value)))
    if kind == DUR:
        return float(duration_to_seconds(str(value)))
    return float(value)


def render(kind: str, value: float, template: Any) -> Any:
    """Map a scalar back into the operand's own units, matching `template`."""
    if kind == DATE:
        # The geometry is clamped to the date domain upstream; this is the last
        # line of defence, so that no rounding at the edge can raise.
        day = min(max(float(value), _DATE_MIN_DAYS), _DATE_MAX_DAYS)
        return days_to_iso(int(round(day)))
    if kind == DUR:
        return seconds_to_duration(max(0.0, float(value)))
    if isinstance(template, int) and not isinstance(template, bool):
        return int(round(value))
    if isinstance(template, str):                # numeric-looking string
        return str(round(float(value), 4))
    decimals = 4
    if isinstance(template, float):
        text = repr(template)
        decimals = max(2, len(text.split(".")[1])) if "." in text else 2
        decimals = min(decimals, 6)
    return round(float(value), decimals)


def clause(operand: str, operator: str, value: Any) -> Dict[str, Any]:
    return {"leftOperand": "odrl:%s" % operand,
            "operator": "odrl:%s" % operator,
            "rightOperand": value}


def short(term: Any) -> str:
    text = str(term)
    return text.split(":", 1)[1] if ":" in text else text


# ======================================================================
# 3. geometry
# ======================================================================
def derive_span(kind: str, operand: str, a_p: float, template: Any) -> float:
    """A declared range for the operand, wide enough that `a_p` sits ~10% in.

    The seeded generator reads lo/hi from its catalogue. Real offers declare no
    range, so one is constructed around the value the provider actually asked
    for, which is what keeps the opening gap at the 0.798 of Table 1.
    """
    if operand in HARD_DOMAINS:
        lo, hi = HARD_DOMAINS[operand]
        return float(hi - lo)
    if kind == DATE:
        return 1095.0                                  # three years, in days
    if kind == DUR:
        return max(2.0 * abs(a_p), 30.0 * 86400.0)
    magnitude = abs(a_p)
    if magnitude <= 1.0:
        return 1.0
    floor = 10.0 if isinstance(template, int) else 1.0
    return max(2.0 * magnitude, floor)


def operand_bounds(kind: str, operand: str,
                   a_p: float) -> Tuple[Optional[float], Optional[float]]:
    """The natural domain of the operand, used to keep the layout admissible.

    A `high`-direction operand puts the consumer's opening *below* the
    provider's, and a range constructed around a small value then runs past
    zero. A negative duration renders as PT0S, which collapses the window and
    both hard bounds onto the same point, so the witness can satisfy neither.
    """
    if operand in HARD_DOMAINS:
        return HARD_DOMAINS[operand]
    if kind == DUR:
        return 0.0, None
    if kind == DATE:
        # Bounded at both ends: a date past 9999-12-31 cannot be rendered at
        # all, and '9999-12-31' is exactly the value a corpus uses to mean
        # 'never expires', so it arrives sitting on the ceiling.
        return _DATE_MIN_DAYS + 1.0, _DATE_MAX_DAYS
    if a_p >= 0:
        return 0.0, None
    return None, None


def _clamp(value: float, floor: Optional[float],
           ceiling: Optional[float]) -> float:
    if floor is not None:
        value = max(value, floor)
    if ceiling is not None:
        value = min(value, ceiling)
    return value


def _fit_points(points: Sequence[float], floor: Optional[float],
                ceiling: Optional[float]) -> List[float]:
    """Move a group of points into the domain without changing their spacing.

    Clamping each point on its own would silently uncross the two hard bounds
    of an infeasible scenario, and would pull the witness off the centre of the
    lawful window in a feasible one. Translating the whole group keeps both
    relations intact; only a group wider than the domain itself is clamped.
    """
    values = [float(p) for p in points]
    shift = 0.0
    if ceiling is not None and max(values) > ceiling:
        shift = ceiling - max(values)
    if floor is not None and min(values) + shift < floor:
        shift = floor - min(values)
    values = [v + shift for v in values]
    return [_clamp(v, floor, ceiling) for v in values]


def numeric_layout(a_p: float, span: float, direction: str, window: float,
                   opening: float, skew: float, feasible: bool,
                   bounds: Tuple[Optional[float], Optional[float]] = (None, None),
                   ) -> Tuple[float, float, float, float, float]:
    """Return ``(a_p, h_c, h_p, a_c, witness)`` in the provider-prefers-low frame.

    Mirrors ``_numeric_layout`` in the seeded generator: the lawful window is a
    ``window`` fraction of the opening gap, ``opening`` scales the gap about the
    centre of the declared range, and ``skew`` places the window's centre along
    the segment from ``a_p`` to ``a_c``.
    """
    if direction == "low":
        lo = a_p - 0.10 * span
    else:
        lo = a_p + 0.10 * span - span
    hi = lo + span
    other = hi - 0.10 * span if direction == "low" else lo + 0.10 * span

    # The seeded generator shrinks both openings toward the centre of the
    # declared range. Here the provider's opening is not free: it is the value
    # the real offer states and passes through verbatim. So the gap is closed
    # from the consumer's side only, anchored on a_p, which scales the opening
    # distance by `opening` while keeping the window centred between the two
    # openings that are actually emitted.
    if opening != 1.0:
        other = a_p + opening * (other - a_p)

    # Keep the consumer's end inside the operand's domain. The provider's end is
    # fixed at the value the offer states, so the clamp lands here; everything
    # else (window, witness) is derived from the interval between the two ends
    # and is therefore admissible by construction.
    floor, ceiling = bounds
    other = _clamp(other, floor, ceiling)
    if abs(other - a_p) < 1e-9:
        # The clamp collapsed the gap, because the provider's own value sits on
        # the edge of the domain -- '9999-12-31' meaning 'no expiry' is the
        # common case. Open the issue away from that edge, whichever side has
        # room, rather than emit a degenerate one.
        for candidate in (a_p + 0.8 * span, a_p - 0.8 * span):
            candidate = _clamp(candidate, floor, ceiling)
            if abs(candidate - a_p) > 1e-9:
                other = candidate
                break
        direction = "low" if direction == "high" else "high"

    low_end, high_end = ((a_p, other) if other >= a_p else (other, a_p))

    mid = (0.5 * (low_end + high_end) if skew == 0.5
           else low_end + skew * (high_end - low_end))
    half = 0.5 * window * (high_end - low_end) * 0.6
    h_c, h_p = mid - half, mid + half

    if not feasible:
        # cross the hard bounds: the provider's ceiling sits below the
        # consumer's floor, so [h_c, h_p] is empty
        gap = max(1.0, 0.12 * span)
        h_c, h_p = mid + gap / 2.0, mid - gap / 2.0

    # The window and its witness are derived from the two ends, and the ends are
    # inside the domain, but the infeasible crossing pushes one bound outside it
    # by half a gap. Translate the three together so the crossing survives.
    h_c, h_p, mid = _fit_points((h_c, h_p, mid), floor, ceiling)

    return low_end, h_c, h_p, high_end, mid


# ======================================================================
# 4. issue construction
# ======================================================================
def build_numeric_issue(rng: random.Random, operand: str, kind: str,
                        p_operator: str, p_value: Any, cfg: Dict[str, Any],
                        window: float, skew: float,
                        feasible: bool) -> Dict[str, Any]:
    a_p_internal = to_internal(kind, p_value)
    direction = DIRECTION.get(operand)
    if direction is None:
        direction = "high" if p_operator in ("gteq", "gt") else "low"

    span = derive_span(kind, operand, a_p_internal, p_value)
    bounds = operand_bounds(kind, operand, a_p_internal)
    low_end, h_c, h_p, high_end, witness = numeric_layout(
        a_p_internal, span, direction, window, cfg["opening"],
        skew if direction == "low" else 1.0 - skew, feasible, bounds)

    # numeric_layout may flip the direction when a domain clamp collapses the
    # gap, so re-derive it from where the provider's end actually landed.
    direction = "low" if abs(low_end - a_p_internal) < 1e-9 else "high"
    a_c_internal = high_end if direction == "low" else low_end
    if direction == "low":
        p_hard_op = rng.choice(HARD_NUM_OPS["low"])
        c_hard_op = rng.choice(HARD_NUM_OPS["high"])
        c_op = "gteq"
        hp_val, hc_val = h_p, h_c
    else:
        p_hard_op = rng.choice(HARD_NUM_OPS["high"])
        c_hard_op = rng.choice(HARD_NUM_OPS["low"])
        c_op = "lteq"
        hp_val, hc_val = h_c, h_p

    # Keep every emitted point inside the operand's domain. The provider's own
    # clause is passed through verbatim and may itself lie outside it (an offer
    # that caps a percentage at 150), so only the derived values are moved, and
    # the three that must keep their relative order move together.
    floor, ceiling = bounds
    a_c_internal = _clamp(a_c_internal, floor, ceiling)
    hp_val, hc_val, witness = _fit_points((hp_val, hc_val, witness),
                                          floor, ceiling)

    # Rendering is lossy: an integer operand with a sub-unit window rounds both
    # hard bounds and the witness onto the same value, and a strict bound then
    # has no room left. Widen the window to two units, and if the rendered
    # witness still fails either bound, fall back to the non-strict operator
    # rather than emit a scenario whose own witness does not verify.
    if feasible:
        renders_int = (isinstance(p_value, int) and not isinstance(p_value, bool)
                       and kind == NUM)
        if renders_int and abs(hp_val - hc_val) < 2.0:
            lo_h, hi_h = min(hp_val, hc_val), max(hp_val, hc_val)
            centre = 0.5 * (lo_h + hi_h)
            lo_h, hi_h = centre - 1.0, centre + 1.0
            hp_val, hc_val = ((hi_h, lo_h) if hp_val >= hc_val else (lo_h, hi_h))
            witness = centre

        w_rendered = render(kind, witness, p_value)
        if not satisfies("eq", w_rendered, p_hard_op,
                         render(kind, hp_val, p_value), kind):
            p_hard_op = {"lt": "lteq", "gt": "gteq"}.get(p_hard_op, p_hard_op)
        if not satisfies("eq", w_rendered, c_hard_op,
                         render(kind, hc_val, p_value), kind):
            c_hard_op = {"lt": "lteq", "gt": "gteq"}.get(c_hard_op, c_hard_op)

    return dict(
        operand=operand, kind=kind, direction=direction, ordered=True,
        # No room was left on either side of the provider's value: the issue is
        # emitted so the offer keeps its clause, but it carries no ZOPA width
        # and is counted in meta rather than passed off as negotiable geometry.
        degenerate=abs(high_end - low_end) < 1e-9,
        p_clause=clause(operand, p_operator, p_value),
        c_clause=clause(operand, c_op, render(kind, a_c_internal, p_value)),
        hp_clause=clause(operand, p_hard_op, render(kind, hp_val, p_value)),
        hc_clause=clause(operand, c_hard_op, render(kind, hc_val, p_value)),
        w_clause=clause(operand, "eq", render(kind, witness, p_value)),
        crossed=not feasible,
    )


def categorical_sets(p_operator: str, p_value: Any,
                     universe: Sequence[str]) -> List[str]:
    """The set of labels the provider's opening clause admits."""
    values = p_value if isinstance(p_value, list) else [p_value]
    values = [str(v) for v in values]
    if p_operator in ("isNoneOf", "neq"):
        allowed = [l for l in universe if l not in values]
        return allowed or list(values)
    return values


def cat_hard_value(operator: str, allow: Sequence[str], core: Sequence[str],
                   universe: Sequence[str]) -> Any:
    if operator == "eq":
        return core[0]
    if operator == "neq":
        banned = [l for l in universe if l not in core]
        return banned[-1] if banned else universe[-1]
    if operator == "isAllOf":
        return list(core[:1])
    if operator == "isNoneOf":
        banned = [l for l in universe if l not in core][-1:]
        return banned or [universe[-1]]
    return list(allow)                                   # isAnyOf


def cat_witness(pairs: Sequence[Tuple[str, Any]], core: Sequence[str],
                universe: Sequence[str]) -> Any:
    """A label satisfying both categorical hard rules, when one exists."""
    def as_set(value: Any) -> set:
        return set(value) if isinstance(value, (list, tuple, set)) else {value}

    banned: set = set()
    required: set = set()
    allowed: Optional[set] = None
    for operator, value in pairs:
        values = as_set(value)
        if operator == "eq":
            required |= values
            allowed = values if allowed is None else (allowed & values)
        elif operator == "isAllOf":
            required |= values
        elif operator == "isAnyOf":
            allowed = values if allowed is None else (allowed & values)
        elif operator in ("isNoneOf", "neq"):
            banned |= values

    candidates = required - banned
    if not candidates:
        pool = (allowed if allowed is not None else set(universe)) - banned
        candidates = {sorted(pool)[0]} if pool else {core[0]}
    return sorted(candidates)[0]


def build_categorical_issue(rng: random.Random, operand: str, p_operator: str,
                            p_value: Any,
                            universe: Sequence[str]) -> Dict[str, Any]:
    p_allow = categorical_sets(p_operator, p_value, universe)
    core = [p_allow[0]]

    outside = [l for l in universe if l not in p_allow]
    rng.shuffle(outside)
    c_allow = core + outside[:2] if outside else list(p_allow)

    # An exclusion operator needs a label outside the shared core to forbid.
    # Where the operand's universe has no such label -- a one-value operand, or
    # a numeric operand carrying a string -- banning anything would ban the only
    # lawful term, so those operators are withheld.
    excludable = [l for l in universe if l not in core]
    choices = ([o for o in HARD_CAT_OPS if o not in ("isNoneOf", "neq")]
               if not excludable else list(HARD_CAT_OPS))
    p_hard_op = rng.choice(choices)
    c_hard_op = rng.choice(choices)
    p_hard_val = cat_hard_value(p_hard_op, p_allow, core, universe)
    c_hard_val = cat_hard_value(c_hard_op, c_allow, core, universe)
    witness = cat_witness([(p_hard_op, p_hard_val), (c_hard_op, c_hard_val)],
                          core, universe)

    # Last resort: if the pair still admits no lawful label, both sides fall
    # back to accepting the witness, so a feasible family stays feasible.
    for operator, value in ((p_hard_op, p_hard_val), (c_hard_op, c_hard_val)):
        if not satisfies("eq", witness, operator, value, CAT):
            p_hard_op = c_hard_op = "isAnyOf"
            p_hard_val, c_hard_val = list(p_allow), list(c_allow)
            witness = core[0]
            break

    return dict(
        operand=operand, kind=CAT, direction="low", ordered=False,
        p_clause=clause(operand, p_operator, p_value),
        c_clause=clause(operand, "isAnyOf", c_allow),
        hp_clause=clause(operand, p_hard_op, p_hard_val),
        hc_clause=clause(operand, c_hard_op, c_hard_val),
        w_clause=clause(operand, "eq", witness),
        crossed=False,
    )


# ======================================================================
# 5. satisfaction predicate, for verifying the witness
# ======================================================================
def satisfies(point_op: str, point_val: Any, hard_op: str, hard_val: Any,
              kind: str) -> bool:
    """Does a committed term ``eq point_val`` satisfy the hard requirement?"""
    if point_op != "eq":
        return True                                   # only points are checked

    if hard_op in SET_OPERATORS or hard_op in ("eq", "neq"):
        committed = {str(point_val)}
        required = ({str(v) for v in hard_val}
                    if isinstance(hard_val, list) else {str(hard_val)})
        if hard_op == "isAnyOf":
            return bool(committed & required)
        if hard_op == "isAllOf":
            return required <= committed
        if hard_op == "isNoneOf":
            return not (committed & required)
        if hard_op == "eq":
            return committed == required
        return not (committed & required)             # neq

    try:
        x = to_internal(kind, point_val)
        y = to_internal(kind, hard_val)
    except Exception:
        return True
    if hard_op == "lteq":
        return x <= y + 1e-9
    if hard_op == "lt":
        return x < y - 1e-9
    if hard_op == "gteq":
        return x >= y - 1e-9
    if hard_op == "gt":
        return x > y + 1e-9
    return True


# ======================================================================
# 6. document assembly
# ======================================================================
def parse_offer(offer: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the offer into an ordered list of (rule_type, action, constraint)."""
    slots = []
    for rule_type in RULE_TYPES:
        for rule_index, rule in enumerate(offer.get(rule_type) or []):
            action = rule.get("action")
            for c_index, constraint in enumerate(rule.get("constraint") or []):
                slots.append(dict(
                    rule_type=rule_type, rule_index=rule_index, action=action,
                    constraint_index=c_index,
                    operand=short(constraint.get("leftOperand")),
                    operator=short(constraint.get("operator")),
                    value=constraint.get("rightOperand"),
                ))
    return slots


def prune_offer(offer: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Drop rules carrying no constraints.

    A constraint-less rule induces no issue, so there is nothing for the two
    parties to be apart on; it would appear in every document as an empty rule
    and the repository's verifier reports it as a structural problem. The rule
    is removed from all six documents, including the provider's, so a scenario
    contains exactly the negotiable surface of the offer.
    """
    pruned = dict(offer)
    dropped = 0
    for rule_type in RULE_TYPES:
        rules = offer.get(rule_type)
        if not rules:
            pruned.pop(rule_type, None)
            continue
        kept = [r for r in rules if r.get("constraint")]
        dropped += len(rules) - len(kept)
        if kept:
            pruned[rule_type] = kept
        else:
            pruned.pop(rule_type, None)
    return pruned, dropped


def label_universe(offers: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Every label seen for each operand across the whole input corpus."""
    seen: Dict[str, List[str]] = {}
    for offer in offers:
        for slot in parse_offer(offer):
            if classify(slot["operand"], slot["operator"], slot["value"]) != CAT:
                continue
            values = (slot["value"] if isinstance(slot["value"], list)
                      else [slot["value"]])
            bucket = seen.setdefault(slot["operand"], [])
            for value in values:
                text = str(value)
                if text not in bucket:
                    bucket.append(text)
    for operand, catalogue in CATALOGUE_LABELS.items():
        bucket = seen.setdefault(operand, [])
        for label in catalogue:
            if label not in bucket:
                bucket.append(label)
    return seen


def rule_skeleton(slots: Sequence[Dict[str, Any]]) -> List[Tuple[str, int, Any]]:
    """The ordered (rule_type, rule_index, action) triples the slots occupy.

    Derived from the slots rather than from the offer, so a padded rule that the
    offer never contained is rebuilt exactly like a real one.
    """
    seen: List[Tuple[str, int, Any]] = []
    for rule_type in RULE_TYPES:
        for slot in slots:
            if slot["rule_type"] != rule_type:
                continue
            key = (rule_type, slot["rule_index"], slot["action"])
            if key not in seen:
                seen.append(key)
    return seen


def assemble(offer: Dict[str, Any], slots: Sequence[Dict[str, Any]],
             issues: Sequence[Dict[str, Any]], clause_key: str,
             doc_type: str, doc_id: str) -> Dict[str, Any]:
    """Rebuild the offer's rule structure with one party's clauses substituted."""
    rules: Dict[str, List[Dict[str, Any]]] = {}
    for rule_type, rule_index, action in rule_skeleton(slots):
        constraints = []
        for slot, issue in zip(slots, issues):
            if slot["rule_type"] == rule_type and slot["rule_index"] == rule_index:
                constraints.append(json.loads(json.dumps(issue[clause_key])))
        rules.setdefault(rule_type, []).append({"action": action,
                                                "constraint": constraints})

    document: Dict[str, Any] = {
        "@context": offer.get("@context", "http://www.w3.org/ns/odrl.jsonld"),
        "@type": doc_type,
        "@id": doc_id,
        "assigner": offer.get("assigner") or "did:provider:P",
        "assignee": offer.get("assignee") or ["did:consumer:C"],
        "target": offer.get("target") or ["urn:asset:dataset:unknown"],
    }
    for rule_type in RULE_TYPES:
        if rule_type in rules:
            document[rule_type] = rules[rule_type]
    return document


def pad_slots(rng: random.Random, slots: List[Dict[str, Any]],
              min_ordered: int, min_issues: int) -> List[Dict[str, Any]]:
    """Top a thin offer up to a usable issue count.

    A generated ODRL offer is often constraint-poor, and the ZOPA geometry lives
    entirely on ordered issues: an offer with none has no numeric window to place
    and can yield no scenario at all. Padding draws further issues from the same
    operand catalogue the seeded generator uses, so every case produces a
    scenario in every family. Padded slots are marked and counted in meta, and
    the real clauses are never touched.
    """
    used = {slot["operand"] for slot in slots}
    ordered = sum(1 for slot in slots
                  if classify(slot["operand"], slot["operator"],
                              slot["value"]) != CAT)

    # Spread the padding over all three rule types. The published corpora carry
    # permissions, prohibitions and duties in every scenario; an offer that
    # states only permissions would otherwise produce a scenario the repository
    # verifier flags for a missing rule type.
    default_actions = {"permission": "use", "prohibition": "share",
                       "obligation": "notify"}
    hosts: List[Tuple[str, int, Any]] = []
    for rule_type in RULE_TYPES:
        existing = [s for s in slots if s["rule_type"] == rule_type]
        if existing:
            hosts.append((rule_type, existing[0]["rule_index"],
                          existing[0]["action"]))
        else:
            hosts.append((rule_type, 0, default_actions[rule_type]))

    def host(position: int) -> Tuple[str, int, Any]:
        return hosts[position % len(hosts)]

    padded: List[Dict[str, Any]] = []

    numeric_pool = [spec for spec in PAD_OPERANDS if spec["operand"] not in used]
    rng.shuffle(numeric_pool)
    while ordered + len(padded) < min_ordered and numeric_pool:
        spec = numeric_pool.pop()
        lo, hi = float(spec["lo"]), float(spec["hi"])
        anchor = lo + 0.10 * (hi - lo) * rng.uniform(0.5, 1.5)
        value = render(spec["kind"], anchor,
                       0.0 if spec.get("float_") else (0 if spec["kind"] == NUM else ""))
        rule_type, rule_index, action = host(len(padded))
        padded.append(dict(rule_type=rule_type, rule_index=rule_index,
                           action=action, constraint_index=-1,
                           operand=spec["operand"], operator="lteq",
                           value=value, padded=True))

    cat_pool = [name for name in PAD_CATEGORICAL if name not in used]
    rng.shuffle(cat_pool)
    while len(slots) + len(padded) < min_issues and cat_pool:
        operand = cat_pool.pop()
        labels = list(CATALOGUE_LABELS.get(operand) or [])
        if len(labels) < 3:
            continue
        rng.shuffle(labels)
        rule_type, rule_index, action = host(len(padded))
        padded.append(dict(rule_type=rule_type, rule_index=rule_index,
                           action=action, constraint_index=-1,
                           operand=operand, operator="isAnyOf",
                           value=labels[:3], padded=True))

    return padded


def build_scenario(offer: Dict[str, Any], index: int, family: str,
                   universe: Dict[str, List[str]], seed: int,
                   min_ordered: int = 0,
                   min_issues: int = 0) -> Optional[Dict[str, Any]]:
    cfg = FAMILIES[family]
    feasible = cfg["feasible"]
    case = "%04d_%s" % (index, family)
    offer, dropped_rules = prune_offer(offer)

    # The categorical half is drawn from a per-case stream shared by every
    # family, so the four feasible corpora differ only in numeric geometry.
    rng_cat = random.Random("%d|%d|cat" % (seed, index))
    rng_num = random.Random("%d|%d|%s|num" % (seed, index, family))

    window = cfg["window"]
    window = (rng_num.uniform(*window) if isinstance(window, tuple)
              else float(window))
    skew = (rng_num.choice(cfg["skew_choices"]) if "skew_choices" in cfg
            else float(cfg["skew"]))

    slots = parse_offer(offer)
    # Padding is drawn from the per-case stream, so the four feasible families
    # receive the same padded issues and differ only in numeric geometry.
    n_real = len(slots)
    if min_ordered or min_issues:
        slots = slots + pad_slots(rng_cat, slots, min_ordered, min_issues)
    ordered_positions = [
        i for i, slot in enumerate(slots)
        if classify(slot["operand"], slot["operator"], slot["value"]) != CAT
    ]
    # Infeasibility is always built on one ordered issue: categorical
    # requirements are overlap tests and cannot be made provably unsatisfiable.
    if not feasible:
        if not ordered_positions:
            # Infeasibility is only provable on an ordered issue. An offer whose
            # constraints are all categorical cannot carry a crossed pair of hard
            # bounds, so it contributes no scenario to this family rather than a
            # scenario that is labelled infeasible but is not.
            return None
        # The seeded generator blocks on an ordered issue inside a PERMISSION
        # rule (`make_infeasible ... and rt == "permission"`), so match that and
        # only fall back to another rule type when the offer has no ordered
        # issue under a permission.
        permission_positions = [i for i in ordered_positions
                                if slots[i]["rule_type"] == "permission"]
        blocking_position = rng_num.choice(permission_positions
                                           or ordered_positions)
    else:
        blocking_position = None

    issues: List[Dict[str, Any]] = []
    for position, slot in enumerate(slots):
        kind = classify(slot["operand"], slot["operator"], slot["value"])
        if kind == CAT:
            pool = universe.get(slot["operand"]) or []
            values = (slot["value"] if isinstance(slot["value"], list)
                      else [slot["value"]])
            pool = list(pool) or [str(v) for v in values]
            issues.append(build_categorical_issue(
                rng_cat, slot["operand"], slot["operator"], slot["value"], pool))
        else:
            issues.append(build_numeric_issue(
                rng_num, slot["operand"], kind, slot["operator"], slot["value"],
                cfg, window, skew, feasible=(position != blocking_position)))

    documents = {
        "provider": assemble(offer, slots, issues, "p_clause", "Offer",
                             "urn:case:%s:provider:aspiration" % case),
        "consumer": assemble(offer, slots, issues, "c_clause", "Offer",
                             "urn:case:%s:consumer:aspiration" % case),
        "hard_provider": assemble(offer, slots, issues, "hp_clause", "Offer",
                                  "urn:case:%s:provider:hard" % case),
        "hard_consumer": assemble(offer, slots, issues, "hc_clause", "Offer",
                                  "urn:case:%s:consumer:hard" % case),
        "witness": assemble(offer, slots, issues, "w_clause", "Agreement",
                            "urn:witness"),
    }
    # provider.json reproduces the input offer's clauses exactly (p_clause is
    # the original constraint), plus any padded issues.

    # --- verify the witness against both hard policies ---
    violations_p, violations_c = [], []
    for slot, issue in zip(slots, issues):
        label = "%s:%s:%s" % (slot["rule_type"], slot["action"], slot["operand"])
        point = issue["w_clause"]["rightOperand"]
        for hard_key, bucket in (("hp_clause", violations_p),
                                 ("hc_clause", violations_c)):
            hard = issue[hard_key]
            if not satisfies("eq", point, short(hard["operator"]),
                             hard["rightOperand"], issue["kind"]):
                bucket.append("%s eq %s violates hard %s %s" % (
                    label, point, short(hard["operator"]), hard["rightOperand"]))

    blocking = []
    if blocking_position is not None:
        slot = slots[blocking_position]
        blocking = ["%s:%s:%s" % (slot["rule_type"], slot["action"],
                                  slot["operand"])]

    witness_verified = not violations_p and not violations_c
    operators_present = sorted({
        short(c["operator"])
        for issue in issues
        for c in (issue["p_clause"], issue["c_clause"], issue["hp_clause"],
                  issue["hc_clause"], issue["w_clause"])
    })

    meta = {
        "case": case,
        "operators_present": operators_present,
        "operators_missing": sorted(ALL_OPERATORS - set(operators_present)),
        "rule_types_with_constraints": sorted(
            {slot["rule_type"] for slot in slots}),
        "family": family,
        "feasible": bool(feasible),
        "witness_verified": bool(witness_verified),
        "analytic_feasible": bool(feasible),
        "blocking_operands": blocking,
        "blocking_intent": blocking,
        "n_issues_spec": len(slots),
        "actions": sorted({"%s:%s" % (slot["rule_type"], slot["action"])
                           for slot in slots}),
        "window": round(window, 4),
        "opening": cfg["opening"],
        "skew": skew,
        "ood": True,
        "strict_or_exclusion_hard_ops": True,
        "witness_violations_provider": violations_p,
        "witness_violations_consumer": violations_c,
        # provenance back to the ODRL corpus this scenario was derived from
        "source": {
            "offer_id": offer.get("@id"),
            "assigner": offer.get("assigner"),
            "target": offer.get("target"),
            "n_ordered_issues": len(ordered_positions),
            "n_categorical_issues": len(slots) - len(ordered_positions),
            "rules_dropped_without_constraints": dropped_rules,
            "n_issues_from_offer": n_real,
            "n_issues_padded": len(slots) - n_real,
            "n_degenerate_issues": sum(1 for issue in issues
                                       if issue.get("degenerate")),
        },
    }

    return dict(case=case, family=family, feasible=feasible,
                documents=documents, meta=meta)


def write_scenario(out_root: str, scenario: Dict[str, Any]) -> str:
    suffix = "feasible" if scenario["feasible"] else "infeasible"
    directory = os.path.join(out_root, scenario["family"],
                             "%s_%s" % (scenario["case"], suffix))
    os.makedirs(directory, exist_ok=True)
    for name, document in scenario["documents"].items():
        with open(os.path.join(directory, "%s.json" % name), "w",
                  encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(scenario["meta"], handle, indent=2, ensure_ascii=False)
    return directory


# ======================================================================
# 7. CLI
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the five BAMBON ZOPA corpora from real ODRL Offers.")
    parser.add_argument("-i", "--input", required=True,
                        help="offers_negotiation.json (a list of compact ODRL Offers)")
    parser.add_argument("-o", "--out", default=os.path.join("data", "generated_agentodrl"),
                        help="output root (default: data/generated_agentodrl)")
    parser.add_argument("--families", default="all",
                        help="comma-separated subset of %s" % ",".join(FAMILIES))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N offers")
    parser.add_argument("--min-ordered", type=int, default=4,
                        help="pad each scenario up to this many ordered "
                             "(numeric/date/duration) issues; 0 disables padding")
    parser.add_argument("--min-issues", type=int, default=8,
                        help="pad each scenario up to this many issues in total")
    parser.add_argument("--no-padding", action="store_true",
                        help="use only what the offers state; scenarios with no "
                             "ordered issue are then dropped")
    args = parser.parse_args()

    min_ordered = 0 if args.no_padding else max(0, args.min_ordered)
    min_issues = 0 if args.no_padding else max(0, args.min_issues)

    families = (list(FAMILIES) if args.families == "all"
                else [f.strip() for f in args.families.split(",") if f.strip()])
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        parser.error("unknown families: %s" % ", ".join(unknown))

    with open(args.input, "r", encoding="utf-8") as handle:
        offers = json.load(handle)
    if isinstance(offers, dict):
        offers = list(offers.values())
    if args.limit:
        offers = offers[:args.limit]
    print("Read %d offers from %s" % (len(offers), args.input))

    universe = label_universe(offers)
    summary: Dict[str, Dict[str, Any]] = {}

    for family in families:
        written = 0
        verified = 0
        skipped = 0
        not_blockable = 0
        windows, gaps = [], []
        for index, offer in enumerate(offers):
            if not parse_offer(offer) and not min_issues:
                skipped += 1
                continue
            scenario = build_scenario(offer, index, family, universe, args.seed,
                                      min_ordered, min_issues)
            if scenario is None:
                not_blockable += 1
                continue
            write_scenario(args.out, scenario)
            written += 1
            if scenario["meta"]["witness_verified"] == scenario["feasible"]:
                verified += 1
            windows.append(scenario["meta"]["window"])
        summary[family] = {
            "scenarios": written,
            "skipped_no_constraints": skipped,
            "skipped_no_ordered_issue": not_blockable,
            "witness_as_expected": verified,
            "median_window": (round(sorted(windows)[len(windows) // 2], 4)
                              if windows else None),
        }
        print("  %-16s %3d scenarios, %3d/%3d witness as expected"
              % (family, written, verified, written))

    report_path = os.path.join(args.out, "generation_report.json")
    os.makedirs(args.out, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"input": args.input, "seed": args.seed,
                   "offers": len(offers), "families": summary},
                  handle, indent=2, ensure_ascii=False)

    print("\nCorpora written under %s" % args.out)
    print("Report: %s" % report_path)


if __name__ == "__main__":
    main()
