"""Parameterised generator for ODRL negotiation scenarios, with a feasibility oracle.

Motivation
----------
The BAMBON evaluation used 50 hand-authored scenarios. Hand authoring buys
realism but limits external validity in three specific ways, all of which a
generator can address:

* **Coverage.** The 50 use 14 left operands, 22 actions, and in their *hard*
  policies only three operators (``lteq``, ``gteq``, ``isAnyOf``). They contain
  no strict inequality, no ``neq``/``isNoneOf``/``isAllOf`` hard rule, no float
  or sub-unit magnitudes, and at most one action per rule type.
* **Geometry.** Every scenario is authored so that a lawful contract exists
  (``a_p <= h_c <= h_p <= a_c``). Nothing tests what an agent does when *no*
  lawful contract exists, which is exactly the case a hard-policy-aware agent is
  supposed to handle by declining.
* **Difficulty.** The width of the legal window is roughly constant, so the
  benchmark cannot separate agents by how narrow a window they can still close.

This module generates scenarios along all three axes and labels each one with a
*verified* feasibility flag.

The feasibility oracle
----------------------
Feasibility is not asserted, it is checked with the same predicate used to score
compliance in the results tables:

* a **feasible** scenario is emitted together with an explicit witness contract,
  and the generator refuses to write the scenario unless
  ``count_hard_violations(witness, H_p) == 0`` *and*
  ``count_hard_violations(witness, H_c) == 0``;
* an **infeasible** scenario carries at least one operand on which the two hard
  policies are provably jointly unsatisfiable -- a crossed scalar interval,
  recorded in ``meta.json`` as ``blocking_operands`` -- and the generator also
  verifies that the best joint witness *does* violate one of the two policies.

Note on ODRL semantics, and why infeasibility is always built on a *scalar*
operand: under the project's satisfaction predicate a categorical requirement is
an **overlap** test, not a containment test. A committed ``isAnyOf S`` satisfies
a requirement ``isAnyOf A`` whenever ``S & A`` is non-empty, so ``S = {a, c}``
satisfies two disjoint allow-lists at once; a committed ``isAllOf`` superset even
satisfies a disjoint ``eq`` rule, because ``sat(eq, v, isAllOf, S)`` is
``{v} subset of S``. Two categorical hard policies can therefore never be made
provably unsatisfiable. Scalars are tested as committed *points*, so a crossed
interval admits nothing -- that is the airtight construction, and it is the one
used here. (The permissiveness of the categorical case is worth stating in the
paper: it is why the commit-time veto's effect on categorical operands is
invisible in the aggregate compliance figure, which is measured with the same
predicate.)

Usage
-----
    # 200 in-distribution scenarios (same geometry as the hand-authored 50)
    python data/generator/generate_scenarios.py --family id --n 200

    # 100 out-of-distribution scenarios: unseen operands/operators, more actions
    python data/generator/generate_scenarios.py --family ood --n 100

    # 100 scenarios with no lawful contract
    python data/generator/generate_scenarios.py --family infeasible --n 100

    # a mixed corpus: 50% ID, 25% OOD, 25% infeasible
    python data/generator/generate_scenarios.py --family mixed --n 400

Output layout matches ``data/50scenarios`` so any harness can consume it::

    data/generated/<family>/<0007_ood_feasible>/
        provider.json  consumer.json  hard_provider.json  hard_consumer.json
        witness.json   meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.bambon.baselines.veto import feasibility_report          # noqa: E402
from src.bambon.utils.helpers import (                            # noqa: E402
    days_to_iso, seconds_to_duration, strip_key,
)
# The feasibility oracle lives in the verifier, so the generator and the
# stand-alone corpus check share one implementation rather than two that can
# drift apart.
from verify_scenarios import (                                    # noqa: E402
    blocking_operands, build_witness, label_universe,
)

RULE_TYPES = ("permission", "obligation", "prohibition")


# ======================================================================
# 1. operand library
# ======================================================================
# direction = the sign of the provider's preference:
#   'low'  the provider wants a SMALL value and caps it   (provider: lteq)
#   'high' the provider wants a LARGE value and floors it (provider: gteq)
# The consumer always takes the mirrored role, so there is real conflict.
#
# ``seen`` marks the operands that appear in the 50 hand-authored scenarios; the
# out-of-distribution family draws from ``seen=False`` instead.

# Days from BASE_DATE (0001-01-01) to 2026-01-01, so generated dates are plausible.
_DAY0 = 739616

NUM = "num"
DATE = "date"
DUR = "duration"
CAT = "cat"

OPERANDS: Dict[str, Dict[str, Any]] = {
    # ---- numeric, seen ----
    "count":                dict(kind=NUM, direction="low",  lo=100, hi=2_000_000, seen=True),
    "percentage":           dict(kind=NUM, direction="low",  lo=1, hi=100, seen=True),
    "payAmount":            dict(kind=NUM, direction="high", lo=50, hi=50_000, seen=True),
    "frequency":            dict(kind=NUM, direction="high", lo=1, hi=365, seen=True),
    "spatialResolution":    dict(kind=NUM, direction="high", lo=1, hi=5000, seen=True),
    # ---- temporal, seen ----
    # Internal space for a date is days since BASE_DATE (0001-01-01), so these
    # bounds are absolute day numbers: DAY0 puts generated dates in 2026 rather
    # than in year 5, which is what a naive 30..1500 range produced.
    "absoluteTemporalPosition": dict(kind=DATE, direction="low",
                                     lo=_DAY0 + 30, hi=_DAY0 + 1500, seen=True),
    "delayPeriod":          dict(kind=DUR, direction="high", lo=3600, hi=90 * 86400, seen=True),
    "elapsedTime":          dict(kind=DUR, direction="low",  lo=3600, hi=365 * 86400, seen=True),
    # ---- categorical, seen ----
    "purpose":              dict(kind=CAT, direction="low", seen=True,
                                 labels=["research", "education", "commercial", "marketing",
                                         "journalism", "statistics", "audit"]),
    "recipient":            dict(kind=CAT, direction="low", seen=True,
                                 labels=["internal-only", "consortium", "affiliate",
                                         "third-party", "public", "regulator"]),
    "fileFormat":           dict(kind=CAT, direction="low", seen=True,
                                 labels=["CSV", "PARQUET", "JSON", "XML", "RDF", "XLSX"]),
    "industry":             dict(kind=CAT, direction="low", seen=True,
                                 labels=["health", "finance", "energy", "mobility",
                                         "public", "retail", "defence"]),
    "language":             dict(kind=CAT, direction="low", seen=True,
                                 labels=["en", "el", "de", "fr", "es"]),
    "spatial":              dict(kind=CAT, direction="low", seen=True,
                                 labels=["EU", "EEA", "CH", "UK", "US", "global"]),

    # ---- OUT OF DISTRIBUTION: operands absent from the hand-authored 50 ----
    "meteredTime":          dict(kind=DUR, direction="low",  lo=60, hi=30 * 86400, seen=False),
    "timeInterval":         dict(kind=DUR, direction="high", lo=3600, hi=180 * 86400, seen=False),
    "relativeTemporalPosition": dict(kind=DATE, direction="high",
                                     lo=_DAY0 + 10, hi=_DAY0 + 900, seen=False),
    "unitOfCount":          dict(kind=NUM, direction="low", lo=1, hi=10_000, seen=False),
    "resolution":           dict(kind=NUM, direction="high", lo=1, hi=600, seen=False),
    # deliberately sub-unit and float-valued: the 50 are integer-only
    "confidenceLevel":      dict(kind=NUM, direction="high", lo=0.5, hi=0.999,
                                 float_=True, seen=False),
    "samplingRate":         dict(kind=NUM, direction="low", lo=0.001, hi=1.0,
                                 float_=True, seen=False),
    "virtualLocation":      dict(kind=CAT, direction="low", seen=False,
                                 labels=["on-prem", "eu-cloud", "us-cloud", "hybrid",
                                         "air-gapped", "edge"]),
    "systemDevice":         dict(kind=CAT, direction="low", seen=False,
                                 labels=["hsm", "tee", "vm", "container", "bare-metal"]),
    "deliveryChannel":      dict(kind=CAT, direction="low", seen=False,
                                 labels=["sftp", "https", "kafka", "s3", "courier"]),
    "media":                dict(kind=CAT, direction="low", seen=False,
                                 labels=["print", "web", "broadcast", "internal-wiki"]),
    "event":                dict(kind=CAT, direction="low", seen=False,
                                 labels=["breach", "audit", "renewal", "termination"]),
    "version":              dict(kind=NUM, direction="high", lo=1, hi=20, seen=False),
}

ACTIONS: Dict[str, List[str]] = {
    "permission": ["use", "read", "distribute", "aggregate", "derive", "index",
                   "extract", "reproduce", "stream", "query"],
    "obligation": ["anonymize", "delete", "attribute", "notify", "encrypt",
                   "watermark", "audit", "report", "compensate", "review"],
    "prohibition": ["reidentify", "sell", "share", "profile", "train",
                    "sublicense", "export", "combine", "advertise"],
}

# Operators used in the *hard* policies of the 50 hand-authored scenarios.
ID_HARD_NUM_OPS = {"low": "lteq", "high": "gteq"}
ID_HARD_CAT_OP = "isAnyOf"
# Strict / exclusion operators that never appear there.
OOD_HARD_NUM_OPS = {"low": ["lteq", "lt"], "high": ["gteq", "gt"]}
OOD_HARD_CAT_OPS = ["isAnyOf", "eq", "isAllOf", "isNoneOf"]


# ======================================================================
# 2. value rendering
# ======================================================================
def _render(spec: Dict[str, Any], x: float) -> Any:
    kind = spec["kind"]
    if kind == DATE:
        return days_to_iso(int(round(x)))
    if kind == DUR:
        return seconds_to_duration(max(0.0, float(x)))
    if spec.get("float_"):
        return round(float(x), 4)
    return int(round(x))


def _clause(operand: str, operator: str, value: Any) -> Dict[str, Any]:
    return {"leftOperand": f"odrl:{operand}",
            "operator": f"odrl:{operator}",
            "rightOperand": value}


def _document(kind: str, case: str, who: str, rules: Dict[str, List[Dict]]) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "@type": "Offer",
        "@id": f"urn:case:{case}:{who}:{kind}",
        "assigner": "did:provider:P",
        "assignee": ["did:consumer:C"],
        "target": [f"urn:asset:dataset:{case}"],
    }
    for rt in RULE_TYPES:
        if rules.get(rt):
            doc[rt] = rules[rt]
    return doc


# ======================================================================
# 3. one issue: four values plus a witness
# ======================================================================
def _numeric_layout(rng: random.Random, spec: Dict[str, Any], width: float,
                    feasible: bool,
                    opening: float = 1.0,
                    skew: float = 0.5) -> Tuple[float, float, float, float, float]:
    """Return ``(a_p, h_c, h_p, a_c, witness)`` in the operand's own units.

    For a ``direction == 'low'`` operand the provider caps the value and prefers
    it small, so the authored ordering is ``a_p <= h_c <= h_p <= a_c``: each
    party's ideal lies outside the other's legal range (real conflict), while the
    legal ranges overlap on ``[h_c, h_p]`` (a lawful contract exists).

    ``width`` in (0, 1] scales that legal window; small values make the scenario
    harder without changing its feasibility. When ``feasible`` is False the two
    hard bounds are crossed (``h_p < h_c``), so no lawful contract exists.

    ``opening`` in (0, 1] scales how far apart the two parties *start*, about the
    centre of the operand's range. At 1.0 each aspiration sits roughly 10% of the
    span in from its own end, so the openings are ~0.80 span apart (the original
    layout). At 0.25 they start ~0.20 span apart. Note that the legal window is
    defined as a fraction of the opening gap, so ``opening`` shrinks the window
    in absolute terms too while leaving ``ZOPA / gap`` unchanged: the two knobs
    are independent, one setting the scale of the problem and the other the
    fraction of it that is lawful.

    ``skew`` in (0, 1) places the *centre* of the legal window along the segment
    from ``a_p`` to ``a_c``. At the default 0.5 the window sits at the midpoint
    of the two opening positions, which is where two agents conceding
    monotonically from opposite ends meet, so neither party has to travel
    further than the other. Below 0.5 the window sits nearer ``a_p`` and the
    party opening there concedes less than its counterparty; above 0.5 the
    advantage is reversed. The caller is responsible for inverting ``skew`` on
    operands whose preference direction is mirrored, so that the same *party*
    is advantaged on every issue of a scenario rather than a random half.
    """
    lo, hi = float(spec["lo"]), float(spec["hi"])
    span = hi - lo
    a_p = lo + 0.10 * span * rng.uniform(0.5, 1.5)
    a_c = hi - 0.10 * span * rng.uniform(0.5, 1.5)
    if opening != 1.0:
        centre = lo + 0.5 * span
        a_p = centre + opening * (a_p - centre)
        a_c = centre + opening * (a_c - centre)
    # the 0.5 branch is kept verbatim so that every corpus generated before the
    # skew parameter existed reproduces bit-for-bit
    mid = 0.5 * (a_p + a_c) if skew == 0.5 else a_p + skew * (a_c - a_p)
    half = 0.5 * width * (a_c - a_p) * 0.6
    h_c, h_p = mid - half, mid + half
    if not feasible:
        # cross the hard bounds: the provider's ceiling sits *below* the
        # consumer's floor, so [h_c, h_p] is empty
        gap = max(1.0, 0.12 * span)
        h_c, h_p = mid + gap / 2, mid - gap / 2
    witness = 0.5 * (h_c + h_p)
    return a_p, h_c, h_p, a_c, witness


def _make_numeric_issue(rng: random.Random, operand: str, spec: Dict[str, Any],
                        cfg: Dict[str, Any], feasible: bool) -> Dict[str, Any]:
    low = spec["direction"] == "low"
    # `_numeric_layout` works in the frame where a_p is the small end. For a
    # "high" operand the caller mirrors that frame below, which would flip which
    # party the skew favours, so invert it here and keep one party advantaged
    # across every issue of the scenario.
    skew = float(cfg.get("skew", 0.5))
    a_p, h_c, h_p, a_c, wit = _numeric_layout(rng, spec, cfg["window"], feasible,
                                              cfg.get("opening", 1.0),
                                              skew if low else 1.0 - skew)
    if cfg.get("strict_num_ops", cfg["strict_ops"]):
        p_hard_op = rng.choice(OOD_HARD_NUM_OPS["low" if low else "high"])
        c_hard_op = rng.choice(OOD_HARD_NUM_OPS["high" if low else "low"])
    else:
        p_hard_op = ID_HARD_NUM_OPS["low" if low else "high"]
        c_hard_op = ID_HARD_NUM_OPS["high" if low else "low"]
    p_op = "lteq" if low else "gteq"
    c_op = "gteq" if low else "lteq"

    if not low:
        # mirrored ordering: the provider wants the LARGE value
        a_p, a_c = a_c, a_p
        h_p, h_c = h_c, h_p
    # keep the witness strictly inside a strict bound
    if feasible and (p_hard_op in ("lt", "gt") or c_hard_op in ("lt", "gt")):
        wit = 0.5 * (h_c + h_p)

    return dict(
        operand=operand, spec=spec, kind="numeric",
        p_clause=_clause(operand, p_op, _render(spec, a_p)),
        c_clause=_clause(operand, c_op, _render(spec, a_c)),
        hp_clause=_clause(operand, p_hard_op, _render(spec, h_p)),
        hc_clause=_clause(operand, c_hard_op, _render(spec, h_c)),
        w_clause=_clause(operand, "eq", _render(spec, wit)),
    )


def _make_categorical_issue(rng: random.Random, operand: str, spec: Dict[str, Any],
                            cfg: Dict[str, Any], feasible: bool) -> Dict[str, Any]:
    labels = list(spec["labels"])
    rng.shuffle(labels)
    # a shared core the two allow-lists overlap on, plus private tails
    core = labels[:1]
    p_only = labels[1:3]
    c_only = labels[3:5] or labels[1:2]

    p_allow = core + p_only
    c_allow = core + c_only

    if cfg.get("strict_cat_ops", cfg["strict_ops"]):
        p_hard_op = rng.choice(OOD_HARD_CAT_OPS)
        c_hard_op = rng.choice(OOD_HARD_CAT_OPS)
    else:
        p_hard_op = c_hard_op = ID_HARD_CAT_OP

    # NOTE: `feasible=False` is never requested for a categorical operand -- see
    # the comment in `generate_scenario`. Categorical requirements are overlap
    # tests under the project's satisfaction predicate, so they cannot be made
    # provably unsatisfiable; infeasibility is always built on a scalar operand.
    p_hard_val = _cat_hard_value(p_hard_op, p_allow, core, labels)
    c_hard_val = _cat_hard_value(c_hard_op, c_allow, core, labels)
    witness = _cat_witness(p_hard_op, p_hard_val, c_hard_op, c_hard_val, core, labels)

    w_op = "eq" if not isinstance(witness, list) else "isAnyOf"
    return dict(
        operand=operand, spec=spec, kind="cat",
        p_clause=_clause(operand, "isAnyOf", p_allow),
        c_clause=_clause(operand, "isAnyOf", c_allow),
        hp_clause=_clause(operand, p_hard_op, p_hard_val),
        hc_clause=_clause(operand, c_hard_op, c_hard_val),
        w_clause=_clause(operand, w_op, witness),
    )


def _cat_hard_value(op: str, allow: List[str], core: List[str], labels: List[str]) -> Any:
    if op == "eq":
        return core[0]
    if op == "isAllOf":
        return core[:1]
    if op == "isNoneOf":
        # forbid something outside the shared core so a lawful term still exists
        banned = [l for l in labels if l not in core][-1:]
        return banned or [labels[-1]]
    return allow                                   # isAnyOf


def _cat_witness(p_op: str, p_val: Any, c_op: str, c_val: Any,
                 core: List[str], labels: List[str]) -> Any:
    """A term satisfying both categorical hard rules, when one exists."""
    def as_set(v: Any) -> set:
        return set(v) if isinstance(v, (list, tuple, set)) else {v}

    banned: set = set()
    required: set = set()
    allowed: Optional[set] = None
    for op, val in ((p_op, p_val), (c_op, c_val)):
        V = as_set(val)
        if op == "eq":
            required |= V
            allowed = V if allowed is None else (allowed & V)
        elif op == "isAllOf":
            required |= V
        elif op == "isAnyOf":
            allowed = V if allowed is None else (allowed & V)
        elif op == "isNoneOf":
            banned |= V

    cand = (required - banned)
    if not cand:
        pool = (allowed if allowed is not None else set(labels)) - banned
        cand = {sorted(pool)[0]} if pool else {core[0]}
    return sorted(cand) if len(cand) > 1 else sorted(cand)[0]


def _make_range_issue(rng: random.Random, operand: str, spec: Dict[str, Any],
                      cfg: Dict[str, Any], feasible: bool) -> Dict[str, Any]:
    """A two-sided numeric interval on one operand (kind RANGE in the paper).

    Both parties state a lower *and* an upper bound on the same operand, which
    the agents merge into a single interval issue. The 50 hand-authored
    scenarios contain few of these, so it is an OOD structure as well as an OOD
    operand.
    """
    lo, hi = float(spec["lo"]), float(spec["hi"])
    span = hi - lo
    p_lo, p_hi = lo + 0.05 * span, lo + 0.45 * span
    c_lo, c_hi = lo + 0.30 * span, lo + 0.90 * span
    hp_lo, hp_hi = lo + 0.02 * span, lo + 0.60 * span
    hc_lo, hc_hi = lo + 0.20 * span, lo + 0.95 * span
    if not feasible:
        hp_hi = lo + 0.10 * span          # provider ceiling below consumer floor
        hc_lo = lo + 0.50 * span
    wit = 0.5 * (max(hp_lo, hc_lo) + min(hp_hi, hc_hi))
    return dict(
        operand=operand, spec=spec, kind="range",
        p_clause=[_clause(operand, "gteq", _render(spec, p_lo)),
                  _clause(operand, "lteq", _render(spec, p_hi))],
        c_clause=[_clause(operand, "gteq", _render(spec, c_lo)),
                  _clause(operand, "lteq", _render(spec, c_hi))],
        hp_clause=[_clause(operand, "gteq", _render(spec, hp_lo)),
                   _clause(operand, "lteq", _render(spec, hp_hi))],
        hc_clause=[_clause(operand, "gteq", _render(spec, hc_lo)),
                   _clause(operand, "lteq", _render(spec, hc_hi))],
        w_clause=[_clause(operand, "gteq", _render(spec, wit)),
                  _clause(operand, "lteq", _render(spec, wit))],
    )



# ======================================================================
# 3b. guaranteed operator coverage
# ======================================================================
# Every ODRL operator the project's satisfaction predicate understands. A corpus
# that omits one cannot say anything about it -- and the 50 hand-authored
# scenarios omit four of the nine in their hard policies. The coverage pass below
# appends purpose-built issues so that *every* generated scenario exercises all
# nine at least once, and so that each of the three rule types carries at least
# one constraint.
ALL_OPERATORS: Tuple[str, ...] = (
    "lt", "lteq", "gt", "gteq", "eq", "neq", "isAnyOf", "isAllOf", "isNoneOf",
)


def _cov(operand, p, c, hp, hc, w):
    """One coverage issue. Each argument is an ``(operator, value)`` pair, or a
    list of them when a document states several clauses on the operand.

    A ``neq`` rule excludes a point but bounds nothing, so an issue whose hard
    policy contains *only* a ``neq`` has no reservation: the utility-aware bidder
    falls back to a synthetic span, the two parties never converge, and the issue
    never closes. That is a real limitation of the agent, but it is also not what
    a policy author would write -- "at most 50, and never exactly 33" is. The
    ``neq`` coverage issues therefore pair the exclusion with an interval bound,
    which keeps the issue well-posed while still exercising the operator.
    """
    def clauses(spec):
        return ([_clause(operand, *x) for x in spec] if isinstance(spec, list)
                else [_clause(operand, *spec)])
    return dict(
        operand=operand, spec=None, kind="coverage",
        p_clause=clauses(p), c_clause=clauses(c),
        hp_clause=clauses(hp), hc_clause=clauses(hc),
        w_clause=clauses(w),
    )


# Each entry is feasible by construction; the witness column is what a lawful
# contract would commit to. Together they cover all nine operators.
COVERAGE_ISSUES: Dict[str, List[Any]] = {
    "permission": [
        # `eq` hard rule on both sides, same label: witness is that label
        lambda: _cov("coverageMode",
                     ("isAnyOf", ["alpha", "beta"]), ("isAnyOf", ["alpha", "gamma"]),
                     ("eq", "alpha"), ("eq", "alpha"), ("eq", "alpha")),
    ],
    "obligation": [
        # `isAllOf` requirement plus an `isNoneOf` exclusion: witness {t1}
        lambda: _cov("coverageTag",
                     ("isAllOf", ["t1"]), ("isAnyOf", ["t1", "t2"]),
                     ("isAllOf", ["t1"]), ("isNoneOf", ["t3"]), ("isAllOf", ["t1"])),
    ],
    "prohibition": [
        # `neq` on a categorical operand: witness c1
        lambda: _cov("coverageChannel",
                     ("isAnyOf", ["c1", "c3"]), ("isAnyOf", ["c1", "c2"]),
                     [("isAnyOf", ["c1", "c3"]), ("neq", "c2")],
                     ("isAnyOf", ["c1", "c2"]), ("eq", "c1")),
    ],
}


def _retag_numeric_operators(rules, rng) -> None:
    """Guarantee `gt`, `lt` and `neq` appear, on *generated* numeric issues.

    Hand-authoring a numeric issue per operator produced a well-formed but
    badly-conditioned one: the two concession paths ended up oscillating one step
    apart around the meeting point and the issue never closed, which blocked every
    negotiation in the corpus. The three operators the random draw may not have
    produced are therefore attached to issues the normal construction already
    made, so every issue in the corpus comes from the same well-tested builder:

    * ``lteq`` -> ``lt`` on one provider hard clause and ``gteq`` -> ``gt`` on one
      consumer hard clause. Each only tightens a bound by one representable step,
      and the witness builder is strictness-aware, so feasibility survives.
    * ``neq v`` appended to both hard policies on one integer operand, with ``v``
      placed far outside the legal window so it never binds. It still exercises
      the exclusion path in the repair operator and in the commit gate.
    """
    def hard_numeric_clauses(doc_key):
        out = []
        for rt in RULE_TYPES:
            for rule in rules[doc_key][rt]:
                for c in rule.get("constraint", []) or []:
                    if isinstance(c["rightOperand"], (int, float)):
                        out.append(c)
        return out

    for doc_key, want, have in (("hp", "lt", "lteq"), ("hc", "gt", "gteq")):
        done = False
        for c in hard_numeric_clauses(doc_key):
            if strip_key(str(c["operator"])) == have:
                c["operator"] = f"odrl:{want}"
                done = True
                break
        if done:
            continue
        # No clause of the right direction to retag. Append a deliberately
        # NON-BINDING strict clause instead, so the operator is present in the
        # corpus without changing what is feasible.
        for c in hard_numeric_clauses(doc_key):
            v = c["rightOperand"]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            slack = int(abs(v) * 50 + 10_000)
            for rt in RULE_TYPES:
                for rule in rules[doc_key][rt]:
                    if c in (rule.get("constraint", []) or []):
                        rule["constraint"].append({
                            "leftOperand": c["leftOperand"],
                            "operator": f"odrl:{want}",
                            "rightOperand": (v + slack) if want == "lt" else (v - slack),
                        })
                        done = True
                        break
                if done:
                    break
            if done:
                break

    # NOTE: `neq` coverage comes from the categorical `coverageChannel` issue
    # above, not from injecting a numeric `neq`. Appending an extra numeric clause
    # to an already-generated operand perturbed that issue's belief domain and
    # reservation enough to stop it converging, so the injection was removed --
    # the operator is still present in every scenario, on a categorical operand.


COVERAGE_ACTION = {"permission": "inspect", "obligation": "certify",
                   "prohibition": "resell"}


def operators_used(*docs) -> set:
    """Every operator appearing anywhere in the given ODRL documents."""
    out = set()
    for doc in docs:
        if not doc:
            continue
        for rt in RULE_TYPES:
            for rule in doc.get(rt, []) or []:
                for c in rule.get("constraint", []) or []:
                    out.add(str(c["operator"]).replace("odrl:", ""))
    return out


# ======================================================================
# 4. scenario assembly
# ======================================================================
FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # in-distribution: matches the hand-authored 50
    "id": dict(seen_only=True, strict_ops=False, ranges=False,
               actions=(1, 1), constraints=(2, 5), window=1.0, feasible=True,
               coverage=False),
    # out-of-distribution: unseen operands and operators, more actions, ranges,
    # narrower legal windows
    "ood": dict(seen_only=False, unseen_bias=0.75, strict_ops=True, ranges=True,
                actions=(1, 3), constraints=(2, 6), window=(0.15, 1.0), feasible=True,
                coverage=False),
    # Randomly generated, every operator guaranteed present, all three rule
    # types populated. This is the corpus the headline tournament runs on.
    "full": dict(seen_only=False, unseen_bias=0.5, strict_ops=True, ranges=True,
                 actions=(1, 2), constraints=(2, 5), window=(0.25, 1.0),
                 feasible=True, coverage=True),
    "full_infeasible": dict(seen_only=False, unseen_bias=0.5, strict_ops=True,
                            ranges=True, actions=(1, 2), constraints=(2, 5),
                            window=1.0, feasible=False, coverage=True),
    # Same as `full` but with a NARROW legal window, and the two parties starting
    # CLOSE together. Isolates whether a small feasible region is what separates
    # a belief-driven bidder from a time-scheduled one.
    "close_narrow": dict(seen_only=False, unseen_bias=0.5, strict_ops=True,
                         ranges=True, actions=(1, 2), constraints=(2, 5),
                         window=(0.10, 0.25), opening=0.25,
                         feasible=True, coverage=True),
    # `full` under the name the paper uses. Parties start FAR apart and the
    # lawful window is WIDE; the parameters are identical to `full`, so the two
    # names generate the same corpus from the same seed.
    "far_wide": dict(seen_only=False, unseen_bias=0.5, strict_ops=True, ranges=True,
                     actions=(1, 2), constraints=(2, 5), window=(0.25, 1.0),
                     opening=1.0, feasible=True, coverage=True),
    # far_narrow with the lawful window moved OFF the midpoint of the two
    # openings, so one party must concede roughly four times as far as the
    # other. Everything else matches far_narrow, which makes the pair a
    # controlled test of window centring, the one geometric axis the other
    # corpora hold fixed.
    "far_narrow_asym": dict(seen_only=False, unseen_bias=0.5, strict_ops=True,
                            ranges=True, actions=(1, 2), constraints=(2, 5),
                            window=(0.10, 0.25), opening=1.0,
                            skew_choices=(0.20, 0.80),
                            feasible=True, coverage=True),
    # Narrow legal window, parties starting FAR apart: the same target size at
    # the original scale, so the pair isolates opening distance from window width.
    "far_narrow": dict(seen_only=False, unseen_bias=0.5, strict_ops=True,
                       ranges=True, actions=(1, 2), constraints=(2, 5),
                       window=(0.10, 0.25), opening=1.0,
                       feasible=True, coverage=True),
    # no lawful contract exists
    "infeasible": dict(seen_only=False, unseen_bias=0.35, strict_ops=False, ranges=True,
                       actions=(1, 2), constraints=(2, 5), window=1.0, feasible=False,
                       coverage=False),
}


def _resolve_cfg(family: str, rng: random.Random, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(FAMILY_DEFAULTS[family])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    # `strict_ops` is a convenience switch over the two finer knobs; either can be
    # set on its own to separate strict *numeric* bounds from *categorical*
    # exclusion/containment rules in the hard policies.
    cfg.setdefault("strict_num_ops", cfg["strict_ops"])
    cfg.setdefault("strict_cat_ops", cfg["strict_ops"])
    cfg.setdefault("coverage", False)
    w = cfg["window"]
    cfg["window"] = rng.uniform(*w) if isinstance(w, tuple) else float(w)
    o = cfg.setdefault("opening", 1.0)
    cfg["opening"] = rng.uniform(*o) if isinstance(o, tuple) else float(o)
    # Which party the lawful window sits near is drawn per scenario, not per
    # issue, and the two directions are equiprobable, so an asymmetric corpus
    # disadvantages provider and consumer equally often in aggregate.
    choices = cfg.get("skew_choices")
    if cfg.get("skew") is None:
        cfg["skew"] = rng.choice(list(choices)) if choices else 0.5
    return cfg


def _pick_operands(rng: random.Random, cfg: Dict[str, Any], n: int) -> List[str]:
    seen = [k for k, v in OPERANDS.items() if v["seen"]]
    unseen = [k for k, v in OPERANDS.items() if not v["seen"]]
    if cfg["seen_only"]:
        pool = seen
        return rng.sample(pool, min(n, len(pool)))
    bias = cfg.get("unseen_bias", 0.5)
    out: List[str] = []
    while len(out) < n:
        pool = unseen if rng.random() < bias else seen
        cand = rng.choice(pool)
        if cand not in out:
            out.append(cand)
        if len(out) >= len(seen) + len(unseen):
            break
    return out


def generate_scenario(rng: random.Random, family: str, index: int,
                      **overrides) -> Dict[str, Any]:
    """Build one scenario. Returns the four documents, a witness and metadata."""
    cfg = _resolve_cfg(family, rng, overrides)
    feasible = bool(cfg["feasible"])
    case = f"{index:04d}_{family}"

    rules = {k: {rt: [] for rt in RULE_TYPES} for k in ("p", "c", "hp", "hc", "w")}
    blocking: List[str] = []
    n_issues = 0
    used_actions: List[str] = []

    # every scenario carries at least one permission, one obligation and one
    # prohibition, as the hand-authored ones do
    for rt in RULE_TYPES:
        n_actions = rng.randint(*cfg["actions"])
        for action in rng.sample(ACTIONS[rt], n_actions):
            used_actions.append(f"{rt}:{action}")
            n_c = rng.randint(*cfg["constraints"])
            operands = _pick_operands(rng, cfg, n_c)
            if (not feasible) and rt == "permission" and not any(
                    OPERANDS[o]["kind"] in (NUM, DATE, DUR) for o in operands):
                # Infeasibility can only be built on a scalar operand, so make
                # sure the rule that carries it has one.
                scalars = [k for k, v in OPERANDS.items()
                           if v["kind"] in (NUM, DATE, DUR)
                           and (v["seen"] or not cfg["seen_only"])]
                operands[0] = rng.choice(scalars)
            # In an infeasible scenario at least one operand must actually block.
            # It has to be a *scalar* one: under the project's ODRL satisfaction
            # predicate a categorical requirement is an overlap test, so two
            # categorical hard policies are never provably unsatisfiable (a
            # committed `isAnyOf {a, c}` satisfies both `isAnyOf {a}` and
            # `isAnyOf {c}`, and a committed `isAllOf` superset even satisfies a
            # disjoint `eq` rule). Scalars are tested as committed points, so a
            # crossed interval is airtight.
            scalar_idx = [i for i, o in enumerate(operands)
                          if OPERANDS[o]["kind"] in (NUM, DATE, DUR)]
            block_at = (rng.choice(scalar_idx) if scalar_idx else None)
            per_rule = {k: [] for k in ("p", "c", "hp", "hc", "w")}
            for i, operand in enumerate(operands):
                spec = OPERANDS[operand]
                make_infeasible = (not feasible) and (i == block_at) and (rt == "permission")
                if cfg["ranges"] and spec["kind"] in (NUM, DUR, DATE) and rng.random() < 0.25:
                    issue = _make_range_issue(rng, operand, spec, cfg, not make_infeasible)
                elif spec["kind"] == CAT:
                    issue = _make_categorical_issue(rng, operand, spec, cfg, not make_infeasible)
                else:
                    issue = _make_numeric_issue(rng, operand, spec, cfg, not make_infeasible)
                if make_infeasible:
                    blocking.append(f"{rt}:{action}:{operand}")
                for key, field in (("p", "p_clause"), ("c", "c_clause"),
                                   ("hp", "hp_clause"), ("hc", "hc_clause"),
                                   ("w", "w_clause")):
                    cl = issue[field]
                    per_rule[key].extend(cl if isinstance(cl, list) else [cl])
                n_issues += 1
            for key in per_rule:
                rules[key][rt].append({"action": action, "constraint": per_rule[key]})

    # ---- operator-coverage pass -------------------------------------------
    if cfg["coverage"]:
        for rt in RULE_TYPES:
            per_rule = {k: [] for k in ("p", "c", "hp", "hc", "w")}
            for make in COVERAGE_ISSUES[rt]:
                issue = make()
                for key, field in (("p", "p_clause"), ("c", "c_clause"),
                                   ("hp", "hp_clause"), ("hc", "hc_clause"),
                                   ("w", "w_clause")):
                    per_rule[key].extend(issue[field])
                n_issues += 1
            action = COVERAGE_ACTION[rt]
            used_actions.append(f"{rt}:{action}")
            for key in per_rule:
                rules[key][rt].append({"action": action, "constraint": per_rule[key]})
        _retag_numeric_operators(rules, rng)

    docs = {
        "provider": _document("aspiration", case, "provider", rules["p"]),
        "consumer": _document("aspiration", case, "consumer", rules["c"]),
        "hard_provider": _document("hard", case, "provider", rules["hp"]),
        "hard_consumer": _document("hard", case, "consumer", rules["hc"]),
    }

    # ---- the oracle, run twice and required to agree ----------------------
    # (1) analytic: intersect the two hard policies operand by operand.
    analytic_blocked = blocking_operands(docs["hard_provider"], docs["hard_consumer"])
    # (2) constructive: build the joint witness and score it with the very
    #     predicate the results tables use to count violations.
    witness = build_witness(docs["hard_provider"], docs["hard_consumer"],
                            label_universe(docs))
    v_p = feasibility_report(witness, docs["hard_provider"])
    v_c = feasibility_report(witness, docs["hard_consumer"])
    witness_ok = not v_p and not v_c
    docs["witness"] = witness

    ops = operators_used(docs["provider"], docs["consumer"],
                         docs["hard_provider"], docs["hard_consumer"])
    meta = {
        "case": case,
        "operators_present": sorted(ops),
        "operators_missing": sorted(set(ALL_OPERATORS) - ops),
        "rule_types_with_constraints": sorted(
            rt for rt in RULE_TYPES
            if any(r.get("constraint") for r in docs["provider"].get(rt, []) or [])),
        "family": family,
        "feasible": feasible,
        # the scenario is only written when all three of these agree with `feasible`
        "witness_verified": witness_ok,
        "analytic_feasible": not analytic_blocked,
        "blocking_operands": analytic_blocked,
        "blocking_intent": blocking,
        "n_issues_spec": n_issues,
        "actions": used_actions,
        "window": round(cfg["window"], 4),
        "opening": round(cfg.get("opening", 1.0), 4),
        "skew": round(float(cfg.get("skew", 0.5)), 4),
        "ood": not cfg["seen_only"],
        "strict_or_exclusion_hard_ops": bool(cfg["strict_ops"]),
        "witness_violations_provider": v_p[:4],
        "witness_violations_consumer": v_c[:4],
    }
    return {"docs": docs, "meta": meta}


# ======================================================================
# 5. CLI
# ======================================================================
def write_scenario(out_root: str, scenario: Dict[str, Any]) -> str:
    meta = scenario["meta"]
    tag = "feasible" if meta["feasible"] else "infeasible"
    directory = os.path.join(out_root, f"{meta['case']}_{tag}")
    os.makedirs(directory, exist_ok=True)
    for name, doc in scenario["docs"].items():
        with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return directory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="mixed",
                    choices=("id", "ood", "infeasible", "full", "full_infeasible",
                             "far_wide", "close_narrow", "far_narrow",
                             "far_narrow_asym", "mixed"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", default=None,
                    help="output root (default data/generated/<family>)")
    ap.add_argument("--window", type=float, default=None,
                    help="override the legal-window width in (0, 1]")
    ap.add_argument("--opening", type=float, default=None,
                    help="override how far apart the two parties start, in (0, 1]")
    ap.add_argument("--skew", type=float, default=None,
                    help="override where the legal window sits between the two "
                         "openings, in (0, 1); 0.5 is the midpoint")
    args = ap.parse_args()

    out_root = args.out or os.path.join(_ROOT, "data", "generated", args.family)
    os.makedirs(out_root, exist_ok=True)
    rng = random.Random(args.seed)

    if args.family == "mixed":
        plan = (["id"] * (args.n // 2)
                + ["ood"] * (args.n // 4)
                + ["infeasible"] * (args.n - args.n // 2 - args.n // 4))
    else:
        plan = [args.family] * args.n

    ok = bad = 0
    counts: Dict[str, int] = {}
    for i, fam in enumerate(plan):
        sc = generate_scenario(rng, fam, i, window=args.window,
                               opening=args.opening, skew=args.skew)
        meta = sc["meta"]
        expect_ok = meta["feasible"]
        if meta["witness_verified"] == expect_ok and meta["analytic_feasible"] == expect_ok:
            ok += 1
        else:
            bad += 1
            print(f"[reject] {meta['case']}: feasible={expect_ok} but "
                  f"witness_verified={meta['witness_verified']} "
                  f"analytic_feasible={meta['analytic_feasible']} "
                  f"blocked={meta['blocking_operands'][:2]} "
                  f"({meta['witness_violations_provider'][:1]}"
                  f"{meta['witness_violations_consumer'][:1]})")
            continue
        write_scenario(out_root, sc)
        counts[fam] = counts.get(fam, 0) + 1

    print(f"wrote {ok} scenarios to {out_root}  {counts}")
    if bad:
        print(f"rejected {bad} scenarios that failed the feasibility oracle")


if __name__ == "__main__":
    main()
