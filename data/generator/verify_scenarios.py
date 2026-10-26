"""Validate a directory of ODRL negotiation scenarios and decide feasibility.

Runs over any corpus laid out like ``data/50scenarios`` (a directory of
scenario folders, each holding ``provider.json``, ``consumer.json``,
``hard_provider.json``, ``hard_consumer.json``) and reports, per scenario:

* **structure** -- all four documents parse, at least one permission, one
  obligation and one prohibition, every hard-policy operand also appears in the
  corresponding aspiration (otherwise the operand is not in the negotiated issue
  set and the party can never state it, so it is violated by construction);
* **feasibility** -- whether a lawful joint contract exists, decided *twice*, by
  two independent routes that must agree:

  1. a **constructive** route: build the joint witness term for every operand
     and score it with ``feasibility_report``, the same predicate the results
     tables use to count violations;
  2. an **analytic** route: intersect the two hard policies operand by operand.
     For scalars this is interval intersection under committed-point semantics;
     for categorical operands it is the allow/require/ban algebra. An operand
     with an empty intersection is *provably* blocking, so the scenario admits no
     lawful contract at all -- not merely "our witness failed".

Disagreement between the two routes is reported as an error, because it means
either the witness construction or the analytic oracle is wrong.

Usage
-----
    python data/generator/verify_scenarios.py data/50scenarios
    python data/generator/verify_scenarios.py data/generated/mixed --expect-meta
    python data/generator/verify_scenarios.py data/generated/infeasible --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.bambon.baselines.veto import (                      # noqa: E402
    _iter_hard_constraints, _labels, _to_num, feasibility_report,
)
from src.bambon.utils.helpers import strip_key               # noqa: E402

RULE_TYPES = ("permission", "obligation", "prohibition")
ALL_OPERATORS = ("lt", "lteq", "gt", "gteq", "eq", "neq",
                 "isAnyOf", "isAllOf", "isNoneOf")


# ======================================================================
# analytic oracle: intersect the two hard policies, operand by operand
# ======================================================================
def _scalar_interval(constraints: List[Tuple[str, float]]):
    """(lo, hi, strict_lo, strict_hi, eq) for one operand's scalar rules."""
    lo = hi = eq = None
    slo = shi = False
    for op, x in constraints:
        if op in ("gteq", "gt"):
            if lo is None or x > lo:
                lo, slo = x, op == "gt"
        elif op in ("lteq", "lt"):
            if hi is None or x < hi:
                hi, shi = x, op == "lt"
        elif op == "eq":
            eq = x
    return lo, hi, slo, shi, eq


def _scalar_satisfiable(constraints: List[Tuple[str, float]]) -> bool:
    lo, hi, slo, shi, eq = _scalar_interval(constraints)
    if eq is not None:
        if lo is not None and (eq < lo or (slo and eq == lo)):
            return False
        if hi is not None and (eq > hi or (shi and eq == hi)):
            return False
        return True
    if lo is None or hi is None:
        return True
    return (lo < hi) if (slo or shi) else (lo <= hi)


def _categorical_satisfiable(constraints: List[Tuple[str, Any]]) -> bool:
    """Is some committed categorical term compatible with all these rules?

    ``isAnyOf`` rules are *overlap* requirements, not containment requirements: a
    committed ``isAnyOf S`` satisfies ``isAnyOf A`` whenever ``S & A`` is
    non-empty. Two ``isAnyOf`` rules are therefore always jointly satisfiable
    (take one label from each). Only ``eq`` / ``isAllOf`` / ``isNoneOf`` can
    actually conflict.
    """
    required: set = set()
    banned: set = set()
    eq_vals: List[set] = []
    anyof: List[set] = []
    for op, val in constraints:
        V = _labels(val)
        if op == "eq":
            eq_vals.append(V)
            required |= V
        elif op == "isAllOf":
            required |= V
        elif op in ("isNoneOf", "neq"):
            banned |= V
        elif op == "isAnyOf":
            anyof.append(V)
    # two `eq` rules on different labels can never both hold
    if len(eq_vals) > 1:
        common = set.intersection(*eq_vals)
        if not common:
            return False
    if required & banned:
        return False
    # an `eq`/`isAllOf` requirement must sit inside every `isAnyOf` allow-list
    for A in anyof:
        if eq_vals and not (required & A):
            return False
    for A in anyof:
        if not (A - banned):
            return False
    return True


def _categorical_satisfiable_containment(constraints: List[Tuple[str, Any]],
                                          universe: Optional[set] = None) -> bool:
    """Strict variant: is there a term *contained* in every allow-list?

    :func:`_categorical_satisfiable` follows the project's satisfaction
    predicate, under which ``isAnyOf``/``eq`` requirements are **overlap** tests.
    That makes a term feasible if it merely mentions one permitted label, so two
    disjoint allow-lists are jointly satisfiable by their union. No agent that
    projects its own proposals into its own allow-list will ever construct such a
    term, however: Eq. 12 intersects, it does not union. This predicate therefore
    asks the stricter question --- is there a non-empty ``S`` with

        required subset of S subset of allowed,   S disjoint from banned

    --- which is what a hard-policy-respecting agent can actually commit to.

    Exclusion-only operands (every rule an ``isNoneOf``/``neq``) leave ``allowed``
    unbounded, so any label outside the banned set will do; those are satisfiable
    whenever the operand's vocabulary has a free label, which is why ``universe``
    is consulted when it is supplied.
    """
    allowed: Optional[set] = None
    required: set = set()
    banned: set = set()
    for op, val in constraints:
        V = _labels(val)
        if op in ("isAnyOf", "eq"):
            allowed = V if allowed is None else (allowed & V)
            if op == "eq":
                required |= V
        elif op == "isAllOf":
            required |= V
        elif op in ("isNoneOf", "neq"):
            banned |= V

    if allowed is None:
        # No rule bounds the universe: a term may be any set disjoint from
        # `banned` that contains `required`.
        if not required.isdisjoint(banned):
            return False
        if required:
            return True
        free = (universe - banned) if universe is not None else None
        return True if free is None else bool(free)

    S = allowed - banned
    return bool(S) and required <= S


def blocking_operands(hard_p: Dict, hard_c: Dict, containment: bool = False,
                      universe: Optional[Dict[Tuple[str, Any, str], set]] = None) -> List[str]:
    """Operands on which the two hard policies are provably unsatisfiable."""
    joint: Dict[Tuple[str, Any, str], List[Tuple[str, Any]]] = {}
    for doc in (hard_p, hard_c):
        for rt, action, operand, op, val in _iter_hard_constraints(doc):
            joint.setdefault((rt, action, operand), []).append((op, val))

    blocked: List[str] = []
    for (rt, action, operand), constraints in joint.items():
        nums = [(op, _to_num(v)) for op, v in constraints if _to_num(v) is not None]
        if len(nums) == len(constraints):
            ok = _scalar_satisfiable(nums)
        elif containment:
            pool = universe.get((rt, action, operand)) if universe else None
            ok = _categorical_satisfiable_containment(constraints, pool)
        else:
            ok = _categorical_satisfiable(constraints)
        if not ok:
            blocked.append(f"{rt}:{action}:{operand}")
    return sorted(blocked)


# ======================================================================
# constructive route: build the joint witness and score it
# ======================================================================
def _scalar_witness_value(nums: List[Tuple[str, float]], template: Any) -> Any:
    """A committed scalar that satisfies every rule, in the template's lexical form.

    The midpoint of the feasible interval is the natural choice, but it is not
    always enough: right operands are rendered back into the template's type, so
    an integer operand with a one-unit-wide window can round the midpoint onto a
    *strict* bound (window ``(10, 11)`` -> midpoint 10.5 -> rendered 10 ->
    violates ``gt 10``). Candidates are therefore tried in order of preference
    and each is re-checked *after rendering*, so the value returned is feasible in
    the representation the agents actually exchange. When none is feasible the
    interval is genuinely empty at the operand's resolution, and the caller's
    feasibility report says so.
    """
    from src.bambon.baselines.veto import _render_like, _step, _to_num as _num

    lo, hi, slo, shi, eq = _scalar_interval(nums)
    step = _step(template)
    cands: List[float] = []
    if eq is not None:
        cands.append(eq)
    if lo is not None and hi is not None:
        cands += [0.5 * (lo + hi), lo + step, hi - step, lo, hi]
    elif lo is not None:
        cands += [lo + step, lo, lo + 10 * step]
    elif hi is not None:
        cands += [hi - step, hi, hi - 10 * step]
    else:
        cands.append(0.0)

    def ok(x: float) -> bool:
        for op, bound in nums:
            if op in ("gteq",) and not x >= bound: return False
            if op in ("gt",) and not x > bound: return False
            if op in ("lteq",) and not x <= bound: return False
            if op in ("lt",) and not x < bound: return False
            if op == "eq" and x != bound: return False
            if op == "neq" and x == bound: return False
        return True

    for x in cands:
        rendered = _render_like(template, x)
        back = _num(rendered)
        if back is not None and ok(back):
            return rendered
    return _render_like(template, cands[0])



def label_universe(docs: Dict[str, Dict]) -> Dict[Tuple[str, Any, str], set]:
    """Every categorical label either party ever mentions, per operand.

    Needed because a pair of ``isNoneOf`` hard rules bounds what a term may *not*
    contain without bounding what it may; the witness has to draw a permitted
    label from somewhere, and the aspirations are where the vocabulary lives.
    """
    out: Dict[Tuple[str, Any, str], set] = {}
    for doc in docs.values():
        for rt in RULE_TYPES:
            for rule in doc.get(rt, []) or []:
                for c in rule.get("constraint", []) or []:
                    if _to_num(c["rightOperand"]) is not None:
                        continue
                    key = (rt, rule.get("action"), strip_key(str(c["leftOperand"])))
                    out.setdefault(key, set()).update(_labels(c["rightOperand"]))
    return out


def build_witness(hard_p: Dict, hard_c: Dict,
                  universe: Optional[Dict[Tuple[str, Any, str], set]] = None) -> Dict:
    """A contract that satisfies both hard policies wherever that is possible."""
    joint: Dict[Tuple[str, Any, str], List[Tuple[str, Any]]] = {}
    for doc in (hard_p, hard_c):
        for rt, action, operand, op, val in _iter_hard_constraints(doc):
            joint.setdefault((rt, action, operand), []).append((op, val))

    out: Dict[str, Any] = {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "@type": "Agreement",
        "@id": "urn:witness",
        "assigner": "did:provider:P",
        "assignee": ["did:consumer:C"],
        "target": hard_p.get("target", []),
    }
    rules: Dict[Tuple[str, Any], List[Dict]] = {}
    for (rt, action, operand), constraints in joint.items():
        nums = [(op, _to_num(v)) for op, v in constraints if _to_num(v) is not None]
        template = next(v for _, v in constraints)
        if len(nums) == len(constraints):
            value: Any = _scalar_witness_value(nums, template)
            op = "eq"
        else:
            required: set = set()
            banned: set = set()
            allow: Optional[set] = None
            anyof: List[set] = []
            for o, v in constraints:
                V = _labels(v)
                if o == "eq":
                    required |= V
                    allow = V if allow is None else (allow & V)
                elif o == "isAllOf":
                    required |= V
                elif o == "isAnyOf":
                    anyof.append(V)
                    allow = V if allow is None else (allow & V)
                elif o in ("isNoneOf", "neq"):
                    banned |= V

            # Prefer a single label inside *every* allow-list: an `eq` term is the
            # cleanest committed contract and satisfies `isAnyOf`, `isAllOf` and
            # `eq` requirements alike.
            common = (allow if allow is not None else set()) - banned
            if required:
                cand = (required - banned)
                if common:
                    cand = cand & common or cand
                value: Any = sorted(cand)[0] if len(cand) == 1 else sorted(cand)
                op = "eq" if isinstance(value, str) else "isAllOf"
            elif common:
                value, op = sorted(common)[0], "eq"
            elif anyof:
                # Disjoint allow-lists. Under ODRL overlap semantics a committed
                # `isAnyOf S` still satisfies `isAnyOf A` whenever `S & A` is
                # non-empty, so one label taken from each list satisfies both.
                S = {sorted(V)[0] for V in anyof} - banned
                value, op = sorted(S), "isAnyOf"
            else:
                # Only exclusion rules: draw any label neither party bans.
                pool = set()
                if universe is not None:
                    pool = universe.get((rt, action, operand), set())
                pool = (pool | _labels(template)) - banned
                value, op = (sorted(pool)[0] if pool else sorted(_labels(template))[0]), "eq"
        rules.setdefault((rt, action), []).append({
            "leftOperand": f"odrl:{operand}",
            "operator": f"odrl:{op}",
            "rightOperand": value,
        })

    for (rt, action), constraints in rules.items():
        out.setdefault(rt, []).append({"action": action, "constraint": constraints})
    return out


# ======================================================================
# structural checks
# ======================================================================
def structural_issues(docs: Dict[str, Dict]) -> List[str]:
    problems: List[str] = []
    for name, doc in docs.items():
        present = [rt for rt in RULE_TYPES if doc.get(rt)]
        if name in ("provider", "consumer") and len(present) < 3:
            problems.append(f"{name}: missing rule types {set(RULE_TYPES) - set(present)}")
        for rt in RULE_TYPES:
            for rule in doc.get(rt, []) or []:
                if not rule.get("action"):
                    problems.append(f"{name}:{rt}: rule without an action")
                if not rule.get("constraint"):
                    problems.append(f"{name}:{rt}:{rule.get('action')}: rule without constraints")
                for c in rule.get("constraint", []) or []:
                    for key in ("leftOperand", "operator", "rightOperand"):
                        if key not in c:
                            problems.append(f"{name}:{rt}: constraint missing {key}")

    # a hard-policy operand absent from the aspiration can never be negotiated
    for hard, asp, who in (("hard_provider", "provider", "provider"),
                           ("hard_consumer", "consumer", "consumer")):
        asp_keys = {(rt, r.get("action"), strip_key(str(c["leftOperand"])))
                    for rt in RULE_TYPES for r in docs[asp].get(rt, []) or []
                    for c in r.get("constraint", []) or []}
        for rt, action, operand, _, _ in _iter_hard_constraints(docs[hard]):
            if (rt, action, operand) not in asp_keys:
                problems.append(
                    f"{who}: hard {rt}:{action}:{operand} is absent from the aspiration")
    return problems


def verify_one(directory: str) -> Dict[str, Any]:
    docs = {}
    for name in ("provider", "consumer", "hard_provider", "hard_consumer"):
        with open(os.path.join(directory, f"{name}.json"), encoding="utf-8") as f:
            docs[name] = json.load(f)

    meta_path = os.path.join(directory, "meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    ops = set()
    rule_types_used = set()
    for doc in docs.values():
        for rt in RULE_TYPES:
            for rule in doc.get(rt, []) or []:
                for c in rule.get("constraint", []) or []:
                    ops.add(strip_key(str(c["operator"])))
                    rule_types_used.add(rt)

    blocked = blocking_operands(docs["hard_provider"], docs["hard_consumer"])
    analytic_feasible = not blocked
    blocked_strict = blocking_operands(docs["hard_provider"], docs["hard_consumer"],
                                       containment=True,
                                       universe=label_universe(docs))
    containment_feasible = not blocked_strict

    witness = build_witness(docs["hard_provider"], docs["hard_consumer"],
                            label_universe(docs))
    v_p = feasibility_report(witness, docs["hard_provider"])
    v_c = feasibility_report(witness, docs["hard_consumer"])
    constructive_feasible = not v_p and not v_c

    return {
        "scenario": os.path.basename(directory),
        "structural": structural_issues(docs),
        "analytic_feasible": analytic_feasible,
        "constructive_feasible": constructive_feasible,
        "blocking_operands": blocked,
        "containment_feasible": containment_feasible,
        "blocking_operands_containment": blocked_strict,
        "witness_violations": (v_p + v_c)[:4],
        "operators_missing": sorted(set(ALL_OPERATORS) - ops),
        "rule_types_missing": sorted(set(RULE_TYPES) - rule_types_used),
        "agree": analytic_feasible == constructive_feasible,
        "labelled_feasible": (None if meta is None else bool(meta.get("feasible"))),
        "n_issues": sum(len(r.get("constraint", []))
                        for rt in RULE_TYPES
                        for r in docs["provider"].get(rt, []) or []),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--expect-meta", action="store_true",
                    help="fail if a scenario's meta.json label disagrees with the oracle")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    names = sorted(d for d in os.listdir(args.root)
                   if os.path.isdir(os.path.join(args.root, d)) and not d.startswith("_"))
    results = [verify_one(os.path.join(args.root, n)) for n in names]

    n_struct = sum(1 for r in results if r["structural"])
    n_disagree = sum(1 for r in results if not r["agree"])
    n_feasible = sum(1 for r in results if r["analytic_feasible"])
    n_mislabel = sum(1 for r in results
                     if r["labelled_feasible"] is not None
                     and r["labelled_feasible"] != r["analytic_feasible"])

    if not args.quiet:
        for r in results:
            if r["structural"] or not r["agree"] or (
                    r["labelled_feasible"] is not None
                    and r["labelled_feasible"] != r["analytic_feasible"]):
                print(f"[!] {r['scenario']}")
                for p in r["structural"][:5]:
                    print(f"      structure: {p}")
                if not r["agree"]:
                    print(f"      oracle disagreement: analytic={r['analytic_feasible']} "
                          f"constructive={r['constructive_feasible']} "
                          f"blocked={r['blocking_operands'][:3]} "
                          f"witness={r['witness_violations'][:2]}")
                if r["labelled_feasible"] is not None and r["labelled_feasible"] != r["analytic_feasible"]:
                    print(f"      label says feasible={r['labelled_feasible']}, "
                          f"oracle says {r['analytic_feasible']}")

    issues = sum(len(r["structural"]) for r in results)
    n_full_ops = sum(1 for r in results if not r["operators_missing"])
    n_all_rt = sum(1 for r in results if not r["rule_types_missing"])
    print(f"all 9 operators present: {n_full_ops}/{len(results)} | "
          f"all 3 rule types present: {n_all_rt}/{len(results)}")
    n_contain = sum(1 for r in results if r["containment_feasible"])
    print(f"containment-feasible (stricter, agent-reachable): {n_contain} "
          f"({n_contain / max(1, len(results)):.0%})")
    print(f"{len(results)} scenarios | feasible {n_feasible} "
          f"({n_feasible / max(1, len(results)):.0%}) | "
          f"mean issues {sum(r['n_issues'] for r in results) / max(1, len(results)):.1f}")
    print(f"structural problems: {issues} across {n_struct} scenarios")
    print(f"oracle disagreements: {n_disagree}")
    if any(r["labelled_feasible"] is not None for r in results):
        print(f"mislabelled: {n_mislabel}")

    bad = n_struct + n_disagree + (n_mislabel if args.expect_meta else 0)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
