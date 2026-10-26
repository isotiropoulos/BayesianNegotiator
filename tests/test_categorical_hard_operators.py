"""Regression tests for categorical hard rules written with ``eq`` / ``neq``.

The bug
-------
``satisfies(my_op, my_val, their_op, their_val)`` normalised a singleton
collection on the *right* operand but not on the left. Its ``eq`` and ``neq``
branches compare ``my_val`` as a bare label, yet a categorical issue's
``to_internal`` wraps every right operand in a set -- so a hard rule
``purpose eq education`` reached the predicate as ``{'education'}`` and

    satisfies("eq", {"education"}, "isAnyOf", {"education"})

returned ``False``. The agent therefore treated the term that *exactly*
satisfies its own hard rule as a violation, vetoed it, and could never close:
every negotiation with a categorical ``eq`` hard rule ran to the deadline.

``isAllOf`` was unaffected because its branch reads the set-ified ``A`` rather
than the raw ``my_val`` -- which is why the failure looked like "containment
operators break BAMBON" when it was really "``eq`` breaks BAMBON".

Measured effect of the fix (``data/generator/diagnose_ood.py``): BAMBON self-play
agreement on the generated out-of-distribution family went from 0.00 to 1.00,
and the aggregate out-of-distribution agreement rate from 0.16 to a competitive
level. Nothing on the 50 hand-authored scenarios could be affected via the
*scoring* path, which
:func:`test_fix_does_not_change_the_compliance_predicate_on_the_corpus` pins
down explicitly.

Run with::

    python -m pytest tests/test_categorical_hard_operators.py -q
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import random
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests"), os.path.join(_ROOT, "data", "generator")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_baselines_full as B                                # noqa: E402
from src.bambon.distributions.belief import BeliefRegistry          # noqa: E402
from src.bambon.models.models import NegotiationInfeasible          # noqa: E402
from src.bambon.negotiator import Bambon                            # noqa: E402
from src.bambon.utils.helpers import satisfies                      # noqa: E402

SCEN_ROOT = os.path.join(_ROOT, "data", "50scenarios")

PROFILE = {
    "inertia": 0.55, "tau_in": 0.60, "tau_out": 0.74, "max_set_k": 3,
    "no_agreement_until": 3, "hopeless_window": 16, "jaccard_snap_threshold": 0.45,
    "max_rounds": 150, "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.60,
    "tau_out_cooled": 0.56, "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5,
    "max_param_delta": 0.12, "hopeless_severity": 0.98, "hopeless_patience": 24,
    "hopeless_context": 0.06,
}

CATEGORICAL_OPS = ("eq", "neq", "isAnyOf", "isAllOf", "isNoneOf")


# ======================================================================
# 1. the predicate itself
# ======================================================================
@pytest.mark.parametrize("their_op", CATEGORICAL_OPS)
@pytest.mark.parametrize("wrap", (list, set, tuple))
def test_satisfies_normalises_a_singleton_left_operand(their_op, wrap):
    """A singleton collection on the left must behave like the bare label."""
    bare = satisfies("eq", "education", their_op, {"education"})
    wrapped = satisfies("eq", wrap(["education"]), their_op, {"education"})
    assert bare == wrapped, (
        f"eq/{their_op}: bare={bare} but singleton-{wrap.__name__}={wrapped}")


def test_the_exact_bug():
    """The specific call that used to return False."""
    assert satisfies("eq", {"education"}, "isAnyOf", {"education"}) is True
    assert satisfies("eq", {"education"}, "eq", "education") is True
    assert satisfies("eq", {"education"}, "isAllOf", {"education"}) is True
    assert satisfies("eq", {"education"}, "isNoneOf", {"education"}) is False


def test_satisfies_still_rejects_what_it_should():
    assert satisfies("eq", {"education"}, "eq", "commercial") is False
    assert satisfies("eq", {"education"}, "isAnyOf", {"commercial"}) is False
    # a multi-label left operand is *not* normalised, so isAllOf stays containment
    assert satisfies("isAllOf", {"a", "b"}, "isAnyOf", {"a"}) is False
    assert satisfies("isAllOf", {"a", "b"}, "isAnyOf", {"a", "b"}) is True


# ======================================================================
# 2. the hard-rule and veto layers that consume it
# ======================================================================
def _registry(offer_op, offer_val, hard_op, hard_val):
    return BeliefRegistry(
        [{"leftOperand": "purpose", "operator": offer_op, "rightOperand": offer_val}],
        [{"leftOperand": "purpose", "operator": hard_op, "rightOperand": hard_val}],
    )


def test_hard_rule_accepts_the_term_that_satisfies_it():
    issue = _registry("isAnyOf", ["research", "education"], "eq", "education").issue_models[0]
    assert issue.hard_rule, "expected a hard rule on the issue"
    for rule in issue.hard_rule:
        assert rule("isAnyOf", {"education"}) is True
        assert rule("eq", "education") is True
        assert rule("isAnyOf", {"commercial"}) is False


def test_commit_gate_does_not_veto_a_satisfying_categorical_term():
    issue = _registry("isAnyOf", ["research", "education"], "eq", "education").issue_models[0]
    assert BeliefRegistry._hard_commit_violation(issue, "isAnyOf", {"education"}) is False
    assert BeliefRegistry._hard_commit_violation(issue, "isAnyOf", {"commercial"}) is True


def test_projection_still_lands_inside_the_eq_allow_set():
    issue = _registry("isAnyOf", ["research", "education"], "eq", "education").issue_models[0]
    projected = set(BeliefRegistry._project_to_hard_categorical(
        issue, {"research", "education", "commercial"}))
    assert projected == {"education"}


# ======================================================================
# 3. end to end: every reachable hard-operator pair must close
# ======================================================================
def _doc(op, val, tag):
    return {
        "@context": "http://www.w3.org/ns/odrl.jsonld", "@type": "Offer",
        "@id": f"urn:test:{tag}", "assigner": "did:provider:P",
        "assignee": ["did:consumer:C"], "target": ["urn:asset:d"],
        "permission": [{"action": "use", "constraint": [
            {"leftOperand": "odrl:purpose", "operator": f"odrl:{op}",
             "rightOperand": val}]}],
    }


def _bambon(role, init, hard):
    agent = Bambon(name=role, lr=0.50, init_offer=copy.deepcopy(init),
                   hard_policy=hard, **PROFILE)
    agent.init_offer = init
    agent.hard_policy = hard
    return agent


# provider aspires to {research, education}, consumer to {education, commercial};
# the shared label is `education`, so a lawful joint term exists in every case
# except the last, where the two hard policies demand different single labels.
ASP_P = _doc("isAnyOf", ["research", "education"], "ap")
ASP_C = _doc("isAnyOf", ["education", "commercial"], "ac")

REACHABLE = [
    ("isAnyOf/isAnyOf", ("isAnyOf", ["research", "education"]), ("isAnyOf", ["education", "commercial"])),
    ("eq/eq same label", ("eq", "education"), ("eq", "education")),
    ("eq/isAnyOf", ("eq", "education"), ("isAnyOf", ["education", "commercial"])),
    ("isAnyOf/eq", ("isAnyOf", ["research", "education"]), ("eq", "education")),
    ("isAllOf/isAnyOf", ("isAllOf", ["education"]), ("isAnyOf", ["education", "commercial"])),
    ("isAllOf/isAllOf", ("isAllOf", ["education"]), ("isAllOf", ["education"])),
    ("isNoneOf/isAnyOf", ("isNoneOf", ["marketing"]), ("isAnyOf", ["education", "commercial"])),
    ("isNoneOf/eq", ("isNoneOf", ["marketing"]), ("eq", "education")),
]


@pytest.mark.parametrize("label,provider_hard,consumer_hard", REACHABLE,
                         ids=[r[0] for r in REACHABLE])
def test_self_play_closes_for_every_reachable_hard_operator_pair(
        label, provider_hard, consumer_hard):
    hp = _doc(*provider_hard, "hp")
    hc = _doc(*consumer_hard, "hc")
    for seed in range(3):
        random.seed(seed)
        try:
            agreement, turns, _ = B.negotiate_generic(
                _bambon("provider", ASP_P, hp), _bambon("consumer", ASP_C, hc))
        except NegotiationInfeasible:
            agreement, turns = None, -1
        assert agreement is not None, f"{label}: no agreement (turns={turns})"
        assert B.count_hard_violations(agreement, hp) == 0
        assert B.count_hard_violations(agreement, hc) == 0


def test_self_play_still_refuses_an_unreachable_pair():
    """Two `eq` rules on different labels: no lawful joint term, so no deal.

    The agent must not close here, and it should stop early rather than burn the
    whole deadline -- the hopelessness detector is what distinguishes "declined"
    from "timed out".
    """
    hp = _doc("eq", "research", "hp")
    hc = _doc("eq", "commercial", "hc")
    random.seed(0)
    try:
        agreement, turns, _ = B.negotiate_generic(
            _bambon("provider", ASP_P, hp), _bambon("consumer", ASP_C, hc))
    except NegotiationInfeasible:
        agreement, turns = None, 0
    assert agreement is None
    assert turns < B.MAX_TURNS, "should decline rather than time out"


# ======================================================================
# 4. the fix must not move the compliance metric
# ======================================================================
def _corpus_hard_clauses():
    """Every distinct categorical hard clause in the 50 hand-authored scenarios."""
    if not os.path.isdir(SCEN_ROOT):
        return [], []
    clauses, labels = set(), set()
    for name in sorted(os.listdir(SCEN_ROOT)):
        directory = os.path.join(SCEN_ROOT, name)
        if not os.path.isdir(directory) or name.startswith("_"):
            continue
        for fn in ("hard_provider", "hard_consumer"):
            with open(os.path.join(directory, f"{fn}.json"), encoding="utf-8") as f:
                doc = json.load(f)
            for rt in ("permission", "obligation", "prohibition"):
                for rule in doc.get(rt, []) or []:
                    for c in rule.get("constraint", []) or []:
                        v = c["rightOperand"]
                        if isinstance(v, (int, float)):
                            continue
                        if isinstance(v, str) and v[:2].isdigit():
                            continue          # ISO date / duration
                        op = c["operator"].replace("odrl:", "")
                        clauses.add((op, tuple(v) if isinstance(v, list) else v))
                        labels.update(v if isinstance(v, list) else [v])
    return sorted(clauses), sorted(labels)


def test_fix_does_not_change_the_compliance_predicate_on_the_corpus():
    """`count_hard_violations` scores the 50 scenarios exactly as before.

    The normalisation only fires on a *singleton collection* left operand.
    ``count_hard_violations`` passes the hard right operand straight from JSON, so
    the only affected clauses would be singleton-list ones -- and for those the
    ``isAnyOf``/``isAllOf``/``isNoneOf`` branches read the set-ified ``A``, which
    the normalisation leaves unchanged. This test asserts that directly, over
    every categorical hard clause in the corpus crossed with every committed term
    that could be scored against it.
    """
    clauses, labels = _corpus_hard_clauses()
    if not clauses:
        pytest.skip("data/50scenarios not present")

    def normalised_only_on_the_right(my_op, my_val, their_op, their_val):
        """`satisfies` as it behaved before the fix."""
        from src.bambon.utils.helpers import _to_setish
        if isinstance(their_val, (set, list, tuple)) and len(their_val) == 1:
            their_val = list(their_val)[0]
        A, B_ = _to_setish(my_val), _to_setish(their_val)
        try:
            if my_op == "isAnyOf":
                return {"eq": lambda: their_val in A, "neq": lambda: len(A - B_) > 0,
                        "isAnyOf": lambda: len(A & B_) > 0,
                        "isAllOf": lambda: B_.issubset(A),
                        "isNoneOf": lambda: len(A & B_) == 0}.get(their_op, lambda: False)()
            if my_op == "eq":
                return {"eq": lambda: their_val == my_val,
                        "neq": lambda: their_val != my_val,
                        "isAnyOf": lambda: my_val in B_,
                        "isAllOf": lambda: A.issubset(B_),
                        "isNoneOf": lambda: my_val not in B_}.get(their_op, lambda: False)()
        except TypeError:
            return "TypeError"
        return None

    # Well-formed committed terms only: an `eq`/`neq` clause carries a single
    # label, a set-valued operator carries a list. (This project's emitters
    # enforce that -- `format_operator_roperand` rewrites `eq` + multi-value to
    # `isAllOf`.) Ill-formed shapes are covered by the crash test below.
    terms = []
    for op in CATEGORICAL_OPS:
        for lab in labels:
            terms.append((op, lab) if op in ("eq", "neq") else (op, [lab]))
        if op not in ("eq", "neq"):
            for a, b in itertools.combinations(labels[:30], 2):
                terms.append((op, [a, b]))

    differences = []
    for hard_op, hard_val in clauses:
        hv = list(hard_val) if isinstance(hard_val, tuple) else hard_val
        for term_op, term_val in terms:
            before = normalised_only_on_the_right(hard_op, hv, term_op, term_val)
            if before is None or before == "TypeError":
                continue
            after = satisfies(hard_op, hv, term_op, term_val)
            if before != after:
                differences.append((hard_op, hv, term_op, term_val, before, after))
    assert not differences, f"{len(differences)} scoring differences, e.g. {differences[:3]}"


def test_multi_label_eq_clause_no_longer_raises():
    """An `eq` clause carrying several labels used to crash the predicate.

    ``satisfies('isAnyOf', A, 'eq', ['a','b'])`` evaluated ``their_val in A`` on a
    list and raised ``TypeError: unhashable type: 'list'``. The categorical block
    is now written entirely on the set-ified operands, so every operator pair
    returns a value. This shape is ill-formed ODRL and this project's own emitters
    never produce it, but an external document can, and a crash inside the
    compliance check is worse than a defined answer.
    """
    for my_op in CATEGORICAL_OPS:
        for their_op in CATEGORICAL_OPS:
            for their_val in (["a", "b"], "a"):
                result = satisfies(my_op, ["a", "c"], their_op, their_val)
                assert isinstance(result, bool) or result is None, (
                    f"{my_op}/{their_op} returned {result!r}")
