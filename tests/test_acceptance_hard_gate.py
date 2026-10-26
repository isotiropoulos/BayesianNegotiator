"""Regression tests for the acceptance-side role asymmetry and the hard gate.

Background
----------
In the FULL-scenario benchmark every pairing of a non-BAMBON *provider* with a
BAMBON *consumer* ran to the 200-turn deadline with **zero** agreements, over
100 seeds, with ``std(turns) == 0``. That is not stubbornness, it is a code path
that cannot fire. Three defects compounded:

1. **Categorical hard rules did not bound a proposal.** ``SetBelief.sample``
   treats ``required["isAnyOf"]`` as labels to *add*, never as a whitelist, so a
   belief-driven set grew past its own hard allow-list — BAMBON proposed
   ``purpose isAllOf {analytics, open-source, publication, research}`` against
   its own hard ``purpose isAnyOf {analytics, publication}``.

2. **The ``isAnyOf`` correction could not repair that.** It only *added* missing
   allowed labels; when the offending set already contained every allowed label
   the pool was empty and the "correction" reproduced the violating set
   verbatim. A fixed point: BAMBON re-sent an offer it could never accept.

3. **Acceptance had no hard gate outside ``kind == "num"``, and no
   mutual-satisfaction shortcut.** For an ``isAllOf`` set of k labels the belief
   confidence is a product of k posterior means, so even a byte-identical echo
   of BAMBON's own offer scored below the acceptance bar forever.

The two invariants pinned here:

* BAMBON accepts a hard-feasible offer that already meets its own demand — in
  particular its own offer echoed back — in **both** roles.
* BAMBON never accepts, and never proposes, a term breaching its own hard
  policy, for numeric **and** categorical operands.

No negmas dependency; runs off ``tests/data/full``.
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys

import numpy as np
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for p in (ROOT, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.bambon.negotiator import Bambon  # noqa: E402
from src.bambon.distributions.belief import BeliefRegistry  # noqa: E402
from src.bambon.utils.helpers import (  # noqa: E402
    satisfies,
    strip_key,
    is_iso_date,
    iso_to_days,
    is_iso_duration,
    duration_to_seconds,
)

RULE_TYPES = ("permission", "obligation", "prohibition")
NUM_OPS = ("lt", "lteq", "gt", "gteq", "eq", "neq")

PROFILE = dict(
    lr=0.50, inertia=0.55, tau_in=0.60, tau_out=0.74, max_set_k=3,
    no_agreement_until=3, hopeless_window=16, jaccard_snap_threshold=0.45,
    max_rounds=150, tau_cooling_threshold=0.55, tau_in_cooled=0.60,
    tau_out_cooled=0.56, lr_min=0.32, lr_max=0.95, vol_window=5,
    max_param_delta=0.12, hopeless_severity=0.98, hopeless_patience=24,
    hopeless_context=0.06,
)


def _load(name):
    with open(os.path.join(TESTS_DIR, "data", "full", f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scenario():
    return {
        "provider": _load("provider"),
        "hard_provider": _load("hard_provider"),
        "consumer": _load("consumer"),
        "hard_consumer": _load("hard_consumer"),
    }


def _agent(role, init_offer, hard_policy):
    kwargs = {k: v for k, v in PROFILE.items() if k != "lr"}
    agent = Bambon(name=role, lr=PROFILE["lr"],
                   init_offer=copy.deepcopy(init_offer),
                   hard_policy=hard_policy, **kwargs)
    agent.init_offer = init_offer
    agent.hard_policy = hard_policy
    return agent


def _to_num(v):
    """ISO date / ISO duration / number -> comparable scalar, else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            if is_iso_date(v):
                return float(iso_to_days(v))
            if is_iso_duration(v):
                return float(duration_to_seconds(v))
            return float(v)
        except Exception:
            return None
    return None


def _clause_ok(hard_op, hard_val, agreed_op, agreed_val):
    """Same compatibility test the benchmark uses, so "violation" means the same
    thing here as in ``benchmark_baselines_full.count_hard_violations``: temporal
    and duration operands are compared on the real line, and the agreed
    right-operand is treated as a committed point."""
    hn, an = _to_num(hard_val), _to_num(agreed_val)
    if hn is not None and an is not None:
        res = satisfies(hard_op if hard_op in NUM_OPS else "eq", hn, "eq", an)
    else:
        res = satisfies(hard_op, hard_val, agreed_op, agreed_val)
    return res is None or bool(res)


def _hard_violations(offer, hard_policy):
    """Hard constraints of ``hard_policy`` that ``offer`` breaches or drops."""
    if offer is None or not hard_policy:
        return []
    bad = []
    for rule_type in RULE_TYPES:
        for hrule in hard_policy.get(rule_type, []) or []:
            action = hrule.get("action")
            for hc in hrule.get("constraint", []) or []:
                operand = strip_key(hc["leftOperand"])
                hop = strip_key(hc["operator"])
                hval = hc["rightOperand"]
                found = False
                for orule in offer.get(rule_type, []) or []:
                    if orule.get("action") != action:
                        continue
                    for oc in orule.get("constraint", []) or []:
                        if strip_key(oc["leftOperand"]) != operand:
                            continue
                        found = True
                        if not _clause_ok(hop, hval, strip_key(oc["operator"]), oc["rightOperand"]):
                            bad.append((rule_type, action, operand, hop, hval,
                                        oc["operator"], oc["rightOperand"]))
                if not found:
                    bad.append((rule_type, action, operand, "MISSING", hval, None, None))
    return bad


class EchoOpponent:
    """Concedes for ``settle_after`` rounds, then simply echoes what it receives.

    This is the shape of every scripted baseline once it stops moving (and of
    ``AspirationNegotiator`` after round 20). Facing it, BAMBON is looking at
    *its own last offer*, so it must be able to close.
    """

    def __init__(self, init_offer, hard_policy, settle_after=6):
        self.init_offer = copy.deepcopy(init_offer)
        self.hard_policy = hard_policy
        self.settle_after = settle_after
        self.round = 0

    def observe(self, received_offer):
        self.round += 1
        if self.round >= self.settle_after:
            return True, received_offer
        return False, copy.deepcopy(self.init_offer)


def _run(agent1, agent2, max_turns=200):
    received = copy.deepcopy(agent2.init_offer)
    turn, accepted, final = 0, 0, None
    while turn < max_turns:
        current = agent1 if turn % 2 == 0 else agent2
        accept, received = current.observe(received)
        if accept:
            accepted += 1
            final = received
        else:
            accepted = 0
        if accepted == 2:
            return final, turn
        turn += 1
    return None, turn


# --------------------------------------------------------------------------- #
# 1. The echo case, in BOTH roles                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("role", ["provider", "consumer"])
def test_bambon_closes_against_an_echoing_opponent(scenario, role):
    """BAMBON must reach agreement with an opponent that echoes its own offer.

    Pre-fix this passed as provider (7 turns vs TitForTat) and failed as
    consumer (0/1500 agreements, every one a 200-turn timeout) — the asymmetry
    was the single largest distortion in the benchmark.
    """
    random.seed(0)
    np.random.seed(0)
    if role == "provider":
        bambon = _agent("provider", scenario["provider"], scenario["hard_provider"])
        other = EchoOpponent(scenario["consumer"], scenario["hard_consumer"])
        agreement, turns = _run(bambon, other)
    else:
        bambon = _agent("consumer", scenario["consumer"], scenario["hard_consumer"])
        other = EchoOpponent(scenario["provider"], scenario["hard_provider"])
        agreement, turns = _run(other, bambon)

    assert agreement is not None, (
        f"BAMBON as {role} never agreed with an echoing opponent (ran {turns} turns)"
    )


@pytest.mark.parametrize("role", ["provider", "consumer"])
def test_bambon_accepts_its_own_offer_echoed_back(scenario, role):
    """The minimal unit form: feed BAMBON exactly what it just proposed.

    Every issue is then trivially satisfied by its own demand, so acceptance
    must fire once the ``no_agreement_until`` warm-up has elapsed. This is the
    assertion that was red before the fix.
    """
    random.seed(0)
    np.random.seed(0)
    if role == "provider":
        bambon = _agent("provider", scenario["provider"], scenario["hard_provider"])
        seed_offer = copy.deepcopy(scenario["consumer"])
    else:
        bambon = _agent("consumer", scenario["consumer"], scenario["hard_consumer"])
        seed_offer = copy.deepcopy(scenario["provider"])

    accept, mine = bambon.observe(seed_offer)
    for _ in range(60):
        accept, mine = bambon.observe(mine)   # hand back exactly what it proposed
        if accept:
            break
    assert accept, f"BAMBON as {role} refused its own offer echoed back for 60 rounds"


# --------------------------------------------------------------------------- #
# 2. The hard-policy guarantee: never accept, never propose, a breaching term   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("role", ["provider", "consumer"])
def test_no_hard_violation_in_any_proposal(scenario, role):
    """Every offer BAMBON emits must satisfy its own hard policy.

    Not just the final agreement: a proposal that breaches BAMBON's own policy
    is one it can never accept back, which is exactly how the deadlock arose.
    Covers categorical operands, which had no enforcement on the sampling path
    at all (``SetBelief.sample`` ignores an ``isAnyOf`` whitelist).
    """
    random.seed(0)
    np.random.seed(0)
    if role == "provider":
        bambon = _agent("provider", scenario["provider"], scenario["hard_provider"])
        hard = scenario["hard_provider"]
        other = EchoOpponent(scenario["consumer"], scenario["hard_consumer"], settle_after=4)
        seq = lambda: _run(bambon, other)  # noqa: E731
    else:
        bambon = _agent("consumer", scenario["consumer"], scenario["hard_consumer"])
        hard = scenario["hard_consumer"]
        other = EchoOpponent(scenario["provider"], scenario["hard_provider"], settle_after=4)
        seq = lambda: _run(other, bambon)  # noqa: E731

    emitted = []
    original_observe = bambon.observe

    def spy(offer):
        accept, mine = original_observe(offer)
        emitted.append(copy.deepcopy(mine))
        return accept, mine

    bambon.observe = spy
    seq()

    assert emitted, "BAMBON never emitted an offer"
    breaches = []
    for i, offer in enumerate(emitted):
        bad = [b for b in _hard_violations(offer, hard) if b[3] != "MISSING"]
        if bad:
            breaches.append((i, bad))
    assert not breaches, (
        f"BAMBON as {role} proposed terms breaching its own hard policy: {breaches[:3]}"
    )


@pytest.mark.parametrize("role", ["provider", "consumer"])
def test_no_hard_violation_in_agreement(scenario, role):
    """The headline guarantee: a signed agreement never breaches own hard policy."""
    random.seed(1)
    np.random.seed(1)
    if role == "provider":
        bambon = _agent("provider", scenario["provider"], scenario["hard_provider"])
        hard = scenario["hard_provider"]
        other = EchoOpponent(scenario["consumer"], scenario["hard_consumer"])
        agreement, _ = _run(bambon, other)
    else:
        bambon = _agent("consumer", scenario["consumer"], scenario["hard_consumer"])
        hard = scenario["hard_consumer"]
        other = EchoOpponent(scenario["provider"], scenario["hard_provider"])
        agreement, _ = _run(other, bambon)

    if agreement is None:
        pytest.skip("no agreement reached; nothing to check")
    bad = [b for b in _hard_violations(agreement, hard) if b[3] != "MISSING"]
    assert not bad, f"BAMBON as {role} signed a deal breaching its own hard policy: {bad}"


def test_accept_gate_rejects_a_breaching_term(scenario):
    """The gate must veto directly, not merely as a side effect of the sampler.

    Hand BAMBON an offer whose categorical term is a strict superset of its hard
    allow-list. The mutual-satisfaction shortcut must not fire and the belief
    model must not be able to wave it through.
    """
    random.seed(0)
    np.random.seed(0)
    bambon = _agent("consumer", scenario["consumer"], scenario["hard_consumer"])

    # purpose: hard policy is isAnyOf {analytics, publication}
    target = None
    for key, registry in bambon.beliefs.items():
        for issue in registry.issue_models:
            if issue.operand == "purpose" and issue.hard_rule:
                target = issue
                break
        if target:
            break
    assert target is not None, "scenario no longer has a hard-constrained purpose issue"

    allowed, _required, _banned = BeliefRegistry._hard_categorical_sets(target)
    assert allowed, "expected an isAnyOf/eq hard rule bounding purpose"
    breaching = set(allowed) | {"open-source", "research"}

    assert BeliefRegistry._hard_commit_violation(target, "isAllOf", breaching) is True
    # ...and the projection repairs it rather than leaving it out of bounds.
    repaired = set(BeliefRegistry._project_to_hard_categorical(target, breaching))
    assert repaired <= set(allowed) and repaired
    assert BeliefRegistry._hard_commit_violation(target, "isAllOf", repaired) is False


def test_numeric_point_gate_still_holds(scenario):
    """A concrete numeric point outside a hard bound is rejected even though the
    two *constraints* overlap as rays (``gteq 897`` vs a ``gteq 900`` floor)."""
    bambon = _agent("consumer", scenario["consumer"], scenario["hard_consumer"])
    checked = 0
    for registry in bambon.beliefs.values():
        for issue in registry.issue_models:
            if issue.kind != "num" or not issue.hard_rule:
                continue
            lo, hi = BeliefRegistry._hard_numeric_bounds(issue)
            if lo is not None:
                assert BeliefRegistry._hard_commit_violation(issue, "gteq", lo - 1) is True
                assert BeliefRegistry._hard_commit_violation(issue, "gteq", lo) is False
                checked += 1
            if hi is not None:
                assert BeliefRegistry._hard_commit_violation(issue, "lteq", hi + 1) is True
                assert BeliefRegistry._hard_commit_violation(issue, "lteq", hi) is False
                checked += 1
    assert checked, "no numeric hard bounds found in the scenario"
