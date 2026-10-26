"""Tests for BAMBON's two infeasibility-detection rules.

The problem
-----------
The rule evaluated in the paper (``hopeless_mode='severity'``) declines when the
two sides have stayed far apart for ``hopeless_patience`` rounds. That measures
*"are we far apart right now"*, which is true early in every hard negotiation --
so on a corpus where a lawful contract always exists it can only ever be wrong.
Measured on the 50 hand-authored scenarios (all feasible by construction) against
a three-opponent panel: **0.378 of encounters ended in a false decline**, and no
setting of ``(hopeless_window, hopeless_patience)`` separates true declines from
false ones -- raising either removes *all* declines, including the correct ones
on genuinely infeasible scenarios. It is a specification problem, not a tuning
problem.

The alternative
---------------
``hopeless_mode='reservation'`` declines only when no signable term remains:

1. the deadline is close enough that waiting cannot help
   (``round_frac >= hopeless_min_round_frac``);
2. the opponent's standing term on some issue is one the agent's own hard policy
   forbids it to commit to; and
3. the agent is already at its reservation on that issue, so it has nothing left
   to concede,

sustained for ``hopeless_patience`` rounds. Condition (3) is what makes it sound:
it cannot fire while the agent still has a concession available.

Default stays ``'severity'`` so every number in the paper remains reproducible.

Run with::

    python -m pytest tests/test_infeasibility_detection.py -q
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_baselines_full as B                            # noqa: E402
from src.bambon.baselines import veto_wrap                      # noqa: E402
from src.bambon.distributions.belief import BeliefRegistry      # noqa: E402
from src.bambon.models.models import ActionState, NegotiationInfeasible  # noqa: E402
from src.bambon.negotiator import Bambon                        # noqa: E402

SCEN_ROOT = os.path.join(_ROOT, "data", "50scenarios")
INF_ROOT = os.path.join(_ROOT, "data", "generated", "infeasible")

PROFILE = {
    "inertia": 0.55, "tau_in": 0.60, "tau_out": 0.74, "max_set_k": 3,
    "no_agreement_until": 3, "hopeless_window": 16, "jaccard_snap_threshold": 0.45,
    "max_rounds": 150, "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.60,
    "tau_out_cooled": 0.56, "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5,
    "max_param_delta": 0.12, "hopeless_severity": 0.98, "hopeless_patience": 24,
    "hopeless_context": 0.06,
}
# The setting measured as strictly better on both axes.
RES = {"hopeless_mode": "reservation", "hopeless_patience": 4,
       "hopeless_min_round_frac": 0.5}


def _scenarios(root, limit):
    if not os.path.isdir(root):
        return []
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
    return [os.path.join(root, n) for n in names[:limit]]


def _load(directory):
    def one(name):
        with open(os.path.join(directory, f"{name}.json"), encoding="utf-8") as f:
            return json.load(f)
    return one("provider"), one("hard_provider"), one("consumer"), one("hard_consumer")


def _bambon(role, init, hard, **over):
    kw = dict(PROFILE)
    kw.update(over)
    agent = Bambon(name=role, lr=0.50, init_offer=copy.deepcopy(init),
                   hard_policy=hard, **kw)
    agent.init_offer = init
    agent.hard_policy = hard
    return agent


HAND = _scenarios(SCEN_ROOT, 8)
INF = _scenarios(INF_ROOT, 5)


# ======================================================================
# the deadlock predicate
# ======================================================================
def _issue(offer_op, offer_val, hard_op, hard_val):
    reg = BeliefRegistry(
        [{"leftOperand": "count", "operator": offer_op, "rightOperand": offer_val}],
        [{"leftOperand": "count", "operator": hard_op, "rightOperand": hard_val}],
    )
    return reg, reg.issue_models[0]


def _state():
    return ActionState(lr=0.5, inertia=0.5, tau_set=[0.6, 0.74], max_set_k=3,
                       jaccard_snap_threshold=0.45)


def test_no_deadlock_when_the_agent_can_still_concede():
    """My cap is 100, I am asking 20, they want 400: I still have room."""
    reg, issue = _issue("lteq", 20, "lteq", 100)
    issue.oppOperator, issue.oppLastValue = "lteq", 400
    issue.lastValue = 20
    assert reg.reservation_deadlock(_state()) is False


def test_deadlock_when_at_the_reservation_and_the_term_is_illegal():
    """I am at my cap of 100, they insist on 400: nothing left to give."""
    reg, issue = _issue("lteq", 20, "lteq", 100)
    issue.oppOperator, issue.oppLastValue = "lteq", 400
    issue.lastValue = 100
    assert reg.reservation_deadlock(_state()) is True


def test_no_deadlock_when_the_standing_term_is_signable():
    """At my cap of 100 and they are offering 80: that is a deal, not a deadlock."""
    reg, issue = _issue("lteq", 20, "lteq", 100)
    issue.oppOperator, issue.oppLastValue = "lteq", 80
    issue.lastValue = 100
    assert reg.reservation_deadlock(_state()) is False


def test_no_deadlock_on_an_accepted_issue():
    reg, issue = _issue("lteq", 20, "lteq", 100)
    issue.oppOperator, issue.oppLastValue = "lteq", 400
    issue.lastValue = 100
    issue.accepted = True
    assert reg.reservation_deadlock(_state()) is False


def test_categorical_deadlock_needs_an_unsignable_term():
    reg = BeliefRegistry(
        [{"leftOperand": "purpose", "operator": "isAnyOf",
          "rightOperand": ["research", "education"]}],
        [{"leftOperand": "purpose", "operator": "isAnyOf", "rightOperand": ["research"]}],
    )
    issue = reg.issue_models[0]
    issue.oppOperator, issue.oppLastValue = "isAllOf", ["marketing"]
    assert reg.reservation_deadlock(_state()) is True
    issue.oppLastValue = ["research"]
    assert reg.reservation_deadlock(_state()) is False


# ======================================================================
# end to end
# ======================================================================
@pytest.mark.parametrize("directory", HAND, ids=os.path.basename)
def test_reservation_mode_does_not_false_decline_against_a_hardliner(directory):
    """A Hardliner never moves but accepts from round 50, and the scenario is
    feasible -- so declining is simply wrong. The severity rule declines here."""
    ip, hp, ic, hc = _load(directory)
    hardliner = veto_wrap(B.HardlinerNegotiator)
    random.seed(0)
    agreement, turns, _ = B.negotiate_generic(
        _bambon("provider", ip, hp, **RES), hardliner(ic, hc))
    assert agreement is not None, f"false decline after {turns} turns"
    assert B.count_hard_violations(agreement, hp) == 0
    assert B.count_hard_violations(agreement, hc) == 0


@pytest.mark.parametrize("directory", HAND, ids=os.path.basename)
def test_reservation_mode_closes_in_self_play_on_a_feasible_scenario(directory):
    ip, hp, ic, hc = _load(directory)
    random.seed(0)
    agreement, turns, _ = B.negotiate_generic(
        _bambon("provider", ip, hp, **RES), _bambon("consumer", ic, hc, **RES))
    assert agreement is not None, f"false decline after {turns} turns"


@pytest.mark.parametrize("directory", INF, ids=os.path.basename)
def test_reservation_mode_still_declines_when_no_lawful_contract_exists(directory):
    """It must not simply be "never decline": on a verified-infeasible scenario it
    has to refuse, and preferably by declining rather than by timing out."""
    ip, hp, ic, hc = _load(directory)
    random.seed(0)
    try:
        agreement, turns, _ = B.negotiate_generic(
            _bambon("provider", ip, hp, **RES), _bambon("consumer", ic, hc, **RES))
    except NegotiationInfeasible:
        agreement, turns = None, 0
    assert agreement is None, "signed a contract on an infeasible scenario"


def test_reservation_mode_declines_rather_than_times_out_somewhere():
    """Aggregate: across the infeasible corpus it should decline, not just stall."""
    if not INF:
        pytest.skip("generated infeasible corpus not present")
    declined = 0
    for directory in INF:
        ip, hp, ic, hc = _load(directory)
        random.seed(0)
        try:
            agreement, turns, _ = B.negotiate_generic(
                _bambon("provider", ip, hp, **RES), _bambon("consumer", ic, hc, **RES))
        except NegotiationInfeasible:
            agreement, turns = None, 0
        if agreement is None and turns < B.MAX_TURNS:
            declined += 1
    assert declined > 0, "never declined on any infeasible scenario"


def test_reservation_is_now_the_default():
    """Adopted after measuring that it dominates 'severity' on every axis.

    'severity' stays available so the originally submitted numbers can still be
    reproduced by passing hopeless_mode='severity'.
    """
    agent = Bambon(name="p", init_offer={"permission": [
        {"action": "use", "constraint": [
            {"leftOperand": "odrl:count", "operator": "odrl:lteq",
             "rightOperand": 10}]}]})
    assert agent.hopeless_mode == "reservation"
    assert agent.hopeless_min_round_frac == 0.5
    legacy = Bambon(name="p", hopeless_mode="severity", init_offer={"permission": [
        {"action": "use", "constraint": [
            {"leftOperand": "odrl:count", "operator": "odrl:lteq",
             "rightOperand": 10}]}]})
    assert legacy.hopeless_mode == "severity"
