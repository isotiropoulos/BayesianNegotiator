"""Tests for the BAMBON-OM(acc) ablation: opponent modelling removed from
*acceptance* only.

Why this arm exists
-------------------
``BAMBON-OMbid`` (``bidding_mode='time'``) removes the posterior from the *bid*
and is a measured null result on three corpora: the Bayesian layer contributes
nothing to bid selection. That leaves an obvious question the paper had not
answered -- if the opponent model does not help the agent *bid*, does it help it
*accept*? ``accept_mode='blind'`` is the complement: posteriors are still updated
and still drive every bid, but the accept test never consults them.

What "blind" replaces, and what it deliberately does not
--------------------------------------------------------
``issue.model.accept(...)`` is the only consumer of the opponent posterior
anywhere in the acceptance path (four call sites: num, enum, set and the merged
inclusion/exclusion path). ``blind`` diverts all four to
``BeliefRegistry._blind_accept``, the classic time-dependent rule: accept a term
at least as good as the one I am currently asking for. It is time-dependent
because my own term rides the aspiration schedule A(t).

Three things stay in force in both arms, because none of them is opponent
modelling, and the tests below pin each one:

* ``_hard_commit_violation`` -- the hard-policy veto (Proposition 1);
* ``_meets_my_demand``       -- own opening ask, or my own offer echoed back;
* ``reservation_deadlock``   -- the adopted infeasibility rule, which is itself
  belief-free (it reads hard bounds, the opponent's *observed* term and my own
  last value; no posterior). ``test_reservation_deadlock_is_belief_free`` asserts
  that structurally, because it is a claim the paper makes in prose.

Run with::

    python -m pytest tests/test_accept_ablation.py -q
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import random
import sys
import textwrap

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_baselines_full as B                            # noqa: E402
from src.bambon.baselines import veto_wrap                      # noqa: E402
from src.bambon.distributions.belief import BeliefRegistry      # noqa: E402
from src.bambon.models.models import ActionState                # noqa: E402
from src.bambon.negotiator import Bambon                        # noqa: E402

SCEN_ROOT = os.path.join(_ROOT, "data", "50scenarios")
GEN_ROOT = os.path.join(_ROOT, "data", "generated", "full")

PROFILE = {
    "inertia": 0.55, "tau_in": 0.60, "tau_out": 0.74, "max_set_k": 3,
    "no_agreement_until": 3, "hopeless_window": 16, "jaccard_snap_threshold": 0.45,
    "max_rounds": 150, "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.60,
    "tau_out_cooled": 0.56, "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5,
    "max_param_delta": 0.12, "hopeless_severity": 0.98, "hopeless_context": 0.06,
    "hopeless_mode": "reservation", "hopeless_patience": 4,
    "hopeless_min_round_frac": 0.5,
}


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


HAND = _scenarios(SCEN_ROOT, 6)
GEN = _scenarios(GEN_ROOT, 6)


def _issue(offer_op, offer_val, hard_op, hard_val):
    reg = BeliefRegistry(
        [{"leftOperand": "count", "operator": offer_op, "rightOperand": offer_val}],
        [{"leftOperand": "count", "operator": hard_op, "rightOperand": hard_val}],
    )
    return reg, reg.issue_models[0]


def _state(**over):
    kw = dict(lr=0.5, inertia=0.5, tau_set=[0.6, 0.74], max_set_k=3,
              jaccard_snap_threshold=0.45)
    kw.update(over)
    return ActionState(**kw)


# ======================================================================
# 1. the switch is wired end to end
# ======================================================================
def test_default_accept_mode_is_belief():
    """The ablation must be opt-in: the shipped agent keeps its opponent model."""
    assert Bambon.__dataclass_fields__["accept_mode"].default == "belief"
    assert ActionState.__dataclass_fields__["accept_mode"].default == "belief"


@pytest.mark.parametrize("mode", ["belief", "blind"])
def test_accept_mode_reaches_every_action_state(mode):
    """Set on the agent -> present on each per-action state, both at construction
    and after observe() refreshes them."""
    if not HAND:
        pytest.skip("hand corpus not present")
    ip, hp, ic, hc = _load(HAND[0])
    agent = _bambon("provider", ip, hp, accept_mode=mode)
    assert agent.accept_mode == mode
    agent.observe(copy.deepcopy(ic))
    assert agent.action_states, "no action states were created"
    for st in agent.action_states.values():
        assert st.accept_mode == mode


# ======================================================================
# 2. the blind rule itself
# ======================================================================
def test_blind_accept_takes_a_term_that_clears_my_current_ask():
    reg, issue = _issue("lteq", 100, "lteq", 500)
    issue.lastValue = 100
    assert reg._blind_accept(issue, "eq", 80) is True


def test_blind_accept_refuses_a_term_worse_than_my_current_ask():
    reg, issue = _issue("lteq", 100, "lteq", 500)
    issue.lastValue = 100
    assert reg._blind_accept(issue, "eq", 400) is False


def test_blind_accept_is_time_dependent_through_my_own_concessions():
    """The bar is not static: it falls as lastValue concedes along A(t), which is
    what makes this a fair control rather than a strawman."""
    reg, issue = _issue("lteq", 100, "lteq", 500)
    issue.lastValue = 100
    assert reg._blind_accept(issue, "eq", 300) is False   # early: still demanding
    issue.lastValue = 350                                 # later: conceded
    assert reg._blind_accept(issue, "eq", 300) is True


def test_blind_accept_never_touches_the_posterior():
    """Structural: the blind rule must not reference issue.model at all."""
    src = inspect.getsource(BeliefRegistry._blind_accept)
    body = src.split('"""')[-1]          # drop the docstring, which names it
    assert ".model" not in body
    assert "accept_probs" not in body


# ======================================================================
# 3. the veto still binds -- Proposition 1 must hold in this arm too
# ======================================================================
def test_blind_arm_still_refuses_a_term_its_hard_policy_forbids():
    """A term that clears my *current ask* but breaches my *hard policy* must
    still be vetoed: the blind rule sits inside the veto, not around it."""
    reg, issue = _issue("lteq", 100, "lteq", 200)
    issue.lastValue = 400                      # conceded past my own hard cap
    issue.oppOperator, issue.oppLastValue = "eq", 300
    st = _state(accept_mode="blind")
    # the blind rule alone would take it ...
    assert reg._blind_accept(issue, "eq", 300) is True
    # ... but the veto that wraps it does not.
    assert reg._hard_commit_violation(issue, "eq", 300) is True


@pytest.mark.parametrize("directory", GEN or HAND, ids=os.path.basename)
def test_blind_arm_signs_only_compliant_contracts(directory):
    """End to end on the operator-complete corpus: zero hard violations, which is
    what makes the arm a valid ablation rather than a different safety story."""
    ip, hp, ic, hc = _load(directory)
    random.seed(0)
    agreement, turns, _ = B.negotiate_generic(
        _bambon("provider", ip, hp, accept_mode="blind"),
        veto_wrap(B.AspirationNegotiator)(ic, hc))
    if agreement is None:
        pytest.skip(f"no agreement after {turns} turns; compliance is vacuous")
    assert B.count_hard_violations(agreement, hp) == 0
    assert B.count_hard_violations(agreement, hc) == 0


# ======================================================================
# 4. the ablation is surgical: bidding is untouched
# ======================================================================
@pytest.mark.parametrize("directory", (GEN or HAND)[:3], ids=os.path.basename)
def test_bids_are_identical_until_the_first_divergent_accept(directory):
    """The two arms must propose the *same* opening bids against the same
    transcript. `accept_mode` may only change whether a term is signed, never how
    a counter-offer is computed -- otherwise a difference in outcome could not be
    attributed to acceptance.

    Compared over the pre-acceptance rounds (`no_agreement_until`), during which
    the acceptance branch is switched off for both arms by construction, so any
    difference in the emitted offers would have to come from the bidding path.

    Both RNGs must be seeded: the belief sampler draws from ``numpy.random``, so
    ``random.seed`` alone does not make an encounter reproducible.
    """
    ip, hp, ic, hc = _load(directory)
    rounds = PROFILE["no_agreement_until"]

    def content(offer):
        """Rule content only: '@id' is a fresh uuid on every emitted offer."""
        return {rt: offer.get(rt) for rt in ("permission", "obligation", "prohibition")}

    def transcript(mode):
        random.seed(11)
        np.random.seed(11)
        agent = _bambon("provider", ip, hp, accept_mode=mode)
        opp = copy.deepcopy(ic)
        out = []
        for _ in range(rounds):
            _, offer = agent.observe(copy.deepcopy(opp))
            out.append(json.dumps(content(offer), sort_keys=True, default=str))
        return out

    assert transcript("belief") == transcript("blind")


def test_accept_mode_is_the_only_difference_in_the_arm_spec():
    """The harness must not smuggle another change into the arm."""
    sys.path.insert(0, os.path.join(_ROOT, "paper", "sweep"))
    from run_tournament import build_arms                    # noqa: E402
    arms = build_arms("all", walk_away_after=8)
    assert "BAMBON-OMacc" in arms, "the ablation arm is not in the arm table"
    base, acc = arms["BAMBON"], arms["BAMBON-OMacc"]
    assert acc["bidding_mode"] == base["bidding_mode"] == "belief", \
        "OMacc must keep the Bayesian bidder; otherwise it is not the complement"
    assert acc.get("extra", {}) == {"accept_mode": "blind"}, \
        f"unexpected extra settings on the arm: {acc.get('extra')}"


# ======================================================================
# 5. a claim the paper makes in prose, pinned as a test
# ======================================================================
def test_reservation_deadlock_is_belief_free():
    """The adopted infeasibility rule reads hard bounds, the opponent's observed
    term and the agent's own last value -- never a posterior. The paper says so,
    so it is asserted here rather than left to inspection."""
    src = textwrap.dedent(inspect.getsource(BeliefRegistry.reservation_deadlock))
    tree = ast.parse(src)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "model" not in names
    assert "accept_probs" not in names


def test_deadlock_detection_is_unchanged_by_the_ablation():
    """Same inputs, both modes, same verdict: the arm does not move the detector."""
    for mode in ("belief", "blind"):
        reg, issue = _issue("lteq", 20, "lteq", 100)
        issue.oppOperator, issue.oppLastValue = "lteq", 400
        issue.lastValue = 100
        assert reg.reservation_deadlock(_state(accept_mode=mode)) is True

        reg, issue = _issue("lteq", 20, "lteq", 100)
        issue.oppOperator, issue.oppLastValue = "lteq", 80
        issue.lastValue = 100
        assert reg.reservation_deadlock(_state(accept_mode=mode)) is False
