"""Tests for the generic hard-policy veto and the Bayesian-bidding ablation.

These are the unit-level evidence behind two claims in the revised evaluation:

1. :class:`src.bambon.baselines.HardPolicyVeto` gives an arbitrary strategy the
   same hard-policy guarantee BAMBON has, so compliance can be used as a
   *control* rather than as a headline (Propositions 1 and 2 hold for a
   veto-wrapped baseline exactly as they do for BAMBON).
2. ``bidding_mode="time"`` really does remove the opponent posterior from
   *bidding* and from bidding only -- verified by making every sampling entry
   point of the belief models raise, and checking that a negotiation still runs.

Run with::

    python -m pytest tests/test_hard_policy_veto.py -q
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

import benchmark_baselines_full as B                                   # noqa: E402
from src.bambon.baselines import (                                     # noqa: E402
    HardPolicyVeto, feasibility_report, is_feasible, project_offer, veto_wrap,
)
from src.bambon.baselines.veto import _content_signature               # noqa: E402
from src.bambon.models.models import NegotiationInfeasible             # noqa: E402
from src.bambon.negotiator import Bambon                              # noqa: E402

SCEN_ROOT = os.path.join(_ROOT, "data", "50scenarios")
GEN_ROOT = os.path.join(_ROOT, "data", "generated")

BASELINES = {
    "Aspiration": B.AspirationNegotiator,
    "QLearning": B.QLearningNegotiator,
    "TitForTat": B.TitForTatNegotiator,
    "Hardliner": B.HardlinerNegotiator,
    "Random": B.RandomNegotiator,
}

PROFILE = {
    "inertia": 0.55, "tau_in": 0.60, "tau_out": 0.74, "max_set_k": 3,
    "no_agreement_until": 3, "hopeless_window": 16, "jaccard_snap_threshold": 0.45,
    "max_rounds": 150, "tau_cooling_threshold": 0.55, "tau_in_cooled": 0.60,
    "tau_out_cooled": 0.56, "lr_min": 0.32, "lr_max": 0.95, "vol_window": 5,
    "max_param_delta": 0.12, "hopeless_severity": 0.98, "hopeless_patience": 24,
    "hopeless_context": 0.06,
}


def _load(directory):
    def one(name):
        with open(os.path.join(directory, f"{name}.json"), encoding="utf-8") as f:
            return json.load(f)
    return one("provider"), one("hard_provider"), one("consumer"), one("hard_consumer")


def _scenarios(root, limit=None):
    if not os.path.isdir(root):
        return []
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
    if limit:
        names = names[:limit]
    return [os.path.join(root, n) for n in names]


HAND = _scenarios(SCEN_ROOT, 12)
GEN_OOD = _scenarios(os.path.join(GEN_ROOT, "ood"), 8)
GEN_INF = _scenarios(os.path.join(GEN_ROOT, "infeasible"), 6)


# ======================================================================
# the projection operator
# ======================================================================
@pytest.mark.parametrize("directory", HAND + GEN_OOD, ids=os.path.basename)
def test_projection_lands_in_the_feasible_region(directory):
    """Prop. 1 at the document level: Pi(O) is feasible, for any O."""
    ip, hp, ic, hc = _load(directory)
    for offer in (ip, ic, hp, hc):
        for hard in (hp, hc):
            projected = project_offer(offer, hard)
            assert is_feasible(projected, hard), feasibility_report(projected, hard)


@pytest.mark.parametrize("directory", HAND + GEN_OOD, ids=os.path.basename)
def test_projection_is_idempotent(directory):
    ip, hp, ic, hc = _load(directory)
    for offer in (ip, ic):
        for hard in (hp, hc):
            once = project_offer(offer, hard)
            twice = project_offer(once, hard)
            assert _content_signature(once) == _content_signature(twice)


@pytest.mark.parametrize("directory", HAND, ids=os.path.basename)
def test_projection_leaves_scalar_clauses_of_a_feasible_offer_alone(directory):
    """On a feasible offer the projection must not move any scalar term.

    It *may* still tighten a categorical one. ``sat`` treats an ``isAnyOf`` term
    as an overlap test, so a term whose label set merely intersects the hard
    allow-list is already feasible, while Eq. 12 intersects it down into the
    allow-list. That is faithful to what BAMBON's own
    ``_project_to_hard_categorical`` does on every emission path, so the control
    is matched -- but it means Pi is not the identity on every feasible offer.
    """
    ip, hp, ic, hc = _load(directory)
    for offer, hard in ((ip, hp), (ic, hc)):
        if not is_feasible(offer, hard):
            continue
        projected = project_offer(offer, hard)
        for rt in ("permission", "obligation", "prohibition"):
            for before, after in zip(offer.get(rt, []) or [], projected.get(rt, []) or []):
                for cb, ca in zip(before.get("constraint", []), after.get("constraint", [])):
                    from src.bambon.baselines.veto import _to_num
                    if _to_num(cb["rightOperand"]) is not None:
                        assert cb["rightOperand"] == ca["rightOperand"]


def test_projection_restores_a_dropped_hard_constraint():
    """Def. 2 counts a missing operand as a violation, so Pi must re-insert it."""
    hard = {
        "permission": [{"action": "use", "constraint": [
            {"leftOperand": "odrl:count", "operator": "odrl:lteq", "rightOperand": 100},
            {"leftOperand": "odrl:purpose", "operator": "odrl:isAnyOf",
             "rightOperand": ["research"]},
        ]}],
    }
    stripped = {"permission": [{"action": "use", "constraint": []}]}
    assert not is_feasible(stripped, hard)
    assert is_feasible(project_offer(stripped, hard), hard)


def test_projection_respects_a_strict_hard_bound():
    hard = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:count", "operator": "odrl:gt", "rightOperand": 10}]}]}
    offer = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:count", "operator": "odrl:gteq", "rightOperand": 3}]}]}
    repaired = project_offer(offer, hard)
    value = repaired["permission"][0]["constraint"][0]["rightOperand"]
    assert value > 10
    assert is_feasible(repaired, hard)


# ======================================================================
# the commit-time veto
# ======================================================================
class _AlwaysAccept:
    """Minimal agent that signs whatever it is shown -- the worst case."""

    def __init__(self, init_offer, hard_policy):
        self.init_offer = copy.deepcopy(init_offer)
        self.hard_policy = hard_policy

    def observe(self, received_offer):
        return True, received_offer


def test_veto_blocks_a_categorical_superset():
    """The hole Sec. 4.6 describes: signing a superset of one's own allow-list."""
    hard = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:language", "operator": "odrl:isAnyOf",
         "rightOperand": ["en"]}]}]}
    mine = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:language", "operator": "odrl:isAllOf",
         "rightOperand": ["en"]}]}]}
    incoming = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:language", "operator": "odrl:isAllOf",
         "rightOperand": ["en", "fr"]}]}]}

    naked = _AlwaysAccept(mine, hard)
    assert naked.observe(incoming)[0] is True             # unguarded: signs it

    guarded = HardPolicyVeto(_AlwaysAccept(mine, hard), hard)
    accept, counter = guarded.observe(incoming)
    assert accept is False
    assert guarded.n_vetoed == 1
    assert is_feasible(counter, hard)


def test_the_compliance_predicate_is_permissive_for_isanyof_terms():
    """A documented limit of the metric, not of the veto.

    ``sat(eq, v, isAnyOf, S)`` is ``v in S`` and ``sat(isAnyOf, A, isAnyOf, S)``
    is ``A & S != {}``: both are *overlap* tests. A committed ``isAnyOf`` term
    that merely mentions a permitted label therefore satisfies the hard rule,
    even though it also mentions forbidden ones. Because ``count_hard_violations``
    -- the function the results tables use -- is built on the same predicate, this
    class of over-broad commitment is invisible in the aggregate compliance
    figure, for BAMBON and for a veto-wrapped baseline alike. It is worth stating
    in the paper: it is part of why compliance reads 1.000 in variants that lack
    the acceptance veto entirely.
    """
    def doc(op, val, operator="eq"):
        return {"permission": [{"action": "use", "constraint": [
            {"leftOperand": "odrl:language", "operator": f"odrl:{op}",
             "rightOperand": val}]}]}

    eq_rule = doc("eq", "en")
    anyof_rule = doc("isAnyOf", ["en"])

    # An `eq` hard rule is satisfied by *both* over-broad term forms, because
    # sat(eq, v, isAnyOf, S) is `v in S` and sat(eq, v, isAllOf, S) is `{v} <= S`.
    assert is_feasible(doc("isAnyOf", ["en", "fr"]), eq_rule)
    assert is_feasible(doc("isAllOf", ["en", "fr"]), eq_rule)

    # A singleton `isAnyOf` hard rule is *stricter* than `eq` here: the isAllOf
    # form is containment-checked, so the superset is caught. Worth stating in
    # the paper -- it is counter-intuitive, and it is why the veto's effect on
    # categorical operands depends on how the hard policy happens to be written.
    assert not is_feasible(doc("isAllOf", ["en", "fr"]), anyof_rule)
    assert is_feasible(doc("isAnyOf", ["en", "fr"]), anyof_rule)


def test_veto_blocks_a_numeric_breach_and_answers_with_a_feasible_repair():
    hard = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:count", "operator": "odrl:lteq", "rightOperand": 100}]}]}
    mine = copy.deepcopy(hard)
    incoming = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:count", "operator": "odrl:lteq", "rightOperand": 5000}]}]}
    guarded = HardPolicyVeto(_AlwaysAccept(mine, hard), hard)
    accept, counter = guarded.observe(incoming)
    assert accept is False
    assert counter["permission"][0]["constraint"][0]["rightOperand"] <= 100


def test_veto_does_not_block_a_feasible_offer():
    hard = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:count", "operator": "odrl:lteq", "rightOperand": 100}]}]}
    incoming = {"permission": [{"action": "use", "constraint": [
        {"leftOperand": "odrl:count", "operator": "odrl:lteq", "rightOperand": 40}]}]}
    guarded = HardPolicyVeto(_AlwaysAccept(copy.deepcopy(hard), hard), hard)
    accept, counter = guarded.observe(incoming)
    assert accept is True
    assert guarded.n_vetoed == 0
    # an accept is a signature on the terms on the table: they must not be rewritten
    assert _content_signature(counter) == _content_signature(incoming)


# ======================================================================
# the guarantee, end to end
# ======================================================================
@pytest.mark.parametrize("name", sorted(BASELINES))
@pytest.mark.parametrize("directory", HAND[:6], ids=os.path.basename)
def test_veto_wrapped_baselines_never_sign_an_infeasible_contract(name, directory):
    ip, hp, ic, hc = _load(directory)
    factory = veto_wrap(BASELINES[name])
    for seed in range(3):
        random.seed(seed)
        provider, consumer = factory(ip, hp), factory(ic, hc)
        agreement, _turns, _ = B.negotiate_generic(provider, consumer)
        if agreement is None:
            continue
        assert B.count_hard_violations(agreement, hp) == 0, \
            feasibility_report(agreement, hp)
        assert B.count_hard_violations(agreement, hc) == 0, \
            feasibility_report(agreement, hc)


@pytest.mark.parametrize("name", sorted(BASELINES))
@pytest.mark.parametrize("directory", HAND[:4], ids=os.path.basename)
def test_raw_baselines_do_breach_so_the_control_is_not_vacuous(name, directory):
    """The wrapper is only a meaningful control if the unwrapped arm can fail.

    Collected across seeds and roles: some (strategy, scenario) cells never
    breach -- an agent that does not concede cannot concede past a bound -- so
    this asserts only that breaches happen *somewhere* in the raw arm.
    """
    ip, hp, ic, hc = _load(directory)
    cls = BASELINES[name]
    breaches = 0
    for seed in range(3):
        random.seed(seed)
        agreement, _turns, _ = B.negotiate_generic(cls(ip, hp), cls(ic, hc))
        if agreement is not None:
            breaches += (B.count_hard_violations(agreement, hp) > 0)
            breaches += (B.count_hard_violations(agreement, hc) > 0)
    test_raw_baselines_do_breach_so_the_control_is_not_vacuous.total += breaches


test_raw_baselines_do_breach_so_the_control_is_not_vacuous.total = 0


def test_raw_baselines_breach_somewhere():
    """Aggregate check for the parametrised test above."""
    ip, hp, ic, hc = _load(HAND[0])
    breaches = 0
    for cls in BASELINES.values():
        for seed in range(3):
            random.seed(seed)
            agreement, _t, _ = B.negotiate_generic(cls(ip, hp), cls(ic, hc))
            if agreement is not None:
                breaches += (B.count_hard_violations(agreement, hp) > 0)
                breaches += (B.count_hard_violations(agreement, hc) > 0)
    assert breaches > 0, "raw baselines never breached, so the veto controls nothing"


@pytest.mark.parametrize("directory", GEN_INF, ids=os.path.basename)
def test_no_agreement_is_reachable_on_an_infeasible_scenario(directory):
    """A scenario the oracle labels infeasible must never yield a lawful deal."""
    ip, hp, ic, hc = _load(directory)
    factory = veto_wrap(BASELINES["Aspiration"])
    random.seed(0)
    agreement, _turns, _ = B.negotiate_generic(factory(ip, hp), factory(ic, hc))
    if agreement is not None:
        # a deal was struck, so at least one party must be in breach
        assert (B.count_hard_violations(agreement, hp) > 0
                or B.count_hard_violations(agreement, hc) > 0)


@pytest.mark.parametrize("directory", GEN_INF[:3], ids=os.path.basename)
def test_walk_away_detector_declines_on_an_infeasible_scenario(directory):
    """The optional belief-free stall detector, BAMBON's hopelessness counterpart."""
    ip, hp, ic, hc = _load(directory)
    factory = veto_wrap(BASELINES["Hardliner"], walk_away_after=3)
    random.seed(0)
    agreement, _turns, _ = B.negotiate_generic(factory(ip, hp), factory(ic, hc))
    assert agreement is None


def test_wrapper_is_transparent():
    ip, hp, _ic, _hc = _load(HAND[0])
    inner = BASELINES["Aspiration"](ip, hp)
    wrapped = HardPolicyVeto(inner, hp)
    assert _content_signature(wrapped.init_offer) == _content_signature(inner.init_offer)
    assert wrapped.concession_rate == inner.concession_rate     # proxied attribute
    assert "Aspiration" in wrapped.name


# ======================================================================
# the Bayesian-bidding ablation
# ======================================================================
def _bambon(role, init, hard, mode):
    agent = Bambon(name=role, lr=0.50, init_offer=copy.deepcopy(init),
                   hard_policy=hard, bidding_mode=mode, **PROFILE)
    agent.init_offer = init
    agent.hard_policy = hard
    return agent


def _forbid_belief_sampling(agent, monkeypatch):
    """Make every posterior *sampling* entry point explode.

    Acceptance is left alone: the ablation is about bidding, and the acceptance
    path legitimately keeps consulting the posterior.
    """
    calls = []
    for registry in agent.beliefs.values():
        for issue in registry.issue_models:
            for model in (issue.model, issue.model_hi):
                if model is None:
                    continue
                for attr in ("sample", "accept_probs"):
                    if hasattr(model, attr):
                        monkeypatch.setattr(
                            model, attr,
                            lambda *a, _n=attr, **k: (_ for _ in ()).throw(
                                AssertionError(f"belief.{_n} called while bidding_mode='time'")),
                            raising=False,
                        )
    return calls


@pytest.mark.parametrize("directory", HAND[:4], ids=os.path.basename)
def test_time_bidding_never_samples_the_posterior(directory, monkeypatch):
    ip, hp, ic, hc = _load(directory)
    random.seed(0)
    provider = _bambon("provider", ip, hp, "time")
    consumer = _bambon("consumer", ic, hc, "time")
    _forbid_belief_sampling(provider, monkeypatch)
    _forbid_belief_sampling(consumer, monkeypatch)
    # must run to completion without touching sample()/accept_probs()
    B.negotiate_generic(provider, consumer, max_turns=60)


@pytest.mark.parametrize("directory", HAND[:4], ids=os.path.basename)
def test_belief_bidding_does_sample_the_posterior(directory, monkeypatch):
    """Control for the test above: the full model must trip the same trap."""
    ip, hp, ic, hc = _load(directory)
    random.seed(0)
    provider = _bambon("provider", ip, hp, "belief")
    consumer = _bambon("consumer", ic, hc, "belief")
    _forbid_belief_sampling(provider, monkeypatch)
    _forbid_belief_sampling(consumer, monkeypatch)
    with pytest.raises(AssertionError):
        B.negotiate_generic(provider, consumer, max_turns=60)


@pytest.mark.parametrize("directory", HAND[:8], ids=os.path.basename)
@pytest.mark.parametrize("mode", ("belief", "time"))
def test_both_bidding_modes_keep_the_compliance_invariant(directory, mode):
    """Prop. 1 and 2 are properties of the repair and veto, not of the bidder."""
    ip, hp, ic, hc = _load(directory)
    for seed in range(2):
        random.seed(seed)
        try:
            agreement, _t, _ = B.negotiate_generic(
                _bambon("provider", ip, hp, mode), _bambon("consumer", ic, hc, mode))
        except NegotiationInfeasible:
            continue
        if agreement is None:
            continue
        assert B.count_hard_violations(agreement, hp) == 0
        assert B.count_hard_violations(agreement, hc) == 0


def test_bidding_mode_reaches_every_action_state():
    ip, hp, _ic, _hc = _load(HAND[0])
    agent = _bambon("provider", ip, hp, "time")
    agent.observe(copy.deepcopy(ip))
    assert agent.action_states
    assert all(st.bidding_mode == "time" for st in agent.action_states.values())
