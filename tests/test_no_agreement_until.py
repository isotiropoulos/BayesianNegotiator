"""``no_agreement_until`` must actually delay agreement.

The bug
-------
The knob gated only the ``propose()`` call inside ``respond()``::

    if self.rounds_seen >= self.no_agreement_until:
        predictions[action_str] = belief.propose(action)
    else:
        predictions[action_str] = belief.propose(action, False)

but ``predictions`` is discarded whenever ``all_accepted`` is true, and
``all_accepted`` did not consult the round counter. So "no agreement until
round N" delayed nothing, for any N. Nobody noticed because on a scenario with
real constraints the beliefs rarely converge on round 1 anyway.

The degenerate case is what exposed it. ``all(... for ... in [])`` is vacuously
true, so an offer whose rules carry **no constraints** — nothing to negotiate —
agreed immediately no matter how the negotiator was configured. That shape is
not hypothetical: it is exactly what the Eclipse DSP TCK puts on the wire
(``permission: [{action: use, constraint: []}]``), and it made a
counter-offering profile impossible to express.

Run with::

    python -m pytest tests/test_no_agreement_until.py -q
"""
from __future__ import annotations

import copy
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.bambon.negotiator import Bambon  # noqa: E402


#: A permission with no constraints — the degenerate offer. Both sides have
#: nothing to negotiate over, so acceptance is governed purely by the round
#: gate.
UNCONSTRAINED = {"permission": [{"action": "use", "constraint": []}]}

#: One scalar constraint, so the beliefs have something to converge on. Used to
#: show the gate holds on ordinary offers too, not just the degenerate one.
CONSTRAINED = {
    "permission": [{
        "action": "use",
        "constraint": [
            {"leftOperand": "count", "operator": "lteq", "rightOperand": 10},
        ],
    }],
}


def _negotiator(offer, **kwargs):
    return Bambon(
        name="test",
        init_offer=copy.deepcopy(offer),
        hopeless_window=0,   # never decline; this test is about acceptance
        max_rounds=500,
        **kwargs,
    )


def test_unconstrained_offer_accepts_immediately_at_zero():
    """The gate open from the start still lets the degenerate offer through."""
    agent = _negotiator(UNCONSTRAINED, no_agreement_until=0)
    accepted, _ = agent.observe(copy.deepcopy(UNCONSTRAINED))
    assert accepted, "with no gate, a constraint-free offer should be agreed at once"


def test_unconstrained_offer_is_not_agreed_before_the_gate_opens():
    """The regression. Previously true on round 1 for every configuration."""
    agent = _negotiator(UNCONSTRAINED, no_agreement_until=3)
    for round_number in (1, 2):
        accepted, response = agent.observe(copy.deepcopy(UNCONSTRAINED))
        assert not accepted, (
            f"agreed on round {round_number} despite no_agreement_until=3"
        )
        assert response is not None, "a refusal must still carry a counter-offer"


def test_unconstrained_offer_is_agreed_once_the_gate_opens():
    agent = _negotiator(UNCONSTRAINED, no_agreement_until=3)
    accepted = False
    for _ in range(3):
        accepted, _ = agent.observe(copy.deepcopy(UNCONSTRAINED))
    assert accepted, "should agree on the round the gate opens"


def test_a_high_gate_never_agrees():
    """How a counter-offer-forever profile is expressed."""
    agent = _negotiator(UNCONSTRAINED, no_agreement_until=999)
    for _ in range(12):
        accepted, response = agent.observe(copy.deepcopy(UNCONSTRAINED))
        assert not accepted
        assert response is not None


@pytest.mark.parametrize("gate", [2, 5])
def test_gate_holds_on_a_constrained_offer_too(gate):
    """Not a quirk of the empty-constraint case.

    ``rounds_seen`` is incremented at the top of ``respond``, so the gate is
    already open on round ``gate`` itself — only rounds ``1 .. gate - 1`` are
    blocked.
    """
    agent = _negotiator(CONSTRAINED, no_agreement_until=gate)
    for round_number in range(1, gate):
        accepted, _ = agent.observe(copy.deepcopy(CONSTRAINED))
        assert not accepted, f"agreed on round {round_number}, gate is {gate}"


def test_init_offer_is_required():
    """The other half of ``bool(self.beliefs)``: there is no belief-free agent.

    Worth pinning, because the guard in ``all_accepted`` reads as though an
    empty registry were reachable at runtime. It is not — the constructor
    refuses first.
    """
    with pytest.raises(ValueError, match="init_offer is required"):
        Bambon(name="empty", init_offer=None, no_agreement_until=0)
