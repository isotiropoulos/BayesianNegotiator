"""End-to-end tests for the part that does not need a running data space:
the bilateral BAMBON loop plus materialisation of its agreement.

Requires ``bambon`` importable (numpy + scipy). Skipped otherwise.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("numpy", reason="BAMBON needs numpy")

from bambon_edc.odrl_edc import operands_used, policy_definition, strip_prefix  # noqa: E402
from bambon_edc.profiles import (ASSET_ID, CONSUMER_HARD_POLICY,  # noqa: E402
                                 PROVIDER_HARD_POLICY)
from bambon_edc.session import (LocalParty, NegotiationSession,  # noqa: E402
                                drive, negotiate_offline)


def _clauses(offer, rule_type="permission", action="use"):
    for rule in offer.get(rule_type) or []:
        if strip_prefix(rule["action"]) == action:
            return {strip_prefix(c["leftOperand"]): c
                    for c in rule.get("constraint", []) or []}
    return {}


@pytest.fixture(scope="module")
def outcome():
    return negotiate_offline(ASSET_ID)


# --------------------------------------------------------------------------- #
# the loop converges                                                           #
# --------------------------------------------------------------------------- #

def test_scenario_starts_out_of_agreement():
    """Sanity: the demo is only interesting because the published offer and the
    consumer's need genuinely conflict."""
    consumer = NegotiationSession(role="consumer", asset_id=ASSET_ID)
    provider = NegotiationSession(role="provider", asset_id=ASSET_ID)
    assert (_clauses(consumer.init_offer)["purpose"]["rightOperand"]
            != _clauses(provider.init_offer)["purpose"]["rightOperand"])


def test_bilateral_negotiation_reaches_agreement(outcome):
    assert outcome.kind == "agreement", outcome.reason
    assert outcome.agreement["@type"] == "odrl:Agreement"
    assert outcome.rounds > 0


def test_agreement_respects_provider_hard_policy(outcome):
    """<= 180 days, EU/EEA only. BAMBON's core guarantee is that it never signs
    outside its own hard policy; this pins it for the demo scenario."""
    agreed = _clauses(outcome.agreement)
    hard = _clauses(PROVIDER_HARD_POLICY)

    assert float(agreed["elapsedTime"]["rightOperand"]) <= \
        float(hard["elapsedTime"]["rightOperand"])
    assert set(agreed["spatial"]["rightOperand"]) <= \
        set(hard["spatial"]["rightOperand"])


def test_agreement_respects_consumer_hard_policy(outcome):
    """>= 60 days, otherwise the data is useless to the consumer."""
    agreed = _clauses(outcome.agreement)
    hard = _clauses(CONSUMER_HARD_POLICY)
    assert float(agreed["elapsedTime"]["rightOperand"]) >= \
        float(hard["elapsedTime"]["rightOperand"])


def test_agreement_beats_the_published_offer_for_the_consumer(outcome):
    """The whole point: the negotiated terms are strictly better for the
    consumer than the 30-day research-only offer in the catalog."""
    agreed = _clauses(outcome.agreement)
    assert agreed["purpose"]["rightOperand"] == "commercial"
    assert float(agreed["elapsedTime"]["rightOperand"]) > 30


def test_prohibitions_survive_the_negotiation(outcome):
    """Both sides agreed on the no-public-redistribution prohibition from the
    start; it must appear in the agreement rather than being dropped."""
    prohibition = _clauses(outcome.agreement, "prohibition", "distribute")
    assert "recipient" in prohibition


# --------------------------------------------------------------------------- #
# the agreement is publishable                                                 #
# --------------------------------------------------------------------------- #

def test_agreement_materialises_into_a_valid_policy_definition(outcome):
    body = policy_definition(outcome.agreement,
                             policy_id="p-test",
                             participant_ids=["consumer"])
    assert body["policy"]["permission"]
    assert body["policy"]["prohibition"]

    clauses = body["policy"]["permission"][0]["constraint"][0]["and"]
    operands = {c["leftOperand"] for c in clauses}
    assert "participantId" in operands
    assert {"purpose", "elapsedTime", "spatial"} <= operands


def test_agreement_survives_the_tractusx_baseline(outcome):
    """Retargeting to Tractus-X must not drop negotiated terms: the mandatory
    Catena-X floor is added alongside them, not instead of them."""
    from bambon_edc.odrl_edc import TRACTUSX

    body = policy_definition(outcome.agreement, policy_id="p-test",
                             dialect=TRACTUSX)
    operands = {c["leftOperand"]
                for c in body["policy"]["permission"][0]["constraint"]["and"]}
    assert {"Membership", "FrameworkAgreement", "UsagePurpose"} <= operands
    assert {"purpose", "elapsedTime", "spatial"} <= operands


def test_every_agreed_operand_is_reported_for_policy_registration(outcome):
    """Anything listed here needs a PolicyEngine function in the connector."""
    assert {"purpose", "elapsedTime", "spatial", "recipient"} == \
        set(operands_used(outcome.agreement))


# --------------------------------------------------------------------------- #
# sessions are isolated                                                        #
# --------------------------------------------------------------------------- #

def test_sessions_do_not_share_state():
    """The service keeps many sessions in one process; ``Bambon`` mutates its
    ``init_offer``, so a shallow copy here would cross-contaminate negotiations
    between different consumers."""
    first = NegotiationSession(role="provider", asset_id=ASSET_ID)
    second = NegotiationSession(role="provider", asset_id=ASSET_ID)

    assert first.init_offer is not second.init_offer
    assert first.session_id != second.session_id

    consumer = NegotiationSession(role="consumer", asset_id=ASSET_ID)
    drive(consumer, LocalParty(first), max_rounds=60)

    fresh = NegotiationSession(role="provider", asset_id=ASSET_ID)
    assert fresh.init_offer == second.init_offer
    assert fresh.agent.rounds_seen == 0


def test_terminated_session_is_idempotent():
    consumer = NegotiationSession(role="consumer", asset_id=ASSET_ID)
    provider = NegotiationSession(role="provider", asset_id=ASSET_ID)
    result = drive(consumer, LocalParty(provider), max_rounds=60)

    if result.kind != "agreement":
        pytest.skip("scenario did not converge; nothing to re-step")

    again = provider.step(consumer.init_offer)
    assert again.kind == "agreement"
    assert again.offer == provider.agreement
