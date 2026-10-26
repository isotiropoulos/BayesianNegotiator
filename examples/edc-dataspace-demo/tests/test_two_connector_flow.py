"""The full flow - catalog, counter-offers, agreement, pull - with fake connectors.

This is the test that pins the *orchestration*: what gets fetched, what gets
requested, what gets republished, what gets retired, and what the transfer
quotes. ``tests/fake_connectors.py`` supplies a connector pair that enforces the
two rules real EDC enforces (fresh offer id per fetch, byte-equal policy or
TERMINATED), so passing here means the sequence is right.

It is not a substitute for ``test_e2e_two_connectors.py``. Nothing here proves
EDC accepts the JSON-LD or that a data plane exists. It proves that when a
connector behaves as documented, this code drives it correctly - which is the
part that used to break silently.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("numpy", reason="BAMBON needs numpy")

from bambon_edc.connector_negotiation import ConnectorNegotiation  # noqa: E402
from bambon_edc.odrl_edc import (VANILLA, access_policy_definition,  # noqa: E402
                                 contract_definition, policy_definition,
                                 strip_prefix)
from bambon_edc.profiles import (ASSET_ID, CONSUMER_HARD_POLICY,  # noqa: E402
                                 PROVIDER_HARD_POLICY, PROVIDER_INIT_OFFER,
                                 PUBLISHED_ACCESS_POLICY_ID,
                                 PUBLISHED_DEFINITION_ID, PUBLISHED_POLICY_ID)
from fake_connectors import (FakeConsumerClient, FakeDataSpace,  # noqa: E402
                             FakeProviderClient, StubPuller, canonical_terms)

MAX_ROUNDS = 14


# --------------------------------------------------------------------------- #
# setup                                                                        #
# --------------------------------------------------------------------------- #

async def _seed(space: FakeDataSpace) -> None:
    """The same three POSTs ``seed_provider.py`` makes against a real connector."""
    provider = FakeProviderClient(space, "http://provider/management")
    await provider.ensure("policy", access_policy_definition(
        policy_id=PUBLISHED_ACCESS_POLICY_ID, dialect=space.dialect))
    await provider.ensure("policy", policy_definition(
        dict(PROVIDER_INIT_OFFER, **{"@type": "odrl:Agreement"}),
        policy_id=PUBLISHED_POLICY_ID, dialect=space.dialect))
    await provider.ensure("contract", contract_definition(
        asset_id=space.asset_id,
        access_policy_id=PUBLISHED_ACCESS_POLICY_ID,
        contract_policy_id=PUBLISHED_POLICY_ID,
        definition_id=PUBLISHED_DEFINITION_ID))


def _run(*, provider_keeps_rejected: bool = True, pull_fails: bool = False):
    space = FakeDataSpace(ASSET_ID,
                          published_definition_id=PUBLISHED_DEFINITION_ID,
                          published_policy_id=PUBLISHED_POLICY_ID,
                          dialect=VANILLA,
                          provider_keeps_rejected=provider_keeps_rejected)
    puller = StubPuller(space, fail=pull_fails)

    async def go():
        await _seed(space)
        negotiation = ConnectorNegotiation(
            consumer_management=FakeConsumerClient(space, "http://consumer/management"),
            provider_management=FakeProviderClient(space, "http://provider/management"),
            provider_protocol_url="http://provider-connector:19194/protocol",
            provider_participant_id="example-provider",
            consumer_participant_id="example-consumer",
            asset_id=ASSET_ID,
            dialect=VANILLA,
            max_rounds=MAX_ROUNDS,
            puller=puller,
            on_event=lambda msg: print(msg),
        )
        return await negotiation.run()

    return space, puller, asyncio.run(go())


@pytest.fixture(scope="module")
def run_result():
    space, puller, outcome = _run()
    print(f"\n=== {outcome.kind} after {len(outcome.rounds)} rounds ===")
    return space, puller, outcome


def _clauses(offer, rule_type="permission", action="use"):
    for rule in offer.get(rule_type) or []:
        if strip_prefix(rule.get("action")) == action:
            return {strip_prefix(c["leftOperand"]): c
                    for c in rule.get("constraint", []) or []}
    return {}


# --------------------------------------------------------------------------- #
# the sequence the user asked for, step by step                                #
# --------------------------------------------------------------------------- #

def test_reaches_a_contract_agreement(run_result):
    _space, _puller, outcome = run_result
    assert outcome.kind == "agreement", outcome.reason
    assert outcome.contract_agreement_id


def test_every_round_fetched_the_catalog_and_sent_a_contract_request(run_result):
    """Steps 1 and 3: nothing is exchanged except through the connectors."""
    space, _puller, outcome = run_result
    assert space.catalog_fetches == len(outcome.rounds)
    assert outcome.dsp_exchanges == len(outcome.rounds)
    for rnd in outcome.rounds:
        assert rnd.published_offer is not None
        assert rnd.negotiation_id


def test_the_catalog_always_carried_exactly_one_offer(run_result):
    """Step 2. The regression this guards: the provider republishing without
    retiring its previous contract definition, so the consumer picks one of
    several offers at random and the transcript becomes meaningless."""
    _space, _puller, outcome = run_result
    counts = [rnd.offers_in_catalog for rnd in outcome.rounds]
    assert set(counts) == {1}, f"offers per fetch: {counts}"


def test_the_seeded_restrictive_offer_was_retired(run_result):
    space, _puller, _outcome = run_result
    assert PUBLISHED_DEFINITION_ID in space.deleted_definitions
    assert PUBLISHED_DEFINITION_ID not in space.definitions


def test_the_consumer_countered_rather_than_accepting(run_result):
    """Step 3. The first request must NOT be the published offer, or there was no
    negotiation - just acceptance."""
    space, _puller, outcome = run_result
    first_published = outcome.rounds[0].published_offer
    assert canonical_terms(space.requested_policies[0]) \
        != canonical_terms(first_published), \
        "the consumer's opening request was the published offer verbatim"


def test_the_provider_countered_by_republishing(run_result):
    """Step 4: EDC has no ContractOfferMessage, so a republished catalog entry is
    the counter-offer. Every non-final round must contain one."""
    _space, _puller, outcome = run_result
    republished = [r.index for r in outcome.rounds if r.republished_policy_id]
    assert len(republished) >= 1
    assert all(r.republished_policy_id for r in outcome.rounds[:-1]), \
        f"rounds without a provider counter: {republished}"


def test_the_provider_read_the_proposal_from_its_own_connector(run_result):
    """The strong path: no out-of-band relay when the connector persists the
    rejected request."""
    _space, _puller, outcome = run_result
    assert all(not r.relayed for r in outcome.rounds), \
        "relayed despite the connector keeping the rejected policy"
    assert any(r.provider_saw_request for r in outcome.rounds)


def test_both_sides_moved(run_result):
    """Step 5: beliefs updating on both sides, not one side conceding."""
    _space, _puller, outcome = run_result
    consumer_terms = {canonical_terms(r.consumer_proposal)
                      for r in outcome.rounds if r.consumer_proposal}
    provider_terms = {canonical_terms(r.provider_counter)
                      for r in outcome.rounds if r.provider_counter}
    assert len(consumer_terms) > 1, "the consumer repeated itself every round"
    assert len(provider_terms) > 1, "the provider never moved"


def test_the_final_request_quoted_the_published_offer_verbatim(run_result):
    """Step 6. EDC only finalises a byte-equal request, so the acceptance round
    has to quote the catalog rather than BAMBON's own agreement object."""
    space, _puller, outcome = run_result
    final_published = outcome.rounds[-1].published_offer
    assert canonical_terms(space.requested_policies[-1]) \
        == canonical_terms(final_published)


def test_terms_moved_away_from_the_opening_offer(run_result):
    _space, _puller, outcome = run_result
    first = _clauses(outcome.rounds[0].published_offer)
    final = _clauses(outcome.agreement)
    moved = {k for k in final if k in first
             and str(final[k]["rightOperand"]) != str(first[k]["rightOperand"])}
    assert moved, f"terms never moved: {final}"


def test_both_hard_policies_survived(run_result):
    _space, _puller, outcome = run_result
    final = _clauses(outcome.agreement)
    floor = _clauses(CONSUMER_HARD_POLICY)
    ceiling = _clauses(PROVIDER_HARD_POLICY)

    elapsed = float(final["elapsedTime"]["rightOperand"])
    assert elapsed >= float(floor["elapsedTime"]["rightOperand"])
    assert elapsed <= float(ceiling["elapsedTime"]["rightOperand"])
    assert set(final["spatial"]["rightOperand"]) <= \
        set(ceiling["spatial"]["rightOperand"])


# --------------------------------------------------------------------------- #
# step 7: the pull                                                             #
# --------------------------------------------------------------------------- #

def test_the_consumer_pulled_the_dataset(run_result):
    _space, _puller, outcome = run_result
    assert outcome.pull is not None, "no transfer was attempted"
    assert outcome.data_moved, getattr(outcome.pull, "reason", outcome.pull)
    assert outcome.pull.record_count == 1
    assert outcome.pull.bytes_received > 0


def test_the_pull_quoted_the_negotiated_agreement(run_result):
    """The claim that matters: the data moved under the agreement BAMBON reached,
    not under a second one obtained by accepting the published offer."""
    _space, puller, outcome = run_result
    assert len(puller.calls) == 1
    assert puller.calls[0]["contract_agreement_id"] == outcome.contract_agreement_id
    assert puller.calls[0]["asset_id"] == ASSET_ID


def test_no_pull_is_attempted_without_an_agreement():
    """A transfer before agreement would be the worst possible bug: data moving
    on unagreed terms."""
    space = FakeDataSpace(ASSET_ID,
                          published_definition_id=PUBLISHED_DEFINITION_ID,
                          published_policy_id=PUBLISHED_POLICY_ID)
    puller = StubPuller(space)

    async def go():
        await _seed(space)
        negotiation = ConnectorNegotiation(
            consumer_management=FakeConsumerClient(space, "http://consumer/management"),
            provider_management=FakeProviderClient(space, "http://provider/management"),
            provider_protocol_url="http://provider/protocol",
            provider_participant_id="example-provider",
            consumer_participant_id="example-consumer",
            asset_id=ASSET_ID,
            max_rounds=2,          # far too few to converge
            puller=puller,
            on_event=lambda msg: None,
        )
        return await negotiation.run()

    outcome = asyncio.run(go())
    assert outcome.kind == "timeout"
    assert outcome.pull is None
    assert puller.calls == []


def test_a_failed_pull_does_not_invalidate_the_agreement():
    """Separating the two claims. The negotiation succeeded; only the end-to-end
    story is weaker, and ``data_moved`` is what says so."""
    _space, _puller, outcome = _run(pull_fails=True)
    assert outcome.kind == "agreement"
    assert outcome.contract_agreement_id
    assert outcome.pull is not None
    assert not outcome.data_moved
    assert outcome.pull.kind == "no_edr"


# --------------------------------------------------------------------------- #
# the weaker mode, disclosed rather than hidden                                 #
# --------------------------------------------------------------------------- #

def test_relay_fallback_when_the_connector_forgets_the_rejected_policy():
    """Some EDC versions do not persist the policy of a request they terminated.
    The run must still work *and* must mark every affected round, so a write-up
    cannot accidentally claim the provider observed proposals purely via EDC."""
    _space, _puller, outcome = _run(provider_keeps_rejected=False)

    assert outcome.kind == "agreement", outcome.reason
    relayed = [r.index for r in outcome.rounds if r.relayed]
    assert relayed, "expected the relay fallback to engage"
    assert all("relayed out of band" in r.note
               for r in outcome.rounds if r.relayed)
    # Still connector-to-connector: every proposal rode a real contract request.
    assert outcome.dsp_exchanges == len(outcome.rounds)
