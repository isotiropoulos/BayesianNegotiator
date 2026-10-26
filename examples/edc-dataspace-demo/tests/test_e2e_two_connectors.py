"""BAMBON negotiating between two live EDC connectors, then pulling the data.

Every round of this negotiation is real Dataspace Protocol traffic between the
two connectors:

* a **catalog request** from the consumer connector to the provider connector,
  carrying the provider's current offer back; and
* a **ContractRequestMessage** from the consumer connector to the provider
  connector, carrying the consumer's counter-offer.

Neither BAMBON talks to the other. Each one only reads and writes its own
connector's Management API. The provider's counter-offers are made by
republishing its catalog entry - the only counter-offer mechanism EDC gives you.

The run ends when a real EDC contract negotiation reaches ``FINALIZED`` and the
consumer pulls the dataset **quoting that agreement id**. Two artifacts are worth
citing: the contract agreement id, and the fact that data moved under it. The
second is the stronger claim, because agreeing terms and transferring bytes are
different things and only one of them is usually demonstrated.

Running it::

    docker compose up -d --build --wait
    docker compose run --rm --no-deps \
        -e EDC_CONNECTOR_HOST=provider-connector \
        -e EDC_CONNECTOR_MANAGEMENT_PORT=19193 \
        provider-negotiator python -m bambon_edc.seed_provider

    pytest tests/test_e2e_two_connectors.py -v -s

Skips (never fails) when the stack is unreachable, so the offline suite still
runs without Docker. The transfer needs ``edcpy`` and a reachable RabbitMQ, and
is skipped - not failed - when either is missing; ``tests/test_two_connector_flow.py``
covers the same sequence with fake connectors and a stub transfer.

Re-running this test without re-seeding the provider starts the negotiation from
wherever the last run left the catalog, because the provider retires its previous
contract definition every time it counter-offers. Re-seed for a clean scenario.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

httpx = pytest.importorskip("httpx", reason="talks to the connectors over HTTP")

from bambon_edc.connector_negotiation import (ConnectorNegotiation,  # noqa: E402
                                              ConnectorOutcome)
from bambon_edc.management import ManagementClient  # noqa: E402
from bambon_edc.odrl_edc import dialect_by_name, strip_prefix  # noqa: E402
from bambon_edc.profiles import (ASSET_ID, CONSUMER_HARD_POLICY,  # noqa: E402
                                 PROVIDER_HARD_POLICY)

# --------------------------------------------------------------------------- #
# where the stack is (defaults match docker-compose.yml, seen from the host)   #
# --------------------------------------------------------------------------- #

CONSUMER_MANAGEMENT_URL = os.getenv(
    "CONSUMER_MANAGEMENT_URL", "http://localhost:29193/management")
PROVIDER_MANAGEMENT_URL = os.getenv(
    "PROVIDER_MANAGEMENT_URL", "http://localhost:19193/management")

#: Dialled by the *consumer connector* from inside the Docker network, so this
#: is a service name rather than localhost.
PROVIDER_PROTOCOL_URL = os.getenv(
    "PROVIDER_PROTOCOL_URL", "http://provider-connector:19194/protocol")

PROVIDER_PARTICIPANT_ID = os.getenv("PROVIDER_PARTICIPANT_ID", "example-provider")
CONSUMER_PARTICIPANT_ID = os.getenv("CONSUMER_PARTICIPANT_ID", "example-consumer")

API_AUTH_KEY = os.getenv("API_AUTH_KEY", "demo-api-key")
DIALECT = dialect_by_name(os.getenv("BAMBON_EDC_DIALECT", "vanilla"))
MAX_ROUNDS = int(os.getenv("E2E_MAX_ROUNDS", "12"))
TRANSFER_TIMEOUT = int(os.getenv("E2E_TRANSFER_TIMEOUT", "120"))

#: Set to "0" to stop at the contract agreement. Off by default because the
#: transfer needs the broker and the consumer backend, which only exist inside
#: the compose network - so from the host you usually want it disabled.
DO_TRANSFER = os.getenv("E2E_TRANSFER", "1") not in ("0", "false", "no")


def _client(url: str) -> ManagementClient:
    return ManagementClient(base_url=url, api_key=API_AUTH_KEY)


def _reachable(url: str) -> bool:
    try:
        httpx.get(url, timeout=3.0)
        return True
    except Exception:
        return False


def _require_stack() -> None:
    missing = [f"{label} ({url})" for label, url in (
        ("consumer connector", f"{CONSUMER_MANAGEMENT_URL}/v3/assets"),
        ("provider connector", f"{PROVIDER_MANAGEMENT_URL}/v3/assets"),
    ) if not _reachable(url)]
    if missing:
        pytest.skip("connectors not reachable: " + "; ".join(missing)
                    + ". Run `docker compose up -d --wait` first.")


def _clauses(offer: Dict[str, Any], rule_type: str = "permission",
             action: str = "use") -> Dict[str, Dict[str, Any]]:
    for rule in offer.get(rule_type) or []:
        if strip_prefix(rule.get("action")) == action:
            return {strip_prefix(c["leftOperand"]): c
                    for c in rule.get("constraint", []) or []}
    return {}


# --------------------------------------------------------------------------- #
# the run, executed once and shared                                            #
# --------------------------------------------------------------------------- #

def _puller():
    """The real transfer, or ``None`` when this environment cannot run one."""
    if not DO_TRANSFER:
        return None
    try:
        from bambon_edc.transfer import make_puller
        return make_puller(provider_protocol_url=PROVIDER_PROTOCOL_URL,
                           provider_participant_id=PROVIDER_PARTICIPANT_ID,
                           timeout=TRANSFER_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - TransferUnavailable or config error
        print(f"\n  transfer disabled: {exc}")
        return None


@pytest.fixture(scope="module")
def outcome() -> ConnectorOutcome:
    _require_stack()

    negotiation = ConnectorNegotiation(
        consumer_management=_client(CONSUMER_MANAGEMENT_URL),
        provider_management=_client(PROVIDER_MANAGEMENT_URL),
        provider_protocol_url=PROVIDER_PROTOCOL_URL,
        provider_participant_id=PROVIDER_PARTICIPANT_ID,
        consumer_participant_id=CONSUMER_PARTICIPANT_ID,
        asset_id=ASSET_ID,
        dialect=DIALECT,
        max_rounds=MAX_ROUNDS,
        puller=_puller(),
        on_event=print,
    )

    print("\n=== BAMBON negotiation between two EDC connectors ===")
    result = asyncio.run(negotiation.run())

    print(f"\n=== {result.kind} after {len(result.rounds)} rounds, "
          f"{result.dsp_exchanges} DSP contract requests ===")
    if result.contract_agreement_id:
        print(f"  contract agreement: {result.contract_agreement_id}")
        print(f"  final negotiation:  {result.final_negotiation_id}")
    if result.reason:
        print(f"  reason: {result.reason}")
    if result.pull is not None:
        from bambon_edc.transfer import describe
        print(f"  transfer:           {describe(result.pull)}")
    for rnd in result.rounds:
        if rnd.note:
            print(f"  round {rnd.index}: {rnd.note}")
    return result


# --------------------------------------------------------------------------- #
# what a successful run must show                                              #
# --------------------------------------------------------------------------- #

def test_the_negotiation_reached_a_contract_agreement(outcome):
    assert outcome.kind == "agreement", outcome.reason
    assert outcome.contract_agreement_id, \
        "FINALIZED without a contract agreement id"


def test_every_round_crossed_the_dataspace_protocol(outcome):
    """The point of this test: each round is connector-to-connector traffic, not
    two agents talking to each other."""
    assert outcome.dsp_exchanges == len(outcome.rounds), \
        "a round completed without issuing a real contract request"
    for rnd in outcome.rounds:
        assert rnd.published_offer is not None, \
            f"round {rnd.index} did not fetch the catalog over DSP"
        assert rnd.negotiation_id, \
            f"round {rnd.index} produced no EDC negotiation record"


def test_the_provider_actually_counter_offered(outcome):
    """Rounds before the last must show the provider republishing its catalog
    entry - that is how a counter-offer is expressed without ContractOfferMessage."""
    intermediate = outcome.rounds[:-1]
    if not intermediate:
        pytest.skip("settled in one round; nothing to counter")
    assert any(r.republished_policy_id for r in intermediate), \
        "the provider never republished; it did not participate in bargaining"


def test_more_than_one_round_was_needed(outcome):
    """A one-round agreement is acceptance, not negotiation."""
    assert len(outcome.rounds) > 1, \
        "settled immediately - the scenario is not adversarial enough to be evidence"


def test_terms_moved_away_from_the_published_offer(outcome):
    """The final terms must differ from what the provider first published."""
    first = _clauses(outcome.rounds[0].published_offer)
    final = _clauses(outcome.agreement)
    moved = {k for k in final if k in first
             and str(final[k]["rightOperand"]) != str(first[k]["rightOperand"])}
    assert moved, f"terms never moved: {final}"
    print(f"\n  operands that moved: {sorted(moved)}")


def test_final_terms_satisfy_both_hard_policies(outcome):
    final = _clauses(outcome.agreement)
    floor = _clauses(CONSUMER_HARD_POLICY)
    ceiling = _clauses(PROVIDER_HARD_POLICY)

    elapsed = float(final["elapsedTime"]["rightOperand"])
    assert elapsed >= float(floor["elapsedTime"]["rightOperand"]), \
        "consumer signed below its own floor"
    assert elapsed <= float(ceiling["elapsedTime"]["rightOperand"]), \
        "provider signed above its own ceiling"
    assert set(final["spatial"]["rightOperand"]) <= \
        set(ceiling["spatial"]["rightOperand"])


def test_the_catalog_offered_one_policy_per_round(outcome):
    """The provider must retire its previous contract definition when it
    counter-offers. If it does not, ``odrl:hasPolicy`` accumulates stale terms
    and "the provider's current offer" stops being well defined - the consumer
    would then be bargaining against whichever policy EDC serialised first."""
    counts = [rnd.offers_in_catalog for rnd in outcome.rounds]
    assert set(counts) <= {1}, f"offers per catalog fetch: {counts}"


# --------------------------------------------------------------------------- #
# the data actually moved                                                      #
# --------------------------------------------------------------------------- #

def test_the_consumer_pulled_the_dataset(outcome):
    """The claim this whole demo exists to support.

    Skips rather than fails when no transfer was attempted: from the host,
    RabbitMQ and the consumer backend are not reachable under their compose
    hostnames, and a red test there would be noise rather than information.
    """
    if outcome.pull is None:
        pytest.skip("no transfer attempted (E2E_TRANSFER=0, or edcpy missing)")

    assert outcome.data_moved, (
        f"terms were agreed but no data arrived: "
        f"{getattr(outcome.pull, 'reason', outcome.pull)}")
    assert outcome.pull.bytes_received > 0
    print(f"\n  pulled {outcome.pull.bytes_received} bytes"
          + (f", {outcome.pull.record_count} records"
             if outcome.pull.record_count is not None else ""))
    print(f"  transfer process: {outcome.pull.transfer_process_id}")


def test_the_transfer_quoted_the_negotiated_agreement(outcome):
    """Data moving is only interesting if it moved under *these* terms.

    A demo that pulled data after separately accepting the published offer would
    show nothing about the negotiation, so the transfer must name the agreement
    id that BAMBON's own contract request produced.
    """
    if outcome.pull is None:
        pytest.skip("no transfer attempted")
    assert outcome.pull.contract_agreement_id == outcome.contract_agreement_id


def test_report_whether_the_provider_read_its_own_connector(outcome):
    """Not an assertion about correctness - a disclosure.

    The provider is supposed to learn the consumer's proposal by reading the
    rejected request from its own connector. Some EDC versions do not persist
    it, in which case the harness relays the proposal instead. That still
    negotiates through the connectors, but it is weaker, so the run must say so
    out loud rather than let a paper overclaim.
    """
    relayed = [r.index for r in outcome.rounds if r.relayed]
    read = [r.index for r in outcome.rounds if r.provider_saw_request]
    print(f"\n  provider read the request from its own connector in rounds: {read}")
    print(f"  proposal was relayed out of band in rounds: {relayed}")
    if relayed:
        print("  -> describe this as 'BAMBON drove EDC contract negotiations', "
              "NOT 'the provider observed proposals purely via EDC'")
    assert read or relayed, "no round produced a proposal for the provider"
