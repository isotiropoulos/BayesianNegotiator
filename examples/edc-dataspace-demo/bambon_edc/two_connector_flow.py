"""Two local data space connectors, negotiating, then moving data.

The whole story, driven from the consumer's side::

    1  consumer connector asks the provider connector for its catalog (DSP)
    2  exactly one offer comes back for the asset, and it is unacceptable
    3  consumer BAMBON counters; the counter travels as a real
       ContractRequestMessage, which the provider terminates
    4  provider BAMBON reads that rejected request from its own connector,
       counters, and republishes its catalog entry - that IS its counter-offer
    5  ... 3 and 4 repeat, both agents updating their beliefs each round ...
    6  consumer BAMBON accepts, requests the published terms verbatim, and the
       EDC negotiation reaches FINALIZED with a real contract agreement id
    7  consumer runs an HttpData-PULL transfer against *that* agreement and
       reads the dataset

Nothing here is a private side channel: every proposal in steps 3-4 crosses the
Dataspace Protocol between the two connectors. ``service.py`` is the other,
weaker arrangement (the two BAMBONs talking over HTTP); this module is the one to
cite.

Run it::

    ./run_demo.sh                              # everything, from scratch
    docker compose run --rm runner             # just this flow
    python -m bambon_edc.two_connector_flow    # from a shell with the env set

Exit codes: 0 agreement and data pulled, 2 agreement but no data, 3 no
agreement, 4 the stack is not reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, Optional

from .connector_negotiation import ConnectorNegotiation, ConnectorOutcome
from .management import ManagementClient
from .odrl_edc import dialect_by_name, operands_used, strip_prefix
from .profiles import (ASSET_ID, CONSUMER_HARD_POLICY, PROVIDER_HARD_POLICY,
                       PUBLISHED_DEFINITION_ID)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
_logger = logging.getLogger("bambon_edc.flow")

# --------------------------------------------------------------------------- #
# where the two connectors are                                                 #
# --------------------------------------------------------------------------- #
#
# Defaults are the compose service names, i.e. correct from inside the Docker
# network. Override with localhost ports to drive the stack from the host.

CONSUMER_MANAGEMENT_URL = os.getenv(
    "CONSUMER_MANAGEMENT_URL", "http://consumer-connector:29193/management")
PROVIDER_MANAGEMENT_URL = os.getenv(
    "PROVIDER_MANAGEMENT_URL", "http://provider-connector:19193/management")

#: Dialled by the *consumer connector*, not by this process, so it must be
#: resolvable inside the data space rather than from wherever this runs.
PROVIDER_PROTOCOL_URL = os.getenv(
    "PROVIDER_PROTOCOL_URL", "http://provider-connector:19194/protocol")

PROVIDER_PARTICIPANT_ID = os.getenv("PROVIDER_PARTICIPANT_ID",
                                    os.getenv("PROVIDER_CONNECTOR_ID",
                                              "example-provider"))
CONSUMER_PARTICIPANT_ID = os.getenv("CONSUMER_PARTICIPANT_ID",
                                    os.getenv("EDC_CONNECTOR_PARTICIPANT_ID",
                                              "example-consumer"))

API_AUTH_KEY = os.getenv("API_AUTH_KEY", os.getenv("EDC_CONNECTOR_API_KEY",
                                                   "demo-api-key"))


# --------------------------------------------------------------------------- #
# presentation                                                                 #
# --------------------------------------------------------------------------- #

def _clauses(offer: Dict[str, Any], rule_type: str = "permission",
             action: str = "use") -> Dict[str, Any]:
    for rule in offer.get(rule_type) or []:
        if strip_prefix(rule.get("action")) == action:
            return {strip_prefix(c["leftOperand"]): c
                    for c in rule.get("constraint", []) or []}
    return {}


def _describe(offer: Optional[Dict[str, Any]]) -> str:
    if not offer:
        return "    (none)"
    lines = []
    for rule_type in ("permission", "prohibition", "obligation"):
        for rule in offer.get(rule_type) or []:
            terms = ", ".join(
                f"{strip_prefix(c['leftOperand'])} "
                f"{strip_prefix(c['operator'])} {c['rightOperand']}"
                for c in rule.get("constraint", []) or [])
            lines.append(f"    {rule_type} {strip_prefix(rule['action'])}: {terms}")
    return "\n".join(lines) or "    (no constraints)"


def _report(outcome: ConnectorOutcome) -> None:
    """Everything a write-up would need, and the caveats it must not omit."""
    print("\n" + "=" * 74)
    print(f"outcome: {outcome.kind} after {len(outcome.rounds)} rounds")
    print(f"real DSP contract requests: {outcome.dsp_exchanges}")

    if outcome.reason:
        print(f"reason: {outcome.reason}")

    if outcome.agreement:
        print("\nagreed terms:")
        print(_describe(outcome.agreement))
        print(f"\noperands in the agreement: {operands_used(outcome.agreement)}")

    if outcome.contract_agreement_id:
        print(f"contract agreement id:  {outcome.contract_agreement_id}")
        print(f"final negotiation id:   {outcome.final_negotiation_id}")

    if outcome.rounds and outcome.agreement:
        first = _clauses(outcome.rounds[0].published_offer or {})
        final = _clauses(outcome.agreement)
        moved = sorted(k for k in final if k in first
                       and str(final[k]["rightOperand"])
                       != str(first[k]["rightOperand"]))
        print(f"operands that moved:    {moved}")

    # Both hard policies survived, or the agreement is not defensible.
    _check_hard_policies(outcome)

    print("\nhow the provider learned each proposal:")
    read = [r.index for r in outcome.rounds if r.provider_saw_request]
    relayed = [r.index for r in outcome.rounds if r.relayed]
    print(f"  read from its own connector: {read}")
    print(f"  relayed out of band:         {relayed}")
    if relayed:
        print("  -> say 'BAMBON drove EDC contract negotiations', NOT")
        print("     'the provider observed proposals purely via EDC'")

    notes = [(r.index, r.note) for r in outcome.rounds if r.note]
    if notes:
        print("\nnotes:")
        for index, note in notes:
            print(f"  round {index}: {note}")

    pull = outcome.pull
    print("\ndata transfer:")
    if pull is None:
        print("  not attempted")
    elif getattr(pull, "ok", False):
        records = (f", {pull.record_count} records"
                   if pull.record_count is not None else "")
        print(f"  transfer process {pull.transfer_process_id}")
        print(f"  HTTP {pull.status_code} from the provider data plane, "
              f"{pull.bytes_received} bytes{records}")
        print(f"  first {len(pull.preview)} chars: {pull.preview}")
    else:
        print(f"  {getattr(pull, 'kind', '?')}: {getattr(pull, 'reason', pull)}")
    print("=" * 74)


def _check_hard_policies(outcome: ConnectorOutcome) -> None:
    """A negotiated agreement that breaches either side's floor is a bug, not a
    result. Checked here as well as in the tests so a manual run cannot be
    misread."""
    if not outcome.agreement:
        return
    final = _clauses(outcome.agreement)
    floor = _clauses(CONSUMER_HARD_POLICY)
    ceiling = _clauses(PROVIDER_HARD_POLICY)
    problems = []

    try:
        elapsed = float(final["elapsedTime"]["rightOperand"])
        if elapsed < float(floor["elapsedTime"]["rightOperand"]):
            problems.append("consumer signed below its own elapsedTime floor")
        if elapsed > float(ceiling["elapsedTime"]["rightOperand"]):
            problems.append("provider signed above its own elapsedTime ceiling")
        def _members(clause):
            value = clause["rightOperand"]
            return set(value) if isinstance(value, (list, tuple, set)) else {value}

        if not _members(final["spatial"]) <= _members(ceiling["spatial"]):
            problems.append("agreed regions exceed the provider's hard policy")
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"could not verify hard policies: {exc}")

    print("hard policies:          "
          + ("both respected" if not problems else "VIOLATED"))
    for problem in problems:
        print(f"  !! {problem}")


# --------------------------------------------------------------------------- #
# the flow                                                                     #
# --------------------------------------------------------------------------- #

async def run(asset_id: str = ASSET_ID,
              *,
              dialect_name: Optional[str] = None,
              max_rounds: int = 12,
              do_transfer: bool = True,
              transfer_timeout: int = 120) -> int:
    dialect = dialect_by_name(dialect_name
                              or os.getenv("BAMBON_EDC_DIALECT", "vanilla"))

    consumer_mgmt = ManagementClient(base_url=CONSUMER_MANAGEMENT_URL,
                                     api_key=API_AUTH_KEY)
    provider_mgmt = ManagementClient(base_url=PROVIDER_MANAGEMENT_URL,
                                     api_key=API_AUTH_KEY)

    print("BAMBON between two local EDC connectors")
    print(f"  consumer management  {CONSUMER_MANAGEMENT_URL}")
    print(f"  provider management  {PROVIDER_MANAGEMENT_URL}")
    print(f"  provider DSP         {PROVIDER_PROTOCOL_URL}")
    print(f"  asset                {asset_id}")
    print(f"  policy dialect       {dialect.name}")

    if not await _connectors_reachable(consumer_mgmt, provider_mgmt):
        return 4

    puller = None
    if do_transfer:
        from .transfer import TransferUnavailable, make_puller
        try:
            puller = make_puller(
                provider_protocol_url=PROVIDER_PROTOCOL_URL,
                provider_participant_id=PROVIDER_PARTICIPANT_ID,
                timeout=transfer_timeout)
        except TransferUnavailable as exc:
            print(f"  transfer disabled: {exc}")

    negotiation = ConnectorNegotiation(
        consumer_management=consumer_mgmt,
        provider_management=provider_mgmt,
        provider_protocol_url=PROVIDER_PROTOCOL_URL,
        provider_participant_id=PROVIDER_PARTICIPANT_ID,
        consumer_participant_id=CONSUMER_PARTICIPANT_ID,
        asset_id=asset_id,
        dialect=dialect,
        max_rounds=max_rounds,
        puller=puller,
        on_event=print,
    )

    outcome = await negotiation.run()
    _report(outcome)

    if outcome.kind != "agreement":
        print("\nNo agreement. If the reason is an empty ZOPA this is a correct "
              "result: BAMBON stopped rather than sign past a hard policy.")
        return 3
    if do_transfer and not outcome.data_moved:
        print("\nTerms were agreed but no data arrived. The negotiation claim "
              "stands; the end-to-end claim does not.")
        return 2

    print("\nThe terms that moved this data were negotiated, not dictated.")
    return 0


async def _connectors_reachable(consumer: ManagementClient,
                                provider: ManagementClient) -> bool:
    """Fail with a useful message rather than a stack trace 30 seconds in.

    Also reports whether the provider is still sitting on its *seeded*
    restrictive offer. It will not be after a previous run, because the provider
    retires each contract definition as it counter-offers - so a bare
    ``docker compose run --rm runner`` would start bargaining from wherever the
    last run left off, which quietly changes the scenario. Better to say so.
    """
    definitions = {}
    for label, client in (("consumer", consumer), ("provider", provider)):
        try:
            definitions[label] = await client.list_contract_definitions()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  {label} connector unreachable at {client.base_url}: {exc}")
            print("  Run `docker compose up -d --wait` first.")
            return False

    ids = [str(d.get("@id", "")) for d in definitions.get("provider", [])]
    if PUBLISHED_DEFINITION_ID not in ids:
        print(f"\n  note: the provider is not offering "
              f"{PUBLISHED_DEFINITION_ID!r}. Its live contract definitions are "
              f"{ids or 'none'}.")
        print("  A previous run retired the seeded offer, so this run starts "
              "from those terms instead of the restrictive opening one.")
        print("  Re-seed with: docker compose run --rm --no-deps "
              "-e EDC_CONNECTOR_HOST=provider-connector "
              "-e EDC_CONNECTOR_MANAGEMENT_PORT=19193 "
              "provider-negotiator python -m bambon_edc.seed_provider")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BAMBON negotiation between two local EDC connectors, "
                    "followed by a consumer pull of the negotiated dataset.")
    parser.add_argument("--asset-id", default=ASSET_ID)
    parser.add_argument("--dialect", default=None,
                        choices=["vanilla", "tractusx"],
                        help="which connector family to render policies for")
    parser.add_argument("--max-rounds", type=int,
                        default=int(os.getenv("E2E_MAX_ROUNDS", "12")))
    parser.add_argument("--no-transfer", action="store_true",
                        help="stop once the contract agreement exists")
    parser.add_argument("--transfer-timeout", type=int, default=120,
                        help="seconds to wait for the pull credentials")
    args = parser.parse_args()

    return asyncio.run(run(args.asset_id,
                           dialect_name=args.dialect,
                           max_rounds=args.max_rounds,
                           do_transfer=not args.no_transfer,
                           transfer_timeout=args.transfer_timeout))


if __name__ == "__main__":
    sys.exit(main())
