"""The consumer-side story, end to end.

    1. fetch the provider catalog with edcpy
    2. show that the published offer is unacceptable, and that a stock EDC
       negotiation on it would be pointless
    3. run BAMBON bilaterally over the side channel until odrl:Agreement
    4. the provider auto-publishes the agreed policy (see service.py)
    5. re-fetch the catalog - the negotiated offer is now there
    6. run edcpy's normal negotiation + transfer flow against it

Steps 1, 5 and 6 are plain ``edcpy``. Steps 2-4 are the contribution.

Run::

    python -m bambon_edc.consumer_flow
    python -m bambon_edc.consumer_flow --offline   # steps 2-4 only, no data space
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .odrl_edc import (offer_from_edc_dataset, offer_from_edc_policy,
                       operands_used, strip_prefix)
from .profiles import ASSET_ID
from .session import (HttpParty, LocalParty, NegotiationSession, Outcome, Step,
                      drive, negotiate_offline, relay)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(message)s")
_logger = logging.getLogger("bambon_edc.consumer")

PROVIDER_PROTOCOL_URL = os.getenv(
    "PROVIDER_PROTOCOL_URL", "http://provider-connector:9194/protocol")
PROVIDER_CONNECTOR_ID = os.getenv("PROVIDER_CONNECTOR_ID", "provider")
PROVIDER_SIDECHANNEL_URL = os.getenv(
    "PROVIDER_SIDECHANNEL_URL", "http://provider-negotiator:8100")
PARTICIPANT_ID = os.getenv("EDC_CONNECTOR_PARTICIPANT_ID", "consumer")


# --------------------------------------------------------------------------- #
# catalog helpers                                                              #
# --------------------------------------------------------------------------- #

def _datasets(catalog: Any) -> List[Dict[str, Any]]:
    """``edcpy``'s Dataset wrapper has moved around between releases; accept
    either the wrapper or a plain dict."""
    out = []
    for dataset in catalog.datasets:
        for attr in ("dct_dict", "data", "raw"):
            value = getattr(dataset, attr, None)
            if isinstance(value, dict):
                out.append(value)
                break
        else:
            out.append(dataset if isinstance(dataset, dict) else vars(dataset))
    return out


def _find_asset(datasets: List[Dict[str, Any]],
                asset_id: str) -> List[Dict[str, Any]]:
    return [d for d in datasets
            if asset_id in (str(d.get("@id", "")), str(d.get("id", "")))
            or asset_id in json.dumps(d.get("@id", ""))]


def _describe(offer: Dict[str, Any]) -> str:
    lines = []
    for rule_type in ("permission", "prohibition", "obligation"):
        for rule in offer.get(rule_type) or []:
            terms = ", ".join(
                f"{strip_prefix(c['leftOperand'])} "
                f"{strip_prefix(c['operator'])} {c['rightOperand']}"
                for c in rule.get("constraint", []) or [])
            lines.append(f"    {rule_type} {rule['action']}: {terms}")
    return "\n".join(lines) or "    (no constraints)"


# --------------------------------------------------------------------------- #
# main flow                                                                    #
# --------------------------------------------------------------------------- #

async def run(asset_id: str = ASSET_ID, *, do_transfer: bool = True) -> int:
    from edcpy.edc_api import ConnectorController

    controller = ConnectorController()

    # --- 1. what does the provider publish? ------------------------------- #
    print("\n[1] Fetching provider catalog via edcpy")
    catalog = await controller.fetch_catalog(
        counter_party_protocol_url=PROVIDER_PROTOCOL_URL)
    datasets = _datasets(catalog)
    matches = _find_asset(datasets, asset_id)
    if not matches:
        print(f"    asset {asset_id!r} not in catalog "
              f"({len(datasets)} datasets seen). Did seed_provider.py run?")
        return 1

    published = offer_from_edc_dataset(matches[0])
    print("    published offer:")
    print(_describe(published))

    # --- 2. why the stock flow stops here --------------------------------- #
    print("\n[2] This offer does not meet our hard policy "
          "(we need commercial use for >= 60 days).")
    print("    EDC only lets us request the catalog offer verbatim, so a stock")
    print("    ContractRequestMessage here would either bind us to unusable")
    print("    terms or be terminated. Escalating to BAMBON.")

    # --- 3. bilateral BAMBON over the side channel ------------------------ #
    print(f"\n[3] Bilateral BAMBON negotiation with {PROVIDER_SIDECHANNEL_URL}")
    consumer = NegotiationSession(role="consumer", asset_id=asset_id,
                                  counter_party_id=PROVIDER_CONNECTOR_ID)
    provider = HttpParty(PROVIDER_SIDECHANNEL_URL, asset_id, PARTICIPANT_ID)

    outcome = drive(consumer, provider, max_rounds=60, on_round=_trace)

    if outcome.kind != "agreement":
        print(f"\n    no agreement: {outcome.kind} - {outcome.reason}")
        print("    This is a legitimate result: BAMBON detected the ZOPA is "
              "empty and stopped, rather than signing a policy that violates "
              "our hard constraints.")
        return 2

    print(f"\n    agreement after {outcome.rounds} rounds "
          f"(declared by {outcome.settled_by}):")
    print(_describe(outcome.agreement))
    print(f"    operands in play: {operands_used(outcome.agreement)}")

    # --- 4. the provider has published it --------------------------------- #
    if not provider.contract_definition_id:
        print("\n[4] Provider did not publish the agreement "
              "(BAMBON_AUTO_MATERIALISE off, or materialisation failed).")
        return 3
    print(f"\n[4] Provider published policy {provider.policy_id} "
          f"and contract definition {provider.contract_definition_id}")

    # --- 5. the catalog now carries the negotiated terms ------------------ #
    print("\n[5] Re-fetching catalog")
    catalog = await controller.fetch_catalog(
        counter_party_protocol_url=PROVIDER_PROTOCOL_URL)
    offers = _offers_for(_find_asset(_datasets(catalog), asset_id))
    print(f"    {len(offers)} offer(s) now visible to us:")
    for offer in offers:
        print(_describe(offer_from_edc_policy(offer)))

    if not do_transfer:
        return 0

    # --- 6. stock EDC negotiation + transfer ------------------------------ #
    print("\n[6] Running the *unmodified* edcpy negotiation + transfer flow")
    transfer_details = await controller.run_negotiation_flow(
        counter_party_protocol_url=PROVIDER_PROTOCOL_URL,
        counter_party_connector_id=PROVIDER_CONNECTOR_ID,
        asset_query=asset_id,
    )
    print(f"    contract agreement: {getattr(transfer_details, 'contract_agreement_id', transfer_details)}")

    transfer_id = await controller.run_transfer_flow(
        transfer_details=transfer_details, is_provider_push=False)
    print(f"    transfer process: {transfer_id}")
    print("\nDone. The terms that moved this data were negotiated, not dictated.")
    return 0


def _offers_for(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for dataset in datasets:
        policy = dataset.get("odrl:hasPolicy") or dataset.get("hasPolicy")
        out.extend(policy if isinstance(policy, list) else [policy] if policy else [])
    return out


def _trace(round_index: int, who: str, offer: Dict[str, Any]) -> None:
    label = "consumer" if who == "consumer" else "provider"
    marker = "==" if offer.get("@type", "").endswith("Agreement") else "->"
    terms = " | ".join(line.strip() for line in _describe(offer).splitlines())
    print(f"    r{round_index:<3} {label:<8} {marker} {terms}")


def _offline(dialect_name: str = "vanilla") -> int:
    from .odrl_edc import (access_policy_definition, contract_definition,
                           dialect_by_name, policy_definition)

    dialect = dialect_by_name(dialect_name)
    print(f"Offline mode: bilateral BAMBON only, no EDC connectors.")
    print(f"Policy dialect: {dialect.name}"
          + (f" (baseline: {[c['leftOperand'] for c in dialect.baseline]})"
             if dialect.baseline else "") + "\n")

    outcome: Outcome = negotiate_offline(ASSET_ID, on_round=_trace)
    print(f"\noutcome: {outcome.kind} after {outcome.rounds} rounds")
    if not outcome.agreement:
        return 2

    print(_describe(outcome.agreement))
    print("\nEDC contract PolicyDefinition that would be POSTed:")
    print(json.dumps(policy_definition(
        outcome.agreement, policy_id="bambon-contract-demo",
        dialect=dialect), indent=2))
    print("\nEDC access PolicyDefinition (who may see the offer):")
    print(json.dumps(access_policy_definition(
        policy_id="bambon-access-demo", participant_ids=["consumer"],
        dialect=dialect), indent=2))
    print("\nEDC ContractDefinition:")
    print(json.dumps(contract_definition(
        asset_id=ASSET_ID,
        access_policy_id="bambon-access-demo",
        contract_policy_id="bambon-contract-demo"), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="run the negotiation and normalisation only")
    parser.add_argument("--dialect", default=os.getenv("BAMBON_EDC_DIALECT",
                                                       "vanilla"),
                        choices=["vanilla", "tractusx"],
                        help="which connector family to render policies for")
    parser.add_argument("--asset-id", default=ASSET_ID)
    parser.add_argument("--no-transfer", action="store_true",
                        help="stop after the catalog re-fetch")
    args = parser.parse_args()

    if args.offline:
        return _offline(args.dialect)
    return asyncio.run(run(args.asset_id, do_transfer=not args.no_transfer))


if __name__ == "__main__":
    sys.exit(main())
