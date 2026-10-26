"""A pair of fake EDC connectors, faithful about the things that break demos.

Why fake connectors rather than mocks
-------------------------------------
``ConnectorNegotiation`` is where the interesting bugs live - offer ids going
stale, contract definitions piling up so the catalog serves two offers at once,
the acceptance round having to quote the published policy byte-for-byte - and
none of those are visible in the offline bargaining tests. But reproducing them
needs a connector, and a connector needs Docker.

So these classes implement the slice of the Management API that
``ConnectorNegotiation`` actually calls, and reproduce the four behaviours that
matter:

1. **A fresh offer id per catalog fetch.** Real EDC mints a new base64 id every
   time, so code that caches one and quotes it later gets TERMINATED.
2. **Byte-equality on the requested policy.** A contract request is FINALIZED
   only if its policy matches the currently published one after normalisation;
   anything else is TERMINATED. That is precisely the take-it-or-leave-it
   behaviour BAMBON exists to work around.
3. **Catalog = whatever contract definitions exist.** Every live definition for
   an asset contributes a policy to ``odrl:hasPolicy``. Forget to retire the old
   one and the dataset carries two offers - the bug the fake catches.
4. **The provider sees the policy it rejected**, via its own
   ``/contractnegotiations/request``, so the non-relay path gets exercised.

The policies travel through the real ``odrl_edc`` writer and parser, so the
JSON-LD normalisation is under test here too, not stubbed out.

What it does NOT do: DSP itself, tokens, identity, or a data plane. A pass here
is evidence the orchestration is correct, never evidence that EDC accepts it -
that is what ``test_e2e_two_connectors.py`` is for.
"""

from __future__ import annotations

import base64
import copy
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from bambon_edc.management import ManagementApiError
from bambon_edc.odrl_edc import (VANILLA, Dialect, offer_from_edc_policy,
                                 strip_prefix)

__all__ = ["FakeDataSpace", "FakeProviderClient", "FakeConsumerClient",
           "StubPuller", "canonical_terms"]


def canonical_terms(offer: Dict[str, Any]) -> Tuple:
    """A comparable, order-insensitive form of an offer's terms.

    Real EDC compares the requested policy against the published one; doing that
    on raw JSON would make the fake fail for reasons no connector cares about
    (key order, ``@id``, numeric formatting). This normalises to what both sides
    would agree is "the same terms": rule type, action, and the set of
    ``(operand, operator, rightOperand)`` triples with right operands stringified
    the way the writer stringifies them.
    """
    out = []
    for rule_type in ("permission", "prohibition", "obligation"):
        for rule in offer.get(rule_type) or []:
            clauses = []
            for clause in rule.get("constraint", []) or []:
                right = clause["rightOperand"]
                if isinstance(right, (list, tuple, set)):
                    rendered: Any = tuple(sorted(str(v) for v in right))
                elif isinstance(right, float) and right.is_integer():
                    rendered = str(int(right))
                else:
                    rendered = str(right)
                clauses.append((strip_prefix(clause["leftOperand"]),
                                strip_prefix(clause["operator"]), rendered))
            out.append((rule_type, strip_prefix(rule["action"]),
                        tuple(sorted(clauses))))
    return tuple(sorted(out))


class FakeDataSpace:
    """Shared state: the provider's catalog and both sides' negotiation records.

    One object stands in for the pair of connectors *and* the DSP link between
    them, because the whole point of the harness is that the consumer only ever
    learns things the provider published.
    """

    def __init__(self, asset_id: str, *,
                 published_definition_id: str,
                 published_policy_id: str,
                 dialect: Dialect = VANILLA,
                 provider_keeps_rejected: bool = True):
        self.asset_id = asset_id
        self.dialect = dialect
        #: When False the provider connector forgets the policy of a request it
        #: rejected, which forces the relay fallback - the weaker mode the demo
        #: has to disclose. Both paths need a test.
        self.provider_keeps_rejected = provider_keeps_rejected

        #: policy definition id -> policy body (as POSTed)
        self.policies: Dict[str, Dict[str, Any]] = {}
        #: contract definition id -> (access policy id, contract policy id)
        self.definitions: Dict[str, Tuple[str, str]] = {}
        self.published_definition_id = published_definition_id
        self.published_policy_id = published_policy_id

        #: Every offer id ever handed out, and the ids from the most recent
        #: fetch. Quoting anything outside the latter is what "stale offer id"
        #: means, and real EDC terminates on it.
        self.offer_ids: List[str] = []
        self.last_offer_ids: List[str] = []
        self.catalog_fetches = 0

        self.consumer_negotiations: Dict[str, Dict[str, Any]] = {}
        self.provider_negotiations: List[Dict[str, Any]] = []

        #: Every policy the consumer requested, in order. Test evidence.
        self.requested_policies: List[Dict[str, Any]] = []
        self.deleted_definitions: List[str] = []
        self.transfers: List[Dict[str, Any]] = []

    # -- provider bookkeeping ------------------------------------------- #

    def live_offers(self) -> List[Dict[str, Any]]:
        """The policy body of every contract definition bound to the asset."""
        offers = []
        for definition_id, (_access, contract_policy_id) in self.definitions.items():
            body = self.policies.get(contract_policy_id)
            if body is not None:
                offers.append(body)
        return offers

    def current_offer(self) -> Optional[Dict[str, Any]]:
        offers = self.live_offers()
        return offers[0] if offers else None

    def mint_offer_id(self) -> str:
        """EDC returns an opaque base64 id; mimic the shape, not just a uuid."""
        raw = f"{uuid.uuid4()}:{self.asset_id}:{uuid.uuid4()}".encode()
        offer_id = base64.b64encode(raw).decode()
        self.offer_ids.append(offer_id)
        return offer_id


class _FakeClient:
    """Shared plumbing so the two fakes present a ManagementClient surface."""

    def __init__(self, space: FakeDataSpace, base_url: str):
        self.space = space
        self.base_url = base_url

    # ManagementClient.ensure dispatches on these three.
    async def create_asset(self, body: Dict[str, Any]) -> Any:
        return {"@id": body.get("@id")}

    async def create_policy_definition(self, body: Dict[str, Any]) -> Any:
        policy_id = body["@id"]
        if policy_id in self.space.policies:
            raise ManagementApiError("POST", self.base_url, 409, "already exists")
        self.space.policies[policy_id] = copy.deepcopy(body["policy"])
        return {"@id": policy_id}

    async def create_contract_definition(self, body: Dict[str, Any]) -> Any:
        definition_id = body["@id"]
        if definition_id in self.space.definitions:
            raise ManagementApiError("POST", self.base_url, 409, "already exists")
        self.space.definitions[definition_id] = (body["accessPolicyId"],
                                                 body["contractPolicyId"])
        return {"@id": definition_id}

    async def list_contract_definitions(self) -> List[Dict[str, Any]]:
        return [{"@id": key} for key in self.space.definitions]

    async def delete_contract_definition(self, definition_id: str) -> Any:
        if definition_id not in self.space.definitions:
            raise ManagementApiError("DELETE", self.base_url, 404, "not found")
        del self.space.definitions[definition_id]
        self.space.deleted_definitions.append(definition_id)
        return None

    async def delete_policy_definition(self, policy_id: str) -> Any:
        self.space.policies.pop(policy_id, None)
        return None

    async def ensure(self, kind: str, body: Dict[str, Any]) -> Any:
        creators = {
            "asset": self.create_asset,
            "policy": self.create_policy_definition,
            "contract": self.create_contract_definition,
        }
        try:
            return await creators[kind](body)
        except ManagementApiError as exc:
            if exc.status == 409:
                return {"@id": body.get("@id"), "reused": True}
            raise


class FakeProviderClient(_FakeClient):
    """The provider's own Management API, as its BAMBON sees it."""

    async def query_negotiations(self, filter_expression=None, limit: int = 200):
        if not self.space.provider_keeps_rejected:
            return []
        return list(reversed(self.space.provider_negotiations))[:limit]


class FakeConsumerClient(_FakeClient):
    """The consumer's own Management API: catalog fetches and contract requests."""

    async def fetch_catalog(self, *, counter_party_address: str,
                            counter_party_id: str,
                            protocol: str = "dataspace-protocol-http",
                            limit: int = 200) -> Dict[str, Any]:
        self.space.catalog_fetches += 1
        offers = self.space.live_offers()
        if not offers:
            # A real connector serves a catalog with no datasets, not an error.
            return {"@context": {}, "@type": "dcat:Catalog", "dcat:dataset": []}

        # A distinct offer id per policy, all newly minted, as EDC does.
        has_policy = []
        self.space.last_offer_ids = []
        for body in offers:
            entry = copy.deepcopy(body)
            entry["@id"] = self.space.mint_offer_id()
            entry["@type"] = "odrl:Offer"
            self.space.last_offer_ids.append(entry["@id"])
            has_policy.append(entry)

        return {
            "@context": {"odrl": "http://www.w3.org/ns/odrl/2/"},
            "@type": "dcat:Catalog",
            "dcat:dataset": {
                "@id": self.space.asset_id,
                "@type": "dcat:Dataset",
                # Single-element lists collapse to a bare object in real
                # catalogs; keep that quirk so the parser stays under test.
                "odrl:hasPolicy": has_policy if len(has_policy) > 1 else has_policy[0],
                "dcat:distribution": [],
            },
        }

    async def initiate_negotiation(self, *, counter_party_address: str,
                                   counter_party_id: str,
                                   offer_id: str,
                                   asset_id: str,
                                   policy: Dict[str, Any],
                                   protocol: str = "dataspace-protocol-http",
                                   ) -> str:
        negotiation_id = f"urn:uuid:{uuid.uuid4()}"
        requested = offer_from_edc_policy(policy)
        self.space.requested_policies.append(requested)

        published = self.space.current_offer()
        published_terms = (canonical_terms(offer_from_edc_policy(published))
                           if published else None)

        stale = offer_id not in self.space.last_offer_ids
        matches = published_terms == canonical_terms(requested)

        if matches and not stale:
            state, agreement_id = "FINALIZED", f"urn:uuid:{uuid.uuid4()}"
        else:
            state, agreement_id = "TERMINATED", None

        self.space.consumer_negotiations[negotiation_id] = {
            "@id": negotiation_id,
            "state": state,
            "type": "CONSUMER",
            "contractAgreementId": agreement_id,
        }
        # The provider's connector records the request it received. Real EDC
        # keeps this under `contractOffers`; whether it survives a rejection is
        # version-dependent, which `provider_keeps_rejected` models.
        self.space.provider_negotiations.append({
            "@id": f"urn:uuid:{uuid.uuid4()}",
            "state": state,
            "type": "PROVIDER",
            "contractOffers": [{"policy": copy.deepcopy(policy)}],
        })
        return negotiation_id

    async def get_negotiation(self, negotiation_id: str) -> Dict[str, Any]:
        return self.space.consumer_negotiations.get(negotiation_id, {})

    async def await_negotiation(self, negotiation_id: str, *,
                                timeout: float = 60.0,
                                interval: float = 1.0) -> Dict[str, Any]:
        return await self.get_negotiation(negotiation_id)


class StubPuller:
    """Stands in for ``transfer.make_puller`` without a data plane.

    Records what it was asked to pull so a test can assert the transfer quoted
    the agreement BAMBON negotiated, and not some later one.
    """

    def __init__(self, space: FakeDataSpace, *, payload: Any = None,
                 fail: bool = False):
        self.space = space
        self.payload = payload if payload is not None else {
            "observations": [{"station": "ES-AST-01", "temperature_c": 12.4}]}
        self.fail = fail
        self.calls: List[Dict[str, str]] = []

    async def __call__(self, *, asset_id: str, contract_agreement_id: str):
        from bambon_edc.transfer import PullResult, _summarise

        self.calls.append({"asset_id": asset_id,
                           "contract_agreement_id": contract_agreement_id})
        self.space.transfers.append({"asset_id": asset_id,
                                     "contract_agreement_id": contract_agreement_id})

        if self.fail:
            return PullResult(kind="no_edr",
                              contract_agreement_id=contract_agreement_id,
                              transfer_process_id=f"urn:uuid:{uuid.uuid4()}",
                              reason="stubbed failure")

        body = json.dumps(self.payload).encode()
        preview, count = _summarise(body)
        return PullResult(kind="pulled",
                          contract_agreement_id=contract_agreement_id,
                          transfer_process_id=f"urn:uuid:{uuid.uuid4()}",
                          endpoint="http://provider-connector:19291/public",
                          status_code=200,
                          bytes_received=len(body),
                          preview=preview,
                          record_count=count)
