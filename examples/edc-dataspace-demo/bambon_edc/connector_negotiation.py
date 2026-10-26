"""A bilateral negotiation conducted *through* two EDC connectors.

The distinction that matters
----------------------------
``service.py`` runs BAMBON over a private HTTP side channel: the agents talk to
each other and the connectors are barely involved. That demonstrates the
bargaining but it is not a negotiation between two connectors.

This module does it the other way round. **Every round is real Dataspace
Protocol traffic between the two connectors.** Neither BAMBON ever talks to the
other; each one only reads and writes its own connector's Management API.

One round
---------
::

    consumer connector  --- DSP catalog request ------->  provider connector
                        <-- provider's current offer ----

    consumer BAMBON observes the offer, produces a counter

    consumer connector  --- DSP ContractRequestMessage ->  provider connector
                            (carrying the counter)
                        <-- TERMINATED (policy != published offer)

    provider BAMBON reads the rejected request from its *own* connector,
    observes it, produces its counter, and republishes it as a new contract
    definition.  That republication is the counter-offer.

    ... loop ...

    once consumer BAMBON accepts, it requests the published offer verbatim
    and the negotiation reaches FINALIZED with a real contract agreement.

    consumer connector  --- transfer process (HttpData-PULL) -->  provider
                        <-- EndpointDataReference --------------
                        --- GET the data plane endpoint ------->
                        <-- the dataset -----------------------

So the counter-offer mechanism EDC lacks is emulated with the two operations it
does support: *request* (consumer) and *republish* (provider). The consumer's
proposals ride inside genuine `ContractRequestMessage`s; the provider's ride in
the catalog. Nothing is exchanged out of band.

The transfer at the end is quoted against *that* contract agreement id, so the
data is moving under the terms BAMBON negotiated and not under some second
agreement obtained by accepting the published offer. See ``transfer.py``.

Caveat: step 4 depends on the provider connector persisting the policy of a
request it rejected. That is version-dependent. ``relay_fallback=True`` (the
default) hands the provider the policy the consumer actually sent when the
connector does not expose it, and records that it did so on the transcript, so a
run never silently degrades into something weaker than it claims to be.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import (Any, Awaitable, Callable, Dict, List, Optional, Protocol,
                    Sequence)

from .management import ManagementApiError, ManagementClient
from .odrl_edc import (Dialect, VANILLA, access_policy_definition,
                       contract_definition, offer_from_edc_policy,
                       policy_definition, strip_prefix)
from .profiles import PUBLISHED_DEFINITION_ID
from .session import NegotiationSession, is_agreement

_logger = logging.getLogger(__name__)

IN_FLIGHT = ("INITIAL", "REQUESTING", "REQUESTED", "OFFERING", "OFFERED",
             "ACCEPTING", "ACCEPTED", "AGREEING", "AGREED", "VERIFYING",
             "VERIFIED", "FINALIZING")


# --------------------------------------------------------------------------- #
# transcript                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class Round:
    """One full pass through both connectors. Every field is evidence."""

    index: int
    published_offer: Optional[Dict[str, Any]] = None
    published_offer_id: Optional[str] = None
    consumer_proposal: Optional[Dict[str, Any]] = None
    negotiation_id: Optional[str] = None
    negotiation_state: Optional[str] = None
    offers_in_catalog: int = 0
    provider_saw_request: bool = False
    relayed: bool = False
    provider_counter: Optional[Dict[str, Any]] = None
    republished_policy_id: Optional[str] = None
    agreement_id: Optional[str] = None
    note: str = ""

    def add_note(self, text: str) -> None:
        """Notes accumulate: a round can be remarkable in more than one way."""
        self.note = f"{self.note}; {text}" if self.note else text


class Puller(Protocol):
    """Moves the data once terms are agreed.

    Injected rather than imported so the negotiation can be exercised without a
    data plane: ``transfer.make_puller`` is the real one, the dry-run harness
    passes a stub with the same signature. Returns a
    :class:`~bambon_edc.transfer.PullResult`, but is typed loosely here so this
    module never has to import ``edcpy``.
    """

    def __call__(self, *, asset_id: str,
                 contract_agreement_id: str) -> Awaitable[Any]: ...


@dataclass
class ConnectorOutcome:
    kind: str                       # agreement | infeasible | timeout | error
    rounds: List[Round] = field(default_factory=list)
    agreement: Optional[Dict[str, Any]] = None
    contract_agreement_id: Optional[str] = None
    final_negotiation_id: Optional[str] = None
    reason: Optional[str] = None
    #: Result of the post-agreement transfer, when a puller was supplied.
    #: ``None`` means no pull was attempted, which is not the same as a pull
    #: that failed - tests must distinguish the two.
    pull: Optional[Any] = None

    @property
    def dsp_exchanges(self) -> int:
        """How many real contract requests crossed between the connectors."""
        return sum(1 for r in self.rounds if r.negotiation_id)

    @property
    def data_moved(self) -> bool:
        return bool(getattr(self.pull, "ok", False))


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _state(record: Dict[str, Any]) -> str:
    return str(record.get("state") or record.get("edc:state") or "")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _datasets(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("dcat:dataset", "dataset", "http://www.w3.org/ns/dcat#dataset"):
        if key in catalog:
            return _as_list(catalog[key])
    return []


def _find_offer(catalog: Dict[str, Any], asset_id: str):
    """Return ``(offer_id, bambon_offer, offer_count)`` for ``asset_id``.

    EDC mints a fresh base64 offer id per catalog fetch, and a contract request
    must quote the *current* one, so it is returned alongside the terms.

    ``offer_count`` is how many policies the dataset carried. It should always be
    one: the provider retires its previous contract definition whenever it
    counter-offers. More than one means stale definitions accumulated, in which
    case "the provider's current offer" is whichever one EDC happened to
    serialise first - so the caller reports it rather than silently negotiating
    against an arbitrary policy.
    """
    for dataset in _datasets(catalog):
        if asset_id not in str(dataset.get("@id", "")):
            continue
        policies = [p for p in _as_list(dataset.get("odrl:hasPolicy")
                                        or dataset.get("hasPolicy")) if p]
        if not policies:
            continue
        raw = policies[0]
        return raw.get("@id"), offer_from_edc_policy(raw), len(policies)
    return None, None, 0


def _odrl_policy_for_request(offer: Dict[str, Any],
                             dialect: Dialect) -> Dict[str, Any]:
    """Render a BAMBON offer as the ``policy`` body of a contract request."""
    body = policy_definition(dict(offer, **{"@type": "odrl:Agreement"}),
                             dialect=dialect)
    return body["policy"]


# --------------------------------------------------------------------------- #
# the loop                                                                     #
# --------------------------------------------------------------------------- #

class ConnectorNegotiation:
    """Drives a BAMBON negotiation through two live EDC connectors."""

    def __init__(self,
                 *,
                 consumer_management: ManagementClient,
                 provider_management: ManagementClient,
                 provider_protocol_url: str,
                 provider_participant_id: str,
                 consumer_participant_id: str,
                 asset_id: str,
                 dialect: Dialect = VANILLA,
                 max_rounds: int = 12,
                 negotiation_timeout: float = float(
                     os.getenv("NEGOTIATION_TIMEOUT", "12")),
                 relay_fallback: bool = True,
                 puller: Optional[Puller] = None,
                 published_definition_ids: Sequence[str] = (
                     PUBLISHED_DEFINITION_ID,),
                 on_event: Optional[Callable[[str], None]] = None):
        self.consumer_mgmt = consumer_management
        self.provider_mgmt = provider_management
        self.provider_protocol_url = provider_protocol_url
        self.provider_participant_id = provider_participant_id
        self.consumer_participant_id = consumer_participant_id
        self.asset_id = asset_id
        self.dialect = dialect
        self.max_rounds = max_rounds
        self.negotiation_timeout = negotiation_timeout
        self.relay_fallback = relay_fallback
        self.puller = puller
        self._emit = on_event or (lambda msg: _logger.info(msg))

        # Contract definitions to retire the first time the provider counters.
        # Seeded with the *published* offer's definition so that the consumer
        # sees exactly one offer per catalog fetch: EDC returns one dataset
        # carrying every matching contract definition's policy in
        # ``odrl:hasPolicy``, in no defined order, so leaving the restrictive
        # opening offer in place would make "the provider's current offer"
        # ambiguous and the negotiation unreadable.
        self._previous_definitions: List[str] = list(published_definition_ids)
        self._access_policy_id: Optional[str] = None

        # One BAMBON per participant. Neither is ever handed the other's state;
        # each only ever sees what arrives through its own connector.
        self.consumer = NegotiationSession(
            role="consumer", asset_id=asset_id,
            counter_party_id=provider_participant_id)
        self.provider = NegotiationSession(
            role="provider", asset_id=asset_id,
            counter_party_id=consumer_participant_id)

    # -- provider side -------------------------------------------------- #

    async def _provider_saw(self, since: int) -> Optional[Dict[str, Any]]:
        """Read the policy the consumer just requested, from the *provider's*
        own connector. Returns ``None`` when the connector did not keep it."""
        try:
            records = await self.provider_mgmt.query_negotiations(limit=20)
        except ManagementApiError as exc:
            self._emit(f"    provider could not query its negotiations: {exc}")
            return None

        for record in records:
            if str(record.get("type") or "").upper() not in ("PROVIDER", ""):
                continue
            policy = (record.get("contractOffers") or record.get("policy")
                      or record.get("offer"))
            for candidate in _as_list(policy):
                if isinstance(candidate, dict) and candidate:
                    inner = candidate.get("policy", candidate)
                    try:
                        return offer_from_edc_policy(inner)
                    except Exception:  # noqa: BLE001
                        continue
        return None

    async def _ensure_access_policy(self) -> str:
        """Create the counter-party-scoped access policy once, then reuse it.

        Created lazily rather than in ``__init__`` because it is a network call,
        and idempotently because every round republishes.
        """
        if self._access_policy_id is None:
            policy_id = (f"bambon-access-{self.consumer_participant_id}"
                         f"-{uuid.uuid4().hex[:6]}")
            await self.provider_mgmt.ensure("policy", access_policy_definition(
                policy_id=policy_id,
                participant_ids=[self.consumer_participant_id],
                dialect=self.dialect))
            self._access_policy_id = policy_id
        return self._access_policy_id

    async def _republish(self, offer: Dict[str, Any], round_index: int) -> str:
        """Publish the provider's counter-offer as its new catalog policy.

        This is the provider's only way to make a counter-offer: EDC has no
        ContractOfferMessage the consumer would act on, but it does have a
        catalog, and the consumer refetches it every round.
        """
        suffix = f"r{round_index}-{uuid.uuid4().hex[:6]}"
        contract_policy_id = f"bambon-offer-{suffix}"
        definition_id = f"bambon-cd-{suffix}"

        await self.provider_mgmt.ensure("policy", policy_definition(
            dict(offer, **{"@type": "odrl:Agreement"}),
            policy_id=contract_policy_id, dialect=self.dialect))

        # A *separate* access policy, scoped to the participant we are bargaining
        # with. EDC asks two questions - who may see the offer (access) and what
        # they may then do with it (contract) - and reusing one document for both
        # works on upstream EDC but is rejected by Tractus-X. Scoping also stops
        # a counter-offer meant for one consumer from being visible data-space
        # wide, which would leak the terms of a bilateral negotiation.
        access_policy_id = await self._ensure_access_policy()

        await self.provider_mgmt.ensure("contract", contract_definition(
            asset_id=self.asset_id,
            access_policy_id=access_policy_id,
            contract_policy_id=contract_policy_id,
            definition_id=definition_id))

        # Retire the previous offer - on the first counter that is the seeded
        # restrictive published one - so the catalog carries exactly one current
        # offer. Delete after creating, never before: a window with no contract
        # definition would make the asset vanish from the catalog and the next
        # round would fail with "not offered".
        for old in self._previous_definitions:
            try:
                await self.provider_mgmt.delete_contract_definition(old)
            except ManagementApiError as exc:
                _logger.debug("could not retire contract definition %s: %s",
                              old, exc)
        self._previous_definitions = [definition_id]

        return contract_policy_id

    # -- data movement --------------------------------------------------- #

    async def _pull(self, outcome: ConnectorOutcome) -> None:
        """Fetch the dataset under the agreement we just reached.

        A failed pull does not turn the outcome into an error: the negotiation
        genuinely succeeded and the contract agreement id is genuine. It is the
        *claim* that weakens, so the failure is recorded on ``outcome.pull``
        instead of being raised, and callers check ``outcome.data_moved``.
        """
        if self.puller is None or not outcome.contract_agreement_id:
            return

        self._emit(f"\n  transfer (agreement {outcome.contract_agreement_id})")
        try:
            outcome.pull = await self.puller(
                asset_id=self.asset_id,
                contract_agreement_id=outcome.contract_agreement_id)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("pull raised: %s", exc)
            self._emit(f"    pull     !! {type(exc).__name__}: {exc}")
            return

        detail = getattr(outcome.pull, "reason", None)
        if getattr(outcome.pull, "ok", False):
            self._emit(f"    pull     <- HTTP "
                       f"{getattr(outcome.pull, 'status_code', '?')}, "
                       f"{getattr(outcome.pull, 'bytes_received', 0)} bytes")
        else:
            self._emit(f"    pull     !! {detail or outcome.pull}")

    # -- the run -------------------------------------------------------- #

    async def run(self) -> ConnectorOutcome:
        rounds: List[Round] = []

        for index in range(1, self.max_rounds + 1):
            rnd = Round(index=index)
            rounds.append(rnd)
            self._emit(f"\n  round {index}")

            # 1. real DSP catalog request, consumer connector -> provider
            catalog = await self.consumer_mgmt.fetch_catalog(
                counter_party_address=self.provider_protocol_url,
                counter_party_id=self.provider_participant_id)
            offer_id, published, offer_count = _find_offer(catalog, self.asset_id)
            if published is None:
                return ConnectorOutcome(
                    kind="error", rounds=rounds,
                    reason=f"asset {self.asset_id!r} not offered in the catalog")
            rnd.published_offer, rnd.published_offer_id = published, offer_id
            rnd.offers_in_catalog = offer_count
            self._emit(f"    catalog  <- {_terms(published)}")
            if offer_count != 1:
                warning = (f"catalog carried {offer_count} offers for this "
                           f"asset - negotiated against the first one")
                rnd.add_note(warning)
                self._emit(f"    catalog  !! {warning}")

            # 2. consumer BAMBON reacts to what the connector fetched
            step = self.consumer.step(published)
            if step.kind == "infeasible":
                return ConnectorOutcome(kind="infeasible", rounds=rounds,
                                        reason=step.reason)
            proposal = step.offer
            rnd.consumer_proposal = proposal
            accepting = is_agreement(proposal)
            self._emit(f"    consumer -> {_terms(proposal)}"
                       + ("   [accepting]" if accepting else ""))

            # 3. real DSP contract request, consumer connector -> provider
            #    When accepting we must quote the published offer verbatim, or
            #    EDC will terminate it like any other mismatch.
            requested = published if accepting else proposal
            try:
                negotiation_id = await self.consumer_mgmt.initiate_negotiation(
                    counter_party_address=self.provider_protocol_url,
                    counter_party_id=self.provider_participant_id,
                    offer_id=offer_id or f"urn:uuid:{uuid.uuid4()}",
                    asset_id=self.asset_id,
                    policy=_odrl_policy_for_request(requested, self.dialect))
            except ManagementApiError as exc:
                return ConnectorOutcome(kind="error", rounds=rounds,
                                        reason=f"contract request failed: {exc}")
            rnd.negotiation_id = negotiation_id

            record = await self.consumer_mgmt.await_negotiation(
                negotiation_id, timeout=self.negotiation_timeout)
            rnd.negotiation_state = _state(record)
            self._emit(f"    DSP      :: negotiation {negotiation_id} "
                       f"-> {rnd.negotiation_state}")

            # 4. finished?
            if rnd.negotiation_state == "FINALIZED":
                rnd.agreement_id = (record.get("contractAgreementId")
                                    or record.get("edc:contractAgreementId"))
                outcome = ConnectorOutcome(
                    kind="agreement", rounds=rounds,
                    agreement=proposal if accepting else published,
                    contract_agreement_id=rnd.agreement_id,
                    final_negotiation_id=negotiation_id)
                await self._pull(outcome)
                return outcome

            if accepting:
                rnd.add_note("consumer accepted but the connector rejected the "
                             "request - offer id was probably stale")
                continue

            # 5. provider BAMBON learns what was requested, from its own connector
            seen = await self._provider_saw(index)
            if seen is None:
                if not self.relay_fallback:
                    return ConnectorOutcome(
                        kind="error", rounds=rounds,
                        reason="provider connector did not expose the rejected "
                               "request and relay_fallback is off")
                seen = proposal
                rnd.relayed = True
                rnd.add_note("provider connector did not persist the rejected "
                             "policy - proposal was relayed out of band")
            else:
                rnd.provider_saw_request = True

            pstep = self.provider.step(seen)
            if pstep.kind == "infeasible":
                return ConnectorOutcome(kind="infeasible", rounds=rounds,
                                        reason=pstep.reason)
            rnd.provider_counter = pstep.offer
            self._emit(f"    provider -> {_terms(pstep.offer)}"
                       + ("   [relayed]" if rnd.relayed else "   [read from own connector]"))

            # 6. the provider's counter-offer *is* a republished catalog entry
            rnd.republished_policy_id = await self._republish(pstep.offer, index)
            self._emit(f"    provider :: republished as {rnd.republished_policy_id}")

            if is_agreement(pstep.offer):
                # Provider has accepted; the consumer will now find terms it
                # asked for in the catalog and request them verbatim next round.
                rnd.add_note("provider accepted - consumer to confirm next round")

        return ConnectorOutcome(
            kind="timeout", rounds=rounds,
            reason=f"no contract agreement within {self.max_rounds} rounds")


def _round_operand(value: Any) -> Any:
    """Shorten ``46.666666666666686`` for the transcript only.

    BAMBON converges on real values and the *published* policy keeps full
    precision - see the README caveat on continuous agreements. This is
    presentation, so nothing that ends up in a PolicyDefinition passes here.
    """
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, list):
        return [_round_operand(v) for v in value]
    return value


def _terms(offer: Dict[str, Any]) -> str:
    parts = []
    for rule_type in ("permission", "prohibition", "obligation"):
        for rule in offer.get(rule_type) or []:
            clauses = ", ".join(
                f"{strip_prefix(c['leftOperand'])} "
                f"{strip_prefix(c['operator'])} {_round_operand(c['rightOperand'])}"
                for c in rule.get("constraint", []) or [])
            parts.append(f"{rule_type} {strip_prefix(rule['action'])}: {clauses}")
    return " | ".join(parts)


async def negotiate_through_connectors(**kwargs) -> ConnectorOutcome:
    """Convenience wrapper around :class:`ConnectorNegotiation`."""
    return await ConnectorNegotiation(**kwargs).run()
