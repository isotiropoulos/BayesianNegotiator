"""Transport-agnostic BAMBON negotiation sessions.

The bilateral loop lives here, deliberately free of HTTP and of EDC, so that it
can be exercised offline in tests and reused unchanged behind the FastAPI
side-channel in ``service.py``.

Both participants are symmetric: each holds its own ``Bambon`` with its own
``init_offer`` and ``hard_policy``, and each side's ``step()`` is one call to
``Bambon.observe``. The only asymmetry is who speaks first.
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .profiles import BAMBON_PROFILE, profile_for

_logger = logging.getLogger(__name__)

# ``Bambon`` and ``NegotiationInfeasible`` are imported lazily so that importing
# this module does not require numpy/scipy to be present (useful for the
# normalisation-only tests).
_BAMBON = None
_INFEASIBLE = None


def _bambon_classes():
    global _BAMBON, _INFEASIBLE
    if _BAMBON is None:
        from bambon.negotiator import Bambon
        from bambon.models.models import NegotiationInfeasible
        _BAMBON, _INFEASIBLE = Bambon, NegotiationInfeasible
    return _BAMBON, _INFEASIBLE


AGREEMENT_TYPES = ("odrl:Agreement", "Agreement")


def is_agreement(offer: Dict[str, Any]) -> bool:
    return offer.get("@type") in AGREEMENT_TYPES


@dataclass
class Step:
    """One turn of the negotiation."""

    kind: str                       # "counter" | "agreement" | "infeasible"
    offer: Optional[Dict[str, Any]] = None
    round: int = 0
    reason: Optional[str] = None
    signals: Optional[Dict[str, Any]] = None

    @property
    def done(self) -> bool:
        return self.kind != "counter"


@dataclass
class NegotiationSession:
    """A single participant's half of one negotiation.

    Attributes
    ----------
    role:
        ``"provider"`` or ``"consumer"``. Selects the profile from
        ``profiles.ROLES``.
    counter_party_id:
        Participant id of the other side; recorded so the provider can scope
        the materialised policy to exactly this consumer.
    """

    role: str
    asset_id: str
    counter_party_id: str = ""
    session_id: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    profile: Dict[str, Any] = field(default_factory=lambda: dict(BAMBON_PROFILE))
    created_at: float = field(default_factory=time.time)

    agent: Any = field(init=False, default=None)
    init_offer: Dict[str, Any] = field(init=False, default_factory=dict)
    transcript: List[Dict[str, Any]] = field(init=False, default_factory=list)
    agreement: Optional[Dict[str, Any]] = field(init=False, default=None)
    terminated: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        Bambon, _ = _bambon_classes()
        init_offer, hard_policy = profile_for(self.role)
        self.init_offer = init_offer
        self.agent = Bambon(
            name=f"{self.role}:{self.session_id[-8:]}",
            init_offer=copy.deepcopy(init_offer),
            hard_policy=hard_policy,
            **self.profile,
        )

    # -- the loop ------------------------------------------------------ #

    def step(self, incoming: Dict[str, Any]) -> Step:
        """Observe the counter-party's offer and produce our reply."""
        _, NegotiationInfeasible = _bambon_classes()

        if self.agreement is not None:
            return Step(kind="agreement", offer=self.agreement,
                        round=self.agent.rounds_seen)
        if self.terminated:
            return Step(kind="infeasible", reason=self.terminated,
                        round=self.agent.rounds_seen)

        try:
            signals, response = self.agent.observe(copy.deepcopy(incoming))
        except NegotiationInfeasible as exc:
            self.terminated = str(exc)
            _logger.info("[%s] infeasible: %s", self.role, exc)
            return Step(kind="infeasible", reason=str(exc),
                        round=self.agent.rounds_seen)

        self.transcript.append({
            "round": self.agent.rounds_seen,
            "received": incoming,
            "sent": response,
        })

        if is_agreement(response):
            self.agreement = response
            _logger.info("[%s] agreement after %d rounds",
                         self.role, self.agent.rounds_seen)
            return Step(kind="agreement", offer=response,
                        round=self.agent.rounds_seen, signals=_jsonable(signals))

        return Step(kind="counter", offer=response,
                    round=self.agent.rounds_seen, signals=_jsonable(signals))

    # -- introspection -------------------------------------------------- #

    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "role": self.role,
            "asset_id": self.asset_id,
            "counter_party_id": self.counter_party_id,
            "rounds": self.agent.rounds_seen,
            "state": ("agreed" if self.agreement
                      else "terminated" if self.terminated else "negotiating"),
            "terminated_reason": self.terminated,
        }


def _jsonable(signals: Any) -> Optional[Dict[str, Any]]:
    """``Bambon`` keys signals by ``(rule_type, action)`` tuples, which JSON
    cannot encode. Flatten to ``"permission/use"`` strings."""
    if not signals:
        return None
    out: Dict[str, Any] = {}
    for key, value in signals.items():
        label = "/".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        out[label] = list(value) if isinstance(value, tuple) else value
    return out


# --------------------------------------------------------------------------- #
# drivers                                                                      #
# --------------------------------------------------------------------------- #

class Party(Protocol):
    """Anything that can answer an offer: a local session or an HTTP peer."""

    def respond(self, offer: Dict[str, Any]) -> Step: ...


class LocalParty:
    """Adapts a :class:`NegotiationSession` to the :class:`Party` protocol."""

    def __init__(self, session: NegotiationSession):
        self.session = session

    def respond(self, offer: Dict[str, Any]) -> Step:
        return self.session.step(offer)


class HttpParty:
    """A participant's BAMBON reached over the side channel.

    Same ``respond`` contract as :class:`LocalParty`, so :func:`drive` does not
    care whether a counter-party is in-process or across the network. Using two
    of these lets a neutral orchestrator relay between two remote participants,
    with neither BAMBON running in the caller's process - which is what an
    honest symmetric demo requires.
    """

    def __init__(self, base_url: str, asset_id: str, participant_id: str,
                 timeout: float = 60.0, label: str = ""):
        self.base_url = base_url.rstrip("/")
        self.asset_id = asset_id
        self.participant_id = participant_id
        self.timeout = timeout
        self.label = label or self.base_url
        self.session_id: Optional[str] = None
        self.policy_id: Optional[str] = None
        self.contract_definition_id: Optional[str] = None
        self.materialisation_error: Optional[str] = None

    def respond(self, offer: Dict[str, Any]) -> Step:
        import httpx

        if self.session_id is None:
            url = f"{self.base_url}/negotiation/sessions"
            payload: Dict[str, Any] = {
                "asset_id": self.asset_id,
                "participant_id": self.participant_id,
                "offer": offer,
            }
        else:
            url = f"{self.base_url}/negotiation/sessions/{self.session_id}/offers"
            payload = {"offer": offer}

        response = httpx.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()

        self.session_id = body["session_id"]
        if body.get("materialisation_error"):
            self.materialisation_error = body["materialisation_error"]
            _logger.warning("[%s] could not publish the agreement: %s",
                            self.label, self.materialisation_error)
        self.policy_id = body.get("policy_id") or self.policy_id
        self.contract_definition_id = (
            body.get("contract_definition_id") or self.contract_definition_id)

        return Step(kind=body["state"], offer=body.get("offer"),
                    round=body.get("round", 0), reason=body.get("reason"),
                    signals=body.get("signals"))

    def health(self) -> Dict[str, Any]:
        import httpx
        return httpx.get(f"{self.base_url}/healthz", timeout=5.0).json()


@dataclass
class Outcome:
    kind: str                                   # "agreement" | "infeasible" | "timeout"
    agreement: Optional[Dict[str, Any]] = None
    rounds: int = 0
    reason: Optional[str] = None
    settled_by: Optional[str] = None            # which side declared agreement


def drive(initiator: NegotiationSession,
          responder: Party,
          *,
          max_rounds: int = 60,
          on_round: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
          ) -> Outcome:
    """Run the bilateral loop, alternating ``initiator`` and ``responder``.

    ``initiator`` opens with its own ``init_offer``. Whichever side first emits
    an ``odrl:Agreement`` ends the negotiation; that agreement is what gets
    materialised into EDC.
    """
    offer = copy.deepcopy(initiator.init_offer)
    if on_round:
        on_round(0, initiator.role, offer)

    for round_index in range(1, max_rounds + 1):
        remote = responder.respond(offer)
        if on_round and remote.offer:
            on_round(round_index, "remote", remote.offer)
        if remote.kind == "infeasible":
            return Outcome(kind="infeasible", rounds=round_index,
                           reason=remote.reason, settled_by="remote")
        if remote.kind == "agreement":
            return Outcome(kind="agreement", agreement=remote.offer,
                           rounds=round_index, settled_by="remote")

        local = initiator.step(remote.offer)
        if on_round and local.offer:
            on_round(round_index, initiator.role, local.offer)
        if local.kind == "infeasible":
            return Outcome(kind="infeasible", rounds=round_index,
                           reason=local.reason, settled_by=initiator.role)
        if local.kind == "agreement":
            return Outcome(kind="agreement", agreement=local.offer,
                           rounds=round_index, settled_by=initiator.role)

        offer = local.offer

    return Outcome(kind="timeout", rounds=max_rounds,
                   reason=f"no agreement within {max_rounds} rounds")


def relay(first: Party,
          second: Party,
          opening_offer: Dict[str, Any],
          *,
          max_rounds: int = 60,
          first_label: str = "first",
          second_label: str = "second",
          on_round: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
          ) -> Outcome:
    """Relay offers between two parties, holding no BAMBON of our own.

    :func:`drive` keeps one half of the negotiation in the caller's process.
    ``relay`` keeps neither: it opens with ``opening_offer`` and passes each
    reply to the other side, so both participants' beliefs live entirely in
    their own services. Use this when the point is to show two independent
    participants negotiating rather than to act as one of them.
    """
    offer = copy.deepcopy(opening_offer)
    if on_round:
        on_round(0, "opening", offer)

    parties = ((first, first_label), (second, second_label))

    for round_index in range(1, max_rounds + 1):
        for party, label in parties:
            step = party.respond(offer)
            if on_round and step.offer:
                on_round(round_index, label, step.offer)

            if step.kind == "infeasible":
                return Outcome(kind="infeasible", rounds=round_index,
                               reason=step.reason, settled_by=label)
            if step.kind == "agreement":
                return Outcome(kind="agreement", agreement=step.offer,
                               rounds=round_index, settled_by=label)
            if step.offer is None:
                return Outcome(kind="infeasible", rounds=round_index,
                               reason=f"{label} returned no offer",
                               settled_by=label)
            offer = step.offer

    return Outcome(kind="timeout", rounds=max_rounds,
                   reason=f"no agreement within {max_rounds} rounds")


def negotiate_offline(asset_id: str, *, max_rounds: int = 60,
                      on_round=None) -> Outcome:
    """Run both halves in-process. Used by the tests and by ``--offline``."""
    consumer = NegotiationSession(role="consumer", asset_id=asset_id,
                                  counter_party_id="provider")
    provider = NegotiationSession(role="provider", asset_id=asset_id,
                                  counter_party_id="consumer")
    return drive(consumer, LocalParty(provider),
                 max_rounds=max_rounds, on_round=on_round)
