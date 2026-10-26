"""The BAMBON side channel: one FastAPI app, run identically by both participants.

Why a side channel at all
-------------------------
The Dataspace Protocol defines an ``OFFERED`` state for provider counter-offers,
but EDC does not implement the iterative loop: a ``ContractRequestMessage`` whose
policy differs from the catalog offer is terminated. So the bargaining has to
happen *before* the DSP negotiation, over a channel the connectors do not own.

This service is that channel. It is deliberately symmetric - the same app runs
on both sides with ``BAMBON_ROLE=provider`` or ``BAMBON_ROLE=consumer`` - so the
protocol below is a candidate DSP extension rather than a provider-side hack.

Protocol
--------
``POST /negotiation/sessions``           open a session, send the opening offer
``POST /negotiation/sessions/{id}/offers`` send a counter-offer
``GET  /negotiation/sessions/{id}``      inspect state and transcript
``GET  /healthz``

When the *provider* side reaches an ``odrl:Agreement`` it immediately
materialises it into its own connector (PolicyDefinition + ContractDefinition
scoped to the requesting participant) and returns the ids, so the consumer knows
what to look for on its next catalog fetch.

Run with::

    BAMBON_ROLE=provider uvicorn bambon_edc.service:app --port 8100
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .management import ManagementApiError, ManagementClient
from .odrl_edc import (access_policy_definition, contract_definition,
                       dialect_by_name, operands_used, policy_definition,
                       strip_prefix)
from .session import NegotiationSession, Step

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
_logger = logging.getLogger("bambon_edc.service")

ROLE = os.getenv("BAMBON_ROLE", "provider").lower()
PARTICIPANT_ID = os.getenv("EDC_CONNECTOR_PARTICIPANT_ID", ROLE)
#: Provider only: publish the agreed policy into the connector automatically.
AUTO_MATERIALISE = os.getenv("BAMBON_AUTO_MATERIALISE", "true").lower() == "true"
#: Which connector family we are publishing into: "vanilla" or "tractusx".
#: Wrong dialect means a 400 from the Management API, or a policy that is
#: accepted and silently never matches.
DIALECT = dialect_by_name(os.getenv("BAMBON_EDC_DIALECT", "vanilla"))
#: Left operands that the connector's policy engine has a function for. Empty
#: means "do not check" - see README, this is the most common demo failure.
REGISTERED_OPERANDS = {
    o.strip() for o in os.getenv("BAMBON_REGISTERED_OPERANDS", "").split(",") if o.strip()
}

app = FastAPI(
    title=f"BAMBON side channel ({ROLE})",
    description=__doc__,
    version="0.1.0",
)

_sessions: Dict[str, NegotiationSession] = {}


# --------------------------------------------------------------------------- #
# wire types                                                                   #
# --------------------------------------------------------------------------- #

class OpenSession(BaseModel):
    asset_id: str = Field(..., description="EDC asset the policy will govern")
    participant_id: str = Field(..., description="Initiator's EDC participant id")
    offer: Dict[str, Any] = Field(..., description="Opening ODRL offer")
    session_id: Optional[str] = None


class CounterOffer(BaseModel):
    offer: Dict[str, Any]


class NegotiationReply(BaseModel):
    session_id: str
    round: int
    state: str                              # counter | agreement | infeasible
    offer: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    signals: Optional[Dict[str, Any]] = None
    # Populated by the provider once the agreement is live in its connector.
    policy_id: Optional[str] = None
    contract_definition_id: Optional[str] = None
    materialisation_error: Optional[str] = None


# --------------------------------------------------------------------------- #
# endpoints                                                                    #
# --------------------------------------------------------------------------- #

@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"status": "ok", "role": ROLE, "participant_id": PARTICIPANT_ID,
            "sessions": len(_sessions)}


@app.post("/negotiation/sessions", response_model=NegotiationReply)
async def open_session(body: OpenSession) -> NegotiationReply:
    session_id = body.session_id or f"urn:uuid:{uuid.uuid4()}"
    if session_id in _sessions:
        raise HTTPException(409, f"session {session_id} already open")

    session = NegotiationSession(
        role=ROLE,
        asset_id=body.asset_id,
        counter_party_id=body.participant_id,
        session_id=session_id,
    )
    _sessions[session_id] = session
    _logger.info("[%s] opened session %s for asset %s with %s",
                 ROLE, session_id, body.asset_id, body.participant_id)

    return await _advance(session, body.offer)


@app.post("/negotiation/sessions/{session_id}/offers",
          response_model=NegotiationReply)
async def counter(session_id: str, body: CounterOffer) -> NegotiationReply:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session {session_id}")
    return await _advance(session, body.offer)


@app.get("/negotiation/sessions/{session_id}")
async def inspect(session_id: str) -> Dict[str, Any]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session {session_id}")
    return {**session.summary(),
            "agreement": session.agreement,
            "transcript": session.transcript}


@app.delete("/negotiation/sessions/{session_id}")
async def close(session_id: str) -> Dict[str, str]:
    _sessions.pop(session_id, None)
    return {"status": "closed"}


# --------------------------------------------------------------------------- #
# internals                                                                    #
# --------------------------------------------------------------------------- #

async def _advance(session: NegotiationSession,
                   incoming: Dict[str, Any]) -> NegotiationReply:
    step: Step = session.step(incoming)
    reply = NegotiationReply(
        session_id=session.session_id,
        round=step.round,
        state=step.kind,
        offer=step.offer,
        reason=step.reason,
        signals=step.signals,
    )

    if step.kind != "agreement" or ROLE != "provider" or not AUTO_MATERIALISE:
        return reply

    try:
        policy_id, definition_id = await _materialise(session, step.offer)
        reply.policy_id = policy_id
        reply.contract_definition_id = definition_id
    except Exception as exc:  # noqa: BLE001 - surfaced to the consumer verbatim
        _logger.exception("materialisation failed")
        reply.materialisation_error = f"{type(exc).__name__}: {exc}"

    return reply


async def _materialise(session: NegotiationSession,
                       agreement: Dict[str, Any]) -> tuple[str, str]:
    """Publish the agreed terms into this connector, scoped to the counter-party.

    Two definitions are created, because EDC asks two different questions:

    * the **access policy** decides who may *see* the offer in the catalog -
      here, only the participant that negotiated it, so a bilateral agreement
      does not leak to the whole data space; and
    * the **contract policy** carries the negotiated ODRL terms plus the
      dialect's non-negotiable baseline.

    Reusing one document for both roles works on upstream EDC but Tractus-X
    rejects it, hence the split.
    """
    _assert_operands_supported(agreement)

    suffix = session.session_id.rsplit(":", 1)[-1][:8]
    contract_policy_id = f"bambon-contract-{suffix}"
    access_policy_id = f"bambon-access-{suffix}"
    definition_id = f"bambon-cd-{suffix}"

    client = ManagementClient()

    await client.ensure("policy", policy_definition(
        agreement, policy_id=contract_policy_id, dialect=DIALECT))

    await client.ensure("policy", access_policy_definition(
        policy_id=access_policy_id,
        participant_ids=[session.counter_party_id],
        dialect=DIALECT,
    ))

    await client.ensure("contract", contract_definition(
        asset_id=session.asset_id,
        access_policy_id=access_policy_id,
        contract_policy_id=contract_policy_id,
        definition_id=definition_id,
    ))

    _logger.info("[provider] published %s / %s for %s",
                 contract_policy_id, definition_id, session.counter_party_id)
    return contract_policy_id, definition_id


def _assert_operands_supported(agreement: Dict[str, Any]) -> None:
    """Fail loudly if BAMBON agreed on an operand the policy engine cannot evaluate.

    EDC silently *denies* on an unregistered left operand in some versions and
    *permits* in others; either way the negotiated policy is meaningless. Set
    ``BAMBON_REGISTERED_OPERANDS`` to the operands your connector extension
    registers via ``PolicyEngine#registerFunction``.
    """
    if not REGISTERED_OPERANDS:
        return
    used = set(operands_used(agreement))
    # Baseline operands come from the dialect, not from BAMBON, and the
    # connector family that mandates them also implements them.
    supplied = {strip_prefix(c["leftOperand"]) for c in DIALECT.baseline}
    missing = sorted(used - REGISTERED_OPERANDS - supplied - {"participantId"})
    if missing:
        raise ManagementApiError(
            "PRECHECK", "policy-engine", 400,
            f"no policy function registered for left operand(s): {missing}. "
            f"Registered: {sorted(REGISTERED_OPERANDS)}",
        )
