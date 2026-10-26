"""The consumer pulls the dataset under the agreement BAMBON negotiated.

Why this module exists
----------------------
Reaching ``FINALIZED`` proves the two connectors agreed on terms. It does not
prove any data moved. The claim worth making is the stronger one - *the
negotiated agreement is the one the data plane honoured* - and that needs a
transfer process bound to the contract agreement id that BAMBON's own
``ContractRequestMessage`` produced.

So this deliberately does **not** call ``edcpy``'s ``run_negotiation_flow()``.
That helper would fetch the catalog and negotiate all over again, and the
agreement it produced would be a second, unrelated one - a demo that pulled data
under *that* id would prove nothing about the negotiation. Instead we build
``TransferProcessDetails`` around our own agreement id and call only the
transfer half of edcpy.

The HTTP-pull handshake, in the order it happens
------------------------------------------------
1. The consumer subscribes to its RabbitMQ pull queue. This must happen *before*
   the transfer starts, or the EndpointDataReference arrives with nobody
   listening and is lost.
2. ``POST /v2/transferprocesses`` with ``transferType: HttpData-PULL``, quoting
   the contract agreement id.
3. The provider's data plane mints a token and POSTs the EDR to the consumer
   connector's ``edc.receiver.http.endpoint``, which is edcpy's
   ``run-http-backend`` service; it republishes the EDR to RabbitMQ.
4. We take the EDR off the queue and GET its endpoint with the auth token. That
   request is proxied by the provider's data plane to the real backend.

Step 3 is why the compose stack has a broker and a ``consumer-backend``: in EDC
the pull credentials are pushed to the consumer out-of-band, not returned from
the Management API call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlparse

_logger = logging.getLogger(__name__)

#: Bytes of the response body to keep on the result. The point is evidence that
#: real content arrived, not to hold a dataset in memory.
PREVIEW_BYTES = 600


class TransferUnavailable(RuntimeError):
    """``edcpy`` (or its messaging extra) is not importable in this process."""


@dataclass
class PullResult:
    """What the consumer got, and under which agreement.

    ``kind`` is one of:

    ``pulled``
        Data arrived. ``status_code``, ``bytes_received`` and ``preview`` are set.
    ``no_edr``
        The transfer process was created but no EndpointDataReference reached the
        broker before the timeout. Usually a missing data plane registration or a
        ``consumer-backend`` that is not running.
    ``error``
        Something else failed; ``reason`` says what.
    """

    kind: str
    contract_agreement_id: Optional[str] = None
    transfer_process_id: Optional[str] = None
    endpoint: Optional[str] = None
    status_code: Optional[int] = None
    bytes_received: int = 0
    preview: str = ""
    record_count: Optional[int] = None
    reason: Optional[str] = None
    request_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.kind == "pulled" and (self.status_code or 0) < 400


def _summarise(body: bytes) -> tuple[str, Optional[int]]:
    """A short preview plus, when the payload is a JSON collection, its size.

    The record count is what makes the demo output convincing: "1 891 bytes"
    could be an error page, "273 observations" could not.
    """
    text = body.decode("utf-8", errors="replace")
    count: Optional[int] = None
    try:
        parsed = json.loads(text)
    except ValueError:
        return text[:PREVIEW_BYTES], None

    if isinstance(parsed, list):
        count = len(parsed)
    elif isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                count = len(value)
                break
    return json.dumps(parsed)[:PREVIEW_BYTES], count


@asynccontextmanager
async def _pull_consumer(client: Any, timeout: float):
    """edcpy's ``pull_consumer`` without the bug that hides its own failures.

    ``with_consumer_messaging_app`` has a ``return`` inside its ``finally``.
    Python discards an exception that is propagating when the ``finally``
    returns, so a messaging app that fails to start surfaces only as
    ``RuntimeError: generator didn't yield`` with the cause destroyed. This
    does the same work - same queue config, same handler, same startup call -
    and lets the real exception through.
    """
    from edcpy.message_handler import (MessageConsumer, create_handler_function,
                                       create_message_handler)
    from edcpy.messaging import HttpPullMessage, start_consumer_messaging_app

    queue_config = client.queue_factory.create_pull_queue_config(
        consumer_id=client.consumer_id, provider_host=None,
        exclusive=False, auto_delete=True)
    handler = create_message_handler(HttpPullMessage, auto_acknowledge=False)

    messaging_app = await start_consumer_messaging_app(
        queue_config, create_handler_function(handler), config=client.config)
    try:
        yield MessageConsumer(handler, timeout)
    finally:
        broker = getattr(messaging_app, "broker", None)
        for name in ("close", "stop"):     # faststream renamed this across versions
            closer = getattr(broker, name, None)
            if closer is not None:
                try:
                    await closer()
                except Exception:          # noqa: BLE001 - shutdown only
                    _logger.warning("could not close the pull broker", exc_info=True)
                break


async def pull_dataset(*,
                       asset_id: str,
                       contract_agreement_id: str,
                       provider_protocol_url: str,
                       provider_participant_id: str,
                       timeout: int = 120,
                       http_timeout: float = 60.0,
                       ) -> PullResult:
    """Run one consumer-pull transfer for ``contract_agreement_id``.

    Reads its connector and broker configuration from the environment, exactly
    as ``edcpy`` does, so the caller needs no extra wiring: ``EDC_CONNECTOR_*``
    for the Management API and ``EDC_RABBIT_URL`` for the EDR queue.
    """
    try:
        import httpx
        from edcpy.edc_api import ConnectorController, TransferProcessDetails
        from edcpy.messaging import MessagingClient
    except ImportError as exc:  # noqa: BLE001
        raise TransferUnavailable(
            f"cannot run a transfer without edcpy and httpx installed: {exc}"
        ) from exc

    details = TransferProcessDetails(
        asset_id=asset_id,
        contract_agreement_id=contract_agreement_id,
        counter_party_protocol_url=provider_protocol_url,
        counter_party_connector_id=provider_participant_id,
    )

    controller = ConnectorController()
    client = MessagingClient(consumer_id=f"bambon-pull-{uuid.uuid4().hex[:8]}")

    # provider_host=None subscribes to `http.pull.#`, i.e. every pull EDR this
    # consumer receives. Narrowing it to the provider host would be tidier but
    # the routing key is built from the *data plane's* hostname, which need not
    # equal the protocol URL's host in a real deployment.
    result = PullResult(kind="error",
                        contract_agreement_id=contract_agreement_id)

    # edcpy's pull_consumer swallows a broker start-up failure and returns
    # without yielding, which surfaces only as "generator didn't yield". Check
    # the broker first so the real reason is reported instead.
    rabbit_url = os.getenv("EDC_RABBIT_URL")
    if not rabbit_url:
        result.reason = ("EDC_RABBIT_URL is not set; the EndpointDataReference "
                         "is delivered over the broker, so the pull cannot run")
        return result
    try:
        import aio_pika
        connection = await aio_pika.connect_robust(rabbit_url, timeout=10)
        await connection.close()
    except Exception as exc:                       # noqa: BLE001 - reported
        result.reason = f"broker {rabbit_url} unreachable: {exc!r}"
        _logger.error(result.reason)
        return result

    try:
        async with _pull_consumer(client, timeout) as consumer:
            transfer_process_id = await controller.run_transfer_flow(
                transfer_details=details, is_provider_push=False)
            result.transfer_process_id = transfer_process_id
            _logger.info("transfer process %s created for agreement %s",
                         transfer_process_id, contract_agreement_id)

            try:
                async with consumer.wait_for_message(
                        expected_id=transfer_process_id,
                        timeout=timeout) as message:
                    request_args = dict(message.request_args)
                    result.endpoint = message.endpoint
                    # The auth token is a bearer credential; keep it out of the
                    # transcript that gets pasted into papers and issues.
                    result.request_args = {
                        k: v for k, v in request_args.items() if k != "headers"}

                    async with httpx.AsyncClient(timeout=http_timeout) as http:
                        response = await http.request(**request_args)

                    body = response.content or b""
                    preview, count = _summarise(body)
                    result.kind = "pulled"
                    result.status_code = response.status_code
                    result.bytes_received = len(body)
                    result.preview = preview
                    result.record_count = count
            except asyncio.TimeoutError:
                result.kind = "no_edr"
                result.reason = (
                    f"no EndpointDataReference within {timeout}s. Check that "
                    "consumer-backend is running and that "
                    "edc.receiver.http.endpoint points at it.")
    except TransferUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        result.kind = "error"
        result.reason = f"{type(exc).__name__}: {exc}"
        _logger.warning("pull failed: %s", result.reason)

    return result


def require_edcpy() -> None:
    """Raise :class:`TransferUnavailable` now rather than after a negotiation.

    Discovering that the transfer half is unusable *after* six rounds of
    bargaining is a waste of everyone's time, so callers check up front.
    """
    try:
        import httpx  # noqa: F401
        from edcpy.edc_api import ConnectorController  # noqa: F401
        from edcpy.messaging import MessagingClient  # noqa: F401
    except ImportError as exc:  # noqa: BLE001
        raise TransferUnavailable(
            f"cannot run a transfer without edcpy and httpx installed: {exc}"
        ) from exc


def make_puller(*,
                provider_protocol_url: str,
                provider_participant_id: str,
                timeout: int = 120):
    """Bind the transport details, leaving a callable the negotiation can call.

    :class:`~bambon_edc.connector_negotiation.ConnectorNegotiation` takes a
    ``puller`` so that the negotiation stays testable without a data plane: the
    dry-run harness passes a stub with this same signature.

    Raises :class:`TransferUnavailable` immediately if ``edcpy`` is missing.
    """
    require_edcpy()

    async def puller(*, asset_id: str, contract_agreement_id: str) -> PullResult:
        return await pull_dataset(
            asset_id=asset_id,
            contract_agreement_id=contract_agreement_id,
            provider_protocol_url=provider_protocol_url,
            provider_participant_id=provider_participant_id,
            timeout=timeout,
        )

    return puller


def describe(result: PullResult) -> str:
    """One line for the demo transcript."""
    if result.kind == "pulled":
        records = (f", {result.record_count} records"
                   if result.record_count is not None else "")
        return (f"HTTP {result.status_code} from the provider data plane, "
                f"{result.bytes_received} bytes{records}")
    if result.kind == "no_edr":
        return f"transfer started but no pull credentials arrived: {result.reason}"
    return f"pull failed: {result.reason}"
