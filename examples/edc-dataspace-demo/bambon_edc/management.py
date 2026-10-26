"""A thin async client for the EDC Management API.

``edcpy`` covers the consumer side of the data space beautifully (catalog,
contract negotiation, transfer) but the *provider* side of this demo needs to
create policy definitions and contract definitions on the fly, which is outside
its remit. This module fills that gap with a dependency-light httpx client.

Only the handful of endpoints the demo touches are implemented.
"""

from __future__ import annotations

import logging
import re
import os
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

EDC_NS = "https://w3id.org/edc/v0.0.1/ns/"
_CTX = {"@context": {"@vocab": EDC_NS,
                     "odrl": "http://www.w3.org/ns/odrl/2/"}}


class ManagementApiError(RuntimeError):
    """Non-2xx from the EDC Management API, with the body attached."""

    def __init__(self, method: str, url: str, status: int, body: str):
        super().__init__(f"{method} {url} -> {status}: {body}")
        self.status = status
        self.body = body


class ManagementClient:
    """Minimal async wrapper over ``/management/v3/*``."""

    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_key_header: str = "X-API-Key",
                 timeout: float = 30.0):
        self.base_url = (base_url or self._base_url_from_env()).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("EDC_CONNECTOR_API_KEY")
        self.api_key_header = api_key_header
        self.timeout = timeout
        # EDC versions its Management API per resource, not globally: a build
        # that serves /v3/assets may still serve /v2/policydefinitions and
        # 404 the v3 spelling. The working version per resource is discovered
        # on first use and cached here rather than hard-coded.
        self._api_version: Dict[str, str] = {}

    # -- construction -------------------------------------------------- #

    @staticmethod
    def _base_url_from_env() -> str:
        """Reuse edcpy's environment variables so both halves stay in sync."""
        scheme = os.getenv("EDC_CONNECTOR_SCHEME", "http")
        host = os.getenv("EDC_CONNECTOR_HOST", "localhost")
        port = os.getenv("EDC_CONNECTOR_MANAGEMENT_PORT", "9193")
        path = os.getenv("EDC_CONNECTOR_MANAGEMENT_PATH", "/management")
        return f"{scheme}://{host}:{port}{path}"

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = self.api_key
        return headers

    # -- transport ----------------------------------------------------- #

    async def _request(self, method: str, path: str,
                       payload: Optional[Dict[str, Any]] = None) -> Any:
        import httpx  # imported lazily so odrl_edc stays importable standalone

        async def attempt(candidate: str):
            url = f"{self.base_url}{candidate}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return url, await client.request(
                    method, url, json=payload, headers=self._headers)

        url, response = await attempt(self._versioned(path))

        # A 404 on a versioned path means this connector serves that resource
        # under a different version, not that the resource is missing. Try the
        # other known versions once, newest first, and remember what worked.
        if response.status_code == 404:
            resource = self._resource_of(path)
            for version in self._API_VERSIONS:
                if resource and version == self._api_version.get(resource):
                    continue
                candidate = self._with_version(path, version)
                if candidate == url[len(self.base_url):]:
                    continue
                retry_url, retry = await attempt(candidate)
                if retry.status_code != 404:
                    if resource:
                        self._api_version[resource] = version
                        _logger.info("management API: %s served under %s",
                                     resource, version)
                    url, response = retry_url, retry
                    break

        if response.status_code >= 400:
            raise ManagementApiError(method, url, response.status_code, response.text)

        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    #: Management API versions to try, newest first.
    _API_VERSIONS = ("v3", "v2")

    @staticmethod
    def _resource_of(path: str) -> str:
        parts = [p for p in path.split("/") if p]
        return parts[1] if len(parts) > 1 and re.fullmatch(r"v\d+", parts[0]) else ""

    @staticmethod
    def _with_version(path: str, version: str) -> str:
        return re.sub(r"^/v\d+/", f"/{version}/", path)

    def _versioned(self, path: str) -> str:
        """Rewrite ``path`` to whichever version this connector answered on."""
        resource = self._resource_of(path)
        known = self._api_version.get(resource)
        return self._with_version(path, known) if known else path

    # -- endpoints ----------------------------------------------------- #

    async def create_asset(self, body: Dict[str, Any]) -> Any:
        return await self._request("POST", "/v3/assets", body)

    async def create_policy_definition(self, body: Dict[str, Any]) -> Any:
        return await self._request("POST", "/v3/policydefinitions", body)

    async def create_contract_definition(self, body: Dict[str, Any]) -> Any:
        return await self._request("POST", "/v3/contractdefinitions", body)

    async def list_contract_definitions(self, limit: int = 200) -> List[Dict[str, Any]]:
        # A bare {} is rejected with 400 by the v2 query endpoints: they expect
        # an expandable JSON-LD QuerySpec, not an empty object.
        return await self._request("POST", "/v3/contractdefinitions/request", {
            **_CTX, "@type": "QuerySpec", "offset": 0, "limit": limit,
        }) or []

    async def delete_contract_definition(self, definition_id: str) -> Any:
        return await self._request("DELETE", f"/v3/contractdefinitions/{definition_id}")

    async def delete_policy_definition(self, policy_id: str) -> Any:
        return await self._request("DELETE", f"/v3/policydefinitions/{policy_id}")

    # -- catalog (DSP) -------------------------------------------------- #

    async def fetch_catalog(self, *, counter_party_address: str,
                            counter_party_id: str,
                            protocol: str = "dataspace-protocol-http",
                            limit: int = 200) -> Dict[str, Any]:
        """Ask *our* connector to fetch a peer's catalog over the Dataspace
        Protocol. This is a real connector-to-connector exchange, not a read of
        the peer's Management API."""
        return await self._request("POST", "/v3/catalog/request", {
            **_CTX,
            "counterPartyAddress": counter_party_address,
            "counterPartyId": counter_party_id,
            "protocol": protocol,
            "querySpec": {"@type": "QuerySpec", "offset": 0, "limit": limit},
        })

    # -- contract negotiations (DSP) ------------------------------------ #

    async def initiate_negotiation(self, *,
                                   counter_party_address: str,
                                   counter_party_id: str,
                                   offer_id: str,
                                   asset_id: str,
                                   policy: Dict[str, Any],
                                   protocol: str = "dataspace-protocol-http",
                                   ) -> str:
        """Send a real ``ContractRequestMessage`` to the counter-party connector.

        ``policy`` must be the ODRL policy exactly as we want to propose it. EDC
        will forward it over DSP; the provider accepts only if it matches one of
        its published offers, and terminates otherwise. That termination is the
        signal BAMBON treats as a rejection.
        """
        body = {
            **_CTX,
            "@type": "ContractRequest",
            "counterPartyAddress": counter_party_address,
            "counterPartyId": counter_party_id,
            "protocol": protocol,
            "policy": {
                **policy,
                "@id": offer_id,
                "@type": "odrl:Offer",
                "assigner": counter_party_id,
                "target": asset_id,
            },
        }
        result = await self._request("POST", "/v3/contractnegotiations", body)
        return (result or {}).get("@id", "")

    async def get_negotiation(self, negotiation_id: str) -> Dict[str, Any]:
        return await self._request(
            "GET", f"/v3/contractnegotiations/{negotiation_id}") or {}

    async def query_negotiations(self,
                                 filter_expression: Optional[List[Dict[str, Any]]] = None,
                                 limit: int = 200) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {**_CTX, "@type": "QuerySpec", "offset": 0,
                                "limit": limit,
                                "sortField": "createdAt", "sortOrder": "DESC"}
        if filter_expression:
            body["filterExpression"] = filter_expression
        return await self._request(
            "POST", "/v3/contractnegotiations/request", body) or []

    async def await_negotiation(self, negotiation_id: str, *,
                                timeout: float = 60.0,
                                interval: float = 1.0) -> Dict[str, Any]:
        """Poll until the negotiation leaves the in-flight states.

        Terminal states are FINALIZED (agreement reached) and TERMINATED
        (rejected) - both are informative to BAMBON.
        """
        import asyncio

        deadline = asyncio.get_event_loop().time() + timeout
        record: Dict[str, Any] = {}
        while asyncio.get_event_loop().time() < deadline:
            record = await self.get_negotiation(negotiation_id)
            state = str(record.get("state") or record.get("edc:state") or "")
            if state in ("FINALIZED", "TERMINATED", "ERROR"):
                return record
            # Some connector builds do not terminate a refused counter-offer.
            # The provider answers with a ContractAgreementMessage carrying its
            # own published policy, the consumer refuses it ("Policy in the
            # contract agreement is not equal to the one in the contract
            # offer"), and the negotiation then sits in REQUESTED forever.
            # That refusal is a rejection, so report it instead of waiting out
            # the timeout.
            detail = (record.get("errorDetail") or record.get("edc:errorDetail")
                      or "")
            if detail:
                record["_refused"] = True
                return record
            await asyncio.sleep(interval)
        record["_timeout"] = True
        return record

    # -- convenience --------------------------------------------------- #

    async def ensure(self, kind: str, body: Dict[str, Any]) -> Any:
        """Create a resource, tolerating a 409 from a previous demo run."""
        creators = {
            "asset": self.create_asset,
            "policy": self.create_policy_definition,
            "contract": self.create_contract_definition,
        }
        try:
            return await creators[kind](body)
        except ManagementApiError as exc:
            if exc.status == 409:
                _logger.info("%s %s already exists, reusing", kind, body.get("@id"))
                return {"@id": body.get("@id"), "reused": True}
            raise
