"""BAMBON <-> Eclipse Dataspace Components integration.

BAMBON negotiates ODRL policies bilaterally; EDC's contract negotiation is
take-it-or-leave-it. This package bridges the two.

There are two arrangements here, and they are not equally strong:

``two_connector_flow`` / ``connector_negotiation``
    Two local connectors bargain *through* the Dataspace Protocol. The
    consumer's proposals travel as real ``ContractRequestMessage``s and the
    provider's as republished catalog entries; the run ends with a genuine
    contract agreement and a consumer pull of the dataset under it. This is the
    demo to cite.

``service`` / ``consumer_flow``
    The two BAMBONs talk to each other over a private HTTP side channel and the
    provider then materialises the agreement for a stock ``edcpy`` negotiation.
    Simpler and useful for exercising the bargaining, but the counter-offers
    never cross DSP.

See ``README.md`` for the sequence and the standardisation caveats.
"""

from __future__ import annotations

import os
import sys

# Convenience for running from a source checkout without `pip install -e ../..`:
# put the repository's `src/` on the path if `bambon` is not importable yet.
try:  # pragma: no cover - environment dependent
    import bambon  # noqa: F401
except ImportError:  # pragma: no cover
    _repo_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    if os.path.isdir(os.path.join(_repo_src, "bambon")) and _repo_src not in sys.path:
        sys.path.insert(0, _repo_src)

__version__ = "0.1.0"

from .odrl_edc import (  # noqa: E402
    DIALECTS,
    TRACTUSX,
    VANILLA,
    Dialect,
    access_policy_definition,
    asset_definition,
    contract_definition,
    dialect_by_name,
    offer_from_edc_dataset,
    offer_from_edc_policy,
    operands_used,
    policy_definition,
    strip_prefix,
)

__all__ = [
    "DIALECTS",
    "TRACTUSX",
    "VANILLA",
    "Dialect",
    "access_policy_definition",
    "asset_definition",
    "contract_definition",
    "dialect_by_name",
    "offer_from_edc_dataset",
    "offer_from_edc_policy",
    "operands_used",
    "policy_definition",
    "strip_prefix",
]
