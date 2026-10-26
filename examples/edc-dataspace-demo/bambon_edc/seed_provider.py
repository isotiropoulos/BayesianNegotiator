"""Seed the provider connector with the asset and its *published* (restrictive)
policy - the state of the world before any BAMBON negotiation happens.

Run inside the provider's network::

    python -m bambon_edc.seed_provider
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from .management import ManagementClient
from .odrl_edc import (access_policy_definition, asset_definition,
                       contract_definition, dialect_by_name, policy_definition)
from .profiles import (ASSET_ID, PROVIDER_INIT_OFFER, PUBLISHED_ACCESS_POLICY_ID,
                       PUBLISHED_DEFINITION_ID, PUBLISHED_POLICY_ID)

BACKEND_URL = os.getenv("PROVIDER_BACKEND_URL", "http://provider-backend:8080/data")
DIALECT = dialect_by_name(os.getenv("BAMBON_EDC_DIALECT", "vanilla"))


async def seed() -> None:
    client = ManagementClient()

    asset = asset_definition(asset_id=ASSET_ID, base_url=BACKEND_URL,
                             name="Climate observations 2026")
    await client.ensure("asset", asset)
    print(f"asset          {ASSET_ID}")

    # Catalog visibility: open, so any participant can discover the offer and
    # find out it does not suit them. That discovery is what triggers BAMBON.
    await client.ensure("policy", access_policy_definition(
        policy_id=PUBLISHED_ACCESS_POLICY_ID, dialect=DIALECT))
    print(f"access policy  {PUBLISHED_ACCESS_POLICY_ID}")

    # The provider's opening offer doubles as its published contract policy.
    published = dict(PROVIDER_INIT_OFFER, **{"@type": "odrl:Agreement"})
    policy = policy_definition(published, policy_id=PUBLISHED_POLICY_ID,
                               dialect=DIALECT)
    await client.ensure("policy", policy)
    print(f"use policy     {PUBLISHED_POLICY_ID}")
    print(json.dumps(policy["policy"], indent=2))

    await client.ensure("contract", contract_definition(
        asset_id=ASSET_ID,
        access_policy_id=PUBLISHED_ACCESS_POLICY_ID,
        contract_policy_id=PUBLISHED_POLICY_ID,
        definition_id=PUBLISHED_DEFINITION_ID,
    ))
    print(f"contract def   {PUBLISHED_DEFINITION_ID}")
    print("\nProvider seeded. This is the take-it-or-leave-it offer a stock "
          "EDC consumer would be stuck with.")


if __name__ == "__main__":
    asyncio.run(seed())
    sys.exit(0)
