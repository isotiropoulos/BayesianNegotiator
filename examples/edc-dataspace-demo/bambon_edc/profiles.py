"""Negotiation profiles for the two demo participants.

The scenario is deliberately one that EDC *cannot* resolve on its own: the
provider publishes a research-only, 30-day, EU-region policy and the consumer
needs commercial use for a year. A stock EDC contract negotiation terminates
here, because the consumer may only request the catalog offer verbatim.

Both sides run BAMBON symmetrically, each with its own hard policy that it will
never concede past.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

ASSET_ID = "climate-observations-2026"

# --------------------------------------------------------------------------- #
# Management-API ids for the provider's *published* (pre-negotiation) offer     #
# --------------------------------------------------------------------------- #
#
# These live here rather than in ``seed_provider`` because two modules need
# them: the seeder creates them, and ``connector_negotiation`` retires the
# contract definition the first time the provider counter-offers. Without that
# retirement the catalog would carry both the restrictive published offer and
# every counter-offer at once, and ``odrl:hasPolicy`` would become a pile of
# stale terms with no defined order - the consumer must see exactly one current
# offer per fetch for the negotiation to mean anything.

PUBLISHED_POLICY_ID = "published-restrictive"
PUBLISHED_ACCESS_POLICY_ID = "published-access"
PUBLISHED_DEFINITION_ID = "cd-published-restrictive"

# --------------------------------------------------------------------------- #
# Provider                                                                     #
# --------------------------------------------------------------------------- #

PROVIDER_INIT_OFFER: Dict[str, Any] = {
    "@id": "urn:offer:provider:opening",
    "@type": "odrl:Offer",
    "permission": [{
        "action": "use",
        "constraint": [
            {"leftOperand": "odrl:purpose", "operator": "odrl:eq",
             "rightOperand": "research"},
            {"leftOperand": "odrl:elapsedTime", "operator": "odrl:lteq",
             "rightOperand": 30},
            {"leftOperand": "odrl:spatial", "operator": "odrl:isAnyOf",
             "rightOperand": ["EU"]},
        ],
    }],
    "prohibition": [{
        "action": "distribute",
        "constraint": [
            {"leftOperand": "odrl:recipient", "operator": "odrl:isAnyOf",
             "rightOperand": ["third-party", "public"]},
        ],
    }],
}

#: Terms the provider will never trade away, whatever the beliefs say.
PROVIDER_HARD_POLICY: Dict[str, Any] = {
    "permission": [{
        "action": "use",
        "constraint": [
            {"leftOperand": "odrl:elapsedTime", "operator": "odrl:lteq",
             "rightOperand": 180},
            {"leftOperand": "odrl:spatial", "operator": "odrl:isAnyOf",
             "rightOperand": ["EU", "EEA"]},
        ],
    }],
    "prohibition": [{
        "action": "distribute",
        "constraint": [
            {"leftOperand": "odrl:recipient", "operator": "odrl:isAnyOf",
             "rightOperand": ["public"]},
        ],
    }],
}

# --------------------------------------------------------------------------- #
# Consumer                                                                     #
# --------------------------------------------------------------------------- #

CONSUMER_INIT_OFFER: Dict[str, Any] = {
    "@id": "urn:offer:consumer:opening",
    "@type": "odrl:Offer",
    "permission": [{
        "action": "use",
        "constraint": [
            {"leftOperand": "odrl:purpose", "operator": "odrl:eq",
             "rightOperand": "commercial"},
            {"leftOperand": "odrl:elapsedTime", "operator": "odrl:lteq",
             "rightOperand": 365},
            {"leftOperand": "odrl:spatial", "operator": "odrl:isAnyOf",
             "rightOperand": ["EU", "EEA", "CH"]},
        ],
    }],
    "prohibition": [{
        "action": "distribute",
        "constraint": [
            {"leftOperand": "odrl:recipient", "operator": "odrl:isAnyOf",
             "rightOperand": ["public"]},
        ],
    }],
}

CONSUMER_HARD_POLICY: Dict[str, Any] = {
    "permission": [{
        "action": "use",
        "constraint": [
            {"leftOperand": "odrl:elapsedTime", "operator": "odrl:gteq",
             "rightOperand": 60},
        ],
    }],
}

# --------------------------------------------------------------------------- #
# Shared BAMBON tuning                                                         #
# --------------------------------------------------------------------------- #

#: Balanced profile from the ODRL-2026 experiments. Symmetric on purpose - both
#: participants use it, so any advantage comes from beliefs, not from tuning.
BAMBON_PROFILE: Dict[str, Any] = dict(
    lr=0.50, inertia=0.55, tau_in=0.60, tau_out=0.74, max_set_k=3,
    no_agreement_until=3, hopeless_window=16, jaccard_snap_threshold=0.45,
    max_rounds=60, tau_cooling_threshold=0.55, tau_in_cooled=0.60,
    tau_out_cooled=0.56, lr_min=0.32, lr_max=0.95, vol_window=5,
    max_param_delta=0.12, hopeless_severity=0.98, hopeless_patience=24,
    hopeless_context=0.06,
)

ROLES = {
    "provider": (PROVIDER_INIT_OFFER, PROVIDER_HARD_POLICY),
    "consumer": (CONSUMER_INIT_OFFER, CONSUMER_HARD_POLICY),
}


def profile_for(role: str):
    """Return ``(init_offer, hard_policy)`` deep copies for ``role``.

    Copies matter: ``Bambon`` mutates ``init_offer`` when building agreements,
    so handing the module-level dict to two sessions would cross-contaminate.
    """
    try:
        init_offer, hard_policy = ROLES[role]
    except KeyError:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLES)}")
    return copy.deepcopy(init_offer), copy.deepcopy(hard_policy)
