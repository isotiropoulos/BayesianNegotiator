"""Normalisation tests. No numpy/scipy/EDC needed - pure data manipulation."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bambon_edc.odrl_edc import (  # noqa: E402
    TRACTUSX,
    VANILLA,
    access_policy_definition,
    contract_definition,
    dialect_by_name,
    offer_from_edc_dataset,
    offer_from_edc_policy,
    operands_used,
    policy_definition,
    strip_prefix,
)


# --------------------------------------------------------------------------- #
# fixtures: real-ish EDC catalog output                                        #
# --------------------------------------------------------------------------- #

# EDC compacts single-element arrays to bare objects and prefixes everything
# with `odrl:`. Both quirks are represented here on purpose.
CATALOG_DATASET = {
    "@id": "climate-observations-2026",
    "@type": "dcat:Dataset",
    "odrl:hasPolicy": {
        "@id": "b2ZmZXI=:YXNzZXQ=:Y29udHJhY3Q=",
        "@type": "odrl:Offer",
        "odrl:permission": {
            "odrl:action": {"@id": "odrl:use"},
            "odrl:constraint": {
                "@type": "odrl:LogicalConstraint",
                "odrl:and": [
                    {"odrl:leftOperand": {"@id": "purpose"},
                     "odrl:operator": {"@id": "odrl:eq"},
                     "odrl:rightOperand": "research"},
                    {"odrl:leftOperand": {"@id": "elapsedTime"},
                     "odrl:operator": {"@id": "odrl:lteq"},
                     "odrl:rightOperand": "30"},
                ],
            },
        },
        "odrl:prohibition": [],
        "odrl:obligation": [],
    },
}

AGREEMENT = {
    "@id": "urn:uuid:abc",
    "@type": "odrl:Agreement",
    "permission": [{
        "action": "use",
        "constraint": [
            {"leftOperand": "odrl:purpose", "operator": "odrl:eq",
             "rightOperand": "commercial"},
            {"leftOperand": "odrl:elapsedTime", "operator": "odrl:lteq",
             "rightOperand": 69.0},
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
# EDC -> BAMBON                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("odrl:purpose", "purpose"),
    ("http://www.w3.org/ns/odrl/2/purpose", "purpose"),
    ("https://w3id.org/edc/v0.0.1/ns/id", "id"),
    ({"@id": "odrl:eq"}, "eq"),
    ({"@value": "research"}, "research"),
    (42, 42),
    (None, None),
])
def test_strip_prefix(raw, expected):
    assert strip_prefix(raw) == expected


def test_dataset_parses_into_bambon_shape():
    offer = offer_from_edc_dataset(CATALOG_DATASET)

    assert offer["@type"] == "odrl:Offer"
    assert len(offer["permission"]) == 1
    permission = offer["permission"][0]
    assert permission["action"] == "use"

    clauses = {c["leftOperand"]: c for c in permission["constraint"]}
    assert set(clauses) == {"odrl:purpose", "odrl:elapsedTime"}
    assert clauses["odrl:purpose"]["operator"] == "odrl:eq"
    assert clauses["odrl:purpose"]["rightOperand"] == "research"


def test_numeric_right_operands_are_coerced_back_to_numbers():
    """EDC serialises every right operand as a string. If we hand BAMBON the
    string "30" it builds a categorical belief instead of a numeric one and the
    negotiation degenerates. This is the single most important coercion here."""
    offer = offer_from_edc_dataset(CATALOG_DATASET)
    elapsed = next(c for c in offer["permission"][0]["constraint"]
                   if c["leftOperand"] == "odrl:elapsedTime")
    assert elapsed["rightOperand"] == 30
    assert isinstance(elapsed["rightOperand"], int)


def test_logical_constraint_wrapper_is_flattened():
    offer = offer_from_edc_policy(CATALOG_DATASET["odrl:hasPolicy"])
    assert len(offer["permission"][0]["constraint"]) == 2
    assert all("and" not in c for c in offer["permission"][0]["constraint"])


def test_empty_rule_lists_are_dropped():
    offer = offer_from_edc_dataset(CATALOG_DATASET)
    assert "prohibition" not in offer
    assert "obligation" not in offer


def test_dataset_without_policy_is_an_error():
    with pytest.raises(ValueError, match="hasPolicy"):
        offer_from_edc_dataset({"@id": "x"})


# --------------------------------------------------------------------------- #
# BAMBON -> EDC                                                                #
# --------------------------------------------------------------------------- #

def test_policy_definition_shape():
    body = policy_definition(AGREEMENT, policy_id="p1")

    assert body["@id"] == "p1"
    assert body["@type"] == "PolicyDefinition"
    assert body["policy"]["@type"] == "Set"
    assert body["policy"]["@context"] == "http://www.w3.org/ns/odrl.jsonld"
    assert json.dumps(body)  # must be serialisable as-is


def test_multiple_clauses_become_an_explicit_conjunction():
    """A bare array of constraints is read as an implicit OR by some EDC
    versions, which would silently widen the negotiated policy."""
    body = policy_definition(AGREEMENT, policy_id="p1")
    constraint = body["policy"]["permission"][0]["constraint"]
    assert len(constraint) == 1
    assert constraint[0]["@type"] == "LogicalConstraint"
    assert len(constraint[0]["and"]) == 2


def test_single_clause_is_not_wrapped():
    body = policy_definition(AGREEMENT, policy_id="p1")
    prohibition = body["policy"]["prohibition"][0]["constraint"]
    assert len(prohibition) == 1
    assert "and" not in prohibition[0]


def test_operands_are_bare_strings_by_default():
    """Both EDC 0.5.x and Tractus-X 0.12 want bare strings. The `{"@id": ...}`
    node form is opt-in, not the default - getting this backwards produces a
    policy the Management API accepts and the engine never matches."""
    body = policy_definition(AGREEMENT, policy_id="p1")
    clause = body["policy"]["permission"][0]["constraint"][0]["and"][0]
    assert clause["leftOperand"] == "purpose"
    assert clause["operator"] == "eq"


def test_id_form_is_available_as_an_override():
    body = policy_definition(AGREEMENT, policy_id="p1", id_form=True)
    clause = body["policy"]["permission"][0]["constraint"][0]["and"][0]
    assert clause["leftOperand"] == {"@id": "purpose"}
    assert clause["operator"] == {"@id": "odrl:eq"}


def test_right_operands_are_stringified_without_trailing_zeros():
    """69.0 must not become "69.0" - EDC compares right operands as strings, so
    a numeric policy function that parses "69.0" as an int would throw."""
    body = policy_definition(AGREEMENT, policy_id="p1")
    clauses = body["policy"]["permission"][0]["constraint"][0]["and"]
    elapsed = next(c for c in clauses if c["leftOperand"] == "elapsedTime")
    assert elapsed["rightOperand"] == "69"


def test_participant_scoping_is_appended_to_every_permission():
    body = policy_definition(AGREEMENT, policy_id="p1",
                             participant_ids=["consumer-a"])
    clauses = body["policy"]["permission"][0]["constraint"][0]["and"]
    scoped = [c for c in clauses if c["leftOperand"] == "participantId"]
    assert len(scoped) == 1
    assert scoped[0]["rightOperand"] == ["consumer-a"]
    # the negotiated terms must survive alongside the scope clause
    assert len(clauses) == 3


# --------------------------------------------------------------------------- #
# dialects                                                                     #
# --------------------------------------------------------------------------- #

def test_vanilla_is_the_default_dialect():
    assert dialect_by_name(None) is VANILLA
    assert dialect_by_name("vanilla") is VANILLA
    assert dialect_by_name("TractusX") is TRACTUSX
    with pytest.raises(ValueError, match="unknown dialect"):
        dialect_by_name("catena-z")


def test_tractusx_uses_the_catenax_context():
    """Tractus-X 0.12+ rejects a policy carrying the plain ODRL context."""
    body = policy_definition(AGREEMENT, policy_id="p1", dialect=TRACTUSX)
    context = body["policy"]["@context"]
    assert isinstance(context, list)
    assert any("catenax" in c for c in context if isinstance(c, str))


def test_tractusx_wraps_constraints_in_a_single_object():
    """Tractus-X wants `"constraint": {"and": [...]}`, not a list. Sending the
    list form is silently accepted and then never matches."""
    body = policy_definition(AGREEMENT, policy_id="p1", dialect=TRACTUSX)
    constraint = body["policy"]["permission"][0]["constraint"]
    assert isinstance(constraint, dict)
    assert "and" in constraint

    prohibition = body["policy"]["prohibition"][0]["constraint"]
    assert isinstance(prohibition, dict), "even single clauses use the object form"


def test_tractusx_injects_the_mandatory_catenax_baseline():
    """Membership / FrameworkAgreement / UsagePurpose are required on any use
    policy. BAMBON negotiates on top of them, never instead of them."""
    body = policy_definition(AGREEMENT, policy_id="p1", dialect=TRACTUSX)
    operands = {c["leftOperand"]
                for c in body["policy"]["permission"][0]["constraint"]["and"]}
    assert {"Membership", "FrameworkAgreement", "UsagePurpose"} <= operands
    # and the negotiated terms are still there
    assert {"purpose", "elapsedTime"} <= operands


def test_baseline_applies_only_to_use_permissions():
    body = policy_definition(AGREEMENT, policy_id="p1", dialect=TRACTUSX)
    prohibition = body["policy"]["prohibition"][0]["constraint"]["and"]
    operands = {c["leftOperand"] for c in prohibition}
    assert "Membership" not in operands
    assert operands == {"recipient"}


def test_baseline_can_be_overridden_or_disabled():
    disabled = policy_definition(AGREEMENT, policy_id="p1",
                                 dialect=TRACTUSX, baseline=[])
    operands = {c["leftOperand"]
                for c in disabled["policy"]["permission"][0]["constraint"]["and"]}
    assert "Membership" not in operands

    custom = policy_definition(
        AGREEMENT, policy_id="p1", dialect=VANILLA,
        baseline=[{"leftOperand": "Tenant", "operator": "eq",
                   "rightOperand": "acme"}])
    operands = {c["leftOperand"]
                for c in custom["policy"]["permission"][0]["constraint"][0]["and"]}
    assert "Tenant" in operands


def test_vanilla_has_no_baseline():
    """Upstream EDC mandates nothing, so the vanilla policy is exactly what was
    negotiated - no silent additions."""
    assert VANILLA.baseline == ()
    body = policy_definition(AGREEMENT, policy_id="p1", dialect=VANILLA)
    operands = {c["leftOperand"]
                for c in body["policy"]["permission"][0]["constraint"][0]["and"]}
    assert operands == {"purpose", "elapsedTime"}


# --------------------------------------------------------------------------- #
# access policies                                                              #
# --------------------------------------------------------------------------- #

def test_access_policy_is_not_a_copy_of_the_use_policy():
    """EDC asks two questions - who may see the offer, and what they may do.
    Tractus-X rejects a use policy reused as an access policy."""
    body = access_policy_definition(policy_id="a1",
                                    participant_ids=["consumer-a"])
    permission = body["policy"]["permission"][0]
    assert permission["action"] == "access"
    clauses = _flat(permission["constraint"])
    assert [c["leftOperand"] for c in clauses] == ["participantId"]


def test_unscoped_access_policy_is_unconditional():
    body = access_policy_definition(policy_id="a1")
    permission = body["policy"]["permission"][0]
    assert permission == {"action": "access"}


def _flat(constraint):
    if isinstance(constraint, dict):
        return constraint.get("and", [constraint])
    out = []
    for item in constraint:
        out.extend(item["and"] if "and" in item else [item])
    return out


def test_non_agreements_are_refused():
    """Only a converged agreement may be published. Materialising a mid-flight
    counter-offer would bind the provider to terms it has not accepted."""
    counter = dict(AGREEMENT, **{"@type": "odrl:Offer"})
    with pytest.raises(ValueError, match="Agreement"):
        policy_definition(counter)


def test_contract_definition_selects_the_asset():
    body = contract_definition(asset_id="asset-1",
                               access_policy_id="a", contract_policy_id="c",
                               definition_id="cd-1")
    selector = body["assetsSelector"][0]
    assert selector["operandRight"] == "asset-1"
    assert selector["operandLeft"].endswith("/id")
    assert body["accessPolicyId"] == "a"
    assert body["contractPolicyId"] == "c"


# --------------------------------------------------------------------------- #
# round trip                                                                   #
# --------------------------------------------------------------------------- #

def test_round_trip_preserves_terms():
    """Publish an agreement, read it back as if from a catalog, and confirm the
    negotiated terms survive the JSON-LD trip."""
    published = policy_definition(AGREEMENT, policy_id="p1")
    recovered = offer_from_edc_policy(published["policy"])

    original = {(strip_prefix(c["leftOperand"]), strip_prefix(c["operator"]))
                for c in AGREEMENT["permission"][0]["constraint"]}
    back = {(strip_prefix(c["leftOperand"]), strip_prefix(c["operator"]))
            for c in recovered["permission"][0]["constraint"]}
    assert original == back


def test_operands_used_reports_everything_needing_a_policy_function():
    assert set(operands_used(AGREEMENT)) == {"purpose", "elapsedTime", "recipient"}
