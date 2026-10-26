"""Normalisation between BAMBON's ODRL dialect and EDC's JSON-LD policy shapes.

BAMBON speaks a compact, prefix-tolerant ODRL dialect::

    {"@type": "odrl:Offer",
     "permission": [{"action": "use",
                     "constraint": [{"leftOperand": "odrl:purpose",
                                     "operator": "odrl:eq",
                                     "rightOperand": "research"}]}]}

EDC speaks expanded/compacted JSON-LD, and the exact shape depends on whether
you are *reading* a catalog (expanded-ish, ``odrl:`` prefixed, single-element
lists collapsed to objects) or *writing* a PolicyDefinition to the Management
API (compacted against the ODRL context, logical constraints wrapped in
``and``).

This module has three jobs:

1. ``offer_from_edc_dataset``  - catalog dataset -> BAMBON offer.
2. ``policy_definition``       - BAMBON agreement -> EDC PolicyDefinition body.
3. ``contract_definition``     - bind that policy to an asset + counter-party.

Everything here is pure data manipulation: no network, no edcpy import. That
keeps it unit-testable without a running data space.
"""

from __future__ import annotations

import json

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "EDC_NS",
    "ODRL_NS",
    "RULE_TYPES",
    "Dialect",
    "VANILLA",
    "TRACTUSX",
    "DIALECTS",
    "dialect_by_name",
    "strip_prefix",
    "offer_from_edc_dataset",
    "offer_from_edc_policy",
    "policy_definition",
    "contract_definition",
    "asset_definition",
    "access_policy_definition",
    "operands_used",
]

EDC_NS = "https://w3id.org/edc/v0.0.1/ns/"
ODRL_NS = "http://www.w3.org/ns/odrl/2/"

#: The three ODRL rule types BAMBON negotiates. EDC supports all three, though
#: only ``permission`` participates in access decisions in most deployments.
RULE_TYPES = ("permission", "prohibition", "obligation")

_MANAGEMENT_CONTEXT = {
    "@vocab": EDC_NS,
    "edc": EDC_NS,
    "odrl": ODRL_NS,
}


# --------------------------------------------------------------------------- #
# dialects                                                                     #
# --------------------------------------------------------------------------- #
#
# "EDC" is not one policy format. Upstream/vanilla connectors and Tractus-X
# disagree on the JSON-LD context, on whether operands are bare strings or
# ``{"@id": ...}`` nodes, and on whether a rule's ``constraint`` is an array or
# a single object wrapping ``and``. Getting any of these wrong yields a 400 from
# the Management API, or worse, a policy that is accepted and never matches.
#
# The Tractus-X shapes here are taken from a working deployment
# (EDC Deploy/services/edc-bridge/edc_client.py against Tractus-X 0.12.1), not
# from documentation.

#: Catena-X policy context required by Tractus-X 0.12+.
_CATENAX_CONTEXT = [
    "https://w3id.org/dspace/2025/1/odrl-profile.jsonld",
    "https://w3id.org/catenax/2025/9/policy/context.jsonld",
    {"@vocab": EDC_NS},
]


@dataclass(frozen=True)
class Dialect:
    """How one connector family wants its policies written.

    Attributes
    ----------
    context:
        Value for the policy's ``@context``.
    id_form:
        ``True`` emits ``{"@id": "purpose"}`` for left operands and operators.
        Both EDC 0.5.x and Tractus-X 0.12 want bare strings, so this defaults
        off; it exists for connectors that require the expanded node form.
    conjunction:
        ``"list"``   -> ``"constraint": [{"@type": "LogicalConstraint", "and": [...]}]``
        ``"object"`` -> ``"constraint": {"and": [...]}``  (Tractus-X)
    baseline:
        Clauses injected into every ``use`` permission and never negotiated.
        Tractus-X 0.12+ rejects a use policy that lacks Membership,
        FrameworkAgreement and UsagePurpose, so BAMBON negotiates *on top of*
        this floor rather than being free to propose any policy at all.
    """

    name: str
    context: Any
    id_form: bool = False
    conjunction: str = "list"
    baseline: Sequence[Dict[str, Any]] = ()


VANILLA = Dialect(
    name="vanilla",
    context="http://www.w3.org/ns/odrl.jsonld",
    id_form=False,
    conjunction="list",
    baseline=(),
)

TRACTUSX = Dialect(
    name="tractusx",
    context=_CATENAX_CONTEXT,
    id_form=False,
    conjunction="object",
    baseline=(
        {"leftOperand": "Membership", "operator": "eq",
         "rightOperand": "active"},
        {"leftOperand": "FrameworkAgreement", "operator": "eq",
         "rightOperand": "DataExchangeGovernance:1.0"},
        {"leftOperand": "UsagePurpose", "operator": "isAnyOf",
         "rightOperand": ["cx.core.digitalTwinRegistry:1"]},
    ),
)

DIALECTS = {d.name: d for d in (VANILLA, TRACTUSX)}


def dialect_by_name(name: Optional[str]) -> Dialect:
    """Look a dialect up by name, defaulting to vanilla."""
    if not name:
        return VANILLA
    try:
        return DIALECTS[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown dialect {name!r}; expected one of {sorted(DIALECTS)}")

# ODRL operators BAMBON can emit. Anything outside this set is passed through
# unchanged so that a custom vocabulary still round-trips.
_KNOWN_OPERATORS = {
    "eq", "neq", "lt", "lteq", "gt", "gteq",
    "isA", "hasPart", "isPartOf", "isAllOf", "isAnyOf", "isNoneOf",
}


#: Operators whose right operand is a collection, whatever EDC compacted it to.
_SET_OPERATORS = {"isAllOf", "isAnyOf", "isNoneOf", "hasPart", "isPartOf"}


# --------------------------------------------------------------------------- #
# small JSON-LD helpers                                                        #
# --------------------------------------------------------------------------- #

def strip_prefix(value: Any) -> Any:
    """Return the local name of a CURIE / IRI, leaving non-strings alone.

    ``"odrl:purpose"`` -> ``"purpose"``;
    ``"http://www.w3.org/ns/odrl/2/purpose"`` -> ``"purpose"``;
    ``{"@id": "odrl:eq"}`` -> ``"eq"``; ``42`` -> ``42``.
    """
    if isinstance(value, dict):
        # Some EDC builds compact an action to {"odrl:type": "<iri>"} rather
        # than to a bare string or {"@id": ...}. Unwrapped here, otherwise the
        # rule's action never matches "use" and its clauses are invisible.
        for key in ("@id", "@value", "@type", "odrl:type",
                    ODRL_NS + "type", "type"):
            if key in value:
                value = value[key]
                break
        else:
            return value
        return strip_prefix(value)
    if not isinstance(value, str):
        return value
    for sep in ("#", "/"):
        if sep in value:
            value = value.rsplit(sep, 1)[-1]
    if ":" in value:
        value = value.rsplit(":", 1)[-1]
    return value


def _as_list(value: Any) -> List[Any]:
    """EDC collapses single-element JSON-LD arrays to bare objects. Undo that."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get(node: Dict[str, Any], *names: str) -> Any:
    """Look a key up under every namespace spelling EDC might have used."""
    for name in names:
        for candidate in (name, f"odrl:{name}", ODRL_NS + name,
                          f"edc:{name}", EDC_NS + name):
            if candidate in node:
                return node[candidate]
    return None


def _scalar(value: Any) -> Any:
    """Unwrap ``{"@value": x}`` / ``{"@id": x}`` wrappers, recursing into lists."""
    if isinstance(value, list):
        return [_scalar(v) for v in value]
    if isinstance(value, dict):
        if "@value" in value:
            return value["@value"]
        if "@id" in value:
            return value["@id"]
        if "@list" in value:
            return _scalar(value["@list"])
    if isinstance(value, str):
        # A multi-valued right operand can come back as JSON *text*, e.g.
        # '[{"@value":"EU"},{"@value":"EEA"}]'. Left as a string it compares
        # unequal to every real value and the clause silently never matches.
        text = value.strip()
        if text[:1] in ("[", "{") and text[-1:] in ("]", "}"):
            try:
                return _scalar(json.loads(text))
            except ValueError:
                return value
    return value


def _coerce_number(value: Any) -> Any:
    """EDC serialises every right operand as a string. BAMBON needs the number
    back, otherwise a numeric issue is misclassified as categorical."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


# --------------------------------------------------------------------------- #
# EDC  ->  BAMBON                                                              #
# --------------------------------------------------------------------------- #

def _flatten_constraint(node: Any) -> List[Dict[str, Any]]:
    """Flatten one ODRL constraint node into BAMBON's flat clause list.

    Handles atomic constraints and ``LogicalConstraint`` wrappers (``and``,
    ``or``, ``andSequence``). BAMBON's model is a conjunction of clauses per
    action, so ``or`` branches are flattened too - they become alternative
    clauses on the same operand, which BAMBON's merged-categorical routing
    already understands. That is lossy for genuine disjunctions of *different*
    operands; see ``README.md`` for the caveat.
    """
    out: List[Dict[str, Any]] = []
    for item in _as_list(node):
        if not isinstance(item, dict):
            continue

        logical = None
        for junction in ("and", "or", "andSequence", "xone"):
            branch = _get(item, junction)
            if branch is not None:
                logical = branch
                break
        if logical is not None:
            out.extend(_flatten_constraint(logical))
            continue

        left = _get(item, "leftOperand")
        operator = _get(item, "operator")
        right = _get(item, "rightOperand")
        if left is None or operator is None:
            continue

        op_name = strip_prefix(operator)
        value = _coerce_number(_scalar(right))
        # EDC collapses a single-element array to a bare value. For a set
        # operator that turns ["EEA"] into "EEA", and every downstream set
        # comparison then iterates the string's characters instead of its one
        # member - which reads as a hard-policy violation that never happened.
        if op_name in _SET_OPERATORS and not isinstance(value, list):
            value = [value]
        out.append({
            "leftOperand": f"odrl:{strip_prefix(left)}",
            "operator": f"odrl:{op_name}",
            "rightOperand": value,
        })
    return out


def offer_from_edc_policy(policy: Dict[str, Any],
                          offer_id: Optional[str] = None) -> Dict[str, Any]:
    """Turn one EDC ``odrl:hasPolicy`` node into a BAMBON offer."""
    offer: Dict[str, Any] = {
        "@id": offer_id or policy.get("@id") or f"urn:uuid:{uuid.uuid4()}",
        "@type": "odrl:Offer",
    }

    for rule_type in RULE_TYPES:
        rules = _as_list(_get(policy, rule_type))
        parsed: List[Dict[str, Any]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            action = strip_prefix(_scalar(_get(rule, "action")))
            if not action:
                continue
            parsed.append({
                "action": action,
                "constraint": _flatten_constraint(_get(rule, "constraint")),
            })
        if parsed:
            offer[rule_type] = parsed

    return offer


def offer_from_edc_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the provider's negotiable offer from a catalog dataset.

    ``edcpy``'s ``ConnectorController.fetch_catalog()`` yields ``Dataset``
    objects; pass ``dataset.dct_dict`` (or the raw dict) here. The EDC offer id
    is preserved on ``@id`` because the *final* EDC negotiation must quote it
    verbatim.
    """
    policy = _get(dataset, "hasPolicy")
    policies = _as_list(policy)
    if not policies:
        raise ValueError(f"dataset {dataset.get('@id')!r} carries no odrl:hasPolicy")
    return offer_from_edc_policy(policies[0])


# --------------------------------------------------------------------------- #
# BAMBON  ->  EDC                                                              #
# --------------------------------------------------------------------------- #

def _edc_constraint(clause: Dict[str, Any], *, id_form: bool) -> Dict[str, Any]:
    """Render one BAMBON clause as an EDC atomic constraint."""
    left = strip_prefix(clause["leftOperand"])
    operator = strip_prefix(clause["operator"])
    right = clause["rightOperand"]

    if operator not in _KNOWN_OPERATORS:
        # Unknown operator: keep whatever the caller wrote rather than guessing.
        operator = str(clause["operator"])

    # EDC compares right operands as strings unless the registered policy
    # function coerces. Lists stay lists (isAnyOf / isNoneOf take collections).
    if isinstance(right, (list, tuple, set)):
        right_out: Any = [str(v) for v in right]
    elif isinstance(right, bool):
        right_out = "true" if right else "false"
    elif isinstance(right, float) and right.is_integer():
        right_out = str(int(right))
    else:
        right_out = str(right)

    if id_form:
        return {
            "leftOperand": {"@id": left},
            "operator": {"@id": f"odrl:{operator}"},
            "rightOperand": right_out,
        }
    return {"leftOperand": left, "operator": operator, "rightOperand": right_out}


def _wrap_conjunction(clauses: List[Dict[str, Any]],
                      dialect: Dialect) -> Any:
    """Package rendered clauses into the rule's ``constraint`` value.

    A bare array of constraints is read as an implicit disjunction by some
    connector versions, so a multi-clause rule always gets an explicit ``and``.
    """
    if not clauses:
        return None
    if dialect.conjunction == "object":
        # Tractus-X: a single object, `and` even for one clause.
        return {"and": clauses}
    if len(clauses) == 1:
        return clauses
    return [{"@type": "LogicalConstraint", "and": clauses}]


def _edc_rule(rule: Dict[str, Any],
              dialect: Dialect,
              *,
              extra_clauses: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    clauses = [_edc_constraint(c, id_form=dialect.id_form)
               for c in list(rule.get("constraint", []) or []) + list(extra_clauses)]
    out: Dict[str, Any] = {"action": strip_prefix(rule["action"])}
    constraint = _wrap_conjunction(clauses, dialect)
    if constraint is not None:
        out["constraint"] = constraint
    return out


def policy_definition(agreement: Dict[str, Any],
                      *,
                      policy_id: Optional[str] = None,
                      participant_ids: Optional[Sequence[str]] = None,
                      dialect: Dialect = VANILLA,
                      baseline: Optional[Sequence[Dict[str, Any]]] = None,
                      id_form: Optional[bool] = None) -> Dict[str, Any]:
    """Materialise a BAMBON ``odrl:Agreement`` as an EDC PolicyDefinition body.

    Parameters
    ----------
    agreement:
        The ``odrl:Agreement`` returned by ``Bambon.observe``.
    policy_id:
        Management-API id for the new definition. Defaults to a fresh urn.
    participant_ids:
        If given, an extra permission constraint restricts the policy to these
        counter-party participant ids. Without it the negotiated terms would be
        published to the whole data space, which defeats the point of a
        *bilateral* negotiation.
    dialect:
        :data:`VANILLA` (upstream EDC) or :data:`TRACTUSX`. Controls context,
        operand form, constraint container and the non-negotiable baseline.
    baseline:
        Override the dialect's baseline clauses. These are appended to every
        ``use`` permission and are never subject to negotiation - BAMBON
        bargains on top of them. Pass ``[]`` to disable.
    id_form:
        Override the dialect's operand form. Rarely needed.

    Returns
    -------
    dict
        Ready to ``POST`` to ``/v3/policydefinitions``.
    """
    if agreement.get("@type") not in ("odrl:Agreement", "Agreement", None):
        raise ValueError(
            f"expected an odrl:Agreement, got {agreement.get('@type')!r}. "
            "Only a converged BAMBON agreement should be materialised."
        )

    if id_form is not None and id_form != dialect.id_form:
        dialect = Dialect(dialect.name, dialect.context, id_form,
                          dialect.conjunction, dialect.baseline)

    floor = list(dialect.baseline if baseline is None else baseline)

    policy: Dict[str, Any] = {
        "@context": dialect.context,
        "@type": "Set",
    }

    for rule_type in RULE_TYPES:
        rules = agreement.get(rule_type) or []
        rendered = []
        for rule in rules:
            # The mandatory floor applies to `use` only; it makes no sense on a
            # prohibition, and connectors only validate the contract policy.
            extra = floor if (rule_type == "permission"
                              and strip_prefix(rule.get("action")) == "use") else ()
            rendered.append(_edc_rule(copy.deepcopy(rule), dialect,
                                      extra_clauses=extra))
        if rendered:
            policy[rule_type] = rendered

    if participant_ids:
        _scope_to_participants(policy, list(participant_ids), dialect)

    return {
        "@context": _MANAGEMENT_CONTEXT,
        "@id": policy_id or f"urn:uuid:{uuid.uuid4()}",
        "@type": "PolicyDefinition",
        "policy": policy,
    }


def access_policy_definition(*,
                             policy_id: str,
                             participant_ids: Optional[Sequence[str]] = None,
                             dialect: Dialect = VANILLA) -> Dict[str, Any]:
    """An access (catalog-visibility) policy, separate from the use policy.

    EDC evaluates the *access* policy to decide whether a participant may see
    the offer at all, and the *contract* policy for what they may then do. A
    bilateral agreement belongs in the contract policy; the access policy just
    has to name who is allowed to look. Tractus-X in particular rejects a use
    policy reused verbatim as an access policy.
    """
    permission: Dict[str, Any] = {"action": "access"}
    if participant_ids:
        clause = _edc_constraint(
            {"leftOperand": "participantId", "operator": "isAnyOf",
             "rightOperand": list(participant_ids)},
            id_form=dialect.id_form)
        constraint = _wrap_conjunction([clause], dialect)
        if constraint is not None:
            permission["constraint"] = constraint

    return {
        "@context": _MANAGEMENT_CONTEXT,
        "@id": policy_id,
        "@type": "PolicyDefinition",
        "policy": {
            "@context": dialect.context,
            "@type": "Set",
            "permission": [permission],
            "prohibition": [],
            "obligation": [],
        },
    }


def _scope_to_participants(policy: Dict[str, Any],
                           participant_ids: List[str],
                           dialect: Dialect) -> None:
    """Append a ``participantId isAnyOf [...]`` clause to every permission."""
    rendered = _edc_constraint(
        {"leftOperand": "participantId", "operator": "isAnyOf",
         "rightOperand": participant_ids},
        id_form=dialect.id_form)

    permissions = policy.get("permission")
    if not permissions:
        policy["permission"] = [{
            "action": "use",
            "constraint": _wrap_conjunction([rendered], dialect),
        }]
        return

    for permission in permissions:
        merged = _unwrap_conjunction(permission.get("constraint"))
        merged.append(rendered)
        permission["constraint"] = _wrap_conjunction(merged, dialect)


def _unwrap_conjunction(constraint: Any) -> List[Dict[str, Any]]:
    """Inverse of :func:`_wrap_conjunction`, for either dialect's shape."""
    out: List[Dict[str, Any]] = []
    for item in _as_list(constraint):
        if isinstance(item, dict) and "and" in item:
            out.extend(item["and"])
        elif isinstance(item, dict):
            out.append(item)
    return out


def contract_definition(*,
                        asset_id: str,
                        access_policy_id: str,
                        contract_policy_id: str,
                        definition_id: Optional[str] = None) -> Dict[str, Any]:
    """Bind a negotiated policy to an asset. ``POST /v3/contractdefinitions``."""
    return {
        "@context": _MANAGEMENT_CONTEXT,
        "@id": definition_id or f"urn:uuid:{uuid.uuid4()}",
        "@type": "ContractDefinition",
        "accessPolicyId": access_policy_id,
        "contractPolicyId": contract_policy_id,
        "assetsSelector": [{
            # The criterion carries its own @context: Tractus-X does not
            # inherit the outer one when expanding the selector.
            "@context": {"@vocab": EDC_NS},
            "@type": "Criterion",
            "operandLeft": f"{EDC_NS}id",
            "operator": "=",
            "operandRight": asset_id,
        }],
    }


def asset_definition(*, asset_id: str, base_url: str,
                     name: str = "", proxy_path: bool = True) -> Dict[str, Any]:
    """Minimal HttpData asset. ``POST /v3/assets``. Used to seed the demo."""
    return {
        "@context": _MANAGEMENT_CONTEXT,
        "@id": asset_id,
        "@type": "Asset",
        "properties": {"name": name or asset_id, "contenttype": "application/json"},
        "dataAddress": {
            "@type": "DataAddress",
            "type": "HttpData",
            "baseUrl": base_url,
            "proxyPath": "true" if proxy_path else "false",
        },
    }


# --------------------------------------------------------------------------- #
# introspection                                                                #
# --------------------------------------------------------------------------- #

def operands_used(offer: Dict[str, Any]) -> List[str]:
    """Every left operand appearing in an offer or agreement.

    Each of these needs a policy function registered in the provider connector
    (``PolicyEngine#registerFunction``) or the materialised policy will not
    evaluate. Call this to fail loudly at publish time rather than silently at
    negotiation time.
    """
    seen: List[str] = []
    for rule_type in RULE_TYPES:
        for rule in offer.get(rule_type) or []:
            for clause in rule.get("constraint", []) or []:
                operand = strip_prefix(clause.get("leftOperand"))
                if operand and operand not in seen:
                    seen.append(operand)
    return seen
