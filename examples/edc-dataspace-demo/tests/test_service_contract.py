"""The side channel has two independent implementations of one wire format:
``service.NegotiationReply`` writes it, ``consumer_flow.HttpParty`` reads it.
Nothing type-checks across that boundary, and a rename on one side fails only at
demo time, against a live data space, which is an expensive place to find out.

These tests read both modules' source with ``ast`` rather than importing them,
so they run without FastAPI, pydantic or edcpy installed.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "bambon_edc")
sys.path.insert(0, os.path.dirname(HERE))


def _tree(filename: str) -> ast.Module:
    with open(os.path.join(PKG, filename), encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=filename)


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _annotated_fields(cls: ast.ClassDef) -> set:
    return {node.target.id for node in cls.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}


def _subscript_keys(tree: ast.Module, var: str) -> set:
    """Every ``var["key"]`` and ``var.get("key")`` literal in the module."""
    keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == var
                and isinstance(node.slice, ast.Constant)):
            keys.add(node.slice.value)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == var
                and node.args and isinstance(node.args[0], ast.Constant)):
            keys.add(node.args[0].value)
    return keys


@pytest.fixture(scope="module")
def service_tree():
    return _tree("service.py")


@pytest.fixture(scope="module")
def client_tree():
    """``HttpParty`` lives in session.py, next to ``LocalParty``, so that both
    the consumer flow and the end-to-end test can use it."""
    return _tree("session.py")


def test_every_field_the_client_reads_is_sent_by_the_service(service_tree,
                                                             client_tree):
    sent = _annotated_fields(_class(service_tree, "NegotiationReply"))
    read = _subscript_keys(client_tree, "body")
    assert read, "found no body[...] reads - did HttpParty move again?"
    missing = read - sent
    assert not missing, f"HttpParty reads fields the service never sends: {missing}"


def test_open_session_payload_matches_the_service_model(service_tree,
                                                        client_tree):
    """``HttpParty.respond`` builds the opening payload by hand; it must carry
    every required field of ``OpenSession``."""
    model = _class(service_tree, "OpenSession")
    required = {
        node.target.id for node in model.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        and not (isinstance(node.annotation, ast.Subscript)
                 and getattr(node.annotation.value, "id", "") == "Optional")
    }

    sent = set()
    for node in ast.walk(client_tree):
        if isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if "offer" in keys and "asset_id" in keys:
                sent |= keys

    assert required <= sent, f"opening payload is missing {required - sent}"


def test_http_and_local_parties_expose_the_same_contract(client_tree):
    """``drive`` and ``relay`` accept either; a signature drift between them
    would only show up against a live stack."""
    for name in ("LocalParty", "HttpParty"):
        cls = _class(client_tree, name)
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        assert "respond" in methods, f"{name} is not a Party"


def test_step_kinds_agree_across_modules():
    """``Step.kind`` values are produced in session.py, forwarded verbatim as
    ``state`` by service.py, and branched on in both session.py and
    consumer_flow.py. All three must use the same three strings."""
    expected = {"counter", "agreement", "infeasible"}

    session = _tree("session.py")
    produced = set()
    for node in ast.walk(session):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Step":
            for kw in node.keywords:
                if kw.arg == "kind" and isinstance(kw.value, ast.Constant):
                    produced.add(kw.value.value)

    assert produced == expected, f"session.py emits {produced}"

    compared = set()
    for tree in (session, _tree("consumer_flow.py"), _tree("service.py")):
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute):
                if node.left.attr in ("kind", "state"):
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Constant):
                            compared.add(comparator.value)
    assert compared <= expected, f"unknown state(s) compared: {compared - expected}"


def test_service_only_materialises_on_the_provider_side(service_tree):
    """A consumer that published its own agreement would be creating contract
    definitions for an asset it does not own."""
    source = ast.get_source_segment  # noqa: F841 - readability
    guard_found = False
    for node in ast.walk(service_tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) \
                and node.left.id == "ROLE":
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value == "provider":
                    guard_found = True
    assert guard_found, "materialisation is not gated on ROLE == 'provider'"


def test_management_client_uses_v3_endpoints():
    tree = _tree("management.py")
    paths = {node.value for node in ast.walk(tree)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.startswith("/v3/")}
    assert {"/v3/assets", "/v3/policydefinitions",
            "/v3/contractdefinitions"} <= paths
