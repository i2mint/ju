"""Tests for ``ju.json_schema.json_schema_to_signature`` honoring ``required``."""

from inspect import Parameter

import pytest

from ju.json_schema import json_schema_to_signature


def _defaults(sig):
    return {name: p.default for name, p in sig.parameters.items()}


def test_properties_branch_empty_required_means_all_optional():
    """Issue i2mint/ju#9: ``required: []`` must not yield required parameters."""
    schema = {
        "type": "object",
        "properties": {"first": {"type": "integer"}, "role": {"type": "string"}},
        "required": [],
    }
    sig = json_schema_to_signature(schema)
    assert _defaults(sig) == {"first": None, "role": None}
    sig.bind()  # no required argument


def test_properties_branch_missing_required_key_keeps_legacy_binding():
    """No ``required`` key: unchanged from before, so positional callers don't re-bind."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer", "default": 5},
            "c": {"type": "integer"},
        },
    }
    sig = json_schema_to_signature(schema)
    assert list(sig.parameters) == ["a", "c", "b"]
    assert sig.bind(1, 2).arguments == {"a": 1, "c": 2}


@pytest.mark.parametrize("bad", ["a", True, {"a": 1}, None])
def test_malformed_required_raises(bad):
    schema = {"type": "object", "properties": {"a": {}}, "required": bad}
    with pytest.raises(ValueError, match="required"):
        json_schema_to_signature(schema)


def test_properties_branch_required_stay_required_and_keep_positional_order():
    schema = {
        "type": "object",
        "properties": {
            "opt1": {"type": "string"},
            "req1": {"type": "integer"},
            "opt2": {"type": "integer", "default": 7},
            "req2": {"type": "boolean"},
        },
        "required": ["req2", "req1"],
    }
    sig = json_schema_to_signature(schema)
    # required first (in property order), then optional (in property order)
    assert list(sig.parameters) == ["req1", "req2", "opt1", "opt2"]
    assert _defaults(sig) == {
        "req1": Parameter.empty,
        "req2": Parameter.empty,
        "opt1": None,
        "opt2": 7,
    }
    # positional calls bind required params in schema order
    assert sig.bind(1, True).arguments == {"req1": 1, "req2": True}
    with pytest.raises(TypeError):
        sig.bind()


def test_properties_branch_required_wins_over_schema_default():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer", "default": 3}},
        "required": ["a"],
    }
    assert _defaults(json_schema_to_signature(schema)) == {"a": Parameter.empty}


def test_required_matches_raw_property_name_not_pyname():
    schema = {
        "type": "object",
        "properties": {"$first": {"type": "integer"}, "x": {"type": "string"}},
        "required": ["$first"],
    }
    sig = json_schema_to_signature(schema)
    (req,) = [n for n, p in sig.parameters.items() if p.default is Parameter.empty]
    assert req.endswith("first")
    assert _defaults(sig)["x"] is None


def test_default_default_is_a_seam():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    sig = json_schema_to_signature(schema, default_default="")
    assert _defaults(sig) == {"a": ""}


def test_parameters_branch_optional_without_default_is_optional():
    """OpenAPI params with no ``required`` (default false) and no schema default."""
    schema = {
        "parameters": [
            {"name": "x_role", "in": "header", "schema": {"type": "string"}},
            {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
            {"name": "n", "in": "query", "schema": {"type": "integer"}},
        ]
    }
    sig = json_schema_to_signature(schema)
    assert list(sig.parameters) == ["id", "x_role", "n"]
    assert _defaults(sig) == {"id": Parameter.empty, "x_role": None, "n": None}
