"""Coverage-pass tests for ``ju.viz.model_digraph`` and the ``ju.rjsf`` helpers."""

from typing import List, Optional

import pytest
from pydantic import BaseModel

from ju.rjsf import FormConfig, display_form_data, func_to_form_spec


class Address(BaseModel):
    street: str


class User(BaseModel):
    name: str
    age: Optional[int] = None
    addresses: List[Address]
    home: Address


def test_model_digraph_nodes_and_edges():
    pytest.importorskip("graphviz")
    from ju.viz import model_digraph

    src = model_digraph(User).source
    for name in ("User", "Address", "name", "addresses", "home"):
        assert name in src
    assert "User -> name" in src
    assert "addresses -> Address" in src
    assert "home -> Address" in src


def test_model_digraph_iterable_and_existing_dot():
    graphviz = pytest.importorskip("graphviz")
    from ju.viz import model_digraph

    dot = graphviz.Digraph(comment="mine")
    out = model_digraph([User, Address, "not a model"], dot=dot)
    assert out is dot
    assert "mine" in dot.source and "User" in dot.source


def test_model_digraph_parent_edge():
    pytest.importorskip("graphviz")
    from ju.viz import model_digraph

    src = model_digraph(Address, parent="holder").source
    assert "holder -> Address" in src


def _f(x: int, y: str = "a"):
    """Doc of f"""


def test_func_to_form_spec_flat_and_nested():
    nested = func_to_form_spec(_f)
    flat = func_to_form_spec(_f, nest_under_field=None)
    assert nested == {"rjsf": flat}
    assert flat["schema"]["properties"]["x"] == {"type": "integer"}
    assert flat["schema"]["properties"]["y"] == {"type": "string", "default": "a"}
    assert flat["schema"]["required"] == ["x"]
    assert flat["schema"]["description"] == "Doc of f"


def test_func_to_form_spec_doc_off_and_custom_title():
    spec = func_to_form_spec(
        _f, doc=False, nest_under_field=None, pyname_to_title=str.upper
    )
    assert "description" not in spec["schema"]


def test_form_config_defaults():
    cfg = FormConfig()
    assert cfg.submit_text == "Submit" and cfg.show_labels
    assert FormConfig(submit_text="Go").submit_text == "Go"


def test_display_form_data_formats_json(monkeypatch):
    pytest.importorskip("IPython")
    shown = []
    import IPython.display as ipd

    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj))
    display_form_data({"a": "é"}, indent=1, prefix="P:\n")
    (md,) = shown
    assert md.data.startswith("P:\n```json") and '"a": "é"' in md.data
