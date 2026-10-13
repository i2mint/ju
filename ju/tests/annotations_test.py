"""Tests for schema inference from string, ``Annotated`` and ``NewType`` annotations
(issue i2mint/ju#27). This module is itself under ``from __future__ import
annotations``, so every annotation in it is a string."""

from __future__ import annotations

from functools import partial
from inspect import Parameter
from typing import Annotated, NewType, Optional, TYPE_CHECKING

from ju.json_schema import DFLT_PARAM_TO_TYPE, signature_to_json_schema

if TYPE_CHECKING:
    from nowhere import Unimportable

UserId = NewType("UserId", int)
Tags = Annotated[list[str], "some metadata"]


def _props(func):
    return signature_to_json_schema(func)["properties"]


def test_string_annotations_are_resolved():
    def f(n: int, xs: list[int], flag: bool = True, opt: Optional[float] = None): ...

    assert _props(f) == {
        "n": {"type": "integer"},
        "xs": {"type": "array", "items": {"type": "integer"}},
        "flag": {"type": "boolean", "default": True},
        "opt": {"type": "number", "default": None},
    }


def test_annotated_and_newtype():
    def f(a: Annotated[list[int], "m"], b: UserId, c: Tags, d: Annotated[UserId, 1]): ...

    assert _props(f) == {
        "a": {"type": "array", "items": {"type": "integer"}},
        "b": {"type": "integer"},
        "c": {"type": "array", "items": {"type": "string"}},
        "d": {"type": "integer"},
    }


def test_wrappers_without_string_annotations():
    # (a real, non-string annotation: built from the objects themselves)
    param = Parameter(
        "p", Parameter.KEYWORD_ONLY, annotation=Optional[Annotated[UserId, "m"]]
    )
    assert DFLT_PARAM_TO_TYPE(param) == "integer"


def test_unresolvable_annotation_falls_back_without_spoiling_the_others():
    def f(a: int, b: Unimportable, c: float): ...

    assert _props(f) == {
        "a": {"type": "integer"},
        "b": {"type": "string"},
        "c": {"type": "number"},
    }


def test_partial_and_callable_instance_and_method():
    def g(a: int, b: list[float]): ...

    class Callme:
        def __call__(self, x: float, ys: list[int]): ...

        def meth(self, z: bool): ...

    assert _props(partial(g, 1)) == {"b": {"type": "array", "items": {"type": "number"}}}
    assert _props(Callme()) == {
        "x": {"type": "number"},
        "ys": {"type": "array", "items": {"type": "integer"}},
    }
    assert _props(Callme().meth) == {"z": {"type": "boolean"}}


def test_custom_param_to_prop_type_sees_the_owning_name():
    seen = []

    def spy(param):
        seen.append(param.name)
        return DFLT_PARAM_TO_TYPE(param)

    def f(xs: list[list[int]], ys=(1, 2)): ...

    signature_to_json_schema(f, param_to_prop_type=spy)
    assert set(seen) == {"xs", "ys"}
