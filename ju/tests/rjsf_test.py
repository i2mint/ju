"""
Test objects from the ju.rjsf module
"""

from functools import partial
from collections.abc import Callable
import inspect
from operator import attrgetter
from ju.json_schema import signature_to_json_schema
from ju.util import FeatureSwitch


def mercury(sweet: float, sour=True):
    return sweet * sour


def venus():
    """Nothing from nothing"""


def earth(north: str, south: bool, east: int = 1, west: float = 2.0):
    """Earth docs"""
    return f"{north=}, {south=}, {east=}, {west=}"


mercury_schema = {
    "title": "mercury",
    "type": "object",
    "properties": {
        "sweet": {"type": "number"},
        "sour": {"type": "boolean", "default": True},
    },
    "required": ["sweet"],
}

venus_schema = {
    "title": "venus",
    "type": "object",
    "properties": {},
    "required": [],
    "description": "Nothing from nothing",
}

earth_schema = {
    "description": "Earth docs",
    "title": "earth",
    "type": "object",
    "properties": {
        "north": {"type": "string"},
        "south": {"type": "boolean"},
        "east": {"type": "integer", "default": 1},
        "west": {"type": "number", "default": 2.0},
    },
    "required": ["north", "south"],
}

expected = {mercury: mercury_schema, venus: venus_schema, earth: earth_schema}


# test
def test_schema_gen(func_to_schema=signature_to_json_schema, expected=expected):
    for func, schema in expected.items():
        assert func_to_schema(func) == schema, (
            f"{func=}, \n" f"{func_to_schema(func)=}, \n" f"{schema=}\n"
        )


feature_to_output_mapping_to_test = {
    bool: "boolean",  # note that bool is a subclass of int, so needs to be before
    int: "integer",
    float: "number",
    str: "string",
}

type_feature_switch_to_test = FeatureSwitch(
    featurizer=attrgetter("annotation"),
    feature_to_output_mapping=feature_to_output_mapping_to_test,
    default="string",
)

func_to_schema_to_test = partial(
    signature_to_json_schema, param_to_prop_type=type_feature_switch_to_test
)

# since type_feature_switch_to_test only switches on the annotation
# and mercury has no annotation, the FeatureSwitch will default to 'string'
# so the expected schema needs to be changed for this test
from copy import deepcopy

modified_mercury_schema = deepcopy(mercury_schema)
modified_mercury_schema["properties"]["sour"]["type"] = "string"

test_schema_gen(
    func_to_schema_to_test,
    expected={
        mercury: modified_mercury_schema,
        venus: venus_schema,
        earth: earth_schema,
    },
)


def test_non_json_defaults_are_serializable():
    """Enum, date and arbitrary-object defaults must not break json.dumps (#15)."""
    import datetime
    import enum
    import json
    from ju.rjsf import func_to_form_spec

    class Color(enum.Enum):
        red = "red"
        blue = "blue"

    class Num(enum.IntEnum):
        one = 1

    def f(
        c: Color = Color.red,
        n: Num = Num.one,
        d: datetime.date = datetime.date(2020, 1, 2),
        dt: datetime.datetime = datetime.datetime(2020, 1, 2, 3, 4, 5),
        o: object = object(),
        keep=3,
        none=None,
    ): ...

    spec = func_to_form_spec(f, nest_under_field=None)
    json.dumps(spec)  # does not raise
    props = spec["schema"]["properties"]
    assert props["c"]["default"] == "red" and props["c"]["enum"] == ["red", "blue"]
    assert props["n"]["default"] == 1 and props["n"]["enum"] == [1]
    assert props["d"]["default"] == "2020-01-02"
    assert props["dt"]["default"] == "2020-01-02T03:04:05"
    assert "default" not in props["o"]
    assert props["keep"]["default"] == 3 and props["none"]["default"] is None


def test_dropped_default_keeps_param_optional():
    from ju.rjsf import func_to_form_spec

    """A default with no JSON form is omitted but the parameter stays optional (#15)."""
    import datetime
    import enum

    class P(enum.Enum):
        x = 1
        y = 2

    class Perm(enum.Flag):
        R = 1
        W = 2

    def f(
        a,
        o=object(),
        td=datetime.timedelta(1),
        lst=[object()],
        p: P = P.x,
        perm: Perm = Perm.R | Perm.W,
        ok=[datetime.date(2020, 1, 2)],
    ): ...

    schema = func_to_form_spec(f, nest_under_field=None)["schema"]
    props = schema["properties"]
    assert schema["required"] == ["a"]
    assert "default" not in props["lst"]
    assert props["p"] == {"type": "integer", "default": 1, "enum": [1, 2]}
    assert props["perm"]["default"] == 3 and "enum" not in props["perm"]
    assert props["ok"]["default"] == ["2020-01-02"]
