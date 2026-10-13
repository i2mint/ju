"""
This module provides tools to transform Python functions to JSON schemas.

The main function in this module is `signature_to_json_schema`, which takes a
Python function as input and returns a JSON schema that can be used to generate
a form in a React application.

Example usage:

>>> def mercury(sweet: float, sour=True):
...     '''Near the sun'''
...     return sweet * sour
>>>
>>> assert signature_to_json_schema(mercury) == {
...         'title': 'mercury',
...         'type': 'object',
...         'properties': {
...             'sweet': {'type': 'number'},
...             'sour': {'type': 'boolean', 'default': True}},
...          'required': ['sweet'],
...          'description': 'Near the sun'
... }

"""

from typing import Union, Optional, Any, get_args, get_origin
from collections import UserString
from collections.abc import Mapping, Sequence, Callable
from inspect import Parameter
from functools import partial
import re
import keyword
from copy import deepcopy

from pydantic import BaseModel
from i2 import name_of_obj as _name_of_obj

from ju.util import is_type, unwrap_optional, resolve_annotations

DFLT_FUNC_TITLE = ""

name_of_obj = partial(_name_of_obj, default_factory=lambda: DFLT_FUNC_TITLE)


class _BasicPythonTypes(BaseModel):
    a_string: str
    a_boolean: bool  # isinstance(True, int) so bool needs to be before int
    an_integer: int
    a_float: float
    a_mapping: Mapping
    a_sequence: Sequence[str]


def pydantic_model_to_type_mapping(pydantic_model: BaseModel) -> dict:
    fields = pydantic_model.model_fields
    properties = pydantic_model.model_json_schema().get("properties")

    def gen():
        for k in fields:
            yield fields[k].annotation, properties[k]["type"]

    return dict(gen())


# We use a Pydantic model to get the types of the basic Python types
_DFLT_TYPE_MAPPING = pydantic_model_to_type_mapping(_BasicPythonTypes)

# _DFLT_TYPE_MAPPING = {
#     int: 'integer',
#     float: 'number',
#     str: 'string',
#     bool: 'boolean',
#     Sequence: 'array',
#     Mapping: 'object',
# }

# Need to get a tuple of items to use type mapping as unmutable default.
# Order matters (first match wins): ``Sequence[str]`` only matches a sequence holding a
# str, so a sequence of anything else (or an empty one) is caught by the plain
# ``Sequence`` entry. ``str``/``bytes`` are Sequences too: ``str`` matched earlier, and
# the string- and bytes-like types are pinned to 'string' before ``Sequence`` can
# claim them.
_EXTRA_PY_JSON_TYPE_PAIRS = (
    (bytes, "string"),
    (bytearray, "string"),
    (memoryview, "string"),
    (UserString, "string"),
    (Sequence, "array"),
)
DFLT_PY_JSON_TYPE_PAIRS = tuple(_DFLT_TYPE_MAPPING.items()) + _EXTRA_PY_JSON_TYPE_PAIRS
DFLT_JSON_PY_TYPE_PAIRS = tuple((v, k) for k, v in _DFLT_TYPE_MAPPING.items())
assert dict(DFLT_JSON_PY_TYPE_PAIRS) != len(DFLT_JSON_PY_TYPE_PAIRS), (
    f"{DFLT_JSON_PY_TYPE_PAIRS=}, {len(DFLT_JSON_PY_TYPE_PAIRS)=}"
    " The mapping is not bijective. The ju devs will need to chose a unique python "
    "type for each json type."
)
DFLT_TYPE_MAPPING = DFLT_PY_JSON_TYPE_PAIRS  # less verbose alias


DFLT_JSON_TYPE = "string"  # TODO: 'string' or 'object'?


def parametrized_param_to_type(
    param: Parameter,
    *,
    type_mapping=DFLT_PY_JSON_TYPE_PAIRS,
    default=DFLT_JSON_TYPE,
):
    # TODO: validate type_mapping. (namely child before parent)
    #   If parent was before child, the child would never be reached.
    for python_type, json_type in type_mapping:
        if is_type(param, python_type):
            return json_type
    return default


DFLT_PARAM_TO_TYPE = partial(
    parametrized_param_to_type, type_mapping=DFLT_PY_JSON_TYPE_PAIRS
)

# -------------------------------------------------------------------------------------
# util

import json
import enum
import datetime


def print_dict(d):
    print(json.dumps(d, indent=2))


def print_schema(func_key, store):
    print_dict(store[func_key]["rjsf"]["schema"])
    print_dict(store[func_key]["rjsf"]["schema"])


form_specs = None


def print_schema(func_key="olab.objects.dpp.accuracy", store=form_specs):
    print_dict(store[func_key]["rjsf"]["schema"])


def wrap_schema_in_opus_spec(schema: dict):
    return {"rjsf": {"schema": schema}}


def merge_with_defaults(defaults: Mapping, overwrites: Mapping) -> dict:
    """
    Returns a new dictionary that combines two dictionaries by using the keys
    and order from the first dictionary (defaults) and overwriting its values
    with those from the second dictionary (overwrites) when they exist.

    Parameters:
    - defaults (dict): The base dictionary containing default key-value pairs.
    - overwrites (dict): The dictionary containing values to overwrite those in defaults.

    Returns:
    - dict: A merged dictionary with values from `overwrites` applied to `defaults`.

    Example:

    >>> defaults = {'a': 1, 'b': 2, 'c': 3}
    >>> overwrites = {'b': 4, 'c': 5, 'd': 6}  # note the extra key 'd', not in defaults
    >>> merge_with_defaults(defaults, overwrites)
    {'a': 1, 'b': 4, 'c': 5}

    """
    return {key: overwrites.get(key, defaults[key]) for key in defaults}


def pyname_to_title(pyname: str) -> str:
    """
    Converts a Python name to a title.

    It does this by replacing underscores with spaces and capitalizing the first
    letter of each word.
    If the name contains camel case, it will be split into words.

    A sort of (imperfect) inverse of `title_to_pyname`.

    Example:
    >>> pyname_to_title('hello_world')
    'Hello World'
    >>> pyname_to_title('helloWorld')
    'Hello World'

    """
    # split camel case
    pyname = re.sub("([a-z0-9])([A-Z])", r"\1 \2", pyname)
    return pyname.replace("_", " ").title()


# TODO: Replace asis with pyname_to_title when time to update tests
# DFLT_PYNAME_TO_TITLE = pyname_to_title
from ju.util import asis

DFLT_PYNAME_TO_TITLE = asis


def title_to_pyname(title: str) -> str:
    r"""
    Converts a title to a Python name.

    A sort of (imperfect) inverse of `pyname_to_title`.

    It does this by replacing spaces with underscores and lowercasing the first
    letter of each word.
    If the title contains camel case, it will be split into words.

    Example:
    >>> title_to_pyname('Hello World')
    'hello_world'
    >>> title_to_pyname('HelloWorld')
    'hello_world'
    >>> title_to_pyname('3: Hello World')
    '_3_hello_world'
    >>> title_to_pyname('   heL----lO  \n\t;; wo__rld')
    '_he_l_l_o_wo__rld'

    """
    # if title begins with a digit, prepend an underscore
    original_title = title
    if title[0].isdigit():
        title = "_" + title
    # split camel case
    title = re.sub("([a-z0-9])([A-Z])", r"\1 \2", title)
    # remove non-alphanumeric characters, replacing them with a single space
    title = re.sub(r"\W+", " ", title)
    # replace any sequence of spaces with a single underscore
    title = re.sub(r"\s+", "_", title)
    pyname = title.lower()  # TODO: Should we? Get control over this?
    # assert pyname is a valid Python identifier
    assert (
        pyname.isidentifier()
    ), f"Invalid Python identifier: {pyname}, computed from title: {original_title}"
    return pyname


# -------------------------------------------------------------------------------------
# The signature_to_json_schema function

from collections.abc import Mapping, Sequence
import inspect
from operator import attrgetter
from i2 import Sig, sort_params
from i2.signatures import ParamsAble

from ju.util import FeatureSwitch, Mapper, ensure_callable_mapper

# dflt_type_mapping = {
#     int: {'type': 'integer'},
#     float: {'type': 'number'},
#     bool: {'type': 'boolean'},
#     str: {'type': 'string'},
# }

dflt_json_types = {
    py_type: {"type": json_type} for py_type, json_type in DFLT_PY_JSON_TYPE_PAIRS
}


type_feature_switch = FeatureSwitch(
    featurizer=attrgetter("annotation"),
    feature_to_output_mapping=dflt_json_types,
    default={"type": "string"},
)

BASE_SCHEMA = {
    "title": DFLT_FUNC_TITLE,
    "type": "object",
    "properties": {},
    "required": [],
}


# TODO: See rjsf _func_to_rjsf_schemas and merge with it
def signature_to_json_schema(
    func: ParamsAble,
    *,
    doc: str | bool | None = True,
    name_of_obj: str | Callable[[Any], str] = name_of_obj,
    pyname_to_title: Callable[[str], str] = DFLT_PYNAME_TO_TITLE,
    param_to_prop_type: Callable = DFLT_PARAM_TO_TYPE,
) -> dict:
    """
    Transforms a Python function to a JSON schema.

    param func: The function to transform
    return: The JSON schema (as a dict) for the function

    >>> def mercury(sweet: float, sour=True):
    ...     '''Near the sun'''
    ...     return sweet * sour
    >>>
    >>> assert signature_to_json_schema(mercury) == {
    ...         'title': 'mercury',
    ...         'type': 'object',
    ...         'properties': {
    ...             'sweet': {'type': 'number'},
    ...             'sour': {'type': 'boolean', 'default': True}},
    ...          'required': ['sweet'],
    ...          'description': 'Near the sun'
    ... }

    See https://github.com/i2mint/i2//blob/f547257c272433b7651d09276afdfb1bb7b2f67b/misc/i2.routing.ipynb#L17.

    """
    # Fetch function metadata
    sig = Sig(func)

    if isinstance(name_of_obj, str):
        func_name = name_of_obj
    else:
        func_name = name_of_obj(func)

    parameters = sig.parameters
    if callable(func):  # (``from __future__ import annotations`` -> strings)
        parameters = resolve_annotations(func, parameters)

    schema = deepcopy(BASE_SCHEMA)
    schema["title"] = pyname_to_title(func_name)

    schema["properties"] = get_properties(
        parameters, param_to_prop_type=param_to_prop_type
    )
    # (not get_required(properties): a default dropped for having no JSON form must
    # not turn its parameter into a required one)
    schema["required"] = [
        name
        for name in schema["properties"]
        if parameters[name].default is Parameter.empty
    ]

    if doc is True:
        # if doc is True, use the function docstring
        doc = inspect.getdoc(func)

    if doc:
        assert isinstance(doc, str), f"doc must be a string, was: {doc}"
        schema["description"] = doc

    # # Build the schema for each parameter
    # for name, param in parameters.items():
    #     field = type_feature_switch(param)

    #     # If there's a default value, add it to the schema
    #     if param.default is not Parameter.empty:
    #         field['default'] = param.default
    #     else:
    #         schema['required'].append(name)

    #     # Add the field to the schema
    #     schema['properties'][name] = field

    return schema


function_to_json_schema = signature_to_json_schema  # backwards compatibility


#: Keys of a `properties` entry that stand for an OpenAPI parameter or body property
#: whose own name was taken by another (``id`` in the query and in a header): the
#: entry is keyed by a distinct name, and these record the spec's name and place.
OAS_NAME_KEY = "x-oas-name"
OAS_IN_KEY = "x-oas-in"


def schema_param_names(
    json_schema: dict,
    *,
    title_to_pyname: Callable[[str], str] = title_to_pyname,
) -> dict[str, str]:
    """Map the Python parameter names `json_schema_to_signature` gives to the original
    names in the schema (the SSOT for the renaming, so callers can build requests
    with the original names).

    OpenAPI ``parameters`` names that already are valid identifiers are kept as is;
    others (e.g. ``X-MS-API-ROLE``) are sanitised with ``title_to_pyname``. Names of
    ``properties`` are always passed through ``title_to_pyname``. Clashes get a
    numeric suffix. A ``properties`` entry carrying ``x-oas-name`` (see ``Route.params``)
    maps to that name, not to its key.

    >>> schema = {'parameters': [
    ...     {'name': 'X-MS-API-ROLE', 'in': 'header', 'schema': {'type': 'string'}},
    ...     {'name': 'thingId', 'in': 'path', 'required': True},
    ...     {'name': 'x_ms_api_role', 'in': 'query'}]}
    >>> schema_param_names(schema)
    {'x_ms_api_role': 'X-MS-API-ROLE', 'thingId': 'thingId', 'x_ms_api_role_2': 'x_ms_api_role'}

    >>> schema = {'properties': {'id': {}, 'id_header': {'x-oas-name': 'id'}}}
    >>> schema_param_names(schema)
    {'id': 'id', 'id_header': 'id'}
    """
    sent_as = {}  # property key -> the spec's name for it, when it is not the key
    if "parameters" in json_schema and isinstance(json_schema["parameters"], list):
        originals = [param["name"] for param in json_schema["parameters"]]
        keep_valid = True
    elif "properties" in json_schema:
        originals = list(json_schema["properties"])
        keep_valid = False
        sent_as = {
            key: prop[OAS_NAME_KEY]
            for key, prop in json_schema["properties"].items()
            if isinstance(prop, Mapping) and OAS_NAME_KEY in prop
        }
    else:
        return {}
    names = {}  # pyname -> original
    for original in originals:
        if keep_valid and original.isidentifier() and not keyword.iskeyword(original):
            pyname = original
        else:
            try:
                pyname = title_to_pyname(original)
            except (AssertionError, IndexError):  # e.g. '' or only symbols
                pyname = "_"
            if keyword.iskeyword(pyname):
                pyname += "_"
        candidate, i = pyname, 1
        while candidate in names:
            i += 1
            candidate = f"{pyname}_{i}"
        names[candidate] = sent_as.get(original, original)
    return names


def json_schema_to_signature(
    json_schema: dict,
    *,
    type_mapper: Mapper = DFLT_JSON_PY_TYPE_PAIRS,
    default_default=None,
    title_to_pyname: Callable[[str], str] = title_to_pyname,
    default_annotation=Parameter.empty,
    default_description: str = "",
):
    """
    Transforms a JSON schema or OpenAPI parameters list to a Python function signature.
    Supports both JSON schema 'properties' and OpenAPI 'parameters'.

    Parameters named in the schema's ``required`` list (or marked ``required: true``,
    for OpenAPI parameters) have no default. All others get their schema ``default``
    if they have one, else ``default_default`` (``None`` unless you say otherwise;
    pass ``Parameter.empty`` to make them required anyway). Required parameters come
    first, optional ones after, each group keeping the schema's order, so positional
    calls bind required parameters in schema order.

    A 'properties' schema with no ``required`` key at all keeps the legacy behavior
    (only properties with their own ``default`` are optional, ``default_default`` is
    not applied), so positional callers of such schemas are not re-bound. Say
    ``'required': []`` to make everything optional. A malformed ``required``
    (not a list) raises ``ValueError``.

    >>> schema = {'title': 'earth',
    ...  'type': 'object',
    ...  'properties': {'north': {'type': 'string'},
    ...   'south': {'type': 'boolean'},
    ...   'east': {'type': 'integer', 'default': 1},
    ...   'west': {'type': 'number', 'default': 2.0}},
    ...  'required': ['north', 'south'],
    ...  'description': 'Earth docs'}
    >>> sig = json_schema_to_signature(schema)
    >>> sig
    <Sig (north: str, south: bool, east: int = 1, west: float = 2.0)>
    >>> sig.name
    'earth'
    >>> sig.docs
    'Earth docs'

    # --- OpenAPI parameters support ---
    >>> openapi_params = {
    ...     'title': 'get_thing',
    ...     'parameters': [
    ...         {'name': 'thing_id', 'in': 'path', 'required': True, 'schema': {'type': 'string'}},
    ...         {'name': 'detail', 'in': 'query', 'required': False, 'schema': {'type': 'boolean', 'default': False}},
    ...         {'name': 'count', 'in': 'query', 'schema': {'type': 'integer', 'default': 1}},
    ...     ],
    ...     'description': 'Get a thing by ID.'
    ... }
    >>> sig2 = json_schema_to_signature(openapi_params)
    >>> sig2
    <Sig (thing_id: str, detail: bool = False, count: int = 1)>
    >>> sig2.name
    'get_thing'
    >>> sig2.docs
    'Get a thing by ID.'

    Properties not in ``required`` are optional (``default_default``, i.e. ``None``,
    unless they have their own default):

    >>> optional = {'properties': {'a': {'type': 'string'}, 'b': {'type': 'integer'}},
    ...             'required': ['b']}
    >>> json_schema_to_signature(optional)
    <Sig (b: int, a: str = None)>
    >>> json_schema_to_signature(optional, default_default='')
    <Sig (b: int, a: str = '')>
    """
    type_mapper = ensure_callable_mapper(type_mapper, default=default_annotation)
    params = []
    # pyname -> original name, in schema order (the SSOT for the renaming)
    pynames = iter(schema_param_names(json_schema, title_to_pyname=title_to_pyname))
    # Support OpenAPI 'parameters' (list of dicts)
    if "parameters" in json_schema and isinstance(json_schema["parameters"], list):
        for param in json_schema["parameters"]:
            name = next(pynames)
            # Try to get type from schema/type, fallback to string
            py_type = type_mapper(param.get("schema", {}).get("type", "string"))
            default = param.get("schema", {}).get("default", default_default)
            if param.get("required", False):
                default = Parameter.empty
            params.append(
                Parameter(
                    name=name,
                    annotation=py_type,
                    default=default,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
    # Support JSON schema 'properties' (as before)
    elif "properties" in json_schema:
        properties = json_schema["properties"]
        required = _required_names(json_schema)  # None when there is no such key
        for name, field in properties.items():
            is_required = required is not None and name in required
            name = next(pynames)
            if (json_type := field.get("type", None)) is not None:
                if not isinstance(json_type, list):
                    py_type = type_mapper(json_type)
                else:
                    json_types = map(type_mapper, json_type)
                    py_type = Union[tuple(json_types)]
            else:
                py_type = Parameter.empty
            if is_required:
                default = Parameter.empty
            elif required is None:  # legacy: no 'required' key, so no opinion
                default = field.get("default", Parameter.empty)
            else:
                default = field.get("default", default_default)
            params.append(
                Parameter(
                    name=name,
                    annotation=py_type,
                    default=default,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
    else:
        # Fallback: no parameters
        params = []
    sig = Sig(sort_params(params))
    if title := json_schema.get("title"):
        sig.name = title_to_pyname(title)
    sig.docs = json_schema.get("description", default_description)
    return sig


# -------------------------------------------------------------------------------------
# Utils


def _required_names(json_schema: dict):
    """Names in the schema's top-level ``required`` list; None if there is no such key."""
    if "required" not in json_schema:
        return None
    required = json_schema["required"]
    if not isinstance(required, (list, tuple, set, frozenset)):
        raise ValueError(
            f"'required' must be a list of property names, got: {required!r}"
        )
    return frozenset(required)


# TODO: The loop body could be factored out
def _json_safe(value):
    """Return ``(ok, json_value)``: ``value`` coerced to something ``json.dumps`` accepts.

    ``ok`` is False if there is no sensible JSON form (the caller should omit it).

    >>> import enum, datetime
    >>> class Color(enum.Enum):
    ...     red = 'red'
    >>> _json_safe(Color.red)
    (True, 'red')
    >>> _json_safe(datetime.date(2020, 1, 2))
    (True, '2020-01-02')
    >>> _json_safe(object())
    (False, None)
    """
    if isinstance(value, enum.Enum):
        return _json_safe(value.value)
    if isinstance(value, (datetime.date, datetime.time)):  # incl. datetime
        return True, value.isoformat()
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        items = [_json_safe(v) for v in value]
        if not all(ok for ok, _ in items):
            return False, None
        return True, (
            type(value)(v for _, v in items) if type(value) in (list, tuple) else value
        )
    if isinstance(value, dict):
        items = {k: _json_safe(v) for k, v in value.items()}
        if not all(ok for ok, _ in items.values()):
            return False, None
        return True, (
            {k: v for k, (_, v) in items.items()} if type(value) is dict else value
        )
    try:
        json.dumps(value)
    except (TypeError, ValueError):  # not serializable (or circular)
        return False, None
    return True, value


def _array_items(
    annotation, default, param_to_prop_type: Callable, name: str = "_"
) -> dict | None:
    """
    The JSON schema of an array parameter's items, or ``None`` if there is no telling.

    The element type comes from the annotation's arguments (``list[int]``,
    ``tuple[int, ...]``, ``Sequence[list[str]]``), else from the default's elements
    when they all have the same JSON type.

    >>> from ju.json_schema import DFLT_PARAM_TO_TYPE as to_type
    >>> _array_items(list[int], inspect.Parameter.empty, to_type)
    {'type': 'integer'}
    >>> _array_items(list[list[str]], inspect.Parameter.empty, to_type)
    {'type': 'array', 'items': {'type': 'string'}}
    >>> _array_items(inspect.Parameter.empty, [1, 2, 3], to_type)
    {'type': 'integer'}
    >>> _array_items(inspect.Parameter.empty, [1, 'a'], to_type) is None
    True
    >>> _array_items(list, [], to_type) is None
    True
    """
    empty = inspect.Parameter.empty
    args = get_args(unwrap_optional(annotation)) if annotation is not empty else ()
    args = tuple(a for a in args if a is not Ellipsis)
    if len(args) == 1:  # list[int], tuple[int, ...]; not tuple[int, str]
        return _value_schema(args[0], empty, param_to_prop_type, name)
    if args:
        return None
    if isinstance(default, (list, tuple)) and default:
        schemas = [_value_schema(empty, d, param_to_prop_type, name) for d in default]
        if all(schema == schemas[0] for schema in schemas):
            return schemas[0]
    return None


def _value_schema(
    annotation, default, param_to_prop_type: Callable, name: str = "_"
) -> dict:
    """``{"type": ..., "items": ...}`` for a value described by an annotation and/or a
    default. ``name`` is that of the parameter the value belongs to (so a custom
    ``param_to_prop_type`` sees it, not a placeholder)."""
    param = Parameter(
        name, Parameter.KEYWORD_ONLY, annotation=annotation, default=default
    )
    schema = {"type": param_to_prop_type(param)}
    if schema["type"] == "array":
        items = _array_items(annotation, default, param_to_prop_type, name)
        if items is not None:
            schema["items"] = items
    return schema


def get_properties(parameters, *, param_to_prop_type: Callable = DFLT_PARAM_TO_TYPE):
    """
    Returns the properties dict for the JSON schema.

    >>> def foo(
    ...     a_bool: bool,
    ...     a_float=3.14,
    ...     an_int=2,
    ...     a_str: str = 'hello',
    ...     something_else=None
    ... ):
    ...     '''A Foo function'''
    >>>
    >>> from ju.json_schema import DFLT_PARAM_TO_TYPE
    >>> parameters = inspect.signature(foo).parameters
    >>> assert (
    ...     get_properties(parameters, param_to_prop_type=DFLT_PARAM_TO_TYPE)
    ...     == {
    ...         'a_bool': {'type': 'boolean'},
    ...         'a_float': {'type': 'number', 'default': 3.14},
    ...         'an_int': {'type': 'integer', 'default': 2},
    ...         'a_str': {'type': 'string', 'default': 'hello'},
    ...         'something_else': {'type': 'string', 'default': None}
    ...     }
    ... )

    """
    # Build the properties dict
    properties = {}
    for i, item in enumerate(parameters.items()):
        name, param = item
        field = {}
        field["type"] = param_to_prop_type(param)
        if field["type"] == "array":
            items = _array_items(
                param.annotation, param.default, param_to_prop_type, name
            )
            if items is not None:
                field["items"] = items

        # If there's a default value, add it
        if param.default is not inspect.Parameter.empty:
            # Defaults with no JSON form (arbitrary objects) are omitted, so the
            # schema stays serializable; Enum members become their value (plus the
            # list of allowed values) and dates become ISO strings.
            ok, default = _json_safe(param.default)
            if ok:
                field["default"] = default
            if ok and isinstance(param.default, enum.Enum):
                allowed = [_json_safe(m) for m in type(param.default)]
                values = [v for _, v in allowed]
                # (Flag members can be combined, so their members don't list the
                # allowed values.)
                if all(ok for ok, _ in allowed) and not isinstance(
                    param.default, enum.Flag
                ):
                    field["enum"] = values
                # the annotation may not say what the enum's values are
                if isinstance(default, bool):
                    field["type"] = "boolean"
                elif isinstance(default, int):
                    field["type"] = "integer"
                elif isinstance(default, float):
                    field["type"] = "number"
                elif isinstance(default, str):
                    field["type"] = "string"

        # Add the field to the schema
        properties[name] = field

    return properties


def get_required(properties: dict):
    return [name for name in properties if "default" not in properties[name]]
