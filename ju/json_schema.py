"""
This module provides tools to transform Python functions to JSON schemas.

The main function in this module is `function_to_json_schema`, which takes a 
Python function as input and returns a JSON schema that can be used to generate 
a form in a React application.

Example usage:

>>> def mercury(sweet: float, sour=True):
...     '''Near the sun'''
...     return sweet * sour
>>>
>>> assert function_to_json_schema(mercury) == {
...         'title': 'mercury', 
...         'type': 'object', 
...         'properties': {
...             'sweet': {'type': 'number'}, 
...             'sour': {'type': 'string', 'default': True}}, 
...          'required': ['sweet'], 
...          'description': 'Near the sun'
... }

"""

from typing import Mapping, Sequence, Callable, Union
from inspect import Parameter
from functools import partial

from pydantic import BaseModel

from ju.util import is_type


class _BasicPythonTypes(BaseModel):
    a_string: str
    an_integer: int
    a_float: float
    a_boolean: bool
    a_mapping: Mapping
    a_sequence: Sequence[str]


def pydantic_model_to_type_mapping(pydantic_model: BaseModel) -> dict:
    fields = pydantic_model.model_fields
    properties = pydantic_model.model_json_schema().get('properties')

    def gen():
        for k in fields:
            yield fields[k].annotation, properties[k]['type']

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
DFLT_PY_JSON_TYPE_PAIRS = tuple(_DFLT_TYPE_MAPPING.items())
DFLT_JSON_PY_TYPE_PAIRS = tuple((v, k) for k, v in _DFLT_TYPE_MAPPING.items())
assert dict(DFLT_JSON_PY_TYPE_PAIRS) != len(DFLT_JSON_PY_TYPE_PAIRS), (
    f'{DFLT_JSON_PY_TYPE_PAIRS=}, {len(DFLT_JSON_PY_TYPE_PAIRS)=}'
    ' The mapping is not bijective. The ju devs will need to chose a unique python '
    'type for each json type.'
)
DFLT_TYPE_MAPPING = DFLT_PY_JSON_TYPE_PAIRS  # less verbose alias


DFLT_JSON_TYPE = 'string'  # TODO: 'string' or 'object'?


def parametrized_param_to_type(
    param: Parameter,
    *,
    type_mapping=DFLT_PY_JSON_TYPE_PAIRS,
    default=DFLT_JSON_TYPE,
):
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


def print_dict(d):
    print(json.dumps(d, indent=2))


def print_schema(func_key, store):
    print_dict(store[func_key]['rjsf']['schema'])
    print_dict(store[func_key]['rjsf']['schema'])


form_specs = None


def print_schema(func_key='olab.objects.dpp.accuracy', store=form_specs):
    print_dict(store[func_key]['rjsf']['schema'])


def wrap_schema_in_opus_spec(schema: dict):
    return {'rjsf': {'schema': schema}}


# -------------------------------------------------------------------------------------
# The function_to_json_schema function

from typing import Mapping, Sequence
import inspect
from operator import attrgetter
from i2 import Sig, sort_params

from ju.util import FeatureSwitch, Mapper, ensure_callable_mapper

# dflt_type_mapping = {
#     int: {'type': 'integer'},
#     float: {'type': 'number'},
#     bool: {'type': 'boolean'},
#     str: {'type': 'string'},
# }

dflt_json_types = {
    py_type: {'type': json_type} for py_type, json_type in DFLT_PY_JSON_TYPE_PAIRS
}


type_feature_switch = FeatureSwitch(
    featurizer=attrgetter('annotation'),
    feature_to_output_mapping=dflt_json_types,
    default={'type': 'string'},
)


# TODO: See
def function_to_json_schema(
    func: Callable,
    *,
    type_feature_switch=type_feature_switch,
):
    """
    Transforms a Python function to a JSON schema.

    param func: The function to transform
    param type_feature_switch: A function that maps a parameter to a JSON schema type
    return: The JSON schema for the function

    >>> def mercury(sweet: float, sour=True):
    ...     '''Near the sun'''
    ...     return sweet * sour
    >>>
    >>> assert function_to_json_schema(mercury) == {
    ...         'title': 'mercury',
    ...         'type': 'object',
    ...         'properties': {
    ...             'sweet': {'type': 'number'},
    ...             'sour': {'type': 'string', 'default': True}},
    ...          'required': ['sweet'],
    ...          'description': 'Near the sun'
    ... }

    See https://github.com/i2mint/i2//blob/f547257c272433b7651d09276afdfb1bb7b2f67b/misc/i2.routing.ipynb#L17.

    """
    # Fetch function metadata
    sig = inspect.signature(func)
    parameters = sig.parameters

    # Start building the JSON schema
    schema = {
        'title': func.__name__,
        'type': 'object',
        'properties': {},
        'required': [],
    }

    if doc := inspect.getdoc(func):
        schema['description'] = doc

    # Build the schema for each parameter
    for name, param in parameters.items():
        field = type_feature_switch(param)

        # If there's a default value, add it to the schema
        if param.default is not Parameter.empty:
            field['default'] = param.default
        else:
            schema['required'].append(name)

        # Add the field to the schema
        schema['properties'][name] = field

    return schema


def json_schema_to_signature(
    json_schema: dict, *, type_mapper: Mapper = DFLT_JSON_PY_TYPE_PAIRS
):
    """
    Transforms a JSON schema to a Python function signature.

    param schema: The JSON schema to transform
    return: The Python function signature

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


    """
    type_mapper = ensure_callable_mapper(type_mapper)
    properties = json_schema['properties']

    def _params():
        for name, field in properties.items():
            if (json_type := field.get('type', None)) is not None:
                if not isinstance(json_type, list):
                    py_type = type_mapper(json_type)
                else:
                    json_types = map(type_mapper, json_type)
                    # Make a typing.Union of the json_types
                    py_type = Union[tuple(json_types)]
            else:
                py_type = Parameter.empty
            default = field.get('default', Parameter.empty)
            yield Parameter(
                name=name,
                annotation=py_type,
                default=default,
                kind=Parameter.POSITIONAL_OR_KEYWORD,
            )

    sig = Sig(sort_params(_params()))
    if title := json_schema.get('title'):
        sig.name = title
    sig.docs = json_schema.get('description', '')

    return sig
