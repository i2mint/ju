"""
Test objects from the ju.rjsf module
"""

from functools import partial
from typing import Callable
import inspect
from operator import attrgetter
from ju.json_schema import function_to_json_schema
from ju.util import FeatureSwitch


def mercury(sweet: float, sour=True):
    return sweet * sour


def venus():
    """Nothing from nothing"""


def earth(north: str, south: bool, east: int = 1, west: float = 2.0):
    """Earth docs"""
    return f'{north=}, {south=}, {east=}, {west=}'


mercury_schema = {
    'title': 'mercury',
    'type': 'object',
    'properties': {
        'sweet': {'type': 'number'},
        'sour': {'type': 'string', 'default': True},
    },
    'required': ['sweet'],
}

venus_schema = {
    'title': 'venus',
    'type': 'object',
    'properties': {},
    'required': [],
    'description': 'Nothing from nothing',
}

earth_schema = {
    'description': 'Earth docs',
    'title': 'earth',
    'type': 'object',
    'properties': {
        'north': {'type': 'string'},
        'south': {'type': 'boolean'},
        'east': {'type': 'integer', 'default': 1},
        'west': {'type': 'number', 'default': 2.0},
    },
    'required': ['north', 'south'],
}

expected = {mercury: mercury_schema, venus: venus_schema, earth: earth_schema}


# test
def test_schema_gen(func_to_schema=function_to_json_schema, expected=expected):
    for func, schema in expected.items():
        assert func_to_schema(func) == schema, (
            f'{func=}, \n' f'{func_to_schema(func)=}, \n' f'{schema=}\n'
        )


feature_to_output_mapping_to_test = {
    int: {'type': 'integer'},
    float: {'type': 'number'},
    bool: {'type': 'boolean'},
    str: {'type': 'string'},
}

type_feature_switch_to_test = FeatureSwitch(
    featurizer=attrgetter('annotation'),
    feature_to_output_mapping=feature_to_output_mapping_to_test,
    default={'type': 'string'},
)

func_to_schema_to_test = partial(
    function_to_json_schema, type_feature_switch=type_feature_switch_to_test
)

test_schema_gen(func_to_schema_to_test)

# -------------------------------------------------------------------------------------
# old version of transform


import inspect


def static_function_to_json_schema(func: Callable):
    """A static (hardcoded) version of function_to_json_schema.
    It's used to test the dynamic version.
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
        if param.annotation is int:
            field = {'type': 'integer'}
        elif param.annotation is float:
            field = {'type': 'number'}
        elif param.annotation is bool:
            field = {'type': 'boolean'}
        else:
            field = {'type': 'string'}

        # If there's a default value, add it to the schema
        if param.default is not inspect.Parameter.empty:
            field['default'] = param.default
        else:
            schema['required'].append(name)

        # Add the field to the schema
        schema['properties'][name] = field

    return schema
