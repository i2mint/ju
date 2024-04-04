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
        'sour': {'type': 'boolean', 'default': True},
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
    bool: 'boolean',  # note that bool is a subclass of int, so needs to be before
    int: 'integer',
    float: 'number',
    str: 'string',
}

type_feature_switch_to_test = FeatureSwitch(
    featurizer=attrgetter('annotation'),
    feature_to_output_mapping=feature_to_output_mapping_to_test,
    default='string',
)

func_to_schema_to_test = partial(
    function_to_json_schema, param_to_prop_type=type_feature_switch_to_test
)

# since type_feature_switch_to_test only switches on the annotation
# and mercury has no annotation, the FeatureSwitch will default to 'string'
# so the expected schema needs to be changed for this test
from copy import deepcopy

modified_mercury_schema = deepcopy(mercury_schema)
modified_mercury_schema['properties']['sour']['type'] = 'string'

test_schema_gen(
    func_to_schema_to_test,
    expected={
        mercury: modified_mercury_schema,
        venus: venus_schema,
        earth: earth_schema,
    },
)
