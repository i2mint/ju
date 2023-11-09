"""Tools for React-JSONSchema-Form (RJSF)"""

from functools import partial
from typing import Callable
import inspect


def func_to_form_spec(func: Callable):
    """
    Returns a JSON object that can be used as a form specification, along with the
    function, to generate a FuncCaller React component in a React application.

    param func: The function to transform
    return: The form specification for the function

    >>> def foo(
    ...     a_bool: bool,
    ...     a_float=3.14,
    ...     an_int=2,
    ...     a_str: str = 'hello',
    ...     something_else=None
    ... ):
    ...     '''A Foo function'''
    >>>
    >>> form_spec = func_to_form_spec(foo)
    >>> assert form_spec == {
    ...     'rjsf': {
    ...         'schema': {
    ...             'title': 'foo',
    ...             'type': 'object',
    ...             'properties': {
    ...                 'a_bool': {'type': 'boolean'},
    ...                 'a_float': {'type': 'number', 'default': 3.14},
    ...                 'an_int': {'type': 'integer', 'default': 2},
    ...                 'a_str': {'type': 'string', 'default': 'hello'},
    ...                 'something_else': {'type': 'string', 'default': None}
    ...             },
    ...             'required': ['a_bool'],
    ...             'description': 'A Foo function'
    ...         },
    ...         'uiSchema': {
    ...             'ui:submitButtonOptions': {
    ...                 'submitText': 'Run'
    ...             },
    ...             'a_bool': {'ui:autofocus': True}
    ...         },
    ...         'liveValidate': False,
    ...         'disabled': False,
    ...         'readonly': False,
    ...         'omitExtraData': False,
    ...         'liveOmit': False,
    ...         'noValidate': False,
    ...         'noHtml5Validate': False,
    ...         'focusOnFirstError': False,
    ...         'showErrorList': 'top'
    ...     }
    ... }
    """
    schema, ui_schema = _func_to_rjsf_schemas(func)

    # Return the form spec
    return {
        'rjsf': {
            'schema': schema,
            'uiSchema': ui_schema,
            'liveValidate': False,
            'disabled': False,
            'readonly': False,
            'omitExtraData': False,
            'liveOmit': False,
            'noValidate': False,
            'noHtml5Validate': False,
            'focusOnFirstError': False,
            'showErrorList': 'top',
        }
    }


def _func_to_rjsf_schemas(func):
    """
    Returns the JSON schema and the UI schema for a function.

    param func: The function to transform
    return: The JSON schema and the UI schema for the function

    >>> def foo(
    ...     a_bool: bool,
    ...     a_float=3.14,
    ...     an_int=2,
    ...     a_str: str = 'hello',
    ...     something_else=None
    ... ):
    ...     '''A Foo function'''
    >>>
    >>> schema, ui_schema = _func_to_rjsf_schemas(foo)
    >>> assert schema == {
    ...     'title': 'foo',
    ...     'type': 'object',
    ...     'properties': {
    ...         'a_bool': {'type': 'boolean'},
    ...         'a_float': {'type': 'number', 'default': 3.14},
    ...         'an_int': {'type': 'integer', 'default': 2},
    ...         'a_str': {'type': 'string', 'default': 'hello'},
    ...         'something_else': {'type': 'string', 'default': None}
    ...     },
    ...     'required': ['a_bool'],
    ...     'description': 'A Foo function'
    ... }
    >>> assert ui_schema == {
    ...     'ui:submitButtonOptions': {'submitText': 'Run'},
    ...     'a_bool': {'ui:autofocus': True}
    ... }
    """

    def is_type(param, type_):
        return param.annotation is type_ or isinstance(param.default, type_)

    def treat_params():
        # Build the schema for each parameter
        for i, item in enumerate(parameters.items()):
            name, param = item
            if is_type(param, bool):
                field = {'type': 'boolean'}
            elif is_type(param, float):
                field = {'type': 'number'}
            elif is_type(param, int):
                field = {'type': 'integer'}
            else:
                field = {'type': 'string'}

                # If there's a default value, add it to the schema
            if param.default is not inspect.Parameter.empty:
                field['default'] = param.default
            else:
                schema['required'].append(name)

                # Add the field to the schema
            schema['properties'][name] = field

            # Add autofocus to the first field
            if i == 0:
                ui_schema[name] = {'ui:autofocus': True}

    # Fetch function metadata
    sig = inspect.signature(func)
    parameters = sig.parameters

    # Start building the JSON schema and the UI schema
    schema = {
        'title': func.__name__,
        'type': 'object',
        'properties': {},
        'required': [],
    }
    ui_schema = {'ui:submitButtonOptions': {'submitText': 'Run',}}

    if doc := inspect.getdoc(func):
        schema['description'] = doc

    treat_params()

    # Return the schemas
    return schema, ui_schema


"""
This module provides tools to transform Python functions to React-JSONSchema-Form specifications.

The main function in this module is `transform_function_to_schema`, which takes a Python function as input and returns a JSON schema that can be used to generate a form in a React application.

The module also provides some helper functions to transform specific types of Python objects to JSON schema types, such as `transform_string_to_schema` and `transform_integer_to_schema`.

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
# utils for routing

from typing import Mapping, Callable
from functools import partial


def display_dag_of_code(func, *args, **kwargs):
    from meshed import code_to_dag

    return code_to_dag(func).dot_digraph(*args, **kwargs)


def apply(func, obj):
    return func(obj)


def _switch_case(mapping, default, feature):
    return mapping.get(feature, default)


def switch_case(mapping, default):
    return partial(_switch_case, mapping, default)


def _feature_based_search(
    feature_processor_pairs, feature_similarity, default, feature
):
    if isinstance(feature_processor_pairs, Mapping):
        feature_processor_pairs = feature_processor_pairs.items()
    feature_matches = partial(feature_similarity, feature)
    for feature_compared_to, then_ in feature_processor_pairs:
        if feature_matches(feature_compared_to):
            return then_
    return default


def feature_based_search(feature_processor_pairs, feature_similarity, default):
    return partial(
        _feature_based_search, feature_processor_pairs, feature_similarity, default
    )


def feature_switch(obj, *, featurizer, feature_to_output_mapping, default):
    feature = apply(featurizer, obj)
    get_output_for_feature = switch_case(feature_to_output_mapping, default)
    output = apply(get_output_for_feature, feature)
    return output


def feature_similarity_search(
    obj,
    *,
    featurizer,
    feature_based_search,
    feature_output_pairs,
    feature_similarity,
    similarity_base_match=lambda x, y: x == y,
):
    feature = apply(obj, featurizer)
    get_output_for_feature = feature_based_search(
        feature_output_pairs, feature_similarity, similarity_base_match
    )
    output = apply(get_output_for_feature, feature)
    return output


# -------------------------------------------------------------------------------------
# The function_to_json_schema function

import inspect
from operator import attrgetter
from i2 import FuncFactory

FeatureSwitch = FuncFactory(feature_switch)

dflt_type_mapping = {
    int: {'type': 'integer'},
    float: {'type': 'number'},
    bool: {'type': 'boolean'},
    str: {'type': 'string'},
}

type_feature_switch = FeatureSwitch(
    featurizer=attrgetter('annotation'),
    feature_to_output_mapping=dflt_type_mapping,
    default={'type': 'string'},
)


def function_to_json_schema(
    func: Callable, *, type_feature_switch=type_feature_switch,
):
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
        if param.default is not inspect.Parameter.empty:
            field['default'] = param.default
        else:
            schema['required'].append(name)

        # Add the field to the schema
        schema['properties'][name] = field

    return schema


# -------------------------------------------------------------------------------------
# tests


def mercury(sweet: float, sour=True):
    return sweet * sour


def venus():
    """Nothing from nothing"""


def earth(north: str, south: bool, east: int = 1, west: float = 2.0):
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


def _static_function_to_json_schema(func: Callable):
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
