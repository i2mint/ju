"""Tools for React-JSONSchema-Form (RJSF)"""

from functools import partial
from typing import Callable, Sequence, Mapping
import inspect
from inspect import Parameter


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


# def is_type(param: Parameter, type_: type):
#     return param.annotation is type_ or isinstance(param.default, type_)

from typing import get_args, get_origin, Any, Union, GenericAlias, Type
from types import GenericAlias

SomeType = Union[Type, GenericAlias, Any]
SomeType.__doc__ = "A type or a GenericAlias, but also Any, just in case"


def is_type(param: Parameter, type_: SomeType):
    """
    Checks if the type of a parameter's default value or its annotation matches a
    given type.

    This function handles both regular types and subscripted generics.

    Args:
        param (Parameter): The parameter to check.
        type_ (type): The type to check against.

    Returns:
        bool: True if the parameter's type matches the given type, False otherwise.

    Doctests:
    >>> from inspect import Parameter
    >>> param = Parameter('p', Parameter.KEYWORD_ONLY, default=3.14)
    >>> is_type(param, float)
    True
    >>> is_type(param, int)
    False
    >>> param = Parameter('p', Parameter.KEYWORD_ONLY, default=[1, 2, 3])
    >>> is_type(param, list)
    True
    >>> from typing import List, Union
    >>> is_type(param, List[int])
    True
    >>> is_type(param, List[str])
    False
    >>> is_type(param, Union[int, List[int]])
    True
    """
    if param.annotation is type_:
        return True
    if isinstance(type_, type):
        return isinstance(param.default, type_)
    if hasattr(type_, '__origin__'):
        origin = get_origin(type_)
        if origin is Union:
            args = get_args(type_)
            return any(is_type(param, arg) for arg in args)
        else:
            args = get_args(type_)
            if isinstance(param.default, origin):
                if all(
                    any(isinstance(element, arg) for element in param.default)
                    for arg in args
                ):
                    return True
    return False


from ju.json_schema import DFLT_PY_JSON_TYPE_PAIRS, DFLT_JSON_TYPE


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


_dflt_param_to_type = partial(
    parametrized_param_to_type, type_mapping=DFLT_PY_JSON_TYPE_PAIRS
)


# TODO: The loop body could be factored out
def get_properties(parameters, *, param_to_prop_type):
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
    >>> parameters = inspect.signature(foo).parameters
    >>> assert (
    ...     get_properties(parameters, param_to_prop_type=_dflt_param_to_type)
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
        field['type'] = param_to_prop_type(param)

        # If there's a default value, add it
        if param.default is not inspect.Parameter.empty:
            field['default'] = param.default

        # Add the field to the schema
        properties[name] = field

    return properties


def get_required(properties: dict):
    return [name for name in properties if 'default' not in properties[name]]


# TODO: This all should really use meshed instead, to be easily composable.
def _func_to_rjsf_schemas(func, *, param_to_prop_type: Callable = _dflt_param_to_type):
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

    # Fetch function metadata
    sig = inspect.signature(func)
    parameters = sig.parameters

    # defaults
    schema = {
        'title': func.__name__,
        'type': 'object',
        'properties': {},
        'required': [],
    }
    ui_schema = {
        'ui:submitButtonOptions': {
            'submitText': 'Run',
        }
    }

    schema['properties'] = get_properties(
        parameters, param_to_prop_type=param_to_prop_type
    )
    schema['required'] = get_required(schema['properties'])

    if doc := inspect.getdoc(func):
        schema['description'] = doc

    # Add autofocus to the first field
    if len(parameters) > 0:
        first_param_name = next(iter(parameters))
        ui_schema[first_param_name] = {'ui:autofocus': True}

    # Return the schemas
    return schema, ui_schema
