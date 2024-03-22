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


from ju.json_schema import DFLT_PARAM_TO_TYPE


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
def _func_to_rjsf_schemas(func, *, param_to_prop_type: Callable = DFLT_PARAM_TO_TYPE):
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
