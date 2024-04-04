"""Tools for React-JSONSchema-Form (RJSF)"""

from typing import Callable, Optional, Union
import inspect
from copy import deepcopy

from ju.json_schema import (
    function_to_json_schema,
    DFLT_PARAM_TO_TYPE,
    DFLT_FUNC_TITLE,
    merge_with_defaults,
)
from ju.util import asis

# TODO: Change to ju.json_schema.pyname_to_title and redo tests
DFLT_PYNAME_TO_TITLE = asis

base_schema = {
    'title': DFLT_FUNC_TITLE,
    'type': 'object',
    'properties': {},
    'required': [],
}

base_ui_schema = {
    'ui:submitButtonOptions': {
        'submitText': 'Run',
    }
}

BASE_RJSF_SPEC = {
    'schema': base_schema,
    'uiSchema': base_ui_schema,
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


def func_to_form_spec(
    func: Callable,
    *,
    doc: Optional[Union[str, bool]] = True,
    param_to_prop_type: Callable = DFLT_PARAM_TO_TYPE,
    nest_under_field: Optional[str] = 'rjsf',
    base_rjsf_spec: dict = BASE_RJSF_SPEC,
    pyname_to_title: Callable[[str], str] = DFLT_PYNAME_TO_TITLE,
):
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
    schema, ui_schema = _func_to_rjsf_schemas(
        func,
        doc=doc,
        param_to_prop_type=param_to_prop_type,
        base_rjsf_spec=base_rjsf_spec,
        pyname_to_title=pyname_to_title,
    )

    # merge these with the base spec
    spec = deepcopy(base_rjsf_spec)
    spec['schema'] = schema
    spec['uiSchema'] = ui_schema

    if nest_under_field:
        return {nest_under_field: spec}
    else:
        return spec


# --------------------------------------------------------------------------------------
# utils


# TODO: This all should really use meshed instead, to be easily composable.
def _func_to_rjsf_schemas(
    func,
    *,
    doc: Optional[Union[str, bool]] = True,
    base_rjsf_spec: dict = BASE_RJSF_SPEC,
    pyname_to_title: Callable[[str], str] = DFLT_PYNAME_TO_TITLE,
    param_to_prop_type: Callable = DFLT_PARAM_TO_TYPE,
):
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

    schema = function_to_json_schema(
        func,
        doc=doc,
        pyname_to_title=pyname_to_title,
        param_to_prop_type=param_to_prop_type,
    )

    ui_schema = deepcopy(base_rjsf_spec['uiSchema'])

    # Add autofocus to the first field
    sig = inspect.signature(func)
    parameters = sig.parameters

    if len(parameters) > 0:
        first_param_name = next(iter(parameters))
        ui_schema[first_param_name] = {'ui:autofocus': True}

    # Return the schemas
    return schema, ui_schema
