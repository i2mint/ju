
# ju
JSON schema Utils


To install:	```pip install ju```


# Examples

## JSON Schema

You have tools to extract JSON schema information from python objects, as well as 
to create python objects from them.


    >>> from ju import function_to_json_schema, json_schema_to_signature
    >>>
    >>>
    >>> def earth(north: str, south: bool, east: int = 1, west: float = 2.0):
    ...     """Earth docs"""
    ...     return f'{north=}, {south=}, {east=}, {west=}'
    ...
    >>> schema = function_to_json_schema(earth)
    >>> assert schema == {
    ...     'description': 'Earth docs',
    ...     'title': 'earth',
    ...     'type': 'object',
    ...     'properties': {
    ...         'north': {'type': 'string'},
    ...         'south': {'type': 'boolean'},
    ...         'east': {'type': 'integer', 'default': 1},
    ...         'west': {'type': 'number', 'default': 2.0},
    ...     },
    ...     'required': ['north', 'south'],
    ... }
    >>>
    >>> sig = json_schema_to_signature(schema)
    >>> sig
    <Sig (north: str, south: bool, east: int = 1, west: float = 2.0)>
    >>> sig.name
    'earth'
    >>> sig.docs
    'Earth docs'


## React JSON Schema Form (rjsf)

You can get a [react-jsonschema-form (rjsf)](https://github.com/rjsf-team/react-jsonschema-form)
specification 
(see the [rjsf playground](https://rjsf-team.github.io/react-jsonschema-form/))
from a python function.


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
        

## OpenAPI Routes

Represents a collection of routes in an OpenAPI specification.

Each instance of this class contains a list of `Route` objects, which can be accessed and manipulated as needed.


    >>> from yaml import safe_load
    >>> spec_yaml = '''
    ... openapi: 3.0.3
    ... paths:
    ...   /items:
    ...     get:
    ...       summary: List items
    ...       responses:
    ...         '200':
    ...           description: An array of items
    ...     post:
    ...       summary: Create item
    ...       responses:
    ...         '201':
    ...           description: Item created
    ... '''
    >>> spec = safe_load(spec_yaml)
    >>> routes = Routes(spec)
    >>> len(routes)
    2
    >>> list(routes)
    [('get', '/items'), ('post', '/items')]
    >>> r = routes['get', '/items']
    >>> r
    Route(method='get', endpoint='/items')
    >>> r.method_data
    {'summary': 'List items', 'responses': {'200': {'description': 'An array of items'}}}

