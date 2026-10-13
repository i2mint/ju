# ju.json_schema

This module provides tools to transform Python functions to JSON schemas.

The main function in this module is `signature_to_json_schema`, which takes a
Python function as input and returns a JSON schema that can be used to generate
a form in a React application.

Example usage:

```pycon
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
```

### Functions

| [`function_to_json_schema`](#ju.json_schema.function_to_json_schema)(func, \*[, doc, ...])    | Transforms a Python function to a JSON schema.                                                                                                                                                                          |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`get_properties`](#ju.json_schema.get_properties)(parameters, \*[, ...])            | Returns the properties dict for the JSON schema.                                                                                                                                                                        |
| `get_required`(properties)                                                                        |                                                                                                                                                                                                                         |
| [`json_schema_to_signature`](#ju.json_schema.json_schema_to_signature)(json_schema, \*[, ...]) | Transforms a JSON schema or OpenAPI parameters list to a Python function signature.                                                                                                                                     |
| [`merge_with_defaults`](#ju.json_schema.merge_with_defaults)(defaults, overwrites)        | Returns a new dictionary that combines two dictionaries by using the keys and order from the first dictionary (defaults) and overwriting its values with those from the second dictionary (overwrites) when they exist. |
| `parametrized_param_to_type`(param, \*[, ...])                                                    |                                                                                                                                                                                                                         |
| `print_dict`(d)                                                                                   |                                                                                                                                                                                                                         |
| `print_schema`([func_key, store])                                                                 |                                                                                                                                                                                                                         |
| `pydantic_model_to_type_mapping`(pydantic_model)                                                  |                                                                                                                                                                                                                         |
| [`pyname_to_title`](#ju.json_schema.pyname_to_title)(pyname)                          | Converts a Python name to a title.                                                                                                                                                                                      |
| [`schema_param_names`](#ju.json_schema.schema_param_names)(json_schema, \*[, ...])       | Map the Python parameter names `json_schema_to_signature` gives to the original names in the schema (the SSOT for the renaming, so callers can build requests with the original names).                                 |
| [`signature_to_json_schema`](#ju.json_schema.signature_to_json_schema)(func, \*[, doc, ...])   | Transforms a Python function to a JSON schema.                                                                                                                                                                          |
| [`title_to_pyname`](#ju.json_schema.title_to_pyname)(title)                           | Converts a title to a Python name.                                                                                                                                                                                      |
| `wrap_schema_in_opus_spec`(schema)                                                                |                                                                                                                                                                                                                         |

### ju.json_schema.function_to_json_schema(func, \*, doc=True, name_of_obj=functools.partial(<function name_of_obj>, default_factory=<function <lambda>>), pyname_to_title=<function asis>, param_to_prop_type=functools.partial(<function parametrized_param_to_type>, type_mapping=((<class 'str'>, 'string'), (<class 'bool'>, 'boolean'), (<class 'int'>, 'integer'), (<class 'float'>, 'number'), (<class 'collections.abc.Mapping'>, 'object'), (collections.abc.Sequence[str], 'array'), (<class 'bytes'>, 'string'), (<class 'bytearray'>, 'string'), (<class 'memoryview'>, 'string'), (<class 'collections.UserString'>, 'string'), (<class 'collections.abc.Sequence'>, 'array'))))

Transforms a Python function to a JSON schema.

param func: The function to transform
return: The JSON schema (as a dict) for the function

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
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
```

See [https://github.com/i2mint/i2//blob/f547257c272433b7651d09276afdfb1bb7b2f67b/misc/i2.routing.ipynb#L17](https://github.com/i2mint/i2//blob/f547257c272433b7651d09276afdfb1bb7b2f67b/misc/i2.routing.ipynb#L17).

### ju.json_schema.get_properties(parameters, \*, param_to_prop_type=functools.partial(<function parametrized_param_to_type>, type_mapping=((<class 'str'>, 'string'), (<class 'bool'>, 'boolean'), (<class 'int'>, 'integer'), (<class 'float'>, 'number'), (<class 'collections.abc.Mapping'>, 'object'), (collections.abc.Sequence[str], 'array'), (<class 'bytes'>, 'string'), (<class 'bytearray'>, 'string'), (<class 'memoryview'>, 'string'), (<class 'collections.UserString'>, 'string'), (<class 'collections.abc.Sequence'>, 'array'))))

Returns the properties dict for the JSON schema.

```pycon
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
```

### ju.json_schema.json_schema_to_signature(json_schema, \*, type_mapper=(('string', <class 'str'>), ('boolean', <class 'bool'>), ('integer', <class 'int'>), ('number', <class 'float'>), ('object', <class 'collections.abc.Mapping'>), ('array', collections.abc.Sequence[str])), default_default=None, title_to_pyname=<function title_to_pyname>, default_annotation, default_description='')

Transforms a JSON schema or OpenAPI parameters list to a Python function signature.
Supports both JSON schema ‘properties’ and OpenAPI ‘parameters’.

Parameters named in the schema’s `required` list (or marked `required: true`,
for OpenAPI parameters) have no default. All others get their schema `default`
if they have one, else `default_default` (`None` unless you say otherwise;
pass `Parameter.empty` to make them required anyway). Required parameters come
first, optional ones after, each group keeping the schema’s order, so positional
calls bind required parameters in schema order.

A ‘properties’ schema with no `required` key at all keeps the legacy behavior
(only properties with their own `default` are optional, `default_default` is
not applied), so positional callers of such schemas are not re-bound. Say
`'required': []` to make everything optional. A malformed `required`
(not a list) raises `ValueError`.

```pycon
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
```

# — OpenAPI parameters support —

```pycon
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
```

Properties not in `required` are optional (`default_default`, i.e. `None`,
unless they have their own default):

```pycon
>>> optional = {'properties': {'a': {'type': 'string'}, 'b': {'type': 'integer'}},
...             'required': ['b']}
>>> json_schema_to_signature(optional)
<Sig (b: int, a: str = None)>
>>> json_schema_to_signature(optional, default_default='')
<Sig (b: int, a: str = '')>
```

### ju.json_schema.merge_with_defaults(defaults, overwrites)

Returns a new dictionary that combines two dictionaries by using the keys
and order from the first dictionary (defaults) and overwriting its values
with those from the second dictionary (overwrites) when they exist.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Parameters

- defaults (dict): The base dictionary containing default key-value pairs.
- overwrites (dict): The dictionary containing values to overwrite those in defaults.

### Returns

- dict: A merged dictionary with values from `overwrites` applied to `defaults`.

### Example

```pycon
>>> defaults = {'a': 1, 'b': 2, 'c': 3}
>>> overwrites = {'b': 4, 'c': 5, 'd': 6}  # note the extra key 'd', not in defaults
>>> merge_with_defaults(defaults, overwrites)
{'a': 1, 'b': 4, 'c': 5}
```

### ju.json_schema.pyname_to_title(pyname)

Converts a Python name to a title.

It does this by replacing underscores with spaces and capitalizing the first
letter of each word.
If the name contains camel case, it will be split into words.

A sort of (imperfect) inverse of `title_to_pyname`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Example

```pycon
>>> pyname_to_title('hello_world')
'Hello World'
>>> pyname_to_title('helloWorld')
'Hello World'
```

### ju.json_schema.schema_param_names(json_schema, \*, title_to_pyname=<function title_to_pyname>)

Map the Python parameter names `json_schema_to_signature` gives to the original
names in the schema (the SSOT for the renaming, so callers can build requests
with the original names).

OpenAPI `parameters` names that already are valid identifiers are kept as is;
others (e.g. `X-MS-API-ROLE`) are sanitised with `title_to_pyname`. Names of
`properties` are always passed through `title_to_pyname`. Clashes get a
numeric suffix.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> schema = {'parameters': [
...     {'name': 'X-MS-API-ROLE', 'in': 'header', 'schema': {'type': 'string'}},
...     {'name': 'thingId', 'in': 'path', 'required': True},
...     {'name': 'x_ms_api_role', 'in': 'query'}]}
>>> schema_param_names(schema)
{'x_ms_api_role': 'X-MS-API-ROLE', 'thingId': 'thingId', 'x_ms_api_role_2': 'x_ms_api_role'}
```

### ju.json_schema.signature_to_json_schema(func, \*, doc=True, name_of_obj=functools.partial(<function name_of_obj>, default_factory=<function <lambda>>), pyname_to_title=<function asis>, param_to_prop_type=functools.partial(<function parametrized_param_to_type>, type_mapping=((<class 'str'>, 'string'), (<class 'bool'>, 'boolean'), (<class 'int'>, 'integer'), (<class 'float'>, 'number'), (<class 'collections.abc.Mapping'>, 'object'), (collections.abc.Sequence[str], 'array'), (<class 'bytes'>, 'string'), (<class 'bytearray'>, 'string'), (<class 'memoryview'>, 'string'), (<class 'collections.UserString'>, 'string'), (<class 'collections.abc.Sequence'>, 'array'))))

Transforms a Python function to a JSON schema.

param func: The function to transform
return: The JSON schema (as a dict) for the function

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
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
```

See [https://github.com/i2mint/i2//blob/f547257c272433b7651d09276afdfb1bb7b2f67b/misc/i2.routing.ipynb#L17](https://github.com/i2mint/i2//blob/f547257c272433b7651d09276afdfb1bb7b2f67b/misc/i2.routing.ipynb#L17).

### ju.json_schema.title_to_pyname(title)

Converts a title to a Python name.

A sort of (imperfect) inverse of `pyname_to_title`.

It does this by replacing spaces with underscores and lowercasing the first
letter of each word.
If the title contains camel case, it will be split into words.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Example

```pycon
>>> title_to_pyname('Hello World')
'hello_world'
>>> title_to_pyname('HelloWorld')
'hello_world'
>>> title_to_pyname('3: Hello World')
'_3_hello_world'
>>> title_to_pyname('   heL----lO  \n\t;; wo__rld')
'_he_l_l_o_wo__rld'
```
