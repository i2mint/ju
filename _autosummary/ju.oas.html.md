# ju.oas

OpenAPI specification tools.

### Module Attributes

| [`BODY_MEDIA_TYPES`](#ju.oas.BODY_MEDIA_TYPES)   | Non-JSON request body media types `Route.params` reads, in order of preference.    |
|---------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`BODY_IN`](#ju.oas.BODY_IN)            | The `OAS_IN_KEY` of a property that comes from the request body.                   |
| [`PARAM_IN`](#ju.oas.PARAM_IN)           | The `OAS_IN_KEY` of a parameter that does not say where it goes (invalid OpenAPI). |

### Functions

| [`add_property`](#ju.oas.add_property)(schema, sent_as, spec_names, ...)   | Add the parameter or body property `name`, sent in `place`, to the object `schema`, keyed by its name; when that is taken (the same name in another place -- `id` in the query and in a header, a query `name` and a body `name`), by a distinct key.                                                                                                                                                                               |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `compare_modules_to_missing_funcs`()                                                              |                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `compare_modules_to_operationids`()                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [`default_func_namer`](#ju.oas.default_func_namer)(method, path[, details, ...]) | Default function name generator for OpenAPI routes.                                                                                                                                                                                                                                                                                                                                                                                 |
| `default_get_response`(method, url, \*\*kwargs)                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `default_response_egress`(method, uri)                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [`ensure_openapi_dict`](#ju.oas.ensure_openapi_dict)(spec)                        | Ensure that the OpenAPI specification is a dictionary.                                                                                                                                                                                                                                                                                                                                                                              |
| [`generate_and_import_openapi_client`](#ju.oas.generate_and_import_openapi_client)(openapi_spec) | openapi_spec: dict, or str (path to YAML/JSON file) output_dir: where to generate the client (default: temp dir) file_format: 'yaml' or 'json' (used if openapi_spec is a dict)                                                                                                                                                                                                                                                     |
| [`generate_openapi_client`](#ju.oas.generate_openapi_client)(openapi_spec[, ...])     | openapi_spec: dict, or str (path to YAML/JSON file) output_dir: where to generate the client (default: temp dir) file_format: 'yaml' or 'json' (used if openapi_spec is a dict)                                                                                                                                                                                                                                                     |
| [`get_routes`](#ju.oas.get_routes)(d[, include_methods])                 | Takes OpenAPI specification dict 'd' and returns the key-paths to all the endpoints.                                                                                                                                                                                                                                                                                                                                                |
| [`merge_parameters`](#ju.oas.merge_parameters)(path_item_params, ...)          | The parameters of an operation: those its path item declares for all its operations, overridden (same name and `in`) or extended by its own.                                                                                                                                                                                                                                                                                        |
| [`merge_request_body_json_schema`](#ju.oas.merge_request_body_json_schema)(details, ...)     | If the operation has a requestBody with a JSON schema, merge its properties and required fields into param_schema.                                                                                                                                                                                                                                                                                                                  |
| [`openapi_to_funcs`](#ju.oas.openapi_to_funcs)(spec, \*[, base_url, ...])      | spec: dict or str (YAML/JSON string or file path) base_url: override the spec's server URL default_servers_url: default URL to use if no servers are specified in the spec get_response: function to get the response object (default: requests.request) response_egress: function (method, uri) -> callable to extract result from response Returns a generator yielding OpenApiFunc instances for each route in the OpenAPI spec. |
| [`openapi_to_generated_funcs`](#ju.oas.openapi_to_generated_funcs)(spec, \*[, ...])      | Like openapi_to_funcs, but yields OpenApiFunc objects for each route, using the generated client for requests.                                                                                                                                                                                                                                                                                                                      |
| `print_generated_modules`()                                                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [`properties_of_schema`](#ju.oas.properties_of_schema)(schema)                     | Returns the properties of the given schema, encapsulating in ArrayOf to indicate that the schema is for an array of objects, and not just a single object.                                                                                                                                                                                                                                                                          |
| [`resolve_refs`](#ju.oas.resolve_refs)(open_api_spec, d)                   | Recursively resolves all references in 'd' using 'open_api_spec'.                                                                                                                                                                                                                                                                                                                                                                   |
| `return_empty_dict_on_error`(e)                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [`validate_openapi_to_generated_funcs_alignment`](#ju.oas.validate_openapi_to_generated_funcs_alignment)()  | Example usage of openapi_to_generated_funcs.                                                                                                                                                                                                                                                                                                                                                                                        |

### Classes

| [`ArrayOf`](#ju.oas.ArrayOf)                                            | A class that is simply meant to mark the fact that some properties dict really represents an array of objects, and not just a single object.   |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| [`OpenApiFunc`](#ju.oas.OpenApiFunc)(\*, method, uri, base_url, ...[, ...]) | Callable class for OpenAPI route functions, supporting introspection and pickling.                                                             |
| [`Route`](#ju.oas.Route)(method, endpoint, spec[, type_mapping])      | Represents a route in an OpenAPI specification.                                                                                                |
| [`Routes`](#ju.oas.Routes)(spec, \*[, type_mapping])                   | Represents a collection of routes in an OpenAPI specification.                                                                                 |

### *class* ju.oas.ArrayOf

Bases: [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

A class that is simply meant to mark the fact that some properties dict really
represents an array of objects, and not just a single object.

### ju.oas.BODY_IN *= 'body'*

The `OAS_IN_KEY` of a property that comes from the request body.

### ju.oas.BODY_MEDIA_TYPES *= ('application/x-www-form-urlencoded', 'multipart/form-data')*

Non-JSON request body media types `Route.params` reads, in order of preference.

### *class* ju.oas.OpenApiFunc(\*, method, uri, base_url, param_schema, get_response=<function default_get_response>, response_egress=None, name=None, qualname=None, doc=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Callable class for OpenAPI route functions, supporting introspection and pickling.

### ju.oas.PARAM_IN *= 'param'*

The `OAS_IN_KEY` of a parameter that does not say where it goes (invalid OpenAPI).

### *class* ju.oas.Route(method, endpoint, spec, type_mapping=(('array', <class 'list'>), ('integer', <class 'int'>), ('object', <class 'dict'>), ('string', <class 'str'>), ('boolean', <class 'bool'>), ('number', <class 'float'>)))

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Represents a route in an OpenAPI specification.

Each route has a method (e.g., ‘get’, ‘post’), an endpoint (e.g., ‘/items’), and a spec, which is a dictionary
containing the details of the route as specified in the OpenAPI document.

The `type_mapping` attribute is a dictionary that maps OpenAPI types to corresponding Python types.

```pycon
>>> from yaml import safe_load
>>> spec_yaml = '''
... openapi: 3.0.3
... paths:
...   /items:
...     get:
...       summary: List items
...       parameters:
...         - in: query
...           name: type
...           schema:
...             type: string
...           required: true
...           description: Type of items to list
...       responses:
...         '200':
...           description: An array of items
... '''
>>> spec = safe_load(spec_yaml)
>>> route_get = Route('get', '/items', spec)
>>> route_get.method
'get'
>>> route_get.endpoint
'/items'
>>> route_get.method_data['summary']
'List items'
>>> route_get.params
{'type': 'object', 'properties': {'type': {'type': 'string'}}, 'required': ['type']}
```

#### *property* body_content_type

The media type of the request body `params` reads, or None for no body.

JSON is preferred; then form-urlencoded, then multipart.

```pycon
>>> from yaml import safe_load
>>> spec = safe_load('''
... paths:
...   /f:
...     post:
...       requestBody:
...         content:
...           multipart/form-data: {schema: {properties: {a: {type: string}}}}
... ''')
>>> route = Route('post', '/f', spec)
>>> route.body_content_type, list(route.params['properties'])
('multipart/form-data', ['a'])
```

#### *property* output_properties

Returns the schema for the response with the given status code.

#### *property* params

Combined parameters from parameters and requestBody
(it should usually just be one or the other, not both).
We’re calling this ‘params’ because that’s what FastAPI calls it.

### *class* ju.oas.Routes(spec, \*, type_mapping=(('array', <class 'list'>), ('integer', <class 'int'>), ('object', <class 'dict'>), ('string', <class 'str'>), ('boolean', <class 'bool'>), ('number', <class 'float'>)))

Bases: `Store`

Represents a collection of routes in an OpenAPI specification.

Each instance of this class contains a list of `Route` objects, which can be accessed and manipulated as needed.

```pycon
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
```

#### update(\*\*F) → None.  Update D from mapping/iterable E and F.

If E present and has a .keys() method, does:     for k in E.keys(): D[k] = E[k]
If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
In either case, this is followed by: for k, v in F.items(): D[k] = v

#### update_keys_cache(keys)

Updates the \_keys_cache by deleting the attribute

### ju.oas.add_property(schema, sent_as, spec_names, name, place, prop_schema, required)

Add the parameter or body property `name`, sent in `place`, to the object
`schema`, keyed by its name; when that is taken (the same name in another
place – `id` in the query and in a header, a query `name` and a body
`name`), by a distinct key. Both then record the spec’s name and place
(OAS_NAME_KEY, OAS_IN_KEY), so a consumer can tell which is which. No distinct
key is a name in `spec_names`. `sent_as` (key -> (name, place)) is the state
to carry between calls.

```pycon
>>> schema, sent_as = {'properties': {}, 'required': []}, {}
>>> add_property(schema, sent_as, {'id'}, 'id', 'query', {'type': 'string'}, True)
>>> add_property(schema, sent_as, {'id'}, 'id', 'header', {'type': 'integer'}, False)
>>> sorted(schema['properties'])
['id', 'id_header']
>>> schema['properties']['id_header']['x-oas-in'], schema['required']
('header', ['id'])
```

### ju.oas.default_func_namer(method, path, details=None, , favor_operation_id=False)

Default function name generator for OpenAPI routes.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> default_func_namer('get', '/stores/{store_name}/{key}')
'get_stores__store_name__key'
```

### ju.oas.ensure_openapi_dict(spec)

Ensure that the OpenAPI specification is a dictionary.

It will handle:

- JSON strings
- YAML strings
- File paths to JSON or YAML files
- URLs pointing to OpenAPI specs
- Direct dictionaries

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ju.oas.generate_and_import_openapi_client(openapi_spec, output_dir=None, , file_format='json')

openapi_spec: dict, or str (path to YAML/JSON file)
output_dir: where to generate the client (default: temp dir)
file_format: ‘yaml’ or ‘json’ (used if openapi_spec is a dict)

* **Returns:**
  imported client module

### ju.oas.generate_openapi_client(openapi_spec, output_dir=None, , file_format='json')

openapi_spec: dict, or str (path to YAML/JSON file)
output_dir: where to generate the client (default: temp dir)
file_format: ‘yaml’ or ‘json’ (used if openapi_spec is a dict)

* **Returns:**
  imported client module

### ju.oas.get_routes(d, include_methods=('get', 'post', 'head', 'put', 'delete', 'options', 'patch'))

Takes OpenAPI specification dict ‘d’ and returns the key-paths to all the endpoints.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ju.oas.merge_parameters(path_item_params, operation_params)

The parameters of an operation: those its path item declares for all its
operations, overridden (same name and `in`) or extended by its own.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

```pycon
>>> merge_parameters(
...     [{'name': 'id', 'in': 'path'}, {'name': 'q', 'in': 'query', 'x': 1}],
...     [{'name': 'q', 'in': 'query', 'x': 2}, {'name': 'q', 'in': 'header'}],
... )
[{'name': 'id', 'in': 'path'}, {'name': 'q', 'in': 'query', 'x': 2}, {'name': 'q', 'in': 'header'}]
```

### ju.oas.merge_request_body_json_schema(details, param_schema)

If the operation has a requestBody with a JSON schema, merge its properties and required fields into param_schema.

### ju.oas.openapi_to_funcs(spec, \*, base_url=None, default_servers_url='http://localhost:8000', func_namer=<function default_func_namer>, get_response=<function default_get_response>, response_egress=None, use_default_func_namer_when_name_is_none=True)

spec: dict or str (YAML/JSON string or file path)
base_url: override the spec’s server URL
default_servers_url: default URL to use if no servers are specified in the spec
get_response: function to get the response object (default: requests.request)
response_egress: function (method, uri) -> callable to extract result from response
Returns a generator yielding OpenApiFunc instances for each route in the OpenAPI spec.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`OpenApiFunc`](#ju.oas.OpenApiFunc)]

### ju.oas.openapi_to_generated_funcs(spec, \*, base_url=None, default_servers_url='http://localhost:8000', output_dir=None, file_format='json', func_namer=<function default_func_namer>, get_response=<function default_get_response>, response_egress=None)

Like openapi_to_funcs, but yields OpenApiFunc objects for each route, using the generated client for requests.
Uses func_namer for naming, and parameter schema from the OpenAPI spec.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`OpenApiFunc`](#ju.oas.OpenApiFunc)]

### ju.oas.properties_of_schema(schema)

Returns the properties of the given schema, encapsulating in ArrayOf to indicate
that the schema is for an array of objects, and not just a single object.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### ju.oas.resolve_refs(open_api_spec, d)

Recursively resolves all references in ‘d’ using ‘open_api_spec’.

* **Parameters:**
  * **open_api_spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The complete OpenAPI specification as a dictionary.
  * **d** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – The dictionary in which references need to be resolved.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  The dictionary with all references resolved.

### ju.oas.validate_openapi_to_generated_funcs_alignment()

Example usage of openapi_to_generated_funcs.
This function is just for demonstration and should be removed in production code.
