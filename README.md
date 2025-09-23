# ju

JSON schema Utils

To install:	```pip install ju```

[Documentation](https://i2mint.github.io/ju/)


# Examples

## JSON Schema

You have tools to extract JSON schema information from python objects, as well as 
to create python objects from them.


    >>> from ju import signature_to_json_schema, json_schema_to_signature
    >>>
    >>>
    >>> def earth(north: str, south: bool, east: int = 1, west: float = 2.0):
    ...     """Earth docs"""
    ...     return f'{north=}, {south=}, {east=}, {west=}'
    ...
    >>> schema = signature_to_json_schema(earth)
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

## pydantic utils

The `ju.pydantic_util` module provides comprehensive tools for working with Pydantic models and JSON schemas.

### Creating Models from Data and Schemas

Generate Pydantic models dynamically from data or JSON schemas:

```python
from ju.pydantic_util import data_to_pydantic_model, schema_to_pydantic_model

# Create model from data dictionary
data = {"name": "Alice", "age": 30, "address": {"city": "NYC", "zip": "10001"}}
UserModel = data_to_pydantic_model(data, "User")

# Create model from JSON schema
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "price": {"type": "number"}
    }
}
ProductModel = schema_to_pydantic_model(schema)
```

### Model Validation and Selection

Check data validity and find matching models:

```python
from ju.pydantic_util import is_valid_wrt_model, valid_models

# Check if data is valid for a specific model
is_valid = is_valid_wrt_model({"name": "Bob", "age": 25}, UserModel)

# Find all models that validate given data
models = [UserModel, ProductModel, AdminModel]
matching = list(valid_models(data, models))
```

### Code Generation with Transformations

Convert models and schemas to Python code with optional transformations:

```python
from ju.pydantic_util import pydantic_model_to_code

# Generate code from model
code = pydantic_model_to_code(UserModel)

# Apply transformations during generation
def fix_field_names(schema):
    # Transform problematic field names
    if 'properties' in schema:
        for old_name, new_name in [('class_', 'class_name'), ('type_', 'type_name')]:
            if old_name in schema['properties']:
                schema['properties'][new_name] = schema['properties'].pop(old_name)
    return schema

code = pydantic_model_to_code(
    source_schema, 
    ingress_transform=fix_field_names,
    egress_transform=lambda code: f"# Auto-generated\n{code}"
)
```

### Model Introspection and Data Extraction

Extract structured information about models and use them as data extraction templates:

```python
from ju.pydantic_util import (
    field_paths_and_annotations, 
    model_field_descriptions,
    ModelExtractor
)

# Get flattened field paths and types
paths = field_paths_and_annotations(UserModel)
# {'name': str, 'age': int, 'address.city': str, 'address.zip': str}

# Extract field descriptions
descriptions = model_field_descriptions(UserModel)

# Use models as data extraction templates
extractor = ModelExtractor([UserModel, AdminModel])
data_reader = extractor(json_data)  # Returns mapping with model-based paths
values = data_reader['address.city']  # Extract nested values easily
```

The module handles complex scenarios like nested models, generic types, collections, and provides flexible validation and transformation capabilities.




