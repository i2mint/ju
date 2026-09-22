# ju.pydantic_util

Tools for working with Pydantic models.

This module provides comprehensive utilities for creating, validating, transforming,
and extracting information from Pydantic models and JSON schemas.

### Key Functionality

### Model Creation and Validation

- `mk_pydantic_model`: Create Pydantic model instances with customizable validation
- `mk_pydantic_models`: Batch version for multiple models
- `is_valid_wrt_model`: Check if data is valid against a model
- `valid_models`: Find all models that validate given data
- `data_to_pydantic_model`: Dynamically create models from data dictionaries

### Schema and Code Generation

- `pydantic_model_to_code`: Convert schemas/models to Pydantic code with transforms
- `schema_to_pydantic_model_simple/advanced`: Convert JSON schemas to Pydantic models

### Model Introspection and Analysis

- `model_field_descriptions`: Extract field descriptions from models
- `field_paths_and_annotations`: Get flattened field paths and types
- `match_typevars_to_args`: Resolve generic type variables
- `is_a_basemodel`: Check if object is a Pydantic BaseModel

### Data Extraction

- `ModelExtractor`: Extract data using model schemas as templates
- Supports nested models and collection types with path notation (e.g., ‘items.\*.ref’)

### Type Classification

- `is_pydantic_model`: Detect Pydantic model classes
- `is_typing_type`: Detect typing module types
- `is_type_hint`: Combined type hint detection

### Error Handling

- `extract_friendly_errors`: Convert ValidationError to user-friendly messages

### Schema Transformations

The `pydantic_model_to_code` function supports:

- `ingress_transform`: Transform schemas before code generation
- `egress_transform`: Transform generated code
- Multiple source types: dicts, models, JSON strings, JSON files

### Examples

```python
# Create models from data
model = data_to_pydantic_model({"name": "John", "age": 30}, "User")

# Generate code with transformations
def fix_field_names(schema):
    # Transform problematic field names
    return schema

code = pydantic_model_to_code(model, ingress_transform=fix_field_names)

# Extract data using model schemas
extractor = ModelExtractor([UserModel, AdminModel])
data_reader = extractor(json_data)  # Returns KeysReader with model-based paths

# Validate data against multiple models
valid_model_list = list(valid_models(data, [Model1, Model2, Model3]))
```

### Functions

| [`PydanticModelFactory`](#ju.pydantic_util.PydanticModelFactory)(schema[, model_name])         | Converts a JSON schema (with 'properties') to a Pydantic model.                                             |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [`data_to_pydantic_model`](#ju.pydantic_util.data_to_pydantic_model)(data[, name, ...])          | Generate a dynamic Pydantic model, optionally creating nested models for nested dictionaries.               |
| [`extract_friendly_errors`](#ju.pydantic_util.extract_friendly_errors)(e)                         | Extracts a generator of user-friendly error messages from a Pydantic ValidationError.                       |
| [`field_paths_and_annotations`](#ju.pydantic_util.field_paths_and_annotations)(data_model)            | Get flattened field paths and their corresponding annotations from a Pydantic model.                        |
| [`fix_generated_code_transform`](#ju.pydantic_util.fix_generated_code_transform)(code)                 | Fix common issues in generated code.                                                                        |
| [`infer_json_friendly_type`](#ju.pydantic_util.infer_json_friendly_type)(value)                    | Infers the type of the value for Pydantic model field.                                                      |
| [`is_a_basemodel`](#ju.pydantic_util.is_a_basemodel)(obj)                                | Check if an object is a Pydantic BaseModel.                                                                 |
| [`is_pydantic_model`](#ju.pydantic_util.is_pydantic_model)(obj)                             | Returns True if the object is a Pydantic model (subclass of BaseModel).                                     |
| [`is_type_hint`](#ju.pydantic_util.is_type_hint)(obj)                                  | Returns True if the object is a Pydantic model or a typing type.                                            |
| [`is_typing_type`](#ju.pydantic_util.is_typing_type)(obj)                                | Returns True if the object is a typing type (e.g., List, Literal, etc.).                                    |
| [`is_valid_wrt_model`](#ju.pydantic_util.is_valid_wrt_model)(data, model, \*[, factory])     | Check if a json object is valid wrt to a pydantic model.                                                    |
| [`jsonschema_to_openapi_transform`](#ju.pydantic_util.jsonschema_to_openapi_transform)(schema)            | Transform JSON Schema to OpenAPI-compatible schema.                                                         |
| [`match_typevars_to_args`](#ju.pydantic_util.match_typevars_to_args)(generic_model)              | Given a Pydantic generic model, returns a mapping of type variables to their concrete types.                |
| [`mk_pydantic_model`](#ju.pydantic_util.mk_pydantic_model)(data, model, \*[, factory, ...]) | Make a Pydantic model instance from data, parametrizing constructor and error handling.                     |
| [`mk_pydantic_models`](#ju.pydantic_util.mk_pydantic_models)(data, models, \*[, ...])        | The iterable-of-models version of `mk_pydantic_model`.                                                      |
| [`model_field_descriptions`](#ju.pydantic_util.model_field_descriptions)(model[, ...])             | Extracts a dictionary of field paths and their descriptions from a Pydantic model, including nested models. |
| [`pydantic_model_to_code`](#ju.pydantic_util.pydantic_model_to_code)(source, \*[, ...])          | Convert a model source (json string, dict, or pydantic model) to pydantic code.                             |
| [`schema_to_pydantic_model`](#ju.pydantic_util.schema_to_pydantic_model)(schema[, model_name])     | Converts a JSON schema (with 'properties') to a Pydantic model.                                             |
| [`schema_to_pydantic_model_simple`](#ju.pydantic_util.schema_to_pydantic_model_simple)(schema[, ...])     | Converts a JSON schema (with 'properties') to a Pydantic model.                                             |
| [`valid_models`](#ju.pydantic_util.valid_models)(json_obj, models, \*[, factory])      | A generator that yields the models that json_obj is valid with respect to.                                  |

### Classes

| [`ModelExtractor`](#ju.pydantic_util.ModelExtractor)(models, \*[, getter])   | Extracts key paths and corresponding values from data based on matching Pydantic models.   |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|

### *class* ju.pydantic_util.ModelExtractor(models, \*, getter=<function glom>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Extracts key paths and corresponding values from data based on matching Pydantic models.

`ModelExtractor` takes a collection of models and extracts all valid key paths from
their (nested) schemas. When called on data, it identifies the first model that
matches the structure of the data and returns a `KeysReader`, which is a mapping
of key paths to the corresponding values in the data.

A `KeysReader` instance is a mapping that gives you lazy-evaluated access to values.
With such an instance, you can list the keys (paths) that are valid according to
the schema of the matched model, and extract the corresponding values from the data
at the moment you need them.

* **Parameters:**
  * **models** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `BaseModel`] | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`BaseModel`]) – A dictionary mapping model names to models, or an iterable of models.
    If an iterable is provided, it will be converted to a dictionary using
    the model names.
  * **getter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function used to extract values from the data based on the identified
    paths. Defaults to `glom.glom`, a tool for nested data extraction.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If the provided models do not have unique names, either as an iterable or
      in the dictionary keys.

### Example

```pycon
>>> from typing import List
>>> from pydantic import BaseModel
>>>
>>> class Item(BaseModel):
...     ref: int
>>> class Playlist(BaseModel):
...     name: str
...     items: List[Item]
>>> class User(BaseModel):
...     name: str
...     age: int
>>>
>>> models = [Playlist, User]
>>> extractors = ModelExtractor(models)
>>> data = {"name": "Digital Reveries", "items": [{"ref": 6}, {"ref": 42}]}
>>> d = extractors(data)
>>> list(d)
['name', 'items.*.ref']
>>> d['name']
'Digital Reveries'
>>> d['items.*.ref']
[6, 42]
```

The example shows how the `ModelExtractor` class automatically detects the model
(in this case, `Playlist`), retrieves the paths defined by the model schema
(e.g., ‘name’ and ‘items.\*.ref’), and extracts the corresponding values from the data.

#### getter(spec, \*\*kwargs)

Access or construct a value from a given *target* based on the
specification declared by *spec*.

Accessing nested data, aka deep-get:

```pycon
>>> target = {'a': {'b': 'c'}}
>>> glom(target, 'a.b')
'c'
```

Here the *spec* was just a string denoting a path,
`'a.b'`. As simple as it should be. You can also use
[`glob`](https://docs.python.org/3/library/glob.html#module-glob)-like wildcard selectors:

```pycon
>>> target = {'a': [{'k': 'v1'}, {'k': 'v2'}]}
>>> glom(target, 'a.*.k')
['v1', 'v2']
```

In addition to `*`, you can also use `**` for recursive access:

```pycon
>>> target = {'a': [{'k': 'v3'}, {'k': 'v4'}], 'k': 'v0'}
>>> glom(target, '**.k')
['v0', 'v3', 'v4']
```

The next example shows how to use nested data to
access many fields at once, and make a new nested structure.

Constructing, or restructuring more-complicated nested data:

```pycon
>>> target = {'a': {'b': 'c', 'd': 'e'}, 'f': 'g', 'h': [0, 1, 2]}
>>> spec = {'a': 'a.b', 'd': 'a.d', 'h': ('h', [lambda x: x * 2])}
>>> output = glom(target, spec)
>>> pprint(output)
{'a': 'c', 'd': 'e', 'h': [0, 2, 4]}
```

`glom` also takes a keyword-argument, *default*. When set,
if a `glom` operation fails with a `GlomError`, the
*default* will be returned, very much like
[`dict.get()`](https://docs.python.org/3/builtins/stdtypes.html#dict.get):

```pycon
>>> glom(target, 'a.xx', default='nada')
'nada'
```

The *skip_exc* keyword argument controls which errors should
be ignored.

```pycon
>>> glom({}, lambda x: 100.0 / len(x), default=0.0, skip_exc=ZeroDivisionError)
0.0
```

* **Parameters:**
  * **target** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) – the object on which the glom will operate.
  * **spec** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) – Specification of the output object in the form
    of a dict, list, tuple, string, other glom construct, or
    any composition of these.
  * **default** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) – An optional default to return in the case
    an exception, specified by *skip_exc*, is raised.
  * **skip_exc** ([*Exception*](https://docs.python.org/3/builtins/exceptions.html#Exception)) – An optional exception or tuple of
    exceptions to ignore and return *default* (None if
    omitted). If *skip_exc* and *default* are both not set,
    glom raises errors through.
  * **scope** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Additional data that can be accessed
    via S inside the glom-spec. Read more: scope.

It’s a small API with big functionality, and glom’s power is
only surpassed by its intuitiveness. Give it a whirl!

### ju.pydantic_util.PydanticModelFactory(schema, model_name='AutoModel')

Converts a JSON schema (with ‘properties’) to a Pydantic model.
Only supports basic types and required fields for demonstration.

* **Return type:**
  [`type`](https://docs.python.org/3/builtins/functions.html#type)[`BaseModel`]

```pycon
>>> schema_simple = {
...     "type": "object",
...     "properties": {
...         "name": {"type": "string"},
...         "age": {"type": "integer"},
...         "city": {"type": "string", "default": "New York"}
...     },
...     "required": ["name"]
... }
>>> model = schema_to_pydantic_model_simple(schema_simple, "User")
>>> model.__name__
'User'
>>> model.model_json_schema()['properties']['name']
{'title': 'Name', 'type': 'string'}
>>> age_schema = model.model_json_schema()['properties']['age']
>>> assert age_schema['title'] == 'Age'
>>> # Ensure both integer and null are allowed (order-independent)
>>> assert {'type': 'integer'} in age_schema.get('anyOf', []) and {'type': 'null'} in age_schema.get('anyOf', [])
>>> city_schema = model.model_json_schema()['properties']['city']
>>> assert city_schema['title'] == 'City' and city_schema['default'] == 'New York' and city_schema['type'] == 'string'
>>> model.model_json_schema()['required']
['name']
```

### ju.pydantic_util.data_to_pydantic_model(data, name='DataBasedModel', , defaults=None, create_nested_models=True, mk_nested_name=None)

Generate a dynamic Pydantic model, optionally creating nested models for nested dictionaries.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Name of the Pydantic model to create.
  * **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – A dictionary representing the structure of the model.
  * **defaults** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A dictionary specifying default values for certain fields.
  * **create_nested_models** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, create nested models for nested dictionaries.
* **Returns:**
  A dynamically created Pydantic model, with nested models if applicable.

```pycon
>>> json_data = {
...     "name": "John", "age": 30, "address": {"city": "New York", "zipcode": "10001"}
... }
>>> defaults = {"age": 18}
>>>
>>> M = data_to_pydantic_model(json_data, "M", defaults=defaults)
>>>
>>> model_instance_custom = M(
... name="John", age=25, address={"city": "Mountain View", "zipcode": "94043"}
... )
>>> model_instance_custom.model_dump()
{'name': 'John', 'age': 25, 'address': {'city': 'Mountain View', 'zipcode': '94043'}}
>>> model_instance_with_defaults = M(
...     name="Jane", address={"city": "Los Angeles", "zipcode": "90001"}
... )
>>> model_instance_with_defaults.model_dump()
{'name': 'Jane', 'age': 18, 'address': {'city': 'Los Angeles', 'zipcode': '90001'}}
```

And note that the nested model is also created:

```pycon
>>> M.Address(city="New York", zipcode="10001")
Address(city='New York', zipcode='10001')
```

### ju.pydantic_util.extract_friendly_errors(e)

Extracts a generator of user-friendly error messages from a Pydantic ValidationError.

### ju.pydantic_util.field_paths_and_annotations(data_model)

Get flattened field paths and their corresponding annotations from a Pydantic model.

Generates a dictionary of dot-separated paths and their corresponding types
from the fields of a given Pydantic BaseModel and any nested BaseModels within it.

The function recursively traverses the fields of the BaseModel and its nested models,
including fields that are lists, sets, tuples, or iterables containing BaseModels.
If the field is a collection containing a BaseModel, the path is marked with a ‘\*’.

This structure is compatible with the `glom` library, allowing extraction of values
from a dictionary that matches the BaseModel structure.

* **Parameters:**
  **data_model** ([`type`](https://docs.python.org/3/builtins/functions.html#type)[`BaseModel`]) – The Pydantic BaseModel to extract field paths and annotations from.
* **Returns:**
  A dictionary where the keys are the dot-separated paths to fields
  : and the values are their corresponding types.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`type`](https://docs.python.org/3/builtins/functions.html#type)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Example

```pycon
>>> from pydantic import BaseModel
>>> from typing import List
```

```pycon
>>> class BItem(BaseModel):
...     c: int
```

```pycon
>>> class A(BaseModel):
...     b: List[BItem]
...     d: str
```

```pycon
>>> class Model(BaseModel):
...     a: A
```

```pycon
>>> paths = field_paths_and_annotations(Model)
>>> expected_paths = {'a.b.*.c': int, 'a.d': str}
>>> assert paths == expected_paths, f"Expected: {expected_paths}, but got: {paths}"
```

See that it works with generics:

```pycon
>>> from typing import TypeVar, List, Generic
>>> T = TypeVar('T')
>>> class A_with_Generic(BaseModel, Generic[T]):
...     b: List[T]
...     d: str
>>> class Model_with_Generic(BaseModel):
...     a: A_with_Generic[BItem]
>>>
>>> field_paths_and_annotations(Model_with_Generic)
{'a.b.*.c': <class 'int'>, 'a.d': <class 'str'>}
```

### ju.pydantic_util.fix_generated_code_transform(code)

Fix common issues in generated code.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ju.pydantic_util.infer_json_friendly_type(value)

Infers the type of the value for Pydantic model field.

```pycon
>>> infer_json_friendly_type(42)
<class 'int'>
>>> infer_json_friendly_type("Hello, World!")
<class 'str'>
>>> infer_json_friendly_type({"key": "value"})
<class 'dict'>
```

### ju.pydantic_util.is_a_basemodel(obj)

Check if an object is a Pydantic BaseModel.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> from typing import List
>>> class MyModel(BaseModel):
...     '''Some model'''
>>> list(map(is_a_basemodel, [BaseModel, MyModel, 3.14, int, List[MyModel]]))
[True, True, False, False, False]
```

### ju.pydantic_util.is_pydantic_model(obj)

Returns True if the object is a Pydantic model (subclass of BaseModel).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ju.pydantic_util.is_type_hint(obj)

Returns True if the object is a Pydantic model or a typing type.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ju.pydantic_util.is_typing_type(obj)

Returns True if the object is a typing type (e.g., List, Literal, etc.).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### ju.pydantic_util.is_valid_wrt_model(data, model, \*, factory=<function \_model_validate>)

Check if a json object is valid wrt to a pydantic model.

### ju.pydantic_util.jsonschema_to_openapi_transform(schema)

Transform JSON Schema to OpenAPI-compatible schema.
This handles $ref resolution and other incompatibilities.

Uses jsonschema2pydantic library approach.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ju.pydantic_util.match_typevars_to_args(generic_model)

Given a Pydantic generic model, returns a mapping of type variables to their
concrete types.

* **Parameters:**
  **generic_model** ([`type`](https://docs.python.org/3/builtins/functions.html#type)[`BaseModel`]) – A generic Pydantic model (e.g., Pair[int, str]).
* **Returns:**
  A dictionary mapping type variables (e.g., T and U)
  to their concrete types (e.g., int and str).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`TypeVar`, bound= <attribute ‘_\_bound_\_’ of ‘typing.TypeVar’ objects>, covariant=<member ‘_\_covariant_\_’ of ‘typing.TypeVar’ objects>, contravariant=<member ‘_\_contravariant_\_’ of ‘typing.TypeVar’ objects>), [`type`](https://docs.python.org/3/builtins/functions.html#type)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

```pycon
>>> from typing import TypeVar, Generic, List
```

```pycon
>>> T = TypeVar('T')
>>> U = TypeVar('U')
```

```pycon
>>> class Pair(Generic[T, U]):
...     first: T
...     second: U
```

```pycon
>>> X = Pair[int, str]
```

```pycon
>>> match_typevars_to_args(X)
{~T: <class 'int'>, ~U: <class 'str'>}
```

### ju.pydantic_util.mk_pydantic_model(data, model, \*, factory=<function \_model_validate>, error_callback=<function \_raise_error>)

Make a Pydantic model instance from data, parametrizing constructor and error handling.

By default, it uses the `model.model_validate` method, but you can pass a custom
constructor function and error handling callback.

* **Parameters:**
  * **data** ([`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Data`, bound= [`Any`](https://docs.python.org/3/library/typing.html#typing.Any))) – A dictionary representing the data to be validated.
  * **model** ([`type`](https://docs.python.org/3/builtins/functions.html#type)[`BaseModel`]) – A Pydantic model class.
  * **factory** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`type`](https://docs.python.org/3/builtins/functions.html#type)[`BaseModel`], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Data`, bound= [`Any`](https://docs.python.org/3/library/typing.html#typing.Any))], `BaseModel`]) – A callable used to construct the model instance.
    Defaults to `model.model_validate`.
  * **error_callback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A callback to handle validation errors.
* **Return type:**
  `BaseModel`
* **Returns:**
  A Pydantic model instance.

### Example

```pycon
>>> from pydantic import BaseModel
>>> class User(BaseModel):
...     name: str
...     code: int
...
>>> data = {"name": "John", "code": 30}
>>> user = mk_pydantic_model(data, User)
>>> user
User(name='John', code=30)
```

Example with custom constructor:

```pycon
>>> user = mk_pydantic_model(data, User, factory=lambda model, data: model.model_construct(**data))
>>> user
User(name='John', code=30)
```

### ju.pydantic_util.mk_pydantic_models(data, models, \*, factory=<function \_model_validate>, error_callback=<function \_raise_error>)

The iterable-of-models version of `mk_pydantic_model`.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`BaseModel`]

### ju.pydantic_util.model_field_descriptions(model, default_description='No description provided', , prefix='')

Extracts a dictionary of field paths and their descriptions from a Pydantic model,
including nested models.

* **Parameters:**
  * **model** ([`type`](https://docs.python.org/3/builtins/functions.html#type)[`BaseModel`]) – A Pydantic model class.
  * **prefix** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A prefix for nested fields (used internally during recursion).
* **Returns:**
  A dictionary of field paths and descriptions.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### Example

```pycon
>>> from pydantic import BaseModel, Field
>>> class Address(BaseModel):
...     city: str = Field(..., description="City name")
...     zipcode: str = Field(..., description="ZIP code")
>>> class User(BaseModel):
...     name: str = Field(..., description="The name of the user")
...     address: Address
>>> model_field_descriptions(User)
{'name': 'The name of the user',
 'address.city': 'City name',
 'address.zipcode': 'ZIP code'}
```

### ju.pydantic_util.pydantic_model_to_code(source, , ingress_transform=None, egress_transform=None, \*\*extra_json_schema_parser_kwargs)

Convert a model source (json string, dict, or pydantic model) to pydantic code.

Requires having datamodel-code-generator installed (pip install datamodel-code-generator)

Code was based on: [https://koxudaxi.github.io/datamodel-code-generator/using_as_module/](https://koxudaxi.github.io/datamodel-code-generator/using_as_module/)

See also this free online converter: [https://jsontopydantic.com/](https://jsontopydantic.com/)

* **Parameters:**
  * **ingress_transform** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Function to transform schema before generation
  * **egress_transform** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Function to transform generated code
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> json_schema: str = '''{
...     "type": "object",
...     "properties": {
...         "number": {"type": "number"},
...         "street_name": {"type": "string"},
...         "street_type": {"type": "string",
...                         "enum": ["Street", "Avenue", "Boulevard"]
...                         }
...     }
... }'''
>>> print(pydantic_model_to_code(json_schema))
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class StreetType(Enum):
    Street = 'Street'
    Avenue = 'Avenue'
    Boulevard = 'Boulevard'


class Model(BaseModel):
    number: Optional[float] = None
    street_name: Optional[str] = None
    street_type: Optional[StreetType] = None
```

This means you can get some model code from an example data dict,
using pydantic_model_to_code

```pycon
>>> M = data_to_pydantic_model({"name": "John", "age": 30}, "Simple")
>>> print(pydantic_model_to_code(M))
from __future__ import annotations

from pydantic import BaseModel, Field


class Simple(BaseModel):
    name: str = Field(..., title='Name')
    age: int = Field(..., title='Age')
```

### ju.pydantic_util.schema_to_pydantic_model(schema, model_name='AutoModel')

Converts a JSON schema (with ‘properties’) to a Pydantic model.
Only supports basic types and required fields for demonstration.

* **Return type:**
  [`type`](https://docs.python.org/3/builtins/functions.html#type)[`BaseModel`]

```pycon
>>> schema_simple = {
...     "type": "object",
...     "properties": {
...         "name": {"type": "string"},
...         "age": {"type": "integer"},
...         "city": {"type": "string", "default": "New York"}
...     },
...     "required": ["name"]
... }
>>> model = schema_to_pydantic_model_simple(schema_simple, "User")
>>> model.__name__
'User'
>>> model.model_json_schema()['properties']['name']
{'title': 'Name', 'type': 'string'}
>>> age_schema = model.model_json_schema()['properties']['age']
>>> assert age_schema['title'] == 'Age'
>>> # Ensure both integer and null are allowed (order-independent)
>>> assert {'type': 'integer'} in age_schema.get('anyOf', []) and {'type': 'null'} in age_schema.get('anyOf', [])
>>> city_schema = model.model_json_schema()['properties']['city']
>>> assert city_schema['title'] == 'City' and city_schema['default'] == 'New York' and city_schema['type'] == 'string'
>>> model.model_json_schema()['required']
['name']
```

### ju.pydantic_util.schema_to_pydantic_model_simple(schema, model_name='AutoModel')

Converts a JSON schema (with ‘properties’) to a Pydantic model.
Only supports basic types and required fields for demonstration.

* **Return type:**
  [`type`](https://docs.python.org/3/builtins/functions.html#type)[`BaseModel`]

```pycon
>>> schema_simple = {
...     "type": "object",
...     "properties": {
...         "name": {"type": "string"},
...         "age": {"type": "integer"},
...         "city": {"type": "string", "default": "New York"}
...     },
...     "required": ["name"]
... }
>>> model = schema_to_pydantic_model_simple(schema_simple, "User")
>>> model.__name__
'User'
>>> model.model_json_schema()['properties']['name']
{'title': 'Name', 'type': 'string'}
>>> age_schema = model.model_json_schema()['properties']['age']
>>> assert age_schema['title'] == 'Age'
>>> # Ensure both integer and null are allowed (order-independent)
>>> assert {'type': 'integer'} in age_schema.get('anyOf', []) and {'type': 'null'} in age_schema.get('anyOf', [])
>>> city_schema = model.model_json_schema()['properties']['city']
>>> assert city_schema['title'] == 'City' and city_schema['default'] == 'New York' and city_schema['type'] == 'string'
>>> model.model_json_schema()['required']
['name']
```

### ju.pydantic_util.valid_models(json_obj, models, \*, factory=<function \_model_validate>)

A generator that yields the models that json_obj is valid with respect to.

```pycon
>>> from pydantic import BaseModel
>>> class User(BaseModel):
...     name: str
...     code: int
...
>>> class Admin(User):
...     pwd: str
...
>>> json_obj = {"name": "John", "code": 30}
>>> models = [User, Admin]
>>> [x.__name__ for x in valid_models(json_obj, models)]
['User']
>>> json_obj = {"name": "Thor", "code": 3, "pwd": "1234"}
>>> [x.__name__ for x in valid_models(json_obj, models)]
['User', 'Admin']
```

Note that valid_models is a generator, it doesn’t return a list.

Tip, to get the first model that is valid, or None if no model is valid:

```pycon
>>> get_name = lambda o: getattr(o, '__name__', 'None')
>>> first_valid_model_name = (
...     lambda o, models: get_name(next(valid_models(o, models), None))
... )
>>> first_valid_model_name({"name": "John", "code": 30}, models)
'User'
>>> first_valid_model_name({"something": "else"}, models)
'None'
```
