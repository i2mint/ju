"""Tools for working with Pydantic models."""

import json

from typing import Any, Dict, Iterable, Optional, Callable, Union
from pydantic import BaseModel, ValidationError, create_model


def is_valid_wrt_model(json_obj, model):
    """
    Check if a json object is valid wrt to a pydantic model.
    """
    try:
        model(**json_obj)
        return True
    except ValidationError as e:
        return False


def valid_models(json_obj, models: Iterable[BaseModel]):
    """
    A generator that yields the models that json_obj is valid wrt to.

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

    Note that valid_models is a generator, it doesn't return a list.

    Tip, to get the first model that is valid, or None if no model is valid:

    >>> get_name = lambda o: getattr(o, '__name__', 'None')
    >>> first_valid_model_name = (
    ...     lambda o, models: get_name(next(valid_models(o, models), None))
    ... )
    >>> first_valid_model_name({"name": "John", "code": 30}, models)
    'User'
    >>> first_valid_model_name({"something": "else"}, models)
    'None'


    """
    return (model for model in models if is_valid_wrt_model(json_obj, model))


def infer_json_friendly_type(value):
    """
    Infers the type of the value for Pydantic model field.

    >>> infer_json_friendly_type(42)
    <class 'int'>
    >>> infer_json_friendly_type("Hello, World!")
    <class 'str'>
    >>> infer_json_friendly_type({"key": "value"})
    <class 'dict'>
    """
    if isinstance(value, dict):
        return dict
    elif isinstance(value, list):
        return list
    else:
        return type(value)


# TODO: Extend to something more robust
#   Perhaps based on datamodel-code-generator (see https://jsontopydantic.com/)?
def data_to_pydantic_model(
    data: Dict[str, Any],
    name: Union[str, Callable[[dict], str]] = "DataBasedModel",
    *,
    defaults: Optional[Dict[str, Any]] = None,
    create_nested_models: bool = True,
    mk_nested_name: Optional[Callable[[str], str]] = None,
):
    """
    Generate a dynamic Pydantic model, optionally creating nested models for nested dictionaries.

    :param name: Name of the Pydantic model to create.
    :param data: A dictionary representing the structure of the model.
    :param defaults: A dictionary specifying default values for certain fields.
    :param create_nested_models: If True, create nested models for nested dictionaries.

    :return: A dynamically created Pydantic model, with nested models if applicable.

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

    And note that the nested model is also created:

    >>> M.Address(city="New York", zipcode="10001")
    Address(city='New York', zipcode='10001')

    """
    defaults = defaults or {}
    nested_models = {}

    if mk_nested_name is None:
        mk_nested_name = lambda key: f"{key.capitalize()}"

    def fields():
        # TODO: Need to handle nested keys as paths to enable more control
        for key, value in data.items():
            if isinstance(value, dict) and create_nested_models:
                # Create a nested model for this dictionary
                nested_model_name = mk_nested_name(key)
                nested_model = data_to_pydantic_model(
                    value, nested_model_name, defaults=defaults.get(key, {})
                )
                nested_models[nested_model_name] = nested_model
                field_type = nested_model
            else:
                field_type = infer_json_friendly_type(value)

            if key in defaults:
                yield key, (field_type, defaults[key])
            else:
                yield key, (field_type, ...)

    model = create_model(name, **dict(fields()))
    for nested_model_name, nested_model in nested_models.items():
        setattr(model, nested_model_name, nested_model)

    return model


ModelSource = Union[str, dict, BaseModel]


def pydantic_model_to_code(
    source: ModelSource, **extra_json_schema_parser_kwargs
) -> str:
    """
    Convert a model source (json string, dict, or pydantic model) to pydantic code.

    Requires having datamodel-code-generator installed (pip install datamodel-code-generator)

    Code was based on: https://koxudaxi.github.io/datamodel-code-generator/using_as_module/

    See also this free online converter: https://jsontopydantic.com/

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
    <BLANKLINE>
    from enum import Enum
    from typing import Optional
    <BLANKLINE>
    from pydantic import BaseModel
    <BLANKLINE>
    <BLANKLINE>
    class StreetType(Enum):
        Street = 'Street'
        Avenue = 'Avenue'
        Boulevard = 'Boulevard'
    <BLANKLINE>
    <BLANKLINE>
    class Model(BaseModel):
        number: Optional[float] = None
        street_name: Optional[str] = None
        street_type: Optional[StreetType] = None
    <BLANKLINE>

    This means you can get some model code from an example data dict,
    using pydantic_model_to_code

    >>> M = data_to_pydantic_model({"name": "John", "age": 30}, "Simple")
    >>> print(pydantic_model_to_code(M))
    from __future__ import annotations
    <BLANKLINE>
    from pydantic import BaseModel, Field
    <BLANKLINE>
    <BLANKLINE>
    class Simple(BaseModel):
        name: str = Field(..., title='Name')
        age: int = Field(..., title='Age')
    <BLANKLINE>

    """
    # pylint: disable=import-outside-toplevel
    from datamodel_code_generator import (
        DataModelType,
        PythonVersion,
    )  # pip install datamodel-code-generator
    from datamodel_code_generator.model import get_data_model_types
    from datamodel_code_generator.parser.jsonschema import JsonSchemaParser

    # isinstance(x, BaseModel) doesn't work (e.g. dynamic models), so defining:
    is_pydantic_model = lambda source: hasattr(source, 'model_json_schema')
    is_json_schema_dict = lambda source: isinstance(source, dict) and 'type' in source

    if is_pydantic_model(source):
        source = source.model_json_schema()
    if is_json_schema_dict(source):
        source = json.dumps(source)

    data_model_types = get_data_model_types(
        DataModelType.PydanticV2BaseModel, target_python_version=PythonVersion.PY_311
    )
    parser = JsonSchemaParser(
        source,
        data_model_type=data_model_types.data_model,
        data_model_root_type=data_model_types.root_model,
        data_model_field_type=data_model_types.field_model,
        data_type_manager_type=data_model_types.data_type_manager,
        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
        **extra_json_schema_parser_kwargs,
    )
    return parser.parse()
