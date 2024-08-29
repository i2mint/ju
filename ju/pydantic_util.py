"""Tools for working with Pydantic models."""

from typing import Any, Dict, Iterable, Optional
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


def create_pydantic_model(
    name: str, data: Dict[str, Any], *, defaults: Optional[Dict[str, Any]] = None
):
    """
    Generate a dynamic model without default values.

    >>> json_data = {
    ...     "name": "John", "age": 30, "address": {"city": "New York", "zipcode": "10001"}
    ... }
    >>> defaults = {"age": 18}
    >>>
    >>> DynamicModel = create_pydantic_model('DynamicModel', json_data, defaults=defaults)
    >>>
    >>> model_instance_custom = DynamicModel(
    ... name="John", age=25, address={"city": "Mountain View", "zipcode": "94043"}
    ... )
    >>> model_instance_custom.model_dump()
    {'name': 'John', 'age': 25, 'address': {'city': 'Mountain View', 'zipcode': '94043'}}
    >>> model_instance_with_defaults = DynamicModel(
    ...     name="Jane", address={"city": "Los Angeles", "zipcode": "90001"}
    ... )
    >>> model_instance_with_defaults.model_dump()
    {'name': 'Jane', 'age': 18, 'address': {'city': 'Los Angeles', 'zipcode': '90001'}}
    """
    defaults = defaults or {}

    def fields():
        for key, value in data.items():
            field_type = infer_json_friendly_type(value)
            if key in defaults:
                # If the key is in defaults, use the provided default value
                yield key, (field_type, defaults[key])
            else:
                # Otherwise, mark the field as required
                yield key, (field_type, ...)

    return create_model(name, **dict(fields()))
