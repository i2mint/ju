"""Tools for working with Pydantic models."""

import json
from functools import partial

from typing import Any, Dict, Iterable, Optional, Callable, Union
from pydantic import BaseModel, ValidationError, create_model
from i2 import ObjectClassifier


# -------------------------------------------------------------------------------------
# Type hint classification


# Example use: Define a type classification instance
def is_pydantic_model(obj: Any) -> bool:
    """Returns True if the object is a Pydantic model (subclass of BaseModel)."""
    from pydantic import BaseModel

    return isinstance(obj, type) and issubclass(obj, BaseModel)


def is_typing_type(obj: Any) -> bool:
    """Returns True if the object is a typing type (e.g., List, Literal, etc.)."""
    from typing import get_origin

    return get_origin(obj) is not None


# Create an instance for type hint classification
type_hint_classifier = ObjectClassifier(
    {"pydantic_model": is_pydantic_model, "typing_type": is_typing_type}
)


def is_type_hint(obj: Any) -> bool:
    """Returns True if the object is a Pydantic model or a typing type."""
    return type_hint_classifier.matches(obj)


# -------------------------------------------------------------------------------------
# Construct and validate Pydantic models

from pydantic import BaseModel, ValidationError
from typing import Callable, Type, TypeVar

Data = TypeVar('Data', bound=Any)
ModelType = Type[BaseModel]
ModelFactory = Callable[[ModelType, Data], BaseModel]


def _raise_error(e: Exception, model: ModelType, data: Data, factory: ModelFactory):
    raise e


def _return_false_on_error(
    e: Exception, model: ModelType, data: Data, factory: ModelFactory
):
    return False


def _model_validate(model: ModelType, data: Data) -> BaseModel:
    return model.model_validate(data)


def _call_and_return_true(
    model: ModelType, data: Data, factory=_model_validate
) -> bool:
    factory(model, data)
    return True


def mk_pydantic_model(
    data: Data,
    model: ModelType,
    *,
    factory: Callable[[ModelType, Data], BaseModel] = _model_validate,
    error_callback: Callable = _raise_error,
) -> BaseModel:
    """
    Make a Pydantic model instance from data, parametrizing constructor and error handling.

    By default, it uses the `model.model_validate` method, but you can pass a custom
    constructor function and error handling callback.

    :param data: A dictionary representing the data to be validated.
    :param model: A Pydantic model class.
    :param factory: A callable used to construct the model instance.
                    Defaults to `model.model_validate`.
    :param error_callback: A callback to handle validation errors.

    :return: A Pydantic model instance.

    Example:

    >>> from pydantic import BaseModel
    >>> class User(BaseModel):
    ...     name: str
    ...     code: int
    ...
    >>> data = {"name": "John", "code": 30}
    >>> user = mk_pydantic_model(data, User)
    >>> user
    User(name='John', code=30)

    Example with custom constructor:

    >>> user = mk_pydantic_model(data, User, factory=lambda model, data: model.model_construct(**data))
    >>> user
    User(name='John', code=30)
    """
    try:
        return factory(model, data)
    except ValidationError as e:
        error_callback(e, model, data, factory)


def mk_pydantic_models(
    data: Data,
    models: Iterable[ModelType],
    *,
    factory: Callable[[ModelType, Data], BaseModel] = _model_validate,
    error_callback: Callable = _raise_error,
) -> Iterable[BaseModel]:
    """
    The iterable-of-models version of `mk_pydantic_model`.
    """
    return (
        mk_pydantic_model(data, m, factory=factory, error_callback=error_callback)
        for m in models
    )


def is_valid_wrt_model(
    data: Data,
    model: ModelType,
    *,
    factory: Callable[[ModelType, Data], BaseModel] = _model_validate,
):
    """
    Check if a json object is valid wrt to a pydantic model.
    """
    return mk_pydantic_model(
        data,
        model,
        factory=partial(_call_and_return_true, factory=factory),
        error_callback=_return_false_on_error,
    )


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


# -------------------------------------------------------------------------------------
# utils for extraction
from typing import Any, Dict, Type, Union, get_args, get_origin, List


def _get_type_parameters(origin: Type[BaseModel]):
    """Helper function to safely retrieve the type parameters of a generic class.

    Note: This is because it's not safe to call `__parameters__` directly on the origin,
    so we do it anyway, but encapsulated in a function, to encapsulate and locate the
    risk.
    """
    return getattr(origin, '__parameters__', ())


def match_typevars_to_args(generic_model: Type[BaseModel]) -> Dict[TypeVar, Type[Any]]:
    """
    Given a Pydantic generic model, returns a mapping of type variables to their
    concrete types.

    Args:
        generic_model (Type[BaseModel]): A generic Pydantic model (e.g., Pair[int, str]).

    Returns:
        Dict[TypeVar, Type[Any]]: A dictionary mapping type variables (e.g., T and U)
        to their concrete types (e.g., int and str).

    >>> from typing import TypeVar, Generic, List

    >>> T = TypeVar('T')
    >>> U = TypeVar('U')

    >>> class Pair(Generic[T, U]):
    ...     first: T
    ...     second: U

    >>> X = Pair[int, str]

    >>> match_typevars_to_args(X)
    {~T: <class 'int'>, ~U: <class 'str'>}
    """
    # Get the origin (the base class without type args, e.g., Response)
    origin = get_origin(generic_model)

    if origin is None:
        # If the model isn't generic, return an empty dictionary
        return {}

    # Get the actual type arguments (e.g., (EmbeddingT,))
    concrete_types = get_args(generic_model)

    # Get the type variables (e.g., (DatumT,))
    type_vars = _get_type_parameters(origin)

    # Create a mapping from type variables to concrete types
    return dict(zip(type_vars, concrete_types))


def is_a_basemodel(obj) -> bool:
    """
    Check if an object is a Pydantic BaseModel.

    >>> from typing import List
    >>> class MyModel(BaseModel):
    ...     '''Some model'''
    >>> list(map(is_a_basemodel, [BaseModel, MyModel, 3.14, int, List[MyModel]]))
    [True, True, False, False, False]

    """
    if not isinstance(obj, type):
        return False
    else:
        # Get the origin in case it's a generic type like List[str]
        if get_origin(obj) is not None:
            return False  # right? Or do we want this to be is_a_basemodel(get_origin(obj))?
        return issubclass(obj, BaseModel)


def field_paths_and_annotations(
    data_model: Type[BaseModel],
) -> Dict[str, Type[Any]]:
    """
    Get flattened field paths and their corresponding annotations from a Pydantic model.

    Generates a dictionary of dot-separated paths and their corresponding types
    from the fields of a given Pydantic BaseModel and any nested BaseModels within it.

    The function recursively traverses the fields of the BaseModel and its nested models,
    including fields that are lists, sets, tuples, or iterables containing BaseModels.
    If the field is a collection containing a BaseModel, the path is marked with a '*'.

    This structure is compatible with the `glom` library, allowing extraction of values
    from a dictionary that matches the BaseModel structure.

    Args:
        data_model (Type[BaseModel]): The Pydantic BaseModel to extract field paths and annotations from.

    Returns:
        Dict[str, Type[Any]]: A dictionary where the keys are the dot-separated paths to fields
                              and the values are their corresponding types.

    Example:

    >>> from pydantic import BaseModel
    >>> from typing import List

    >>> class BItem(BaseModel):
    ...     c: int

    >>> class A(BaseModel):
    ...     b: List[BItem]
    ...     d: str

    >>> class Model(BaseModel):
    ...     a: A

    >>> paths = field_paths_and_annotations(Model)
    >>> expected_paths = {'a.b.*.c': int, 'a.d': str}
    >>> assert paths == expected_paths, f"Expected: {expected_paths}, but got: {paths}"

    See that it works with generics:
    
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

    """

    def get_field_type(field_type, model: Type[BaseModel]):
        """Resolves the actual type of a field, replacing generics with their concrete types."""
        typevar_mapping = match_typevars_to_args(model)

        # Replace any type variables in the field_type with their corresponding concrete types
        if typevar_mapping:
            if field_type in typevar_mapping:
                return typevar_mapping[field_type]

            # If the field_type is a generic collection (e.g., List[DatumT]), replace the args
            origin_type = get_origin(field_type)
            if origin_type:
                args = get_args(field_type)
                resolved_args = tuple(typevar_mapping.get(arg, arg) for arg in args)
                return origin_type[resolved_args]

        return field_type

    def recurse_model(model: Type[BaseModel], prefix: str = '') -> Dict[str, Type[Any]]:
        paths = {}
        for field_name, field_info in model.model_fields.items():
            field_type = get_field_type(field_info.annotation, model)
            current_path = f"{prefix}.{field_name}" if prefix else field_name
            origin = get_origin(field_type)
            args = get_args(field_type)

            if is_a_basemodel(field_type) or is_a_basemodel(origin):
                # the field type is a BaseModel or a generic BaseModel
                paths.update(recurse_model(field_type, current_path))
            elif (
                origin in {list, set, tuple, List}
                and args
                and is_a_basemodel(args[0])  # TODO: What if args[1] is a generic?
            ):
                paths.update(recurse_model(args[0], f"{current_path}.*"))
            else:
                paths[current_path] = field_type

        return paths

    return recurse_model(data_model)
