"""Visualization functions for Ju."""

from typing import get_origin, get_args, Iterable, Union
from pydantic import BaseModel


# TODO: Handle ids and labels more carefully (e.g. what happens when the field names appear in different models?)
# TODO: Test loops and recursion artifacts
def model_digraph(
    model: Union[BaseModel, Iterable[BaseModel]], *, dot=None, parent=None
):
    r"""
    Visualize Pydantic models using Graphviz, showing the relationship between models
    and their fields.

    This function creates a diagram where model nodes are represented as rectangles,
    and field nodes are represented as ellipses. Relationships between models and
    fields, including nested models, are illustrated with directed edges.

    Parameters:
    - model: The Pydantic model or an Iterable of Pydantic models to visualize.
    - dot: An optional existing Graphviz Digraph object. If not provided, a new one will be created.
    - parent: The parent node to which the current model node will be connected.

    Returns:
    - dot: The Graphviz Digraph object representing the model structure.

    Example usage:

    (See: https://github.com/i2mint/ju/discussions/4#discussioncomment-10530803)

    >>> from pydantic import BaseModel
    >>> from typing import List, Optional

    >>> class Address(BaseModel):
    ...     street: str
    ...     city: str
    ...     state: str

    >>> class User(BaseModel):
    ...     name: str
    ...     email: str
    ...     age: Optional[int] = None
    ...     addresses: List[Address]

    >>> dot = model_digraph(User) # doctest: +SKIP
    >>> dot.source  # doctest: +SKIP +ELLIPSIS
    '// Pydantic Model Diagram\ndigraph {\n\tUser [label=User...Address -> Address\n}\n'

    To save to a file and get the filepath it was saved to:

    >>> dot.render(f"User_model_graph", format="png", cleanup=True)  # doctest: +SKIP
    'User_model_graph.png'

    See also the online tool: https://navneethg.github.io/jsonschemaviewer/
    """

    # pylint: disable=import-outside-toplevel
    from graphviz import Digraph  # pip install graphviz

    if dot is None:
        dot = Digraph(comment="Pydantic Model Diagram")

    # Check if model is an Iterable (like a list or set) of models
    if isinstance(model, Iterable) and not isinstance(model, (str, BaseModel)):
        for sub_model in model:
            if isinstance(sub_model, type) and issubclass(sub_model, BaseModel):
                model_digraph(sub_model, dot=dot)
        return dot

    # Create a node for the model itself
    model_name = model.__name__
    dot.node(model_name, model_name, shape="rectangle")

    # Add nodes and edges for each field
    for field_name, field in model.model_fields.items():
        field_type = (
            field.annotation.__name__
            if hasattr(field.annotation, "__name__")
            else str(field.annotation)
        )
        dot.node(field_name, label=f"{field_name}: {field_type}", shape="ellipse")
        dot.edge(model_name, field_name)

        # Handle nested models
        origin_type = get_origin(field.annotation)
        if (
            origin_type
            and isinstance(origin_type, type)
            and issubclass(origin_type, list)
        ):
            nested_model = get_args(field.annotation)[0]
            if isinstance(nested_model, type) and issubclass(nested_model, BaseModel):
                nested_model_name = nested_model.__name__
                dot.node(nested_model_name, nested_model_name, shape="rectangle")
                dot.edge(field_name, nested_model_name)
                model_digraph(nested_model, dot=dot, parent=field_name)
        elif isinstance(field.annotation, type) and issubclass(
            field.annotation, BaseModel
        ):
            model_digraph(field.annotation, dot=dot, parent=field_name)
            dot.edge(field_name, field.annotation.__name__)

    if parent:
        dot.edge(parent, model_name)

    return dot
