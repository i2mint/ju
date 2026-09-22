# ju.viz

Visualization functions for Ju.

### Functions

| [`model_digraph`](#ju.viz.model_digraph)(model, \*[, dot, parent])   | Visualize Pydantic models using Graphviz, showing the relationship between models and their fields.   |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|

### ju.viz.model_digraph(model, , dot=None, parent=None)

Visualize Pydantic models using Graphviz, showing the relationship between models
and their fields.

This function creates a diagram where model nodes are represented as rectangles,
and field nodes are represented as ellipses. Relationships between models and
fields, including nested models, are illustrated with directed edges.

### Parameters

- model: The Pydantic model or an Iterable of Pydantic models to visualize.
- dot: An optional existing Graphviz Digraph object. If not provided, a new one will be created.
- parent: The parent node to which the current model node will be connected.

### Returns

- dot: The Graphviz Digraph object representing the model structure.

Example usage:

(See: [https://github.com/i2mint/ju/discussions/4#discussioncomment-10530803](https://github.com/i2mint/ju/discussions/4#discussioncomment-10530803))

```pycon
>>> from pydantic import BaseModel
>>> from typing import List, Optional
```

```pycon
>>> class Address(BaseModel):
...     street: str
...     city: str
...     state: str
```

```pycon
>>> class User(BaseModel):
...     name: str
...     email: str
...     age: Optional[int] = None
...     addresses: List[Address]
```

```pycon
>>> dot = model_digraph(User)
>>> dot.source
'// Pydantic Model Diagram\ndigraph {\n\tUser [label=User...Address -> Address\n}\n'
```

To save to a file and get the filepath it was saved to:

```pycon
>>> dot.render(f"User_model_graph", format="png", cleanup=True)
'User_model_graph.png'
```

See also the online tool: [https://navneethg.github.io/jsonschemaviewer/](https://navneethg.github.io/jsonschemaviewer/)
