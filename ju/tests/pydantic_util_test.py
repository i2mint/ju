import json
from ju.pydantic_util import ModelSource, pydantic_model_to_code, data_to_pydantic_model


def test_pydantic_model_to_code() -> str:
    """
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
