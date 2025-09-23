import json
import tempfile
from pathlib import Path
from pydantic import BaseModel
from ju.pydantic_util import ModelSource, pydantic_model_to_code, data_to_pydantic_model


def test_ingress_transform_with_dict_schema():
    """Test ingress_transform with dictionary schema source."""
    schema = {"type": "object", "properties": {"old_field": {"type": "string"}}}

    def rename_field(schema_dict):
        new_schema = json.loads(json.dumps(schema_dict))  # Deep copy
        if "properties" in new_schema and "old_field" in new_schema["properties"]:
            new_schema["properties"]["new_field"] = new_schema["properties"].pop(
                "old_field"
            )
        return new_schema

    result = pydantic_model_to_code(schema, ingress_transform=rename_field)
    assert "new_field" in result
    assert "old_field" not in result


def test_ingress_transform_with_pydantic_model():
    """Test ingress_transform with Pydantic model source."""

    class TestModel(BaseModel):
        old_field: str

    def rename_field(schema_dict):
        new_schema = json.loads(json.dumps(schema_dict))  # Deep copy
        if "properties" in new_schema and "old_field" in new_schema["properties"]:
            new_schema["properties"]["new_field"] = new_schema["properties"].pop(
                "old_field"
            )
        return new_schema

    result = pydantic_model_to_code(TestModel, ingress_transform=rename_field)
    assert "new_field" in result
    assert "old_field" not in result


def test_ingress_transform_with_json_string():
    """Test ingress_transform with JSON string source."""
    schema_str = json.dumps(
        {"type": "object", "properties": {"old_field": {"type": "string"}}}
    )

    def rename_field(schema_dict):
        new_schema = json.loads(json.dumps(schema_dict))  # Deep copy
        if "properties" in new_schema and "old_field" in new_schema["properties"]:
            new_schema["properties"]["new_field"] = new_schema["properties"].pop(
                "old_field"
            )
        return new_schema

    result = pydantic_model_to_code(schema_str, ingress_transform=rename_field)
    assert "new_field" in result
    assert "old_field" not in result


def test_ingress_transform_with_json_file():
    """Test ingress_transform with JSON file source."""
    schema_data = {"type": "object", "properties": {"old_field": {"type": "string"}}}

    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(schema_data, f)
        temp_file = f.name

    try:

        def rename_field(schema_dict):
            new_schema = json.loads(json.dumps(schema_dict))  # Deep copy
            if "properties" in new_schema and "old_field" in new_schema["properties"]:
                new_schema["properties"]["new_field"] = new_schema["properties"].pop(
                    "old_field"
                )
            return new_schema

        result = pydantic_model_to_code(temp_file, ingress_transform=rename_field)
        assert "new_field" in result
        assert "old_field" not in result
    finally:
        # Clean up
        Path(temp_file).unlink(missing_ok=True)


def test_ingress_transform_validation():
    """Test that ingress_transform validates callable input."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    try:
        pydantic_model_to_code(schema, ingress_transform="not_callable")
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "ingress_transform must be callable" in str(e)


def test_no_ingress_transform():
    """Test that function works without ingress_transform (default behavior)."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = pydantic_model_to_code(schema)
    assert "name" in result
    assert "BaseModel" in result


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
