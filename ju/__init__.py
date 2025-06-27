"""JSON schema Utils"""

from ju.oas import Route, Routes, openapi_to_funcs
from ju.rjsf import func_to_form_spec, create_rjsf_viewer
from ju.json_schema import (
    signature_to_json_schema,
    json_schema_to_signature,
    function_to_json_schema,  # deprecated
)
from ju.util import truncate_dict_values
from ju.pydantic_util import (
    ModelExtractor,  # Extracts key paths and corresponding values from data based on matching Pydantic models.
    is_valid_wrt_model,
    valid_models,
    data_to_pydantic_model,  # data to pydantic model
    pydantic_model_to_code,  # pydantic model to code
    field_paths_and_annotations,  # flattened field paths & annotations from model
    model_field_descriptions,  # Extracts a dictionary of field paths & their descriptions from model.
)
from ju.viz import model_digraph  # visualize pydantic models
from ju.traitlets_util import trait_to_py
