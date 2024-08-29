"""JSON schema Utils"""

from ju.oas import Route, Routes
from ju.rjsf import func_to_form_spec
from ju.json_schema import function_to_json_schema, json_schema_to_signature
from ju.util import truncate_dict_values
from ju.pydantic_util import is_valid_wrt_model, valid_models, create_pydantic_model