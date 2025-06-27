"""OpenAPI specification tools."""

from typing import Any, Dict, Iterable, Iterator
from functools import cached_property, partial

from dataclasses import dataclass, field

from dol import KvReader, cached_keys, path_get as _path_get
from i2 import Sig


http_methods = {"get", "post", "put", "delete", "patch", "options", "head"}


# Make a function that gets the value of a key in a dict, given a path to that key
# but returning an empty dict if any element of the path doesn't exist
def return_empty_dict_on_error(e):
    return dict()


path_get = partial(
    _path_get, get_value=_path_get.get_item, on_error=return_empty_dict_on_error
)


def get_routes(d: Dict[str, Any], include_methods=tuple(http_methods)) -> Iterable[str]:
    """
    Takes OpenAPI specification dict 'd' and returns the key-paths to all the endpoints.
    """
    if isinstance(include_methods, str):
        include_methods = {include_methods}
    for endpoint in (paths := d.get("paths", {})):
        for method in paths[endpoint]:
            if method in include_methods:
                yield method, endpoint


dflt_type_mapping = tuple(
    {
        "array": list,
        "integer": int,
        "object": dict,
        "string": str,
        "boolean": bool,
        "number": float,
    }.items()
)


@cached_keys
class Routes(KvReader):
    """
    Represents a collection of routes in an OpenAPI specification.

    Each instance of this class contains a list of `Route` objects, which can be accessed and manipulated as needed.

    >>> from yaml import safe_load
    >>> spec_yaml = '''
    ... openapi: 3.0.3
    ... paths:
    ...   /items:
    ...     get:
    ...       summary: List items
    ...       responses:
    ...         '200':
    ...           description: An array of items
    ...     post:
    ...       summary: Create item
    ...       responses:
    ...         '201':
    ...           description: Item created
    ... '''
    >>> spec = safe_load(spec_yaml)
    >>> routes = Routes(spec)
    >>> len(routes)
    2
    >>> list(routes)
    [('get', '/items'), ('post', '/items')]
    >>> r = routes['get', '/items']
    >>> r
    Route(method='get', endpoint='/items')
    >>> r.method_data
    {'summary': 'List items', 'responses': {'200': {'description': 'An array of items'}}}

    """

    def __init__(self, spec: dict, *, type_mapping: dict = dflt_type_mapping) -> None:
        self.spec = spec
        self._mk_route = partial(Route, spec=spec, type_mapping=type_mapping)
        self._title = spec.get("info", {}).get("title", "OpenAPI spec")

    @classmethod
    def from_yaml(cls, yaml_str: str):
        import yaml

        return cls(yaml.safe_load(yaml_str))

    @property
    def _paths(self):
        self.spec["paths"]

    def __iter__(self):
        return get_routes(self.spec)

    def __getitem__(self, k):
        return self._mk_route(*k, spec=self.spec)

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self._title}')"


class ArrayOf(dict):
    """A class that is simply meant to mark the fact that some properties dict really
    represents an array of objects, and not just a single object.
    """


def properties_of_schema(schema: dict) -> dict:
    """Returns the properties of the given schema, encapsulating in ArrayOf to indicate
    that the schema is for an array of objects, and not just a single object."""
    if "items" in schema:
        # the schema is for an array
        return ArrayOf(path_get(schema, "items.properties"))
    else:
        return path_get(schema, "properties")


@dataclass
class Route:
    """
    Represents a route in an OpenAPI specification.

    Each route has a method (e.g., 'get', 'post'), an endpoint (e.g., '/items'), and a spec, which is a dictionary
    containing the details of the route as specified in the OpenAPI document.

    The `type_mapping` attribute is a dictionary that maps OpenAPI types to corresponding Python types.

    >>> from yaml import safe_load
    >>> spec_yaml = '''
    ... openapi: 3.0.3
    ... paths:
    ...   /items:
    ...     get:
    ...       summary: List items
    ...       parameters:
    ...         - in: query
    ...           name: type
    ...           schema:
    ...             type: string
    ...           required: true
    ...           description: Type of items to list
    ...       responses:
    ...         '200':
    ...           description: An array of items
    ... '''
    >>> spec = safe_load(spec_yaml)
    >>> route_get = Route('get', '/items', spec)
    >>> route_get.method
    'get'
    >>> route_get.endpoint
    '/items'
    >>> route_get.method_data['summary']
    'List items'
    >>> route_get.params
    {'type': 'object', 'properties': {'type': {'type': 'string'}}, 'required': ['type']}
    """

    method: str
    endpoint: str
    spec: dict = field(repr=False)
    # TODO: When moving to 3.9+, make below keyword-only
    type_mapping: dict = field(default=dflt_type_mapping, repr=False)

    def __post_init__(self):
        self.type_mapping = dict(self.type_mapping)

    @cached_property
    def method_data(self):
        method, endpoint = self.method, self.endpoint
        method_data = self.spec.get("paths", {}).get(endpoint, {}).get(method, None)
        if method_data is None:
            raise KeyError(f"Endpoint '{endpoint}' has no method '{method}'")
        return resolve_refs(self.spec, method_data)

    @cached_property
    def input_specs(self):
        return {
            "parameters": self.method_data.get("parameters", []),
            "requestBody": self.method_data.get("requestBody", {}),
        }

    @cached_property
    def output_specs(self):
        return self.method_data.get("responses", {})

    @cached_property
    def params(self):
        """Combined parameters from parameters and requestBody
        (it should usually just be one or the other, not both).
        We're calling this 'params' because that's what FastAPI calls it.
        """
        schema = {"type": "object", "properties": {}, "required": []}

        # Process query and path parameters
        for param in self.method_data.get("parameters", []):
            # Add each parameter to the properties
            schema["properties"][param["name"]] = param.get("schema", {})

            # Mark as required if specified
            if param.get("required", False):
                schema["required"].append(param["name"])
        # list(t['content']['application/json']['schema']['items']['properties'])
        # Process requestBody
        request_body = self.method_data.get("requestBody", {})
        content = path_get(request_body, "content.application/json")
        if "schema" in content:
            # Merge the requestBody schema with the existing properties
            body_schema = content["schema"]
            schema["properties"].update(properties_of_schema(body_schema))
            # Add required properties from requestBody
            if "required" in body_schema:
                schema["required"].extend(body_schema["required"])

        return schema

    @cached_property
    def output_properties(self, status_code: int = 200):
        """Returns the schema for the response with the given status code."""
        schema = path_get(
            self.output_specs, f"{status_code}.content.application/json.schema"
        )
        return properties_of_schema(schema)


# def resolve_ref(oas, ref):
#     from glom import glom

#     if ref.startswith('#/'):
#         ref = ref[2:]
#     ref_path = '.'.join(ref.split('/'))
#     return glom(oas, ref_path)


def resolve_refs(open_api_spec: dict, d: dict) -> dict:
    """
    Recursively resolves all references in 'd' using 'open_api_spec'.

    :param open_api_spec: The complete OpenAPI specification as a dictionary.
    :param d: The dictionary in which references need to be resolved.
    :return: The dictionary with all references resolved.
    """
    if isinstance(d, dict):
        if "$ref" in d:
            # Extract the path from the reference and resolve it
            ref_path = d["$ref"].split("/")[1:]
            ref_value = open_api_spec
            for key in ref_path:
                ref_value = ref_value.get(key, {})
            return resolve_refs(open_api_spec, ref_value)
        else:
            # Recursively resolve references in each key-value pair
            return {k: resolve_refs(open_api_spec, v) for k, v in d.items()}
    elif isinstance(d, list):
        # Recursively resolve references in each item of the list
        return [resolve_refs(open_api_spec, item) for item in d]
    else:
        # If 'd' is neither a dict nor a list, return it as is
        return d


# --------------------------------------------------------------------------------------
# OpenAPI specifications to Python functions

import json
import os
from typing import Union, Generator, Tuple, Callable, Optional
import requests
from ju.json_schema import json_schema_to_signature
import operator as operations
from functools import partial, update_wrapper
import inspect
import dill  # Optional: for advanced pickling, but not required for class-based approach

OpenAPISpec = Union[str, dict]
DFLT_SERVERS_URL = "http://localhost:8000"


def ensure_openapi_dict(spec: OpenAPISpec) -> dict:
    """
    Ensure that the OpenAPI specification is a dictionary.

    It will handle:
    - JSON strings
    - YAML strings
    - File paths to JSON or YAML files
    - URLs pointing to OpenAPI specs
    - Direct dictionaries

    """
    if isinstance(spec, str):
        if spec.strip().startswith("{") or spec.strip().startswith("openapi:"):
            # It's a string spec
            try:
                spec = json.loads(spec)
            except Exception:
                import yaml

                spec = yaml.safe_load(spec)
        elif spec.startswith("http://") or spec.startswith("https://"):
            # It's a URL, fetch the spec
            response = requests.get(spec)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                spec = response.json()
            elif "application/x-yaml" in content_type or "text/yaml" in content_type:
                import yaml

                spec = yaml.safe_load(response.text)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        elif os.path.isfile(spec):
            # It's a file path
            with open(spec) as f:
                if spec.endswith(".json"):
                    spec = json.load(f)
                elif spec.endswith(".yaml") or spec.endswith(".yml"):
                    import yaml

                    spec = yaml.safe_load(f)
        else:
            raise ValueError(
                "spec must be a dict, JSON string, YAML string, or file path"
            )
    elif not isinstance(spec, dict):
        raise ValueError("spec must be a dict, JSON string, or YAML string/file path")

    return spec


def default_get_response(method, url, **kwargs):
    import requests

    return requests.request(method.upper(), url, **kwargs)


def default_response_egress(method, uri):
    return operations.methodcaller("json")


class OpenApiFunc:
    """
    Callable class for OpenAPI route functions, supporting introspection and pickling.
    """

    def __init__(
        self,
        *,
        method,
        uri,
        base_url,
        param_schema,
        get_response=default_get_response,
        response_egress=None,
        name=None,
        qualname=None,
        doc=None,
    ):
        self.method = method
        self.uri = uri
        self.base_url = base_url
        self.param_schema = param_schema
        self.get_response = get_response
        # Always set response_egress, defaulting to named function
        if response_egress is None:
            response_egress = default_response_egress
        self.response_egress = response_egress
        self.__signature__ = Sig(json_schema_to_signature(param_schema))
        self.__name__ = (
            name
            or f"{method.lower()}_{uri.strip('/').replace('/', '_').replace('{','').replace('}','')}"
        )
        self.__qualname__ = qualname or self.__name__
        self.__doc__ = doc or f"OpenAPI route function for {method.upper()} {uri}"

    def __call__(self, *args, **kwargs):
        _kwargs = self.__signature__.map_arguments(args, kwargs)
        # Split _kwargs into path/query params and body params
        import re

        # Get path param names from the URL
        path_param_names = re.findall(r"{(\w+)}", self.uri)
        # Get parameter names from OpenAPI 'parameters'
        oas_parameters = self.param_schema.get("_oas_parameters", [])
        param_names = set(p["name"] for p in oas_parameters)
        # Get requestBody schema
        oas_request_body = self.param_schema.get("_oas_request_body", None)
        # Path params
        path_params = {k: _kwargs[k] for k in path_param_names if k in _kwargs}
        # Query params (those in parameters but not in path)
        query_params = {
            k: _kwargs[k]
            for k in param_names
            if k not in path_param_names and k in _kwargs
        }
        # Body params
        body = None
        if oas_request_body:
            if oas_request_body.get("type") == "object":
                # All properties of the object schema
                body = {
                    k: _kwargs[k]
                    for k in oas_request_body.get("properties", {})
                    if k in _kwargs
                }
            elif oas_request_body.get("type") == "array":
                # The array param is named by the schema's title or 'body'
                array_param_name = oas_request_body.get("title", "body")
                body = _kwargs.get(array_param_name)
        url = self.base_url + self.uri.format(**path_params)
        if self.method.lower() in ("post", "put", "patch"):
            resp = self.get_response(self.method, url, params=query_params, json=body)
        else:
            resp = self.get_response(self.method, url, params=query_params)
        egress = self.response_egress(self.method, self.uri)
        return egress(resp)

    def __repr__(self):
        import inspect

        sig = getattr(self, "__signature__", None)
        sig_str = str(sig) if sig is not None else "(…)"
        return (
            f"<OpenApiFunc {self.__name__}{sig_str} [{self.method.upper()} {self.uri}]>"
        )


def simple_func_namer(method: str, path: str, details: dict = None) -> str:
    trans_path = path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
    return f"{method.lower()}_{trans_path}"


def openapi_python_client_style_func_namer(
    method: str, path: str, details: dict = None, *, favor_operation_id: bool = False
) -> str:
    """
    Mimics openapi-python-client's operationId naming:
    - method + '_' + path, with path params as double underscores, and always ending with _<method>.
    E.g. get /stores/{store_name}/{key} -> get_stores__store_name___key__get
    """
    if favor_operation_id and details and "operationId" in details:
        # If operationId is provided, use it directly
        return details["operationId"]

    import re

    # Remove leading/trailing slashes
    path = path.strip("/")
    # Replace {param} with _param_
    path = re.sub(r"\{([^}]+)\}", r"_\1_", path)
    # Replace all non-alphanumeric characters (including dashes and slashes) with underscores
    path = re.sub(r"[^0-9a-zA-Z_]", "_", path)
    # Remove duplicate underscores
    path = re.sub(r"_+", "_", path)
    # Remove leading/trailing underscores
    path = path.strip("_")
    # Compose name: <method>_<path>
    if path:
        return f"{method.lower()}_{path}"
    else:
        return f"{method.lower()}"


def merge_request_body_json_schema(details, param_schema):
    """
    If the operation has a requestBody with a JSON schema, merge its properties and required fields into param_schema.
    """
    if "requestBody" in details:
        content = details["requestBody"].get("content", {})
        json_schema = None
        for ct in content:
            if ct.endswith("json") and "schema" in content[ct]:
                json_schema = content[ct]["schema"]
                break
        if json_schema and "properties" in json_schema:
            param_schema.setdefault("properties", {}).update(json_schema["properties"])
            if "required" in json_schema:
                param_schema.setdefault("required", []).extend(json_schema["required"])
    return param_schema


def openapi_to_funcs(
    spec: OpenAPISpec,
    *,
    base_url: Optional[str] = None,
    default_servers_url: str = DFLT_SERVERS_URL,
    path_to_func_name: Callable[
        [str, str], str
    ] = openapi_python_client_style_func_namer,
    get_response=default_get_response,
    response_egress=None,
) -> Iterator["OpenApiFunc"]:
    """
    spec: dict or str (YAML/JSON string or file path)
    base_url: override the spec's server URL
    default_servers_url: default URL to use if no servers are specified in the spec
    get_response: function to get the response object (default: requests.request)
    response_egress: function (method, uri) -> callable to extract result from response
    Returns a generator yielding OpenApiFunc instances for each route in the OpenAPI spec.
    """
    if response_egress is None:
        response_egress = default_response_egress
    spec_dict = ensure_openapi_dict(spec)
    routes = Routes(spec_dict)
    if not base_url:
        base_url = spec_dict.get("servers", [{}])[0].get("url", default_servers_url)
    for method, uri in routes:
        route = routes[method, uri]
        func_name = path_to_func_name(method, uri, route.method_data)
        # Build param_schema: include both parameters and requestBody fields, using resolved schemas
        param_schema = {
            "title": func_name,
            "type": "object",
            "properties": {},
            "required": [],
            "_oas_parameters": [],  # for internal use: list of parameter dicts
            "_oas_request_body": None,  # for internal use: requestBody schema
        }
        # Add parameters (query, path, etc.)
        for param in route.input_specs.get("parameters", []):
            pname = param["name"]
            param_schema["properties"][pname] = param.get("schema", {})
            if param.get("required", False):
                param_schema["required"].append(pname)
            param_schema["_oas_parameters"].append(param)
        # Add requestBody
        request_body = route.input_specs.get("requestBody", {})
        content = request_body.get("content", {})
        json_schema = None
        for ct in content:
            if ct.endswith("json") and "schema" in content[ct]:
                json_schema = content[ct]["schema"]
                break
        if json_schema:
            param_schema["_oas_request_body"] = json_schema
            if json_schema.get("type") == "object":
                # Add each property as a parameter
                for pname, pschema in json_schema.get("properties", {}).items():
                    param_schema["properties"][pname] = pschema
                if "required" in json_schema:
                    param_schema["required"].extend(json_schema["required"])
            elif json_schema.get("type") == "array":
                # Add a single parameter for the array body
                array_param_name = json_schema.get("title", "body")
                param_schema["properties"][array_param_name] = json_schema
                param_schema["required"].append(array_param_name)
        func = OpenApiFunc(
            method=method,
            uri=uri,
            base_url=base_url,
            param_schema=param_schema,
            get_response=get_response,
            response_egress=response_egress,
            name=func_name,
            qualname=func_name,
            doc=route.method_data.get("summary")
            or route.method_data.get("description"),
        )
        yield func


# TODO: Everything below needs some work! Ideally, we want to be able to use openapi-python-client to generate the code,
#   but this method creates a lot of boilerplate for the user, as opposed to the straightforward dynamic function approach above.
# --------------------------------------------------------------------------------------
# The code generation way. Requires `openapi-python-client` to be installed
# (e.g. with pip install openapi-python-client)


import tempfile
import subprocess
import importlib.util
import json
import re


def _import_generated_client_module(output_dir):
    """
    Given an output directory, find and import the generated client package.
    Returns the imported module.
    """
    import sys, os, importlib

    for entry in os.listdir(output_dir):
        pkg_path = os.path.join(output_dir, entry)
        if (
            os.path.isdir(pkg_path)
            and not entry.startswith(".")
            and os.path.isfile(os.path.join(pkg_path, "__init__.py"))
        ):
            sys.path.insert(0, output_dir)
            return importlib.import_module(entry)
    raise RuntimeError(f"No client package found in output_dir: {output_dir}")


def generate_openapi_client(openapi_spec, output_dir=None, *, file_format="json"):
    """
    openapi_spec: dict, or str (path to YAML/JSON file)
    output_dir: where to generate the client (default: temp dir)
    file_format: 'yaml' or 'json' (used if openapi_spec is a dict)
    Returns: imported client module
    """
    # Step 1: If openapi_spec is a dict, write to temp file
    if isinstance(openapi_spec, dict):
        suffix = ".yaml" if file_format == "yaml" else ".json"
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
            if file_format == "yaml":
                import yaml

                yaml.safe_dump(openapi_spec, f)
            else:
                json.dump(openapi_spec, f)
            openapi_path = f.name
    else:
        openapi_path = openapi_spec

    # Step 2: Set output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # Step 3: Use --url for URLs, --path for local files
    if str(openapi_path).startswith("http://") or str(openapi_path).startswith(
        "https://"
    ):
        cli_args = [
            "openapi-python-client",
            "generate",
            "--url",
            openapi_path,
            "--output-path",
            output_dir,
            "--overwrite",
        ]
    else:
        cli_args = [
            "openapi-python-client",
            "generate",
            "--path",
            openapi_path,
            "--output-path",
            output_dir,
            "--overwrite",
        ]

    try:
        result = subprocess.run(
            cli_args,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("openapi-python-client failed with the following output:")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    return _import_generated_client_module(output_dir)


generate_and_import_openapi_client = generate_openapi_client


def openapi_to_generated_funcs(
    spec: OpenAPISpec,
    *,
    base_url: Optional[str] = None,
    default_servers_url: str = DFLT_SERVERS_URL,
    output_dir=None,
    file_format="json",
    path_to_func_name: Callable[
        [str, str], str
    ] = openapi_python_client_style_func_namer,
    get_response=default_get_response,
    response_egress=None,
) -> Iterator[OpenApiFunc]:
    """
    Like openapi_to_funcs, but yields OpenApiFunc objects for each route, using the generated client for requests.
    Uses path_to_func_name for naming, and parameter schema from the OpenAPI spec.
    """
    import types
    import copy

    client_module = generate_openapi_client(
        spec, output_dir=output_dir, file_format=file_format
    )
    spec_dict = ensure_openapi_dict(spec)
    paths = spec_dict.get("paths", {})
    api_default_dir = os.path.join(client_module.__path__[0], "api", "default")
    if not os.path.isdir(api_default_dir):
        raise RuntimeError(
            f"No api/default directory found in generated client at {api_default_dir}"
        )
    py_files = [
        f
        for f in os.listdir(api_default_dir)
        if f.endswith(".py") and f != "__init__.py"
    ]
    opid_to_modbase = {f[:-3]: f[:-3] for f in py_files}
    for uri, methods in paths.items():
        for method, details in methods.items():
            func_name = path_to_func_name(method, uri, details)
            opid = details.get("operationId")
            mod_base = opid if opid in opid_to_modbase else None
            if not mod_base:
                mod_base = openapi_python_client_style_func_namer(method, uri, details)
                if mod_base not in opid_to_modbase:
                    continue
            mod_name = f"{client_module.__name__}.api.default.{mod_base}"
            try:
                mod = importlib.import_module(mod_name)
            except ModuleNotFoundError:
                continue
            if not hasattr(mod, "sync_detailed"):
                continue
            sync_func = getattr(mod, "sync_detailed")
            param_schema = {
                "title": func_name,
                "parameters": details.get("parameters", []),
            }
            param_schema = merge_request_body_json_schema(details, param_schema)
            yield OpenApiFunc(
                method=method,
                uri=uri,
                base_url=base_url
                or spec_dict.get("servers", [{}])[0].get("url", default_servers_url),
                param_schema=param_schema,
                get_response=sync_func,
                response_egress=response_egress,
                name=func_name,
                qualname=func_name,
                doc=details.get("summary") or details.get("description"),
            )


# --------------------------------------------------------------------------------------
# Test and demo functions


# TODO: Will deprecate this, since I'm not managing to align the names.
def validate_openapi_to_generated_funcs_alignment():
    """
    Example usage of openapi_to_generated_funcs.
    This function is just for demonstration and should be removed in production code.
    """
    from ju.oas import generate_openapi_client

    import json
    import os

    with open(os.path.expanduser("~/tmp/test_oas_gen.json"), "r") as f:
        openapi_spec = json.load(f)

    generated_funcs = list(openapi_to_generated_funcs(openapi_spec))

    dynamic_funcs = list(openapi_to_funcs(openapi_spec))

    # make dicts that list the names of functions that are in generated_funcs but not in dynamic_funcs
    # and vice versa, for easy comparison
    gen_func_names = {f.__name__ for f in generated_funcs}
    dyn_func_names = {f.__name__ for f in dynamic_funcs}
    only_in_gen = gen_func_names - dyn_func_names
    only_in_dyn = dyn_func_names - gen_func_names
    # validate that the two sets are equal, and if not, print the differences
    if only_in_gen or only_in_dyn:
        print("Functions only in generated_funcs:", only_in_gen)
        print("Functions only in dynamic_funcs:", only_in_dyn)
        raise ValueError(
            "Mismatch between generated_funcs and dynamic_funcs. \n"
            f"  Generated functions: {gen_func_names} \n"
            f"  Dynamic functions: {dyn_func_names} \n"
            f"  Only in generated_funcs: {only_in_gen} \n"
            f"  Only in dynamic_funcs: {only_in_dyn}\n"
        )
    else:
        print("All functions match between generated_funcs and dynamic_funcs.")
        return True


def print_generated_modules():
    import os
    from ju.oas import generate_openapi_client
    import json

    openapi_spec = json.load(open(os.path.expanduser("~/tmp/test_oas_gen.json")))
    client_module = generate_openapi_client(openapi_spec)
    api_default_dir = os.path.join(client_module.__path__[0], "api", "default")
    print("api/default/ modules:")
    print(sorted([f for f in os.listdir(api_default_dir) if f.endswith(".py")]))


def compare_modules_to_missing_funcs():
    import os
    import json
    from ju.oas import generate_openapi_client, openapi_to_funcs

    openapi_spec = json.load(open(os.path.expanduser("~/tmp/test_oas_gen.json")))
    client_module = generate_openapi_client(openapi_spec)
    api_default_dir = os.path.join(client_module.__path__[0], "api", "default")
    module_files = sorted(
        [
            f
            for f in os.listdir(api_default_dir)
            if f.endswith(".py") and f != "__init__.py"
        ]
    )
    print("api/default/ modules:")
    for f in module_files:
        print("  ", f)
    # Also print the missing dynamic function names
    dynamic_funcs = list(openapi_to_funcs(openapi_spec))
    dyn_func_names = {f.__name__ for f in dynamic_funcs}
    # Print all dynamic function names for reference
    print("\nDynamic function names:")
    for name in sorted(dyn_func_names):
        print("  ", name)


def compare_modules_to_operationids():
    import os
    import json
    from ju.oas import generate_openapi_client

    openapi_spec = json.load(open(os.path.expanduser("~/tmp/test_oas_gen.json")))
    client_module = generate_openapi_client(openapi_spec)
    api_default_dir = os.path.join(client_module.__path__[0], "api", "default")
    module_files = sorted(
        [
            f[:-3]
            for f in os.listdir(api_default_dir)
            if f.endswith(".py") and f != "__init__.py"
        ]
    )
    # Collect all operationIds from the OpenAPI spec
    operation_ids = set()
    for path, methods in openapi_spec["paths"].items():
        for method, details in methods.items():
            opid = details.get("operationId")
            if opid:
                operation_ids.add(opid)
    print("Generated modules:")
    for mod in module_files:
        print("  ", mod)
    print("\nOperationIds in OpenAPI spec:")
    for opid in sorted(operation_ids):
        print("  ", opid)
    print("\nOperationIds missing generated modules:")
    for opid in sorted(operation_ids - set(module_files)):
        print("  ", opid)
    print("\nGenerated modules with no matching operationId:")
    for mod in sorted(set(module_files) - operation_ids):
        print("  ", mod)


if __name__ == "__main__":
    compare_modules_to_operationids()
