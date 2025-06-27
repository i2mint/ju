"""OpenAPI specification tools."""

from typing import Any, Dict, Iterable, Iterator
from functools import cached_property, partial
from dol import KvReader, cached_keys, path_get as _path_get
from dataclasses import dataclass, field

http_methods = {"get", "post", "put", "delete", "patch", "options", "head"}

from dol import path_get
from functools import partial


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


def make_openapi_func(
    *,
    method,
    uri,
    base_url,
    param_schema,
    get_response=default_get_response,
    response_egress=lambda method, uri: operations.methodcaller("json"),
):
    """
    Helper to create a function for a given OpenAPI route, with signature and egress.
    """
    sig = json_schema_to_signature(param_schema)
    egress_for_this_route = response_egress(method, uri)

    def func(**kwargs):
        url = base_url + uri.format(**kwargs)
        params = {k: v for k, v in kwargs.items() if "{" + k + "}" not in uri}
        resp = get_response(method, url, json=params)
        return egress_for_this_route(resp)

    func.__signature__ = sig
    func.__name__ = f"{method.lower()}_{uri.strip('/').replace('/', '_').replace('{','').replace('}','')}"
    update_wrapper(func, func)
    return func


def openapi_to_funcs(
    spec: OpenAPISpec,
    *,
    base_url: Optional[str] = None,
    default_servers_url: str = DFLT_SERVERS_URL,
    path_to_func_name: Callable[
        [str, str], str
    ] = lambda m, p: f"{m.lower()}_{p.strip('/').replace('/', '_').replace('{','').replace('}','')}",
    get_response=default_get_response,
    response_egress=lambda method, uri: operations.methodcaller("json"),
):
    """
    spec: dict or str (YAML/JSON string or file path)
    base_url: override the spec's server URL
    default_servers_url: default URL to use if no servers are specified in the spec
    get_response: function to get the response object (default: requests.request)
    response_egress: function (method, uri) -> callable to extract result from response
    Returns a generator yielding function names and their corresponding callable functions.
    """
    spec = ensure_openapi_dict(spec)
    if not base_url:
        base_url = spec.get("servers", [{}])[0].get("url", default_servers_url)
    for uri, methods in spec["paths"].items():
        for method, details in methods.items():
            func_name = path_to_func_name(method, uri)
            # Build param schema for signature
            param_schema = {
                "title": func_name,
                "parameters": details.get("parameters", []),
            }
            # If requestBody exists, merge its schema properties
            if "requestBody" in details:
                content = details["requestBody"].get("content", {})
                json_schema = None
                for ct in content:
                    if ct.endswith("json") and "schema" in content[ct]:
                        json_schema = content[ct]["schema"]
                        break
                if json_schema and "properties" in json_schema:
                    param_schema.setdefault("properties", {}).update(
                        json_schema["properties"]
                    )
                    if "required" in json_schema:
                        param_schema.setdefault("required", []).extend(
                            json_schema["required"]
                        )
            f = partial(
                make_openapi_func,
                method=method,
                uri=uri,
                base_url=base_url,
                param_schema=param_schema,
                get_response=get_response,
                response_egress=response_egress,
            )
            func = f()
            func.__qualname__ = func_name
            yield func_name, func
