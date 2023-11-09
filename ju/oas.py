"""OpenAPI specification tools."""

from typing import Any, Dict, Iterable, Iterator
from functools import cached_property, partial
from dol import KvReader, cached_keys
from dataclasses import dataclass, field

http_methods = {'get', 'post', 'put', 'delete', 'patch', 'options', 'head'}


def get_routes(d: Dict[str, Any], include_methods=tuple(http_methods)) -> Iterable[str]:
    """
    Takes OpenAPI specification dict 'd' and returns the key-paths to all the endpoints.
    """
    if isinstance(include_methods, str):
        include_methods = {include_methods}
    for endpoint in (paths := d.get('paths', {})) :
        for method in paths[endpoint]:
            if method in include_methods:
                yield method, endpoint


dflt_type_mapping = tuple(
    {
        'array': list,
        'integer': int,
        'object': dict,
        'string': str,
        'boolean': bool,
        'number': float,
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
        self._title = spec.get('info', {}).get('title', 'OpenAPI spec')

    @classmethod
    def from_yaml(cls, yaml_str: str):
        import yaml

        return cls(yaml.safe_load(yaml_str))

    @property
    def _paths(self):
        self.spec['paths']

    def __iter__(self):
        return get_routes(self.spec)

    def __getitem__(self, k):
        return self._mk_route(*k)

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self._title}')"


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
    [{'in': 'query', 'name': 'type', 'schema': {'type': 'string'}, 'required': True, 'description': 'Type of items to list'}]
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
        method_data = self.spec.get('paths', {}).get(endpoint, {}).get(method, None)
        if method_data is None:
            raise KeyError(f"Endpoint '{endpoint}' has no method '{method}'")
        return method_data

    @cached_property
    def input_specs(self):
        return {
            'parameters': self.method_data.get('parameters', []),
            'requestBody': self.method_data.get('requestBody', {}),
        }

    @cached_property
    def output_specs(self):
        return self.method_data.get('responses', {})

    @cached_property
    def params(self):
        """Combined parameters from parameters and requestBody
        (it should usually just be one or the other, not both).
        We're calling this 'params' because that's what FastAPI calls it.
        """
        # Start with the parameters defined in the 'parameters' section
        p = self.method_data.get('parameters', [])

        # Check if requestBody is defined and has content with a JSON content type
        request_body = self.method_data.get('requestBody', {})
        content = request_body.get('content', {}).get('application/json', {})

        # If there's a schema, we extract its properties and merge with the parameters
        if 'schema' in content:
            schema_props = content['schema'].get('properties', {})
            for name, details in schema_props.items():
                p.append(
                    {'in': 'requestBody', 'name': name, 'schema': details,}
                )

        return p
