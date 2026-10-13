"""Test the oas module."""

import yaml
import pytest
from ju.util import ju_files
from ju.oas import get_routes, Routes

test_data_path = ju_files / "tests" / "test_data"
simple_openapi_spec_path = test_data_path / "simple_openapi_spec.yaml"


def _openapi_spec():
    return yaml.safe_load(simple_openapi_spec_path.read_text())


@pytest.fixture
def openapi_spec():
    return _openapi_spec()


def test_get_routes(openapi_spec):
    result = list(get_routes(openapi_spec))
    assert result == [("get", "/items"), ("post", "/items")]

    result = list(get_routes(openapi_spec, include_methods="get"))
    assert result == [("get", "/items")]

    result = list(get_routes(openapi_spec, include_methods="post"))
    assert result == [("post", "/items")]

    result = list(get_routes(openapi_spec, include_methods="put"))
    assert result == []


def test_routes(openapi_spec):
    routes = Routes(openapi_spec)
    assert list(routes) == [("get", "/items"), ("post", "/items")]

    get_route = routes["get", "/items"]

    assert get_route.input_specs == {
        "parameters": [
            {
                "in": "query",
                "name": "type",
                "schema": {"type": "string"},
                "required": True,
                "description": "Type of items to list",
            },
            {
                "in": "query",
                "name": "limit",
                "schema": {"type": "integer", "default": 10},
                "required": False,
                "description": "Maximum number of items to return",
            },
        ],
        "requestBody": {},
    }

    assert get_route.output_specs == {
        "200": {
            "description": "An array of items",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "active": {"type": "boolean"},
                            },
                        },
                    }
                }
            },
        }
    }

    assert get_route.params == {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["type"],
    }

    post_route = routes["post", "/items"]

    assert post_route.input_specs == {
        "parameters": [],
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer", "default": 42},
                        },
                    }
                }
            },
        },
    }

    assert post_route.output_specs == {
        "201": {
            "description": "Item created",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Unique identifier of the created item",
                            }
                        },
                    }
                }
            },
        }
    }

    assert post_route.params == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "default": 42},
        },
        "required": [],
    }


# ---------------------------------------------------------------------------
# Path-item parameters and same-name parameters (i2mint/ju#20)

_SHARED_PARAMS_SPEC = """
openapi: 3.0.3
paths:
  /items/{id}:
    parameters:
      - {in: path, name: id, required: true, schema: {type: integer}}
      - {in: query, name: fields, schema: {type: string}}
      - $ref: '#/components/parameters/Trace'
    get:
      parameters:
        - {in: query, name: fields, required: true, schema: {type: array}}
      responses: {'200': {description: ok}}
    delete:
      responses: {'204': {description: gone}}
  /plain:
    get:
      responses: {'200': {description: ok}}
components:
  parameters:
    Trace:
      {in: header, name: X-Trace, schema: {type: string}}
"""


def test_path_item_parameters_are_those_of_every_operation():
    from yaml import safe_load

    routes = Routes(safe_load(_SHARED_PARAMS_SPEC))
    get, delete = routes["get", "/items/{id}"], routes["delete", "/items/{id}"]
    assert [(p["name"], p["in"]) for p in delete.input_specs["parameters"]] == [
        ("id", "path"),
        ("fields", "query"),
        ("X-Trace", "header"),
    ]
    # the operation's own parameter wins over the path item's on (name, in)
    assert [(p["name"], p.get("required")) for p in get.input_specs["parameters"]] == [
        ("id", True),
        ("fields", True),
        ("X-Trace", None),
    ]
    assert get.params["properties"]["id"] == {"type": "integer"}
    assert get.params["properties"]["fields"] == {"type": "array"}
    assert get.params["required"] == ["id", "fields"]


def test_a_route_without_any_parameters_gets_no_parameters_key():
    from yaml import safe_load

    route = Routes(safe_load(_SHARED_PARAMS_SPEC))["get", "/plain"]
    assert "parameters" not in route.method_data
    assert route.input_specs["parameters"] == []


def test_same_name_in_two_places_stay_two_properties():
    from yaml import safe_load

    spec = safe_load("""
openapi: 3.0.3
paths:
  /things:
    post:
      parameters:
        - {in: query, name: id, required: true, schema: {type: string}}
        - {in: header, name: id, schema: {type: integer}}
        - {in: query, name: name, schema: {type: string}}
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [name, other]
              properties:
                name: {type: boolean}
                other: {type: string}
""")
    props = Routes(spec)["post", "/things"].params
    assert props["properties"] == {
        "id": {"type": "string", "x-oas-name": "id", "x-oas-in": "query"},
        "id_header": {"type": "integer", "x-oas-name": "id", "x-oas-in": "header"},
        "name": {"type": "string", "x-oas-name": "name", "x-oas-in": "query"},
        "name_body": {"type": "boolean", "x-oas-name": "name", "x-oas-in": "body"},
        "other": {"type": "string"},
    }
    assert props["required"] == ["id", "name_body", "other"]
    from ju.json_schema import json_schema_to_signature, schema_param_names

    assert schema_param_names(props) == {
        "id": "id",
        "id_header": "id",
        "name": "name",
        "name_body": "name",
        "other": "other",
    }
    assert list(json_schema_to_signature(props).names) == [
        "id",
        "name_body",
        "other",
        "id_header",
        "name",
    ]


def test_a_distinct_key_never_takes_a_name_the_spec_uses():
    from yaml import safe_load

    spec = safe_load("""
openapi: 3.0.3
paths:
  /x:
    get:
      parameters:
        - {in: query, name: id, schema: {type: string}}
        - {in: header, name: id, schema: {type: string}}
        - {in: query, name: id_header, schema: {type: integer}}
        - {in: query, schema: {type: string}, name: other}
        - {in: cookie, name: other, schema: {type: string}}
""")
    props = Routes(spec)["get", "/x"].params["properties"]
    assert props["id_header"] == {"type": "integer"}  # the spec's own, untouched
    where = {
        k: (v.get("x-oas-name"), v.get("x-oas-in"))
        for k, v in props.items()
        if "x-oas-name" in v
    }
    assert where == {
        "id": ("id", "query"),
        "id_header_2": ("id", "header"),
        "other": ("other", "query"),
        "other_cookie": ("other", "cookie"),
    }


def test_form_and_multipart_bodies_are_read_and_json_is_preferred():
    from yaml import safe_load

    spec = safe_load("""
paths:
  /a: {post: {requestBody: {content: {application/x-www-form-urlencoded:
         {schema: {required: [f], properties: {f: {type: string}}}}}}}}
  /b: {post: {requestBody: {content: {multipart/form-data: {schema: {properties: {m: {}}}},
         application/json: {schema: {properties: {j: {}}}}}}}}
  /c: {post: {responses: {}}}
""")
    routes = Routes(spec)
    a, b, c = (routes["post", p] for p in ("/a", "/b", "/c"))
    assert a.body_content_type == "application/x-www-form-urlencoded"
    assert list(a.params["properties"]) == ["f"] and a.params["required"] == ["f"]
    assert b.body_content_type == "application/json"
    assert list(b.params["properties"]) == ["j"]
    assert c.body_content_type is None and c.params["properties"] == {}


@pytest.mark.parametrize(
    "content, expected",
    [
        (  # plain JSON beats other JSON types, whatever the order
            ["application/json-patch+json", "application/json", "text/json"],
            "application/json",
        ),
        (
            ["application/vnd.api+json; charset=utf-8", "text/plain"],
            "application/vnd.api+json; charset=utf-8",
        ),
        (["application/x-ndjson", "application/notjson"], None),
        (
            ["application/X-WWW-Form-Urlencoded; charset=utf-8"],
            "application/X-WWW-Form-Urlencoded; charset=utf-8",
        ),
        (["Multipart/Form-Data"], "Multipart/Form-Data"),
        (["text/plain", "*/*"], None),
    ],
)
def test_body_content_type_preference(content, expected):
    from ju.oas import _preferred_media_type

    assert _preferred_media_type(content) == expected


def test_a_media_type_with_a_schema_beats_one_without():
    from yaml import safe_load

    spec = safe_load("""
paths:
  /a: {post: {requestBody: {content: {application/json: {},
         application/x-www-form-urlencoded: {schema: {properties: {f: {}}}}}}}}
  /b: {post: {requestBody: {content: }}}
  /c: {post: {requestBody: }}
""")
    routes = Routes(spec)
    assert list(routes["post", "/a"].params["properties"]) == ["f"]
    assert routes["post", "/b"].params["properties"] == {}  # content: null
    assert routes["post", "/c"].body_content_type is None


def test_openapi_funcs_send_each_argument_where_the_spec_says():
    from yaml import safe_load
    from ju.oas import openapi_to_funcs

    spec = safe_load("""
openapi: 3.0.3
servers: [{url: 'http://e.com'}]
paths:
  /things/{id}:
    parameters:
      - {in: path, name: id, required: true, schema: {type: string}}
    post:
      parameters:
        - {in: query, name: name, schema: {type: string}}
        - {in: header, name: X-Role, schema: {type: string}}
        - {in: header, name: id, schema: {type: integer}}
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [name]
              properties: {name: {type: string}, size: {type: integer}}
  /list:
    post:
      requestBody:
        content:
          application/json:
            schema: {type: array, title: Rows, items: {type: object}}
""")
    sent = []

    def get_response(method, url, **kwargs):
        sent.append((method, url, kwargs))
        return type("R", (), {"json": lambda self: {}})()

    funcs = {f.__name__: f for f in openapi_to_funcs(spec, get_response=get_response)}
    func = funcs["post_things__id"]
    func(id="7", name="Q", x_role="admin", id_header=3, name_body="B", size=2)
    ((method, url, kwargs),) = sent
    assert url == "http://e.com/things/7"
    assert kwargs["params"] == {"name": "Q"}
    assert kwargs["headers"] == {"X-Role": "admin", "id": "3"}
    assert kwargs["json"] == {"name": "B", "size": 2}

    funcs["post_list"](rows=[{"a": 1}])
    assert sent[-1][2]["json"] == [{"a": 1}] and "headers" not in sent[-1][2]


def test_openapi_funcs_with_a_parameters_list_schema_and_non_object_bodies():
    from ju.oas import OpenApiFunc

    sent = []

    def get_response(method, url, **kwargs):
        sent.append((url, kwargs))
        return type("R", (), {"json": lambda self: {}})()

    # the shape the generated-client route builds: a `parameters` list, no `properties`
    schema = {
        "title": "f",
        "parameters": [
            {"name": "thingId", "in": "path", "required": True},
            {"name": "q", "in": "query"},
        ],
    }
    func = OpenApiFunc(
        method="get",
        uri="/t/{thingId}",
        base_url="http://e.com",
        param_schema=schema,
        get_response=get_response,
    )
    func("7", q="x")
    assert sent[0] == ("http://e.com/t/7", {"params": {"q": "x"}})

    # a string body is not an object: nothing to assemble, so no `{}` body
    schema = {
        "title": "g",
        "type": "object",
        "required": [],
        "properties": {"a": {"type": "string"}},
        "_oas_parameters": [],
        "_oas_request_body": {"type": "string"},
    }
    func = OpenApiFunc(
        method="post",
        uri="/s",
        base_url="http://e.com",
        param_schema=schema,
        get_response=get_response,
    )
    func(a="v")
    assert sent[1][1]["json"] is None
