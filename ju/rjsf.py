"""Tools for React-JSONSchema-Form (RJSF)"""

from typing import Callable, Optional, Union
import inspect

from copy import deepcopy
from i2 import Sig

from ju.json_schema import (
    signature_to_json_schema,
    DFLT_PARAM_TO_TYPE,
    DFLT_FUNC_TITLE,
    merge_with_defaults,
)
from ju.util import asis

# TODO: Change to ju.json_schema.pyname_to_title and redo tests
DFLT_PYNAME_TO_TITLE = asis

base_schema = {
    "title": DFLT_FUNC_TITLE,
    "type": "object",
    "properties": {},
    "required": [],
}

base_ui_schema = {
    "ui:submitButtonOptions": {
        "submitText": "Run",
    }
}

BASE_RJSF_SPEC = {
    "schema": base_schema,
    "uiSchema": base_ui_schema,
    "liveValidate": False,
    "disabled": False,
    "readonly": False,
    "omitExtraData": False,
    "liveOmit": False,
    "noValidate": False,
    "noHtml5Validate": False,
    "focusOnFirstError": False,
    "showErrorList": "top",
}


def func_to_form_spec(
    func: Callable,
    *,
    doc: Optional[Union[str, bool]] = True,
    param_to_prop_type: Callable = DFLT_PARAM_TO_TYPE,
    nest_under_field: Optional[str] = "rjsf",
    base_rjsf_spec: dict = BASE_RJSF_SPEC,
    pyname_to_title: Callable[[str], str] = DFLT_PYNAME_TO_TITLE,
):
    """
    Returns a JSON object that can be used as a form specification, along with the
    function, to generate a FuncCaller React component in a React application.

    param func: The function to transform
    return: The form specification for the function

    >>> def foo(
    ...     a_bool: bool,
    ...     a_float=3.14,
    ...     an_int=2,
    ...     a_str: str = 'hello',
    ...     something_else=None
    ... ):
    ...     '''A Foo function'''
    >>>
    >>> form_spec = func_to_form_spec(foo)
    >>> assert form_spec == {
    ...     'rjsf': {
    ...         'schema': {
    ...             'title': 'foo',
    ...             'type': 'object',
    ...             'properties': {
    ...                 'a_bool': {'type': 'boolean'},
    ...                 'a_float': {'type': 'number', 'default': 3.14},
    ...                 'an_int': {'type': 'integer', 'default': 2},
    ...                 'a_str': {'type': 'string', 'default': 'hello'},
    ...                 'something_else': {'type': 'string', 'default': None}
    ...             },
    ...             'required': ['a_bool'],
    ...             'description': 'A Foo function'
    ...         },
    ...         'uiSchema': {
    ...             'ui:submitButtonOptions': {
    ...                 'submitText': 'Run'
    ...             },
    ...             'a_bool': {'ui:autofocus': True}
    ...         },
    ...         'liveValidate': False,
    ...         'disabled': False,
    ...         'readonly': False,
    ...         'omitExtraData': False,
    ...         'liveOmit': False,
    ...         'noValidate': False,
    ...         'noHtml5Validate': False,
    ...         'focusOnFirstError': False,
    ...         'showErrorList': 'top'
    ...     }
    ... }
    """
    schema, ui_schema = _func_to_rjsf_schemas(
        func,
        doc=doc,
        param_to_prop_type=param_to_prop_type,
        base_rjsf_spec=base_rjsf_spec,
        pyname_to_title=pyname_to_title,
    )

    # merge these with the base spec
    spec = deepcopy(base_rjsf_spec)
    spec["schema"] = schema
    spec["uiSchema"] = ui_schema

    if nest_under_field:
        return {nest_under_field: spec}
    else:
        return spec


# --------------------------------------------------------------------------------------
# utils


# TODO: This all should really use meshed instead, to be easily composable.
def _func_to_rjsf_schemas(
    func,
    *,
    doc: Optional[Union[str, bool]] = True,
    base_rjsf_spec: dict = BASE_RJSF_SPEC,
    pyname_to_title: Callable[[str], str] = DFLT_PYNAME_TO_TITLE,
    param_to_prop_type: Callable = DFLT_PARAM_TO_TYPE,
):
    """
    Returns the JSON schema and the UI schema for a function.

    param func: The function to transform
    return: The JSON schema and the UI schema for the function

    >>> def foo(
    ...     a_bool: bool,
    ...     a_float=3.14,
    ...     an_int=2,
    ...     a_str: str = 'hello',
    ...     something_else=None
    ... ):
    ...     '''A Foo function'''

    >>> schema, ui_schema = _func_to_rjsf_schemas(foo)
    >>> assert schema == {
    ...     'title': 'foo',
    ...     'type': 'object',
    ...     'properties': {
    ...         'a_bool': {'type': 'boolean'},
    ...         'a_float': {'type': 'number', 'default': 3.14},
    ...         'an_int': {'type': 'integer', 'default': 2},
    ...         'a_str': {'type': 'string', 'default': 'hello'},
    ...         'something_else': {'type': 'string', 'default': None}
    ...     },
    ...     'required': ['a_bool'],
    ...     'description': 'A Foo function'
    ... }
    >>> assert ui_schema == {
    ...     'ui:submitButtonOptions': {'submitText': 'Run'},
    ...     'a_bool': {'ui:autofocus': True}
    ... }

    """

    schema = signature_to_json_schema(
        func,
        doc=doc,
        pyname_to_title=pyname_to_title,
        param_to_prop_type=param_to_prop_type,
    )

    ui_schema = deepcopy(base_rjsf_spec["uiSchema"])

    # Add autofocus to the first field
    sig = inspect.signature(func)
    parameters = sig.parameters

    if len(parameters) > 0:
        first_param_name = next(iter(parameters))
        ui_schema[first_param_name] = {"ui:autofocus": True}

    # Return the schemas
    return schema, ui_schema


# --------------------------------------------------------------------------------------
# RJSF jupyter notebook viewer


from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field

# TODO: Make this import conditional
import ipywidgets as widgets


@dataclass
class FormConfig:
    """Configuration for form rendering behavior."""

    submit_text: str = "Submit"
    show_labels: bool = True
    layout_width: str = "400px"
    autofocus_first: bool = True


def display_in_notebook(string: str) -> None:
    """Display a string in a Jupyter notebook cell."""
    from IPython.display import display, Markdown

    display(Markdown(string))


def display_form_data(
    form_data: Dict[str, Any],
    *,
    indent=2,
    ensure_ascii=False,
    prefix: str = "**Submitted data:**\n\n",
) -> None:
    """Prints form_data in a nicely formatted way.

    Args:
        form_data: The collected form data
    """
    import json

    formatted = json.dumps(form_data, indent=indent, ensure_ascii=ensure_ascii)
    display_in_notebook(f"{prefix}```json\n{formatted}\n```")


DFLT_RJSF_VIEWER_ON_SUBMIT = display_form_data


class RJSFViewer:
    """Renders RJSF specifications as Jupyter widgets.

    Provides a bridge between RJSF form specifications and ipywidgets,
    allowing interactive form viewing and data collection in notebooks.

    Example:

    >>> rjsf_dict = {'rjsf': {'schema': {...}, 'uiSchema': {...}}}
    >>> viewer = RJSFViewer(rjsf_dict)  # doctest: +SKIP
    >>> viewer.display()  # doctest: +SKIP
    """

    def __init__(
        self,
        rjsf_spec: Dict[str, Any],
        *,
        on_submit: Callable = DFLT_RJSF_VIEWER_ON_SUBMIT,
        name: str = None,
        unpack_form_data: bool = None,
        config: Optional[FormConfig] = None,
    ):
        """Initialize the RJSF viewer.

        Args:
            rjsf_spec: The RJSF specification dictionary
            on_submit: Optional callback for form submission
            config: Optional form configuration
        """
        self.rjsf_spec = rjsf_spec
        self.on_submit = on_submit
        self._on_submit_sig = Sig(on_submit)
        self.config = config or FormConfig()
        if unpack_form_data is None:
            # if unpack_form_data not given, use the signature of on_submit
            # to determine if it (might) require unpacking (if the names of the rjsf
            # schema are a subset of the names of on_submit, and all required_names of
            # on_submit are in the rjsf schema)
            on_submit_names = set(self._on_submit_sig.names)
            on_submit_required_names = set(self._on_submit_sig.required_names)
            rjsf_names = set(
                rjsf_spec.get("rjsf", {}).get("schema", {}).get("properties", {}).keys()
            )
            unpack_form_data = rjsf_names.issubset(
                on_submit_names
            ) and on_submit_required_names.issubset(rjsf_names)

        self.unpack_form_data = unpack_form_data

        if name is None:
            # set the name to the name of the rjsf_spec if it has one
            name = (
                self.rjsf_spec.get("rjsf", {})
                .get("schema", {})
                .get("title", self._on_submit_sig.name or "RJSF Form")
            )
        self.name = name

        self._widgets = {}
        self._form_widget = None
        self._build_form()

    def _default_submit_handler(self, form_data: Dict[str, Any]) -> None:
        """Default handler that prints submitted data in a nicely formatted way.

        Args:
            form_data: The collected form data
        """
        from IPython.display import display, Markdown
        import json

        formatted = json.dumps(form_data, indent=2, ensure_ascii=False)
        display(Markdown(f"**Submitted data:**\n\n```json\n{formatted}\n```"))

    def _extract_schema_info(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract schema and UI schema from RJSF specification.

        Returns:
            Tuple of (schema, ui_schema)
        """
        rjsf_data = self.rjsf_spec.get("rjsf", {})
        schema = rjsf_data.get("schema", {})
        ui_schema = rjsf_data.get("uiSchema", {})
        return schema, ui_schema

    def _create_widget_for_property(
        self, prop_name: str, prop_schema: Dict[str, Any], ui_config: Dict[str, Any]
    ) -> widgets.Widget:
        """Create appropriate widget for a schema property.

        Args:
            prop_name: Property name
            prop_schema: Property schema definition
            ui_config: UI configuration for this property

        Returns:
            Configured widget instance
        """
        prop_type = prop_schema.get("type", "string")
        default_value = prop_schema.get("default", "")

        widget_kwargs = {
            "description": prop_name.replace("_", " ").title() + ":",
            "style": {"description_width": "initial"},
            "layout": widgets.Layout(width="100%"),
        }

        if prop_type == "string":
            widget = widgets.Text(
                value=str(default_value),
                placeholder=f"Enter {prop_name}...",
                **widget_kwargs,
            )
        elif prop_type == "number":
            widget = widgets.FloatText(
                value=float(default_value) if default_value else 0.0, **widget_kwargs
            )
        elif prop_type == "integer":
            widget = widgets.IntText(
                value=int(default_value) if default_value else 0, **widget_kwargs
            )
        elif prop_type == "boolean":
            widget = widgets.Checkbox(value=bool(default_value), **widget_kwargs)
        else:
            # Fallback to text input
            widget = widgets.Text(value=str(default_value), **widget_kwargs)

        # Apply autofocus if specified
        if ui_config.get("ui:autofocus"):
            # Note: ipywidgets doesn't have direct autofocus,
            # but we can store this info for later use
            widget._autofocus = True

        return widget

    def _build_form(self) -> None:
        """Build the complete form widget structure."""
        schema, ui_schema = self._extract_schema_info()

        title = schema.get("title", "Form")
        description = schema.get("description", "")
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        # Create title widget
        title_widget = widgets.HTML(f"<h3>{title}</h3>")

        # Create description widget if present
        widgets_list = [title_widget]
        if description:
            desc_widget = widgets.HTML(f"<p><em>{description}</em></p>")
            widgets_list.append(desc_widget)

        # Create input widgets for each property
        for prop_name, prop_schema in properties.items():
            ui_config = ui_schema.get(prop_name, {})
            widget = self._create_widget_for_property(prop_name, prop_schema, ui_config)

            # Mark required fields
            if prop_name in required_fields:
                widget.description = widget.description + " *"

            self._widgets[prop_name] = widget
            widgets_list.append(widget)

        # Create submit button
        submit_options = ui_schema.get("ui:submitButtonOptions", {})
        submit_text = submit_options.get("submitText", self.config.submit_text)

        submit_button = widgets.Button(
            description=submit_text,
            button_style="primary",
            layout=widgets.Layout(width="auto", margin="10px 0 0 0"),
        )
        submit_button.on_click(self._handle_submit)

        widgets_list.append(submit_button)

        # Create the main form container
        self._form_widget = widgets.VBox(
            widgets_list,
            layout=widgets.Layout(
                width=self.config.layout_width,
                padding="10px",
                border="1px solid #ddd",
                border_radius="5px",
            ),
        )

    def _handle_submit(self, button: widgets.Button) -> None:
        """Handle form submission.

        Args:
            button: The submit button widget
        """
        form_data = self.get_form_data()
        self.on_submit(form_data)

    def get_form_data(self) -> Dict[str, Any]:
        """Extract current form data from widgets.

        Returns:
            Dictionary of form field values
        """
        return {name: widget.value for name, widget in self._widgets.items()}

    def set_form_data(self, data: Dict[str, Any]) -> None:
        """Set form data programmatically.

        Args:
            data: Dictionary of field values to set
        """
        for name, value in data.items():
            if name in self._widgets:
                self._widgets[name].value = value

    def display(self) -> None:
        """Display the form in the notebook."""
        from IPython.display import display

        if self._form_widget:
            display(self._form_widget)

    @property
    def widget(self) -> widgets.Widget:
        """Access the underlying widget for custom layouts."""
        return self._form_widget


def create_rjsf_viewer(
    rjsf_spec: Dict[str, Any],
    *,
    on_submit: Callable = DFLT_RJSF_VIEWER_ON_SUBMIT,
    name: str = None,
    unpack_form_data: bool = None,
    config: Optional[FormConfig] = None,
    display: bool = False,
) -> RJSFViewer:
    """Factory function to create and display an RJSF viewer.

    Args:
        rjsf_spec: The RJSF specification
        on_submit: Optional submission callback
        config: Optional form configuration

    Returns:
        Configured RJSFViewer instance

    Example:

    >>> def handle_data(data):
    ...     print(f"Received: {data}")
    >>> viewer = create_rjsf_viewer(rjsf_dict, on_submit=handle_data)  # doctest: +SKIP
    >>> viewer.display()  # doctest: +SKIP
    """
    viewer = RJSFViewer(
        rjsf_spec,
        on_submit=on_submit,
        name=name,
        unpack_form_data=unpack_form_data,
        config=config,
    )
    if display:
        viewer.display()
    return viewer
