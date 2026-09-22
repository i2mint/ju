# ju.rjsf

Tools for React-JSONSchema-Form (RJSF)

### Functions

| [`DFLT_RJSF_VIEWER_ON_SUBMIT`](#ju.rjsf.DFLT_RJSF_VIEWER_ON_SUBMIT)(form_data, \*[, ...])   | Prints form_data in a nicely formatted way.                                                                                                               |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`create_rjsf_viewer`](#ju.rjsf.create_rjsf_viewer)(rjsf_spec, \*[, ...])           | Factory function to create and display an RJSF viewer.                                                                                                    |
| [`display_form_data`](#ju.rjsf.display_form_data)(form_data, \*[, indent, ...])    | Prints form_data in a nicely formatted way.                                                                                                               |
| [`display_in_notebook`](#ju.rjsf.display_in_notebook)(string)                        | Display a string in a Jupyter notebook cell.                                                                                                              |
| [`func_to_form_spec`](#ju.rjsf.func_to_form_spec)(func, \*[, doc, ...])            | Returns a JSON object that can be used as a form specification, along with the function, to generate a FuncCaller React component in a React application. |

### Classes

| [`FormConfig`](#ju.rjsf.FormConfig)([submit_text, show_labels, ...])       | Configuration for form rendering behavior.      |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [`RJSFViewer`](#ju.rjsf.RJSFViewer)(rjsf_spec, \*[, on_submit, name, ...]) | Renders RJSF specifications as Jupyter widgets. |

### ju.rjsf.DFLT_RJSF_VIEWER_ON_SUBMIT(form_data, , indent=2, ensure_ascii=False, prefix='\*\*Submitted data:\*\*\\\\n\\\\n')

Prints form_data in a nicely formatted way.

* **Parameters:**
  **form_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The collected form data
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* ju.rjsf.FormConfig(submit_text='Submit', show_labels=True, layout_width='400px', autofocus_first=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for form rendering behavior.

### *class* ju.rjsf.RJSFViewer(rjsf_spec, \*, on_submit=<function display_form_data>, name=None, unpack_form_data=None, config=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Renders RJSF specifications as Jupyter widgets.

Provides a bridge between RJSF form specifications and ipywidgets,
allowing interactive form viewing and data collection in notebooks.

### Example

```pycon
>>> rjsf_dict = {'rjsf': {'schema': {...}, 'uiSchema': {...}}}
>>> viewer = RJSFViewer(rjsf_dict)
>>> viewer.display()
```

#### display()

Display the form in the notebook.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_form_data()

Extract current form data from widgets.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dictionary of form field values

#### set_form_data(data)

Set form data programmatically.

* **Parameters:**
  **data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Dictionary of field values to set
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* widget *: Widget*

Access the underlying widget for custom layouts.

### ju.rjsf.create_rjsf_viewer(rjsf_spec, \*, on_submit=<function display_form_data>, name=None, unpack_form_data=None, config=None, display=False)

Factory function to create and display an RJSF viewer.

* **Parameters:**
  * **rjsf_spec** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The RJSF specification
  * **on_submit** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Optional submission callback
  * **config** ([`FormConfig`](#ju.rjsf.FormConfig) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional form configuration
* **Return type:**
  [`RJSFViewer`](#ju.rjsf.RJSFViewer)
* **Returns:**
  Configured RJSFViewer instance

### Example

```pycon
>>> def handle_data(data):
...     print(f"Received: {data}")
>>> viewer = create_rjsf_viewer(rjsf_dict, on_submit=handle_data)
>>> viewer.display()
```

### ju.rjsf.display_form_data(form_data, , indent=2, ensure_ascii=False, prefix='\*\*Submitted data:\*\*\\\\n\\\\n')

Prints form_data in a nicely formatted way.

* **Parameters:**
  **form_data** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The collected form data
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ju.rjsf.display_in_notebook(string)

Display a string in a Jupyter notebook cell.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ju.rjsf.func_to_form_spec(func, \*, doc=True, param_to_prop_type=functools.partial(<function parametrized_param_to_type>, type_mapping=((<class 'str'>, 'string'), (<class 'bool'>, 'boolean'), (<class 'int'>, 'integer'), (<class 'float'>, 'number'), (<class 'collections.abc.Mapping'>, 'object'), (collections.abc.Sequence[str], 'array'))), nest_under_field='rjsf', base_rjsf_spec={'disabled': False, 'focusOnFirstError': False, 'liveOmit': False, 'liveValidate': False, 'noHtml5Validate': False, 'noValidate': False, 'omitExtraData': False, 'readonly': False, 'schema': {'properties': {}, 'required': [], 'title': '', 'type': 'object'}, 'showErrorList': 'top', 'uiSchema': {'ui:submitButtonOptions': {'submitText': 'Run'}}}, pyname_to_title=<function asis>)

Returns a JSON object that can be used as a form specification, along with the
function, to generate a FuncCaller React component in a React application.

param func: The function to transform
return: The form specification for the function

```pycon
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
```
