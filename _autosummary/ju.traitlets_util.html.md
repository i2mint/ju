# ju.traitlets_util

Utils for traitlets

### Functions

| [`extract_type_params`](#ju.traitlets_util.extract_type_params)(trait_type, trait)   | Extract type parameters for a trait.                                               |
|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`trait_to_py`](#ju.traitlets_util.trait_to_py)(trait)                       | Convert a traitlets trait (instance or type) to a Python object (instance or type) |

### ju.traitlets_util.extract_type_params(trait_type, trait)

Extract type parameters for a trait.

```pycon
>>> extract_type_params(
...     traitlets.Union, traitlets.Union([traitlets.Unicode(), traitlets.Float()])
... )
(<class 'str'>, <class 'float'>)
```

### ju.traitlets_util.trait_to_py(trait)

Convert a traitlets trait (instance or type) to a Python object (instance or type)

* **Return type:**
  [`object`](https://docs.python.org/3/builtins/functions.html#object) | [`type`](https://docs.python.org/3/builtins/functions.html#type)

```pycon
>>> trait_to_py(traitlets.Bool())
<class 'bool'>
>>> trait_to_py(traitlets.Union([traitlets.Unicode(), traitlets.Float()]))
typing.Union[str, float]
```
