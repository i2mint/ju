# ju.util

Ju base utils.

### Functions

| [`apply`](#ju.util.apply)(func, obj)                                  | Calls a function to an object, returing the result                                                                                                  |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `asis`(obj)                                                                                        |                                                                                                                                                     |
| `display_dag_of_code`(func, \*args, \*\*kwargs)                                                    |                                                                                                                                                     |
| [`ensure_callable_mapper`](#ju.util.ensure_callable_mapper)(mapper, \*[, default])     | Will return a CallableMapper from the specification of a Mapper, which could be a Mapping itself, a callable, or (key, value) pairs.                |
| [`ensure_mapping_mapper`](#ju.util.ensure_mapping_mapper)(mapper, \*[, default])      | Will return a MappingMapper (a dict really) from the specification of a Mapper, which could be a Mapping itself, a callable, or (key, value) pairs. |
| [`feature_based_search`](#ju.util.feature_based_search)(...)                         | Returns a function that searches for a feature in a list of feature_processor_pairs                                                                 |
| [`feature_similarity_search`](#ju.util.feature_similarity_search)(obj, \*, ...[, ...])    | Returns the output of a feature based on a featurizer and a list of feature_output_pairs                                                            |
| [`feature_switch`](#ju.util.feature_switch)(obj, \*, featurizer, ...)          | Returns the output of a feature based on a featurizer and a mapping                                                                                 |
| `is_jsonable`(x)                                                                                   |                                                                                                                                                     |
| [`is_type`](#ju.util.is_type)(param, type_)                             | Checks if the type of a parameter's default value or its annotation matches a given type.                                                           |
| `mk_function_returning`(_obj)                                                                      |                                                                                                                                                     |
| [`switch_case`](#ju.util.switch_case)(mapping, default)                     | Returns a function that switches between cases based on a feature                                                                                   |
| [`truncate_dict_values`](#ju.util.truncate_dict_values)(d, \*[, max_list_size, ...]) | Returns a new dictionary with the same nested keys structure, where:                                                                                |

### Classes

| [`GetWithCaller`](#ju.util.GetWithCaller)(getter_func[, default, ...])   | Implements the [.] operator with a backend caller.     |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [`Gettable`](#ju.util.Gettable)(\*args, \*\*kwargs)                 | The missing type for objects that can be fetched from. |

### *class* ju.util.GetWithCaller(getter_func, default=Sentinel('NotSpecified'), handle_exceptions=(<class 'KeyError'>, ))

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Implements the [.] operator with a backend caller.

```pycon
>>> g = GetWithCaller(lambda x: x + 1)
>>> g[4]
5
```

### *class* ju.util.Gettable(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

The missing type for objects that can be fetched from.
The contract is that we can fetch an element from `obj` with brackets: `obj[k]`.
That is, `obj` has a `__getitem__` method.

```pycon
>>> isinstance(3, Gettable)  # 3 is not Gettable (can't do 3[...])
False
```

But `dict`, `list`, and `str` are Gettable:

```pycon
>>> isinstance([1, 2, 3], Gettable)
True
>>> isinstance({'foo': 'bar'}, Gettable)
True
>>> isinstance('foo', Gettable)
True
```

Note that so are their types:

```pycon
>>> all(isinstance(c, Gettable) for c in (list, dict, str))
True
```

### ju.util.apply(func, obj)

Calls a function to an object, returing the result

### ju.util.ensure_callable_mapper(mapper, , default=Sentinel('NotSpecified'))

Will return a CallableMapper from the specification of a Mapper,
which could be a Mapping itself, a callable, or (key, value) pairs.

Further, if default is specified, the mapper will return that default if the
requested key is not found.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)], [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]

```pycon
>>> mapping = {'a': 1, 'b': 2}
>>> func = ensure_callable_mapper(mapping, default=3)
>>> func('a')
1
>>> func('c')
3
>>> items = [('a', 1), ('b', 2)]
>>> func = ensure_callable_mapper(items, default=3)
>>> func('a')
1
>>> func('c')
3
```

### ju.util.ensure_mapping_mapper(mapper, , default=Sentinel('NotSpecified'))

Will return a MappingMapper (a dict really) from the specification of a Mapper,
which could be a Mapping itself, a callable, or (key, value) pairs.

Further, if default is specified, the mapper will return that default if the
requested key is not found.

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)]

```pycon
>>> func = lambda k: {'a': 1, 'b': 2}[k]
>>> d = ensure_mapping_mapper(func, default=3)
>>> d['a']
1
>>> d['c']
3
>>> items = [('a', 1), ('b', 2)]
>>> d = ensure_mapping_mapper(items, default=3)
>>> d['a']
1
>>> d['c']
3
```

### ju.util.feature_based_search(feature_processor_pairs, feature_similarity, default)

Returns a function that searches for a feature in a list of feature_processor_pairs

### ju.util.feature_similarity_search(obj, \*, featurizer, feature_based_search, feature_output_pairs, feature_similarity, similarity_base_match=<function <lambda>>)

Returns the output of a feature based on a featurizer and a list of feature_output_pairs

### ju.util.feature_switch(obj, , featurizer, feature_to_output_mapping, default)

Returns the output of a feature based on a featurizer and a mapping

### ju.util.is_type(param, type_)

Checks if the type of a parameter’s default value or its annotation matches a
given type.

This function handles both regular types and subscripted generics.

* **Parameters:**
  * **param** ([`Parameter`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)) – The parameter to check.
  * **type_** (`Union`[[`type`](https://docs.python.org/3/builtins/functions.html#type), [`GenericAlias`](https://docs.python.org/3/library/types.html#types.GenericAlias), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The type to check against.
* **Returns:**
  True if the parameter’s type matches the given type, False otherwise.
* **Return type:**
  [*bool*](https://docs.python.org/3/builtins/functions.html#bool)

Doctests:

```pycon
>>> from inspect import Parameter
>>> param = Parameter('p', Parameter.KEYWORD_ONLY, default=3.14)
>>> is_type(param, float)
True
>>> is_type(param, int)
False
>>> param = Parameter('p', Parameter.KEYWORD_ONLY, default=[1, 2, 3])
>>> is_type(param, list)
True
>>> from typing import List, Union
>>> is_type(param, List[int])
True
>>> is_type(param, List[str])
False
>>> is_type(param, Union[int, List[int]])
True
```

A parameterized generic built from an `collections.abc` type (e.g.
`Sequence[str]`) is itself an instance of `type` – unlike `typing`
generics such as `List[str]` – so it must still fall through to the
`__origin__`-based branch below rather than being handed to
`isinstance(param.default, type_)`, which raises `TypeError` for a
parameterized generic:

```pycon
>>> from collections.abc import Sequence
>>> is_type(param, Sequence[int])
True
>>> is_type(param, Sequence[str])
False
```

### ju.util.switch_case(mapping, default)

Returns a function that switches between cases based on a feature

### ju.util.truncate_dict_values(d, , max_list_size=2, max_string_size=66, middle_marker='...')

Returns a new dictionary with the same nested keys structure, where:

- List values are reduced to a maximum size of max_list_size.
- String values longer than max_string_size are truncated in the middle.

### Parameters

d (dict): The input dictionary.
max_list_size (int, optional): Maximum size for lists. Defaults to 2.
max_string_size (int, optional): Maximum length for strings. Defaults to None (no truncation).
middle_marker (str, optional): String to insert in the middle of truncated strings. Defaults to ‘…’.

### Returns

dict: A new dictionary with truncated lists and strings.

This can be useful when you have a large dictionary that you want to investigate,
but printing/logging it takes too much space.

### Example

```pycon
>>> large_dict = {'a': [1, 2, 3, 4, 5], 'b': {'c': [6, 7, 8, 9], 'd': 'A string like this that is too long'}, 'e': [10, 11]}
>>> truncate_dict_values(large_dict, max_list_size=3, max_string_size=20)
{'a': [1, 2, 3], 'b': {'c': [6, 7, 8], 'd': 'A string...too long'}, 'e': [10, 11]}
```

You can use `None` to indicate “no max”:

```pycon
>>> assert (
...     truncate_dict_values(large_dict, max_list_size=None, max_string_size=None)
...     == large_dict
... )
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
