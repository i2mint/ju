"""Ju base utils."""

import json
from functools import partial
from typing import (
    Mapping,
    Callable,
    Union,
    KT,
    VT,
    Sequence,
    Tuple,
    runtime_checkable,
    Protocol,
)
from collections import defaultdict
from dataclasses import dataclass
from i2 import mk_sentinel

try:
    import importlib.resources

    _files = importlib.resources.files  # only valid in 3.9+
except AttributeError:
    import importlib_resources  # needs pip install

    _files = importlib_resources.files

ju_files = _files('ju')


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


CallableMapper = Callable[[KT], VT]
MappingMapper = Mapping[KT, VT]
PairsMapper = Sequence[Tuple[KT, VT]]
Mapper = Union[PairsMapper, MappingMapper, CallableMapper]
Mapper.__doc__ = "A Mapper is a specification of (key, value) pairs"

Exceptions = Sequence[BaseException]
NotSpecified = mk_sentinel('NotSpecified')


def _return_this(_obj):
    return _obj


def mk_function_returning(_obj):
    return partial(_return_this, _obj)


@runtime_checkable
class Gettable(Protocol):
    """The missing type for objects that can be fetched from.
    The contract is that we can fetch an element from ``obj`` with brackets: ``obj[k]``.
    That is, ``obj`` has a ``__getitem__`` method.

    >>> isinstance(3, Gettable)  # 3 is not Gettable (can't do 3[...])
    False

    But ``dict``, ``list``, and ``str`` are Gettable:

    >>> isinstance([1, 2, 3], Gettable)
    True
    >>> isinstance({'foo': 'bar'}, Gettable)
    True
    >>> isinstance('foo', Gettable)
    True

    Note that so are their types:
    >>> all(isinstance(c, Gettable) for c in (list, dict, str))
    True

    """

    def __getitem__(self, k: KT) -> VT:
        pass


def _defaulted_function_call(
    func: Callable, arg, default=NotSpecified, handle_exceptions: Exceptions = None
):
    if handle_exceptions is None and default is not NotSpecified:
        return func(arg)
    else:
        try:
            return func(arg)
        except handle_exceptions as e:
            if default is NotSpecified:
                raise e
            return default


@dataclass
class GetWithCaller:
    """
    Implements the [.] operator with a backend caller.

    >>> g = GetWithCaller(lambda x: x + 1)
    >>> g[4]
    5
    """

    getter_func: Callable[[KT], VT]
    default: VT = NotSpecified
    handle_exceptions: Exceptions = (KeyError,)

    def __getitem__(self, k: KT) -> VT:
        """Get the value for a given a key k."""
        return _defaulted_function_call(
            self.getter_func, k, self.default, self.handle_exceptions
        )


# def ensure_mapping_mapper(mapper: Mapper, *, default=NotSpecified) -> MappingMapper:
#     if isinstance(mapper, Mapping):
#         return mapper
#     elif callable(mapper):
#         return dict(mapper).__getitem__  # not .get, because want to raise KeyError
#     elif isinstance(mapper, Sequence):
#         return dict(mapper)
#     else:
#         raise TypeError(f'Cannot convert {mapper} to a MappingMapper')


def _handle_default(obj, default=NotSpecified):
    """
    Adds default handling to an object.
    """
    if default is NotSpecified:
        return obj
    if isinstance(obj, Mapping):
        return defaultdict(mk_function_returning(default), obj)
    elif callable(obj):
        return partial(
            _defaulted_function_call,
            obj,
            default=default,
            handle_exceptions=(KeyError,),
        )
    else:
        raise TypeError(f'Cannot add defaulting handling to this type of object: {obj}')


def ensure_mapping_mapper(mapper: Mapper, *, default=NotSpecified) -> MappingMapper:
    """
    Will return a MappingMapper (a dict really) from the specification of a Mapper,
    which could be a Mapping itself, a callable, or (key, value) pairs.

    Further, if default is specified, the mapper will return that default if the
    requested key is not found.

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

    """
    if callable(mapper):
        getter_func = mapper
        return GetWithCaller(getter_func, default=default)
    elif isinstance(mapper, Sequence):
        mapper = dict(mapper)

    return _handle_default(mapper, default=default)


def ensure_callable_mapper(mapper: Mapper, *, default=NotSpecified) -> CallableMapper:
    """
    Will return a CallableMapper from the specification of a Mapper,
    which could be a Mapping itself, a callable, or (key, value) pairs.

    Further, if default is specified, the mapper will return that default if the
    requested key is not found.

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

    """
    if isinstance(mapper, Sequence):
        mapper = dict(mapper)
    if isinstance(mapper, Mapping):
        mapper = mapper.__getitem__
    assert callable(mapper), f"Cannot convert {mapper} to a CallableMapper"
    return _handle_default(mapper, default=default)



# def ensure_callable_mapper(mapper: Mapper, *, default=NotSpecified) -> CallableMapper:
#     if callable(mapper):
#         return mapper
#     elif isinstance(mapper, Mapping):
#         return mapper.__getitem__
#     elif isinstance(mapper, Sequence):
#         return dict(mapper).get
#     else:
#         raise TypeError(f'Cannot convert {mapper} to a CallableMapper')


# def ensure_pairs_mapper(mapper: Mapper, *, default=NotSpecified) -> PairsMapper:
#     if isinstance(mapper, Sequence):
#         return mapper
#     elif isinstance(mapper, Mapping):
#         return list(mapper.items())
#     elif callable(mapper):
#         return list(mapper())
#     else:
#         raise TypeError(f'Cannot convert {mapper} to a PairsMapper')


# -------------------------------------------------------------------------------------
# utils for routing
# See https://github.com/i2mint/i2//blob/f547257c272433b7651d09276afdfb1bb7b2f67b/misc/i2.routing.ipynb#L17

from typing import Mapping, Callable
from functools import partial


def display_dag_of_code(func, *args, **kwargs):
    from meshed import code_to_dag

    return code_to_dag(func).dot_digraph(*args, **kwargs)


def apply(func, obj):
    """
    Calls a function to an object, returing the result
    """
    return func(obj)


def _switch_case(mapping, default, feature):
    """
    Returns the value of a feature in a mapping or a default value
    """
    return mapping.get(feature, default)


def switch_case(mapping, default):
    """
    Returns a function that switches between cases based on a feature
    """
    return partial(_switch_case, mapping, default)


def _feature_based_search(
    feature_processor_pairs, feature_similarity, default, feature
):
    """
    Returns the output of a feature based on a list of feature_processor_pairs
    """
    if isinstance(feature_processor_pairs, Mapping):
        feature_processor_pairs = feature_processor_pairs.items()
    feature_matches = partial(feature_similarity, feature)
    for feature_compared_to, then_ in feature_processor_pairs:
        if feature_matches(feature_compared_to):
            return then_
    return default


def feature_based_search(feature_processor_pairs, feature_similarity, default):
    """
    Returns a function that searches for a feature in a list of feature_processor_pairs
    """
    return partial(
        _feature_based_search, feature_processor_pairs, feature_similarity, default
    )


def feature_switch(obj, *, featurizer, feature_to_output_mapping, default):
    """
    Returns the output of a feature based on a featurizer and a mapping
    """
    feature = apply(featurizer, obj)
    get_output_for_feature = switch_case(feature_to_output_mapping, default)
    output = apply(get_output_for_feature, feature)
    return output


def feature_similarity_search(
    obj,
    *,
    featurizer,
    feature_based_search,
    feature_output_pairs,
    feature_similarity,
    similarity_base_match=lambda x, y: x == y,
):
    """
    Returns the output of a feature based on a featurizer and a list of feature_output_pairs
    """
    feature = apply(obj, featurizer)
    get_output_for_feature = feature_based_search(
        feature_output_pairs, feature_similarity, similarity_base_match
    )
    output = apply(get_output_for_feature, feature)
    return output


from i2 import FuncFactory

FeatureSwitch = FuncFactory(feature_switch)
