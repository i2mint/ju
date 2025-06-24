"""Tests for traitlets_util"""

import pytest
import traitlets
import typing
import enum
import re

from ju.traitlets_util import py_type_for_traitlet_type, trait_to_py


def test_trait_to_py():
    uninstantiable_traitlet_types = {traitlets.Container}
    one_traitlet_type = traitlets.Int()
    one_py_type = int
    two_traitlet_types = (traitlets.Unicode(), traitlets.Int())
    two_py_types = (str, int)

    for traitlet_type, expected_py_type in py_type_for_traitlet_type.items():
        # Test with the trait class (type)
        try:
            result = trait_to_py(traitlet_type)
            assert (
                result == expected_py_type
            ), f"Type mismatch for {traitlet_type}: expected {expected_py_type}, got {result}"
        except Exception as e:
            pytest.fail(f"trait_to_py({traitlet_type}) raised an exception: {e}")

        # Now test with an instance
        try:
            # Handle special cases where instantiation requires arguments
            if traitlet_type in [traitlets.Instance, traitlets.Type]:

                class DummyClass:
                    pass

                trait_instance = traitlet_type(klass=DummyClass)
            elif traitlet_type is traitlets.UseEnum:

                class Color(enum.Enum):
                    RED = 1
                    GREEN = 2
                    BLUE = 3

                trait_instance = traitlet_type(enum_class=Color)
            elif traitlet_type in [
                traitlets.Enum,
                traitlets.CaselessStrEnum,
                traitlets.FuzzyEnum,
                traitlets.ObserveHandler,
            ]:
                trait_instance = traitlet_type(["a", "b", "c"])
            elif traitlet_type in {traitlets.List, traitlets.Set}:
                trait_instance = traitlet_type(one_traitlet_type)
            elif traitlet_type is traitlets.Tuple:
                trait_instance = traitlet_type(*two_traitlet_types)
            elif traitlet_type is traitlets.Dict:
                trait_instance = traitlet_type(
                    **dict(zip(["key_trait", "value_trait"], two_traitlet_types))
                )
            elif traitlet_type is traitlets.ForwardDeclaredInstance:
                trait_instance = traitlet_type("DummyClass")
            elif traitlet_type in {traitlets.TCPAddress, traitlets.CRegExp}:
                trait_instance = traitlet_type()
            elif traitlet_type is traitlets.Union:
                trait_instance = traitlet_type(two_traitlet_types)
            elif traitlet_type not in uninstantiable_traitlet_types:
                trait_instance = traitlet_type()
        except Exception as e:
            pytest.fail(f"Failed to instantiate {traitlet_type}: {e}")
            continue

        # Now test trait_to_py with the instance
        try:
            result = trait_to_py(trait_instance)
            if traitlet_type is traitlets.Instance:
                assert (
                    result == DummyClass
                ), f"Instance mismatch for {traitlet_type}: expected {DummyClass}, got {result}"
            elif traitlet_type is traitlets.Type:
                assert (
                    result == typing.Type[DummyClass]
                ), f"Type mismatch for {traitlet_type}: expected {typing.Type[DummyClass]}, got {result}"
            elif traitlet_type is traitlets.UseEnum:
                assert (
                    result == Color
                ), f"Enum mismatch for {traitlet_type}: expected {Color}, got {result}"
            elif traitlet_type is traitlets.List:
                expected = list[one_py_type]
                assert (
                    result == expected
                ), f"List mismatch for {traitlet_type}: expected {expected}, got {result}"
            elif traitlet_type is traitlets.Set:
                expected = set[one_py_type]
                assert (
                    result == expected
                ), f"List mismatch for {traitlet_type}: expected {expected}, got {result}"
            elif traitlet_type is traitlets.Tuple:
                expected = tuple[two_py_types]
                assert (
                    result == expected
                ), f"Tuple mismatch for {traitlet_type}: expected {expected}, got {result}"
            elif traitlet_type is traitlets.Dict:
                expected = dict[two_py_types]
                assert (
                    result == expected
                ), f"Dict mismatch for {traitlet_type}: expected {expected}, got {result}"
                assert (
                    result == expected
                ), f"Set mismatch for {traitlet_type}: expected {expected}, got {result}"
            elif traitlet_type is traitlets.Union:
                trait_instance = traitlets.Union(two_traitlet_types)
                result = trait_to_py(trait_instance)
                expected = typing.Union[two_py_types]
                assert (
                    result == expected
                ), f"Union mismatch for {traitlet_type}: expected {expected}, got {result}"
            elif traitlet_type is traitlets.CRegExp:
                expected = re.Pattern
                assert (
                    result == expected
                ), f"CRegExp mismatch: expected {expected}, got {result}"
            elif traitlet_type not in uninstantiable_traitlet_types:
                assert (
                    result == expected_py_type
                ), f"Type mismatch for instance of {traitlet_type}: expected {expected_py_type}, got {result}"
        except Exception as e:
            pytest.fail(
                f"trait_to_py(instance of {traitlet_type}) raised an exception: {e}"
            )
