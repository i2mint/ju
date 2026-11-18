"""ju's typing aliases are plain aliases: nothing assigns attributes to them.

From Python 3.14 a typing.Union's attributes are read-only, and `SomeType.__doc__ = ...` at
module level made `import ju` fail with AttributeError on that version. The CI matrix stops
before 3.14, so the rule is pinned on the source rather than on an import.
"""

import ast
from pathlib import Path

import ju


def test_no_module_level_attribute_assignment_on_a_name():
    offenders = []
    for path in Path(ju.__file__).parent.rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "__doc__":
                        offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
