"""The layer rule of AGENTS.md, checked on the source without importing it.

Every import statement of src/astroviper counts: at module level and inside
functions, ``import x`` and ``from x import y``, relative ones resolved against
the importing module. A module of a layer imports from its own layer or a lower
one, in the order distributed_applications, node_tasks, processing_functions,
utils, and no module outside parked/ imports from parked/. The other top-level
packages, calibration, data and parked, are not layers: the order does not
apply to imports from or into them.
"""

import ast
import collections
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "astroviper"
LAYERS = ("distributed_applications", "node_tasks", "processing_functions", "utils")

#: The imports of the tree that go up a layer, each inside a function:
#: (importing file, module imported from). The test accepts them only inside
#: a function and fails when one of them is gone, so that this list stays exact.
UPWARD_IMPORTS = [
    ("utils/io.py", "astroviper.processing_functions.imaging.imaging_conclude.restore"),
    (
        "utils/measurement_set_v2_input.py",
        "astroviper.processing_functions.imaging.config_check",
    ),
    (
        "utils/node_task_io.py",
        "astroviper.processing_functions.imaging.utils.frequency_mapping",
    ),
    (
        "utils/node_task_io.py",
        "astroviper.processing_functions.imaging.utils.frequency_mapping",
    ),
]


def _import_statements():
    """(file, line, module imported from, top-level package of astroviper reached,
    whether the statement is inside a function).

    One entry for each top-level package of astroviper that an import statement
    reaches; ``file`` is relative to src/astroviper.
    """
    for path in sorted(SOURCE.rglob("*.py")):
        file = path.relative_to(SOURCE).as_posix()
        package = ["astroviper", *path.relative_to(SOURCE).parts[:-1]]
        tree = ast.parse(path.read_bytes(), filename=str(path))
        nested = {
            id(inner)
            for outer in ast.walk(tree)
            if isinstance(outer, ast.FunctionDef | ast.AsyncFunctionDef)
            for inner in ast.walk(outer)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                named = [(alias.name, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join([*base, *([node.module] if node.module else [])])
                named = [(module, f"{module}.{alias.name}") for alias in node.names]
            else:
                continue
            reached = {
                (module, name.split(".")[1])
                for module, name in named
                if name.startswith("astroviper.")
            }
            for module, target in sorted(reached):
                yield file, node.lineno, module, target, id(node) in nested


def test_imports_go_down_or_sideways():
    assert all((SOURCE / layer).is_dir() for layer in LAYERS), SOURCE
    upward = [
        (file, line, module, target, nested)
        for file, line, module, target, nested in _import_statements()
        if file.split("/")[0] in LAYERS
        and target in LAYERS
        and LAYERS.index(target) < LAYERS.index(file.split("/")[0])
    ]
    found = collections.Counter(
        (file, module) for file, _, module, _, nested in upward if nested
    )
    known = collections.Counter(UPWARD_IMPORTS)
    new = [
        f"{file}:{line} imports from {module} ({target})"
        for file, line, module, target, nested in upward
        if not nested or (file, module) in found - known
    ]
    assert not new, "Imports from a higher layer:\n" + "\n".join(new)
    assert not known - found, f"No longer in the tree: {sorted(known - found)}"


def test_nothing_outside_parked_imports_from_it():
    assert (SOURCE / "parked").is_dir(), SOURCE
    imports = [
        f"{file}:{line} imports from {module} (parked)"
        for file, line, module, target, _ in _import_statements()
        if target == "parked" and file.split("/")[0] != "parked"
    ]
    assert not imports, "Imports from parked/:\n" + "\n".join(imports)
