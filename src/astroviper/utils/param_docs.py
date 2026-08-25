"""Central parameter-documentation sync: marker + codegen engine.

A parameter such as ``image_params`` is spelled out in every layer of the
imaging stack (the distributed-graph driver, the node task, and the science
processing functions).  To keep those NumPy-style ``Parameters`` descriptions
identical without editing each docstring by hand, the canonical description of
each shared parameter lives in **one** registry
(:data:`astroviper.processing_functions.imaging._param_docs.IMAGING_PARAM_DOCS`),
and the :func:`sync_param_docs` codegen rewrites the matching ``Parameters``
entries in the source files of every function marked with
:func:`shares_param_docs`.

Workflow
--------
1. Edit a description in the registry.
2. Run ``python -m astroviper.utils.param_docs sync`` to rewrite the source
   docstrings (or ``--check`` to verify they are in sync, e.g. in CI).

This module is intentionally dependency-light at import time; ``libcst`` is
imported lazily inside the engine so that decorating a function with
:func:`shares_param_docs` never drags that tool into the runtime import path.
"""


def shares_param_docs(func):
    """Mark ``func`` as participating in central parameter-doc syncing.

    Pure marker -- it returns ``func`` unchanged (no runtime behaviour) and sets
    ``func.__shares_param_docs__ = True``.  The :func:`sync_param_docs` codegen
    discovers decorated functions by this decorator name in the source and
    rewrites the ``Parameters`` entries whose names appear in the central
    registry, leaving every other parameter (and the Summary / Returns / Notes
    sections) untouched.

    Parameters
    ----------
    func : callable
        The function whose shared parameter descriptions should be kept in sync.

    Returns
    -------
    callable
        ``func``, unchanged.
    """
    try:
        func.__shares_param_docs__ = True
    except (AttributeError, TypeError):
        pass
    return func


# --------------------------------------------------------------------------- #
# Codegen engine (lazy heavy imports).
# --------------------------------------------------------------------------- #
def _load_registry():
    """Return the imaging shared-parameter registry ``{param_name: description}``.

    The registry module (``processing_functions/imaging/_param_docs.py``) is pure
    data, so it is loaded directly from its file path rather than imported
    through the package.  That lets the codegen run as a standalone script
    (``python src/astroviper/utils/param_docs.py check``) with only ``libcst``
    installed -- no need to build the C++ extensions or import ``astroviper``.
    """
    import importlib.util
    import pathlib

    reg_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "processing_functions"
        / "imaging"
        / "_param_docs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_astroviper_imaging_param_docs", reg_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IMAGING_PARAM_DOCS


def _target_files():
    """Return the absolute paths of the source files the codegen scans."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]  # src/astroviper
    rel = [
        "distributed_applications/imaging/image_cube_single_field.py",
        "distributed_applications/imaging/image_continuum_single_field.py",
        "node_tasks/imaging/image_cube_single_field.py",
        "node_tasks/imaging/image_continuum_single_field.py",
        "processing_functions/imaging/image_cube_single_field.py",
        "processing_functions/imaging/calculate_imaging_weights.py",
        "processing_functions/imaging/residual_cycle.py",
        "processing_functions/imaging/model_update_cycle.py",
        "processing_functions/imaging/make_point_spread_function.py",
        "processing_functions/imaging/make_undeconvolved_image.py",
    ]
    return [root / r for r in rel]


def _rewrite_docstring(docstring, registry):
    """Return ``docstring`` with registry-known ``Parameters`` descriptions replaced.

    Only the *description* of each parameter present in the registry is changed;
    the ``name : type`` line and every other docstring section are preserved.
    Returns ``None`` if nothing changed.
    """
    import re

    lines = docstring.split("\n")

    # Locate the "Parameters" / "----------" section header.
    p_idx = None
    for i in range(len(lines) - 1):
        if lines[i].strip() == "Parameters" and set(lines[i + 1].strip()) == {"-"}:
            p_idx = i
            break
    if p_idx is None:
        return None

    header_indent = len(lines[p_idx]) - len(lines[p_idx].lstrip())
    body_indent = header_indent + 4  # numpydoc: descriptions indented 4 from name

    # Find the end of the Parameters section (next same-indent section header,
    # or end of docstring).
    end = len(lines)
    for i in range(p_idx + 2, len(lines) - 1):
        cur_indent = len(lines[i]) - len(lines[i].lstrip())
        if (
            lines[i].strip()
            and cur_indent == header_indent
            and set(lines[i + 1].strip()) == {"-"}
            and len(lines[i + 1].strip()) >= len(lines[i].strip())
        ):
            end = i
            break

    name_re = re.compile(r"^(\s*)([*\w]+)\s*:")
    out = list(lines[: p_idx + 2])
    i = p_idx + 2
    changed = False
    while i < end:
        line = lines[i]
        m = name_re.match(line)
        cur_indent = len(line) - len(line.lstrip())
        if m and cur_indent == header_indent:
            pname = m.group(2)
            out.append(line)  # keep the "name : type" line
            # Consume the existing description block (more-indented lines until
            # the next name line at header_indent or section end).
            j = i + 1
            while j < end:
                nxt = lines[j]
                nxt_indent = len(nxt) - len(nxt.lstrip())
                nm = name_re.match(nxt)
                if nxt.strip() and nm and nxt_indent == header_indent:
                    break
                j += 1
            if pname in registry:
                desc = registry[pname].rstrip("\n")
                for dl in desc.split("\n"):
                    out.append((" " * body_indent + dl).rstrip() if dl else "")
                # Preserve a trailing blank line from the consumed block (e.g.
                # the separator before the next section when this parameter is
                # the last one in the Parameters section).
                if j > i + 1 and not lines[j - 1].strip():
                    out.append("")
                changed = True
            else:
                out.extend(lines[i + 1 : j])
            i = j
        else:
            out.append(line)
            i += 1

    out.extend(lines[end:])
    if not changed:
        return None
    new_doc = "\n".join(out)
    return new_doc if new_doc != docstring else None


def sync_param_docs(check=False):
    """Sync (or check) shared parameter descriptions across the imaging stack.

    Parameters
    ----------
    check : bool, optional
        If ``True``, do not write any file; instead return the list of files that
        are *out of sync* with the registry (empty when everything matches).  If
        ``False`` (default), rewrite the source files in place and return the list
        of files that were changed.

    Returns
    -------
    list of str
        Paths of files changed (``check=False``) or out of sync (``check=True``).
    """
    import libcst as cst

    registry = _load_registry()
    affected = []

    def _has_marker(node):
        for dec in node.decorators:
            d = dec.decorator
            name = None
            if isinstance(d, cst.Name):
                name = d.value
            elif isinstance(d, cst.Attribute):
                name = d.attr.value
            elif isinstance(d, cst.Call):
                f = d.func
                if isinstance(f, cst.Name):
                    name = f.value
                elif isinstance(f, cst.Attribute):
                    name = f.attr.value
            if name == "shares_param_docs":
                return True
        return False

    class _DocTransformer(cst.CSTTransformer):
        def __init__(self):
            self.changed = False

        def _maybe_rewrite(self, node):
            # Only rewrite functions explicitly marked with @shares_param_docs.
            if not _has_marker(node):
                return node
            body = node.body.body
            if not body or not isinstance(body[0], cst.SimpleStatementLine):
                return node
            stmt = body[0].body[0]
            if not isinstance(stmt, cst.Expr) or not isinstance(
                stmt.value, cst.SimpleString
            ):
                return node
            raw = stmt.value
            try:
                current = raw.raw_value  # source content (escapes preserved)
            except Exception:
                return node
            if not isinstance(current, str):
                return node
            new_doc = _rewrite_docstring(current, registry)
            if new_doc is None:
                return node
            self.changed = True
            prefix = raw.prefix
            quote = raw.quote
            new_value = f"{prefix}{quote}{new_doc}{quote}"
            new_raw = raw.with_changes(value=new_value)
            new_stmt = stmt.with_changes(value=new_raw)
            new_simple = body[0].with_changes(body=[new_stmt])
            new_body = node.body.with_changes(body=[new_simple, *body[1:]])
            return node.with_changes(body=new_body)

        def leave_FunctionDef(self, original_node, updated_node):
            return self._maybe_rewrite(updated_node)

    for path in _target_files():
        src = path.read_text()
        module = cst.parse_module(src)
        transformer = _DocTransformer()
        new_module = module.visit(transformer)
        if transformer.changed and new_module.code != src:
            affected.append(str(path))
            if not check:
                path.write_text(new_module.code)

    return affected


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m astroviper.utils.param_docs",
        description="Sync shared imaging parameter docstrings from the central registry.",
    )
    parser.add_argument(
        "command",
        choices=["sync", "check"],
        help="'sync' rewrites the source docstrings; 'check' verifies they match.",
    )
    args = parser.parse_args(argv)

    affected = sync_param_docs(check=(args.command == "check"))
    if args.command == "check":
        if affected:
            print("Out of sync (run `python -m astroviper.utils.param_docs sync`):")
            for f in affected:
                print("  " + f)
            return 1
        print("Parameter docs are in sync.")
        return 0
    if affected:
        print("Updated parameter docs in:")
        for f in affected:
            print("  " + f)
    else:
        print("Parameter docs already in sync.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
