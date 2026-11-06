"""
``python -m astroviper._kernels --check``: validate ``ASTROVIPER_KERNELS``.

Parses the variable, imports the module of every selected implementation and,
when ``ASTROVIPER_KERNELS_TRACE`` is set, checks that its directory can be
written, so that a typo, a missing module or an unwritable trace fails before
a long run starts (a driver may import the hooked shims only late, in its
workers). Exits with status 1 and the reason on standard error when the
selection is not usable. Warns when the trace is set without a selection: the
shims are then not hooked and write no trace.
"""

import argparse
import os
import sys
import tempfile

from astroviper._kernels._switch import _ENTRIES, ENV, TRACE_ENV, impl, selected


def _check_trace_dir(trace_dir):
    """Raise ``OSError`` unless a trace file can be created in ``trace_dir``."""
    os.makedirs(trace_dir, exist_ok=True)
    with tempfile.TemporaryFile(dir=trace_dir, prefix="kernels-check-"):
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m astroviper._kernels")
    parser.add_argument(
        "--check",
        action="store_true",
        required=True,
        help=f"validate {ENV} and import every selected implementation",
    )
    parser.parse_args(argv)

    value = os.environ.get(ENV, "")
    trace_dir = os.environ.get(TRACE_ENV)
    try:
        selection = selected()
        for kernel, name in selection.items():
            for entry in _ENTRIES[kernel]:
                impl(kernel, entry, name)
        if trace_dir:
            _check_trace_dir(trace_dir)
    except (ValueError, ImportError, OSError) as e:
        print(f"{ENV} check failed: {e}", file=sys.stderr)
        return 1
    if trace_dir and not value:
        print(
            f"warning: {TRACE_ENV} is set but {ENV} is not: the shims are not "
            "hooked and no call is traced",
            file=sys.stderr,
        )
    choices = ", ".join(f"{kernel}={name}" for kernel, name in selection.items())
    state = "hooked" if value else "unset: shims not hooked"
    print(f"{ENV}={value!r} ({state}): {choices}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
