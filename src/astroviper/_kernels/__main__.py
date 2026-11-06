"""
``python -m astroviper._kernels --check``: validate ``ASTROVIPER_KERNELS``.

Parses the variable and imports the module of every selected implementation,
so that a typo or a missing module fails before a long run starts (a driver
may import the hooked shims only late, in its workers). Exits with status 1
and the reason on standard error when the selection is not usable.
"""

import argparse
import os
import sys

from astroviper._kernels._switch import _ENTRIES, ENV, impl, selected


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
    try:
        selection = selected()
        for kernel, name in selection.items():
            for entry in _ENTRIES[kernel]:
                impl(kernel, entry, name)
    except (ValueError, ImportError) as e:
        print(f"{ENV} check failed: {e}", file=sys.stderr)
        return 1
    choices = ", ".join(f"{kernel}={name}" for kernel, name in selection.items())
    state = "hooked" if value else "unset: shims not hooked"
    print(f"{ENV}={value!r} ({state}): {choices}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
