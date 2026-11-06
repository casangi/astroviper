"""
Run-time selection between the reference kernels and alternative ones.

The reference C++ extensions stay the default and are never modified. When
``ASTROVIPER_KERNELS`` is set (and non-empty) at the import of a hooked
extension shim, the shim's hooked entries are replaced by wrappers that run,
at each call, the implementation the variable selects at that moment. When it
is unset at import, the shim is unchanged: nothing from this package is
imported and nothing can switch later in that process.

``ASTROVIPER_KERNELS``
    Comma-separated, case-insensitive, whitespace ignored, later tokens win.
    ``<impl>`` (or ``*=<impl>``) selects ``<impl>`` for every kernel that has
    it; ``<kernel>=<impl>`` for one kernel; unnamed kernels stay ``ref``.
    ``ref`` installs the wrappers but runs the reference (the overhead
    control). Anything invalid raises ``ValueError`` naming the options.
    ``python -m astroviper._kernels --check`` validates the value and imports
    every selected implementation up front.

Hooked kernels and entries (implementations registered so far):

=========  ================================  ===============
kernel     entries                           implementations
=========  ================================  ===============
``dft``    ``dft_predict``                   ``ref``
``hogbom`` ``clean_cube``,                   ``ref``
           ``clean_cube_many_threads``
``asp``    ``clean``, ``clean_cube``         ``ref``
=========  ================================  ===============

``ASTROVIPER_KERNELS_TRACE=<dir>``
    Each wrapped call appends one JSON line to
    ``<dir>/kernels-<host>-<pid>.jsonl``: kernel, entry, impl, path, t_start,
    t_end (epoch seconds), wall_s, rusage_cpu_s (process-wide), first_call,
    and the shapes and dtypes of the array arguments.

Processes inherit the selection only through their environment: set the
variable before a Dask cluster starts. A worker that imported a shim with it
unset runs the reference.
"""

from astroviper._kernels._switch import dispatch, impl, selected

__all__ = ["dispatch", "impl", "selected"]
