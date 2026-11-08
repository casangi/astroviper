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

==========  ================================  ===============
kernel      entries                           implementations
==========  ================================  ===============
``dft``     ``dft_predict``                   ``ref``, ``fast``
``hogbom``  ``clean_cube``,                   ``ref``, ``fast``
            ``clean_cube_many_threads``
``asp``     ``clean``, ``clean_cube``         ``ref``, ``fast``
==========  ================================  ===============

With ``hogbom=fast`` a call may put several threads on one plane, where the
reference ``clean_cube`` runs one per plane, and it never starts more threads
than the CPUs it may run on.

``ASTROVIPER_KERNELS_TRACE=<dir>``
    Each wrapped call appends one JSON line to
    ``<dir>/kernels-<host>-<pid>.jsonl``: kernel, entry, impl, path (the
    path the implementation reported for the call, such as ``"S"`` or
    ``"declined"`` for ``dft=fast``, else the impl), error
    (the exception type name, or null), t_start and t_end (epoch seconds,
    ``t_end = t_start + wall_s``), wall_s, rusage_cpu_s with
    ``cpu_scope="process"`` (the process-wide CPU time of the call),
    first_call, host, pid, thread, and the shapes and dtypes of the array
    arguments. A line that cannot be written is logged once and dropped; it
    never changes the call's result. Only hooked shims trace: setting the
    trace without ``ASTROVIPER_KERNELS`` traces nothing.

``ASTROVIPER_KERNELS_ISA=generic``
    A test override read once per process, at the import of a fast module:
    its baseline instruction-set routines instead of the best ones the CPU
    supports (they give the same bits). Any other value is an error.

Processes inherit the selection only through their environment: set the
variable before a Dask cluster starts. A worker that imported a shim with it
unset runs the reference.
"""

from astroviper._kernels._switch import dispatch, impl, selected

__all__ = ["dispatch", "impl", "selected"]
