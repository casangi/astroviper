"""
Fast Hogbom minor cycle, selected by ``ASTROVIPER_KERNELS=hogbom=fast``.

``clean_cube`` and ``clean_cube_many_threads`` take the arguments of the
reference entries (``deconvolvers/hogbom``) and give their results bit for
bit: the residual and model cubes, the iterations, stop codes, flags,
``final_peak``, ``total_flux_cleaned`` and the iteration history. Bit for bit
holds for every value that is not a NaN; a NaN result is a NaN in both, with
its sign and payload unspecified (they depend on the operand order that the
compiler picks for a commutative operation, in the reference too). Each
plane's searched pixels (the clean box inside the mask) are kept current by
one fused pass per iteration that subtracts the last component and searches
for the next; the subtractions from the other pixels, which no iteration
reads, are applied at the end of the model update in the original order
(``include/hclean_fast.hpp``). Small planes (at most 8,192 pixels, or at
most 64 columns and 65,536 pixels) are not split that way: each iteration
subtracts the last component from its whole footprint and then scans the
box, as the reference does, row by row or, for a narrow box, column by
column. The bits do not depend on the number of threads, on the instruction
set or on these choices.

Threads: a call uses up to ``processing_function_threads`` (< 1 counts as 1,
as in the reference), and no more than the CPUs it may run on and its work
justify: many small planes get up to about the square root of their work in
units of the cost of starting a thread (tiny planes that stop at their first
search count for none), often fewer than the reference starts, and a thread
starts only while planes are left that no thread has taken.
Unlike the reference ``clean_cube``, which runs one thread per plane, the
fast one may put several threads on one plane: the same results, from more
cores at once. Its threads poll for a bounded time between the steps of an
iteration and then sleep; none outlives a call.

Memory: besides its inputs and the iteration history, a call keeps the plans
of its larger planes and at most 16,384 deferred subtractions of 32 bytes per
plane in flight; it does not grow with the iterations. Each thread that holds
small planes column by column has a copy of one plane, its mask and its PSF.

Speed, measured on one machine (12 cores, 24 threads; the WP4 reports):
``clean_cube_many_threads`` was faster than the reference in every case
measured, and ``clean_cube`` in most, among them calls of many small planes
at 12 and 24 threads and calls of larger planes. ``clean_cube`` was slower on
these:

* Planes of one row. Six planes of 1 x 2,410 to 1 x 8,192 pixels ran at 0.60
  to 0.73 times the reference's speed without a mask, and at 0.51 to 0.95
  times with random masks, at 1, 12 and 24 threads. Planes of two rows ran at
  0.73 to 1.5 times its speed; planes of four rows or more were faster.
* Small planes with random masks. 24 planes of 3 x 48, with 5 to 50 % of
  their pixels in the mask, ran at 0.81 to 0.90 times its speed (without a
  mask, or with a stripe, at 1.16 to 1.62 times).
* Tiny planes on one thread. Planes of 4 x 4 to 8 x 8 pixels ran at 0.65 to
  1.2 times its speed (it varied from process to process).
* Calls that do almost nothing, by the fixed cost of a call: up to about 15
  microseconds longer for small planes that stop within a few iterations,
  and about 75 microseconds longer for a plane of 65,536 x 1 pixels whose
  mask stops it at once (311 against 235 microseconds).
* Narrow planes just over 65,536 pixels (1,400 x 48) ran at 0.88 to 1.1 times
  its speed.

Narrow planes of 32,768 to 65,536 pixels (at most 64 columns) take one thread
each, as small planes do, so one or two of them gain nothing from more
threads. Both entries were still faster than the reference on them, but up to
3.2 times slower than an earlier version, which put a crew of threads on each
(one plane of 1,024 x 64 with a disk mask, at 12 or 24 threads: 27.4 ms,
against 8.6 to 8.9 ms for the earlier version and 46 to 70 ms for the
reference).

Each call takes one path, reported to the kernel switch for its trace line:

``"fast"``
    The fast kernel ran (the path is the implementation's name).
``"declined"``
    Every call that the reference binding would not run exactly as given
    (another argument type or dtype, a failing check, an array the call
    writes that shares memory with another argument, image planes without
    pixels) or that cannot allocate its buffers before its first write: the
    reference runs with the same arguments, so the result is the reference's,
    bit for bit, and so is any exception. Names are compared in place, so a
    keyword whose name cannot be encoded as UTF-8 declines too, and the
    reference binding then aborts the process, as it does when called
    directly.

Misaligned arrays (such as Dask's deserialised ones) are read through aligned
copies, as much memory again as those arrays, and the residual and model
written back. ``ISA`` is the instruction set of this process: ``"avx2"`` when
the CPU has it, or ``"generic"`` (set ``ASTROVIPER_KERNELS_ISA=generic``
before the import to force it, for tests).
"""

# The extension by its module path: when it is not built, the import raises
# ModuleNotFoundError naming it (``from <package> import <module>`` would
# raise a plain ImportError, which ``pytest.importorskip`` does not skip).
import astroviper._kernels.fast.hogbom._hogbom_fast_ext as _fast
from astroviper._kernels._switch import report_path
from astroviper._kernels.fast.hogbom._hogbom_fast_ext import ISA
from astroviper.processing_functions.imaging.deconvolvers.hogbom import (
    _hogbom_ext as _reference,
)

__all__ = ["DECLINED", "ISA", "clean_cube", "clean_cube_many_threads"]

#: The path of a call that the reference ran.
DECLINED = "declined"


def clean_cube(*args, **kwargs):
    """``clean_cube`` of the reference, with the same arguments and results.

    Cleans the cube in place by the fast kernel, or, when the call declines
    it, by the reference binding (which also raises the reference's
    exceptions).
    """
    result = _fast.clean_cube(*args, **kwargs)
    if result is None:
        report_path("hogbom", DECLINED)
        return _reference.clean_cube(*args, **kwargs)
    return result


def clean_cube_many_threads(*args, **kwargs):
    """``clean_cube_many_threads`` of the reference, with the same arguments
    and results; see :func:`clean_cube`."""
    result = _fast.clean_cube_many_threads(*args, **kwargs)
    if result is None:
        report_path("hogbom", DECLINED)
        return _reference.clean_cube_many_threads(*args, **kwargs)
    return result
