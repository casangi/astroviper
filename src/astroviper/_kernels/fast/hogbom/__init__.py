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
(``include/hclean_fast.hpp``). The bits do not depend on the number of
threads or on the instruction set.

Threads: a call uses up to ``processing_function_threads`` (< 1 counts as 1,
as in the reference), and no more than the CPUs it may run on and its work
justify. Unlike the reference ``clean_cube``, which runs one thread per plane,
the fast one may put several threads on one plane: the same results, from
more cores at once. Its threads poll for a bounded time between the steps of
an iteration and then sleep; none outlives a call.

Memory: besides its inputs and the iteration history, a call keeps the plans
of its planes and at most 16,384 deferred subtractions of 32 bytes per plane
in flight; it does not grow with the iterations.

Speed: on planes smaller or narrower than about 64 x 64 pixels,
``clean_cube`` can be slower than the reference (one 8 x 8 plane on one
thread took 1.1 to 1.7 times as long, measured on one machine), and a call of
either entry that does almost nothing takes a few microseconds longer: there
the fast path's fixed costs per call, plane and row are as large as the
reference's whole work. On larger planes it is faster, and so is
``clean_cube_many_threads`` from 2 threads.

Each call takes one path, reported to the kernel switch for its trace line:

``"fast"``
    The fast kernel ran (the path is the implementation's name).
``"declined"``
    Every call that the reference binding would not run exactly as given
    (another argument type or dtype, a failing check, an array the call
    writes that shares memory with another argument, image planes without
    pixels) or that cannot allocate its buffers before its first write: the
    reference runs with the same arguments, so the result is the reference's,
    bit for bit, and so is any exception. One error differs: a keyword whose
    name cannot be encoded as UTF-8 raises ``RuntimeError``, where the
    reference binding aborts the process.

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
