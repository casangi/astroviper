"""
Fast Hogbom minor cycle, selected by ``ASTROVIPER_KERNELS=hogbom=fast``.

``clean_cube`` and ``clean_cube_many_threads`` take the arguments of the
reference entries (``deconvolvers/hogbom``) and give their results bit for
bit: the residual and model cubes, the iterations, stop codes, flags,
``final_peak``, ``total_flux_cleaned`` and the iteration history. Each plane's
searched pixels (the clean box inside the mask) are kept current by one fused
pass per iteration that subtracts the last component and searches for the
next; the subtractions from the other pixels, which no iteration reads, are
applied at the end of the model update in the original order
(``include/hclean_fast.hpp``). The bits do not depend on the number of
threads or on the instruction set.

Threads: a call uses up to ``processing_function_threads`` (< 1 counts as 1,
as in the reference), and no more than the CPUs it may run on and its work
justify. Unlike the reference ``clean_cube``, which runs one thread per plane,
the fast one may put several threads on one plane: the same results, from
more cores at once. Its threads poll for a bounded time between the steps of
an iteration and then sleep; none outlives a call.

Each call takes one path, reported to the kernel switch for its trace line:

``"fast"``
    The fast kernel ran (the path is the implementation's name).
``"declined"``
    Every call that the reference binding would not run exactly as given
    (another argument type or dtype, a failing check, an array the call
    writes that shares memory with another argument, image planes without
    pixels) or that cannot allocate its buffers: the reference runs with the
    same arguments, so the result is the reference's, bit for bit, and so is
    any exception.

Misaligned arrays (such as Dask's deserialised ones) are read through aligned
copies, and the residual and model written back. ``ISA`` is the instruction
set of this process: ``"avx2"`` when the CPU has it, or ``"generic"`` (set
``ASTROVIPER_KERNELS_ISA=generic`` before the import to force it, for tests).
"""

from astroviper._kernels._switch import report_path
from astroviper._kernels.fast.hogbom import _hogbom_fast_ext as _fast
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
