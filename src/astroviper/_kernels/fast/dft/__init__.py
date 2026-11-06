"""
Fast ``dft_predict``, selected by ``ASTROVIPER_KERNELS=dft=fast``.

It computes the reference's exact direct sum (``direct_fourier_transform_cpp``)

    V_k = sum_q (M_q / n_q) exp(+2 pi i (u_k l_q + v_k m_q + w_k (n_q - 1)))

by exact algebraic refactorings only, with results within rounding of the
exact value; it is not bitwise equal to the reference. Each call takes one
path, chosen in C++ from its inputs alone before anything is written:

``"S"``
    Every used row (all three uvw finite) has ``w == 0``: the separable sum,
    a GEMM over the folded model per model plane. Its bits do not depend on
    the threads, the instruction set or on how the rows are split into calls.
``"declined"``
    Anything else, including every call the reference binding would convert
    or reject: the reference binding runs with the same arguments, so the
    result is the reference's, bit for bit, and so is any exception.

The path is reported to the kernel switch, so that each trace line names it.
``ISA`` is the instruction set of path S in this process: ``"avx2"`` when the
CPU has it, or ``"generic"`` (set ``ASTROVIPER_KERNELS_ISA=generic`` before
the import to force it, for tests).
"""

from astroviper._kernels._switch import report_path
from astroviper._kernels.fast.dft._dft_fast_ext import DECLINED, ISA
from astroviper._kernels.fast.dft._dft_fast_ext import dft_predict as _fast_predict
from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp._direct_fourier_transform_ext import (
    dft_predict as _reference_predict,
)

__all__ = ["ISA", "dft_predict"]


def dft_predict(*args, **kwargs):
    """``dft_predict`` of the reference, with the same arguments and contract.

    Writes the predicted visibilities into ``vis_data`` in place by path S,
    or, when the call declines it, by the reference binding (which also
    raises the reference's exceptions).
    """
    path = _fast_predict(*args, **kwargs)
    report_path(path)
    if path == DECLINED:
        _reference_predict(*args, **kwargs)
