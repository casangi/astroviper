"""
Fast ``dft_predict``, selected by ``ASTROVIPER_KERNELS=dft=fast``.

It computes the reference's exact direct sum (``direct_fourier_transform_cpp``)

    V_k = sum_q (M_q / n_q) exp(+2 pi i (u_k l_q + v_k m_q + w_k (n_q - 1)))

by exact algebraic refactorings only, with results within rounding of the
exact value; it is not bitwise equal to the reference. Each call takes one
path, chosen in C++ from its inputs alone (all of its model planes, rows and
w) before anything is written:

``"S"``
    Every used row (all three uvw finite) has ``w == 0``: the separable sum,
    a GEMM over the folded model per model plane, with the phases ``u l`` and
    ``v m`` exact before one rounding of the reduced angle, so that its error
    does not grow with the phase (the reference's does). What remains is the
    rounding of the weights M / n, which it shares with the reference bit for
    bit (large only near the horizon, where n is ill-conditioned), and of its
    own sums: usually a few eps sum|M/n|, at most about the chain length's
    worth, and in another order than the reference's, so that at a given
    sample the reference can be the closer (both lose relative precision
    where values are subnormal). Where a sample's terms reach beyond
    float32's range, a complex64 sample of one kernel can overflow to inf
    where the other's is finite. It writes the samples that the reference
    writes and no other. A
    sample's bits do not depend on the threads or the instruction set, nor on
    how the rows and channels are split into calls that take S (a split can
    change the path of a part, as the path is decided per call).
``"declined"``
    Anything else, including every call the reference binding would convert
    or reject and a call that cannot allocate its buffers or start its
    threads: the reference binding runs with the same arguments, so the
    result is the reference's, bit for bit, and so is any exception.

Misaligned inputs (such as Dask's deserialised arrays) are read through
aligned copies, and a misaligned ``vis_data`` is written in place. The path
is reported to the kernel switch, so that each trace line
names it. ``ISA`` is the instruction set of path S in this process:
``"avx2"`` when the CPU has it, or ``"generic"`` (set
``ASTROVIPER_KERNELS_ISA=generic`` before the import to force it, for tests).
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
    report_path("dft", path)
    if path == DECLINED:
        _reference_predict(*args, **kwargs)
