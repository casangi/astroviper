"""FFTs in the library that ``fft_backend`` names.

``"numpy"`` runs ``numpy.fft``, ``"scipy"`` ``scipy.fft`` and ``"pyfftw"`` FFTW through
pyFFTW. The transforms take ``scipy.fft``'s arguments, its ``workers`` named
``processing_function_threads``, and ``fft_backend``. The shifts, their fold into sign
checkerboards on even axes and ``next_fast_len`` do not depend on the library. AGENTS.md
section 6 states the memory each library takes. The C++ kernels transform with pocketfft
through ``processing_functions/include/fft.hpp``.
"""

import numpy as np
import scipy.fft
from numpy.fft import fftshift, ifftshift
from scipy.fft import next_fast_len

__all__ = [
    "fft2",
    "ifft2",
    "fftshift",
    "ifftshift",
    "next_fast_len",
    "fold_shift_checkerboard",
]

_NUMPY_LINES = 8


def _library(fft_backend):
    """The FFT module of ``fft_backend``, checked by ``check_fft_backend``; under
    ``"pyfftw"`` the plan cache is enabled, so that repeated transforms of one shape
    reuse their plans."""
    from astroviper.processing_functions.imaging.config_check._common import (
        check_fft_backend,
    )

    library = check_fft_backend(fft_backend)
    if fft_backend == "pyfftw":
        import pyfftw

        pyfftw.interfaces.cache.enable()
    return library


def fold_shift_checkerboard(arr, axes, centred=False):
    """Multiply ``arr`` in place by a separable checkerboard of signs along ``axes``.

    On an even axis of length ``n`` the roll by ``n/2`` of an ``fftshift`` /
    ``ifftshift`` is a pointwise multiply by ``c[k] = (-1)**k`` in the other
    domain, for the forward and the inverse transform ``F`` alike:
    ``F(ifftshift(x)) == c * F(x)`` and ``fftshift(F(y)) == F(c * y)``. The
    two shifts of a centred transform fold to::

        fftshift(F(ifftshift(x))) == c_centred * F(c * x)

    with ``c_centred[k] = (-1)**(k - n/2) = (-1)**(n/2) * c[k]``, the
    checkerboard that is ``+1`` at the centre pixel ``k = n/2``, where
    ``fftshift`` puts the origin. ``centred=False`` multiplies by ``c`` along
    each axis (before the transform), ``centred=True`` by ``c_centred`` (after
    it). ``c`` on both sides would be off by ``(-1)**(n0/2 + n1/2)``: the
    negated transform when that sum is odd (e.g. a 90 x 120 plane).

    Each checkerboard is a 1-D sign broadcast per axis (no full 2-D kernel),
    which avoids the two full-array roll-copies the shifts would make; the
    centring only flips that 1-D sign, so it costs no extra pass. ``arr``
    keeps its dtype (the real sign array broadcasts without upcasting a
    complex array).

    Parameters
    ----------
    arr : numpy.ndarray
        Array to multiply in place; its ``axes`` are even.
    axes : tuple of int
        Axes of the checkerboard.
    centred : bool, optional
        ``False`` (default) multiplies by ``c``, ``True`` by ``c_centred``.

    Returns
    -------
    numpy.ndarray
        ``arr``.
    """
    sign_dtype = arr.real.dtype
    for ax in axes:
        n = arr.shape[ax]
        sign = np.ones(n, dtype=sign_dtype)
        sign[1::2] = -1.0
        if centred and (n // 2) % 2:
            sign = -sign
        shape = [1] * arr.ndim
        shape[ax] = n
        arr *= sign.reshape(shape)
    return arr


def _fftw_unaligned_inplace(work, axes, processing_function_threads, inverse):
    """Transform external unaligned storage using only one aligned line of scratch.

    FFTW requires natural alignment even with SIMD disabled. A separable FFT
    lets spilled buffers stay in place while each row/column uses a reusable
    aligned line; no plane-sized realignment copy is needed.
    """
    import pyfftw

    for axis in axes:
        lines = np.moveaxis(work, axis, -1)
        scratch = pyfftw.empty_aligned(lines.shape[-1], dtype=work.dtype)
        plan = pyfftw.FFTW(
            scratch,
            scratch,
            direction="FFTW_BACKWARD" if inverse else "FFTW_FORWARD",
            flags=("FFTW_ESTIMATE",),
            threads=processing_function_threads,
        )
        for index in np.ndindex(lines.shape[:-1]):
            np.copyto(scratch, lines[index])
            plan()
            np.copyto(lines[index], scratch)
        del plan, scratch
    return work


def _numpy_lines(function, x, out, axis, other, **kwargs):
    """numpy.fft's 1-D ``function`` along ``axis`` of ``x``, into ``out`` (``x`` itself or a
    new array of its shape).

    Plane by plane (the planes of ``other`` and ``axis``): when ``x`` and ``out`` are aligned,
    numpy is given ``_NUMPY_LINES`` lines of the plane at a time; otherwise each line goes
    through one aligned row, as numpy would copy every block it is given into aligned memory.
    """
    sources = np.moveaxis(x, (other, axis), (-2, -1))
    targets = np.moveaxis(out, (other, axis), (-2, -1))
    if x.flags.aligned and out.flags.aligned:
        for plane in np.ndindex(sources.shape[:-2]):
            source, target = sources[plane], targets[plane]
            for start in range(0, source.shape[0], _NUMPY_LINES):
                block = slice(start, start + _NUMPY_LINES)
                function(source[block], out=target[block], **kwargs)
        return out
    row = np.empty(sources.shape[-1], dtype=x.dtype)
    for plane in np.ndindex(sources.shape[:-2]):
        source, target = sources[plane], targets[plane]
        for j in range(source.shape[0]):
            np.copyto(row, source[j])
            if out is x:
                function(row, out=row, **kwargs)
                np.copyto(target[j], row)
            else:
                function(row, out=target[j], **kwargs)
    return out


def _transform2(
    x, s, axes, norm, overwrite_x, processing_function_threads, fft_backend, inverse
):
    """fft2 (``inverse=False``) or ifft2 in the library ``fft_backend`` names."""
    library = _library(fft_backend)
    if fft_backend == "numpy":
        if s is None and x.dtype.kind == "c":
            function = library.ifft if inverse else library.fft
            first, second = axes
            out = x if overwrite_x else np.empty(x.shape, dtype=x.dtype)
            _numpy_lines(function, x, out, second, first, norm=norm)
            return _numpy_lines(function, out, out, first, second, norm=norm)
        transform = library.ifft2 if inverse else library.fft2
        return transform(x, s=s, axes=axes, norm=norm)
    if (
        fft_backend == "pyfftw"
        and overwrite_x
        and s is None
        and norm is None
        and not x.flags.aligned
    ):
        threads = (
            scipy.fft.get_workers()
            if processing_function_threads is None
            else processing_function_threads
        )
        return _fftw_unaligned_inplace(x, axes, threads, inverse)
    transform = library.ifft2 if inverse else library.fft2
    return transform(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=processing_function_threads,
    )


def fft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    processing_function_threads=None,
    *,
    fft_backend,
):
    """Two-dimensional FFT in the library ``fft_backend`` names.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    s, axes, norm
        As ``scipy.fft.fft2``; ``axes`` names two axes.
    overwrite_x : bool, optional
        As ``scipy.fft.fft2``: ``x`` may be destroyed. Under ``"numpy"`` a complex
        ``x`` without ``s`` is transformed by numpy's 1-D transforms, 8 lines of a
        plane at a time, or one aligned row at a time when ``x`` is not aligned: in
        place with ``overwrite_x``, else into one new array. Under ``"pyfftw"`` an
        ``x`` that is not aligned, without ``s`` and ``norm``, is transformed in
        place through one aligned row with ``overwrite_x``.
    processing_function_threads : int, optional
        Threads, handed as given to scipy.fft (its ``workers``) and pyFFTW, whose
        scipy interfaces refuse ``0`` and read a negative ``n`` (down to
        ``-os.cpu_count()``) as ``os.cpu_count() + 1 + n``; ``None`` is the count of
        ``scipy.fft.set_workers``, 1 outside it. The aligned row of ``"pyfftw"``
        hands the count to ``pyfftw.FFTW``, which takes it as an unsigned int (a
        negative count raises OverflowError). numpy.fft takes no count: under
        ``"numpy"`` the transform runs one thread whatever the count.
    fft_backend : {"numpy", "scipy", "pyfftw"}
        FFT library.

    Returns
    -------
    numpy.ndarray
        The transform; with ``overwrite_x`` it may be held in the memory of ``x``.

    Raises
    ------
    ValueError
        ``fft_backend`` is not one of the three names.
    ImportError
        The library is not installed.
    """
    return _transform2(
        x,
        s,
        axes,
        norm,
        overwrite_x,
        processing_function_threads,
        fft_backend,
        inverse=False,
    )


def ifft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    processing_function_threads=None,
    *,
    fft_backend,
):
    """Two-dimensional inverse FFT in the library ``fft_backend`` names.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    s, axes, norm
        As ``scipy.fft.ifft2``; ``axes`` names two axes.
    overwrite_x : bool, optional
        As ``scipy.fft.ifft2``: ``x`` may be destroyed. Under ``"numpy"`` a complex
        ``x`` without ``s`` is transformed by numpy's 1-D transforms, 8 lines of a
        plane at a time, or one aligned row at a time when ``x`` is not aligned: in
        place with ``overwrite_x``, else into one new array. Under ``"pyfftw"`` an
        ``x`` that is not aligned, without ``s`` and ``norm``, is transformed in
        place through one aligned row with ``overwrite_x``.
    processing_function_threads : int, optional
        Threads, handed as given to scipy.fft (its ``workers``) and pyFFTW, whose
        scipy interfaces refuse ``0`` and read a negative ``n`` (down to
        ``-os.cpu_count()``) as ``os.cpu_count() + 1 + n``; ``None`` is the count of
        ``scipy.fft.set_workers``, 1 outside it. The aligned row of ``"pyfftw"``
        hands the count to ``pyfftw.FFTW``, which takes it as an unsigned int (a
        negative count raises OverflowError). numpy.fft takes no count: under
        ``"numpy"`` the transform runs one thread whatever the count.
    fft_backend : {"numpy", "scipy", "pyfftw"}
        FFT library.

    Returns
    -------
    numpy.ndarray
        The transform; with ``overwrite_x`` it may be held in the memory of ``x``.

    Raises
    ------
    ValueError
        ``fft_backend`` is not one of the three names.
    ImportError
        The library is not installed.
    """
    return _transform2(
        x,
        s,
        axes,
        norm,
        overwrite_x,
        processing_function_threads,
        fft_backend,
        inverse=True,
    )
