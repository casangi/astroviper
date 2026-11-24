"""FFTs in the library that ``fft_backend`` names.

``"numpy"`` runs ``numpy.fft``, ``"scipy"`` ``scipy.fft`` and ``"pyfftw"`` FFTW through
pyFFTW. The transforms take ``scipy.fft``'s arguments, its ``workers`` named
``processing_function_threads``, and ``fft_backend``. The shifts, their fold into sign
checkerboards on even axes, the sample frequencies and ``next_fast_len`` do not depend on
the library. AGENTS.md section 6 states the memory each library takes. The C++ kernels
transform with pocketfft through ``processing_functions/include/fft.hpp``.
"""

import contextlib

import numpy as np
import scipy.fft
from numpy.fft import fftfreq, fftshift, ifftshift, rfftfreq
from scipy.fft import next_fast_len

__all__ = [
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "ifft",
    "irfft",
    "set_workers",
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
    "next_fast_len",
    "fold_shift_checkerboard",
]

_NUMPY_LINES = 8
_FFTW_FLAGS = ("FFTW_ESTIMATE", "FFTW_UNALIGNED")


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


def _fftw_threads(processing_function_threads):
    """The count as scipy.fft reads it: ``0`` is refused and a negative ``n`` is
    ``os.cpu_count() + 1 + n``; ``None`` is the count of ``scipy.fft.set_workers``."""
    if processing_function_threads is None:
        return scipy.fft.get_workers()
    with scipy.fft.set_workers(processing_function_threads):
        return scipy.fft.get_workers()


def _fftw_rfft2(x, processing_function_threads, s=None):
    """rfft2 of a plane with direct ``pyfftw.FFTW`` plans: the rows into a new half
    spectrum (zero padded to ``s``; through one aligned row, on one thread, when ``x`` is
    not aligned), then its columns in place."""
    import pyfftw

    threads = _fftw_threads(processing_function_threads)
    if s is not None:
        padded = np.zeros(s, dtype=x.dtype)
        padded[: x.shape[0], : x.shape[1]] = x
        x = padded
    ny, nx = x.shape
    spectrum = np.empty((ny, nx // 2 + 1), dtype=np.result_type(x.dtype, np.complex64))
    if x.flags.aligned:
        pyfftw.FFTW(
            x, spectrum, axes=(-1,), flags=_FFTW_FLAGS, threads=threads
        ).execute()
    else:
        row = pyfftw.empty_aligned(nx, dtype=x.dtype)
        plan = pyfftw.FFTW(row, spectrum[0], flags=_FFTW_FLAGS, threads=1)
        for j in range(ny):
            np.copyto(row, x[j])
            plan(output_array=spectrum[j])
        del plan, row
    pyfftw.FFTW(
        spectrum, spectrum, axes=(-2,), flags=_FFTW_FLAGS, threads=threads
    ).execute()
    return spectrum


def _fftw_irfft2(x, s, processing_function_threads):
    """irfft2 of a half spectrum to the plane shape ``s`` with direct ``pyfftw.FFTW``
    plans: the columns of a copy in place, then the rows into a new plane, scaled by
    ``1 / (s[0] s[1])``."""
    import pyfftw

    threads = _fftw_threads(processing_function_threads)
    work = np.array(x, dtype=np.result_type(x.dtype, np.complex64))
    pyfftw.FFTW(
        work,
        work,
        axes=(-2,),
        direction="FFTW_BACKWARD",
        flags=_FFTW_FLAGS,
        threads=threads,
    ).execute()
    out = np.empty(s, dtype=work.real.dtype)
    pyfftw.FFTW(
        work,
        out,
        axes=(-1,),
        direction="FFTW_BACKWARD",
        flags=_FFTW_FLAGS,
        threads=threads,
    ).execute()
    del work
    out *= 1.0 / (s[0] * s[1])
    return out


def _numpy_lines(function, x, out, axis, other, **kwargs):
    """numpy.fft's 1-D ``function`` along ``axis`` of ``x``, into ``out`` (``x`` itself or a
    new array).

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


def rfft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    processing_function_threads=None,
    *,
    fft_backend,
):
    """Two-dimensional FFT of a real array in the library ``fft_backend`` names.

    Parameters
    ----------
    x : numpy.ndarray
        Real input array.
    s, axes, norm, overwrite_x
        As ``scipy.fft.rfft2``. Under ``"numpy"`` a 2-D ``x`` on axes ``(-2, -1)``
        without ``s`` is transformed by numpy's 1-D transforms, 8 lines at a time, or
        one aligned row at a time when ``x`` is not aligned: its rows into a new half
        spectrum, whose columns are then transformed in place. Under ``"pyfftw"`` such
        an ``x``, also zero padded to a larger ``s``, without ``norm``, is transformed
        by direct ``pyfftw.FFTW`` plans: the rows into a new half spectrum (an ``x``
        that is not aligned through one aligned row, on one thread), then its columns
        in place.
    processing_function_threads : int, optional
        As ``fft2``; the direct plans read the count as ``scipy.fft`` does.
    fft_backend : {"numpy", "scipy", "pyfftw"}
        FFT library.

    Returns
    -------
    numpy.ndarray
        The half spectrum, complex in the precision of ``x``.

    Raises
    ------
    ValueError
        ``fft_backend`` is not one of the three names.
    ImportError
        The library is not installed.
    """
    library = _library(fft_backend)
    plane = x.ndim == 2 and tuple(axes) == (-2, -1)
    if fft_backend == "numpy":
        if plane and s is None:
            spectrum = np.empty(
                (x.shape[0], x.shape[1] // 2 + 1),
                dtype=np.result_type(x.dtype, np.complex64),
            )
            _numpy_lines(library.rfft, x, spectrum, -1, -2, norm=norm)
            return _numpy_lines(library.fft, spectrum, spectrum, -2, -1, norm=norm)
        return library.rfft2(x, s=s, axes=axes, norm=norm)
    padded = s is not None and len(s) == 2 and s[0] >= x.shape[0] and s[1] >= x.shape[1]
    if fft_backend == "pyfftw" and plane and norm is None and (s is None or padded):
        return _fftw_rfft2(x, processing_function_threads, s)
    return library.rfft2(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=processing_function_threads,
    )


def irfft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    processing_function_threads=None,
    *,
    fft_backend,
):
    """Two-dimensional inverse FFT to a real array in the library ``fft_backend`` names.

    Parameters
    ----------
    x : numpy.ndarray
        Half spectrum.
    s, axes, norm, overwrite_x
        As ``scipy.fft.irfft2``. Under ``"pyfftw"`` a 2-D ``x`` on axes ``(-2, -1)``
        whose shape is that of the half spectrum of ``s``, without ``norm``, is
        transformed by direct ``pyfftw.FFTW`` plans: the columns of a copy in place,
        then the rows into the new plane.
    processing_function_threads : int, optional
        As ``fft2``; the direct plans read the count as ``scipy.fft`` does.
    fft_backend : {"numpy", "scipy", "pyfftw"}
        FFT library.

    Returns
    -------
    numpy.ndarray
        The real transform, in the precision of ``x``.

    Raises
    ------
    ValueError
        ``fft_backend`` is not one of the three names.
    ImportError
        The library is not installed.
    """
    library = _library(fft_backend)
    if fft_backend == "numpy":
        return library.irfft2(x, s=s, axes=axes, norm=norm)
    if (
        fft_backend == "pyfftw"
        and s is not None
        and norm is None
        and x.ndim == 2
        and tuple(axes) == (-2, -1)
        and x.shape == (s[0], s[1] // 2 + 1)
    ):
        return _fftw_irfft2(x, tuple(s), processing_function_threads)
    return library.irfft2(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=processing_function_threads,
    )


def ifft(
    x,
    n=None,
    axis=-1,
    norm=None,
    overwrite_x=False,
    processing_function_threads=None,
    *,
    fft_backend,
):
    """One-dimensional inverse FFT in the library ``fft_backend`` names.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    n, axis, norm, overwrite_x
        As ``scipy.fft.ifft``. With ``overwrite_x`` and no ``n``, a complex 2-D ``x``
        is transformed in place under ``"numpy"`` by numpy's 1-D transform, 8 lines at
        a time, or one aligned row at a time when ``x`` is not aligned; under
        ``"pyfftw"`` a complex aligned ``x`` with ``norm="forward"`` by an in-place
        ``pyfftw.FFTW`` plan.
    processing_function_threads : int, optional
        As ``fft2``; the direct plan reads the count as ``scipy.fft`` does.
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
    library = _library(fft_backend)
    complex_x = x.dtype.kind == "c"
    if fft_backend == "numpy":
        if overwrite_x and n is None and complex_x and x.ndim == 2:
            other = -1 if axis in (-2, 0) else -2
            return _numpy_lines(library.ifft, x, x, axis, other, norm=norm)
        return library.ifft(x, n=n, axis=axis, norm=norm)
    if (
        fft_backend == "pyfftw"
        and overwrite_x
        and n is None
        and norm == "forward"
        and complex_x
        and x.flags.aligned
    ):
        import pyfftw

        pyfftw.FFTW(
            x,
            x,
            axes=(axis,),
            direction="FFTW_BACKWARD",
            flags=_FFTW_FLAGS,
            threads=_fftw_threads(processing_function_threads),
        ).execute()
        return x
    return library.ifft(
        x,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=processing_function_threads,
    )


def irfft(
    x,
    n=None,
    axis=-1,
    norm=None,
    overwrite_x=False,
    processing_function_threads=None,
    *,
    fft_backend,
):
    """One-dimensional inverse FFT to a real array in the library ``fft_backend`` names.

    Parameters
    ----------
    x : numpy.ndarray
        Half spectrum.
    n, axis, norm, overwrite_x
        As ``scipy.fft.irfft``. With ``n``, a 2-D ``x`` along its last axis is
        transformed under ``"numpy"`` by numpy's 1-D transform into a new array, 8
        lines at a time, or one aligned row at a time when ``x`` is not aligned; under
        ``"pyfftw"`` a complex aligned ``x`` of ``n // 2 + 1`` points along ``axis``,
        with ``norm="forward"``, by a direct ``pyfftw.FFTW`` plan into a new array.
    processing_function_threads : int, optional
        As ``fft2``; the direct plan reads the count as ``scipy.fft`` does.
    fft_backend : {"numpy", "scipy", "pyfftw"}
        FFT library.

    Returns
    -------
    numpy.ndarray
        The real transform, in the precision of ``x``.

    Raises
    ------
    ValueError
        ``fft_backend`` is not one of the three names.
    ImportError
        The library is not installed.
    """
    library = _library(fft_backend)
    if fft_backend == "numpy":
        if n is not None and x.ndim == 2 and axis in (-1, 1):
            out = np.empty((x.shape[0], n), dtype=x.real.dtype)
            return _numpy_lines(library.irfft, x, out, -1, -2, n=n, norm=norm)
        return library.irfft(x, n=n, axis=axis, norm=norm)
    if (
        fft_backend == "pyfftw"
        and n is not None
        and norm == "forward"
        and x.dtype.kind == "c"
        and x.flags.aligned
        and x.shape[axis] == n // 2 + 1
    ):
        import pyfftw

        shape = list(x.shape)
        shape[axis] = n
        out = np.empty(shape, dtype=x.real.dtype)
        pyfftw.FFTW(
            x,
            out,
            axes=(axis,),
            direction="FFTW_BACKWARD",
            flags=_FFTW_FLAGS,
            threads=_fftw_threads(processing_function_threads),
        ).execute()
        return out
    return library.irfft(
        x,
        n=n,
        axis=axis,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=processing_function_threads,
    )


def set_workers(processing_function_threads, *, fft_backend):
    """Context manager: the count of the transforms called inside it without one.

    Parameters
    ----------
    processing_function_threads : int
        Threads; under ``"scipy"`` and ``"pyfftw"`` ``scipy.fft.set_workers``'s
        context, which ``scipy.fft``, pyFFTW's interface and the direct plans read.
        numpy.fft takes no count: under ``"numpy"`` the context changes nothing.
    fft_backend : {"numpy", "scipy", "pyfftw"}
        FFT library.

    Returns
    -------
    contextlib.AbstractContextManager
        The context.

    Raises
    ------
    ValueError
        ``fft_backend`` is not one of the three names, or, outside ``"numpy"``, the
        count is ``0`` or below ``-os.cpu_count()``.
    """
    _library(fft_backend)
    if fft_backend == "numpy":
        return contextlib.nullcontext()
    return scipy.fft.set_workers(processing_function_threads)
