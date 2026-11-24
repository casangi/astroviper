"""The FFT module runs each transform in the library its option names.

``processing_functions/fft.py`` (the C++ wrapper ``include/fft.hpp`` is tested in
``test_fft.py``): under ``"numpy"`` a complex transform without ``s``, a plane's
rfft2 without ``s``, a complex plane's ifft in place and irfft with ``n`` along a
plane's last axis run numpy's 1-D transforms on blocks of lines of each plane, or
one aligned row at a time when the array is not aligned, with the bits of numpy's
own transforms; under ``"pyfftw"`` an unaligned array transformed in place goes
through pyFFTW's line path, and a plane's rfft2 and irfft2, and ifft and irfft with
``norm="forward"``, run direct ``pyfftw.FFTW`` plans; every other call is the
library's own transform. The shifts, the sample frequencies and ``next_fast_len``
are the libraries' own.
"""

import os
import tracemalloc

import numpy as np
import pyfftw
import pyfftw.interfaces.cache
import pyfftw.interfaces.numpy_fft
import pyfftw.interfaces.scipy_fft
import pytest
import scipy.fft

from astroviper.processing_functions import fft

LIBRARIES = {
    "numpy": np.fft,
    "scipy": scipy.fft,
    "pyfftw": pyfftw.interfaces.scipy_fft,
}
LIBRARY_TRANSFORMS = (
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
)
# Planes, odd and even, and the stacks N and feather transform.
CASES = [((n, n), (-2, -1)) for n in (48, 63, 64, 97, 128, 160, 250)] + [
    ((3, 2, 48, 64), (-2, -1)),
    ((6, 3, 10), (0, 2)),
    ((1, 16, 1, 64, 64), (3, 4)),
]
# Bookkeeping of a transform besides its scratch arrays.
BOOKKEEPING = 8192


def _values(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return values.astype(dtype)


def _spilled(values):
    """``values`` copied into a view at byte offset 1, not aligned for its dtype."""
    backing = bytearray(values.nbytes + 2)
    view = np.ndarray(values.shape, dtype=values.dtype, buffer=backing, offset=1)
    view[...] = values
    return view


def _laid_out(values, layout):
    return values.copy() if layout == "aligned" else _spilled(values)


def _peak(call):
    """Peak traced memory of ``call()`` above the memory traced before it, after one
    untraced call (a process's first call of the module imports config_check). The
    thread of pyFFTW's plan cache copies the cache's dictionary at each cull, about
    every 0.1 s (pyFFTW's keepalive time, which the module keeps); culls are put off
    during the measured call."""
    call()
    cache = pyfftw.interfaces.cache.is_enabled()
    if cache:
        pyfftw.interfaces.cache.set_keepalive_time(3600.0)
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        result = call()
        return tracemalloc.get_traced_memory()[1] - before, result
    finally:
        tracemalloc.stop()
        if cache:
            pyfftw.interfaces.cache.set_keepalive_time(0.1)


@pytest.mark.parametrize("overwrite_x", [False, True])
@pytest.mark.parametrize("layout", ["aligned", "spilled"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape,axes", CASES, ids=str)
@pytest.mark.parametrize("name", ["fft2", "ifft2"])
@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_each_option_gives_its_library_bits(
    fft_backend, name, shape, axes, dtype, layout, overwrite_x
):
    """numpy's in-place blocks and aligned row give the bits of numpy's own 2-D
    transform; scipy and pyFFTW's interface are called as given; pyFFTW's line path
    gives its own bits, within 8 eps of the peak of the interface's."""
    values = _values(shape, dtype, sum(shape))
    x = _laid_out(values, layout)
    got = getattr(fft, name)(
        x,
        axes=axes,
        overwrite_x=overwrite_x,
        processing_function_threads=1,
        fft_backend=fft_backend,
    )
    assert got.dtype == dtype and got.shape == shape
    line_path = fft_backend == "pyfftw" and overwrite_x and layout == "spilled"
    if fft_backend == "numpy":
        expected = getattr(np.fft, name)(values.copy(), axes=axes)
    elif line_path:
        expected = fft._fftw_unaligned_inplace(
            _spilled(values), axes, 1, inverse=name == "ifft2"
        )
        interface = getattr(pyfftw.interfaces.scipy_fft, name)(values.copy(), axes=axes)
        bound = 8 * np.finfo(dtype).eps * np.abs(interface).max()
        assert np.abs(got - interface).max() <= bound
    else:
        expected = getattr(LIBRARIES[fft_backend], name)(
            _laid_out(values, layout), axes=axes, overwrite_x=overwrite_x, workers=1
        )
    np.testing.assert_array_equal(got, expected)
    if overwrite_x and (fft_backend == "numpy" or line_path):
        assert got is x
    elif not overwrite_x:
        np.testing.assert_array_equal(x, values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_real_input_takes_the_library_transform(fft_backend, dtype):
    """A real array is not transformed in place: its complex transform is a new
    array, in the precision of the input."""
    plane = np.random.default_rng(5).standard_normal((60, 80)).astype(dtype)
    for name in ("fft2", "ifft2"):
        x = plane.copy()
        got = getattr(fft, name)(
            x, overwrite_x=True, processing_function_threads=1, fft_backend=fft_backend
        )
        expected = getattr(LIBRARIES[fft_backend], name)(plane.copy())
        assert got.dtype == np.result_type(dtype, np.complex64)
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("layout", ["aligned", "spilled"])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("name", ["fft2", "ifft2", "rfft2"])
def test_numpy_passes_take_norm(name, norm, layout):
    """Under "numpy" the module's 1-D passes take norm as numpy.fft's own 2-D
    transform does."""
    if name == "rfft2":
        values = np.random.default_rng(9).standard_normal((48, 64))
    else:
        values = _values((48, 64), np.complex128)
    got = getattr(fft, name)(_laid_out(values, layout), norm=norm, fft_backend="numpy")
    np.testing.assert_array_equal(got, getattr(np.fft, name)(values.copy(), norm=norm))


@pytest.mark.parametrize("layout", ["aligned", "spilled"])
@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_only_the_selected_library_transforms(fft_backend, layout, monkeypatch):
    """The transforms of the other libraries are never called; under "numpy" a
    complex array, transformed in place or not, goes through numpy's 1-D
    transforms only; under "pyfftw" an unaligned plane transformed in place
    builds one pyfftw.FFTW plan per axis and calls no transform of the
    interface."""
    called = []
    for backend, library in LIBRARIES.items():
        for function_name in LIBRARY_TRANSFORMS:
            real = getattr(library, function_name)

            def spy(*args, _real=real, _label=f"{backend}.{function_name}", **kwargs):
                called.append(_label)
                return _real(*args, **kwargs)

            def refuse(*args, _label=f"{backend}.{function_name}", **kwargs):
                raise AssertionError(f"{_label} called under {fft_backend}")

            monkeypatch.setattr(
                library, function_name, spy if backend == fft_backend else refuse
            )
    plans = []
    real_plan = pyfftw.FFTW

    def plan(*args, **kwargs):
        plans.append(kwargs.get("threads"))
        return real_plan(*args, **kwargs)

    line_path = fft_backend == "pyfftw" and layout == "spilled"
    if line_path:
        monkeypatch.setattr(pyfftw, "FFTW", plan)
    values = _values((64, 80), np.complex128)
    for overwrite_x in (True, False) if fft_backend == "numpy" else (True,):
        for name in ("fft2", "ifft2"):
            getattr(fft, name)(
                _laid_out(values, layout),
                overwrite_x=overwrite_x,
                processing_function_threads=1,
                fft_backend=fft_backend,
            )
    if line_path:
        assert called == [] and plans == [1, 1, 1, 1]
    elif fft_backend == "numpy":
        assert set(called) == {"numpy.fft", "numpy.ifft"}
    else:
        assert called == [f"{fft_backend}.fft2", f"{fft_backend}.ifft2"]


@pytest.mark.parametrize("name", [None, "", "fftw", "NumPy", "SciPy", "pyFFTW"])
def test_unknown_names_are_refused(name):
    values = _values((8, 8), np.complex128)
    for function in (fft.fft2, fft.ifft2):
        with pytest.raises(
            ValueError, match="fft_backend must be 'numpy', 'scipy' or 'pyfftw'"
        ):
            function(values.copy(), fft_backend=name)


@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_counts_below_one(fft_backend, monkeypatch):
    """numpy takes no count; scipy and pyFFTW's interface refuse 0 and a count below
    -os.cpu_count() and take -1, None and 3; the line path hands the count to
    pyfftw.FFTW as an unsigned int and reads None as the set_workers count."""
    values = _values((32, 40), np.complex128)
    one = fft.ifft2(
        values.copy(), processing_function_threads=1, fft_backend=fft_backend
    )
    too_negative = -(os.cpu_count() + 1)
    for count in (0, -1, too_negative, None, 3):
        call = lambda count=count: fft.ifft2(  # noqa: E731
            values.copy(), processing_function_threads=count, fft_backend=fft_backend
        )
        if fft_backend != "numpy" and count in (0, too_negative):
            with pytest.raises(ValueError):
                call()
        elif fft_backend == "numpy":
            np.testing.assert_array_equal(call(), one)
        else:
            call()
    if fft_backend != "pyfftw":
        return
    plans = []
    real_plan = pyfftw.FFTW

    def plan(*args, **kwargs):
        plans.append(kwargs["threads"])
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(pyfftw, "FFTW", plan)

    def line_path(count):
        return fft.ifft2(
            _spilled(values),
            overwrite_x=True,
            processing_function_threads=count,
            fft_backend="pyfftw",
        )

    line_path(0)
    with pytest.raises(OverflowError):
        line_path(-1)
    plans.clear()
    with scipy.fft.set_workers(3):
        line_path(None)
    line_path(None)
    assert plans == [3, 3, 1, 1]


@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_single_precision_stays_single(fft_backend):
    for layout in ("aligned", "spilled"):
        for overwrite_x in (False, True):
            for name in ("fft2", "ifft2"):
                got = getattr(fft, name)(
                    _laid_out(_values((40, 48), np.complex64), layout),
                    overwrite_x=overwrite_x,
                    fft_backend=fft_backend,
                )
                assert got.dtype == np.complex64


def test_the_shifts_and_next_fast_len_are_the_libraries():
    """numpy's shifts give the bits of scipy's and of both pyFFTW interfaces' at the
    sites' shapes; next_fast_len is scipy's (pyFFTW's sizes FFTW's own lengths)."""
    assert fft.fftshift is np.fft.fftshift
    assert fft.ifftshift is np.fft.ifftshift
    assert fft.next_fast_len is scipy.fft.next_fast_len
    others = (scipy.fft, pyfftw.interfaces.scipy_fft, pyfftw.interfaces.numpy_fft)
    for shape, axes in [((75, 75), (-2, -1)), ((1, 16, 1, 63, 65), (3, 4))]:
        for dtype in (np.complex64, np.complex128, np.float32, np.float64):
            x = _values(shape, np.complex128).real.astype(dtype)
            for name in ("fftshift", "ifftshift"):
                got = getattr(fft, name)(x, axes=axes)
                for other in others:
                    np.testing.assert_array_equal(
                        got, getattr(other, name)(x, axes=axes)
                    )


def test_pyfftw_enables_the_plan_cache():
    """Only a "pyfftw" call enables pyFFTW's plan cache, which is the process's."""
    was_enabled = pyfftw.interfaces.cache.is_enabled()
    pyfftw.interfaces.cache.disable()
    try:
        values = _values((16, 16), np.complex128)
        for fft_backend in ("numpy", "scipy"):
            fft.fft2(values.copy(), fft_backend=fft_backend)
            assert not pyfftw.interfaces.cache.is_enabled()
        fft.fft2(values.copy(), fft_backend="pyfftw")
        assert pyfftw.interfaces.cache.is_enabled()
    finally:
        if not was_enabled:
            pyfftw.interfaces.cache.disable()


@pytest.mark.parametrize("layout", ["aligned", "spilled"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("name", ["fft2", "ifft2"])
def test_numpy_transforms_in_place(name, dtype, layout):
    """Under "numpy" a 512 x 512 plane transformed in place takes at most one
    aligned row of scratch (8 KiB of bookkeeping allowed), and in a single-precision
    forward pass numpy's double-precision copies of the lines of one call: 8 lines
    of an aligned plane, 1 of an unaligned one."""
    n = 512
    x = _laid_out(_values((n, n), dtype), layout)
    peak, got = _peak(
        lambda: getattr(fft, name)(x, overwrite_x=True, fft_backend="numpy")
    )
    assert got is x
    lines = 8 if layout == "aligned" else 1
    copies = 2 * lines * n * 16 if (dtype == np.complex64 and name == "fft2") else 0
    row = 0 if layout == "aligned" else n * np.dtype(dtype).itemsize
    assert peak <= row + copies + BOOKKEEPING


@pytest.mark.parametrize("layout", ["aligned", "spilled"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("name", ["fft2", "ifft2"])
def test_numpy_out_of_place_takes_one_new_plane(name, dtype, layout):
    """Under "numpy" a 511 x 511 plane transformed out of place is left as it is
    and the transform takes one new plane, plus one aligned row when the plane is
    not aligned, plus in a single-precision forward pass numpy's double-precision
    copies of a block of 8 lines (8 KiB of bookkeeping allowed)."""
    n = 511
    values = _values((n, n), dtype)
    x = _laid_out(values, layout)
    peak, got = _peak(lambda: getattr(fft, name)(x, fft_backend="numpy"))
    assert not np.shares_memory(got, x)
    np.testing.assert_array_equal(x, values)
    copies = 2 * 8 * n * 16 if (dtype == np.complex64 and name == "fft2") else 0
    row = 0 if layout == "aligned" else n * np.dtype(dtype).itemsize
    assert peak <= values.nbytes + row + copies + BOOKKEEPING


def test_numpy_blocks_stay_in_one_plane_of_a_stack():
    """Under "numpy" the blocks of 8 lines of an aligned (8, 256, 256) complex64
    stack transformed forward in place stay in one plane: the double-precision
    copies of one block, not of 8 lines of every plane."""
    x = _values((8, 256, 256), np.complex64)
    peak, got = _peak(lambda: fft.fft2(x, overwrite_x=True, fft_backend="numpy"))
    assert got is x
    assert peak <= 2 * 8 * 256 * 16 + BOOKKEEPING


def test_scipy_transforms_an_aligned_plane_in_place():
    x = _values((512, 512), np.complex128)
    peak, got = _peak(lambda: fft.ifft2(x, overwrite_x=True, fft_backend="scipy"))
    assert np.shares_memory(got, x)
    assert peak <= 4096


# The forms of the restore and the automatic mask: a plane's rfft2 (also zero padded
# to s), irfft2 to a plane's shape, ifft in place along the first axis and irfft along
# the last, both with norm="forward".
RESTORE_SIZES = [48, 63, 64, 97, 128, 160, 250]


def _real_plane(n, dtype, seed=0):
    return np.random.default_rng(seed).standard_normal((n, n + 1)).astype(dtype)


def _forms(x, fft_backend, threads=1):
    """The four forms on ``x`` (a real plane), each on its own copy; ``x`` is left as
    it is."""
    ny, nx = x.shape
    spectrum = fft.rfft2(
        x, processing_function_threads=threads, fft_backend=fft_backend
    )
    half = fft.ifft(
        spectrum.copy(),
        axis=-2,
        norm="forward",
        overwrite_x=True,
        processing_function_threads=threads,
        fft_backend=fft_backend,
    )
    plane = fft.irfft(
        half,
        n=nx,
        axis=-1,
        norm="forward",
        processing_function_threads=threads,
        fft_backend=fft_backend,
    )
    back = fft.irfft2(
        spectrum,
        (ny, nx),
        processing_function_threads=threads,
        fft_backend=fft_backend,
    )
    return spectrum, half, plane, back


def _library_forms(x, library, **workers):
    ny, nx = x.shape
    spectrum = library.rfft2(x, **workers)
    half = library.ifft(spectrum, axis=-2, norm="forward", **workers)
    plane = library.irfft(half, n=nx, axis=-1, norm="forward", **workers)
    back = library.irfft2(spectrum, (ny, nx), **workers)
    return spectrum, half, plane, back


@pytest.mark.parametrize("layout", ["aligned", "spilled"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", RESTORE_SIZES)
@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_restore_forms_give_their_library_bits(fft_backend, n, dtype, layout):
    """numpy's line blocks give numpy.fft's own bits and scipy is called as given;
    pyFFTW's direct plans stay within 8 eps of the peak of pyFFTW's interface. The
    input is left as it is and the precision kept."""
    values = _real_plane(n, dtype, n)
    x = _laid_out(values, layout)
    got = _forms(x, fft_backend)
    np.testing.assert_array_equal(x, values)
    complex_dtype = np.result_type(dtype, np.complex64)
    for result, kind in zip(got, ("c", "c", "f", "f"), strict=True):
        assert result.dtype == (complex_dtype if kind == "c" else dtype)
    if fft_backend == "pyfftw":
        expected = _library_forms(values.copy(), pyfftw.interfaces.scipy_fft)
        for result, reference in zip(got, expected, strict=True):
            bound = 8 * np.finfo(dtype).eps * np.abs(reference).max()
            assert np.abs(result - reference).max() <= bound
    else:
        workers = {"workers": 1} if fft_backend == "scipy" else {}
        expected = _library_forms(values.copy(), LIBRARIES[fft_backend], **workers)
        for result, reference in zip(got, expected, strict=True):
            np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("size", [64, 182, 600])
def test_mask_padded_forms_under_pyfftw_are_the_interface_bits(size):
    """The automatic mask's padded transforms (a box zero padded to next_fast_len
    of the box plus the kernel) under "pyfftw" give the bits of pyFFTW's
    interface."""
    rng = np.random.default_rng(size)
    box = (rng.random((size, size)) > 0.7).astype(np.float64)
    kernel = rng.random((37, 37))
    shape = [fft.next_fast_len(s + 36, real=True) for s in box.shape]
    interface = pyfftw.interfaces.scipy_fft
    spectrum = fft.rfft2(box, shape, fft_backend="pyfftw")
    np.testing.assert_array_equal(spectrum, interface.rfft2(box, shape))
    spectrum *= fft.rfft2(kernel, shape, fft_backend="pyfftw")
    expected = interface.rfft2(box, shape) * interface.rfft2(kernel, shape)
    np.testing.assert_array_equal(
        fft.irfft2(spectrum, shape, fft_backend="pyfftw"),
        interface.irfft2(expected, shape),
    )


@pytest.mark.parametrize("layout", ["aligned", "spilled"])
@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_restore_forms_run_only_the_selected_library(fft_backend, layout, monkeypatch):
    """The transforms of the other libraries are never called; under "numpy" the
    forms of a plane run numpy's 1-D transforms, and irfft2 numpy's own; under
    "pyfftw" they build pyfftw.FFTW plans and call no transform of the interface."""
    called = []
    for backend, library in LIBRARIES.items():
        for function_name in LIBRARY_TRANSFORMS:
            real = getattr(library, function_name)

            def spy(*args, _real=real, _label=f"{backend}.{function_name}", **kwargs):
                called.append(_label)
                return _real(*args, **kwargs)

            def refuse(*args, _label=f"{backend}.{function_name}", **kwargs):
                raise AssertionError(f"{_label} called under {fft_backend}")

            monkeypatch.setattr(
                library,
                function_name,
                refuse if backend != fft_backend or backend == "pyfftw" else spy,
            )
    plans = []
    real_plan = pyfftw.FFTW

    def plan(*args, **kwargs):
        plans.append(kwargs.get("threads"))
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(pyfftw, "FFTW", plan)
    _forms(_laid_out(_real_plane(64, np.float64), layout), fft_backend)
    if fft_backend == "pyfftw":
        assert called == [] and len(plans) >= 6 and set(plans) == {1}
    elif fft_backend == "numpy":
        assert set(called) == {
            "numpy.rfft",
            "numpy.fft",
            "numpy.ifft",
            "numpy.irfft2",
            "numpy.irfft",
        }
        assert plans == []
    else:
        assert called == ["scipy.rfft2", "scipy.ifft", "scipy.irfft", "scipy.irfft2"]
        assert plans == []


@pytest.mark.parametrize("name", [None, "", "fftw", "NumPy", "SciPy", "pyFFTW"])
def test_restore_forms_refuse_unknown_names(name):
    plane = _real_plane(8, np.float64)
    spectrum = np.fft.rfft2(plane)
    calls = (
        lambda: fft.rfft2(plane, fft_backend=name),
        lambda: fft.irfft2(spectrum, plane.shape, fft_backend=name),
        lambda: fft.ifft(spectrum.copy(), axis=-2, overwrite_x=True, fft_backend=name),
        lambda: fft.irfft(spectrum, n=9, fft_backend=name),
        lambda: fft.set_workers(1, fft_backend=name),
    )
    for call in calls:
        with pytest.raises(
            ValueError, match="fft_backend must be 'numpy', 'scipy' or 'pyfftw'"
        ):
            call()


@pytest.mark.parametrize("length", [7, 9, 10])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_irfft_takes_any_half_spectrum_length(fft_backend, axis, length):
    """irfft with ``n=16`` and norm="forward" truncates or zero pads a half spectrum
    of another length than 9 as scipy.fft.irfft does, within 8 eps of its peak."""
    x = _values((length, 8) if axis == 0 else (8, length), np.complex128)
    expected = scipy.fft.irfft(x, n=16, axis=axis, norm="forward")
    got = fft.irfft(x.copy(), n=16, axis=axis, norm="forward", fft_backend=fft_backend)
    assert got.shape == expected.shape
    bound = 8 * np.finfo(np.float64).eps * np.abs(expected).max()
    assert np.abs(got - expected).max() <= bound


@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_restore_forms_counts_below_one(fft_backend, monkeypatch):
    """numpy takes no count; scipy and pyFFTW's direct plans refuse 0 and a count
    below -os.cpu_count(); the direct plans read -1 as every CPU and None as the
    set_workers count."""
    plane = _real_plane(32, np.float64)
    one = _forms(plane, fft_backend, threads=1)
    plans = []
    real_plan = pyfftw.FFTW

    def plan(*args, **kwargs):
        plans.append(kwargs["threads"])
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(pyfftw, "FFTW", plan)
    cpus = os.cpu_count()
    for count in (0, -(cpus + 1)):
        if fft_backend == "numpy":
            for result, reference in zip(
                _forms(plane, fft_backend, count), one, strict=True
            ):
                np.testing.assert_array_equal(result, reference)
        else:
            with pytest.raises(ValueError):
                fft.rfft2(
                    plane, processing_function_threads=count, fft_backend=fft_backend
                )
    if fft_backend != "pyfftw":
        return
    plans.clear()
    _forms(plane, "pyfftw", -1)
    assert set(plans) == {cpus}
    plans.clear()
    with fft.set_workers(3, fft_backend="pyfftw"):
        _forms(plane, "pyfftw", None)
    _forms(plane, "pyfftw", None)
    assert plans == [3] * (len(plans) // 2) + [1] * (len(plans) // 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["aligned", "spilled"])
@pytest.mark.parametrize("n", [60, 160])
def test_rfft2_pyfftw_gives_one_thread_bits_with_three(n, layout, dtype):
    """pyFFTW's direct rfft2 plans with 3 threads give the bits of one thread on
    planes of 60 and 160, aligned and spilled (the spilled rows run one thread)."""
    values = _real_plane(n, dtype, n)[:, :n]
    one = fft.rfft2(
        _laid_out(values, layout), processing_function_threads=1, fft_backend="pyfftw"
    )
    three = fft.rfft2(
        _laid_out(values, layout), processing_function_threads=3, fft_backend="pyfftw"
    )
    np.testing.assert_array_equal(three, one)


def test_set_workers_is_scipys_but_under_numpy():
    """Under "scipy" and "pyfftw" set_workers is scipy.fft's context; under
    "numpy" it changes nothing."""
    with fft.set_workers(3, fft_backend="numpy"):
        assert scipy.fft.get_workers() == 1
    for fft_backend in ("scipy", "pyfftw"):
        with fft.set_workers(3, fft_backend=fft_backend):
            assert scipy.fft.get_workers() == 3
        with pytest.raises(ValueError):
            with fft.set_workers(0, fft_backend=fft_backend):
                pass


def test_forward_plans_keep_their_input():
    """The direct plans are made without FFTW_DESTROY_INPUT."""
    flags = []
    real_plan = pyfftw.FFTW

    def plan(*args, **kwargs):
        flags.append(kwargs["flags"])
        return real_plan(*args, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(pyfftw, "FFTW", plan):
        _forms(_real_plane(64, np.float32), "pyfftw")
    assert flags and all("FFTW_DESTROY_INPUT" not in f for f in flags)


def _restore_helpers(dtype, fft_backend):
    """The restore's helpers on 512 x 512 planes, as calls (design v2 10.1)."""
    from astroviper.processing_functions.imaging.imaging_conclude import restore

    n, delta = 512, 1e-6
    rng = np.random.default_rng(0)
    plane = rng.standard_normal((n, n)).astype(dtype)
    primary_beam = np.linspace(0.05, 1.0, n * n).reshape(n, n).astype(dtype)
    covariance = np.array([[3.0, 0.4], [0.4, 2.0]])
    beam = [6 * delta, 4 * delta, 0.3]
    kernel_ft = restore._clean_beam_kernel_ft(beam, n, n, delta, dtype, 1, fft_backend)
    return {
        "_clean_beam_kernel_ft": lambda: restore._clean_beam_kernel_ft(
            beam, n, n, delta, dtype, 1, fft_backend
        ),
        "_convolve_with_clean_beam": lambda: restore._convolve_with_clean_beam(
            plane, kernel_ft, 1, fft_backend
        ),
        "_residual_to_beam": lambda: restore._residual_to_beam(
            plane, covariance, 1.3, 1, fft_backend
        ),
        "primary_beam_corrected_plane": lambda: restore.primary_beam_corrected_plane(
            primary_beam,
            0.2,
            plane,
            plane,
            kernel_ft=kernel_ft,
            workers=1,
            fft_backend=fft_backend,
        ),
        "_restore_plane_to_beam": lambda: restore._restore_plane_to_beam(
            plane,
            plane,
            kernel_ft,
            covariance,
            1.3,
            1,
            apparent_plane=np.empty_like(plane),
            fft_backend=fft_backend,
        ),
        "_restore_plane_to_beam corrected": lambda: restore._restore_plane_to_beam(
            plane,
            plane,
            kernel_ft,
            covariance,
            1.3,
            1,
            apparent_plane=np.empty_like(plane),
            corrected_plane=np.empty_like(plane),
            primary_beam_plane=primary_beam,
            fft_backend=fft_backend,
        ),
    }


def _largest_peak(call, repeats=3):
    """Largest of ``repeats`` peaks of ``_peak(call)``; the garbage collector runs
    before each and not during it, so that the finalizers of other tests' objects do
    not count."""
    import gc

    peaks = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            peak, result = _peak(call)
        finally:
            gc.enable()
        del result
        peaks.append(peak)
    return max(peaks)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fft_backend", ["numpy", "pyfftw"])
def test_restore_helpers_take_scipys_memory(fft_backend, dtype):
    """Each helper of the restore on a 512 x 512 plane takes at most its "scipy"
    peak and 8 KiB of bookkeeping; under "numpy" in single precision also numpy's
    double-precision copies of one block of 8 lines of 512."""
    scipy_calls = _restore_helpers(dtype, "scipy")
    copies = 2 * 8 * 512 * 16 if (fft_backend == "numpy" and dtype == np.float32) else 0
    for name, call in _restore_helpers(dtype, fft_backend).items():
        bound = _largest_peak(scipy_calls[name]) + copies + BOOKKEEPING
        assert _largest_peak(call) <= bound, name


@pytest.mark.parametrize("fft_backend", ["numpy", "pyfftw"])
def test_mask_padded_forms_take_scipys_memory(fft_backend):
    """The automatic mask's convolution of a box of 600 takes at most its "scipy"
    peak and 8 KiB of bookkeeping."""
    import importlib

    make_auto_mask = importlib.import_module(
        "astroviper.processing_functions.image_analysis.make_auto_mask"
    )
    rng = np.random.default_rng(3)
    box = rng.random((600, 600)) > 0.7
    kernel = rng.random((37, 37))
    scipy_peak = _largest_peak(lambda: make_auto_mask._convolve(box, kernel, "scipy"))
    peak = _largest_peak(lambda: make_auto_mask._convolve(box, kernel, fft_backend))
    assert peak <= scipy_peak + BOOKKEEPING


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pyfftw_spilled_rfft2_takes_one_row(dtype):
    """Under "pyfftw" the rfft2 of a spilled 512 x 512 plane takes its half spectrum,
    one aligned row and 8 KiB of bookkeeping: no plane-sized realignment copy."""
    n = 512
    x = _spilled(np.random.default_rng(4).standard_normal((n, n)).astype(dtype))
    peak = _largest_peak(lambda: fft.rfft2(x, fft_backend="pyfftw"))
    half = n * (n // 2 + 1) * np.dtype(np.result_type(dtype, np.complex64)).itemsize
    assert peak <= half + n * np.dtype(dtype).itemsize + BOOKKEEPING
