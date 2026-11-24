"""The FFT module runs each transform in the library its option names.

``processing_functions/fft.py`` (the C++ wrapper ``include/fft.hpp`` is tested in
``test_fft.py``): under ``"numpy"`` a complex transform without ``s`` runs numpy's
1-D transforms on blocks of lines of each plane, or one aligned row at a time when
the array is not aligned, in place or into one new array, with the bits of numpy's
own 2-D transform; under ``"pyfftw"`` an unaligned array transformed in place goes
through pyFFTW's line path; every other call is the library's own 2-D transform.
The shifts and ``next_fast_len`` are the libraries' own.
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
@pytest.mark.parametrize("name", ["fft2", "ifft2"])
def test_numpy_passes_take_norm(name, norm, layout):
    """Under "numpy" the module's 1-D passes take norm as numpy.fft's own 2-D
    transform does."""
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
