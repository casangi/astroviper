"""Tests of the GPU uv <-> lm transforms (``ASTROVIPER_KERNELS=fft=gpu``).

The implementation is called directly (``astroviper._kernels.impl``), so these
tests run in any session; those that need the GPU skip without CuPy or a CUDA
device. Planes of up to 127 x 131 pixels (4800 x 4800 on the DATA side). The
oracle is the reference's transform evaluated in long double: the shifts, then
a direct DFT along each axis (twiddles of the reduced angles, the inverse
divided by the pixel count), skipped where long double is double.

* Accuracy: random, smooth and point-source planes of even, odd and mixed
  sizes (primes included), complex64 and complex128, 2-D and stacked, through
  both entries: the GPU's result and the reference's (both backends) are
  within the bound of the module docstring, ``ACCURACY * eps * log2(N) *
  ||y||_2`` per plane, in norm and element by element. The GPU's bits do not
  change from call to call, nor with a plane's place in a stack.
* Contract: the result's dtype, shape and C order, real inputs and
  ``complex_dtype``; without ``overwrite_input`` the input is kept, with it the
  result is in the input's buffer; the thread count and the backend change no
  bit.
* Paths: the calls the GPU entries do not take run the reference (its results,
  exceptions and input changes; ``declined``); planes below the floor or with
  an axis of one pixel, and calls that do not fit in ``CUPY_GPU_MEMORY_LIMIT``,
  run the reference (``routed``); the trace names each call's path and a GPU
  call's times; the floor is validated; the FFT module's own callers and
  feather.py go through its hook.
* Failures: without a device the first GPU call raises RuntimeError and the
  routed calls still run; the prewarm compiles the kernels and loads cuFFT.
* The generated code: no fused or flushed operation in the PTX.
"""

import importlib.resources
import importlib.util
import json
import math
import re
import socket

import numpy as np
import pytest

from astroviper._kernels import impl
from astroviper._kernels.__main__ import main as check_main
from astroviper._kernels._switch import ENV, TRACE_ENV, dispatch
from astroviper._kernels.gpu import _runtime, fft, prewarm
from tests.unit.kernels.conftest import SELF_HOOKED
from tests.unit.kernels.oracles import AVAILABLE, TWO_PI

ENTRIES = ("ifft_uv_to_lm", "fft_lm_to_uv")
GPU = {entry: impl("fft", entry, "gpu") for entry in ENTRIES}
REFERENCE = {entry: impl("fft", entry, "ref") for entry in ENTRIES}
LONG = np.clongdouble
needs_oracle = pytest.mark.skipif(not AVAILABLE, reason="long double is double")


@pytest.fixture(scope="module")
def device():
    try:
        return _runtime.require_device()
    except RuntimeError as e:
        pytest.skip(f"no GPU: {e}")


@pytest.fixture
def gpu(device, monkeypatch):
    """Every plane on the GPU; the list of the entries of the calls it ran."""
    monkeypatch.setenv(fft.MIN_PIXELS_ENV, "0")
    runs, run = [], fft._run

    def counted(call, stats=None):
        runs.append(call.entry)
        return run(call, stats)

    monkeypatch.setattr(fft, "_run", counted)
    return runs


@pytest.fixture
def paths(monkeypatch):
    """The paths the GPU entries report, as (path, fields) in call order."""
    reported = []
    monkeypatch.setattr(
        fft, "report_path", lambda kernel, path, **f: reported.append((path, f))
    )
    return reported


# ---------------------------------------------------------------- the oracle
def _dft(n, sign):
    """exp(sign 2 pi i j k / n) in long double, from the angles of j k mod n."""
    jk = np.outer(np.arange(n), np.arange(n)) % n
    angle = sign * TWO_PI * jk.astype(np.longdouble) / n
    return np.cos(angle) + 1j * np.sin(angle)


def _oracle(values, inverse):
    """The reference's transform of the planes of ``values`` (the last two
    axes), in long double: the DFT along each axis (divided by the pixel count
    for the inverse) between the checkerboard sign flips when both axes are
    even, else between ifftshift and fftshift. (The checkerboard form is the
    shifted transform times (-1)**(n0 / 2 + n1 / 2), which the reference keeps:
    so does the GPU.)"""
    x = np.asarray(values).astype(LONG)
    n0, n1 = x.shape[-2:]
    even = n0 % 2 == 0 and n1 % 2 == 0
    signs = (-1.0) ** np.add.outer(np.arange(n0), np.arange(n1))
    x = x * signs if even else np.fft.ifftshift(x, axes=(-2, -1))
    sign = 1 if inverse else -1
    y = _dft(n0, sign) @ x @ _dft(n1, sign)  # the DFT matrices are symmetric
    if inverse:
        y /= n0 * n1
    return y * signs if even else np.fft.fftshift(y, axes=(-2, -1))


def _ratios(result, y):
    """Per plane, the error of ``result`` from the oracle's ``y`` in norm and
    the largest error of an element, in units of the bound ACCURACY eps
    log2(N) ||y||_2 (within it: <= 1)."""
    eps = np.finfo(result.dtype).eps
    n0, n1 = y.shape[-2:]
    errors = np.abs(result.astype(LONG) - y).reshape(-1, n0 * n1)
    norms = np.sqrt((np.abs(y) ** 2).reshape(-1, n0 * n1).sum(axis=1))
    bound = fft.ACCURACY * eps * math.log2(n0 * n1) * norms
    return (
        np.sqrt((errors**2).sum(axis=1)) / bound,
        errors.max(axis=1) / bound,
    )


def _plane(shape, dtype, kind, seed=0):
    """A plane (or stack) of ``kind``: random, smooth (a Gaussian off the
    centre) or point (one pixel off the centre), complex or real."""
    rng = np.random.default_rng(seed)
    if kind == "random":
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    else:
        i = np.arange(shape[-2])[:, None] - (shape[-2] // 2 + 1)
        j = np.arange(shape[-1])[None, :] - (shape[-1] // 2 - 1)
        if kind == "smooth":
            x = np.exp(-(i**2 + j**2) / 8.0) * np.exp(0.3j * (i - j))
        else:
            x = np.where((i == 0) & (j == 0), 2.5 - 1.0j, 0)
        x = np.broadcast_to(x, shape) * (1 + rng.random(shape[:-2] + (1, 1)))
    if np.dtype(dtype).kind == "f":
        x = x.real
    return np.ascontiguousarray(x, dtype=dtype)


# ---------------------------------------------------------------- accuracy
SHAPES = [
    (8, 8),
    (6, 10),
    (10, 12),
    (64, 64),
    (9, 15),
    (8, 7),
    (13, 8),
    (127, 131),
    (2, 3, 16, 12),
    (3, 33, 31),
]


@needs_oracle
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("kind", ["random", "smooth", "point"])
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_the_gpu_and_the_reference_are_within_the_fft_bound(
    gpu, entry, dtype, kind, shape
):
    x = _plane(shape, dtype, kind, seed=sum(shape))
    kwargs = {"complex_dtype": dtype} if entry == "fft_lm_to_uv" else {}
    inverse = entry == "ifft_uv_to_lm"
    result = GPU[entry](x.copy(), **kwargs)
    assert gpu == [entry]
    y = _oracle(x, inverse)
    for norm, element in [_ratios(result, y)] + [
        _ratios(REFERENCE[entry](x.copy(), fft_backend=b, **kwargs), y)
        for b in ("pyfftw", "scipy")
    ]:
        assert norm.max() <= 1.0 and element.max() <= 1.0, (norm, element)
    again = GPU[entry](x.copy(), **kwargs)
    assert again.tobytes() == result.tobytes()


@needs_oracle
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_images_and_complex_dtypes(gpu, dtype):
    """``fft_lm_to_uv`` of a real image at either ``complex_dtype`` (cast as
    NumPy casts it), and ``ifft_uv_to_lm`` of a real grid (the complex type of
    its precision), as the reference types them."""
    x = _plane((24, 18), dtype, "random", seed=3)
    for complex_dtype in (np.complex64, np.complex128):
        result = GPU["fft_lm_to_uv"](x, complex_dtype=complex_dtype)
        reference = REFERENCE["fft_lm_to_uv"](x, complex_dtype=complex_dtype)
        assert result.dtype == reference.dtype == complex_dtype
        y = _oracle(x.astype(complex_dtype), False)
        assert max(r.max() for r in _ratios(result, y)) <= 1.0
    result = GPU["ifft_uv_to_lm"](x)
    assert result.dtype == REFERENCE["ifft_uv_to_lm"](x).dtype
    y = _oracle(x.astype(result.dtype), True)
    assert max(r.max() for r in _ratios(result, y)) <= 1.0
    assert len(gpu) == 3


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_planes_bits_do_not_depend_on_its_stack(gpu, entry):
    stack = _plane((2, 3, 20, 14), np.complex128, "random", seed=4)
    together = GPU[entry](stack)
    for index in np.ndindex(stack.shape[:-2]):
        alone = GPU[entry](stack[index].copy())
        assert alone.tobytes() == together[index].tobytes()


# ---------------------------------------------------------------- contract
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("shape", [(12, 10), (2, 11, 9)], ids=str)
def test_the_result_is_typed_and_laid_out_as_the_references(gpu, entry, shape):
    for dtype in (np.complex64, np.complex128, np.float32, np.float64):
        x = _plane(shape, dtype, "random", seed=5)
        original = x.copy()
        result = GPU[entry](x)
        reference = REFERENCE[entry](x.copy())
        assert (result.dtype, result.shape) == (reference.dtype, reference.shape)
        assert result.flags.c_contiguous and result.flags.writeable
        assert x.tobytes() == original.tobytes()  # without overwrite_input
        assert not np.shares_memory(result, x)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("shape", [(12, 10), (11, 9)], ids=str)
def test_overwrite_input_puts_the_result_in_the_inputs_buffer(gpu, entry, shape):
    x = _plane(shape, np.complex128, "random", seed=6)
    expected = GPU[entry](x.copy())
    result = GPU[entry](x, overwrite_input=True)
    assert result is x and result.tobytes() == expected.tobytes()
    # an input of another dtype is converted, and kept
    image = _plane(shape, np.float64, "random", seed=7)
    original = image.copy()
    result = GPU[entry](image, overwrite_input=True)
    assert result.dtype == np.complex128 and image.tobytes() == original.tobytes()


@pytest.mark.parametrize("entry", ENTRIES)
def test_threads_and_backends_change_no_bit(gpu, entry):
    x = _plane((3, 16, 20), np.complex64, "random", seed=8)
    results = {
        GPU[entry](x, (-2, -1), threads, backend).tobytes()
        for threads in (1, 7)
        for backend in ("pyfftw", "scipy")
    }
    results.add(
        GPU[entry](x, fft_plane_dims=[1, 2], processing_function_threads=2).tobytes()
    )
    assert len(results) == 1 and len(gpu) == 5


# ---------------------------------------------------------------- paths
def _bytes(array):
    """The bytes of an array's values: of a long double, the 80 bits of the
    x87 format, not the padding that no operation defines."""
    array = np.ascontiguousarray(array)
    if array.dtype.kind in "fc" and np.finfo(array.dtype).nmant == 63:
        size = np.dtype(np.longdouble).itemsize
        return array.view(np.uint8).reshape(-1, size)[:, :10].tobytes()
    return array.tobytes()


def _outcome(function, make):
    """The result of ``function(*args, **kwargs)`` for ``(args, kwargs) =
    make()`` as bytes, dtype, shape and C order, or its exception, and the
    bytes of the array arguments after the call."""
    args, kwargs = make()
    try:
        result = function(*args, **kwargs)
        out = (_bytes(result), result.dtype, result.shape, result.flags.c_contiguous)
    except Exception as e:
        out = (type(e), str(e))
    arrays = [a for a in (*args, *kwargs.values()) if isinstance(a, np.ndarray)]
    return out, [_bytes(a) for a in arrays]


def _args(*args, plane=None, **kwargs):
    def make():
        x = (
            _plane((12, 10), np.complex128, "random", seed=9)
            if plane is None
            else plane()
        )
        return (x, *args), kwargs

    return make


def _readonly():
    x = _plane((12, 10), np.complex128, "random", seed=10)
    x.flags.writeable = False
    return x


#: Calls the GPU entries decline: the reference runs them as given.
DECLINED = {
    "fortran order": _args(
        plane=lambda: np.asfortranarray(_plane((12, 10), np.complex128, "random"))
    ),
    "strided": _args(plane=lambda: _plane((12, 20), np.complex128, "random")[:, ::2]),
    "float16": _args(plane=lambda: _plane((12, 10), np.float16, "random")),
    "int64": _args(plane=lambda: np.arange(120).reshape(12, 10)),
    "clongdouble": _args(plane=lambda: _plane((12, 10), np.clongdouble, "random")),
    "big endian": _args(plane=lambda: _plane((12, 10), ">c16", "random")),
    "list": _args(plane=lambda: _plane((4, 6), np.complex128, "random").tolist()),
    "matrix": _args(
        plane=lambda: np.asmatrix(_plane((12, 10), np.complex128, "random"))
    ),
    "one axis": _args(plane=lambda: _plane((2, 10), np.complex128, "random")[0]),
    "empty": _args(plane=lambda: np.zeros((0, 10), np.complex128)),
    "axes reversed": _args((1, 0)),
    "axes repeated": _args((0, 0)),
    "one transform axis": _args((-1,)),
    "axes out of range": _args((1, 2)),
    "first two of three axes": _args(
        (0, 1), plane=lambda: _plane((12, 10, 3), np.complex128, "random")
    ),
    "a bool axis": _args((False, True)),
    "zero threads": _args(processing_function_threads=0),
    "all threads": _args(processing_function_threads=-1),
    "threads None": _args(processing_function_threads=None),
    "float threads": _args(processing_function_threads=2.0),
    "bool threads": _args(processing_function_threads=True),
    "unknown backend": _args(fft_backend="numpy"),
    "backend not a string": _args(fft_backend=3),
    "overwrite 1": _args(overwrite_input=1),
    "overwrite None": _args(overwrite_input=None),
    "read-only, overwrite": _args(plane=_readonly, overwrite_input=True),
    "unknown keyword": _args(threads=2),
    "too many arguments": _args((-2, -1), 1, "pyfftw", False, np.complex128, 7),
}
FORWARD_DECLINED = {
    "complex_dtype float64": _args(complex_dtype=np.float64),
    "complex_dtype clongdouble": _args(complex_dtype=np.clongdouble),
    "complex_dtype None": _args(complex_dtype=None),
    "complex_dtype not a dtype": _args(complex_dtype="complex999"),
}


@pytest.mark.parametrize(
    "entry, name",
    [(e, n) for e in ENTRIES for n in DECLINED]
    + [("fft_lm_to_uv", n) for n in FORWARD_DECLINED],
)
def test_declined_calls_are_the_reference(monkeypatch, paths, entry, name):
    """Their outcome (result or exception, and the arguments' bytes) is the
    reference's, and the GPU does not run."""
    monkeypatch.setenv(fft.MIN_PIXELS_ENV, "0")
    monkeypatch.setattr(fft, "_run", None)  # the GPU must not run
    make = (DECLINED | FORWARD_DECLINED)[name]
    assert _outcome(GPU[entry], make) == _outcome(REFERENCE[entry], make)
    assert [path for path, _ in paths] == ["declined"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "shape, floor, reason",
    [
        ((64, 64), None, f"below {fft.MIN_PIXELS_ENV}={fft.DEFAULT_MIN_PIXELS}"),
        ((2, 100, 50), "5001", "below ASTROVIPER_KERNELS_GPU_FFT_MIN_PIXELS=5001"),
        ((1, 64), "0", "an axis of one pixel"),
        ((3, 64, 1), "0", "an axis of one pixel"),
    ],
    ids=str,
)
def test_routed_calls_are_the_reference(
    monkeypatch, paths, entry, shape, floor, reason
):
    if floor is None:
        monkeypatch.delenv(fft.MIN_PIXELS_ENV, raising=False)
    else:
        monkeypatch.setenv(fft.MIN_PIXELS_ENV, floor)
    monkeypatch.setattr(fft, "_run", None)  # the GPU must not run
    make = _args(plane=lambda: _plane(shape, np.complex128, "random", seed=11))
    assert _outcome(GPU[entry], make) == _outcome(REFERENCE[entry], make)
    ((path, fields),) = paths
    assert path == "routed" and reason in fields["reason"]


def test_the_pixel_floor_is_validated(monkeypatch, capsys):
    monkeypatch.setenv(ENV, "fft=gpu")
    for value in ("abc", "-1", "1.5"):
        monkeypatch.setenv(fft.MIN_PIXELS_ENV, value)
        with pytest.raises(ValueError, match="expected a pixel count >= 0"):
            GPU["ifft_uv_to_lm"](np.zeros((4, 4), np.complex128))
        assert check_main(["--check"]) == 1
        assert f"{fft.MIN_PIXELS_ENV}={value!r}" in capsys.readouterr().err
    monkeypatch.setenv(fft.MIN_PIXELS_ENV, "0")
    assert fft.min_pixels() == 0


def _trace(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "fft=gpu")
    monkeypatch.setenv(TRACE_ENV, str(tmp_path))
    wrappers = {entry: dispatch("fft", REFERENCE[entry]) for entry in ENTRIES}

    def records():
        (path,) = tmp_path.glob(f"kernels-{socket.gethostname()}-*.jsonl")
        return [json.loads(line) for line in path.read_text().splitlines()]

    return wrappers, records


def test_the_trace_names_each_calls_path(device, monkeypatch, tmp_path):
    wrappers, records = _trace(monkeypatch, tmp_path)
    x = _plane((256, 192), np.complex128, "random", seed=12)
    monkeypatch.delenv(fft.MIN_PIXELS_ENV, raising=False)
    wrappers["ifft_uv_to_lm"](x)  # 49152 pixels: on the GPU
    wrappers["fft_lm_to_uv"](x[:64, :64].copy())  # below the floor
    wrappers["fft_lm_to_uv"](np.asfortranarray(x))  # declined
    with pytest.raises(ValueError):  # the reference's error, declined
        wrappers["ifft_uv_to_lm"](x, fft_backend="numpy")
    gpu_record, routed, declined, failed = records()
    assert [r["path"] for r in records()] == ["gpu", "routed", "declined", "declined"]
    assert {r["impl"] for r in records()} == {"gpu"}
    assert gpu_record["init_s"] >= 0 and gpu_record["device_s"] > 0
    assert set(gpu_record["versions"]) == {"driver", "runtime", "nvrtc"}
    assert "below" in routed["reason"] and "init_s" not in routed
    assert (declined["error"], failed["error"]) == (None, "ValueError")


LIMITED = """
import json, logging, numpy as np
from astroviper._kernels.gpu import fft

logging.basicConfig()
reported = []
fft.report_path = lambda kernel, path, **fields: reported.append([path, fields.get("reason")])
out = {}
for n in (64, 512):
    x = np.random.default_rng(n).standard_normal((2, n, n)).astype(np.complex128)
    reference = fft.impl("fft", "ifft_uv_to_lm", "ref")(x.copy(), overwrite_input=True)
    result = fft.ifft_uv_to_lm(x, overwrite_input=True)
    out[n] = [bool(np.array_equal(result, reference)), result is x]
fft.ifft_uv_to_lm(np.ones((512, 512), np.complex128))
print("RESULT", json.dumps({"out": out, "paths": reported}))
"""


def test_a_call_that_does_not_fit_runs_the_reference(device, run_python):
    """With a 1 MiB limit a plane of 64 x 64 (64 KiB, its plan as much) runs
    on the GPU and one of 512 x 512 (4 MiB) on the CPU, as the reference runs
    it, unchanged by the attempt; a warning is logged once."""
    done = run_python(
        "-c",
        LIMITED,
        check=False,
        CUPY_GPU_MEMORY_LIMIT="1048576",
        ASTROVIPER_KERNELS_GPU_FFT_MIN_PIXELS="0",
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.rpartition("RESULT ")[2])
    assert result["out"]["512"] == [True, False]  # the reference's own bits
    assert result["out"]["64"][1] is True
    paths = [p for p, _ in result["paths"]]
    assert paths == ["gpu", "routed", "routed"]
    assert "bytes of device memory" in result["paths"][1][1]
    assert done.stderr.count("the reference runs such calls") == 1


WITHOUT_DEVICE = """
import json, os, numpy as np
from astroviper._kernels.gpu import fft

x = np.ones((8, 8), np.complex128)
out = {"routed": bool(np.array_equal(fft.fft_lm_to_uv(x), fft.impl("fft", "fft_lm_to_uv", "ref")(x)))}
os.environ[fft.MIN_PIXELS_ENV] = "0"
try:
    fft.fft_lm_to_uv(x)
except RuntimeError as e:
    out["error"] = str(e)
print("RESULT", json.dumps(out))
"""


def test_without_a_device_the_first_gpu_call_raises(device, run_python):
    out = run_python("-c", WITHOUT_DEVICE, CUDA_VISIBLE_DEVICES="")
    result = json.loads(out.rpartition("RESULT ")[2])
    assert result["routed"] is True
    assert result["error"].startswith("no usable CUDA device")


HOOKED = """
import json, numpy as np, xarray as xr
from astroviper.processing_functions.imaging import feather, fft_normalize_anti_alias_gridder as m
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    CASA_GRIDDING_KERNEL_PARAMS, make_gridding_kernel,
)

kernel = make_gridding_kernel(CASA_GRIDDING_KERNEL_PARAMS)
names = {f"{c}.{e}": getattr(c_, e) is getattr(m, e) and hasattr(getattr(m, e), "__wrapped__")
         for c, c_ in (("feather", feather),) for e in ("ifft_uv_to_lm", "fft_lm_to_uv")}
dims = ("time", "frequency", "polarization")
grid = np.random.default_rng(0).standard_normal((1, 1, 2, 40, 40)) + 0j
image = xr.Dataset(
    {"VISIBILITY": (dims + ("u", "v"), grid), "VISIBILITY_NORMALIZATION": (dims, np.full((1, 1, 2), 2.0)),
     "SKY_MODEL": (dims + ("l", "m"), np.ones((1, 1, 2, 32, 32)))},
    coords={"time": [0], "frequency": [1.5e9], "polarization": ["XX", "YY"], "l": np.arange(32), "m": np.arange(32)},
    attrs={"type": "image_dataset", "data_groups": {
        "residual": {"visibility": "VISIBILITY", "visibility_normalization": "VISIBILITY_NORMALIZATION"},
        "model": {"sky": "SKY_MODEL"}}},
)
m.ifft_norm_img_xds(image, {"image_size": [32, 32]}, image_data_group_in_name="residual",
                    image_data_group_out_name="residual", image_data_group_out_modified={"sky": "SKY_RESIDUAL"},
                    gridding_kernel=kernel)
m.fft_norm_img_xds(image, {"image_size": [32, 32], "fft_padding": 1.25}, gridding_kernel=kernel)
print("RESULT", json.dumps(names))
"""


def test_the_modules_callers_and_feather_go_through_the_hook(
    device, run_python, tmp_path
):
    """With fft=gpu the FFT module's image and model transforms call its
    hooked entries, on the GPU, and feather.py holds the hooked entries."""
    trace = tmp_path / "trace"
    out = run_python(
        "-c",
        HOOKED,
        ASTROVIPER_KERNELS="fft=gpu",
        ASTROVIPER_KERNELS_TRACE=str(trace),
        ASTROVIPER_KERNELS_GPU_FFT_MIN_PIXELS="0",
    )
    names = json.loads(out.rpartition("RESULT ")[2])
    assert names == dict.fromkeys(names, True) and len(names) == 2
    (path,) = trace.iterdir()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["entry"], r["path"]) for r in records] == [
        ("ifft_uv_to_lm", "gpu"),
        ("ifft_uv_to_lm", "gpu"),
        ("fft_lm_to_uv", "gpu"),
        ("fft_lm_to_uv", "gpu"),
    ]
    assert {r["shapes"]["0"] == [40, 40] for r in records} == {True}
    module, entries, callers = SELF_HOOKED["fft"]
    assert entries == ENTRIES and callers == (f"{module.rpartition('.')[0]}.feather",)


def test_the_prewarm_compiles_the_kernels_and_loads_cufft(device, monkeypatch):
    monkeypatch.setenv(ENV, "fft=gpu")
    assert prewarm.prewarm()["kernels"] == ["fft"]
    assert ("fft_shift", fft._KERNELS, False) in device._modules
    assert {(64, 64, np.dtype(np.complex128)), (63, 65, np.dtype(np.complex64))} <= set(
        fft._plans
    )


# ---------------------------------------------------------------- the generated code
def _ptx(arch):
    """The PTX of cuda/fft_shift.cu for compute_<arch>, compiled by NVRTC with
    the options CuPy gives it; no device is needed."""
    cp = pytest.importorskip("cupy", reason="no CuPy")
    try:
        cp.cuda.nvrtc.getVersion()
    except Exception as e:  # cuda.pathfinder's DynamicLibNotFoundError
        pytest.skip(f"no NVRTC: {e}")
    nvrtc = cp.cuda.nvrtc
    source = (
        importlib.resources.files("astroviper._kernels.gpu")
        .joinpath("cuda", "fft_shift.cu")
        .read_text(encoding="utf-8")
    )
    program = nvrtc.createProgram(source, "fft_shift.cu", [], [])
    try:
        for name in fft._KERNELS:
            nvrtc.addNameExpression(program, name)
        options = ("-ftz=true", "--device-as-default-execution-space")
        nvrtc.compileProgram(
            program, (*_runtime.OPTIONS, *options, f"-arch=compute_{arch}")
        )
        ptx = nvrtc.getPTX(program)
    finally:
        nvrtc.destroyProgram(program)
    return ptx.decode() if isinstance(ptx, bytes) else ptx


@pytest.mark.parametrize("arch", (75, 86, 100, 120))
def test_the_kernels_compile_without_fused_or_flushed_operations(arch):
    """CuPy compiles with -ftz=true: no float operation of the shifts may
    flush a subnormal, be fused or approximate, and the products of the
    scaling round to nearest."""
    ptx = _ptx(arch)
    ops = re.findall(r"^\s*(?:@!?%p\d+\s+)?([a-z][\w.]*)\s", ptx, re.M)
    floats = [op.split(".") for op in ops if re.search(r"\.f(16|32|64)\b", op)]
    assert ".ftz" not in ptx
    exact = ("fma", "mad", "div", "rcp", "sqrt", "abs", "neg", "min", "max")
    assert not [".".join(op) for op in floats if op[0] in exact]
    products = [op for op in floats if op[0] in ("add", "sub", "mul")]
    assert products and all(op == ["mul", "rn", "f64"] for op in products)
