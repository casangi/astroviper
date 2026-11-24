"""The FFT module rule of AGENTS.md, checked on the source without importing it.

Outside processing_functions/fft.py and parked/, no module imports numpy.fft,
scipy.fft, scipy.fftpack or pyFFTW with an import statement, takes the ``fft`` or
``fftpack`` attribute of a name bound to numpy or scipy, or uses one of the
scipy.signal and astropy functions listed here, which transform inside (also as
attributes of a bare ``scipy`` or ``astropy``); no module passes a string naming one
of the libraries to ``import_module`` or ``__import__``, except ``check_fft_backend``,
which imports the library to check that it is installed; the module's transforms are
called with ``fft_backend=``; and the module is imported as
``from astroviper.processing_functions import fft``, so that its functions are looked
up at call time.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "astroviper"
FFT_MODULE = "processing_functions/fft.py"
FFT_LIBRARIES = ("numpy.fft", "scipy.fft", "scipy.fftpack", "pyfftw")
#: Functions of scipy.signal and astropy.convolution that transform inside.
SIGNAL_TRANSFORMING = {
    "fftconvolve",
    "oaconvolve",
    "convolve",
    "correlate",
    "choose_conv_method",
    "hilbert",
    "hilbert2",
    "resample",
    "czt",
    "zoom_fft",
    "CZT",
    "ZoomFFT",
    "ShortTimeFFT",
    "periodogram",
    "welch",
    "csd",
    "coherence",
    "spectrogram",
    "stft",
    "istft",
    "freqz",
    "firwin2",
    "minimum_phase",
    "envelope",
    "wiener",
}
ASTROPY_TRANSFORMING = {"convolve_fft"}
#: The module's functions that take ``fft_backend``.
MODULE_TRANSFORMS = {"fft2", "ifft2", "rfft2", "irfft2", "ifft", "irfft", "set_workers"}
#: The one function that may import a library by name: (file, function).
DYNAMIC_IMPORT = (
    "processing_functions/imaging/config_check/_common.py",
    "check_fft_backend",
)


def _names_library(name):
    return any(
        name == library or name.startswith(library + ".") for library in FFT_LIBRARIES
    )


def _dotted(node):
    """``a`` or ``a.b`` of a Name or an Attribute of a Name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _findings(path, file):
    """(file, line, kind, text) of every use of an FFT library in one file."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    function_of = {}
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            for node in ast.walk(function):
                function_of[id(node)] = function.name
    found = []
    numpy_scipy, signal, convolution, module = set(), set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _names_library(alias.name):
                    found.append((node.lineno, "import", f"import {alias.name}"))
                top = alias.name.split(".")[0]
                binds_top = alias.asname is None or alias.name == top
                if top in ("numpy", "scipy") and binds_top:
                    numpy_scipy.add(alias.asname or top)
                if top == "scipy" and binds_top:
                    signal.add(f"{alias.asname or top}.signal")
                if top == "astropy" and binds_top:
                    convolution.add(f"{alias.asname or top}.convolution")
                if alias.name == "scipy.signal":
                    signal.add(alias.asname or "scipy.signal")
                if alias.name == "astropy.convolution":
                    convolution.add(alias.asname or "astropy.convolution")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for alias in node.names:
                if _names_library(base) or _names_library(f"{base}.{alias.name}"):
                    found.append(
                        (node.lineno, "import", f"from {base} import {alias.name}")
                    )
                if base == "scipy" and alias.name == "signal":
                    signal.add(alias.asname or "signal")
                if base == "astropy" and alias.name == "convolution":
                    convolution.add(alias.asname or "convolution")
                if base == "scipy.signal" and alias.name in SIGNAL_TRANSFORMING:
                    found.append(
                        (node.lineno, "signal", f"from {base} import {alias.name}")
                    )
                if base == "astropy.convolution" and alias.name in ASTROPY_TRANSFORMING:
                    found.append(
                        (node.lineno, "signal", f"from {base} import {alias.name}")
                    )
                if base == "astroviper.processing_functions" and alias.name == "fft":
                    module.add(alias.asname or "fft")
                if base == "astroviper.processing_functions.fft":
                    found.append(
                        (node.lineno, "from fft", f"from {base} import {alias.name}")
                    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = _dotted(node.value)
            if (
                isinstance(node.value, ast.Name)
                and node.attr in ("fft", "fftpack")
                and base in numpy_scipy
            ):
                found.append((node.lineno, "attribute", f"{base}.{node.attr}"))
            if base in signal and node.attr in SIGNAL_TRANSFORMING:
                found.append((node.lineno, "signal", f"{base}.{node.attr}"))
            if base in convolution and node.attr in ASTROPY_TRANSFORMING:
                found.append((node.lineno, "signal", f"{base}.{node.attr}"))
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else getattr(function, "id", None)
        )
        if name in ("import_module", "__import__"):
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            named = sorted(
                {
                    constant.value
                    for argument in arguments
                    for constant in ast.walk(argument)
                    if isinstance(constant, ast.Constant)
                    and isinstance(constant.value, str)
                    and _names_library(constant.value)
                }
            )
            if named:
                kind = (
                    "allowed dynamic import"
                    if (file, function_of.get(id(node))) == DYNAMIC_IMPORT
                    else "dynamic import"
                )
                found.append((node.lineno, kind, f"{name}({', '.join(named)})"))
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in module
            and function.attr in MODULE_TRANSFORMS
            and not any(keyword.arg == "fft_backend" for keyword in node.keywords)
        ):
            found.append((node.lineno, "no fft_backend", f"fft.{function.attr}"))
    return [(file, line, kind, text) for line, kind, text in sorted(set(found))]


def _scan(source=None):
    source = SOURCE if source is None else source
    found = []
    for path in sorted(source.rglob("*.py")):
        file = path.relative_to(source).as_posix()
        if file == FFT_MODULE or file.startswith("parked/"):
            continue
        found.extend(_findings(path, file))
    return found


def test_only_the_fft_module_uses_an_fft_library():
    violations = [entry for entry in _scan() if entry[2] != "allowed dynamic import"]
    assert violations == []


def test_check_fft_backend_is_the_one_dynamic_import():
    """The exception is exact: the test fails when the check no longer imports."""
    allowed = [entry for entry in _scan() if entry[2] == "allowed dynamic import"]
    assert [entry[0] for entry in allowed] == [DYNAMIC_IMPORT[0]]
