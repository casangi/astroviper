"""Mode context and small, read-only numerical validation helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagingMode:
    """Identify both independent imaging axes.

    Parameters
    ----------
    imaging_kind : {"cube", "continuum"}
        Spectral imaging workflow.
    field_layout : {"single_field", "multifield"}
        Field layout. Multifield modes are reserved and not yet implemented.
    """

    imaging_kind: str
    field_layout: str

    def __post_init__(self):
        if self.imaging_kind not in ("cube", "continuum"):
            raise ValueError("imaging_kind must be 'cube' or 'continuum'.")
        if self.field_layout not in ("single_field", "multifield"):
            raise ValueError("field_layout must be 'single_field' or 'multifield'.")

    def require_supported(self):
        """Reject execution of the reserved multifield workflows."""
        if self.field_layout == "multifield":
            raise NotImplementedError(
                f"{self.imaging_kind} multifield imaging is not implemented."
            )


CUBE_SINGLE_FIELD = ImagingMode("cube", "single_field")
CONTINUUM_SINGLE_FIELD = ImagingMode("continuum", "single_field")


def require_finite_real(name, value, *, positive=False, scalar=False):
    """Check real numerical values without coercing configuration values.

    Parameters
    ----------
    name : str
        Parameter label included in errors.
    value : array-like
        Nonempty finite real values; unit-bearing quantities are rejected.
    positive : bool, optional
        Require every value to be strictly positive.
    scalar : bool, optional
        Require a zero-dimensional value.

    Returns
    -------
    numpy.ndarray
        Validated NumPy view/conversion of the value, without coercing its dtype
        or changing the caller input.

    Raises
    ------
    TypeError
        The input is non-real, unit-bearing, or nonscalar when a scalar is required.
    ValueError
        The input is empty, non-finite, or fails the requested positivity condition.
    """
    import numpy as np

    if hasattr(value, "unit"):
        raise TypeError(
            f"{name} must use plain numerical SI values, not a unit-bearing quantity."
        )
    values = np.asarray(value)
    if values.dtype.kind not in "iuf" or (scalar and values.ndim != 0):
        raise TypeError(
            f"{name} must contain real numbers" + (" (a scalar)." if scalar else ".")
        )
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be nonempty and finite.")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    return values


def require_mapping(name, value):
    """Require a mapping without copying or filling it.

    Parameters
    ----------
    name : str
        Parameter label included in errors.
    value : object
        Original object, which must implement the mapping interface.

    Returns
    -------
    mapping
        The original validated mapping, without copying or inserting keys.

    Raises
    ------
    TypeError
        The supplied object does not implement the mapping interface.
    """
    from collections.abc import Mapping

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def require_integer(name, value, *, minimum=0, allow_minus_one=False):
    """Require an integer count, optionally allowing the documented -1 sentinel.

    Parameters
    ----------
    name : str
        Parameter label included in errors.
    value : int
        Integral count; Boolean values are rejected.
    minimum : int, optional
        Inclusive lower bound, default zero.
    allow_minus_one : bool, optional
        Also accept the sentinel -1 independently of the lower bound.

    Raises
    ------
    TypeError
        The supplied value is Boolean or not integral.
    ValueError
        The count is below the permitted minimum and is not an allowed sentinel.
    """
    from numbers import Integral

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if allow_minus_one and value == -1:
        return
    if value < minimum:
        suffix = ", or -1" if allow_minus_one else ""
        raise ValueError(f"{name} must be >= {minimum}{suffix}.")


def require_boolean(name, value):
    """Require a Boolean rather than a truthy string or number.

    Parameters
    ----------
    name : str
        Parameter label included in errors.
    value : bool
        Python or NumPy Boolean; numbers and truthy strings are rejected.

    Raises
    ------
    TypeError
        The supplied value is not a Python or NumPy Boolean.
    """
    import numpy as np

    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a bool.")


def imaging_weighting_scope(params, *, mode):
    """Resolve the execution scope without modifying the supplied configuration.

    Natural weighting needs no uv-density grid and always uses the local path,
    independently of a supplied local/global scope. Other methods retain their
    explicit scope or the mode-specific default. Validate supplied values before
    calling this resolver so irrelevant settings cannot conceal malformed input.

    Parameters
    ----------
    params : mapping
        Validated imaging-weight configuration.
    mode : ImagingMode
        Imaging kind that determines density-weighting defaults.

    Returns
    -------
    str
        Effective local/global execution scope, without changing ``params``.
    """
    if params.get("weighting", "natural") == "natural":
        return "local"
    if "weighting_scope" in params:
        return params["weighting_scope"]
    if mode.imaging_kind == "continuum" and params.get("weighting", "natural") in (
        "briggs",
        "uniform",
    ):
        return "global"
    return "local"


def selected_frequency_mask(frequencies, requested_frequencies):
    """Select source channels assigned to requested image frequencies.

    Parameters
    ----------
    frequencies : array-like
        One-dimensional source channel coordinates in Hz.
    requested_frequencies : array-like
        Requested image or execution-block frequencies in Hz.

    Returns
    -------
    numpy.ndarray of bool
        Mask with the source frequency shape identifying selected channels.

    Notes
    -----
    Cube and continuum imaging use the same nearest-channel half cells, allowing
    shifted execution blocks and channel averaging. The supplied coordinates are
    not changed.
    """
    import numpy as np

    frequencies = np.asarray(frequencies)
    requested = np.asarray(requested_frequencies)
    if not frequencies.size or not requested.size:
        return np.zeros(frequencies.shape, dtype=bool)

    from graphviper.graph_tools.coordinate_utils import map_to_nearest_coordinate

    return map_to_nearest_coordinate(frequencies, requested) >= 0


def check_continuum_mode(value):
    """Require one of the implemented continuum image representations.

    Parameters
    ----------
    value : {'mfs', 'mvc'}
        Requested continuum representation.
    """
    if value not in ("mfs", "mvc"):
        raise ValueError("continuum_mode must be 'mfs' or 'mvc'.")


def check_fft_backend(backend):
    """Require a supported and locally importable FFT implementation.

    Parameters
    ----------
    backend : {'scipy', 'pyfftw'}
        FFT implementation to validate and import.

    Returns
    -------
    module
        Imported SciPy-compatible FFT implementation.

    Raises
    ------
    ValueError
        The backend name is unsupported.
    ImportError
        The selected FFT implementation is not installed.
    """
    from importlib import import_module

    if backend not in ("scipy", "pyfftw"):
        raise ValueError(
            f"Unknown FFT backend {backend!r}; fft_backend must be 'scipy' or 'pyfftw'."
        )
    return import_module(
        "scipy.fft" if backend == "scipy" else "pyfftw.interfaces.scipy_fft"
    )
