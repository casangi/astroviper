"""Antenna beam models shipped with AstroVIPER and their readers.

Three kinds of **antenna (voltage) beam models** are understood by the simulator
(``astroviper.processing_functions.simulation.antenna_beams``):

1. **Analytic** — a dict ``{"func": "casa_airy" | "airy" | "none",
   "dish_diameter": m, "blockage_diameter": m, "max_rad_1GHz": rad}``.
   ``max_rad_1GHz`` is the radius (at 1 GHz) beyond which the beam is zero; the
   values in :data:`AIRY_DISK_MODELS` are CASA's ``PBMath`` defaults.
2. **Aperture (Zernike) polynomial coefficients** — an ``xr.Dataset`` with
   ``ZPC[frequency, polarization, coefficient]`` (complex) and
   ``ETA[frequency, polarization, coefficient]`` and attrs ``dish_diameter``,
   ``max_rad_1GHz``, ``telescope_name``.  Converted to Jones beam images by the
   simulator (Sekhar et al. 2022 models for the EVLA and MeerKAT ship as CSV).
3. **Beam polynomial coefficients** — an ``xr.Dataset`` with
   ``BPC[frequency, polarization, coefficient]``: CASA ``PBMath1DPoly``-style
   1-D power-beam polynomials in ``r`` (arcmin at 1 GHz), e.g. the EVLA models.

The legacy SIRIUS coordinate names (``chan``, ``pol``, ``coef_indx``,
``dish_diam``) are translated by :func:`normalize_beam_model_xds`.
"""

from __future__ import annotations

import importlib.resources
import os
from datetime import date

import numpy as np
import xarray as xr

DEG_TO_RAD = np.pi / 180.0

# CASA PBMath.cc max radii (degrees at 1 GHz) converted to radians.
AIRY_DISK_MODELS: dict[str, dict] = {
    "vla": {
        "func": "casa_airy",
        "dish_diameter": 24.5,
        "blockage_diameter": 0.0,
        "max_rad_1GHz": 0.8564 * DEG_TO_RAD,
    },
    "aca": {
        "func": "casa_airy",
        "dish_diameter": 6.25,
        "blockage_diameter": 0.75,
        "max_rad_1GHz": 3.568 * DEG_TO_RAD,
    },
    "alma": {
        "func": "casa_airy",
        "dish_diameter": 10.7,
        "blockage_diameter": 0.75,
        "max_rad_1GHz": 1.784 * DEG_TO_RAD,
    },
    "ngvla": {
        "func": "casa_airy",
        "dish_diameter": 18.0,
        "blockage_diameter": 0.0,
        "max_rad_1GHz": 1.5 * DEG_TO_RAD,
    },
}

_MAX_RAD_1GHZ_BY_TELESCOPE = {
    "evla": 0.8564 * DEG_TO_RAD,
    "vla": 0.8564 * DEG_TO_RAD,
    "ngvla": 1.5 * DEG_TO_RAD,
    "alma": 1.784 * DEG_TO_RAD,
    "aca": 3.568 * DEG_TO_RAD,
    "meerkat": 1.5 * DEG_TO_RAD,
}

_LEGACY_ATTR_NAMES = {
    "dish_diam": "dish_diameter",
    "blockage_diam": "blockage_diameter",
}
_LEGACY_COORD_NAMES = {
    "chan": "frequency",
    "pol": "polarization",
    "coef_indx": "coefficient",
    "pa": "parallactic_angle",
}
_POL_CODE_TO_NAME = {
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
}


def beam_model_directory(kind: str) -> str:
    """Directory of shipped model files.

    Parameters
    ----------
    kind : {"aperture_polynomial_coefficient_models", "beam_polynomial_coefficient_models"}
    """
    return str(importlib.resources.files("astroviper") / "data" / "simulation" / kind)


def list_aperture_polynomial_coefficient_models() -> list[str]:
    """Names of the shipped Zernike aperture-coefficient models (CSV basenames)."""
    d = beam_model_directory("aperture_polynomial_coefficient_models")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".csv"))


def list_beam_polynomial_coefficient_models() -> list[str]:
    """Names of the shipped beam-polynomial models (TXT basenames)."""
    d = beam_model_directory("beam_polynomial_coefficient_models")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".txt"))


def airy_disk_model(name: str, func: str | None = None) -> dict:
    """Copy of one of :data:`AIRY_DISK_MODELS` (``"vla"``, ``"aca"``, ``"alma"``, ``"ngvla"``).

    Parameters
    ----------
    name : str
    func : str, optional
        Override the beam function (``"casa_airy"``, ``"airy"`` or ``"none"``).
    """
    model = dict(AIRY_DISK_MODELS[name.lower()])
    if func is not None:
        model["func"] = func
    return model


def normalize_beam_model_dict(model: dict) -> dict:
    """Return an analytic beam-model dict with the canonical key names."""
    out = {}
    for key, value in model.items():
        out[_LEGACY_ATTR_NAMES.get(key, key)] = value
    out.setdefault("func", "casa_airy")
    out.setdefault("blockage_diameter", 0.0)
    for required in ("dish_diameter", "max_rad_1GHz"):
        if required not in out:
            raise ValueError(f"Analytic beam model is missing '{required}': {model}")
    return out


def normalize_beam_model_xds(xds: xr.Dataset) -> xr.Dataset:
    """Translate legacy SIRIUS coordinate/attribute names of a beam-model dataset.

    ``chan -> frequency``, ``pol -> polarization`` (casacore codes -> MSv4 labels),
    ``coef_indx -> coefficient``, ``pa -> parallactic_angle``, ``J -> JONES`` and
    ``dish_diam -> dish_diameter``.  Datasets already using the new names are
    returned unchanged (a shallow copy).
    """
    renames = {
        k: v for k, v in _LEGACY_COORD_NAMES.items() if k in xds.dims or k in xds.coords
    }
    out = xds.rename(renames) if renames else xds.copy()
    if "J" in out.data_vars:
        out = out.rename({"J": "JONES"})
    if "polarization" in out.coords and np.issubdtype(
        out.polarization.dtype, np.integer
    ):
        out = out.assign_coords(
            polarization=[_POL_CODE_TO_NAME[int(p)] for p in out.polarization.values]
        )
    attrs = dict(out.attrs)
    for old, new in _LEGACY_ATTR_NAMES.items():
        if old in attrs:
            attrs[new] = attrs.pop(old)
    out.attrs = attrs
    return out


def _max_rad_for_telescope(telescope_name: str) -> float:
    try:
        return _MAX_RAD_1GHZ_BY_TELESCOPE[telescope_name.lower()]
    except KeyError:
        raise ValueError(
            f"No default max_rad_1GHz for telescope '{telescope_name}'; pass max_rad_1GHz explicitly."
        ) from None


def read_aperture_polynomial_coefficients(
    model: str,
    dish_diameter: float | None = None,
    max_rad_1GHz: float | None = None,
    telescope_name: str | None = None,
    frequency_unit_to_hertz: float = 1e6,
) -> xr.Dataset:
    """Read a Zernike aperture-coefficient CSV (``#stokes,freq,ind,real,imag[,eta]``).

    Parameters
    ----------
    model : str
        Shipped model name (see :func:`list_aperture_polynomial_coefficient_models`,
        e.g. ``"EVLA_avg_zcoeffs_SBand_lookup"``) or a path to a CSV file.
    dish_diameter : float, optional
        Metres; defaults to 25 m for EVLA, 13.5 m for MeerKAT (taken from the file name).
    max_rad_1GHz : float, optional
        Radians; default from the telescope name (CASA PBMath values).
    telescope_name : str, optional
        Defaults to the prefix of the file name (``EVLA``, ``MeerKAT``).
    frequency_unit_to_hertz : float
        Scale of the ``freq`` column (MHz in the shipped files).

    Returns
    -------
    xr.Dataset
        ``ZPC[frequency, polarization, coefficient]`` (complex128) and ``ETA``
        (float64) with attrs ``dish_diameter``, ``max_rad_1GHz``, ``telescope_name``,
        ``model_file``, ``conversion_date``.
    """
    path = model
    if not os.path.isfile(path):
        path = os.path.join(
            beam_model_directory("aperture_polynomial_coefficient_models"),
            model + ".csv",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Unknown aperture coefficient model {model!r}.")
    base = os.path.basename(path)
    if telescope_name is None:
        telescope_name = base.partition("_")[0]
    if dish_diameter is None:
        dish_diameter = 13.5 if telescope_name.lower() == "meerkat" else 25.0
    if max_rad_1GHz is None:
        max_rad_1GHz = _max_rad_for_telescope(telescope_name)

    rows = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    pol_codes = np.unique(rows[:, 0]).astype(int)
    frequency = np.unique(rows[:, 1])
    n_coef = int(rows[:, 2].max()) + 1
    zpc = np.zeros((len(frequency), len(pol_codes), n_coef), dtype=np.complex128)
    eta = np.ones((len(frequency), len(pol_codes), n_coef), dtype=np.float64)
    i_pol = np.searchsorted(pol_codes, rows[:, 0].astype(int))
    i_freq = np.searchsorted(frequency, rows[:, 1])
    i_coef = rows[:, 2].astype(int)
    zpc[i_freq, i_pol, i_coef] = rows[:, 3] + 1j * rows[:, 4]
    if rows.shape[1] > 5:
        eta[i_freq, i_pol, i_coef] = rows[:, 5]

    xds = xr.Dataset(
        {
            "ZPC": (("frequency", "polarization", "coefficient"), zpc),
            "ETA": (("frequency", "polarization", "coefficient"), eta),
        },
        coords={
            "frequency": frequency * frequency_unit_to_hertz,
            "polarization": [_POL_CODE_TO_NAME[int(p)] for p in pol_codes],
            "coefficient": np.arange(n_coef),
        },
        attrs={
            "model_type": "aperture_polynomial_coefficients",
            "model_file": base,
            "telescope_name": telescope_name,
            "conversion_date": str(date.today()),
            "dish_diameter": float(dish_diameter),
            "max_rad_1GHz": float(max_rad_1GHz),
        },
    )
    xds.frequency.attrs.update({"units": "Hz", "type": "spectral_coord"})
    return xds


def read_beam_polynomial_coefficients(
    model: str,
    dish_diameter: float = 25.0,
    max_rad_1GHz: float | None = None,
    telescope_name: str | None = None,
    max_coefficients: int = 5,
    polarization: str = "RR",
) -> xr.Dataset:
    """Read a CASA ``PBMath1DPoly`` style coefficient text file (e.g. the EVLA bands).

    The file lists ``# Band definitions`` (``name start_Hz end_Hz``) followed by
    ``# Coefficients`` blocks ``coeffmap_p[freq_MHz]={c0, c1, ...};`` grouped per band.
    The power beam is ``sum_i c_i r^(2 i)`` with ``r`` in arcmin scaled to 1 GHz.

    Parameters
    ----------
    model : str
        Shipped model name (see :func:`list_beam_polynomial_coefficient_models`,
        e.g. ``"EVLA_"``) or a path.
    dish_diameter : float
    max_rad_1GHz : float, optional
    telescope_name : str, optional
    max_coefficients : int
    polarization : str
        Label of the single (unpolarised) polarization axis entry.

    Returns
    -------
    xr.Dataset
        ``BPC[frequency, polarization, coefficient]`` (float64) with a ``band``
        coordinate along ``frequency`` and the same attrs as
        :func:`read_aperture_polynomial_coefficients`.
    """
    path = model
    if not os.path.isfile(path):
        path = os.path.join(
            beam_model_directory("beam_polynomial_coefficient_models"), model + ".txt"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Unknown beam polynomial model {model!r}.")
    base = os.path.basename(path)
    if telescope_name is None:
        telescope_name = base.partition("_")[0]
    if max_rad_1GHz is None:
        max_rad_1GHz = _max_rad_for_telescope(telescope_name)

    band_names: list[str] = []
    frequency, coefficients, bands = [], [], []
    mode = None
    band = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# Band definitions"):
                mode = "band"
                continue
            if line.startswith("# Coefficients"):
                mode = "coef"
                continue
            if line.startswith("#"):
                continue
            if mode == "band":
                parts = line.split()
                band_names.append(parts[0])
            elif mode == "coef":
                if line[0] in band_names and "{" not in line:
                    band = line[0]
                    continue
                freq_mhz = float(line[line.index("[") + 1 : line.index("]")])
                values = [
                    float(v)
                    for v in line[line.index("{") + 1 : line.index("}")].split(",")
                ]
                if len(values) > max_coefficients:
                    raise ValueError(
                        f"{path}: more than {max_coefficients} coefficients on line {line!r}"
                    )
                coef = np.zeros(max_coefficients)
                coef[: len(values)] = values
                frequency.append(freq_mhz * 1e6)
                coefficients.append(coef)
                bands.append(band)
    frequency = np.array(frequency)
    order = np.argsort(frequency, kind="stable")
    bpc = np.array(coefficients)[order][:, None, :]
    xds = xr.Dataset(
        {"BPC": (("frequency", "polarization", "coefficient"), bpc)},
        coords={
            "frequency": frequency[order],
            "polarization": [polarization],
            "coefficient": np.arange(max_coefficients),
            "band": ("frequency", np.array(bands, dtype=str)[order]),
        },
        attrs={
            "model_type": "beam_polynomial_coefficients",
            "model_file": base,
            "telescope_name": telescope_name,
            "conversion_date": str(date.today()),
            "dish_diameter": float(dish_diameter),
            "max_rad_1GHz": float(max_rad_1GHz),
        },
    )
    xds.frequency.attrs.update({"units": "Hz", "type": "spectral_coord"})
    return xds
