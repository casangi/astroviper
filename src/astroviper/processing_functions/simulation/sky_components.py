"""Sky components of the simulator: one registry and one interface for every source kind.

A **sky component** is a dictionary::

    {"kind": "gaussian", "flux": 0.5, "ra_dec": [ra, dec], "major": 2e-5, "minor": 1e-5, "pa": 0.3}

with the keys

``kind``
    One of :data:`COMPONENT_KINDS` (``"point"``, ``"gaussian"``, ``"disk"``,
    ``"gaussian_ring"``, ``"m_ring"``, ``"crescent"``, ``"annulus"``,
    ``"exponential_disk"``, ``"tapered_power_law"``, ``"shapelet"``).
``flux``
    Integrated flux in Jy: a scalar (Stokes I, put into both parallel-hand
    correlations), ``[4]`` (the four instrumental correlations ``RR, RL, LR,
    LL`` or ``XX, XY, YX, YY``), ``[n_frequency | 1, 4]`` (a spectrum) or
    ``[n_time | 1, n_frequency | 1, 4]`` (time and frequency dependent).
``ra_dec``
    Right ascension and declination in radians, ``[2]`` or ``[n_time | 1, 2]``
    (a moving source), or an :class:`astropy.coordinates.SkyCoord`.
shape parameters
    The kind's own parameters (table below); angles in radians or as astropy
    angle :class:`~astropy.units.Quantity` objects.  Missing optional
    parameters take their defaults.
``name``
    An optional label (kept, unused).

===================  ==========================================================
kind                 shape parameters (required, *optional = default*)
===================  ==========================================================
point                (none)
gaussian             major, minor, *pa = 0* (FWHM diameters)
disk                 major, minor, *pa = 0*, *limb_darkening = 0*, *fwhm = 0*
gaussian_ring        radius, fwhm, *inclination = 0*, *pa = 0*
m_ring               radius, beta, *fwhm = 0*, *inclination = 0*, *pa = 0*
crescent             radius, inner_radius, offset, *pa = 0*, *floor = 0*, *fwhm = 0*
annulus              radius, inner_radius, *inclination = 0*, *pa = 0*, *fwhm = 0*
exponential_disk     scale_radius, *inclination = 0*, *pa = 0*
tapered_power_law    cutoff_radius, index, *inclination = 0*, *pa = 0*
shapelet             scale, coefficients, *pa = 0*
===================  ==========================================================

Every kind has a unit-flux analytic visibility ``T(u, v, w)`` in the frame of
the component **including the w term** (:func:`component_uv_response`) and an
image-plane twin (:func:`component_image`, per steradian, unit flux), and
:func:`sky_model_image` rasterises a whole component list in Jy/pixel.  All
position angles are measured from ``+m`` (north) towards ``+l`` (east), the
clean-beam convention; inclinations are ``0`` face-on with the minor axis
``cos(inclination)`` times the major axis.

The ``<kind>_source_flux / _ra_dec / _shape`` array parameters of the
simulator (point, Gaussian, disk and Gaussian-ring sources) remain supported
as a bulk interface and are converted with :func:`sky_components_from_arrays`;
:func:`normalize_sky_components` validates either form once, in the
distributed application, and the node tasks and processing functions consume
the normalised list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

REQUIRED = object()


@dataclass(frozen=True)
class ComponentKind:
    """Description of one component kind: its parameters and analytic functions."""

    name: str
    label: str
    parameters: dict[str, Any]  # parameter -> default (REQUIRED for mandatory ones)
    angle_parameters: frozenset
    uv_response: str | None  # "module:function" (imported lazily)
    image: str | None

    def required(self):
        return [p for p, d in self.parameters.items() if d is REQUIRED]


_SIM = "astroviper.processing_functions.simulation."

COMPONENT_KINDS: dict[str, ComponentKind] = {
    kind.name: kind
    for kind in [
        ComponentKind("point", "point", {}, frozenset(), None, None),
        ComponentKind(
            "gaussian",
            "Gaussian",
            {"major": REQUIRED, "minor": REQUIRED, "pa": 0.0},
            frozenset({"major", "minor", "pa"}),
            _SIM + "gaussian_source:elliptical_gaussian_uv_response",
            _SIM + "gaussian_source:elliptical_gaussian_image",
        ),
        ComponentKind(
            "disk",
            "limb-darkened disk",
            {"major": REQUIRED, "minor": REQUIRED, "pa": 0.0, "limb_darkening": 0.0, "fwhm": 0.0},
            frozenset({"major", "minor", "pa", "fwhm"}),
            _SIM + "limb_darkened_disk:limb_darkened_disk_uv_response",
            _SIM + "limb_darkened_disk:limb_darkened_disk_image",
        ),
        ComponentKind(
            "gaussian_ring",
            "Gaussian ring",
            {"radius": REQUIRED, "fwhm": REQUIRED, "inclination": 0.0, "pa": 0.0},
            frozenset({"radius", "fwhm", "inclination", "pa"}),
            _SIM + "gaussian_ring:gaussian_ring_uv_response",
            _SIM + "gaussian_ring:gaussian_ring_image",
        ),
        ComponentKind(
            "m_ring",
            "m-ring",
            {"radius": REQUIRED, "beta": REQUIRED, "fwhm": 0.0, "inclination": 0.0, "pa": 0.0},
            frozenset({"radius", "fwhm", "inclination", "pa"}),
            _SIM + "m_ring:m_ring_uv_response",
            _SIM + "m_ring:m_ring_image",
        ),
        ComponentKind(
            "crescent",
            "crescent",
            {"radius": REQUIRED, "inner_radius": REQUIRED, "offset": REQUIRED, "pa": 0.0, "floor": 0.0, "fwhm": 0.0},
            frozenset({"radius", "inner_radius", "offset", "pa", "fwhm"}),
            _SIM + "crescent:crescent_uv_response",
            _SIM + "crescent:crescent_image",
        ),
        ComponentKind(
            "annulus",
            "annulus",
            {"radius": REQUIRED, "inner_radius": REQUIRED, "inclination": 0.0, "pa": 0.0, "fwhm": 0.0},
            frozenset({"radius", "inner_radius", "inclination", "pa", "fwhm"}),
            _SIM + "annulus:annulus_uv_response",
            _SIM + "annulus:annulus_image",
        ),
        ComponentKind(
            "exponential_disk",
            "exponential disk",
            {"scale_radius": REQUIRED, "inclination": 0.0, "pa": 0.0},
            frozenset({"scale_radius", "inclination", "pa"}),
            _SIM + "exponential_disk:exponential_disk_uv_response",
            _SIM + "exponential_disk:exponential_disk_image",
        ),
        ComponentKind(
            "tapered_power_law",
            "tapered power-law",
            {"cutoff_radius": REQUIRED, "index": REQUIRED, "inclination": 0.0, "pa": 0.0},
            frozenset({"cutoff_radius", "inclination", "pa"}),
            _SIM + "tapered_power_law:tapered_power_law_uv_response",
            _SIM + "tapered_power_law:tapered_power_law_image",
        ),
        ComponentKind(
            "shapelet",
            "shapelet",
            {"scale": REQUIRED, "coefficients": REQUIRED, "pa": 0.0},
            frozenset({"scale", "pa"}),
            _SIM + "shapelet:shapelet_uv_response",
            _SIM + "shapelet:shapelet_image",
        ),
    ]
}  # fmt: skip

# columns of the legacy ``<kind>_source_shape`` arrays
_LEGACY_SHAPE_COLUMNS = {
    "gaussian": ("major", "minor", "pa"),
    "disk": ("major", "minor", "pa"),
    "gaussian_ring": ("radius", "fwhm", "inclination", "pa"),
}


def _resolve(path: str):
    module_name, function_name = path.split(":")
    import importlib

    return getattr(importlib.import_module(module_name), function_name)


def _as_angle(value, name, kind):
    """Angle in radians from a float or an astropy angle Quantity (scalar or array)."""
    if hasattr(value, "unit"):
        import astropy.units as units

        try:
            value = value.to_value(units.rad)
        except units.UnitConversionError as error:
            raise ValueError(
                f"{kind} {name}: expected an angle, got {value.unit}."
            ) from error
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{kind} {name} must be finite.")
    return float(array) if array.ndim == 0 else array


def _normalize_flux(flux, kind, n_time, n_frequency):
    flux = np.asarray(flux, dtype=np.float64)
    if flux.ndim == 0:
        flux = np.array([flux, 0.0, 0.0, flux])
    if flux.ndim == 1:
        flux = flux[None, None, :]
    elif flux.ndim == 2:
        flux = flux[None, :, :]
    if flux.ndim != 3 or flux.shape[2] != 4:
        raise ValueError(
            f"{kind} flux must be a scalar, [4], [n_frequency|1, 4] or [n_time|1, n_frequency|1, 4]; "
            f"got shape {np.shape(flux)}."
        )
    if n_time is not None and flux.shape[0] not in (1, n_time):
        raise ValueError(
            f"{kind} flux time axis must be 1 or n_time={n_time}; got {flux.shape[0]}."
        )
    if n_frequency is not None and flux.shape[1] not in (1, n_frequency):
        raise ValueError(
            f"{kind} flux frequency axis must be 1 or n_frequency={n_frequency}; got {flux.shape[1]}."
        )
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{kind} flux must be finite.")
    return flux


def _normalize_ra_dec(ra_dec, kind, n_time, direction_frame):
    if hasattr(ra_dec, "transform_to"):  # astropy SkyCoord
        import astropy.units as units

        coord = ra_dec.transform_to(direction_frame)
        ra_dec = np.stack(
            [
                np.atleast_1d(coord.ra.to_value(units.rad)),
                np.atleast_1d(coord.dec.to_value(units.rad)),
            ],
            axis=-1,
        )
    ra_dec = np.asarray(ra_dec, dtype=np.float64)
    if ra_dec.ndim == 1:
        ra_dec = ra_dec[None, :]
    if ra_dec.ndim != 2 or ra_dec.shape[1] != 2:
        raise ValueError(
            f"{kind} ra_dec must be [2] or [n_time|1, 2] radians; got shape {ra_dec.shape}."
        )
    if n_time is not None and ra_dec.shape[0] not in (1, n_time):
        raise ValueError(
            f"{kind} ra_dec time axis must be 1 or n_time={n_time}; got {ra_dec.shape[0]}."
        )
    if not np.all(np.isfinite(ra_dec)):
        raise ValueError(f"{kind} ra_dec must be finite.")
    return ra_dec


def normalize_sky_component(
    component, n_time=None, n_frequency=None, direction_frame="icrs"
):
    """Validate one component dictionary and return it in canonical form.

    Parameters
    ----------
    component : dict
        ``{"kind", "flux", "ra_dec", <shape parameters>, ["name"]}`` (module
        docstring).
    n_time, n_frequency : int, optional
        Simulated axis lengths the flux / position arrays must be compatible with.
    direction_frame : str
        Frame of ``ra_dec`` (``SkyCoord`` inputs are transformed to it).

    Returns
    -------
    dict
        ``kind`` (str), ``flux`` (float64 ``[n_time|1, n_frequency|1, 4]``),
        ``ra_dec`` (float64 ``[n_time|1, 2]``), every shape parameter of the
        kind as a float / array, and ``name``.
    """
    if not isinstance(component, dict):
        raise TypeError(
            f"a sky component must be a dict; got {type(component).__name__}."
        )
    try:
        kind_name = component["kind"]
    except KeyError as error:
        raise ValueError("a sky component needs a 'kind'.") from error
    if kind_name not in COMPONENT_KINDS:
        raise ValueError(
            f"unknown sky component kind {kind_name!r}; expected one of {sorted(COMPONENT_KINDS)}."
        )
    kind = COMPONENT_KINDS[kind_name]
    allowed = {"kind", "flux", "ra_dec", "name", *kind.parameters}
    unknown = set(component) - allowed
    if unknown:
        raise ValueError(
            f"{kind_name} component has unknown parameter(s) {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}."
        )
    missing = [p for p in ("flux", "ra_dec", *kind.required()) if p not in component]
    if missing:
        raise ValueError(f"{kind_name} component is missing {missing}.")

    out = {"kind": kind_name, "name": component.get("name")}
    out["flux"] = _normalize_flux(component["flux"], kind_name, n_time, n_frequency)
    out["ra_dec"] = _normalize_ra_dec(
        component["ra_dec"], kind_name, n_time, direction_frame
    )
    for parameter, default in kind.parameters.items():
        value = component.get(parameter, default)
        if parameter in kind.angle_parameters:
            value = _as_angle(value, parameter, kind_name)
            if parameter == "scale":
                value = np.atleast_1d(value)
            elif np.ndim(value) != 0:
                raise ValueError(f"{kind_name} {parameter} must be a scalar angle.")
        elif parameter == "beta":
            value = np.atleast_1d(np.asarray(value, dtype=np.complex128)).ravel()
        elif parameter == "coefficients":
            value = np.atleast_2d(np.asarray(value, dtype=np.float64))
        else:
            value = float(value)
        out[parameter] = value
    if (
        kind.uv_response is not None
    ):  # let the analytic function validate its own ranges
        _resolve(kind.uv_response)(np.zeros(1), np.zeros(1), **_shape_parameters(out))
    return out


def normalize_sky_components(
    components, n_time=None, n_frequency=None, direction_frame="icrs"
):
    """Validate a list of component dictionaries (see :func:`normalize_sky_component`)."""
    if components is None:
        return []
    if isinstance(components, dict):
        components = [components]
    return [
        normalize_sky_component(component, n_time, n_frequency, direction_frame)
        for component in components
    ]


def _shape_parameters(component):
    kind = COMPONENT_KINDS[component["kind"]]
    return {p: component[p] for p in kind.parameters}


def component_uv_response(component, u, v, w=0.0):
    """Unit-flux analytic visibility ``T(u, v, w)`` of a (normalised) component in its own frame.

    ``u, v, w`` in wavelengths, rotated into the frame of the component
    (:func:`~astroviper.processing_functions.simulation.calculate_visibilities.source_frame_uvw`).
    A point source returns ones.
    """
    kind = COMPONENT_KINDS[component["kind"]]
    if kind.uv_response is None:
        return np.ones(np.broadcast(u, v, w).shape)
    return _resolve(kind.uv_response)(u, v, w=w, **_shape_parameters(component))


def component_image(component, l, m, pixel_area=None):  # noqa: E741
    """Unit-flux surface brightness (per steradian) of a (normalised) extended component.

    ``l, m`` are sky offsets (radians) from the component centre.  ``pixel_area``
    (steradian) is used by profiles that diverge at the centre (the tapered
    power law) to give the central pixel its pixel-averaged brightness.
    Point sources have no surface brightness (``ValueError``).
    """
    kind = COMPONENT_KINDS[component["kind"]]
    if kind.image is None:
        raise ValueError(
            "a point source has no surface brightness; see sky_model_image."
        )
    parameters = _shape_parameters(component)
    if component["kind"] == "tapered_power_law":
        parameters["pixel_area"] = pixel_area
    return _resolve(kind.image)(l, m, **parameters)


def component_flux(component, time_index=0, frequency_index=0, correlation=None):
    """Flux of a normalised component at one time / channel: Stokes I (``(c0 + c3) / 2``) or one correlation."""
    flux = component["flux"]
    row = flux[
        time_index if flux.shape[0] > 1 else 0,
        frequency_index if flux.shape[1] > 1 else 0,
    ]
    if correlation is None:
        return 0.5 * (row[0] + row[3])
    return row[int(correlation)]


def sky_model_image(
    components,
    l_axis,
    m_axis,
    phase_center_ra_dec,
    time_index=0,
    frequency_index=0,
    correlation=None,
):
    """Rasterise a component list on a SIN-projected grid, in Jy/pixel.

    Parameters
    ----------
    components : list of dict
        Sky components (normalised or not; see :func:`normalize_sky_components`).
    l_axis, m_axis : numpy.ndarray, [n_l], [n_m], radians
        Regular direction-cosine axes of the image (the ``l`` and ``m``
        coordinates of an AstroVIPER image, ``l`` increasing to the east).
    phase_center_ra_dec : array_like, [2], radians
        Direction of the tangent point of the grid.
    time_index, frequency_index : int
        Which time / channel of time- or frequency-dependent fluxes and positions.
    correlation : int, optional
        Instrumental correlation (0..3) to rasterise; ``None`` gives Stokes I.

    Returns
    -------
    numpy.ndarray, [n_l, n_m]
        Point sources are added to the nearest pixel; extended components are
        their unit-flux surface brightness times flux and pixel area.  No
        primary beam is applied.
    """
    from astroviper.utils.coordinate_transforms import sin_project

    l_axis = np.asarray(l_axis, dtype=np.float64)
    m_axis = np.asarray(m_axis, dtype=np.float64)
    phase_center = np.asarray(phase_center_ra_dec, dtype=np.float64).reshape(2)
    dl = l_axis[1] - l_axis[0]
    dm = m_axis[1] - m_axis[0]
    pixel_area = abs(dl * dm)
    l_grid, m_grid = np.meshgrid(l_axis, m_axis, indexing="ij")
    image = np.zeros((l_axis.size, m_axis.size), dtype=np.float64)
    for component in normalize_sky_components(components):
        ra_dec = component["ra_dec"]
        ra_dec = ra_dec[time_index if ra_dec.shape[0] > 1 else 0]
        l0, m0 = sin_project(phase_center, ra_dec)
        flux = component_flux(component, time_index, frequency_index, correlation)
        if component["kind"] == "point":
            i_l = int(np.round((l0 - l_axis[0]) / dl))
            i_m = int(np.round((m0 - m_axis[0]) / dm))
            if 0 <= i_l < l_axis.size and 0 <= i_m < m_axis.size:
                image[i_l, i_m] += flux
            continue
        image += (
            flux
            * pixel_area
            * component_image(component, l_grid - l0, m_grid - m0, pixel_area)
        )
    return image


# Coefficients of the four instrumental correlations ([XX, XY, YX, YY] or
# [RR, RL, LR, LL], the order of a component's flux vector) that make each
# Stokes parameter: I = (XX + YY)/2, Q = (XX - YY)/2, U = (XY + YX)/2,
# V = (XY - YX)/(2i) for linear feeds; I = (RR + LL)/2, V = (RR - LL)/2,
# Q = (RL + LR)/2, U = (RL - LR)/(2i) for circular feeds.
_STOKES_FROM_CORRELATIONS = {
    "linear": {
        "I": ((0, 0.5), (3, 0.5)),
        "Q": ((0, 0.5), (3, -0.5)),
        "U": ((1, 0.5), (2, 0.5)),
        "V": ((1, -0.5j), (2, 0.5j)),
    },
    "circular": {
        "I": ((0, 0.5), (3, 0.5)),
        "Q": ((1, 0.5), (2, 0.5)),
        "U": ((1, -0.5j), (2, 0.5j)),
        "V": ((0, 0.5), (3, -0.5)),
    },
}


def polarization_basis_of(polarization):
    """``"linear"`` for ``XX/XY/YX/YY`` labels, ``"circular"`` for ``RR/RL/LR/LL``."""
    labels = {str(label).upper() for label in polarization}
    if labels and labels <= {"XX", "XY", "YX", "YY"}:
        return "linear"
    if labels and labels <= {"RR", "RL", "LR", "LL"}:
        return "circular"
    raise ValueError(
        f"polarization labels {sorted(labels)} are not one instrumental basis."
    )


def stokes_sky_model_images(
    components,
    l_axis,
    m_axis,
    phase_center_ra_dec,
    polarization_basis,
    stokes,
    time_index=0,
    frequency_index=0,
):
    """Rasterise a component list into Stokes planes, ``[n_stokes, n_l, n_m]`` in Jy/pixel.

    Component fluxes are per instrumental correlation (see
    :func:`normalize_sky_components`); each requested Stokes parameter is the
    standard combination of the correlation images rasterised by
    :func:`sky_model_image` (linear feeds: ``I = (XX + YY)/2``,
    ``Q = (XX - YY)/2``, ``U = (XY + YX)/2``, ``V = (XY - YX)/(2i)``; circular
    feeds: ``I = (RR + LL)/2``, ``V = (RR - LL)/2``, ``Q = (RL + LR)/2``,
    ``U = (RL - LR)/(2i)``).  Component fluxes are real, so the cross-hand
    difference (``V`` for linear feeds, ``U`` for circular feeds) is zero.

    Parameters
    ----------
    components, l_axis, m_axis, phase_center_ra_dec, time_index, frequency_index
        As for :func:`sky_model_image`.
    polarization_basis : str
        ``"linear"`` or ``"circular"`` (:func:`polarization_basis_of`).
    stokes : list of str
        Stokes parameters to produce, a subset of ``["I", "Q", "U", "V"]``.
    """
    table = _STOKES_FROM_CORRELATIONS[polarization_basis]
    stokes = [str(label).upper() for label in stokes]
    unknown = [label for label in stokes if label not in table]
    if unknown:
        raise ValueError(
            f"unknown Stokes parameters {unknown}; expected a subset of I, Q, U, V."
        )
    needed = sorted({corr for label in stokes for corr, _ in table[label]})
    planes = {
        corr: sky_model_image(
            components,
            l_axis,
            m_axis,
            phase_center_ra_dec,
            time_index,
            frequency_index,
            correlation=corr,
        )
        for corr in needed
    }
    out = np.zeros((len(stokes), len(l_axis), len(m_axis)), dtype=np.float64)
    for k, label in enumerate(stokes):
        out[k] = np.real(
            sum(coefficient * planes[corr] for corr, coefficient in table[label])
        )
    return out


def sky_components_from_arrays(kind, flux, ra_dec, shape=None, limb_darkening=None):
    """Convert the bulk ``<kind>_source_flux / _ra_dec / _shape`` arrays into component dicts.

    Parameters
    ----------
    kind : {"point", "gaussian", "disk", "gaussian_ring"}
    flux : array_like, [n_source, n_time | 1, n_frequency | 1, 4]
    ra_dec : array_like, [n_time | 1, n_source, 2]
    shape : array_like, [n_source, n_columns], optional
        ``[major, minor, pa]`` (gaussian, disk) or ``[radius, fwhm, inclination, pa]``
        (gaussian_ring).
    limb_darkening : array_like, [n_source], optional
        Disk limb-darkening exponents (default 0).

    Returns
    -------
    list of dict (not yet normalised).
    """
    flux = np.asarray(flux, dtype=np.float64)
    ra_dec = np.asarray(ra_dec, dtype=np.float64)
    if flux.ndim != 4 or ra_dec.ndim != 3 or flux.shape[0] != ra_dec.shape[1]:
        raise ValueError(
            f"{kind}_source_flux must be [n_source, n_time|1, n_frequency|1, 4] and "
            f"{kind}_source_ra_dec [n_time|1, n_source, 2]; got {flux.shape} and {ra_dec.shape}."
        )
    n_source = flux.shape[0]
    components = []
    for i_source in range(n_source):
        component = {
            "kind": kind,
            "flux": flux[i_source],
            "ra_dec": ra_dec[:, i_source, :],
        }
        if kind in _LEGACY_SHAPE_COLUMNS:
            columns = _LEGACY_SHAPE_COLUMNS[kind]
            shape_array = np.asarray(shape, dtype=np.float64)
            if shape_array.shape != (n_source, len(columns)):
                raise ValueError(
                    f"{kind}_source_shape must have shape [n_source, {len(columns)}]; got {shape_array.shape}."
                )
            component.update(zip(columns, shape_array[i_source], strict=True))
            if kind == "disk" and limb_darkening is not None:
                component["limb_darkening"] = float(
                    np.asarray(limb_darkening, dtype=np.float64)[i_source]
                )
        components.append(component)
    return components


def slice_sky_components(components, time_slice, frequency_slice):
    """Cut the time / frequency axes of normalised components to one chunk (singleton axes broadcast)."""

    def cut(array, axis, index):
        if array.shape[axis] == 1:
            return array
        selector = [slice(None)] * array.ndim
        selector[axis] = index
        return array[tuple(selector)]

    out = []
    for component in components:
        chunk = dict(component)
        chunk["flux"] = cut(cut(component["flux"], 0, time_slice), 1, frequency_slice)
        chunk["ra_dec"] = cut(component["ra_dec"], 0, time_slice)
        out.append(chunk)
    return out


def describe_sky_components(components):
    """``"3 point source(s), 1 m-ring source(s)"`` for the MSv4 description."""
    counts: dict[str, int] = {}
    for component in components:
        counts[component["kind"]] = counts.get(component["kind"], 0) + 1
    return ", ".join(
        f"{count} {COMPONENT_KINDS[kind].label} source(s)"
        for kind, count in counts.items()
    )
