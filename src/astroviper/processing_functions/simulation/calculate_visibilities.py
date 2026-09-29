"""Point-source visibilities with direction-dependent antenna beams (DFT).

For every time ``t``, baseline ``b = (a1, a2)``, channel ``nu`` and point source
``s``::

    V[t, b, nu] += M(J_a1, J_a2) @ S_s(t, nu) * exp(i 2 pi k_s(t) . uvw[t, b] nu / c) / n_s(t)

where ``S`` is the 4-correlation flux of the source, ``J_a`` the antenna Jones
vector sampled at the source direction relative to the antenna pointing
(:func:`~astroviper.processing_functions.simulation.antenna_beams.sample_jones`),
``M`` the Mueller matrix restricted to ``mueller_selection``
(:func:`~astroviper.processing_functions.simulation.antenna_beams.apply_mueller`),
``k_s`` the rotated direction vector of the source with respect to the phase
centre and ``n_s`` its ``n`` direction cosine
(:func:`~astroviper.utils.coordinate_transforms.calculate_uvw_rotation`).

Extended sources (Gaussians, limb-darkened disks, Gaussian rings, m-rings,
crescents, annuli, exponential disks, tapered power laws and shapelets -- the
kinds of :mod:`~astroviper.processing_functions.simulation.sky_components`)
are point sources whose visibilities are multiplied by the analytic
visibility ``T(u', v', w')`` of their unit-flux sky profile
(:func:`~astroviper.processing_functions.simulation.sky_components.component_uv_response`).
The point-source kernel supplies the exact phase and ``1/n`` of the component
centre (no approximation, including the w term of the centre); the response is
evaluated at the baseline coordinates **rotated into the frame of the
component** (:func:`source_frame_uvw`, so ``(u', v', w')`` are relative to the
component centre) and **includes the w term** of the extended structure through
the paraxial expansion ``w' (n' - 1) ~ -w' r'^2 / 2``.  Neither approximation of
the tangent-plane treatment (phase-centre ``(u, v)``, no w) is made; what
remains is second order in the component size (``pi w' theta^4 / 4`` and
``theta^2 / 2``, negligible below a degree).  The antenna beams are sampled at
the source centre (valid for sources much smaller than the primary beam).

This module holds the vectorised NumPy implementation (the reference used in the
tests); ``implementation="cpp"`` selects the multithreaded C++ kernel when it is
built.
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.antenna_beams import (
    SPEED_OF_LIGHT,
    apply_mueller,
    sample_jones,
)
from astroviper.utils.coordinate_transforms import calculate_uvw_rotation, sin_project


def _broadcast_index(n: int, size: int) -> int:
    """Divisor mapping an index ``0..n-1`` onto a singleton (size 1) or full axis."""
    return n if size == 1 else 1


def source_frame_uvw(uvw, frequency, phase_center_ra_dec, source_ra_dec):
    """Baseline coordinates in wavelengths, rotated into the frame of a source.

    The phase-centre ``uvw`` (``w`` towards the phase centre) are rotated so
    that ``w'`` points at the source: ``uvw' = uvw @ R`` with the rotation of
    :func:`~astroviper.utils.coordinate_transforms.calculate_uvw_rotation`
    (the same rotation the point-source kernel uses for the phase, so the
    extended-source responses and the kernel are exactly consistent).  For a
    source at the phase centre the rotation is the identity.

    Parameters
    ----------
    uvw : np.ndarray, [n_time, n_baseline, 3], metres
    frequency : np.ndarray, [n_frequency], Hz
    phase_center_ra_dec : np.ndarray, [n_time | 1, 2], radians
    source_ra_dec : np.ndarray, [n_time | 1, 2], radians

    Returns
    -------
    u, v, w : np.ndarray, [n_time, n_baseline, n_frequency], wavelengths
    """
    uvw = np.asarray(uvw, dtype=np.float64)
    inverse_wavelength = np.asarray(frequency, dtype=np.float64) / SPEED_OF_LIGHT
    phase_center = np.asarray(phase_center_ra_dec, dtype=np.float64).reshape(-1, 2)
    source = np.asarray(source_ra_dec, dtype=np.float64).reshape(-1, 2)
    n_time = uvw.shape[0]
    f_pc_time = _broadcast_index(n_time, phase_center.shape[0])
    f_src_time = _broadcast_index(n_time, source.shape[0])
    rotated = np.empty_like(uvw)
    for i_time in range(n_time):
        rotation, _ = calculate_uvw_rotation(
            phase_center[i_time // f_pc_time], source[i_time // f_src_time]
        )
        rotated[i_time] = uvw[i_time] @ rotation
    u = rotated[:, :, 0, None] * inverse_wavelength
    v = rotated[:, :, 1, None] * inverse_wavelength
    w = rotated[:, :, 2, None] * inverse_wavelength
    return u, v, w


def calculate_visibilities(
    uvw: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    frequency: np.ndarray,
    polarization_index: np.ndarray,
    point_source_flux: np.ndarray,
    point_source_ra_dec: np.ndarray,
    phase_center_ra_dec: np.ndarray,
    pointing_ra_dec: np.ndarray | None,
    beam_model_map: np.ndarray,
    packed_beam_models: list[dict],
    parallactic_angle: np.ndarray,
    mueller_selection: np.ndarray,
    processing_function_threads: int = 1,
    implementation: str = "cpp",
    gaussian_source_flux: np.ndarray | None = None,
    gaussian_source_ra_dec: np.ndarray | None = None,
    gaussian_source_shape: np.ndarray | None = None,
    disk_source_flux: np.ndarray | None = None,
    disk_source_ra_dec: np.ndarray | None = None,
    disk_source_shape: np.ndarray | None = None,
    disk_source_limb_darkening: np.ndarray | None = None,
    gaussian_ring_source_flux: np.ndarray | None = None,
    gaussian_ring_source_ra_dec: np.ndarray | None = None,
    gaussian_ring_source_shape: np.ndarray | None = None,
    sky_components: list | None = None,
) -> np.ndarray:
    """Simulate the visibilities of point sources and analytic extended components for one in-memory chunk.

    Parameters
    ----------
    uvw : np.ndarray, [n_time, n_baseline, 3], metres
    antenna1, antenna2 : np.ndarray, [n_baseline] int
    frequency : np.ndarray, [n_frequency], Hz
    polarization_index : np.ndarray, [n_polarization] int
        Index (0..3, row-major correlation order) of each output polarization.
    point_source_flux : np.ndarray, [n_source, n_time | 1, n_frequency | 1, 4], Jy, or None
        Flux of each point source in the four instrumental correlations
        (``None``: no bulk point sources; give sources through ``sky_components``).
    point_source_ra_dec : np.ndarray, [n_time | 1, n_source, 2], radians, or None
    phase_center_ra_dec : np.ndarray, [n_time | 1, 2], radians
    pointing_ra_dec : np.ndarray, [n_time | 1, n_antenna | 1, 2], radians, or None
        Antenna pointing directions; ``None`` means every antenna points at the
        phase centre.
    beam_model_map : np.ndarray, [n_antenna] int
        Index into ``packed_beam_models`` for each antenna.
    packed_beam_models : list of dict
        From :func:`~astroviper.processing_functions.simulation.antenna_beams.pack_beam_models`.
    parallactic_angle : np.ndarray, [n_time], radians
    mueller_selection : np.ndarray of int
    processing_function_threads : int
        Threads used by the C++ kernel (ignored by the NumPy implementation).
    implementation : {"numpy", "cpp"}
    gaussian_source_flux : np.ndarray, [n_gaussian, n_time | 1, n_frequency | 1, 4], Jy, optional
        Integrated flux of each Gaussian source in the four instrumental
        correlations.  ``None`` (default) simulates no Gaussian sources.
    gaussian_source_ra_dec : np.ndarray, [n_time | 1, n_gaussian, 2], radians, optional
        Right ascension and declination of the Gaussian sources.
    gaussian_source_shape : np.ndarray, [n_gaussian, 3], radians, optional
        ``[major, minor, position angle]`` FWHM shape of each Gaussian source,
        in the convention of the imaging clean beam
        (:func:`astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper`).
    disk_source_flux : np.ndarray, [n_disk, n_time | 1, n_frequency | 1, 4], Jy, optional
        Integrated flux of each limb-darkened disk source in the four
        instrumental correlations; singleton time/frequency axes broadcast.
        ``None`` (default) simulates no disk sources.
    disk_source_ra_dec : np.ndarray, [n_time | 1, n_disk, 2], radians, optional
        Right ascension and declination of the disk sources (per time or fixed).
    disk_source_shape : np.ndarray, [n_disk, 3], radians, optional
        ``[major, minor, position angle]`` outer diameters and orientation of
        each (inclined) disk, in the Gaussian-source / clean-beam position-angle
        convention
        (:func:`astroviper.processing_functions.simulation.limb_darkened_disk.limb_darkened_disk_uv_response`).
    disk_source_limb_darkening : np.ndarray, [n_disk] float, optional
        Power-law limb-darkening exponent ``alpha`` of each disk
        (``I ~ mu**alpha``, Hestroffer 1997): ``0`` uniform disk (the default
        when ``None``), ``> 0`` darker towards the limb, ``-2 < alpha < 0``
        limb brightened, ``-2`` an infinitely thin ring.
    gaussian_ring_source_flux : np.ndarray, [n_ring, n_time | 1, n_frequency | 1, 4], Jy, optional
        Integrated flux of each Gaussian-broadened ring source.
    gaussian_ring_source_ra_dec : np.ndarray, [n_time | 1, n_ring, 2], radians, optional
        Right ascension and declination of the ring sources.
    gaussian_ring_source_shape : np.ndarray, [n_ring, 4], radians, optional
        ``[radius, fwhm, inclination, position angle]`` of each ring.
    sky_components : list of dict, optional
        Sky components of any kind (point, gaussian, disk, gaussian_ring,
        m_ring, crescent, annulus, exponential_disk, tapered_power_law,
        shapelet): ``{"kind", "flux", "ra_dec", <shape parameters>}`` as
        described in :mod:`~astroviper.processing_functions.simulation.sky_components`
        (validated with
        :func:`~astroviper.processing_functions.simulation.sky_components.normalize_sky_components`).
        The ``<kind>_source_*`` arrays above are the bulk form of the same
        components and may be combined with this list.

    Returns
    -------
    np.ndarray, [n_time, n_baseline, n_frequency, n_polarization] complex128

    Notes
    -----
    Every extended component is a point source whose visibilities are
    multiplied by the analytic visibility of its unit-flux sky profile
    (:func:`~astroviper.processing_functions.simulation.sky_components.component_uv_response`;
    for Gaussians that is the imaging restore module's
    :func:`~astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper`
    at ``w = 0``), so it shares the beam response, phase and kernel
    implementations of the point sources and its integrated flux equals its
    ``flux``.  All responses are evaluated at the source-frame
    ``(u', v', w')`` of :func:`source_frame_uvw` and include the w term
    (module docstring).  Point components of ``sky_components`` are grouped
    with the bulk point sources into as few kernel calls as their array
    shapes allow.
    """
    from astroviper.processing_functions.simulation.sky_components import (
        component_uv_response,
        normalize_sky_components,
        sky_components_from_arrays,
    )

    uvw = np.asarray(uvw, dtype=np.float64)
    frequency = np.asarray(frequency, dtype=np.float64)
    n_time = uvw.shape[0]
    n_frequency = frequency.shape[0]

    def kernel(flux, ra_dec):
        return _calculate_point_source_visibilities(
            uvw,
            antenna1,
            antenna2,
            frequency,
            polarization_index,
            flux,
            ra_dec,
            phase_center_ra_dec,
            pointing_ra_dec,
            beam_model_map,
            packed_beam_models,
            parallactic_angle,
            mueller_selection,
            processing_function_threads,
            implementation,
        )

    # All sources as one normalised component list: the bulk arrays are converted
    # (points are kept as bulk groups so the kernel sees them in one call).
    components = []
    for kind, flux, ra_dec, shape, extra in [
        ("gaussian", gaussian_source_flux, gaussian_source_ra_dec, gaussian_source_shape, None),
        ("disk", disk_source_flux, disk_source_ra_dec, disk_source_shape, disk_source_limb_darkening),
        ("gaussian_ring", gaussian_ring_source_flux, gaussian_ring_source_ra_dec, gaussian_ring_source_shape, None),
    ]:  # fmt: skip
        if flux is not None:
            components += sky_components_from_arrays(kind, flux, ra_dec, shape, extra)
    components = normalize_sky_components(
        components, n_time, n_frequency
    ) + normalize_sky_components(sky_components, n_time, n_frequency)

    point_groups = []
    if point_source_flux is not None:
        point_source_flux = np.asarray(point_source_flux, dtype=np.float64)
        point_source_ra_dec = np.asarray(point_source_ra_dec, dtype=np.float64)
        if point_source_flux.shape[0] > 0:
            point_groups.append((point_source_flux, point_source_ra_dec))
    # point components: stack those with identical time/frequency layouts
    by_layout = {}
    for component in components:
        if component["kind"] == "point":
            key = (component["flux"].shape, component["ra_dec"].shape)
            by_layout.setdefault(key, []).append(component)
    for group in by_layout.values():
        point_groups.append(
            (
                np.stack([c["flux"] for c in group]),
                np.stack([c["ra_dec"] for c in group], axis=1),
            )
        )

    n_baseline = uvw.shape[1]
    n_polarization = np.shape(polarization_index)[0]
    visibility = np.zeros(
        (n_time, n_baseline, n_frequency, n_polarization), dtype=np.complex128
    )
    for flux, ra_dec in point_groups:
        visibility += kernel(flux, ra_dec)

    for component in components:
        if component["kind"] == "point":
            continue
        ra_dec = component["ra_dec"]
        # baseline coordinates in the frame of this source, in wavelengths
        # per channel: [n_time, n_baseline, n_frequency]
        u, v, w = source_frame_uvw(uvw, frequency, phase_center_ra_dec, ra_dec)
        source_visibility = kernel(component["flux"][None], ra_dec[:, None, :])
        source_visibility *= component_uv_response(component, u, v, w=w)[..., None]
        visibility += source_visibility

    return visibility


def _calculate_point_source_visibilities(
    uvw,
    antenna1,
    antenna2,
    frequency,
    polarization_index,
    point_source_flux,
    point_source_ra_dec,
    phase_center_ra_dec,
    pointing_ra_dec,
    beam_model_map,
    packed_beam_models,
    parallactic_angle,
    mueller_selection,
    processing_function_threads,
    implementation,
):
    """Dispatch one point-source kernel evaluation (numpy or C++)."""
    if implementation == "numpy":
        return _calculate_visibilities_numpy(
            uvw,
            antenna1,
            antenna2,
            frequency,
            polarization_index,
            point_source_flux,
            point_source_ra_dec,
            phase_center_ra_dec,
            pointing_ra_dec,
            beam_model_map,
            packed_beam_models,
            parallactic_angle,
            mueller_selection,
        )
    if implementation == "cpp":
        from astroviper.processing_functions.simulation.calculate_visibilities_cpp import (
            calculate_visibilities_cpp,
        )

        return calculate_visibilities_cpp(
            uvw,
            antenna1,
            antenna2,
            frequency,
            polarization_index,
            point_source_flux,
            point_source_ra_dec,
            phase_center_ra_dec,
            pointing_ra_dec,
            beam_model_map,
            packed_beam_models,
            parallactic_angle,
            mueller_selection,
            processing_function_threads,
        )
    raise ValueError(
        f"implementation must be 'numpy' or 'cpp', got {implementation!r}."
    )


def _calculate_visibilities_numpy(
    uvw,
    antenna1,
    antenna2,
    frequency,
    polarization_index,
    point_source_flux,
    point_source_ra_dec,
    phase_center_ra_dec,
    pointing_ra_dec,
    beam_model_map,
    packed_beam_models,
    parallactic_angle,
    mueller_selection,
):
    uvw = np.asarray(uvw, dtype=np.float64)
    antenna1 = np.asarray(antenna1, dtype=np.int64)
    antenna2 = np.asarray(antenna2, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64)
    polarization_index = np.asarray(polarization_index, dtype=np.int64)
    flux = np.asarray(point_source_flux, dtype=np.complex128)
    source_ra_dec = np.asarray(point_source_ra_dec, dtype=np.float64)
    phase_center = np.asarray(phase_center_ra_dec, dtype=np.float64)
    beam_model_map = np.asarray(beam_model_map, dtype=np.int64)
    parallactic_angle = np.asarray(parallactic_angle, dtype=np.float64)
    mueller_selection = np.asarray(mueller_selection, dtype=np.int64)

    n_time, n_baseline, _ = uvw.shape
    n_chan = frequency.shape[0]
    n_pol = polarization_index.shape[0]
    n_antenna = beam_model_map.shape[0]
    n_source = source_ra_dec.shape[1]
    if flux.shape[0] != n_source or flux.shape[3] != 4:
        raise ValueError(
            "point_source_flux must have shape [n_source, n_time | 1, n_frequency | 1, 4]."
        )

    f_pc_time = _broadcast_index(n_time, phase_center.shape[0])
    f_src_time = _broadcast_index(n_time, source_ra_dec.shape[0])
    f_flux_time = _broadcast_index(n_time, flux.shape[1])
    f_flux_chan = _broadcast_index(n_chan, flux.shape[2])
    do_pointing = pointing_ra_dec is not None
    if do_pointing:
        pointing = np.asarray(pointing_ra_dec, dtype=np.float64)
        f_pt_time = _broadcast_index(n_time, pointing.shape[0])

    chan_index = np.arange(n_chan) // f_flux_chan
    visibility = np.zeros((n_time, n_baseline, n_chan, n_pol), dtype=np.complex128)
    jones_all = np.empty((n_antenna, n_chan, 4), dtype=np.complex128)

    for i_time in range(n_time):
        pa = parallactic_angle[i_time]
        pc = phase_center[i_time // f_pc_time]
        if do_pointing:
            pointing_t = np.broadcast_to(pointing[i_time // f_pt_time], (n_antenna, 2))
        for i_source in range(n_source):
            ra_dec = source_ra_dec[i_time // f_src_time, i_source]
            uvw_rotation, lmn_rot = calculate_uvw_rotation(pc, ra_dec)
            k_vector = uvw_rotation @ lmn_rot
            phase = 2 * np.pi * (uvw[i_time] @ k_vector)  # [n_baseline]
            fringe = np.exp(
                1j * phase[:, None] * frequency[None, :] / SPEED_OF_LIGHT
            ) / (1.0 - lmn_rot[2])

            # Jones of every antenna at this source (grouped per beam model)
            if do_pointing:
                lm_antenna = sin_project_per_antenna(
                    pointing_t, ra_dec
                )  # [n_antenna, 2]
            else:
                lm_antenna = np.broadcast_to(sin_project(pc, ra_dec), (n_antenna, 2))
            for i_model in np.unique(beam_model_map):
                ants = np.where(beam_model_map == i_model)[0]
                if do_pointing:
                    jones_all[ants] = sample_jones(
                        packed_beam_models[i_model], lm_antenna[ants], frequency, pa
                    )
                else:
                    jones_all[ants] = sample_jones(
                        packed_beam_models[i_model], lm_antenna[:1], frequency, pa
                    )[0]

            source_flux = flux[i_source, i_time // f_flux_time][
                chan_index
            ]  # [n_chan, 4]
            flux_scaled = apply_mueller(
                jones_all[antenna1],
                jones_all[antenna2],
                source_flux[None, :, :],
                mueller_selection,
            )  # [n_baseline, n_chan, 4]
            visibility[i_time] += (
                flux_scaled[:, :, polarization_index] * fringe[:, :, None]
            )
    return visibility


def sin_project_per_antenna(
    pointing_ra_dec: np.ndarray, ra_dec: np.ndarray
) -> np.ndarray:
    """SIN-projected source direction relative to each antenna's pointing.

    Parameters
    ----------
    pointing_ra_dec : np.ndarray, [n_antenna, 2], radians
    ra_dec : np.ndarray, [2], radians

    Returns
    -------
    np.ndarray, [n_antenna, 2]
    """
    pointing_ra_dec = np.asarray(pointing_ra_dec, dtype=np.float64)
    ra_o, dec_o = pointing_ra_dec[:, 0], pointing_ra_dec[:, 1]
    ra, dec = float(ra_dec[0]), float(ra_dec[1])
    d_ra = ra - ra_o
    l = np.cos(dec) * np.sin(d_ra)  # noqa: E741
    m = np.sin(dec) * np.cos(dec_o) - np.cos(dec) * np.sin(dec_o) * np.cos(d_ra)
    return np.stack([l, m], axis=-1)
