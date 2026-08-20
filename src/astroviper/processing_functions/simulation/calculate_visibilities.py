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
) -> np.ndarray:
    """Simulate the visibilities of point and Gaussian sources for one in-memory chunk.

    Parameters
    ----------
    uvw : np.ndarray, [n_time, n_baseline, 3], metres
    antenna1, antenna2 : np.ndarray, [n_baseline] int
    frequency : np.ndarray, [n_frequency], Hz
    polarization_index : np.ndarray, [n_polarization] int
        Index (0..3, row-major correlation order) of each output polarization.
    point_source_flux : np.ndarray, [n_source, n_time | 1, n_frequency | 1, 4], Jy
        Flux of each source in the four instrumental correlations.
    point_source_ra_dec : np.ndarray, [n_time | 1, n_source, 2], radians
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

    Returns
    -------
    np.ndarray, [n_time, n_baseline, n_frequency, n_polarization] complex128

    Notes
    -----
    A Gaussian source is a point source whose visibilities are multiplied by
    the analytic Fourier transform of its sky Gaussian -- the imaging restore
    module's :func:`~astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper`
    (the single source of truth for the Gaussian parametrisation) -- so it
    shares the beam response, phase and kernel implementations of the point
    sources and its integrated flux equals ``gaussian_source_flux``.
    """
    point_args = (
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
    )
    visibility = _calculate_point_source_visibilities(*point_args)

    if gaussian_source_flux is not None:
        from astroviper.processing_functions.imaging.restore import (
            elliptical_gaussian_uv_taper,
        )

        gaussian_source_flux = np.asarray(gaussian_source_flux, dtype=np.float64)
        gaussian_source_ra_dec = np.asarray(gaussian_source_ra_dec, dtype=np.float64)
        gaussian_source_shape = np.asarray(gaussian_source_shape, dtype=np.float64)
        # Baseline coordinates in wavelengths per channel:
        # [n_time, n_baseline, n_frequency].
        inverse_wavelength = frequency / 299792458.0
        u = uvw[:, :, 0, None] * inverse_wavelength
        v = uvw[:, :, 1, None] * inverse_wavelength
        for source in range(gaussian_source_flux.shape[0]):
            source_visibility = _calculate_point_source_visibilities(
                uvw,
                antenna1,
                antenna2,
                frequency,
                polarization_index,
                gaussian_source_flux[source : source + 1],
                gaussian_source_ra_dec[:, source : source + 1, :],
                phase_center_ra_dec,
                pointing_ra_dec,
                beam_model_map,
                packed_beam_models,
                parallactic_angle,
                mueller_selection,
                processing_function_threads,
                implementation,
            )
            major, minor, pa = gaussian_source_shape[source]
            source_visibility *= elliptical_gaussian_uv_taper(u, v, major, minor, pa)[
                ..., None
            ]
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
