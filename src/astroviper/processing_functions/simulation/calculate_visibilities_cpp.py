"""Python wrapper of the multithreaded C++ visibility kernel.

Mirrors :func:`astroviper.processing_functions.simulation.calculate_visibilities`
(NumPy implementation): the per-(time, source) rotation vectors are computed
here in NumPy (cheap, ``n_time * n_source`` 3x3 products) and the heavy
``time x baseline x source x frequency`` loop runs in C++ with the GIL released.
"""

from __future__ import annotations

import numpy as np

from astroviper.utils.coordinate_transforms import calculate_uvw_rotation


def cpp_kernel_available() -> bool:
    """True when the compiled ``_visibility_kernel_ext`` module can be imported."""
    try:
        from astroviper.processing_functions.simulation import (
            visibility_kernel_cpp,  # noqa: F401
        )
    except ImportError:
        return False
    return True


def _packed_model_for_cpp(model: dict) -> dict:
    """Ensure the array members of a packed beam model are C-contiguous with the kernel dtypes."""
    out = dict(model)
    if model["kind"] == "polynomial":
        out["frequency"] = np.ascontiguousarray(model["frequency"], dtype=np.float64)
        out["coefficients"] = np.ascontiguousarray(
            model["coefficients"], dtype=np.float64
        )
    elif model["kind"] == "jones_image":
        out["jones"] = np.ascontiguousarray(model["jones"], dtype=np.complex128)
        out["parallactic_angle"] = np.ascontiguousarray(
            model["parallactic_angle"], dtype=np.float64
        )
        out["frequency"] = np.ascontiguousarray(model["frequency"], dtype=np.float64)
        out["polarization_index"] = np.ascontiguousarray(
            model["polarization_index"], dtype=np.int64
        )
    return out


def calculate_visibilities_cpp(
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
    processing_function_threads: int = 1,
) -> np.ndarray:
    """C++ implementation of :func:`~astroviper.processing_functions.simulation.calculate_visibilities`.

    Same parameters and result as the NumPy implementation; ``processing_function_threads``
    ``<= 0`` uses all hardware threads.
    """
    from astroviper.processing_functions.simulation.visibility_kernel_cpp import (
        calculate_visibilities as _kernel,
    )

    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    antenna1 = np.ascontiguousarray(antenna1, dtype=np.int64)
    antenna2 = np.ascontiguousarray(antenna2, dtype=np.int64)
    frequency = np.ascontiguousarray(frequency, dtype=np.float64)
    polarization_index = np.ascontiguousarray(polarization_index, dtype=np.int64)
    flux = np.ascontiguousarray(point_source_flux, dtype=np.complex128)
    source_ra_dec = np.ascontiguousarray(point_source_ra_dec, dtype=np.float64)
    phase_center = np.asarray(phase_center_ra_dec, dtype=np.float64)
    beam_model_map = np.ascontiguousarray(beam_model_map, dtype=np.int64)
    parallactic_angle = np.ascontiguousarray(parallactic_angle, dtype=np.float64)
    mueller_selection = np.ascontiguousarray(mueller_selection, dtype=np.int64)

    n_time, n_baseline, _ = uvw.shape
    n_source = source_ra_dec.shape[1]
    f_pc_time = n_time if phase_center.shape[0] == 1 else 1
    f_src_time = n_time if source_ra_dec.shape[0] == 1 else 1

    # rotation vectors k = R @ lmn_rot and 1/n per (time, source)
    k_vector = np.empty((n_time, n_source, 3), dtype=np.float64)
    inverse_n = np.empty((n_time, n_source), dtype=np.float64)
    for i_time in range(n_time):
        pc = phase_center[i_time // f_pc_time]
        for i_source in range(n_source):
            rotation, lmn_rot = calculate_uvw_rotation(
                pc, source_ra_dec[i_time // f_src_time, i_source]
            )
            k_vector[i_time, i_source] = rotation @ lmn_rot
            inverse_n[i_time, i_source] = 1.0 / (1.0 - lmn_rot[2])

    if pointing_ra_dec is None:
        pointing = np.ascontiguousarray(
            phase_center.reshape(-1, 1, 2), dtype=np.float64
        )
    else:
        pointing = np.ascontiguousarray(pointing_ra_dec, dtype=np.float64)

    visibility = np.zeros(
        (n_time, n_baseline, frequency.shape[0], polarization_index.shape[0]),
        dtype=np.complex128,
    )
    _kernel(
        visibility,
        uvw,
        antenna1,
        antenna2,
        frequency,
        polarization_index,
        flux,
        k_vector,
        inverse_n,
        source_ra_dec,
        pointing,
        beam_model_map,
        [_packed_model_for_cpp(m) for m in packed_beam_models],
        parallactic_angle,
        mueller_selection,
        int(processing_function_threads),
    )
    return visibility
