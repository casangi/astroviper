"""Visibilities of a sky image, by the exact direct Fourier transform or by degridding.

The image is predicted with the imager's own prediction functions:
``"dft"`` with
:func:`~astroviper.processing_functions.imaging.direct_fourier_transform.predict_dft_visibilities_single_field`
(the exact measurement equation of the simulator, the w term included when the
``uvw`` hold it), ``"gridded"`` with the anti-aliasing kernel of
``gridding_kernel_params``
(:func:`~astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder.fft_norm_img_xds`
and
:func:`~astroviper.processing_functions.imaging.get_visibility_grid.get_visibility_grid_single_field`),
which is two dimensional: it has no w term.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

SKY_IMAGE_PREDICTIONS = ("dft", "gridded")


def sky_image_visibilities(
    sky_image: xr.Dataset,
    uvw: np.ndarray,
    frequency: np.ndarray,
    polarization: list,
    primary_beam: np.ndarray | None = None,
    predict: str = "dft",
    gridding_kernel_params: dict | None = None,
    fft_padding: float = 1.2,
    processing_function_threads: int = 1,
) -> np.ndarray:
    """Visibilities of a Stokes sky image on the simulated baselines.

    Parameters
    ----------
    sky_image : xarray.Dataset
        XRADIO image with ``SKY`` (``time, frequency, polarization, l, m``) in
        Jy per pixel, Stokes ``polarization`` coordinates (``["I"]``, the two
        Stokes planes of the parallel hands of the basis of ``polarization``,
        or ``["I", "Q", "U", "V"]``) and ``l, m`` direction cosines about the
        simulation's phase centre. Time plane 0 is used; every simulated
        channel takes the image channel nearest in frequency.
    uvw : numpy.ndarray
        ``[n_time, n_baseline, 3]`` baseline coordinates in metres (w already
        zero when the simulation has no w term).
    frequency : numpy.ndarray
        ``[n_frequency]`` channel frequencies in Hz.
    polarization : list of str
        Simulated correlations, in the order of the output axis.
    primary_beam : numpy.ndarray, optional
        ``[n_frequency, n_l, n_m]`` power primary beam on the image grid,
        applied per simulated channel. ``None``: the image is already the
        apparent sky.
    predict : {"dft", "gridded"}, optional
        Exact direct Fourier transform over the non-zero pixels, or FFT and
        degridding with ``gridding_kernel_params`` (no w term).
    gridding_kernel_params : dict, optional
        Kernel of ``predict="gridded"`` (see
        :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel.resolve_gridding_kernel_params`).
    fft_padding : float, optional
        FFT padding of ``predict="gridded"``.
    processing_function_threads : int, optional
        Threads of the prediction kernels.

    Returns
    -------
    numpy.ndarray
        ``complex128`` ``[n_time, n_baseline, n_frequency, n_polarization]``.

    Raises
    ------
    ValueError
        For an unknown ``predict``, a primary beam of the wrong shape, or Stokes
        planes that do not give the simulated correlations.
    """
    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        map_visibility_frequencies_to_image,
    )
    from astroviper.processing_functions.simulation.sky_components import (
        polarization_basis_of,
    )
    from astroviper.utils.data_group_tools import modify_data_groups_xds

    if predict not in SKY_IMAGE_PREDICTIONS:
        raise ValueError(
            f"predict must be one of {SKY_IMAGE_PREDICTIONS}; got {predict!r}."
        )
    frequency = np.asarray(frequency, dtype=np.float64)
    polarization = [str(p) for p in polarization]
    basis = polarization_basis_of(polarization)

    sky = sky_image["SKY"].transpose("time", "frequency", "polarization", "l", "m")
    stokes = [str(s) for s in sky.polarization.values]
    planes = sky.isel(time=0).values.astype(np.float64)
    nearest = map_visibility_frequencies_to_image(frequency, sky.frequency.values)
    planes = planes[nearest]  # [n_frequency, n_stokes, n_l, n_m]
    if stokes == ["I"]:
        # The Stokes plane of the parallel hands that is not I is zero.
        planes = np.concatenate([planes, np.zeros_like(planes)], axis=1)
        stokes = ["I", "Q" if basis == "linear" else "V"]
    if primary_beam is not None:
        primary_beam = np.asarray(primary_beam, dtype=np.float64)
        if primary_beam.shape != (frequency.size,) + planes.shape[-2:]:
            raise ValueError(
                "primary_beam must have shape (n_frequency, n_l, n_m) = "
                f"{(frequency.size,) + planes.shape[-2:]}; got {primary_beam.shape}."
            )
        planes = planes * primary_beam[:, None]

    img_xds = xr.Dataset(
        {"SKY": (("time", "frequency", "polarization", "l", "m"), planes[None])},
        coords={
            "time": [0],
            "frequency": frequency,
            "polarization": stokes,
            "l": sky.l.values,
            "m": sky.m.values,
        },
        attrs={"type": "image_dataset", "data_groups": {"model": {"sky": "SKY"}}},
    )
    img_xds = transform_polarization_basis(img_xds, new_polarization_basis=basis)
    correlations = [str(p) for p in img_xds.polarization.values]
    if correlations != polarization:
        missing = sorted(set(polarization) - set(correlations))
        if missing:
            raise ValueError(
                f"the Stokes planes {stokes} give the correlations {correlations}, "
                f"not the simulated {polarization}."
            )
        img_xds = img_xds.sel(polarization=polarization)

    n_time, n_baseline = uvw.shape[:2]
    ms_xds = xr.Dataset(
        {
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.zeros(
                    (n_time, n_baseline, frequency.size, len(polarization)),
                    np.complex128,
                ),
            ),
            "UVW": (
                ("time", "baseline_id", "uvw_label"),
                np.ascontiguousarray(uvw, dtype=np.float64),
            ),
        },
        coords={
            "time": np.arange(n_time),
            "baseline_id": np.arange(n_baseline),
            "frequency": frequency,
            "polarization": polarization,
            "uvw_label": ["u", "v", "w"],
        },
    )
    ms_xds.attrs["data_groups"] = {}
    modify_data_groups_xds(
        ms_xds,
        data_group_out_name="base",
        data_group_out={"correlated_data": "VISIBILITY", "uvw": "UVW"},
        description="Sky image prediction samples.",
    )

    if predict == "dft":
        from astroviper.processing_functions.imaging.direct_fourier_transform import (
            predict_dft_visibilities_single_field,
        )

        predict_dft_visibilities_single_field(
            ms_xds,
            img_xds,
            ms_data_group_in_name="base",
            ms_data_group_out_name="model",
            image_data_group_in_name="model",
            chan_mode="cube",
            processing_function_threads=processing_function_threads,
        )
    else:
        from astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder import (
            fft_norm_img_xds,
        )
        from astroviper.processing_functions.imaging.get_visibility_grid import (
            get_visibility_grid_single_field,
        )
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
            make_gridding_kernel,
        )

        kernel = make_gridding_kernel(gridding_kernel_params, fft_padding)
        l, m = img_xds.l.values, img_xds.m.values
        image_params = {
            "image_size": [l.size, m.size],
            "cell_size": np.array([l[1] - l[0], m[1] - m[0]]),
            "fft_padding": fft_padding,
        }
        img_xds = fft_norm_img_xds(
            img_xds,
            image_params=image_params,
            image_data_group_in_name="model",
            image_data_group_out_name="model",
            image_data_group_out_modified={"visibility": "VISIBILITY_MODEL"},
            image_data_variables_keep=["sky"],
            processing_function_threads=processing_function_threads,
            complex_dtype=np.complex128,
            gridding_kernel=kernel,
        )
        get_visibility_grid_single_field(
            ms_xds,
            kernel,
            img_xds,
            ms_data_group_in_name="base",
            ms_data_group_out_name="model",
            ms_data_group_out_modified={"correlated_data": "VISIBILITY_MODEL"},
            image_data_group_in_name="model",
            chan_mode="cube",
            fft_padding=fft_padding,
            processing_function_threads=processing_function_threads,
        )
    return np.asarray(ms_xds["VISIBILITY_MODEL"].values, dtype=np.complex128)
