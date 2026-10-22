"""Taylor-weighted point-spread-function creation for MT-MFS continuum imaging.

This module mirrors the cube-imaging PSF path, but grids a stack of Taylor-weighted
UV-sampling functions.  For ``n_taylor_terms`` sky-model Taylor coefficients, the MT-MFS
normal equations require ``2 * n_taylor_terms - 1`` PSF/Hessian Taylor orders.

For a channel frequency ``nu`` and a shared reference frequency ``nu0``,

    x = (nu - nu0) / nu0

the UV-sampling function for PSF Taylor order ``k`` is gridded with the effective
weight

    weight_imaging * x**k.

All PSF Taylor orders use the zeroth-order sum of weights for normalization.  In
particular, odd Taylor orders must not be normalized by their own signed sum,
which can be close to zero.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import xarray as xr

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.param_docs import shares_param_docs


def _validate_mtmfs_parameters(
    image_params: dict,
) -> tuple[int, int, float]:
    """Return validated ``n_taylor_terms``, number of PSF terms, and reference frequency."""
    if "n_taylor_terms" not in image_params:
        raise KeyError("image_params must contain 'n_taylor_terms' for MT-MFS imaging.")
    if "reference_frequency" not in image_params:
        raise KeyError(
            "image_params must contain one globally shared 'reference_frequency'."
        )

    n_taylor_terms = int(image_params["n_taylor_terms"])
    if n_taylor_terms < 1:
        raise ValueError(
            f"n_taylor_terms must be at least 1; received {n_taylor_terms}."
        )

    reference_frequency = float(image_params["reference_frequency"])
    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "reference_frequency must be finite and positive; "
            f"received {reference_frequency}."
        )

    return n_taylor_terms, 2 * n_taylor_terms - 1, reference_frequency


def add_uv_sampling_grid_continuum_single_field(
    ms_xdt: xr.Dataset,
    cgk_1D: np.ndarray,
    img_xds: xr.Dataset,
    *,
    n_psf_taylor_terms: int,
    reference_frequency: float,
    ms_data_group_in_name: str = "base",
    image_data_group_in_name: str = "residual",
    image_data_group_out_name: str = "residual",
    image_data_group_out_modified: dict | None = None,
    overwrite: bool = True,
    fft_padding: float = 1.2,
    processing_function_threads: int = 1,
    gcf_support: int = 7,
    gcf_oversampling: int = 100,
    complex_dtype=None,
) -> None:
    """Accumulate Taylor-weighted UV-sampling grids for one measurement set.

    The output arrays have dimensions

    ``(time, psf_taylor_order, polarization, u, v)``

    and

    ``(time, psf_taylor_order, polarization)``.

    The C++ cube gridder is reused one Taylor order at a time. All channels map
    directly into one Taylor plane of the output grid, while their imaging weights
    are multiplied by ``x**k`` for Taylor order ``k``.

    Parameters
    ----------
    ms_xdt : xarray.Dataset
        One Measurement Set partition with visibility coordinates and imaging
        weights.
    cgk_1D : numpy.ndarray
        Oversampled one-dimensional gridding convolution kernel.
    img_xds : xarray.Dataset
        Image dataset whose UV-sampling arrays are updated in place.
    n_psf_taylor_terms : int
        Number of Taylor PSF/Hessian orders to grid.
    reference_frequency : float
        Taylor-expansion reference frequency in Hz.
    ms_data_group_in_name : str, optional
        Measurement Set data group registering the input weights.
    image_data_group_in_name : str, optional
        Existing image data group used as the output template.
    image_data_group_out_name : str, optional
        Image data group in which to register the UV-sampling arrays.
    image_data_group_out_modified : dict or None, optional
        Role-to-variable-name overrides for the output data group.
    overwrite : bool, optional
        Permit replacement of existing registered output variables.
    fft_padding : float, optional
        Image-to-UV-grid padding factor.
    processing_function_threads : int, optional
        Threads supplied to the C++ gridder.
    gcf_support : int, optional
        Convolution-kernel support in grid cells.
    gcf_oversampling : int, optional
        Convolution-kernel oversampling factor.
    complex_dtype : numpy.dtype or None, optional
        Complex dtype of the accumulated UV grids.

    Notes
    -----
    ``UV_SAMPLING_NORMALIZATION`` is repeated along ``psf_taylor_order`` but every
    entry contains the zeroth-order sum of weights.  This gives all Hessian PSFs a
    common normalization and avoids dividing odd Taylor orders by a signed sum
    that may be nearly zero.
    """
    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "uv_sampling": "UV_SAMPLING",
            "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
        }

    if n_psf_taylor_terms < 1:
        raise ValueError(
            f"n_psf_taylor_terms must be at least 1; received {n_psf_taylor_terms}."
        )
    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "reference_frequency must be finite and positive; "
            f"received {reference_frequency}."
        )

    if complex_dtype is None:
        complex_dtype = np.complex128

    output_mapping = copy.deepcopy(image_data_group_out_modified)
    ms_data_group_in = ms_xdt.attrs["data_groups"][ms_data_group_in_name]

    _, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=output_mapping,
        overwrite=overwrite,
    )

    weight_imaging = np.asarray(ms_xdt[ms_data_group_in["weight_imaging"]].values)
    if weight_imaging.ndim != 4:
        raise ValueError(
            "The imaging-weight array must have dimensions "
            "(time, baseline, frequency, polarization); "
            f"received shape {weight_imaging.shape}."
        )

    n_time, _, n_chan, n_pol = weight_imaging.shape
    frequency_coord = np.asarray(ms_xdt.frequency.values, dtype=np.float64)

    if frequency_coord.ndim != 1 or frequency_coord.size != n_chan:
        raise ValueError(
            "The frequency coordinate must be one-dimensional and match the "
            f"weight-imaging channel axis: {frequency_coord.shape} versus {n_chan}."
        )

    # The current single-field image gridder maps all input times into one image
    # time plane, matching the cube implementation.
    n_image_time = 1
    time_map = np.zeros(n_time, dtype=np.int64)
    frequency_map = np.zeros(n_chan, dtype=np.int64)
    pol_map = np.arange(n_pol, dtype=np.int64)

    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    n_uv = padded_grid_size(
        [img_xds.sizes["l"], img_xds.sizes["m"]],
        fft_padding,
    )
    delta_lm = img_xds.xr_img.get_lm_cell_size()

    uv_name = image_data_group_out["uv_sampling"]
    norm_name = image_data_group_out["uv_sampling_normalization"]

    if uv_name not in img_xds:
        img_xds[uv_name] = xr.DataArray(
            np.zeros(
                (
                    n_image_time,
                    n_psf_taylor_terms,
                    n_pol,
                    n_uv[0],
                    n_uv[1],
                ),
                dtype=complex_dtype,
            ),
            dims=[
                "time",
                "psf_taylor_order",
                "polarization",
                "u",
                "v",
            ],
            coords={
                "psf_taylor_order": np.arange(
                    n_psf_taylor_terms,
                    dtype=np.int64,
                )
            },
        )

        img_xds[norm_name] = xr.DataArray(
            np.zeros(
                (n_image_time, n_psf_taylor_terms, n_pol),
                dtype=np.float64,
            ),
            dims=[
                "time",
                "psf_taylor_order",
                "polarization",
            ],
            coords={
                "psf_taylor_order": np.arange(
                    n_psf_taylor_terms,
                    dtype=np.int64,
                )
            },
        )

        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            image_data_group_out,
            description=(
                "Added MT-MFS Taylor-weighted UV-sampling grids with "
                "add_uv_sampling_grid_continuum_single_field."
            ),
        )
    else:
        expected_grid_shape = (
            n_image_time,
            n_psf_taylor_terms,
            n_pol,
            n_uv[0],
            n_uv[1],
        )
        if img_xds[uv_name].shape != expected_grid_shape:
            raise ValueError(
                f"Existing {uv_name} has shape {img_xds[uv_name].shape}, "
                f"but {expected_grid_shape} is required."
            )

    target_grid = img_xds[uv_name].values
    target_normalization = img_xds[norm_name].values

    uvw = np.asarray(ms_xdt[ms_data_group_in["uvw"]].values)

    x_frequency = (frequency_coord - reference_frequency) / reference_frequency

    from astroviper.processing_functions.imaging.gridders.anti_alias_grid_cpp import (
        anti_alias_grid_uv_sampling,
    )

    # Reuse the existing continuum channel map. With one image-time plane, each
    # Taylor-order view has the contiguous layout required by the C++ gridder.
    for taylor_order in range(n_psf_taylor_terms):
        taylor_factor = np.power(x_frequency, taylor_order)

        if (
            taylor_order == 0
            and weight_imaging.dtype == np.float64
            and weight_imaging.flags.c_contiguous
        ):
            effective_weight = weight_imaging
        else:
            effective_weight = np.ascontiguousarray(
                weight_imaging * taylor_factor[np.newaxis, np.newaxis, :, np.newaxis]
            )

        grid_view = target_grid[:, taylor_order : taylor_order + 1, ...]
        temporary_normalization = np.zeros(
            (n_image_time, 1, n_pol),
            dtype=np.float64,
        )

        anti_alias_grid_uv_sampling(
            grid_view,
            temporary_normalization,
            uvw,
            frequency_coord,
            frequency_map,
            time_map,
            pol_map,
            effective_weight,
            cgk_1D,
            n_uv,
            delta_lm,
            support=gcf_support,
            oversampling=gcf_oversampling,
            processing_function_threads=processing_function_threads,
        )

        # Only order zero defines the common MT-MFS normalization.
        if taylor_order == 0:
            target_normalization[:, 0, ...] += temporary_normalization[:, 0, ...]

    # Every Taylor PSF is normalized by the zeroth-order sum of weights.
    target_normalization[...] = target_normalization[:, 0:1, :]


@shares_param_docs
def make_point_spread_function_mvc_single_field(
    ps_xdt,
    img_xds,
    image_params,
    ms_data_group_in_name="base",
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    gcf_oversampling=100,
    gcf_support=7,
    processing_function_threads=1,
    complex_dtype=None,
):
    """Grid a frequency-resolved PSF cube for an MVC map partition.

    Unlike direct MFS, MVC does not apply Taylor weights during gridding.  Channels use the shared cube nearest-channel policy. Map tasks subsequently
    FFT these planes and form ``2*n_taylor_terms-1`` Taylor Hessian contributions;
    reduction combines the contributions before global normalization.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Processing Set partition containing prepared imaging weights.
    img_xds : xarray.Dataset
        Frequency-resolved MVC image dataset updated with UV-sampling grids.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
        Continuum imaging also requires ``frequency_coords``,
        ``n_taylor_terms`` (CASA ``nterms``), and ``reference_frequency`` in Hz;
        ``reference_frequency_hz`` is a deprecated alias. ``spectral_reference``
        sets the spectral frame (default ``"lsrk"``). ``primary_beam_model``
        selects ``auto``, ``airy`` or ``casa_airy``; ``list_dish_diameters`` and
        ``list_blockage_diameters`` are arrays in metres,
        ``primary_beam_max_radius_1ghz`` is a radius in radians at 1 GHz, and
        ``primary_beam_ipower`` is the exponent applied to the voltage pattern
        (default 2, the power beam). These primary-beam keys override the inferred
        beam prescription.
    ms_data_group_in_name : str, optional
        Measurement Set data group registering the imaging weights.
    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    image_data_group_out_name : str, optional
        Output image data-group name.
    gcf_oversampling : int, optional
        Convolution-kernel oversampling factor.
    gcf_support : int, optional
        Convolution-kernel support in grid cells.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    complex_dtype : numpy.dtype or None, optional
        Complex dtype of the UV-sampling grids.

    Returns
    -------
    img_xds : xarray.Dataset
        Dataset containing frequency-resolved ``UV_SAMPLING`` and
        ``UV_SAMPLING_NORMALIZATION`` arrays.
    return_df : pandas.DataFrame
        One-row timing dataframe for kernel creation, visibility filtering, and
        UV-sampling gridding.
    """
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )
    from astroviper.processing_functions.imaging.make_point_spread_function import (
        add_uv_sampling_grid_single_field,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations
    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        map_visibility_frequencies_to_image,
    )

    # Validate the shared cube nearest-channel policy before gridding.
    for ms_xdt in ps_xdt.values():
        map_visibility_frequencies_to_image(
            ms_xdt.frequency.values, img_xds.frequency.values, matching="nearest"
        )

    if complex_dtype is None:
        complex_dtype = np.complex128

    start = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(
        gcf_oversampling,
        gcf_support,
    )
    T_gcf = time.time() - start

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0

    for _, ms_xdt in ps_xdt.items():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_uv_sampling_grid_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_group_out_modified={
                "uv_sampling": "UV_SAMPLING",
                "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
            },
            overwrite=True,
            chan_mode="cube",
            fft_padding=image_params["fft_padding"],
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )
        T_uv_sampling_grid += time.time() - start

    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [T_uv_sampling_grid],
        }
    )

    return img_xds, return_df


@shares_param_docs
def make_point_spread_function_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    n_taylor_terms=None,
    reference_frequency=None,
    ms_data_group_in_name="base",
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    image_data_variables_keep=None,
    gcf_oversampling=100,
    gcf_support=7,
    processing_function_threads=1,
    fft_backend="pyfftw",
    complex_dtype=None,
):
    """Build the MT-MFS Taylor UV-sampling stack for one continuum-imaging chunk.

    For ``n_taylor_terms`` sky-model Taylor coefficients, this function produces

    ``2 * n_taylor_terms - 1``

    PSF/Hessian Taylor orders.  Each measurement-set channel is weighted by
    ``((nu - nu0) / nu0)**k`` before gridding order ``k``.  The reference
    frequency ``nu0`` must be supplied globally in ``image_params`` and must be
    identical for every distributed frequency chunk.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Processing Set partition containing prepared imaging weights.
    img_xds : xarray.Dataset
        Continuum image dataset updated with Taylor UV-sampling grids.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
        Continuum imaging also requires ``frequency_coords``,
        ``n_taylor_terms`` (CASA ``nterms``), and ``reference_frequency`` in Hz;
        ``reference_frequency_hz`` is a deprecated alias. ``spectral_reference``
        sets the spectral frame (default ``"lsrk"``). ``primary_beam_model``
        selects ``auto``, ``airy`` or ``casa_airy``; ``list_dish_diameters`` and
        ``list_blockage_diameters`` are arrays in metres,
        ``primary_beam_max_radius_1ghz`` is a radius in radians at 1 GHz, and
        ``primary_beam_ipower`` is the exponent applied to the voltage pattern
        (default 2, the power beam). These primary-beam keys override the inferred
        beam prescription.
    n_taylor_terms : int or None, optional
        Number of sky-model Taylor coefficients; defaults to ``image_params``.
    reference_frequency : float or None, optional
        Taylor reference frequency in Hz; defaults to ``image_params``.
    ms_data_group_in_name : str, optional
        Measurement Set data group registering the imaging weights.
    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    image_data_group_out_name : str, optional
        Output image data-group name.
    image_data_variables_keep : list of str or None, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    gcf_oversampling : int, optional
        Convolution-kernel oversampling factor.
    gcf_support : int, optional
        Convolution-kernel support in grid cells.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    complex_dtype : numpy.dtype or None, optional
        Complex dtype of the UV-sampling grids.

    Returns
    -------
    img_xds : xarray.Dataset
        Image dataset containing ``UV_SAMPLING``,
        ``UV_SAMPLING_NORMALIZATION`` with a
        ``psf_taylor_order`` dimension.
    return_df : pandas.DataFrame
        One-row timing frame with ``T_gcf``, ``T_vis_mask``,
        and ``T_uv_sampling_grid``.
    """
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    if complex_dtype is None:
        complex_dtype = np.complex128
    if image_data_variables_keep is None:
        image_data_variables_keep = []

    # Work on a local copy so that this function does not modify the caller's
    # image-parameter dictionary.
    image_params = copy.deepcopy(image_params)

    # Support both explicit keyword arguments and values stored in image_params.
    if n_taylor_terms is not None:
        image_params["n_taylor_terms"] = int(n_taylor_terms)

    if reference_frequency is not None:
        image_params["reference_frequency"] = float(reference_frequency)

    _, n_psf_taylor_terms, reference_frequency = _validate_mtmfs_parameters(
        image_params
    )

    start = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(
        gcf_oversampling,
        gcf_support,
    )
    T_gcf = time.time() - start

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0

    for _, ms_xdt in ps_xdt.items():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_uv_sampling_grid_continuum_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            n_psf_taylor_terms=n_psf_taylor_terms,
            reference_frequency=reference_frequency,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_group_out_modified={
                "uv_sampling": "UV_SAMPLING",
                "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
            },
            overwrite=True,
            fft_padding=image_params["fft_padding"],
            processing_function_threads=processing_function_threads,
            gcf_support=gcf_support,
            gcf_oversampling=gcf_oversampling,
            complex_dtype=complex_dtype,
        )
        T_uv_sampling_grid += time.time() - start

    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [T_uv_sampling_grid],
            # "T_fft_norm": [T_fft_norm],
        }
    )

    return img_xds, return_df
