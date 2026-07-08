"""Point-spread-function creation for the single-field cube imager.

This module also owns the UV-sampling gridders (``add_uv_sampling_grid_*``) that
build the PSF numerator: gridding the imaging weights onto the UV plane *is* the
first step of forming the point spread function.
"""

import copy

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.param_docs import shares_param_docs


def add_uv_sampling_grid_mosaic(
    ms_xds: xr.Dataset,
    gcf_xds: xr.Dataset,
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    image_data_group_in_name: str = "mosaic",
    image_data_group_out_name: str = "mosaic",
    image_data_group_out_modified: dict = {
        "uv_sampling": "UV_SAMPLING",
        "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
    },
    overwrite: bool = True,
    chan_mode: str = "cube",
    fft_padding: float = 1.2,
):
    """Accumulate the UV-sampling grid for a mosaic observation into an image dataset.

    Reads imaging weights from ``ms_xds``, grids them onto the UV plane using a
    direction-dependent convolution function (GCF), and accumulates the result
    into ``img_xds``.  The UV-sampling grid and its normalization (sum of
    weights) are stored as new data variables in ``img_xds``, and a
    corresponding output data group is registered via
    :func:`~astroviper.utils.data_group_tools.modify_data_groups_xds`.

    This function is designed to be called once per measurement set in a
    mosaic loop; the grid arrays in ``img_xds`` are accumulated in place so
    that repeated calls sum contributions from multiple MSes.

    Parameters
    ----------
    ms_xds : xr.Dataset
        Measurement set dataset.  Must contain the data variables referenced
        by the input data group (``uvw``, ``weight_imaging``, ``flag``) and a
        ``frequency`` coordinate.
    gcf_xds : xr.Dataset
        Gridding convolution function dataset.  Must contain
        ``CF_BASELINE_MAP``, ``CF_CHAN_MAP``, ``CF_POL_MAP``, ``CONV_KERNEL``,
        ``SUPPORT``, and ``PHASE_GRADIENT`` data variables, as well as an
        ``oversampling`` attribute.
    img_xds : xr.Dataset
        Image dataset that accumulates the UV-sampling grid.  The arrays
        named by ``image_data_group_out_modified`` are created on the first call
        (if absent) and accumulated on subsequent calls.  Cell size and image
        dimensions are read directly from ``img_xds`` via
        ``img_xds.xr_img.get_lm_cell_size()`` and ``img_xds.sizes``.
    ms_data_group_in_name : str, default ``"base"``
        Key of the input data group in ``ms_xds.attrs["data_groups"]``.
        Must provide ``"uvw"`` and ``"weight_imaging"`` role keys.
    image_data_group_in_name : str, default ``"mosaic"``
        Key of the input data group in ``img_xds.attrs["data_groups"]``.
    image_data_group_out_name : str, default ``"mosaic"``
        Key under which the output data group is registered in
        ``img_xds.attrs["data_groups"]``.
    image_data_group_out_modified : dict, default ``{"uv_sampling": "UV_SAMPLING", "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION"}``
        Mapping of role keys to the data-variable names that will be written
        into ``img_xds``.
    overwrite : bool, default ``True``
        If ``True``, an existing output data group or output data variables
        are silently overwritten.  Defaults to ``True`` because this function
        is typically called in a loop that accumulates into the same arrays.
    chan_mode : str, default ``"cube"``
        Channel mapping mode.  ``"cube"`` maps each input channel to its own
        image channel; ``"continuum"`` collapses all input channels onto a
        single image channel.
    fft_padding : float, default ``1.2``
        Padding factor applied to the image size when computing the UV-grid
        dimensions: ``n_uv = fft_padding * [img_xds.sizes["l"], img_xds.sizes["m"]]``.
        Values greater than ``1.0`` reduce aliasing from the FFT.

    Returns
    -------
    None
        Modifies ``img_xds`` in place (data variables and ``data_groups``
        attribute); no return value.

    See Also
    --------
    add_uv_sampling_grid_single_field : Non-mosaic (standard gridder) variant.
    """
    from astroviper.processing_functions.imaging.gridders.mosaic_grid import (
        mosaic_grid_jit,
    )

    _image_data_group_out_modified = copy.deepcopy(image_data_group_out_modified)

    # Read the MS input data group directly; the MS is read-only here.
    ms_data_group_in = ms_xds.attrs["data_groups"][ms_data_group_in_name]

    # Resolve the image input and output data groups, guarding against
    # accidental overwrites according to the overwrite flag.
    _, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=_image_data_group_out_modified,
        overwrite=overwrite,
    )

    weight_imaging = ms_xds[ms_data_group_in["weight_imaging"]].values
    n_chan = weight_imaging.shape[2]

    if chan_mode == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Single continuum image collapsed across all channels.
        chan_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight_imaging.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    n_uv = padded_grid_size([img_xds.sizes["l"], img_xds.sizes["m"]], fft_padding)
    delta_lm = img_xds.xr_img.get_lm_cell_size()
    oversampling = gcf_xds.attrs["oversampling"]

    # Initialise output arrays on the first call; subsequent calls accumulate.
    if image_data_group_out["uv_sampling"] not in img_xds:
        img_xds[image_data_group_out["uv_sampling"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128),
            dims=["frequency", "polarization", "u", "v"],
        )
        img_xds[image_data_group_out["uv_sampling_normalization"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol), dtype=np.double),
            dims=["frequency", "polarization"],
        )

    grid = img_xds[image_data_group_out["uv_sampling"]].values
    sum_weight = img_xds[image_data_group_out["uv_sampling_normalization"]].values

    # vis_data is unused for PSF-only gridding (do_psf=True); a minimal
    # placeholder array is passed to satisfy the gridder's signature.
    vis_data = np.zeros((1, 1, 1, 1), dtype=bool)
    uvw = ms_xds[ms_data_group_in["uvw"]].values
    freq_chan = ms_xds.frequency.values
    imaging_weight = ms_xds[ms_data_group_in["weight_imaging"]].values

    do_psf = True

    cf_baseline_map = gcf_xds["CF_BASELINE_MAP"].values
    cf_chan_map = gcf_xds["CF_CHAN_MAP"].values
    cf_pol_map = gcf_xds["CF_POL_MAP"].values
    conv_kernel = gcf_xds["CONV_KERNEL"].values
    weight_support = gcf_xds["SUPPORT"].values
    phase_gradient = gcf_xds["PHASE_GRADIENT"].values

    mosaic_grid_jit(
        grid,
        sum_weight,
        do_psf,
        vis_data,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        cf_baseline_map,
        cf_chan_map,
        cf_pol_map,
        imaging_weight,
        conv_kernel,
        n_uv,
        delta_lm,
        weight_support,
        oversampling,
        phase_gradient,
    )

    modify_data_groups_xds(
        img_xds,
        image_data_group_out_name,
        image_data_group_out,
        description="Added UV sampling grid to img_xds with add_uv_sampling_grid_mosaic.",
    )


def add_uv_sampling_grid_single_field(
    ms_xdt: xr.Dataset,
    cgk_1D: np.ndarray,
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    image_data_group_in_name: str = "residual",
    image_data_group_out_name: str = "residual",
    image_data_group_out_modified: dict = {
        "uv_sampling": "UV_SAMPLING",
        "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
    },
    overwrite: bool = True,
    chan_mode: str = "cube",
    fft_padding: float = 1.2,
    num_threads: int = 1,
    complex_dtype=None,
):
    """Accumulate the UV-sampling grid for a single-field observation into an image dataset.

    Reads imaging weights from ``ms_xdt``, grids them onto the UV plane using a
    separable 1-D convolutional gridding kernel (``cgk_1D``), and accumulates
    the result into ``img_xds``.  The UV-sampling grid and its normalization
    (sum of weights) are stored as new data variables in ``img_xds``, and a
    corresponding output data group is registered via
    :func:`~astroviper.utils.data_group_tools.modify_data_groups_xds`.

    This function is the non-mosaic counterpart of
    :func:`add_uv_sampling_grid_mosaic`.  It uses the standard separable prolate
    spheroidal gridder rather than the direction-dependent mosaic gridder, so no
    GCF dataset is required.

    Parameters
    ----------
    ms_xdt : xr.Dataset
        Measurement set dataset.  Must contain the data variables referenced
        by ``ms_data_group_in_name`` (``uvw``, ``weight_imaging``) and a
        ``frequency`` coordinate.
    cgk_1D : np.ndarray
        1-D convolutional gridding kernel used by the standard gridder.
        Shape ``(oversampling * support,)``.
    img_xds : xr.Dataset
        Image dataset that accumulates the UV-sampling grid.  The arrays
        named by ``image_data_group_out_modified`` are created on the first call
        (if absent) and accumulated on subsequent calls.
    ms_data_group_in_name : str, default ``"base"``
        Key of the MS input data group in ``ms_xdt.attrs["data_groups"]``.
        Must provide ``"uvw"`` and ``"weight_imaging"`` role keys.
    image_data_group_in_name : str, default ``"residual"``
        Key of the image input data group in ``img_xds.attrs["data_groups"]``.
    image_data_group_out_name : str, default ``"residual"``
        Key under which the output data group is registered in
        ``img_xds.attrs["data_groups"]``.
    image_data_group_out_modified : dict, default ``{"uv_sampling": "UV_SAMPLING", "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION"}``
        Mapping of role keys to the data-variable names written into
        ``img_xds``.  ``"uv_sampling"`` stores the complex UV grid and
        ``"uv_sampling_normalization"`` stores the per-channel,
        per-polarization sum of imaging weights.
    overwrite : bool, default ``True``
        If ``True``, an existing output data group or output data variables
        are silently overwritten.  Defaults to ``True`` because this function
        is typically called in a loop that accumulates into the same arrays.
    chan_mode : str, default ``"cube"``
        Channel mapping mode.  ``"cube"`` maps each input channel to its own
        image channel; ``"continuum"`` collapses all input channels onto a
        single image channel.
    fft_padding : float, default ``1.2``
        Padding factor applied to the image size when computing the UV-grid
        dimensions: ``n_uv = fft_padding * [img_xds.sizes["l"], img_xds.sizes["m"]]``.
        Values greater than ``1.0`` reduce aliasing from the FFT.
    num_threads : int, default ``1``
        Threads handed to the C++ gridding kernel.
    complex_dtype : numpy.dtype, optional
        Complex precision of the gridded UV-sampling data (``complex64`` for a
        single-precision image, ``complex128`` otherwise).  Defaults to
        ``numpy.complex128``.

    Returns
    -------
    None
        Modifies ``img_xds`` in place (data variables and ``data_groups``
        attribute); no return value.

    See Also
    --------
    add_uv_sampling_grid_mosaic : Mosaic (direction-dependent GCF) variant.
    """
    # Deep copy so that inputs are not modified
    _image_data_group_out_modified = copy.deepcopy(image_data_group_out_modified)

    ms_data_group_in = ms_xdt.attrs["data_groups"][ms_data_group_in_name]

    image_data_group_in, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=_image_data_group_out_modified,
        overwrite=overwrite,
    )

    weight_imaging = ms_xdt[ms_data_group_in["weight_imaging"]].values
    n_chan = weight_imaging.shape[2]

    if chan_mode == "cube":
        n_imag_chan = n_chan
        frequency_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        frequency_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight_imaging.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    # Time Map #Currently not implemented.
    n_image_time = 1
    n_time = ms_xdt.sizes["time"]
    time_map = (np.zeros(n_time)).astype(int)

    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    n_uv = padded_grid_size([img_xds.sizes["l"], img_xds.sizes["m"]], fft_padding)
    delta_lm = img_xds.xr_img.get_lm_cell_size()

    if complex_dtype is None:
        complex_dtype = np.complex128

    if image_data_group_out["uv_sampling"] not in img_xds:
        img_xds[image_data_group_out["uv_sampling"]] = xr.DataArray(
            np.zeros(
                (n_image_time, n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]),
                dtype=complex_dtype,
            ),
            dims=["time", "frequency", "polarization", "u", "v"],
        )
        img_xds[image_data_group_out["uv_sampling_normalization"]] = xr.DataArray(
            np.zeros((n_image_time, n_imag_chan, n_imag_pol), dtype=np.double),
            dims=["time", "frequency", "polarization"],
        )
        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            image_data_group_out,
            description="Added UV sampling grid to img_xds with add_uv_sampling_grid_single_field.",
        )

    grid = img_xds[image_data_group_out["uv_sampling"]].values
    normalization = img_xds[image_data_group_out["uv_sampling_normalization"]].values

    uvw = ms_xdt[ms_data_group_in["uvw"]].values
    frequency_coord = ms_xdt.frequency.values
    imaging_weight = ms_xdt[ms_data_group_in["weight_imaging"]].values

    cpp_gridder = True
    if cpp_gridder:
        from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp import (
            prolate_spheroidal_grid_uv_sampling,
        )

        prolate_spheroidal_grid_uv_sampling(
            grid,
            normalization,
            uvw,
            frequency_coord,
            frequency_map,
            time_map,
            pol_map,
            imaging_weight,
            cgk_1D,
            n_uv,
            delta_lm,
            support=7,
            oversampling=100,
            num_threads=num_threads,
        )
    else:
        from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid import (
            prolate_spheroidal_grid_uv_sampling_jit,
        )

        prolate_spheroidal_grid_uv_sampling_jit(
            grid,
            normalization,
            uvw,
            frequency_coord,
            frequency_map,
            time_map,
            pol_map,
            imaging_weight,
            cgk_1D,
            n_uv,
            delta_lm,
            support=7,
            oversampling=100,
        )


@shares_param_docs
def make_point_spread_function_single_field(
    ps_xdt,
    img_xds,
    image_params,
    ms_data_group_in_name="base",
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    image_data_variables_keep=None,
    gcf_oversampling=100,
    gcf_support=7,
    num_threads=1,
    fft_backend="pyfftw",
    complex_dtype=None,
):
    """Build the point spread function for a single-field image chunk.

    Grids the UV-sampling function of every measurement set in ``ps_xdt`` onto
    the ``UV_SAMPLING`` grid (via :func:`add_uv_sampling_grid_single_field`) and
    inverse-transforms it (with prolate-spheroidal normalization) to the
    ``POINT_SPREAD_FUNCTION`` image.  Auto-correlations are dropped in place
    before gridding.  The resulting PSF is produced in whatever polarization
    basis ``img_xds`` currently carries (the correlation basis for the
    single-field imager); it is stamped with ``type="point_spread_function"`` by
    the inverse transform so a later
    :func:`~astroviper.processing_functions.image_analysis.transform_polarization_basis.transform_polarization_basis`
    leaves it untouched.

    This is the once-per-chunk PSF builder; the per-cycle visibility grid is made
    separately by
    :func:`~astroviper.processing_functions.imaging.make_undeconvolved_image.make_undeconvolved_image_single_field`.
    The Gaussian beam fit of the PSF is performed afterwards by
    :func:`~astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit.point_spread_function_gaussian_fit`.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set whose measurement sets are gridded.  Auto-correlations
        are dropped in place.
    img_xds : xarray.Dataset
        Image dataset to populate.  The data group ``image_data_group_out_name``
        must already exist.  Modified in place.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    ms_data_group_in_name : str, optional
        Measurement-set data group that supplies the weights/uvw used for the
        UV-sampling grid.  Default ``"base"``.
    image_data_group_in_name : str, optional
        Image data group read by the gridders/FFT (the same group the
        ``UV_SAMPLING`` grid accumulates into).  Default ``"residual"``.
    image_data_group_out_name : str, optional
        Image data group that the ``UV_SAMPLING`` and ``POINT_SPREAD_FUNCTION``
        variables are registered under.  Default ``"residual"``.
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    gcf_oversampling : int, optional
        Oversampling of the prolate-spheroidal gridding convolution kernel.
        Default ``100``.
    gcf_support : int, optional
        Support (in pixels) of the gridding convolution kernel.  Default ``7``.
    num_threads : int, optional
        Number of threads handed to the gridding and FFT kernels.  Default ``1``.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    complex_dtype : numpy.dtype, optional
        Complex precision of the gridded data.  Defaults to
        ``numpy.complex128``.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with ``UV_SAMPLING`` and ``POINT_SPREAD_FUNCTION``
        added.
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_gcf``, ``T_vis_mask``,
        ``T_uv_sampling_grid`` and ``T_fft_norm`` columns.

    See Also
    --------
    astroviper.processing_functions.imaging.primary_beam.make_primary_beam.make_primary_beam_single_field
    """
    import time

    import pandas as pd

    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        ifft_norm_img_xds,
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    if complex_dtype is None:
        complex_dtype = np.complex128
    if image_data_variables_keep is None:
        image_data_variables_keep = []

    T_start_gcf = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(gcf_oversampling, gcf_support)
    T_gcf = time.time() - T_start_gcf

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    for ms_name, ms_xdt in ps_xdt.items():
        T_start_vis_mask = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - T_start_vis_mask

        T_start_uv = time.time()
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
            num_threads=num_threads,
            complex_dtype=complex_dtype,
        )  # Will become the PSF.
        T_uv_sampling_grid += time.time() - T_start_uv

    T_start_fft_norm = time.time()
    img_xds = ifft_norm_img_xds(
        img_xds,
        image_params=image_params,
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
        image_data_group_out_modified={
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        },
        image_data_variables_keep=image_data_variables_keep,
        num_threads=num_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )
    T_fft_norm = time.time() - T_start_fft_norm

    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [T_uv_sampling_grid],
            "T_fft_norm": [T_fft_norm],
        }
    )

    return img_xds, return_df
