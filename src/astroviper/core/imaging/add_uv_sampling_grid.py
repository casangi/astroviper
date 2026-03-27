import numpy as np
import scipy
from scipy import constants
from numba import jit
import numba
import xarray as xr

from astroviper.core.imaging.gridders.mosaic_grid import mosaic_grid_jit
from astroviper.core.imaging.gridders.prolate_spheroidal_grid import prolate_spheroidal_grid_uv_sampling_jit
import copy

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)


def add_uv_sampling_grid_mosaic(
    ms_xds: xr.Dataset,
    gcf_xds: xr.Dataset,
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    img_data_group_in_name: str = "mosaic",
    img_data_group_out_name: str = "mosaic",
    img_data_group_out_modified: dict = {
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
        named by ``img_data_group_out_modified`` are created on the first call
        (if absent) and accumulated on subsequent calls.  Cell size and image
        dimensions are read directly from ``img_xds`` via
        ``img_xds.xr_img.get_lm_cell_size()`` and ``img_xds.sizes``.
    ms_data_group_in_name : str, default ``"base"``
        Key of the input data group in ``ms_xds.attrs["data_groups"]``.
        Must provide ``"uvw"`` and ``"weight_imaging"`` role keys.
    img_data_group_in_name : str, default ``"mosaic"``
        Key of the input data group in ``img_xds.attrs["data_groups"]``.
    img_data_group_out_name : str, default ``"mosaic"``
        Key under which the output data group is registered in
        ``img_xds.attrs["data_groups"]``.
    img_data_group_out_modified : dict, default ``{"uv_sampling": "UV_SAMPLING", "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION"}``
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
    mosaic_grid_jit : Numba-compiled mosaic gridding kernel.
    """
    _img_data_group_out_modified = copy.deepcopy(img_data_group_out_modified)

    # Read the MS input data group directly; the MS is read-only here.
    ms_data_group_in = ms_xds.attrs["data_groups"][ms_data_group_in_name]

    # Resolve the image input and output data groups, guarding against
    # accidental overwrites according to the overwrite flag.
    _, img_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=img_data_group_in_name,
        data_group_out_name=img_data_group_out_name,
        data_group_out_modified=_img_data_group_out_modified,
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

    n_uv = fft_padding * np.array([img_xds.sizes["l"], img_xds.sizes["m"]])
    delta_lm = img_xds.xr_img.get_lm_cell_size()
    oversampling = gcf_xds.attrs["oversampling"]

    # Initialise output arrays on the first call; subsequent calls accumulate.
    if img_data_group_out["uv_sampling"] not in img_xds:
        img_xds[img_data_group_out["uv_sampling"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128),
            dims=["frequency", "polarization", "u", "v"],
        )
        img_xds[img_data_group_out["uv_sampling_normalization"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol), dtype=np.double),
            dims=["frequency", "polarization"],
        )

    grid = img_xds[img_data_group_out["uv_sampling"]].values
    sum_weight = img_xds[img_data_group_out["uv_sampling_normalization"]].values

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
        img_data_group_out_name,
        img_data_group_out,
        description="Added UV sampling grid to img_xds with add_uv_sampling_grid_mosaic.",
    )


def add_uv_sampling_grid_single_field(
    ms_xdt: xr.Dataset,
    cgk_1D: np.ndarray,
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    img_data_group_in_name: str = "single_field",
    img_data_group_out_name: str = "single_field",
    img_data_group_out_modified: dict = {"uv_sampling": "UV_SAMPLING", "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION"},
    overwrite: bool = True,
    chan_mode: str = "cube",
    fft_padding: float = 1.2,
):
    """Accumulate the UV-sampling grid for a single-field observation into an image dataset.

    Reads imaging weights from ``ms_xdt``, grids them onto the UV plane using a
    separable 1-D convolutional gridding kernel (``cgk_1D``), and accumulates
    the result into ``img_xds``.  The UV-sampling grid and its normalization
    (sum of weights) are stored as new data variables in ``img_xds``, and a
    corresponding output data group is registered via
    :func:`~astroviper.utils.data_group_tools.modify_data_groups_xds`.

    This function is the non-mosaic counterpart of
    :func:`add_uv_sampling_grid_mosaic`.  It uses the standard separable
    gridder (:func:`standard_grid_jit`) rather than the direction-dependent
    mosaic gridder, so no GCF dataset is required.

    Parameters
    ----------
    ms_xdt : xr.Dataset
        Measurement set dataset.  Must contain the data variables referenced
        by ``ms_data_group_in_name`` (``uvw``, ``weight_imaging``) and a
        ``frequency`` coordinate.
    cgk_1D : np.ndarray
        1-D convolutional gridding kernel used by the standard gridder.
        Shape ``(oversampling * support,)``; passed directly to
        :func:`standard_grid_jit`.
    img_xds : xr.Dataset
        Image dataset that accumulates the UV-sampling grid.  The arrays
        named by ``img_data_group_out_modified`` are created on the first call
        (if absent) and accumulated on subsequent calls.
    ms_data_group_in_name : str, default ``"base"``
        Key of the MS input data group in ``ms_xdt.attrs["data_groups"]``.
        Must provide ``"uvw"`` and ``"weight_imaging"`` role keys.
    img_data_group_in_name : str, default ``"single_field"``
        Key of the image input data group in ``ms_xdt.attrs["data_groups"]``.
        Typically the data group written by
        :func:`~astroviper.core.imaging.calculate_imaging_weights.calculate_imaging_weights`.
    img_data_group_out_name : str, default ``"imaging"``
        Key under which the output data group is registered in
        ``img_xds.attrs["data_groups"]``.
    img_data_group_out_modified : dict, default ``{"uv_sampling": "UV_SAMPLING", "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION"}``
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

    Returns
    -------
    None
        Modifies ``img_xds`` in place (data variables and ``data_groups``
        attribute); no return value.

    See Also
    --------
    add_uv_sampling_grid_mosaic : Mosaic (direction-dependent GCF) variant.
    standard_grid_jit : Numba-compiled standard separable gridding kernel.
    """
    # Deep copy so that inputs are not modified
    _img_data_group_out_modified = copy.deepcopy(img_data_group_out_modified)
    
    ms_data_group_in = ms_xdt.attrs["data_groups"][ms_data_group_in_name]
    
    img_data_group_in, img_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=img_data_group_in_name,
        data_group_out_name=img_data_group_out_name,
        data_group_out_modified=_img_data_group_out_modified,
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

    #Time Map #Currently not implemented.
    n_image_time = 1
    n_time = ms_xdt.sizes["time"]
    time_map = (np.zeros(n_time)).astype(int)

    n_uv = (fft_padding * np.array([img_xds.sizes["l"], img_xds.sizes["m"]])).astype(int)
    delta_lm = img_xds.xr_img.get_lm_cell_size()

    if img_data_group_out["uv_sampling"] not in img_xds:
        img_xds[img_data_group_out["uv_sampling"]] = xr.DataArray(
            np.zeros((n_image_time, n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128),
            dims=["time","frequency", "polarization", "u", "v"],
        )
        img_xds[img_data_group_out["uv_sampling_normalization"]] = xr.DataArray(
            np.zeros((n_image_time, n_imag_chan, n_imag_pol), dtype=np.double),
            dims=["time","frequency", "polarization"],
        )

    grid = img_xds[img_data_group_out["uv_sampling"]].values
    normalization = img_xds[img_data_group_out["uv_sampling_normalization"]].values

    vis_data = vis_data = np.zeros((1, 1, 1, 1), dtype=bool)
    uvw = ms_xdt[ms_data_group_in["uvw"]].values
    frequency_coord = ms_xdt.frequency.values
    imaging_weight = ms_xdt[ms_data_group_in["weight_imaging"]].values

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
    
    modify_data_groups_xds(
        img_xds,
        img_data_group_out_name,
        img_data_group_out,
        description="Added UV sampling grid to img_xds with add_uv_sampling_grid_single_field.",
    )
    
