"""Point-spread-function creation for the single-field cube imager.

This module also owns the UV-sampling gridders (``add_uv_sampling_grid_*``) that
build the PSF numerator: gridding the imaging weights onto the UV plane *is* the
first step of forming the point spread function.
"""

import copy
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.param_docs import shares_param_docs

if TYPE_CHECKING:
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        GriddingKernel,
    )


def add_uv_sampling_grid_mosaic(
    ms_xds: xr.Dataset,
    gcf_xds: xr.Dataset,
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    image_data_group_in_name: str = "mosaic",
    image_data_group_out_name: str = "mosaic",
    image_data_group_out_modified: dict | None = None,
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
    add_uv_sampling_grid_single_field : Non-mosaic (anti-aliasing gridder) variant.
    """
    from astroviper.processing_functions.imaging.gridders.mosaic_grid import (
        mosaic_grid_jit,
    )

    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "uv_sampling": "UV_SAMPLING",
            "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
        }
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
    cgk_1D: "GriddingKernel | np.ndarray",
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    image_data_group_in_name: str = "residual",
    image_data_group_out_name: str = "residual",
    image_data_group_out_modified: dict | None = None,
    overwrite: bool = True,
    chan_mode: str = "cube",
    fft_padding: float = 1.2,
    processing_function_threads: int = 1,
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
    :func:`add_uv_sampling_grid_mosaic`.  It uses the separable anti-aliasing gridder (the standard gridder in CASA) rather than the
    direction-dependent mosaic gridder, so no GCF dataset is required.

    Parameters
    ----------
    ms_xdt : xr.Dataset
        Measurement set dataset.  Must contain the data variables referenced
        by ``ms_data_group_in_name`` (``uvw``, ``weight_imaging``) and a
        ``frequency`` coordinate.
    cgk_1D : GriddingKernel or np.ndarray
        The gridding kernel
        (:class:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel.GriddingKernel`),
        or a bare table of CASA's prolate spheroidal kernel of support 7.
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
        Channel mapping mode.  ``"cube"`` maps each input channel to the image
        channel nearest in frequency (see
        :func:`~astroviper.processing_functions.imaging.utils.frequency_mapping.map_visibility_frequencies_to_image`);
        ``"continuum"`` collapses all input channels onto a single image
        channel.
    fft_padding : float, default ``1.2``
        Padding factor applied to the image size when computing the UV-grid
        dimensions: ``n_uv = fft_padding * [img_xds.sizes["l"], img_xds.sizes["m"]]``.
        Values greater than ``1.0`` reduce aliasing from the FFT.
    processing_function_threads : int, default ``1``
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
    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "uv_sampling": "UV_SAMPLING",
            "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
        }
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
        from astroviper.processing_functions.imaging.utils.frequency_mapping import (
            map_visibility_frequencies_to_image,
        )

        n_imag_chan = img_xds.sizes["frequency"]
        frequency_map = map_visibility_frequencies_to_image(
            ms_xdt.frequency.values,
            img_xds.frequency.values,
        )
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

    from astroviper.processing_functions.imaging.gridders.anti_alias_grid_cpp import (
        anti_alias_grid_uv_sampling,
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        as_gridding_kernel,
    )

    kernel = as_gridding_kernel(cgk_1D)
    anti_alias_grid_uv_sampling(
        grid,
        normalization,
        uvw,
        frequency_coord,
        frequency_map,
        time_map,
        pol_map,
        imaging_weight,
        kernel.table,
        n_uv,
        delta_lm,
        support=kernel.support,
        oversampling=kernel.oversampling,
        interpolation=kernel.interpolation_code,
        processing_function_threads=processing_function_threads,
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
    gridding_kernel_params=None,
    processing_function_threads=1,
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

    With ``gridding_kernel_params={"type": "dft"}`` the PSF is the exact direct
    Fourier transform of the imaging weights, w term included, over the
    unpadded image
    (:func:`~astroviper.processing_functions.imaging.direct_fourier_transform.make_dft_point_spread_function_single_field`);
    no ``UV_SAMPLING`` grid is made.

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
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
    ms_data_group_in_name : str, optional
        Measurement-set data group that supplies the weights/uvw used for the
        UV-sampling grid.  Default ``"base"``.
    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    image_data_group_out_name : str, optional
        Image data group that the ``UV_SAMPLING`` and ``POINT_SPREAD_FUNCTION``
        variables are registered under.  Default ``"residual"``.
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    gridding_kernel_params : dict, optional
        The gridding convolution kernel the gridder and degridder use:
        ``type`` (``"prolate_spheroidal_casa"``, Schwab's rational
        approximation of the ``m = 6``, ``alpha = 1`` function with its
        analytic correction, the kernel CASA uses; ``"prolate_spheroidal"``,
        the generalised prolate spheroidal function of Schwab 1984 with its
        bandwidth and weight exponent chosen for the image region the
        ``fft_padding`` keeps; ``"exponential_semicircle"``, the modified
        exponential of a semicircle of Barnett et al. 2019 and Arras et al.
        2021 shaped for that region; or
        ``"least_misfit"``, the table that minimises the map error over that
        region for the lookup used, after Ye et al. 2020), ``support`` (the
        odd number of grid points the kernel spans along one axis; 7 for the
        ``prolate_spheroidal_casa`` kernel, any odd number from 3 for the others),
        ``oversampling`` (samples of the kernel table per uv cell) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, so the lookup error,
        and with it the floor the imaging cycle converges to, falls as 1 over
        the oversampling; ``"linear"`` interpolates the two neighbouring
        samples and the error falls as 1 over its square). Missing keys take
        the defaults ``{"type": "least_misfit", "support": 9,
        "oversampling": 300, "interpolation": "linear"}``, the kernel RADPS
        Memo 13a selects for kernel errors of at most 0.1 sigma at a dynamic
        range of 1e4 (its table for ``fft_padding`` 1.2 ships with the
        package, other paddings solve it at run time); ``None`` is all
        defaults. With the type ``"prolate_spheroidal_casa"`` missing keys
        take CASA's values instead, so that type alone is CASA's kernel and
        sampling, ``{"type": "prolate_spheroidal_casa", "support": 7,
        "oversampling": 100, "interpolation": "nearest"}``, with Schwab's
        analytic correction. A ``prolate_spheroidal``,
        ``exponential_semicircle`` or ``least_misfit`` kernel whose image
        plane correction falls below 1e-6 of its centre value at the edge of
        the image kept (a support too large for the ``fft_padding``, e.g. 17
        at 1.05) is refused. The kernel of the
        former ``gridder="prolate_spheroidal"`` is ``"prolate_spheroidal_casa"``,
        not ``"prolate_spheroidal"``. ``{"type": "dft"}`` (no other key) replaces
        the gridding and the FFTs by the exact direct Fourier transform over the
        unpadded image, w term included: with the phase
        ``2 pi (u l + v m + w (n - 1))`` and ``n = sqrt(1 - l**2 - m**2)``, the
        model visibilities are ``sum M exp(+i phase) / n`` over the model pixels
        and the image is ``n sum W V exp(-i phase) / sum W`` over the samples,
        the measurement equation of ``simulate_processing_set``. Its cost is
        proportional to the samples times the image pixels (30 to 40 s per
        residual update of 451,500 samples, four correlations and 160 x 160
        pixels on one core), so give it several
        ``processing_function_threads``; its results do not depend on their
        number. Its ``VISIBILITY_NORMALIZATION`` and
        ``UV_SAMPLING_NORMALIZATION`` hold the sum of the imaging weights, the
        gridders' that sum times the kernel's stencil sums.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
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

    from astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder import (
        ifft_norm_img_xds,
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        is_direct_fourier_transform,
        make_gridding_kernel,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    if complex_dtype is None:
        complex_dtype = np.complex128
    if image_data_variables_keep is None:
        image_data_variables_keep = []

    if is_direct_fourier_transform(gridding_kernel_params):
        from astroviper.processing_functions.imaging.direct_fourier_transform import (
            make_dft_point_spread_function_single_field,
        )

        return make_dft_point_spread_function_single_field(
            ps_xdt,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_out_name,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )

    T_start_gcf = time.time()
    cgk_1D = make_gridding_kernel(gridding_kernel_params, image_params["fft_padding"])
    T_gcf = time.time() - T_start_gcf

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    for ms_xdt in ps_xdt.values():
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
            processing_function_threads=processing_function_threads,
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
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
        gridding_kernel=cgk_1D,
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


@shares_param_docs
def make_subtraction_point_spread_function_single_field(
    ps_xdt,
    img_xds,
    image_params,
    correlation_coords,
    ms_data_group_in_name="base",
    gridding_kernel_params=None,
    processing_function_threads=1,
    fft_backend="pyfftw",
    complex_dtype=None,
):
    """Build the PSF of twice the image size that exact PSF subtraction uses.

    Makes the PSF with :func:`make_point_spread_function_single_field` on an
    empty image of ``2 n_l x 2 n_m`` pixels with the cell size, gridding kernel
    and ``fft_padding`` of the image, from the imaging weights (``WEIGHT_IMAGING``)
    already in ``ps_xdt``, so it is the response of exactly the weights the
    residual images are made with. The weights are not recomputed: Briggs
    weights depend on the uv cell, which the larger grid halves. The peak lies
    at ``(n_l, n_m)``, the centre the Hogbom kernels take for a PSF of twice the
    image size; the ``n_l x n_m`` pixels from ``(n_l - n_l // 2, n_m - n_m // 2)``
    on are the image-sized PSF up to the gridding errors of the two: the lookup
    of the kernel table, on uv cells that differ by a factor of two, and the
    aliasing at the edge of the padded grid, which lies at the edge of the image
    for the image-sized PSF and twice as far out for this one. With the direct
    Fourier transform (``gridding_kernel_params={"type": "dft"}``) those
    pixels are the image-sized PSF to rounding.

    Imaging weights equal in every correlation (``NaN`` at the same samples),
    as the imaging weights of the single-field imager are, give every
    correlation the same uv sampling grid and so the same PSF, to the bit.
    The PSF is then made once, for the first correlation, and copied to the
    others: the uv grid of the ``2 n_l x 2 n_m`` image, the largest array
    while the PSF is made, holds one correlation instead of all.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set holding the imaging weights of the image. Its
        auto-correlations are dropped in place.
    img_xds : xarray.Dataset
        Image whose ``l``, ``m`` sizes and ``frequency`` coordinate the PSF
        follows. Not modified.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
    correlation_coords : sequence of str
        The correlations of the imaging weights, in their order: the PSF
        planes, as the image-sized PSF has them before the transform to the
        Stokes basis (which leaves PSF planes unchanged).
    ms_data_group_in_name : str, optional
        Measurement-set data group that supplies the weights/uvw used for the
        UV-sampling grid.  Default ``"base"``.
    gridding_kernel_params : dict, optional
        The gridding convolution kernel the gridder and degridder use:
        ``type`` (``"prolate_spheroidal_casa"``, Schwab's rational
        approximation of the ``m = 6``, ``alpha = 1`` function with its
        analytic correction, the kernel CASA uses; ``"prolate_spheroidal"``,
        the generalised prolate spheroidal function of Schwab 1984 with its
        bandwidth and weight exponent chosen for the image region the
        ``fft_padding`` keeps; ``"exponential_semicircle"``, the modified
        exponential of a semicircle of Barnett et al. 2019 and Arras et al.
        2021 shaped for that region; or
        ``"least_misfit"``, the table that minimises the map error over that
        region for the lookup used, after Ye et al. 2020), ``support`` (the
        odd number of grid points the kernel spans along one axis; 7 for the
        ``prolate_spheroidal_casa`` kernel, any odd number from 3 for the others),
        ``oversampling`` (samples of the kernel table per uv cell) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, so the lookup error,
        and with it the floor the imaging cycle converges to, falls as 1 over
        the oversampling; ``"linear"`` interpolates the two neighbouring
        samples and the error falls as 1 over its square). Missing keys take
        the defaults ``{"type": "least_misfit", "support": 9,
        "oversampling": 300, "interpolation": "linear"}``, the kernel RADPS
        Memo 13a selects for kernel errors of at most 0.1 sigma at a dynamic
        range of 1e4 (its table for ``fft_padding`` 1.2 ships with the
        package, other paddings solve it at run time); ``None`` is all
        defaults. With the type ``"prolate_spheroidal_casa"`` missing keys
        take CASA's values instead, so that type alone is CASA's kernel and
        sampling, ``{"type": "prolate_spheroidal_casa", "support": 7,
        "oversampling": 100, "interpolation": "nearest"}``, with Schwab's
        analytic correction. A ``prolate_spheroidal``,
        ``exponential_semicircle`` or ``least_misfit`` kernel whose image
        plane correction falls below 1e-6 of its centre value at the edge of
        the image kept (a support too large for the ``fft_padding``, e.g. 17
        at 1.05) is refused. The kernel of the
        former ``gridder="prolate_spheroidal"`` is ``"prolate_spheroidal_casa"``,
        not ``"prolate_spheroidal"``. ``{"type": "dft"}`` (no other key) replaces
        the gridding and the FFTs by the exact direct Fourier transform over the
        unpadded image, w term included: with the phase
        ``2 pi (u l + v m + w (n - 1))`` and ``n = sqrt(1 - l**2 - m**2)``, the
        model visibilities are ``sum M exp(+i phase) / n`` over the model pixels
        and the image is ``n sum W V exp(-i phase) / sum W`` over the samples,
        the measurement equation of ``simulate_processing_set``. Its cost is
        proportional to the samples times the image pixels (30 to 40 s per
        residual update of 451,500 samples, four correlations and 160 x 160
        pixels on one core), so give it several
        ``processing_function_threads``; its results do not depend on their
        number. Its ``VISIBILITY_NORMALIZATION`` and
        ``UV_SAMPLING_NORMALIZATION`` hold the sum of the imaging weights, the
        gridders' that sum times the kernel's stencil sums.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    complex_dtype : numpy.dtype, optional
        Complex precision of the gridded data: ``complex64`` gives a float32
        PSF, ``complex128`` (the default) a float64 one.

    Returns
    -------
    point_spread_function : numpy.ndarray
        C-contiguous ``(n_time, n_frequency, n_correlation, 2 n_l, 2 n_m)``
        PSF, peak 1 at ``(n_l, n_m)``.
    return_df : pandas.DataFrame
        The one-row timing frame of
        :func:`make_point_spread_function_single_field`.
    """
    from xradio.image import make_empty_sky_image

    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    for ms_xdt in ps_xdt.values():
        drop_auto_correlations(ms_xdt)
    n_correlation = len(correlation_coords)
    made_once = n_correlation > 1 and _correlation_weights_equal(
        ps_xdt, ms_data_group_in_name
    )
    if made_once:
        ps_xdt = _first_correlation(ps_xdt, ms_data_group_in_name)
        correlation_coords = correlation_coords[:1]

    image_size = [2 * img_xds.sizes["l"], 2 * img_xds.sizes["m"]]
    psf_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_size,
        cell_size=image_params["cell_size"],
        frequency_coords=img_xds.frequency.values,
        pol_coords=correlation_coords,
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )
    psf_xds.attrs["type"] = "image_dataset"
    psf_xds = psf_xds.xr_img.add_data_group(
        new_data_group_name="residual",
        new_data_group={"description": "PSF of twice the image size."},
    )
    psf_xds, return_df = make_point_spread_function_single_field(
        ps_xdt,
        psf_xds,
        {**image_params, "image_size": image_size},
        ms_data_group_in_name=ms_data_group_in_name,
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
        gridding_kernel_params=gridding_kernel_params,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )
    point_spread_function = psf_xds["POINT_SPREAD_FUNCTION"].values
    if made_once:
        point_spread_function = np.repeat(point_spread_function, n_correlation, axis=2)
    return point_spread_function, return_df


def _correlation_weights_equal(ps_xdt, ms_data_group_in_name):
    """Whether the imaging weights of every measurement set in ``ps_xdt`` are
    equal in every correlation (``NaN`` at the same samples), so that one uv
    sampling grid serves all correlations."""
    for ms_xdt in ps_xdt.values():
        name = ms_xdt.attrs["data_groups"][ms_data_group_in_name]["weight_imaging"]
        weight = ms_xdt[name].values
        for p in range(1, weight.shape[-1]):
            if not np.array_equal(weight[..., p], weight[..., 0], equal_nan=True):
                return False
    return True


def _first_correlation(ps_xdt, ms_data_group_in_name):
    """``{ms_name: measurement set}`` of ``ps_xdt`` (a processing set or, as
    the node task hands it over, a dict of measurement sets) with the first
    correlation alone: views of its arrays, except the imaging weights, a
    C-contiguous copy for the gridders."""
    first = {}
    for ms_name, ms_xdt in ps_xdt.items():
        ms_first = ms_xdt.isel(polarization=slice(0, 1))
        name = ms_first.attrs["data_groups"][ms_data_group_in_name]["weight_imaging"]
        ms_first[name] = ms_first[name].copy(
            data=np.ascontiguousarray(ms_first[name].values)
        )
        first[ms_name] = ms_first
    return first
