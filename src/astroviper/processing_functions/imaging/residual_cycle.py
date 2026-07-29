import time

from astroviper.processing_functions.imaging.get_visibility_grid import (
    get_visibility_grid_single_field,
)
from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def imaging_setup_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    processing_set_data_group_name="corrected",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    image_data_group_out_name="residual",
):
    """Perform the once-per-chunk imaging setup before the major-cycle loop.

    Everything here is independent of the sky model and therefore only needs to
    be computed a single time per chunk:

    * the imaging weights (calculated on the unmasked visibilities),
    * the UV-sampling grid and its inverse transform, giving the
      ``POINT_SPREAD_FUNCTION`` plus its Gaussian beam fit, and
    * the primary beam.

    The dirty image and the residual visibilities are NOT made here -- they are
    per-cycle work done by :func:`residual_cycle_cube_single_field`.  The image
    dataset is returned in the Stokes basis (the state the first residual cycle
    expects), with the residual data group created and populated with the PSF and
    primary beam.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data.  Auto-correlations are dropped in place while gridding
        the UV-sampling function.
    img_xds : xarray.Dataset
        Image dataset to populate with the PSF and primary beam.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the minor
        cycle runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    image_data_group_out_name : str, optional
        Image data group that the PSF, primary beam and residual image are
        registered under.  Default ``"residual"``.

    Returns
    -------
    img_xds : xarray.Dataset
        Image dataset in the Stokes basis with the PSF and primary beam added.
    return_df : pandas.DataFrame
        One-row timing frame for the setup step.
    """

    import numpy as np
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
        point_spread_function_gaussian_fit,
    )
    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )
    from astroviper.processing_functions.imaging.make_point_spread_function import (
        make_point_spread_function_single_field,
    )
    from astroviper.processing_functions.imaging.primary_beam.make_primary_beam import (
        make_primary_beam_single_field,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    if single_precision_image:
        float_dtype = np.float32
        complex_dtype = np.complex64
    else:
        float_dtype = np.float64
        complex_dtype = np.complex128

    ps_data_group_name = processing_set_data_group_name

    # Create the residual data group up front: the PSF, the primary beam and the
    # residual image all live in it, and gridding writes into it.
    if image_data_group_out_name not in img_xds.attrs["data_groups"]:
        img_xds.attrs["type"] = "image_dataset"
        img_xds = img_xds.xr_img.add_data_group(
            new_data_group_name=image_data_group_out_name,
            new_data_group={"description": "test", "date": "2026"},
        )
        logger.debug("img_xds size " + str(img_xds.nbytes / 1e9) + " GB")

    # Imaging weights are computed once, on the unmasked data (the auto-
    # correlations are only dropped while gridding the UV-sampling grid below).
    T_start_weight = time.time()
    calculate_imaging_weights(
        ps_xdt,
        img_xds,
        imaging_weights_params=imaging_weights_params,
        return_weight_density_grid=False,
        ms_data_group_in_name=ps_data_group_name,
        ms_data_group_out_name=ps_data_group_name,
        ms_data_group_out_modified={"weight_imaging": "WEIGHT_IMAGING"},
        processing_function_threads=processing_function_threads,
    )
    T_weights = time.time() - T_start_weight

    # Point spread function: grid the UV-sampling function and inverse-transform
    # it.  Auto-correlations are dropped in place inside this call.
    T_start_psf = time.time()
    img_xds, point_spread_function_return_df = make_point_spread_function_single_field(
        ps_xdt,
        img_xds,
        image_params,
        ms_data_group_in_name=ps_data_group_name,
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        image_data_variables_keep=image_data_variables_keep,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )
    T_make_point_spread_function = time.time() - T_start_psf

    # Primary beam (azimuthally-symmetric obscured Airy disk).  It is independent
    # of the PSF and is created before the transform to the Stokes basis.
    img_xds, primary_beam_return_df = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        float_dtype=float_dtype,
    )
    T_primary_beam = float(primary_beam_return_df["T_primary_beam"].iloc[0])

    start = time.time()
    img_xds = transform_polarization_basis(
        img_xds, new_polarization_basis="stokes", overwrite=True
    )
    T_transform_pol = time.time() - start

    start = time.time()
    img_xds = point_spread_function_gaussian_fit(
        img_xds,
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        image_data_group_out_modified={
            "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
            "max_sidelobe_point_spread_function": "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
        },
        overwrite=True,
        processing_function_threads=processing_function_threads,
    )
    T_psf_fit = time.time() - start

    import pandas as pd

    return_df = pd.DataFrame(
        {
            "T_weights": [T_weights],
            "T_make_point_spread_function": [T_make_point_spread_function],
            "T_primary_beam": [T_primary_beam],
            "T_transform_pol": [T_transform_pol],
            "T_psf_fit": [T_psf_fit],
        }
    )
    # Merge the fine-grained PSF timings (T_gcf, T_vis_mask, T_uv_sampling_grid,
    # T_fft_norm) produced by make_point_spread_function_single_field.
    return_df = pd.concat([return_df, point_spread_function_return_df], axis=1)

    return img_xds, return_df


# from memory_profiler import profile
# @profile(precision=1)
@shares_param_docs
def residual_cycle_cube_single_field(
    ps_xdt,
    img_xds,
    image_params,
    is_n_iter_0,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    image_data_group_in_name="model",
    image_data_group_out_name="residual",
    last_residual_cycle=False,
):
    """Run one residual (major) cycle.

    Degrids the current sky model, forms the residual visibilities, grids them
    and inverse-transforms the grid to a residual image.  The once-per-chunk
    setup (imaging weights, PSF, primary beam and PSF fit) is done beforehand in
    :func:`imaging_setup_single_field`, so this function only performs the
    per-cycle work.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this chunk.  Model and residual visibilities are
        written into it in place on every non-first cycle.
    img_xds : xarray.Dataset
        Image dataset holding the sky model (input) and residual image (output).
        Modified in place.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    is_n_iter_0 : bool
        ``True`` for the very first (dirty image) cycle, where there is no sky
        model to degrid yet.  ``False`` for every later cycle.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    instrument_polarization_basis : str, optional
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` (``XX``/``YY``) or ``"circular"`` (``RR``/``LL``). The
        output image is always produced in the Stokes basis.
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the minor
        cycle runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    image_data_group_in_name : str, optional
        Image data group holding the sky model that is degridded.  Default
        ``"model"``.
    image_data_group_out_name : str, optional
        Image data group that the residual image is written into.  Default
        ``"residual"``.
    last_residual_cycle : bool, optional
        Unused placeholder retained for call-site compatibility.

    Returns
    -------
    img_xds : xarray.Dataset
        Image dataset with the updated residual image (Stokes basis).
    return_df : pandas.DataFrame
        One-row timing frame for this cycle (``T_gcf``, ``T_degrid``,
        ``T_residual_vis``, ``T_grid``, ``T_fft_norm``, ``T_transform_pol`` and
        the fine-grained gridding timings).
    """

    import numpy as np
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        fft_norm_img_xds,
        ifft_norm_img_xds,
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    if single_precision_image:
        complex_dtype = np.complex64
    else:
        complex_dtype = np.complex128

    ps_data_group_name = processing_set_data_group_name

    T_start_gcf = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(100, 7)
    T_gcf = time.time() - T_start_gcf

    T_transform_pol = 0.0
    T_fft_degrid = 0.0
    T_fft_grid = 0.0
    T_degrid = 0.0
    T_residual_vis = 0.0

    # Degrid the current model and form the residual visibilities.
    if not is_n_iter_0:
        residual_data_group = img_xds.attrs["data_groups"][image_data_group_out_name]
        # Delete the SKY_RESIDUAL so the gridded residual image is rebuilt below.
        img_xds.xr_img.delete_data_variables(variables=[residual_data_group["sky"]])

        # Stokes to correlation (instrument) basis for the model visibilities.
        start = time.time()
        img_xds = transform_polarization_basis(
            img_xds,
            new_polarization_basis=instrument_polarization_basis,
            overwrite=True,
        )
        T_transform_pol += time.time() - start

        start = time.time()
        img_xds = fft_norm_img_xds(
            img_xds,
            image_params=image_params,
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_in_name,
            image_data_group_out_modified={
                "visibility": "VISIBILITY_MODEL",
            },
            image_data_variables_keep=["sky"],
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            complex_dtype=complex_dtype,
        )
        T_fft_degrid += time.time() - start

        start = time.time()
        make_visibility_model_single_field(
            ps_xdt,
            img_xds,
            cgk_1D,
            ms_data_group_out_name="model",
            ms_data_group_out_modified={
                "correlated_data": "VISIBILITY_MODEL",
            },
            image_data_group_in_name=image_data_group_in_name,
            processing_function_threads=processing_function_threads,
            fft_padding=image_params["fft_padding"],
        )
        T_degrid += time.time() - start

        start = time.time()
        calculate_residual_visibilities(
            ps_xdt,
            ms_data_group_out_residual="residual",
            ms_data_group_in_model="model",
            ms_data_group_in_observed=ps_data_group_name,
        )
        T_residual_vis += time.time() - start

        # After the first cycle the residual data group becomes the input group.
        ps_data_group_name = "residual"
    else:
        # First (dirty image) cycle: the image dataset arrives from
        # imaging_setup_single_field in the Stokes basis. Flip it to the
        # correlation (instrument) basis so the gridded residual image is created
        # in the same basis as the visibility grid, before the transform back to
        # Stokes below.
        start = time.time()
        img_xds = transform_polarization_basis(
            img_xds,
            new_polarization_basis=instrument_polarization_basis,
            overwrite=True,
        )
        T_transform_pol += time.time() - start

    # Grid the (residual) visibilities into the undeconvolved image grid. The
    # PSF / UV-sampling grid is built once in imaging_setup_single_field, so only
    # the visibility grid is made here.
    from astroviper.processing_functions.imaging.make_undeconvolved_image import (
        make_undeconvolved_image_single_field,
    )

    T_start_grid = time.time()
    img_xds, make_undeconvolved_image_return_df = make_undeconvolved_image_single_field(
        ps_xdt,
        img_xds,
        image_params,
        cgk_1D,
        False,
        ms_data_group_in_name=ps_data_group_name,
        image_data_group_out_name=image_data_group_out_name,
        processing_function_threads=processing_function_threads,
        complex_dtype=complex_dtype,
    )
    T_grid = time.time() - T_start_grid

    start = time.time()
    img_xds = ifft_norm_img_xds(
        img_xds,
        image_params=image_params,
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        image_data_group_out_modified={
            "sky": "SKY_RESIDUAL",
        },
        image_data_variables_keep=image_data_variables_keep,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )
    T_fft_grid += time.time() - start

    from toolviper.utils.memory_management import get_rss_gb

    logger.debug("Memory usage after residual cycle " + str(get_rss_gb()) + " GB")
    start = time.time()
    img_xds = transform_polarization_basis(
        img_xds, new_polarization_basis="stokes", overwrite=True
    )
    T_transform_pol += time.time() - start
    logger.debug(
        "Memory usage after transform polarization " + str(get_rss_gb()) + " GB"
    )

    import pandas as pd

    # Per-cycle timing for each processing function.  T_uv_sampling_grid inside
    # make_undeconvolved_image_return_df is zero here (the UV-sampling grid is
    # built once in imaging_setup_single_field); T_degrid/T_residual_vis are zero
    # on the first (dirty image) cycle.
    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_degrid": [T_degrid],
            "T_fft_degrid": [T_fft_degrid],
            "T_residual_vis": [T_residual_vis],
            "T_grid": [T_grid],
            "T_fft_grid": [T_fft_grid],
            "T_transform_pol": [T_transform_pol],
        }
    )
    # Add the fine-grained gridding timings (T_vis_mask, T_uv_sampling_grid,
    # T_vis_grid) from make_undeconvolved_image_single_field.
    return_df = pd.concat([return_df, make_undeconvolved_image_return_df], axis=1)

    return img_xds, return_df


def make_visibility_model_single_field(
    ps_xdt,
    img_xds,
    cgk_1D,
    ms_data_group_out_name="model",
    ms_data_group_out_modified=None,
    image_data_group_in_name="model",
    processing_function_threads=1,
    fft_padding=1.2,
):
    """Degrid the model image into model visibilities for every measurement set.

    Wraps
    :func:`~astroviper.processing_functions.imaging.get_visibility_grid.get_visibility_grid_single_field`
    over each measurement set in the processing set, writing the degridded model
    visibilities (``VISIBILITY_MODEL``) into the ``model`` data group.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; model visibilities are written into each measurement set
        in place.
    img_xds : xarray.Dataset
        Image dataset holding the model uv-grid to degrid from.
    cgk_1D : numpy.ndarray
        1-D prolate-spheroidal gridding convolution kernel.
    ms_data_group_out_name : str, optional
        Measurement-set data group for the model visibilities.  Default
        ``"model"``.
    ms_data_group_out_modified : dict, optional
        Data-variable override for the output group.  Default
        ``{"correlated_data": "VISIBILITY_MODEL"}``.
    image_data_group_in_name : str, optional
        Image data group holding the model uv-grid.  Default ``"model"``.
    processing_function_threads : int, optional
        Threads handed to the degridder kernel.  Default ``1``.
    fft_padding : float, optional
        Padding factor used during degridding.  Default ``1.2``.
    """
    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {
            "correlated_data": "VISIBILITY_MODEL",
        }
    for ms_xdt in ps_xdt.values():
        get_visibility_grid_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            ms_data_group_out_name=ms_data_group_out_name,
            ms_data_group_out_modified=ms_data_group_out_modified,
            image_data_group_in_name=image_data_group_in_name,
            overwrite=True,
            chan_mode="cube",
            fft_padding=fft_padding,
            processing_function_threads=processing_function_threads,
        )


def calculate_residual_visibilities(
    ps_xdt,
    ms_data_group_out_residual="residual",
    ms_data_group_in_model="model",
    ms_data_group_in_observed="base",
):
    """Form residual visibilities (observed minus model) for every measurement set.

    For each measurement set, subtracts the model visibilities from the observed
    visibilities and registers the result as ``VISIBILITY_RESIDUAL`` under
    ``ms_data_group_out_residual``.

    This helper consumes two measurement-set input groups (the observed
    visibilities and the model visibilities) and writes one output group, so it
    uses the appended-role naming convention rather than a single
    ``ms_data_group_in_name`` / ``ms_data_group_out_name`` pair.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; residual visibilities are written into each measurement
        set in place.
    ms_data_group_out_residual : str, optional
        Output data group for the residual visibilities.  Default ``"residual"``.
    ms_data_group_in_model : str, optional
        Input data group holding the model visibilities.  Default ``"model"``.
    ms_data_group_in_observed : str, optional
        Input data group holding the observed visibilities.  Default ``"base"``.
    """

    for ms_xdt in ps_xdt.values():
        ms_data_group_model = ms_xdt.attrs["data_groups"][ms_data_group_in_model]

        ms_data_group_observed, ms_data_group_residual = create_data_groups_in_and_out(
            ms_xdt,
            data_group_in_name=ms_data_group_in_observed,
            data_group_out_name=ms_data_group_out_residual,
            data_group_out_modified={
                "correlated_data": "VISIBILITY_RESIDUAL",
            },
            overwrite=True,
        )

        ms_xdt[ms_data_group_residual["correlated_data"]] = (
            ms_xdt[ms_data_group_observed["correlated_data"]]
            - ms_xdt[ms_data_group_model["correlated_data"]]
        )

        modify_data_groups_xds(
            ms_xdt,
            data_group_out_name=ms_data_group_out_residual,
            data_group_out=ms_data_group_residual,
            description="Calculated residual visibilities by subtracting model visibilities from observed visibilities.",
        )
