from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def residual_cycle_continuum_single_field(
    ps_xdt,
    img_xds,
    model_xds,
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
    """Create continuum residual Taylor products for the initial dirty-image cycle.

    This implementation supports the current ``niter=0`` path. It grids the
    observed visibilities into MT-MFS residual Taylor uv grids, inverse-transforms
    them into ``SKY_RESIDUAL``, and transforms the result back to Stokes.

    Model degridding and residual-visibility formation for later major cycles are
    intentionally not implemented yet.
    """
    import time

    import numpy as np
    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        fft_norm_continuum_img_xds,
        fft_norm_img_xds,
        ifft_norm_img_xds,
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )
    from astroviper.processing_functions.imaging.make_undeconvolved_image_continuum import (
        make_undeconvolved_image_continuum_single_field,
    )
    from astroviper.processing_functions.imaging.residual_cycle import (
        calculate_residual_visibilities,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    nterms = int(image_params.get("nterms", 2))
    reference_frequency = float(image_params["reference_frequency_hz"])

    if nterms < 1:
        raise ValueError("image_params['nterms'] must be at least 1.")

    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "image_params['reference_frequency'] must be a positive finite "
            "frequency in Hz."
        )

    complex_dtype = np.complex64 if single_precision_image else np.complex128

    ps_data_group_name = processing_set_data_group_name

    start = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(100, 7)
    T_gcf = time.time() - start

    T_transform_pol = 0.0
    T_fft_degrid = 0.0
    T_degrid = 0.0
    T_residual_vis = 0.0

    # ------------------------------------------------------------
    # Later major cycles:
    # predict the current model and form residual visibilities.
    # ------------------------------------------------------------
    if not is_n_iter_0:

        residual_data_group = img_xds.attrs["data_groups"].get(
            image_data_group_out_name
        )

        if residual_data_group is None:
            raise KeyError(
                f"Image data group {image_data_group_out_name!r} is missing."
            )

        residual_sky_name = residual_data_group.get("sky")

        if residual_sky_name is not None and residual_sky_name in img_xds:
            img_xds.xr_img.delete_data_variables(variables=[residual_sky_name])

        #
        # Work on the updated model produced by the previous minor cycle.
        #
        model_xds = transform_polarization_basis(
            model_xds,
            new_polarization_basis=instrument_polarization_basis,
            overwrite=True,
        )

        start = time.time()

        model_xds = fft_norm_continuum_img_xds(
            model_xds,
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

        make_visibility_model_continuum_single_field(
            ps_xdt,
            model_xds,
            cgk_1D,
            nterms=nterms,
            reference_frequency=reference_frequency,
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

        ps_data_group_name = "residual"

        #
        # The residual image is produced in correlation basis.
        #
        start = time.time()

        img_xds = transform_polarization_basis(
            img_xds,
            new_polarization_basis=instrument_polarization_basis,
            overwrite=True,
        )

        T_transform_pol += time.time() - start

    # ------------------------------------------------------------
    # First dirty-image cycle.
    # ------------------------------------------------------------
    else:

        start = time.time()

        img_xds = transform_polarization_basis(
            img_xds,
            new_polarization_basis=instrument_polarization_basis,
            overwrite=True,
        )

        T_transform_pol += time.time() - start

    start = time.time()
    (
        img_xds,
        make_undeconvolved_image_return_df,
    ) = make_undeconvolved_image_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        cgk_1D,
        nterms=nterms,
        reference_frequency=reference_frequency,
        ms_data_group_in_name=ps_data_group_name,
        image_data_group_out_name=image_data_group_out_name,
        processing_function_threads=processing_function_threads,
        complex_dtype=complex_dtype,
    )
    T_grid = time.time() - start

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
    T_fft_grid = time.time() - start

    if "SKY_RESIDUAL" not in img_xds:
        raise RuntimeError("ifft_norm_img_xds did not create SKY_RESIDUAL.")

    if "taylor_term" not in img_xds["SKY_RESIDUAL"].dims:
        raise RuntimeError(
            "Continuum inverse FFT did not preserve the taylor_term dimension."
        )

    img_xds["SKY_RESIDUAL"].attrs.update(
        {
            "description": "Continuum residual Taylor products.",
            "nterms": nterms,
            "reference_frequency": reference_frequency,
            "placeholder": False,
        }
    )

    start = time.time()
    img_xds = transform_polarization_basis(
        img_xds,
        new_polarization_basis="stokes",
        overwrite=True,
    )
    T_transform_pol += time.time() - start

    logger.debug(
        "Created continuum residual Taylor products with dimensions "
        f"{img_xds['SKY_RESIDUAL'].dims} and shape "
        f"{img_xds['SKY_RESIDUAL'].shape}."
    )

    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_degrid": [T_degrid],
            "T_fft_degrid": [T_fft_degrid],
            "T_residual_vis": [T_residual_vis],
            "T_grid": [T_grid],
            "T_fft_grid": [T_fft_grid],
            "T_transform_pol": [T_transform_pol],
            "nterms": [nterms],
            "is_n_iter_0": [bool(is_n_iter_0)],
            "last_residual_cycle": [bool(last_residual_cycle)],
        }
    )

    return_df = pd.concat(
        [return_df, make_undeconvolved_image_return_df],
        axis=1,
    )

    return img_xds, return_df


def make_visibility_model_continuum_single_field(
    ps_xdt,
    model_xds,
    cgk_1D,
    nterms,
    reference_frequency,
    ms_data_group_out_name="model",
    ms_data_group_out_modified=None,
    image_data_group_in_name="model",
    processing_function_threads=1,
    fft_padding=1.2,
):
    """Degrid an MT-MFS Taylor model into channel-dependent model visibilities.

    The input ``model_xds`` is expected to contain Fourier-transformed model
    Taylor terms,

        M_t(u, v),  t = 0, ..., nterms - 1,

    registered as the ``visibility`` variable of
    ``image_data_group_in_name``. For every measurement-set frequency channel,
    this function reconstructs the model uv grid as

        M(u, v, nu) =
            sum_t M_t(u, v)
                  ((nu - reference_frequency) / reference_frequency)**t

    and passes the resulting frequency cube to the existing cube degridder.

    Model visibilities are written into each measurement-set dataset under
    ``ms_data_group_out_name``.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set containing one or more measurement-set datasets.
    model_xds : xarray.Dataset
        Image dataset containing the Fourier-transformed Taylor model. The
        input model data group must define a ``visibility`` variable with a
        ``taylor_term`` dimension.
    cgk_1D : numpy.ndarray
        One-dimensional prolate-spheroidal convolution kernel.
    nterms : int
        Number of Taylor model terms to use.
    reference_frequency : float
        MT-MFS reference frequency in Hz.
    ms_data_group_out_name : str, optional
        Measurement-set data group receiving the predicted model
        visibilities.
    ms_data_group_out_modified : dict, optional
        Output data-variable mapping. Defaults to
        ``{"correlated_data": "VISIBILITY_MODEL"}``.
    image_data_group_in_name : str, optional
        Image data group containing the Fourier-transformed model.
    processing_function_threads : int, optional
        Number of threads supplied to the degridder.
    fft_padding : float, optional
        FFT padding factor used by the degridder.

    Raises
    ------
    ValueError
        If the Taylor configuration or reference frequency is invalid.
    KeyError
        If the model data group, model uv grid, or frequency coordinate is
        missing.
    """
    import copy

    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.imaging.get_visibility_grid import (
        get_visibility_grid_single_field,
    )

    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {
            "correlated_data": "VISIBILITY_MODEL",
        }

    nterms = int(nterms)
    reference_frequency = float(reference_frequency)

    if nterms < 1:
        raise ValueError("nterms must be at least 1.")

    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "reference_frequency must be a positive finite frequency in Hz."
        )

    data_groups = model_xds.attrs.get("data_groups", {})

    if image_data_group_in_name not in data_groups:
        raise KeyError(
            f"Image data group {image_data_group_in_name!r} is not present "
            "in model_xds.attrs['data_groups']."
        )

    model_data_group = data_groups[image_data_group_in_name]
    model_visibility_name = model_data_group.get("visibility")

    if model_visibility_name is None:
        raise KeyError(
            f"Image data group {image_data_group_in_name!r} does not define "
            "a 'visibility' variable."
        )

    if model_visibility_name not in model_xds:
        raise KeyError(
            f"Model visibility-grid variable {model_visibility_name!r} is "
            "not present in model_xds."
        )

    model_taylor_grid = model_xds[model_visibility_name]

    if "taylor_term" not in model_taylor_grid.dims:
        raise ValueError(
            f"Model visibility-grid variable {model_visibility_name!r} must "
            "contain a 'taylor_term' dimension."
        )

    available_nterms = model_taylor_grid.sizes["taylor_term"]

    if available_nterms < nterms:
        raise ValueError(
            f"The model contains {available_nterms} Taylor terms, but "
            f"nterms={nterms} was requested."
        )

    # Ignore any extra Taylor terms that may be present.
    model_taylor_grid = model_taylor_grid.isel(taylor_term=slice(0, nterms))

    for _, ms_xdt in ps_xdt.items():
        if "frequency" in ms_xdt.coords:
            frequency = ms_xdt.coords["frequency"]
        elif "frequency" in ms_xdt:
            frequency = ms_xdt["frequency"]
        else:
            raise KeyError(
                "The measurement-set dataset does not contain a "
                "'frequency' coordinate."
            )

        if frequency.ndim != 1:
            raise ValueError(
                "The measurement-set frequency coordinate must be "
                f"one-dimensional; received dimensions {frequency.dims}."
            )

        frequency_dimension = frequency.dims[0]

        # Use the canonical name expected by the cube degridder.
        if frequency_dimension != "frequency":
            frequency = frequency.rename({frequency_dimension: "frequency"})

        frequency_values = np.asarray(
            frequency.values,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(frequency_values)):
            raise ValueError(
                "The measurement-set frequency coordinate contains "
                "non-finite values."
            )

        taylor_coordinate = np.arange(nterms, dtype=np.int64)

        spectral_coordinate = (
            frequency_values - reference_frequency
        ) / reference_frequency

        # dimensions:
        #
        #     frequency x taylor_term
        #
        taylor_weights = xr.DataArray(
            spectral_coordinate[:, np.newaxis] ** taylor_coordinate[np.newaxis, :],
            dims=("frequency", "taylor_term"),
            coords={
                "frequency": frequency_values,
                "taylor_term": model_taylor_grid.coords.get(
                    "taylor_term",
                    taylor_coordinate,
                ),
            },
        )

        # Reconstruct the uv model for every actual visibility channel:
        #
        #     M(nu) = sum_t M_t x^t.
        #
        frequency_model_grid = (model_taylor_grid * taylor_weights).sum(
            dim="taylor_term",
            keep_attrs=True,
        )

        preferred_dimension_order = [
            dimension
            for dimension in (
                "time",
                "frequency",
                "polarization",
                "u",
                "v",
            )
            if dimension in frequency_model_grid.dims
        ]

        remaining_dimensions = [
            dimension
            for dimension in frequency_model_grid.dims
            if dimension not in preferred_dimension_order
        ]

        frequency_model_grid = frequency_model_grid.transpose(
            *preferred_dimension_order,
            *remaining_dimensions,
        )

        # Construct a temporary cube-compatible image dataset. Its model data
        # group still points to the same logical visibility-grid name, but that
        # variable now has a frequency dimension rather than taylor_term.
        cube_model_xds = xr.Dataset(attrs=copy.deepcopy(model_xds.attrs))

        for coordinate_name, coordinate in model_xds.coords.items():
            if coordinate_name in {"taylor_term", "frequency"}:
                continue

            if "taylor_term" in coordinate.dims:
                continue

            cube_model_xds = cube_model_xds.assign_coords(
                {coordinate_name: coordinate.copy(deep=False)}
            )

        cube_model_xds = cube_model_xds.assign_coords(
            frequency=("frequency", frequency_values)
        )

        cube_model_xds[model_visibility_name] = frequency_model_grid

        get_visibility_grid_single_field(
            ms_xdt,
            cgk_1D,
            cube_model_xds,
            ms_data_group_out_name=ms_data_group_out_name,
            ms_data_group_out_modified=ms_data_group_out_modified,
            image_data_group_in_name=image_data_group_in_name,
            overwrite=True,
            chan_mode="cube",
            fft_padding=fft_padding,
            processing_function_threads=processing_function_threads,
        )

    return
