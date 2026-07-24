import xarray as xr

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def imaging_preparation_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    processing_set_data_group_name="corrected",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    task_id=0,
):
    """Run the once-per-chunk continuum-imaging setup.

    Everything performed only once for one frequency-chunk map task belongs
    here:

    * construction of the :class:`IterationController`;
    * construction of the initially empty combined deconvolution return dict;
    * imaging-weight calculation;
    * creation of the local Taylor PSF/Hessian products;
    * creation of the primary-beam products, when requested;
    * creation of any continuum/Taylor coordinates and data groups required by
      the returned :class:`xarray.Dataset`.

    The dirty/residual Taylor images are deliberately not calculated here.
    They are produced by the first call to
    :func:`residual_cycle_continuum_single_field`.

    Notes
    -----
    For ``nterms=2`` the setup implementation is expected to create local
    Taylor PSF/Hessian products of orders 0, 1, and 2. The preferred xarray
    representation is one point-spread-function variable with a Taylor-order
    dimension rather than separate unrelated variables.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this frequency chunk.
    img_xds : xarray.Dataset
        Empty image dataset for this chunk.
    image_params : dict
        Image geometry and continuum configuration. In addition to the normal
        image geometry, this should contain or resolve:

        * ``nterms``;
        * ``reference_frequency``;
        * the output polarization and time coordinates;
        * gridding/FFT parameters such as ``fft_padding``.
    imaging_weights_params : dict
        Weighting configuration.
    iteration_control_params : dict
        Major/minor-cycle control configuration.
    processing_set_data_group_name : str, optional
        Processing-set data group to image.
    single_precision_image : bool, optional
        Whether image-domain products use single precision.
    processing_function_threads : int, optional
        Number of threads supplied to lower-level processing kernels.
    fft_backend : str, optional
        FFT backend used during image normalization.
    image_data_variables_keep : list of str, optional
        Logical output variables retained in ``img_xds``.
    task_id : int, optional
        Frequency-chunk identifier.

    Returns
    -------
    controller : IterationController
        Freshly initialized iteration controller.
    img_xds : xarray.Dataset
        Dataset carrying the local Taylor PSF/Hessian and setup products.
    return_df : pandas.DataFrame
        One-row timing frame returned by the setup function.
    combined_deconvolve_dict : ReturnDict
        Initially empty deconvolution-statistics accumulator.
    T_setup : float
        Wall-clock duration of this setup phase.
    """
    import time

    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
        imaging_setup_continuum_single_field,
    )

    logger.debug("Processing continuum chunk " + str(task_id))

    start = time.time()

    img_xds, return_df = imaging_setup_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        processing_set_data_group_name=processing_set_data_group_name,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
    )

    T_setup = time.time() - start

    return (
        img_xds,
        return_df,
        T_setup,
    )


@shares_param_docs
def residual_update_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    processing_set_data_group_name="corrected",
    deconvolver="hogbom",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    is_n_iter_0=True,
    model_uv_xds=None,
    task_id=0,
):
    """Perform setup and exactly one continuum residual update."""

    import time

    import pandas as pd

    from astroviper.processing_functions.imaging.residual_cycle_continuum_single_field import (
        residual_cycle_continuum_single_field,
    )
    from astroviper.processing_functions.imaging.utils import accumulate_timing

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    timing = {
        "T_prep": 0.0,
        "T_residual_cycle": 0.0,
        "task_id": task_id,
    }

    # -------------------------------------------------------------
    # Setup phase
    # -------------------------------------------------------------

    if is_n_iter_0:

        (
            img_xds,
            setup_return_df,
            T_setup,
        ) = imaging_preparation_continuum_single_field(
            ps_xdt,
            img_xds,
            image_params,
            imaging_weights_params,
            processing_set_data_group_name=processing_set_data_group_name,
            single_precision_image=single_precision_image,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            image_data_variables_keep=image_data_variables_keep,
            task_id=task_id,
        )

        timing["T_prep"] = T_setup
        accumulate_timing(
            timing,
            setup_return_df,
            phase="prep",
        )

    else:
        # Do the empty registrations ... that are needed down the line
        # from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        #    transform_polarization_basis,
        # )
        from astroviper.processing_functions.imaging.calculate_imaging_weights import (
            calculate_imaging_weights,
        )

        # make_empty_sky_image() supplies the geometry and coordinates, but the
        # xradio image accessor requires this dataset-type marker.
        img_xds.attrs["type"] = "image_dataset"

        # Residual needs to be registered
        image_data_group_out_name = "residual"
        data_groups = img_xds.attrs.setdefault("data_groups", {})

        if image_data_group_out_name not in data_groups:
            img_xds = img_xds.xr_img.add_data_group(
                new_data_group_name=image_data_group_out_name,
                new_data_group={
                    "description": "Continuum residual products.",
                    "date": "2026",
                },
            )

        # Need to transform basis to Stokes
        # img_xds = transform_polarization_basis(
        #    img_xds,
        #    new_polarization_basis="stokes",
        #    overwrite=True,
        # )

        # Needs to be refactored at a later point when decided what to do with weights
        start = time.time()

        calculate_imaging_weights(
            ps_xdt,
            img_xds,
            imaging_weights_params=imaging_weights_params,
            return_weight_density_grid=False,
            ms_data_group_in_name=processing_set_data_group_name,
            ms_data_group_out_name=processing_set_data_group_name,
            ms_data_group_out_modified={
                "weight_imaging": "WEIGHT_IMAGING",
            },
            processing_function_threads=processing_function_threads,
        )

        T_weights = time.time() - start

        setup_return_df = pd.DataFrame({})

        timing["T_prep"] = T_weights
        accumulate_timing(
            timing,
            setup_return_df,
            phase="prep",
        )

    # -------------------------------------------------------------
    # Exactly one residual update
    # -------------------------------------------------------------
    start = time.time()

    img_xds, residual_return_df = residual_cycle_continuum_single_field(
        ps_xdt,
        img_xds,
        model_uv_xds,
        image_params,
        is_n_iter_0,
        processing_set_data_group_name=processing_set_data_group_name,
        instrument_polarization_basis=instrument_polarization_basis,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
    )

    timing["T_residual_cycle"] = time.time() - start
    accumulate_timing(timing, residual_return_df)

    timing["n_channels"] = img_xds.sizes.get(
        "frequency",
        len(img_xds.coords.get("frequency", [])),
    )
    timing["nterms"] = image_params.get("nterms", 2)

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df


@shares_param_docs
def model_update_mtmfs_single_field(
    img_xds,
    deconvolver,
    deconvolve_params,
    is_n_iter_0=True,
    processing_function_threads=1,
    image_data_group_in_name="residual",
    image_data_group_out_name="model",
):
    """Run a temporary continuum minor cycle using Taylor order zero.

        This is an interim implementation until a true MT-MFS deconvolution backend
        is available. It performs an ordinary Högbom CLEAN using only

            SKY_RESIDUAL[taylor_term=0]

        and

            POINT_SPREAD_FUNCTION[psf_taylor_order=0].

        The resulting model is copied into

            SKY_MODEL[taylor_term=0].

        Higher-order Taylor residuals, PSFs, and model terms are left unchanged.

        For now, we follow this strategy as a placeholder
        continuum Taylor dataset
            │
            ├── take residual Taylor term 0
            ├── take PSF Taylor order 0
            ├── take model Taylor term 0
            │
            ▼
    build temporary one-channel cube dataset
    (time, frequency=1, polarization, l, m)
            │
            ▼
    call existing cube model-update function
            │
            ├── create primary-beam mask if needed
            ├── run Högbom CLEAN
            └── update the temporary cube model
            │
            ▼
    copy cleaned model plane back into SKY_MODEL[taylor_term=0]

        Parameters
        ----------
        img_xds : xarray.Dataset
            Globally reduced continuum image dataset. Residual/model images are
            expected to use ``taylor_term`` while the PSF uses
            ``psf_taylor_order``.
        deconvolver : str
            Deconvolver name. Currently only ``"hogbom"`` is supported.
        deconvolve_params : dict
            Minor-cycle parameters, including ``cycleniter``,
            ``cyclethreshold``, ``niter_per_plane``, and
            ``cyclethreshold_per_plane``.
        is_n_iter_0 : bool, optional
            Whether this is the first model-update cycle.
        processing_function_threads : int, optional
            Number of threads used by the deconvolution backend.
        image_data_group_in_name : str, optional
            Input image data-group name.
        image_data_group_out_name : str, optional
            Output model data-group name.

        Returns
        -------
        deconvolve_dict : ReturnDict
            Per-plane deconvolution statistics returned by the Högbom backend.
        return_df : pandas.DataFrame
            Timing information for this model-update cycle.

        Notes
        -----
        Despite the function name, this is not yet a true MT-MFS minor cycle.
        It is a compatibility layer around the existing single-frequency Högbom
        backend.
    """
    import copy
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    if deconvolver.lower() != "hogbom":
        raise NotImplementedError(
            "Continuum deconvolution currently supports only "
            "'hogbom' cleaning of Taylor term zero."
        )

    if "SKY_RESIDUAL" not in img_xds:
        raise KeyError("The continuum image dataset does not contain SKY_RESIDUAL.")

    if "POINT_SPREAD_FUNCTION" not in img_xds:
        raise KeyError(
            "The continuum image dataset does not contain " "POINT_SPREAD_FUNCTION."
        )

    data_groups = img_xds.attrs.get("data_groups", {})

    if image_data_group_in_name not in data_groups:
        raise KeyError(
            f"Input data group {image_data_group_in_name!r} is not present "
            "in img_xds.attrs['data_groups']."
        )

    residual_data_group = data_groups[image_data_group_in_name]

    residual_name = residual_data_group.get("sky")
    psf_name = residual_data_group.get("point_spread_function")

    residual = img_xds["SKY_RESIDUAL"]
    psf = img_xds["POINT_SPREAD_FUNCTION"]

    if "taylor_term" not in residual.dims:
        raise ValueError("SKY_RESIDUAL must contain a 'taylor_term' dimension.")

    if "psf_taylor_order" not in psf.dims:
        raise ValueError(
            "POINT_SPREAD_FUNCTION must contain a " "'psf_taylor_order' dimension."
        )

    if residual.sizes["taylor_term"] < 1:
        raise ValueError("SKY_RESIDUAL contains no Taylor terms.")

    if psf.sizes["psf_taylor_order"] < 1:
        raise ValueError("POINT_SPREAD_FUNCTION contains no PSF Taylor terms.")

    # make a cube-compatible copy of the per-plane controls
    cube_deconvolve_params = copy.deepcopy(deconvolve_params)

    for parameter_name in (
        "niter_per_plane",
        "cyclethreshold_per_plane",
    ):
        parameter_value = cube_deconvolve_params.get(parameter_name)

        if parameter_value is None:
            continue

        parameter_array = np.asarray(parameter_value)

        # Continuum layout:
        #     time, taylor_term/frequency-like plane, polarization
        #
        # Temporary Högbom layout:
        #     time, frequency=1, polarization
        if parameter_array.ndim != 3:
            raise ValueError(
                f"{parameter_name} must have three dimensions "
                "(time, plane, polarization); "
                f"received shape {parameter_array.shape}."
            )

        if parameter_array.shape[1] < 1:
            raise ValueError(f"{parameter_name} contains no image planes.")

        cube_deconvolve_params[parameter_name] = np.ascontiguousarray(
            parameter_array[:, 0:1, :]
        )

    start = time.time()

    max_sidelobe_name = "MAX_SIDELOBE_POINT_SPREAD_FUNCTION"

    if max_sidelobe_name not in img_xds:
        raise KeyError(
            f"{max_sidelobe_name} is required before running the "
            "continuum minor cycle. It must be created by "
            "point_spread_function_gaussian_fit_continuum() during "
            "the first append node and restored from static_xds during "
            "later append nodes."
        )

    data_groups = img_xds.attrs.setdefault(
        "data_groups",
        {},
    )

    residual_data_group = data_groups.setdefault(
        image_data_group_in_name,
        {},
    )

    residual_data_group["max_sidelobe_point_spread_function"] = max_sidelobe_name

    # ------------------------------------------------------------------
    # Build a temporary cube-like image dataset.
    #
    # The existing Högbom backend expects
    #
    #     time x frequency x polarization x l x m
    #
    # whereas the continuum products use Taylor dimensions.
    # ------------------------------------------------------------------

    if "frequency" in img_xds.coords and img_xds.sizes.get("frequency", 0) > 0:
        frequency_coord = np.atleast_1d(img_xds.coords["frequency"].values)[:1]
    elif "reference_frequency" in img_xds.attrs:
        frequency_coord = np.atleast_1d(img_xds.attrs["reference_frequency"])
    else:
        # The numerical value is irrelevant to Högbom CLEAN. The singleton
        # dimension exists only to satisfy the cube image interface.
        frequency_coord = np.asarray([0.0], dtype=np.float64)

    hogbom_xds = xr.Dataset(attrs=dict(img_xds.attrs))

    # Preserve coordinates that do not depend on a Taylor-order dimension.
    for coord_name, coord in img_xds.coords.items():
        if (
            "taylor_term" not in coord.dims
            and "psf_taylor_order" not in coord.dims
            and coord_name != "frequency"
        ):
            hogbom_xds = hogbom_xds.assign_coords({coord_name: coord.copy(deep=False)})

    hogbom_xds = hogbom_xds.assign_coords(frequency=("frequency", frequency_coord))

    def _to_single_frequency(data_array, plane_dimension):
        """Select plane zero and insert a singleton frequency dimension."""
        plane = data_array.isel({plane_dimension: 0}, drop=True)

        if "frequency" in plane.dims:
            if plane.sizes["frequency"] != 1:
                plane = plane.isel(frequency=0, drop=False)

            return plane.assign_coords(frequency=frequency_coord)

        # Insert frequency immediately after time to preserve the conventional
        # image dimension ordering.
        if "time" in plane.dims:
            axis = plane.dims.index("time") + 1
        else:
            axis = 0

        return plane.expand_dims(
            frequency=frequency_coord,
            axis=axis,
        )

    # Copy Taylor-dependent variables into the temporary cube layout.
    max_sidelobe_name = "MAX_SIDELOBE_POINT_SPREAD_FUNCTION"

    for variable_name, data_array in img_xds.data_vars.items():

        if variable_name == max_sidelobe_name:
            if data_array.dims == ("time", "polarization"):
                hogbom_xds[variable_name] = xr.DataArray(
                    data_array.values[:, np.newaxis, :],
                    dims=(
                        "time",
                        "frequency",
                        "polarization",
                    ),
                    coords={
                        "time": data_array.coords["time"],
                        "frequency": frequency_coord,
                        "polarization": data_array.coords["polarization"],
                    },
                    attrs=data_array.attrs.copy(),
                )

            elif "psf_taylor_order" in data_array.dims:
                hogbom_xds[variable_name] = _to_single_frequency(
                    data_array,
                    "psf_taylor_order",
                )

            elif "frequency" in data_array.dims:
                hogbom_xds[variable_name] = data_array.isel(
                    frequency=slice(0, 1)
                ).assign_coords(frequency=frequency_coord)

            else:
                raise ValueError(
                    f"{variable_name} has unsupported dimensions " f"{data_array.dims}."
                )

        elif "taylor_term" in data_array.dims:
            hogbom_xds[variable_name] = _to_single_frequency(
                data_array,
                "taylor_term",
            )

        elif "psf_taylor_order" in data_array.dims:
            hogbom_xds[variable_name] = _to_single_frequency(
                data_array,
                "psf_taylor_order",
            )

        elif "frequency" in data_array.dims:
            # Primary beam and similar products may retain a frequency
            # dimension. Högbom only needs one plane here.
            if data_array.sizes["frequency"] == 1:
                hogbom_xds[variable_name] = data_array.assign_coords(
                    frequency=frequency_coord
                )
            else:
                hogbom_xds[variable_name] = data_array.isel(
                    frequency=0, drop=False
                ).assign_coords(frequency=frequency_coord)

        else:
            hogbom_xds[variable_name] = data_array.copy(deep=False)

    # Ensure the temporary model variable exists.
    if "SKY_MODEL" not in hogbom_xds:
        hogbom_xds["SKY_MODEL"] = xr.zeros_like(hogbom_xds["SKY_RESIDUAL"])

    # Register the temporary model through the data-group interface expected by
    # deconvolve() and starting_statistics().
    hogbom_xds.attrs.setdefault("data_groups", {})

    hogbom_xds.attrs["data_groups"][image_data_group_out_name] = {
        "sky": "SKY_MODEL",
    }
    model_name = hogbom_xds.attrs["data_groups"][image_data_group_out_name]["sky"]

    T_make_hogbom_view = time.time() - start

    # Convert to real numbers
    hogbom_xds[residual_name] = hogbom_xds[residual_name].real.astype(np.float32)
    hogbom_xds[psf_name] = hogbom_xds[psf_name].real.astype(np.float32)
    hogbom_xds[model_name] = hogbom_xds[model_name].real.astype(np.float32)

    # ------------------------------------------------------------------
    # Run the existing single-frequency Högbom model-update backend.
    # ------------------------------------------------------------------
    start = time.time()

    from astroviper.processing_functions.imaging.model_update_cycle import (
        model_update_cycle_cube_single_field,
    )

    deconvolve_dict, hogbom_return_df = model_update_cycle_cube_single_field(
        hogbom_xds,
        deconvolver,
        cube_deconvolve_params,
        is_n_iter_0,
        processing_function_threads=processing_function_threads,
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
    )

    T_hogbom = time.time() - start

    # ------------------------------------------------------------------
    # Copy the Högbom model back into Taylor term zero.
    # ------------------------------------------------------------------
    start = time.time()

    if "SKY_MODEL" not in hogbom_xds:
        raise RuntimeError("The Högbom model-update backend did not produce SKY_MODEL.")

    model_zero = hogbom_xds["SKY_MODEL"].isel(
        frequency=0,
        drop=True,
    )

    if "SKY_MODEL" not in img_xds:
        # Create all model Taylor terms, initialized to zero.
        img_xds["SKY_MODEL"] = xr.zeros_like(img_xds["SKY_RESIDUAL"])

    if "taylor_term" not in img_xds["SKY_MODEL"].dims:
        raise ValueError(
            "The continuum SKY_MODEL variable must contain a "
            "'taylor_term' dimension."
        )

    # Use positional assignment because the Taylor coordinate need not
    # literally contain the value zero.
    img_xds["SKY_MODEL"].data[:, 0, ...] = model_zero.data

    T_copy_model = time.time() - start

    return_df = pd.DataFrame(
        {
            "T_make_hogbom_view": [T_make_hogbom_view],
            "T_hogbom": [T_hogbom],
            "T_copy_model": [T_copy_model],
            "T_model_update_cycle": [T_make_hogbom_view + T_hogbom + T_copy_model],
        }
    )

    if hogbom_return_df is not None:
        hogbom_return_df = hogbom_return_df.reset_index(drop=True)
        return_df = pd.concat(
            [return_df, hogbom_return_df],
            axis=1,
        )

    return deconvolve_dict, return_df


def restore_image(
    img_xds,
    image_data_group_in_residual_name="residual",
    image_data_group_in_model_name="model",
    image_data_group_out_restore_name="restored",
    processing_function_threads=1,
):
    """Restore the Taylor-zero continuum model.

    The continuum model and residual use a ``taylor_term`` dimension, while
    the existing cube restoration routine expects a ``frequency`` dimension.
    This wrapper constructs a temporary one-channel cube containing:

    * residual Taylor term zero;
    * model Taylor term zero;
    * PSF Taylor order zero;
    * the fitted restoring-beam parameters.

    It then calls the existing cube restoration routine and copies the restored
    reference-frequency image back into the continuum dataset.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Continuum image dataset containing the residual, model, PSF, and fitted
        restoring-beam parameters.
    image_data_group_in_residual_name : str, optional
        Data group containing the residual image.
    image_data_group_in_model_name : str, optional
        Data group containing the sky model.
    image_data_group_out_restore_name : str, optional
        Data group under which the restored image is registered.
    processing_function_threads : int, optional
        Number of threads passed to the cube restoration routine.

    Returns
    -------
    img_xds : xarray.Dataset
        Input continuum dataset with the restored reference-frequency image.
    timing_df : pandas.DataFrame
        Timing information returned by the cube restoration routine.
    """
    import copy

    import numpy as np
    import xarray as xr

    # Alias the existing cube routine to avoid colliding with this wrapper.
    from astroviper.processing_functions.imaging.restore import (
        restore_image as restore_cube_image,
    )

    data_groups = img_xds.attrs.get("data_groups", {})

    # ------------------------------------------------------------------
    # Resolve the residual and model variables.
    # ------------------------------------------------------------------
    if image_data_group_in_residual_name not in data_groups:
        raise KeyError(
            "Residual data group "
            f"{image_data_group_in_residual_name!r} is missing. "
            f"Available groups are {list(data_groups)}."
        )

    if image_data_group_in_model_name not in data_groups:
        raise KeyError(
            "Model data group "
            f"{image_data_group_in_model_name!r} is missing. "
            f"Available groups are {list(data_groups)}."
        )

    residual_group = data_groups[image_data_group_in_residual_name]
    model_group = data_groups[image_data_group_in_model_name]

    residual_name = residual_group.get("sky")
    model_name = model_group.get("sky")

    if residual_name is None:
        raise KeyError(
            f"Residual data group "
            f"{image_data_group_in_residual_name!r} does not define 'sky'."
        )

    if model_name is None:
        raise KeyError(
            f"Model data group "
            f"{image_data_group_in_model_name!r} does not define 'sky'."
        )

    if residual_name not in img_xds:
        raise KeyError(f"Residual variable {residual_name!r} is missing from img_xds.")

    if model_name not in img_xds:
        raise KeyError(f"Model variable {model_name!r} is missing from img_xds.")

    residual_da = img_xds[residual_name]
    model_da = img_xds[model_name]

    for variable_name, data_array in (
        (residual_name, residual_da),
        (model_name, model_da),
    ):
        if "taylor_term" not in data_array.dims:
            raise ValueError(
                f"{variable_name!r} must contain a 'taylor_term' "
                f"dimension. Found {data_array.dims}."
            )

        if data_array.sizes["taylor_term"] < 1:
            raise ValueError(f"{variable_name!r} contains no Taylor terms.")

    # ------------------------------------------------------------------
    # Resolve the PSF.
    # ------------------------------------------------------------------
    psf_name = residual_group.get("point_spread_function")

    if psf_name is None:
        # Permit older datasets where the logical registration is missing.
        if "POINT_SPREAD_FUNCTION" in img_xds:
            psf_name = "POINT_SPREAD_FUNCTION"
        else:
            raise KeyError(
                "The residual data group does not define "
                "'point_spread_function', and POINT_SPREAD_FUNCTION is absent."
            )

    if psf_name not in img_xds:
        raise KeyError(f"Point-spread-function variable {psf_name!r} is missing.")

    psf_da = img_xds[psf_name]

    # ------------------------------------------------------------------
    # Resolve the fitted restoring beam.
    # ------------------------------------------------------------------
    beam_fit_key = "beam_fit_params_point_spread_function"
    beam_name = residual_group.get(beam_fit_key)

    if beam_name is None:
        if "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION" in img_xds:
            beam_name = "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
        else:
            raise KeyError(
                f"The residual data group does not define {beam_fit_key!r}, "
                "and BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION is absent."
            )

    if beam_name not in img_xds:
        raise KeyError(f"Restoring-beam variable {beam_name!r} is missing.")

    beam_da = img_xds[beam_name]

    # ------------------------------------------------------------------
    # Construct a one-channel cube view.
    # ------------------------------------------------------------------
    cube_xds = xr.Dataset(attrs=copy.deepcopy(img_xds.attrs))

    # Preserve all coordinates that do not depend on a spectral/Taylor axis.
    for coord_name, coord in img_xds.coords.items():
        if coord_name in {
            "frequency",
            "taylor_term",
            "psf_taylor_order",
        }:
            continue

        if (
            "frequency" in coord.dims
            or "taylor_term" in coord.dims
            or "psf_taylor_order" in coord.dims
        ):
            continue

        cube_xds = cube_xds.assign_coords({coord_name: coord.copy(deep=False)})

    # The numerical frequency value is not used by restoration, but use the
    # reference frequency when it is available to preserve meaningful metadata.
    reference_frequency = img_xds.attrs.get("reference_frequency") or img_xds.attrs.get(
        "reference_frequency_hz"
    )

    if reference_frequency is None:
        frequency_value = np.asarray([0.0], dtype=np.float64)
    else:
        frequency_value = np.asarray(
            [float(reference_frequency)],
            dtype=np.float64,
        )

    cube_xds = cube_xds.assign_coords(frequency=("frequency", frequency_value))

    def _taylor_zero_to_frequency(data_array):
        """Convert Taylor term zero into a singleton frequency plane."""
        return (
            data_array.isel(taylor_term=slice(0, 1))
            .rename({"taylor_term": "frequency"})
            .assign_coords(frequency=cube_xds.coords["frequency"])
        )

    cube_xds[residual_name] = _taylor_zero_to_frequency(residual_da)
    cube_xds[model_name] = _taylor_zero_to_frequency(model_da)

    if "psf_taylor_order" in psf_da.dims:
        if psf_da.sizes["psf_taylor_order"] < 1:
            raise ValueError(f"PSF variable {psf_name!r} contains no Taylor orders.")

        cube_xds[psf_name] = (
            psf_da.isel(psf_taylor_order=slice(0, 1))
            .rename({"psf_taylor_order": "frequency"})
            .assign_coords(frequency=cube_xds.coords["frequency"])
        )

    elif "taylor_term" in psf_da.dims:
        cube_xds[psf_name] = (
            psf_da.isel(taylor_term=slice(0, 1))
            .rename({"taylor_term": "frequency"})
            .assign_coords(frequency=cube_xds.coords["frequency"])
        )

    elif "frequency" in psf_da.dims:
        cube_xds[psf_name] = psf_da.isel(frequency=slice(0, 1)).assign_coords(
            frequency=cube_xds.coords["frequency"]
        )

    else:
        frequency_axis = psf_da.dims.index("time") + 1 if "time" in psf_da.dims else 0

        cube_xds[psf_name] = psf_da.expand_dims(
            frequency=cube_xds.coords["frequency"].values,
            axis=frequency_axis,
        )

    # ------------------------------------------------------------------
    # Convert the beam parameters into cube layout.
    # ------------------------------------------------------------------
    if "frequency" in beam_da.dims:
        cube_beam_da = beam_da.isel(frequency=slice(0, 1)).assign_coords(
            frequency=cube_xds.coords["frequency"]
        )

    elif "taylor_term" in beam_da.dims:
        cube_beam_da = (
            beam_da.isel(taylor_term=slice(0, 1))
            .rename({"taylor_term": "frequency"})
            .assign_coords(frequency=cube_xds.coords["frequency"])
        )

    elif "psf_taylor_order" in beam_da.dims:
        cube_beam_da = (
            beam_da.isel(psf_taylor_order=slice(0, 1))
            .rename({"psf_taylor_order": "frequency"})
            .assign_coords(frequency=cube_xds.coords["frequency"])
        )

    else:
        frequency_axis = beam_da.dims.index("time") + 1 if "time" in beam_da.dims else 0

        cube_beam_da = beam_da.expand_dims(
            frequency=cube_xds.coords["frequency"].values,
            axis=frequency_axis,
        )

    expected_beam_dims = (
        "time",
        "frequency",
        "polarization",
        "beam_params",
    )

    missing_beam_dims = [
        dim for dim in expected_beam_dims if dim not in cube_beam_da.dims
    ]

    if missing_beam_dims:
        raise ValueError(
            f"Beam variable {beam_name!r} cannot be converted to the "
            f"expected cube dimensions {expected_beam_dims}. "
            f"Missing dimensions: {missing_beam_dims}; "
            f"found {cube_beam_da.dims}."
        )

    cube_xds[beam_name] = cube_beam_da.transpose(*expected_beam_dims)

    # Ensure the temporary cube data-group registrations refer to variables
    # actually installed above.
    cube_data_groups = cube_xds.attrs.setdefault("data_groups", {})

    cube_residual_group = cube_data_groups.setdefault(
        image_data_group_in_residual_name,
        {},
    )
    cube_residual_group["sky"] = residual_name
    cube_residual_group["point_spread_function"] = psf_name
    cube_residual_group[beam_fit_key] = beam_name

    cube_model_group = cube_data_groups.setdefault(
        image_data_group_in_model_name,
        {},
    )
    cube_model_group["sky"] = model_name

    # ------------------------------------------------------------------
    # Run the existing cube restoration implementation.
    # ------------------------------------------------------------------
    cube_xds, timing_df = restore_cube_image(
        cube_xds,
        image_data_group_in_residual_name=(image_data_group_in_residual_name),
        image_data_group_in_model_name=image_data_group_in_model_name,
        image_data_group_out_restore_name=(image_data_group_out_restore_name),
        processing_function_threads=processing_function_threads,
    )

    restored_group = cube_xds.attrs["data_groups"][image_data_group_out_restore_name]
    restored_name = restored_group["sky"]

    if restored_name not in cube_xds:
        raise RuntimeError(
            "The cube restoration routine registered restored variable "
            f"{restored_name!r}, but did not create it."
        )

    # ------------------------------------------------------------------
    # Copy the restored reference-frequency plane back to Taylor layout.
    # ------------------------------------------------------------------
    restored_da = (
        cube_xds[restored_name]
        .rename({"frequency": "taylor_term"})
        .assign_coords(
            taylor_term=img_xds.coords["taylor_term"].isel(taylor_term=slice(0, 1))
        )
    )

    img_xds[restored_name] = restored_da

    img_xds.attrs.setdefault("data_groups", {})[
        image_data_group_out_restore_name
    ] = copy.deepcopy(restored_group)

    return img_xds, timing_df


def point_spread_function_gaussian_fit_continuum(
    img_xds,
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    processing_function_threads=1,
):
    """Fit the restoring beam to the zeroth-order continuum PSF.

    The existing cube ``point_spread_function_gaussian_fit`` routine expects
    a PSF with dimensions

        (time, frequency, polarization, l, m).

    A continuum MT-MFS PSF instead has dimensions

        (time, psf_taylor_order, polarization, l, m).

    This wrapper selects the globally reduced zeroth-order Taylor PSF,
    presents it to the cube fitter as a single-frequency PSF, and copies the
    fitted beam parameters and maximum sidelobe back into the continuum
    image dataset.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Globally reduced continuum image dataset.
    image_data_group_in_name : str, optional
        Data group containing the continuum point-spread function.
    image_data_group_out_name : str, optional
        Data group in which the beam-fit products are registered.
    processing_function_threads : int, optional
        Number of threads supplied to the Gaussian-fit routine.

    Returns
    -------
    img_xds : xarray.Dataset
        Continuum dataset containing the fitted restoring beam and maximum
        PSF sidelobe.
    return_df : pandas.DataFrame
        One-row timing dataframe.
    """
    import time
    from copy import deepcopy

    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
        point_spread_function_gaussian_fit,
    )

    beam_fit_key = "beam_fit_params_point_spread_function"
    max_sidelobe_key = "max_sidelobe_point_spread_function"

    # ------------------------------------------------------------------
    # Validate the requested data groups and obtain the PSF variable.
    # ------------------------------------------------------------------
    data_groups = img_xds.attrs.get("data_groups", {})

    if image_data_group_in_name not in data_groups:
        raise KeyError(
            f"Input data group {image_data_group_in_name!r} was not found. "
            f"Available groups are {list(data_groups)}."
        )

    input_group = data_groups[image_data_group_in_name]

    if "point_spread_function" not in input_group:
        raise KeyError(
            "'point_spread_function' was not found in data group "
            f"{image_data_group_in_name!r}."
        )

    psf_name = input_group["point_spread_function"]

    if psf_name not in img_xds:
        raise KeyError(
            f"PSF variable {psf_name!r}, registered in data group "
            f"{image_data_group_in_name!r}, was not found in img_xds."
        )

    psf_da = img_xds[psf_name]

    if "psf_taylor_order" not in psf_da.dims:
        raise ValueError(
            f"Continuum PSF {psf_name!r} must contain the dimension "
            f"'psf_taylor_order'. Found dimensions {psf_da.dims}."
        )

    if psf_da.sizes["psf_taylor_order"] < 1:
        raise ValueError(f"Continuum PSF {psf_name!r} has no Taylor-order entries.")

    required_psf_dims = {"time", "polarization", "l", "m"}
    missing_psf_dims = required_psf_dims.difference(psf_da.dims)

    if missing_psf_dims:
        raise ValueError(
            f"Continuum PSF {psf_name!r} is missing required dimensions "
            f"{sorted(missing_psf_dims)}. Found dimensions {psf_da.dims}."
        )

    # ------------------------------------------------------------------
    # Select the globally reduced zeroth-order PSF.
    # ------------------------------------------------------------------
    psf0_da = psf_da.isel(psf_taylor_order=0, drop=True)

    # The numerical frequency value is not used by the fit itself, but the
    # cube routine requires a one-element frequency coordinate.
    continuum_metadata = img_xds.attrs.get("continuum_imaging", {})

    if "reference_frequency_hz" in continuum_metadata:
        reference_frequency_hz = float(continuum_metadata["reference_frequency_hz"])
    elif "frequency" in img_xds.coords and img_xds.frequency.size > 0:
        reference_frequency_hz = float(
            np.asarray(img_xds.frequency.values).reshape(-1)[0]
        )
    else:
        # This is only a coordinate label for the temporary cube.
        reference_frequency_hz = 0.0

    frequency_coord = np.asarray(
        [reference_frequency_hz],
        dtype=np.float64,
    )

    cube_psf_da = psf0_da.expand_dims(frequency=frequency_coord,).transpose(
        "time",
        "frequency",
        "polarization",
        "l",
        "m",
    )

    # ------------------------------------------------------------------
    # Build a minimal, one-frequency cube dataset.
    # ------------------------------------------------------------------
    cube_xds = xr.Dataset(
        {
            psf_name: cube_psf_da,
        }
    )

    cube_xds.attrs = deepcopy(img_xds.attrs)
    cube_xds.attrs["type"] = "image_dataset"

    # The Gaussian fitter only needs the PSF registration. Using the same
    # input and output group lets it add its beam-fit entries to this group.
    cube_xds.attrs["data_groups"] = {
        image_data_group_out_name: {
            "point_spread_function": psf_name,
        }
    }

    # ------------------------------------------------------------------
    # Run the existing cube Gaussian-fit routine.
    #
    # The routine modifies and returns only the xarray.Dataset; it does not
    # return a timing dataframe.
    # ------------------------------------------------------------------
    start = time.time()

    cube_xds = point_spread_function_gaussian_fit(
        cube_xds,
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        processing_function_threads=processing_function_threads,
    )

    return_df = pd.DataFrame(
        {
            "T_psf_fit": [time.time() - start],
        }
    )

    cube_group = cube_xds.attrs["data_groups"][image_data_group_out_name]

    if beam_fit_key not in cube_group:
        raise KeyError(
            f"The Gaussian fitter did not register {beam_fit_key!r} "
            f"in data group {image_data_group_out_name!r}."
        )

    if max_sidelobe_key not in cube_group:
        raise KeyError(
            f"The Gaussian fitter did not register {max_sidelobe_key!r} "
            f"in data group {image_data_group_out_name!r}."
        )

    beam_fit_name = cube_group[beam_fit_key]
    max_sidelobe_name = cube_group[max_sidelobe_key]

    if beam_fit_name not in cube_xds:
        raise KeyError(
            f"The fitted-beam variable {beam_fit_name!r} is not present "
            "in the temporary cube dataset."
        )

    if max_sidelobe_name not in cube_xds:
        raise KeyError(
            f"The maximum-sidelobe variable {max_sidelobe_name!r} is not "
            "present in the temporary cube dataset."
        )

    # ------------------------------------------------------------------
    # Copy the one-channel fit back without retaining a frequency dimension.
    #
    # This avoids conflicting with any existing continuum frequency
    # coordinate. The continuum restore wrapper should expand these arrays
    # back to one frequency channel when constructing its temporary cube.
    # ------------------------------------------------------------------
    beam_fit_da = cube_xds[beam_fit_name]

    if "frequency" in beam_fit_da.dims:
        beam_fit_da = beam_fit_da.isel(frequency=0, drop=True)

    max_sidelobe_da = cube_xds[max_sidelobe_name]

    if "frequency" in max_sidelobe_da.dims:
        max_sidelobe_da = max_sidelobe_da.isel(
            frequency=0,
            drop=True,
        )

    if "beam_params_label" in cube_xds.coords:
        img_xds = img_xds.assign_coords(
            beam_params_label=cube_xds.coords["beam_params_label"],
        )

    img_xds[beam_fit_name] = beam_fit_da
    img_xds[max_sidelobe_name] = max_sidelobe_da

    # Create the output group if input and output names differ.
    if image_data_group_out_name not in img_xds.attrs["data_groups"]:
        img_xds.attrs["data_groups"][image_data_group_out_name] = deepcopy(input_group)

    output_group = img_xds.attrs["data_groups"][image_data_group_out_name]

    output_group[beam_fit_key] = beam_fit_name
    output_group[max_sidelobe_key] = max_sidelobe_name

    return img_xds, return_df


def copy_variable_without_alignment(
    destination: xr.Dataset,
    source: xr.Dataset,
    name: str,
) -> xr.Dataset:
    source_da = source[name]

    for dim, size in source_da.sizes.items():
        if dim in destination.sizes and destination.sizes[dim] != size:
            raise ValueError(
                f"{name}: incompatible size for dimension {dim!r}: "
                f"source={size}, destination={destination.sizes[dim]}"
            )

    destination[name] = xr.Variable(
        dims=source_da.dims,
        data=source_da.data,
        attrs=source_da.attrs.copy(),
    )
    return destination


@shares_param_docs
def prepare_model_uv_continuum_single_field(
    model_xds,
    image_params,
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_group_name="model",
):
    """Prepare global Fourier-domain Taylor model grids for degridding.

    This function should be called once after each global continuum model
    update. It converts the image-domain Taylor model from Stokes parameters
    to the instrumental correlation basis and Fourier-transforms every Taylor
    term.

    The returned dataset can be shared by all frequency-chunk map workers.
    Each worker then only reconstructs the model at its local frequencies and
    degrids it, avoiding repeated polarization transformations and FFTs.

    Parameters
    ----------
    model_xds : xarray.Dataset
        Image-domain continuum model dataset. The data group selected by
        ``image_data_group_name`` must register a ``sky`` variable with a
        ``taylor_term`` dimension.
    image_params : dict
        Image geometry and FFT parameters. It must contain ``nterms``.
    instrument_polarization_basis : {"linear", "circular"}, optional
        Instrumental correlation basis into which the Stokes model is
        transformed.
    single_precision_image : bool, optional
        If true, create complex64 Fourier grids. Otherwise, create complex128
        Fourier grids.
    processing_function_threads : int, optional
        Number of threads used by the polarization and FFT processing
        functions.
    fft_backend : str, optional
        FFT backend passed to ``fft_norm_continuum_img_xds``.
    image_data_group_name : str, optional
        Data group containing the model sky variable and receiving the model
        visibility grid.

    Returns
    -------
    model_uv_xds : xarray.Dataset
        Model dataset in the instrumental polarization basis containing
        ``VISIBILITY_MODEL`` with a ``taylor_term`` dimension.
    """
    import copy

    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        fft_norm_continuum_img_xds,
    )

    if not isinstance(model_xds, xr.Dataset):
        raise TypeError(
            "model_xds must be an xarray.Dataset; received "
            f"{type(model_xds).__name__}."
        )

    if instrument_polarization_basis not in ("linear", "circular"):
        raise ValueError(
            "instrument_polarization_basis must be either 'linear' or "
            f"'circular'; received {instrument_polarization_basis!r}."
        )

    nterms = int(image_params.get("nterms", 0))

    if nterms < 1:
        raise ValueError("image_params['nterms'] must be at least 1.")

    data_groups = model_xds.attrs.get("data_groups", {})

    if image_data_group_name not in data_groups:
        raise KeyError(
            f"Model data group {image_data_group_name!r} is not present in "
            "model_xds.attrs['data_groups']."
        )

    model_data_group = data_groups[image_data_group_name]
    model_sky_name = model_data_group.get("sky")

    if model_sky_name is None:
        raise KeyError(
            f"Model data group {image_data_group_name!r} does not define "
            "a 'sky' variable."
        )

    if model_sky_name not in model_xds:
        raise KeyError(
            f"Model sky variable {model_sky_name!r} is not present in " "model_xds."
        )

    model_sky = model_xds[model_sky_name]

    if "taylor_term" not in model_sky.dims:
        raise ValueError(
            f"Model sky variable {model_sky_name!r} must contain a "
            "'taylor_term' dimension."
        )

    if model_sky.sizes["taylor_term"] != nterms:
        raise ValueError(
            "The number of model Taylor terms does not match "
            "image_params['nterms']: "
            f"{model_sky.sizes['taylor_term']} != {nterms}."
        )

    complex_dtype = np.complex64 if single_precision_image else np.complex128

    # The accumulated model must remain in Stokes basis for the next model
    # update and for final restoration. Work on an independent copy.
    model_uv_xds = model_xds.copy(deep=True)
    model_uv_xds.attrs = copy.deepcopy(model_xds.attrs)

    model_uv_xds = transform_polarization_basis(
        model_uv_xds,
        new_polarization_basis=instrument_polarization_basis,
        overwrite=True,
    )

    model_uv_xds = fft_norm_continuum_img_xds(
        model_uv_xds,
        image_params=image_params,
        image_data_group_in_name=image_data_group_name,
        image_data_group_out_name=image_data_group_name,
        image_data_group_out_modified={
            "visibility": "VISIBILITY_MODEL",
        },
        image_data_variables_keep=["sky"],
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )

    output_data_groups = model_uv_xds.attrs.get("data_groups", {})

    if image_data_group_name not in output_data_groups:
        raise RuntimeError(
            "The model FFT removed or failed to create data group "
            f"{image_data_group_name!r}."
        )

    output_model_group = output_data_groups[image_data_group_name]
    model_visibility_name = output_model_group.get("visibility")

    if model_visibility_name is None:
        raise RuntimeError(
            "fft_norm_continuum_img_xds did not register a model "
            "visibility variable."
        )

    if model_visibility_name not in model_uv_xds:
        raise RuntimeError(
            "fft_norm_continuum_img_xds registered model visibility "
            f"{model_visibility_name!r}, but that variable is absent."
        )

    model_visibility = model_uv_xds[model_visibility_name]

    if "taylor_term" not in model_visibility.dims:
        raise RuntimeError(
            f"Model visibility variable {model_visibility_name!r} must "
            "contain a 'taylor_term' dimension."
        )

    if model_visibility.sizes["taylor_term"] != nterms:
        raise RuntimeError(
            "The Fourier-domain model has the wrong number of Taylor terms: "
            f"{model_visibility.sizes['taylor_term']} != {nterms}."
        )

    model_visibility.attrs.update(
        {
            "description": (
                "Global Fourier-domain continuum Taylor model used for "
                "distributed degridding."
            ),
            "nterms": nterms,
            "instrument_polarization_basis": (instrument_polarization_basis),
        }
    )

    return model_uv_xds
