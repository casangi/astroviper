from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def residual_cycle_continuum_single_field(
    ps_xdt,
    img_xds,
    model_uv_xds,
    image_params,
    is_first_cycle,
    continuum_mode="mfs",
    model_xds=None,
    primary_beam_xds=None,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    visibility_memory_mode="recompute",
    observed_visibility_grid_xds=None,
    image_data_group_in_name="model",
    image_data_group_out_name="residual",
    last_residual_cycle=False,
):
    """Grid residual products for one continuum visibility partition.

    Use prepared imaging weights. The first cycle grids observed data; later
    cycles predict the model and grid residuals, or model contributions for
    subtraction from cached observed grids. MFS produces Taylor UV planes;
    MVC produces frequency-resolved UV planes. FFT and normalization occur
    in the calling workflow.

    Parameters
    ----------
    continuum_mode : {"mfs", "mvc"}
        Continuum construction mode. ``"mfs"`` grids Taylor-weighted
        visibilities directly. Its ``SKY_MODEL`` Taylor coefficients describe
        the apparent sky, ``A_nu * I_nu``; consequently ``tt1 / tt0`` contains
        both the intrinsic source spectral index and the primary-beam spectral
        index. Its reference-frequency primary beam corrects the Taylor-zero
        restored product, but not the higher Taylor coefficients. ``"mvc"`` forms
        channel images and common-beam Taylor equations, and predicts
        ``M_nu * A_nu / Abar``. Its ``SKY_MODEL`` coefficients therefore describe
        the intrinsic spectrum under the common-beam convention, so
        ``tt1 / tt0`` excludes the primary-beam spectral index. These conventions
        match CASA's standard-gridder MT-MFS and MVC behavior. MVC images the two
        parallel hands only; ``["I", "Q", "U", "V"]`` is refused.
    model_xds : xarray.Dataset, optional
        Accumulated image-domain Taylor model used for MVC prediction.
    primary_beam_xds : xarray.Dataset, optional
        Cached primary beam for this MVC partition.
    observed_visibility_grid_xds : xarray.Dataset, optional
        Task-local cached observed visibility grid used by later residual-update
        cycles. The distributed application supplies this only for in-memory MVC
        caching; other modes and first cycles leave it unset.
    ps_xdt : xarray.DataTree or mapping
        Processing-set partition containing the observed (or residual)
        visibilities together with previously prepared imaging weights.
    img_xds : xarray.Dataset
        Continuum image dataset containing the partition-local static products.
    model_uv_xds : xarray.Dataset
        Globally prepared Fourier-domain Taylor model. This is only used after the
        first imaging cycle.
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
    is_first_cycle : bool
        True for the first imaging cycle, in which no model prediction is required.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    instrument_polarization_basis : str, optional
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` or ``"circular"``. The residual update grids and degrids the
        correlations of this basis and the model update deconvolves in the
        Stokes basis, in which the image is written. The Stokes planes requested
        in ``image_params["polarization_coords"]`` fix the correlations that are
        loaded and gridded: the two parallel hands give ``I, Q`` (linear) or
        ``I, V`` (circular), all four correlations give ``I, Q, U, V``. A sample
        is used only if none of its loaded correlations is flagged.
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the model
        update runs in single precision; the visibilities always stay double
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
    visibility_memory_mode : {"in_memory", "in_place", "recompute"}, optional
        Continuum residual-update storage policy for observed visibility grids.
        ``"in_memory"`` retains the first-cycle grid in driver memory, while
        ``"in_place"`` stores it temporarily in the image Zarr store. Later
        cycles grid only the predicted-model contribution and subtract it from
        the cached observed grid. ``"recompute"`` instead reloads the original
        observed visibilities and grids their visibility-domain residual every
        cycle. MFS caches the globally reduced Taylor UV grid; MVC caches each
        map task's exclusively owned frequency-resolved UV planes.
    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    image_data_group_out_name : str, optional
        Image data group receiving the residual Taylor products.
    last_residual_cycle : bool, optional
        Indicates whether this is the final residual calculation after the last
        model update.

    Returns
    -------
    img_xds : xarray.Dataset
        Partition-local continuum image dataset containing the residual Taylor
        products.
    return_df : pandas.DataFrame
        Timing summary for the residual-update stage.

    """
    import time

    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        prepare_model_uv_mvc_single_field,
    )
    from astroviper.processing_functions.imaging.residual_update import (
        calculate_residual_visibilities,
        make_visibility_model_single_field,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    continuum_mode = str(continuum_mode).lower()
    if continuum_mode not in ("mfs", "mvc"):
        raise ValueError(
            f"continuum_mode must be either 'mfs' or 'mvc'; received {continuum_mode!r}."
        )
    if visibility_memory_mode not in ("in_memory", "in_place", "recompute"):
        raise ValueError(
            "visibility_memory_mode must be 'in_memory', 'in_place', or "
            f"'recompute'; received {visibility_memory_mode!r}."
        )

    n_taylor_terms = int(image_params.get("n_taylor_terms", 2))
    reference_frequency = float(image_params["reference_frequency"])

    if n_taylor_terms < 1:
        raise ValueError("image_params['n_taylor_terms'] must be at least 1.")

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

    T_degrid = 0.0
    T_residual_vis = 0.0

    # ------------------------------------------------------------
    # Later imaging cycles:
    # predict the current model and form residual visibilities.
    # ------------------------------------------------------------
    if not is_first_cycle:
        if continuum_mode == "mfs":
            # delete an old residual if present
            residual_data_group = img_xds.attrs.get("data_groups", {}).get(
                image_data_group_out_name,
                {},
            )

            residual_sky_name = residual_data_group.get("sky")

            if residual_sky_name is not None and residual_sky_name in img_xds:
                img_xds.xr_img.delete_data_variables(variables=[residual_sky_name])

            degrid_start = time.time()

            make_visibility_model_continuum_single_field(
                ps_xdt,
                model_uv_xds,
                cgk_1D,
                n_taylor_terms=n_taylor_terms,
                reference_frequency=reference_frequency,
                ms_data_group_in_name=ps_data_group_name,
                ms_data_group_out_name="model",
                ms_data_group_out_modified={
                    "correlated_data": "VISIBILITY_MODEL",
                },
                image_data_group_in_name=image_data_group_in_name,
                processing_function_threads=processing_function_threads,
                fft_padding=image_params["fft_padding"],
            )

            T_degrid += time.time() - degrid_start

            if visibility_memory_mode == "recompute":
                residual_vis_start = time.time()

                calculate_residual_visibilities(
                    ps_xdt,
                    ms_data_group_out_residual="residual",
                    ms_data_group_in_model="model",
                    ms_data_group_in_observed=ps_data_group_name,
                )

                T_residual_vis += time.time() - residual_vis_start
                ps_data_group_name = "residual"
            else:
                # The globally reduced observed-data grid is retained by the
                # append node. Grid only the predicted model contribution here;
                # the append node forms observed minus model after reduction.
                ps_data_group_name = "model"

        else:
            if model_xds is None:
                raise ValueError(
                    "MVC requires the accumulated image-domain Taylor model."
                )

            if primary_beam_xds is None:
                raise ValueError(
                    "MVC requires the task-local frequency-dependent primary beam."
                )

            frequency_values = np.asarray(
                img_xds.coords["frequency"].values,
                dtype=np.float64,
            )

            degrid_start = time.time()
            local_model_uv_xds = prepare_model_uv_mvc_single_field(
                model_xds,
                primary_beam_xds,
                frequency_values,
                image_params,
                instrument_polarization_basis=(instrument_polarization_basis),
                single_precision_image=(single_precision_image),
                processing_function_threads=(processing_function_threads),
                fft_backend=fft_backend,
                image_data_group_in_name=image_data_group_in_name,
                image_data_group_out_name=image_data_group_in_name,
            )

            make_visibility_model_single_field(
                ps_xdt,
                local_model_uv_xds,
                cgk_1D,
                ms_data_group_in_name=ps_data_group_name,
                ms_data_group_out_name="model",
                ms_data_group_out_modified={
                    "correlated_data": "VISIBILITY_MODEL",
                },
                image_data_group_in_name=(image_data_group_in_name),
                processing_function_threads=(processing_function_threads),
                fft_padding=image_params["fft_padding"],
            )
            # MVC prepares its channel-dependent UV model locally immediately
            # before degridding, so report both prediction steps together.
            T_degrid += time.time() - degrid_start

            if visibility_memory_mode == "recompute":
                residual_vis_start = time.time()
                calculate_residual_visibilities(
                    ps_xdt,
                    ms_data_group_out_residual="residual",
                    ms_data_group_in_model="model",
                    ms_data_group_in_observed=ps_data_group_name,
                )

                T_residual_vis += time.time() - residual_vis_start
                ps_data_group_name = "residual"
            else:
                ps_data_group_name = "model"

    # ------------------------------------------------------------
    # Grid residual
    # ------------------------------------------------------------

    start = time.time()

    if continuum_mode == "mfs":
        from astroviper.processing_functions.imaging.make_undeconvolved_image_continuum import (
            make_undeconvolved_image_continuum_single_field,
        )

        (
            img_xds,
            make_undeconvolved_image_return_df,
        ) = make_undeconvolved_image_continuum_single_field(
            ps_xdt,
            img_xds,
            image_params,
            cgk_1D,
            n_taylor_terms=n_taylor_terms,
            reference_frequency=reference_frequency,
            ms_data_group_in_name=ps_data_group_name,
            image_data_group_out_name=image_data_group_out_name,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )

    elif continuum_mode == "mvc":
        from astroviper.processing_functions.imaging.make_undeconvolved_image_continuum_mvc import (
            make_undeconvolved_image_mvc_single_field,
        )

        (
            img_xds,
            make_undeconvolved_image_return_df,
        ) = make_undeconvolved_image_mvc_single_field(
            ps_xdt,
            img_xds,
            image_params,
            cgk_1D,
            ms_data_group_in_name=ps_data_group_name,
            image_data_group_out_name=image_data_group_out_name,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )

        if not is_first_cycle and visibility_memory_mode != "recompute":
            if observed_visibility_grid_xds is None:
                raise ValueError(
                    "Cached-grid MVC cycles require a task-local observed "
                    "visibility grid."
                )
            from astroviper.processing_functions.imaging.image_continuum_single_field import (
                form_residual_grid_from_cache,
            )

            img_xds = form_residual_grid_from_cache(
                observed_visibility_grid_xds,
                img_xds,
                image_data_group_in_observed=image_data_group_out_name,
                image_data_group_in_model=image_data_group_out_name,
                grid_kind="MVC",
            )
            img_xds.attrs["continuum_mode"] = "mvc"

    else:
        raise ValueError(
            f"continuum_mode must be either 'mfs' or 'mvc'; received {continuum_mode!r}."
        )

    T_grid = time.time() - start

    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_degrid": [T_degrid],
            "T_residual_vis": [T_residual_vis],
            "T_grid": [T_grid],
            "n_taylor_terms": [n_taylor_terms],
            "is_first_cycle": [bool(is_first_cycle)],
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
    n_taylor_terms,
    reference_frequency,
    ms_data_group_in_name="base",
    ms_data_group_out_name="model",
    ms_data_group_out_modified=None,
    image_data_group_in_name="model",
    processing_function_threads=1,
    fft_padding=1.2,
):
    """Predict channel-dependent model visibilities from a Fourier Taylor model.

    The continuum imaging cycle maintains the sky model as Fourier-transformed Taylor
    coefficients rather than frequency images. This function reconstructs the model
    visibility at the actual observing frequencies of one processing-set
    partition. Each Taylor grid is degridded once and its channel-dependent
    Taylor contribution is accumulated in visibility space.

    For each frequency channel ν, the model visibility grid is reconstructed as

        M(u,v,ν) =
            Σ_t M_t(u,v)
                ((ν-ν_ref)/ν_ref)^t,

    where ``M_t`` denotes the Fourier-domain Taylor coefficient and ``ν_ref`` is
    the common continuum reference frequency.

    This avoids materializing a channel-sized UV-grid cube. The shared standard
    degridding primitive samples each Taylor plane for every visibility channel,
    after which the corresponding spectral weights are applied to its output.

    This reconstruction is repeated independently for every visibility partition,
    whereas the Fourier Taylor model itself is prepared only once after each global
    model-update update.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing-set partition receiving the predicted model visibilities.
    model_xds : xarray.Dataset
        Globally prepared Fourier-domain Taylor model. The selected image data
        group must register a visibility variable containing the Taylor
        coefficients.
    cgk_1D : numpy.ndarray
        One-dimensional prolate-spheroidal convolution kernel.
    n_taylor_terms : int
        Number of Taylor terms to reconstruct.
    reference_frequency : float
        Common MT-MFS reference frequency in Hz.
    ms_data_group_in_name : str, optional
        Processing-set data group being imaged (the caller's
        ``processing_set_data_group_name``). The degridder reads its ``uvw``
        and the shape of its ``correlated_data`` (or of its ``weight_imaging``
        when the observed visibilities are not loaded), and the output group
        inherits its remaining roles.
    ms_data_group_out_name : str, optional
        Processing-set data group receiving the predicted model visibilities.
    ms_data_group_out_modified : dict, optional
        Mapping describing the output visibility variable. Defaults to
        ``{"correlated_data": "VISIBILITY_MODEL"}``.
    image_data_group_in_name : str, optional
        Image data group containing the Fourier Taylor model.
    processing_function_threads : int, optional
        Number of threads supplied to the standard degridder.
    fft_padding : float, optional
        FFT padding factor used to interpret the continuum UV-grid geometry.

    Returns
    -------
    None

    Notes
    -----
    For each processing-set partition this continuum implementation degrids each
    stored Fourier Taylor coefficient through the shared standard primitive. It
    combines the results in visibility space and does not route model prediction
    through the cube processing API.

    The Fourier-domain Taylor model is prepared only once after each global
    continuum model update. Consequently, this function performs only the spectral
    reconstruction and degridding required for the current visibility partition.
    """
    import numpy as np

    from astroviper.processing_functions.imaging.degrid_visibility_grid import (
        degrid_visibility_grid_single_field,
    )

    # ------------------------------------------------------------
    # Load grids from memory and perform sanity checks
    # ------------------------------------------------------------

    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {
            "correlated_data": "VISIBILITY_MODEL",
        }
    output_visibility_name = ms_data_group_out_modified["correlated_data"]

    n_taylor_terms = int(n_taylor_terms)
    reference_frequency = float(reference_frequency)

    if n_taylor_terms < 1:
        raise ValueError("n_taylor_terms must be at least 1.")

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

    if available_nterms < n_taylor_terms:
        raise ValueError(
            f"The model contains {available_nterms} Taylor terms, but "
            f"n_taylor_terms={n_taylor_terms} was requested."
        )

    # Ignore any extra Taylor terms that may be present.
    model_taylor_grid = model_taylor_grid.isel(taylor_term=slice(0, n_taylor_terms))
    model_grid_dtype = np.dtype(model_taylor_grid.dtype)
    if not np.issubdtype(model_grid_dtype, np.complexfloating):
        raise TypeError(
            "The Fourier Taylor model must use complex64 or complex128; "
            f"received {model_grid_dtype}."
        )
    if model_grid_dtype.itemsize == np.dtype(np.complex64).itemsize:
        native_grid_dtype = np.complex64
    elif model_grid_dtype.itemsize == np.dtype(np.complex128).itemsize:
        native_grid_dtype = np.complex128
    else:
        raise TypeError(
            "The Fourier Taylor model must use complex64 or complex128; "
            f"received {model_grid_dtype}."
        )

    # ------------------------------------------------------------
    # Wrapper around cube implementation, may need refactoring into a native continuum implementation later
    # ------------------------------------------------------------

    for _, ms_xdt in ps_xdt.items():
        if "frequency" in ms_xdt.coords:
            frequency = ms_xdt.coords["frequency"]
        elif "frequency" in ms_xdt:
            frequency = ms_xdt["frequency"]
        else:
            raise KeyError(
                "The measurement-set dataset does not contain a 'frequency' coordinate."
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
                "The measurement-set frequency coordinate contains non-finite values."
            )

        spectral_coordinate = (
            frequency_values - reference_frequency
        ) / reference_frequency
        frequency_map = np.zeros(frequency_values.size, dtype=np.int64)
        accumulated_model = None

        for taylor_term in range(n_taylor_terms):
            taylor_grid = model_taylor_grid.isel(
                taylor_term=taylor_term,
                drop=True,
            )
            preferred_dimension_order = [
                dimension
                for dimension in ("time", "polarization", "u", "v")
                if dimension in taylor_grid.dims
            ]
            remaining_dimensions = [
                dimension
                for dimension in taylor_grid.dims
                if dimension not in preferred_dimension_order
            ]
            taylor_grid = taylor_grid.transpose(
                *preferred_dimension_order,
                *remaining_dimensions,
            )

            # A single Taylor plane supplies every channel; the spectral
            # dependence is applied afterward in the much smaller visibility
            # domain instead of constructing a channel-sized UV-grid cube.
            single_plane_grid = np.ascontiguousarray(
                np.expand_dims(taylor_grid.values, axis=1),
                dtype=native_grid_dtype,
            )
            degrid_visibility_grid_single_field(
                ms_xdt,
                cgk_1D,
                model_xds,
                single_plane_grid,
                frequency_map,
                ms_data_group_in_name=ms_data_group_in_name,
                ms_data_group_out_name=ms_data_group_out_name,
                ms_data_group_out_modified=ms_data_group_out_modified,
                overwrite=True,
                fft_padding=fft_padding,
                processing_function_threads=processing_function_threads,
                description=(
                    "Continuum Taylor model sampled with the standard degridder."
                ),
            )

            predicted_model = ms_xdt[output_visibility_name].values
            if accumulated_model is None:
                # The zeroth Taylor weight is one for every channel.
                accumulated_model = predicted_model.copy()
                continue

            weight_shape = [1] * predicted_model.ndim
            weight_shape[ms_xdt[output_visibility_name].get_axis_num("frequency")] = (
                frequency_values.size
            )
            np.multiply(
                predicted_model,
                (spectral_coordinate**taylor_term).reshape(weight_shape),
                out=predicted_model,
            )
            accumulated_model += predicted_model

        ms_xdt[output_visibility_name].data[...] = accumulated_model

    return
