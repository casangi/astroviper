import xarray as xr

from astroviper.utils.param_docs import shares_param_docs

###############################################################################
# Generic helper functions
###############################################################################


def copy_variable_without_alignment(
    destination: xr.Dataset,
    source: xr.Dataset,
    name: str,
) -> xr.Dataset:
    """Copy a data variable without coordinate alignment.

    This helper copies the numerical values and metadata of a data variable from
    one :class:`xarray.Dataset` to another while preserving the destination
    dataset's coordinates and indexing.

    Unlike a normal xarray assignment, which aligns arrays by coordinate labels,
    this function performs a positional copy of the underlying array data. This is
    useful when the source and destination datasets are known to have identical
    array layouts but differ in coordinates or auxiliary metadata.

    Parameters
    ----------
    destination_xds : xarray.Dataset
        Dataset receiving the copied variable.

    source_xds : xarray.Dataset
        Dataset providing the variable to copy.

    variable_name : str
        Name of the data variable to copy.

    Returns
    -------
    xarray.Dataset
        The destination dataset with the copied variable. Variable attributes are
        preserved, while the destination dataset's coordinates and data-group
        definitions remain unchanged.

    Notes
    -----
    This helper should only be used when the source and destination variables are
    known to have identical dimensions and shapes. No coordinate alignment or
    broadcasting is performed."""

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


###############################################################################
# Processing Function level functionality related to the residual update
###############################################################################


@shares_param_docs
def imaging_preparation_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    specmode="mfs",
    processing_set_data_group_name="corrected",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    task_id=0,
):
    """Run the once-per-chunk continuum imaging preparation.

    This helper performs the setup work required only during the first major
    cycle for one frequency-chunk map task. It delegates to
    :func:`imaging_setup_continuum_single_field`, which prepares the chunk-local
    visibility and UV-sampling products needed by the subsequent continuum
    residual update.

    The setup stage may include

    * calculating and registering imaging weights;
    * constructing the chunk-local visibility and UV-sampling grids;
    * creating the corresponding normalization products;
    * creating the primary beam when requested;
    * establishing the continuum coordinates, metadata, and image data groups
      required by later processing stages.

    No model visibility prediction, residual visibility calculation, global
    reduction, inverse FFT, minor cycle, or restoration is performed here. Those
    operations are handled by later stages of the continuum imaging workflow.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this frequency chunk.

    img_xds : xarray.Dataset
        Empty image dataset carrying the geometry and coordinates for this chunk.

    image_params : dict
        Image geometry and continuum configuration. This includes the image size,
        cell size, reference frequency, number of Taylor terms, and gridding or FFT
        configuration required by the setup processing functions.

    imaging_weights_params : dict
        Imaging-weight configuration.

    processing_set_data_group_name : str, optional
        Processing-set data group used as the input for imaging.

    single_precision_image : bool, optional
        Whether continuum image and grid products use single precision.

    processing_function_threads : int, optional
        Number of threads supplied to lower-level processing functions.

    fft_backend : str, optional
        FFT backend supplied to setup functions that require Fourier transforms.

    image_data_variables_keep : list of str, optional
        Logical image products to retain in the returned dataset.

    task_id : int, optional
        Identifier of the frequency-chunk task, used for logging.

    Returns
    -------
    img_xds : xarray.Dataset
        Chunk-local continuum dataset containing the setup products required by
        the residual-update and reduce stages.

    return_df : pandas.DataFrame
        Timing information returned by
        :func:`imaging_setup_continuum_single_field`.

    T_setup : float
        Total wall-clock duration of the setup call, in seconds.
    """

    import time

    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
        imaging_setup_continuum_single_field,
    )

    logger.debug("Processing continuum chunk " + str(task_id))

    start = time.time()

    # wrap around imaging setup
    img_xds, return_df = imaging_setup_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        specmode=specmode,
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
def prepare_model_uv_continuum_single_field(
    model_xds,
    image_params,
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_group_name="model",
):
    """Prepare the Fourier-domain continuum model for the next major cycle.

    This function is called once after each continuum minor cycle. It converts the
    updated image-domain continuum model from the Stokes basis into the
    instrumental correlation basis and Fourier-transforms every Taylor term to
    produce the corresponding model visibility grids.

    The resulting Fourier-domain model is shared by all frequency-chunk map
    workers during the subsequent major cycle. Each worker reconstructs the model
    visibilities at its local frequencies and degrids them directly, avoiding
    repeated polarization transformations and FFTs on every worker.

    Parameters
    ----------
    model_xds : xarray.Dataset
        Image-domain continuum model dataset. The data group selected by
        ``image_data_group_name`` must contain a ``sky`` variable with a
        ``taylor_term`` dimension.

    image_params : dict
        Continuum imaging configuration, including the image geometry, FFT
        parameters, and the number of Taylor terms.

    instrument_polarization_basis : {"linear", "circular"}, optional
        Instrumental correlation basis into which the Stokes model is
        transformed before the Fourier transform.

    single_precision_image : bool, optional
        If ``True``, create ``complex64`` Fourier grids; otherwise create
        ``complex128`` grids.

    processing_function_threads : int, optional
        Number of threads supplied to the polarization transformation and FFT
        routines.

    fft_backend : str, optional
        FFT backend passed to ``fft_norm_continuum_img_xds``.

    image_data_group_name : str, optional
        Data group containing the continuum sky model and receiving the Fourier-
        domain model visibility grids.

    Returns
    -------
    model_uv_xds : xarray.Dataset
        Continuum model dataset in the instrumental correlation basis containing
        the Fourier-domain Taylor coefficients (``VISIBILITY_MODEL``), indexed by
        ``taylor_term``.

    Notes
    -----
    This function performs the expensive polarization transformation and Fourier
    transform only once per major cycle. The resulting Fourier-domain model is
    reused by every distributed map task, substantially reducing the computational
    cost of continuum degridding.
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


def convert_mvc_cubes_to_taylor_normal_equations(
    residual_cube,
    psf_cube,
    primary_beam_cube,
    residual_normalization,
    psf_normalization,
    *,
    nterms,
    reference_frequency,
    pblimit=0.2,
):
    """Convert MVC channel cubes into CASA-style MT-MFS normal equations.

    The channel residuals are first remapped from the frequency-dependent
    primary-beam convention to one common effective beam.  The returned
    residual Taylor planes are weighted normal-equation right-hand sides, not
    pixel-wise polynomial coefficients.  The PSF cube is converted with the
    same spectral basis into ``2 * nterms - 1`` Hessian planes.

    Parameters
    ----------
    residual_cube : xarray.DataArray
        Channel residual images with dimensions
        ``(time, frequency, polarization, l, m)``.
    psf_cube : xarray.DataArray
        Channel PSF images with the same dimensions as ``residual_cube``.
    primary_beam_cube : xarray.DataArray
        Frequency-dependent primary beam with the same dimensions.
    residual_normalization : xarray.DataArray
        Per-channel residual sum of imaging weights with dimensions
        ``(time, frequency, polarization)``.
    psf_normalization : xarray.DataArray
        Per-channel PSF sum of imaging weights with dimensions
        ``(time, frequency, polarization)``.
    nterms : int
        Number of sky-model Taylor coefficients.
    reference_frequency : float
        Taylor reference frequency in Hz.
    pblimit : float, default 0.2
        Primary-beam validity cutoff.

    Returns
    -------
    residual_taylor : xarray.DataArray
        ``nterms`` Taylor normal-equation right-hand sides.
    psf_taylor : xarray.DataArray
        ``2 * nterms - 1`` Taylor PSF/Hessian planes.
    effective_primary_beam : xarray.DataArray
        Imaging-weighted common primary beam, without a frequency dimension.

    Notes
    -----
    With ``x_nu = (nu - nu0) / nu0`` and common beam ``Abar``, this implements

    ``R'_nu = Abar * R_nu / A_nu``

    followed by

    ``b_t = sum(w_nu * x_nu**t * A_nu * R'_nu) / (sum(w_nu) * Abar)``

    and

    ``B_k = sum(w_nu * x_nu**k * A_nu * B_nu) / (sum(w_nu) * Abar)``.

    The first expression reduces algebraically to the weighted residual moment,
    while retaining CASA's primary-beam mask and common-beam convention.
    """
    import numpy as np
    import xarray as xr

    required_cube_dims = ("time", "frequency", "polarization", "l", "m")
    required_normalization_dims = ("time", "frequency", "polarization")

    cubes = {
        "residual_cube": residual_cube,
        "psf_cube": psf_cube,
        "primary_beam_cube": primary_beam_cube,
    }
    for name, cube in cubes.items():
        if cube.dims != required_cube_dims:
            raise ValueError(
                f"{name} has dimensions {cube.dims}; expected {required_cube_dims}."
            )

    normalizations = {
        "residual_normalization": residual_normalization,
        "psf_normalization": psf_normalization,
    }
    for name, normalization in normalizations.items():
        if normalization.dims != required_normalization_dims:
            raise ValueError(
                f"{name} has dimensions {normalization.dims}; expected "
                f"{required_normalization_dims}."
            )

    try:
        xr.align(
            residual_cube,
            psf_cube,
            primary_beam_cube,
            join="exact",
            copy=False,
        )
        xr.align(
            residual_normalization,
            psf_normalization,
            residual_cube,
            join="exact",
            copy=False,
            exclude={"l", "m"},
        )
    except ValueError as exc:
        raise ValueError(
            "MVC cubes and normalizations are not coordinate-aligned."
        ) from exc

    nterms = int(nterms)
    reference_frequency = float(reference_frequency)
    pblimit = float(pblimit)

    if nterms < 1:
        raise ValueError(f"nterms must be positive; received {nterms}.")
    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError("reference_frequency must be finite and positive.")
    if not np.isfinite(pblimit) or pblimit < 0.0:
        raise ValueError("pblimit must be finite and non-negative.")

    frequency = np.asarray(residual_cube.frequency.values, dtype=np.float64)
    if frequency.size < nterms:
        raise ValueError(
            f"MVC requires at least nterms={nterms} channels; received "
            f"{frequency.size}."
        )

    x_frequency = xr.DataArray(
        (frequency - reference_frequency) / reference_frequency,
        dims=("frequency",),
        coords={"frequency": residual_cube.frequency},
    )

    def _clean_weights(normalization):
        return xr.where(
            np.isfinite(normalization) & (normalization > 0.0),
            normalization.astype(np.float64),
            0.0,
        )

    residual_weight = _clean_weights(residual_normalization)
    psf_weight = _clean_weights(psf_normalization)
    residual_weight_sum = residual_weight.sum(dim="frequency")
    psf_weight_sum = psf_weight.sum(dim="frequency")

    if bool((residual_weight_sum <= 0.0).any()):
        raise ValueError("MVC residual normalization has an empty image plane.")
    if bool((psf_weight_sum <= 0.0).any()):
        raise ValueError("MVC PSF normalization has an empty image plane.")

    finite_primary_beam = xr.where(
        np.isfinite(primary_beam_cube),
        primary_beam_cube,
        0.0,
    )
    effective_primary_beam = (
        (finite_primary_beam * psf_weight).sum(dim="frequency") / psf_weight_sum
    ).transpose("time", "polarization", "l", "m")
    effective_primary_beam.attrs = {
        "description": "Imaging-weighted effective primary beam for MVC.",
        "specmode": "mvc",
        "pblimit": pblimit,
    }

    valid_channel_pb = np.isfinite(primary_beam_cube) & (primary_beam_cube > pblimit)
    remapped_residual = xr.where(
        valid_channel_pb,
        effective_primary_beam * residual_cube / primary_beam_cube,
        0.0,
    )
    valid_effective_pb = np.isfinite(effective_primary_beam) & (
        effective_primary_beam > pblimit
    )

    residual_terms = []
    for order in range(nterms):
        numerator = (
            residual_weight
            * x_frequency**order
            * finite_primary_beam
            * remapped_residual
        ).sum(dim="frequency")
        denominator = residual_weight_sum * effective_primary_beam
        term = xr.where(valid_effective_pb, numerator / denominator, 0.0)
        residual_terms.append(term)

    residual_taylor = xr.concat(
        residual_terms,
        dim=xr.IndexVariable("taylor_term", np.arange(nterms, dtype=np.int64)),
    ).transpose("time", "taylor_term", "polarization", "l", "m")
    residual_taylor.attrs = {
        "description": "MVC Taylor residual normal-equation right-hand sides.",
        "specmode": "mvc",
        "reference_frequency": reference_frequency,
        "nterms": nterms,
        "pblimit": pblimit,
    }

    n_psf_taylor_terms = 2 * nterms - 1
    psf_terms = []
    finite_psf = xr.where(np.isfinite(psf_cube), psf_cube, 0.0)
    for order in range(n_psf_taylor_terms):
        numerator = (
            psf_weight * x_frequency**order * finite_primary_beam * finite_psf
        ).sum(dim="frequency")
        denominator = psf_weight_sum * effective_primary_beam
        term = xr.where(valid_effective_pb, numerator / denominator, 0.0)
        psf_terms.append(term)

    psf_taylor = xr.concat(
        psf_terms,
        dim=xr.IndexVariable(
            "psf_taylor_order",
            np.arange(n_psf_taylor_terms, dtype=np.int64),
        ),
    ).transpose("time", "psf_taylor_order", "polarization", "l", "m")
    psf_taylor.attrs = {
        "description": "MVC Taylor PSF/Hessian planes formed from the PSF cube.",
        "type": "point_spread_function",
        "specmode": "mvc",
        "reference_frequency": reference_frequency,
        "nterms": nterms,
        "n_psf_taylor_terms": n_psf_taylor_terms,
        "pblimit": pblimit,
    }

    return residual_taylor, psf_taylor, effective_primary_beam


def apply_mvc_primary_beam_convention(
    model_cube,
    primary_beam_cube,
    effective_primary_beam,
):
    """Map a common-beam MVC model cube to the channel-beam convention.

    This applies ``M'_nu = M_nu * A_nu / Abar`` and sets pixels with a
    non-finite or non-positive effective primary beam to zero.
    """
    import numpy as np

    required_cube_dims = ("time", "frequency", "polarization", "l", "m")
    required_beam_dims = ("time", "polarization", "l", "m")

    if model_cube.dims != required_cube_dims:
        raise ValueError(
            f"model_cube has dimensions {model_cube.dims}; expected "
            f"{required_cube_dims}."
        )
    if primary_beam_cube.dims != required_cube_dims:
        raise ValueError(
            "primary_beam_cube has dimensions "
            f"{primary_beam_cube.dims}; expected {required_cube_dims}."
        )
    if effective_primary_beam.dims != required_beam_dims:
        raise ValueError(
            "effective_primary_beam has dimensions "
            f"{effective_primary_beam.dims}; expected {required_beam_dims}."
        )

    if model_cube.shape != primary_beam_cube.shape:
        raise ValueError(
            "MVC model and channel primary beam have incompatible shapes: "
            f"{model_cube.shape} and {primary_beam_cube.shape}."
        )
    expected_effective_shape = (
        model_cube.sizes["time"],
        model_cube.sizes["polarization"],
        model_cube.sizes["l"],
        model_cube.sizes["m"],
    )
    if effective_primary_beam.shape != expected_effective_shape:
        raise ValueError(
            "MVC model and effective primary beam have incompatible shapes: "
            f"{model_cube.shape} and {effective_primary_beam.shape}."
        )

    # The Taylor model is in Stokes coordinates while the cached scalar PB is
    # still labelled with the instrumental correlations.  The airy-disk PB is
    # identical plane-by-plane, so validate every physical coordinate except
    # the polarization labels and apply it positionally.
    for coordinate in ("time", "frequency", "l", "m"):
        if not np.array_equal(
            model_cube.coords[coordinate].values,
            primary_beam_cube.coords[coordinate].values,
        ):
            raise ValueError(
                f"MVC model and channel primary beam {coordinate} coordinates "
                "are not aligned."
            )
    for coordinate in ("time", "l", "m"):
        if not np.array_equal(
            model_cube.coords[coordinate].values,
            effective_primary_beam.coords[coordinate].values,
        ):
            raise ValueError(
                f"MVC model and effective primary beam {coordinate} coordinates "
                "are not aligned."
            )

    effective_pb_data = np.asarray(effective_primary_beam.data)[:, None, ...]
    valid_effective_pb = np.isfinite(effective_pb_data) & (effective_pb_data > 0.0)
    result = model_cube.copy(
        data=np.where(
            valid_effective_pb,
            np.asarray(model_cube.data)
            * np.asarray(primary_beam_cube.data)
            / effective_pb_data,
            0.0,
        )
    )
    result.attrs = model_cube.attrs.copy()
    result.attrs["primary_beam_convention"] = "channel_pb_over_effective_pb"
    return result


@shares_param_docs
def prepare_model_uv_mvc_single_field(
    model_xds,
    primary_beam_xds,
    frequency_values,
    image_params,
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_group_name="model",
):
    """Construct a local MVC model UV cube in the channel-PB convention.

    The Taylor model uses the common effective primary-beam convention.  Before
    prediction this function evaluates the Taylor polynomial at each channel and
    applies ``A_nu / Abar``, matching CASA's MVC major-cycle convention.
    """
    import copy

    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        fft_norm_img_xds,
    )
    from astroviper.utils.data_group_tools import modify_data_groups_xds

    frequency_values = np.asarray(
        frequency_values,
        dtype=np.float64,
    )

    reference_frequency = float(
        image_params.get(
            "reference_frequency",
            image_params["reference_frequency_hz"],
        )
    )
    nterms = int(image_params["nterms"])

    model_name = model_xds.attrs["data_groups"][image_data_group_name]["sky"]

    model_taylor = model_xds[model_name].isel(taylor_term=slice(0, nterms))

    x = (frequency_values - reference_frequency) / reference_frequency

    basis = xr.DataArray(
        x[:, None] ** np.arange(nterms)[None, :],
        dims=("frequency", "taylor_term"),
        coords={
            "frequency": frequency_values,
            "taylor_term": model_taylor.coords["taylor_term"],
        },
    )

    # Result:
    # (time, frequency, polarization, l, m)
    # Reconstruct the frequency-dependent image cube from the
    # image-domain Taylor model.
    model_cube = (model_taylor * basis).sum(dim="taylor_term")

    # Xarray appends the new frequency dimension after the existing
    # model dimensions. Restore the canonical image-cube ordering.
    model_cube = model_cube.transpose(
        "time",
        "frequency",
        "polarization",
        "l",
        "m",
    )

    pb_name = primary_beam_xds.attrs.get(
        "primary_beam_name",
        "PRIMARY_BEAM",
    )
    primary_beam = primary_beam_xds[pb_name]

    if primary_beam.sizes["frequency"] != len(frequency_values):
        raise ValueError(
            "The cached MVC PB cube does not match the " "local model frequency axis."
        )

    primary_beam = primary_beam.transpose(
        "time",
        "frequency",
        "polarization",
        "l",
        "m",
    )

    if model_cube.shape != primary_beam.shape:
        raise ValueError(
            "The reconstructed MVC model cube and cached primary "
            "beam have incompatible shapes: "
            f"model={model_cube.shape}, PB={primary_beam.shape}."
        )

    if "PRIMARY_BEAM" not in model_xds:
        raise KeyError(
            "MVC model prediction requires the effective PRIMARY_BEAM carried "
            "with the Taylor model."
        )

    effective_primary_beam = model_xds["PRIMARY_BEAM"]
    if "frequency" not in effective_primary_beam.dims:
        raise ValueError("MVC effective PRIMARY_BEAM must contain frequency.")
    if effective_primary_beam.sizes["frequency"] != 1:
        raise ValueError("MVC effective PRIMARY_BEAM must have one frequency plane.")

    effective_primary_beam = effective_primary_beam.isel(
        frequency=0,
        drop=True,
    ).transpose("time", "polarization", "l", "m")

    model_cube = apply_mvc_primary_beam_convention(
        model_cube,
        primary_beam,
        effective_primary_beam,
    )

    mvc_xds = model_xds.drop_vars(
        list(model_xds.data_vars),
        errors="ignore",
    ).copy(deep=False)

    if "taylor_term" in mvc_xds.dims:
        mvc_xds = mvc_xds.drop_dims(
            "taylor_term",
            errors="ignore",
        )

    mvc_xds = mvc_xds.assign_coords(frequency=frequency_values)

    mvc_xds["SKY_MODEL_MVC"] = xr.DataArray(
        model_cube.data,
        dims=(
            "time",
            "frequency",
            "polarization",
            "l",
            "m",
        ),
        coords={
            "time": model_cube.coords["time"],
            "frequency": frequency_values,
            "polarization": model_cube.coords["polarization"],
            "l": model_cube.coords["l"],
            "m": model_cube.coords["m"],
        },
    )

    modify_data_groups_xds(
        mvc_xds,
        data_group_out_name=image_data_group_name,
        data_group_out={
            "sky": "SKY_MODEL_MVC",
        },
        description=(
            "MVC model converted from the common effective-beam convention "
            "to the channel-dependent primary-beam convention."
        ),
    )

    mvc_xds = transform_polarization_basis(
        mvc_xds,
        new_polarization_basis=(instrument_polarization_basis),
        overwrite=True,
    )

    complex_dtype = np.complex64 if single_precision_image else np.complex128

    mvc_xds = fft_norm_img_xds(
        mvc_xds,
        image_params=image_params,
        image_data_group_in_name=(image_data_group_name),
        image_data_group_out_name=(image_data_group_name),
        image_data_group_out_modified={
            "visibility": "VISIBILITY_MODEL",
        },
        image_data_variables_keep=["sky"],
        processing_function_threads=(processing_function_threads),
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )

    return mvc_xds


@shares_param_docs
def residual_update_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    specmode="mfs",
    primary_beam_xds=None,
    processing_set_data_group_name="corrected",
    deconvolver="hogbom",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    is_n_iter_0=True,
    model_xds=None,
    model_uv_xds=None,
    task_id=0,
):
    """Perform one continuum major-cycle update for a single frequency chunk.

    This function is the primary processing entry point executed by the continuum
    map node task. It performs the operations required to compute the chunk-local
    continuum products that are later accumulated by the GraphViper reduce stage.

    During the first major cycle, the function first executes the one-time imaging
    preparation for the frequency chunk before computing the initial residual
    products. During subsequent major cycles, it reuses the existing imaging
    geometry, updates the imaging weights when required, and computes a new
    residual using the globally prepared Fourier-domain continuum model.

    The residual-update stage predicts the model visibilities (except during the
    first major cycle), subtracts them from the observed visibilities, grids the
    resulting residual visibilities into Taylor-weighted UV-domain products, and
    returns those products for global reduction. For MVC, the channel-resolved
    residual and first-cycle PSF grids are normalized and inverse transformed
    locally before being returned. MFS retains the globally reduced inverse-FFT
    path. No minor cycle or restoration is performed here.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this frequency chunk.

    img_xds : xarray.Dataset
        Chunk-local continuum image dataset used to accumulate the UV-domain
        products.

    image_params : dict
        Image geometry and continuum imaging configuration.

    imaging_weights_params : dict
        Imaging-weight configuration.

    processing_set_data_group_name : str, optional
        Processing-set data group to image.

    deconvolver : str, optional
        Reserved for future continuum deconvolution implementations.

    instrument_polarization_basis : {"linear", "circular"}, optional
        Instrument correlation basis used during gridding and degridding.

    single_precision_image : bool, optional
        Whether continuum products use single precision.

    processing_function_threads : int, optional
        Number of threads supplied to the lower-level processing functions.

    fft_backend : str, optional
        FFT backend used by the underlying processing functions.

    image_data_variables_keep : list of str, optional
        Logical image products retained in the returned dataset.

    is_n_iter_0 : bool, optional
        Indicates whether this is the first major cycle.

    model_uv_xds : xarray.Dataset, optional
        Globally prepared Fourier-domain Taylor model used for degridding during
        all major cycles after the first.

    task_id : int, optional
        Identifier of the current frequency chunk.

    Returns
    -------
    img_xds : xarray.Dataset
        Chunk-local continuum dataset containing the UV-domain products required
        by the GraphViper reduce stage.

    timing_df : pandas.DataFrame
        Timing summary for the setup (when applicable) and residual-update
        processing performed by this function."""

    import time

    import numpy as np
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
            specmode=specmode,
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

        # Needs to be refactored at a later point when decided what to do with weights
        # start = time.time()

        # calculate_imaging_weights(
        #    ps_xdt,
        #    img_xds,
        #    imaging_weights_params=imaging_weights_params,
        #    return_weight_density_grid=False,
        #    ms_data_group_in_name=processing_set_data_group_name,
        #    ms_data_group_out_name=processing_set_data_group_name,
        #    ms_data_group_out_modified={
        #        "weight_imaging": "WEIGHT_IMAGING",
        #    },
        #    processing_function_threads=processing_function_threads,
        # )

        # T_weights = time.time() - start

        setup_return_df = pd.DataFrame({})

        # timing["T_prep"] = T_weights
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
        specmode=specmode,
        model_xds=model_xds,
        primary_beam_xds=primary_beam_xds,
        processing_set_data_group_name=processing_set_data_group_name,
        instrument_polarization_basis=instrument_polarization_basis,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
    )

    timing["T_residual_cycle"] = time.time() - start
    accumulate_timing(timing, residual_return_df)

    # MVC channels are exclusively owned by this map task. Normalize and
    # inverse-transform them here so the distributed reducer carries cropped
    # image cubes rather than padded UV grids. The normalization arrays remain
    # in the dataset because the global Taylor conversion reuses them as
    # spectral weights.
    timing["T_local_ifft"] = 0.0

    if str(specmode).lower() == "mvc":
        from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
            ifft_norm_img_xds,
        )

        start = time.time()

        img_xds = ifft_norm_img_xds(
            img_xds,
            image_params=image_params,
            image_data_group_in_name="residual",
            image_data_group_out_name="residual",
            image_data_group_out_modified={
                "sky": "SKY_RESIDUAL_MVC_CUBE",
            },
            image_data_variables_keep=[],
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            complex_dtype=(np.complex64 if single_precision_image else np.complex128),
        )

        if "SKY_RESIDUAL_MVC_CUBE" not in img_xds:
            raise RuntimeError(
                "The MVC map task did not create its channel residual image cube."
            )

        if is_n_iter_0:
            img_xds = ifft_norm_img_xds(
                img_xds,
                image_params=image_params,
                image_data_group_in_name="residual",
                image_data_group_out_name="residual",
                image_data_group_out_modified={
                    "point_spread_function": "POINT_SPREAD_FUNCTION_MVC_CUBE",
                },
                image_data_variables_keep=[],
                processing_function_threads=processing_function_threads,
                fft_backend=fft_backend,
                complex_dtype=(
                    np.complex64 if single_precision_image else np.complex128
                ),
            )

            if "POINT_SPREAD_FUNCTION_MVC_CUBE" not in img_xds:
                raise RuntimeError(
                    "The first MVC map task did not create its channel PSF image "
                    "cube."
                )

        timing["T_local_ifft"] = time.time() - start

    timing["n_channels"] = img_xds.sizes.get(
        "frequency",
        len(img_xds.coords.get("frequency", [])),
    )
    timing["nterms"] = image_params.get("nterms", 2)

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df


###############################################################################
# Processing Function level functionality related to the model update
###############################################################################


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
    """Perform one continuum minor-cycle model update.

    This function implements the current continuum deconvolution backend used by
    the distributed MT-MFS imaging workflow. Until a native MT-MFS deconvolver is
    available, the minor cycle is performed by temporarily projecting the
    continuum dataset onto a single-frequency cube representation and reusing the
    existing cube Högbom implementation.

    The procedure is

    continuum image dataset
        │
        ├── select residual Taylor term 0
        ├── select PSF Taylor order 0
        ├── select model Taylor term 0
        │
        ▼
    construct temporary one-channel cube dataset
    (time, frequency=1, polarization, l, m)
        │
        ▼
    call existing cube model-update implementation
        │
        ├── construct the deconvolution mask (if required)
        ├── determine CLEAN components
        ├── update the temporary cube model
        └── return deconvolution statistics
        │
        ▼
    copy the updated model back into
    SKY_MODEL[taylor_term=0]

    Only the zeroth Taylor coefficient is modified during the minor cycle.
    Higher-order Taylor model terms are intentionally left unchanged and are
    updated indirectly through the subsequent major cycle.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Globally reduced continuum image dataset. The residual and model images
        are expected to use the ``taylor_term`` dimension, while the point-spread
        function uses ``psf_taylor_order``.

    deconvolver : str
        Name of the continuum deconvolver. Currently only ``"hogbom"`` is
        supported.

    deconvolve_params : dict
        Minor-cycle control parameters. These typically include entries such as
        ``cycleniter``, ``cyclethreshold``, ``niter_per_plane``, and
        ``cyclethreshold_per_plane``.

    is_n_iter_0 : bool, optional
        Indicates whether this is the first minor cycle.

    processing_function_threads : int, optional
        Number of threads supplied to the deconvolution backend.

    image_data_group_in_name : str, optional
        Name of the residual image data group.

    image_data_group_out_name : str, optional
        Name of the output model data group.

    Returns
    -------
    deconvolve_dict : ReturnDict
        Deconvolution statistics returned by the Högbom implementation.

    return_df : pandas.DataFrame
        Timing information for the continuum minor cycle.

    Notes
    -----
    This function is a compatibility layer that allows the continuum imaging
    pipeline to reuse the existing cube deconvolution backend. Although the
    surrounding imaging algorithm is MT-MFS, the current minor cycle operates
    only on the zeroth Taylor coefficient. A future native MT-MFS deconvolver
    will replace this implementation."""
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


###############################################################################
# Processing function level functionality to initialize and finish imaging
###############################################################################


def primary_beam_correct_restored_continuum(
    img_xds,
    *,
    pblimit=0.2,
    primary_beam_name="PRIMARY_BEAM",
    restored_data_group_name="restored",
    output_data_group_name="restored_pbcor",
    output_variable_name="SKY_RESTORED_PBCOR",
):
    """PB-correct the restored Taylor-zero continuum image.

    Only the final restored reference-frequency intensity image is corrected.
    The model, residual Taylor stack, higher Taylor terms, and minor-cycle
    calculations remain in the apparent-sky convention.
    """
    import numpy as np
    import xarray as xr

    from astroviper.utils.data_group_tools import modify_data_groups_xds

    if not 0.0 <= float(pblimit) < 1.0:
        raise ValueError(
            f"pblimit must satisfy 0 <= pblimit < 1; " f"received {pblimit}."
        )

    data_groups = img_xds.attrs.get("data_groups", {})

    if restored_data_group_name not in data_groups:
        raise KeyError(
            f"Restored data group {restored_data_group_name!r} " "is missing."
        )

    restored_name = data_groups[restored_data_group_name].get("sky")

    if restored_name is None or restored_name not in img_xds:
        raise KeyError(
            f"Restored data group {restored_data_group_name!r} "
            "does not contain an accessible sky image."
        )

    if primary_beam_name not in img_xds:
        raise KeyError(f"Primary-beam variable {primary_beam_name!r} is missing.")

    restored = img_xds[restored_name]
    primary_beam = img_xds[primary_beam_name]

    # Restoration operates on the Taylor-zero continuum plane.  The restored
    # dataset may retain the model's Taylor axis for bookkeeping, but PB
    # correction is a reference-frequency image product rather than a Taylor
    # stack, so discard that singleton science selection explicitly.
    if "taylor_term" in restored.dims:
        restored = restored.isel(taylor_term=0, drop=True)

    # The averaged PB should no longer have a frequency axis. This check
    # catches accidental retention of a chunk-local PB cube.
    if "frequency" in primary_beam.dims:
        if primary_beam.sizes["frequency"] != 1:
            raise ValueError(
                "The continuum PB correction requires one averaged "
                "primary beam, not a multi-channel PB cube."
            )

        primary_beam = primary_beam.isel(
            frequency=0,
            drop=True,
        )

    valid = np.isfinite(primary_beam) & (primary_beam >= float(pblimit))

    corrected = xr.where(
        valid,
        restored / primary_beam,
        np.nan,
    )

    corrected.attrs = restored.attrs.copy()
    corrected.attrs.update(
        {
            "description": (
                "Restored Taylor-zero continuum intensity divided by "
                "the globally averaged primary beam."
            ),
            "primary_beam_corrected": True,
            "primary_beam_variable": primary_beam_name,
            "pblimit": float(pblimit),
        }
    )

    img_xds[output_variable_name] = corrected

    modify_data_groups_xds(
        img_xds,
        data_group_out_name=output_data_group_name,
        data_group_out={
            "sky": output_variable_name,
        },
        description=("Primary-beam-corrected restored continuum intensity."),
    )

    return img_xds


def restore_image(
    img_xds,
    image_data_group_in_residual_name="residual",
    image_data_group_in_model_name="model",
    image_data_group_out_restore_name="restored",
    processing_function_threads=1,
):
    """Restore the continuum image.

    This function restores the final continuum image by temporarily projecting the
    continuum dataset onto a one-channel cube representation and reusing the
    existing cube restoration implementation.

    The procedure is

    continuum image dataset
        │
        ├── select residual Taylor term 0
        ├── select model Taylor term 0
        ├── select PSF Taylor order 0
        └── copy the fitted restoring-beam parameters
        │
        ▼
    construct temporary one-channel cube dataset
    (time, frequency=1, polarization, l, m)
        │
        ▼
    call existing cube restoration implementation
        │
        ├── convolve the model with the restoring beam
        ├── add the residual image
        └── produce the restored image
        │
        ▼
    copy the restored image back into the
    continuum dataset

    The restored image corresponds to the reference-frequency (Taylor-zero)
    continuum image.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Continuum image dataset containing the residual image, sky model,
        point-spread function, and fitted restoring-beam parameters.

    image_data_group_in_residual_name : str, optional
        Name of the residual image data group.

    image_data_group_in_model_name : str, optional
        Name of the sky-model data group.

    image_data_group_out_restore_name : str, optional
        Name under which the restored image is registered.

    processing_function_threads : int, optional
        Number of threads supplied to the cube restoration backend.

    Returns
    -------
    img_xds : xarray.Dataset
        Continuum dataset with the restored Taylor-zero image added.

    timing_df : pandas.DataFrame
        Timing information returned by the restoration backend.

    Notes
    -----
    This function is a compatibility layer around the existing cube restoration
    routine. Although the surrounding imaging algorithm is MT-MFS, the current
    restoration operates only on the zeroth Taylor coefficient and therefore
    produces the restored reference-frequency continuum image."""

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
    """Fit the restoring beam from the continuum point-spread function.

    This function determines the restoring beam by temporarily projecting the
    continuum point-spread function onto a one-channel cube representation and
    reusing the existing cube Gaussian-fitting implementation.

    The procedure is

    continuum image dataset
        │
        ├── select PSF Taylor order 0
        │
        ▼
    construct temporary one-channel cube dataset
    (time, frequency=1, polarization, l, m)
        │
        ▼
    call existing cube Gaussian-fit implementation
        │
        ├── fit the restoring beam
        ├── determine the maximum PSF sidelobe
        └── return beam-fit statistics
        │
        ▼
    copy the fitted restoring-beam parameters and
    maximum PSF sidelobe back into the
    continuum dataset

    Only the zeroth Taylor-order point-spread function is used for the beam fit.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Continuum image dataset containing the point-spread function.

    image_data_group_in_name : str, optional
        Name of the data group containing the continuum point-spread function.

    image_data_group_out_name : str, optional
        Name of the data group in which the fitted restoring-beam parameters are
        registered.

    processing_function_threads : int, optional
        Number of threads supplied to the Gaussian-fitting backend.

    Returns
    -------
    img_xds : xarray.Dataset
        Continuum dataset with the fitted restoring-beam parameters and maximum
        PSF sidelobe added.

    return_df : pandas.DataFrame
        Timing information returned by the Gaussian-fitting backend.

    Notes
    -----
    This function is a compatibility layer around the existing cube
    ``point_spread_function_gaussian_fit`` implementation. The restoring beam is
    determined exclusively from the zeroth Taylor-order point-spread function,
    which is the standard convention for MT-MFS imaging."""

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
