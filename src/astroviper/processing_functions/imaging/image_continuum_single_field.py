from astroviper.utils.continuum_cache import (
    MVC_OBSERVED_NORMALIZATION_CACHE as _MVC_OBSERVED_NORMALIZATION_CACHE,
)
from astroviper.utils.continuum_cache import (
    MVC_OBSERVED_VISIBILITY_CACHE as _MVC_OBSERVED_VISIBILITY_CACHE,
)
from astroviper.utils.param_docs import shares_param_docs


def _continuum_frequency_coord(img_xds):
    """Return the canonical singleton frequency coordinate for Taylor products."""
    import numpy as np

    continuum_metadata = img_xds.attrs.get("continuum_imaging", {})
    reference_frequency = continuum_metadata.get("reference_frequency_hz")
    if reference_frequency is None:
        reference_frequency = img_xds.attrs.get("reference_frequency_hz")
    if reference_frequency is None:
        reference_frequency = img_xds.attrs.get("reference_frequency")
    if reference_frequency is None and img_xds.sizes.get("frequency", 0) > 0:
        reference_frequency = np.asarray(img_xds.frequency.values).reshape(-1)[0]
    if reference_frequency is None:
        reference_frequency = 0.0
    return np.asarray([float(reference_frequency)], dtype=np.float64)


def _continuum_plane_to_frequency(img_xds, data_array, plane_dimension):
    """Project plane zero of a Taylor axis onto one canonical frequency plane."""
    frequency_coord = _continuum_frequency_coord(img_xds)
    plane = data_array.isel({plane_dimension: 0}, drop=True)
    if "frequency" in plane.dims:
        if plane.sizes["frequency"] != 1:
            plane = plane.isel(frequency=0, drop=False)
        return plane.assign_coords(frequency=frequency_coord)
    axis = plane.dims.index("time") + 1 if "time" in plane.dims else 0
    return plane.expand_dims(frequency=frequency_coord, axis=axis)


def _mtmfs_principal_residual_for_restoration(residual_da, psf_da):
    """Return a principal-solution residual copy for MT-MFS restoration."""
    import numpy as np

    from astroviper.processing_functions.imaging.deconvolvers import mtmfs

    residual_dims = ("time", "taylor_term", "polarization", "l", "m")
    psf_dims = ("time", "psf_taylor_order", "polarization", "l", "m")
    missing_residual_dims = [
        dim for dim in residual_dims if dim not in residual_da.dims
    ]
    missing_psf_dims = [dim for dim in psf_dims if dim not in psf_da.dims]
    if missing_residual_dims:
        raise ValueError(
            "MT-MFS restoration requires residual dimensions "
            f"{residual_dims}; missing {missing_residual_dims}."
        )
    if missing_psf_dims:
        raise ValueError(
            "MT-MFS restoration requires PSF dimensions "
            f"{psf_dims}; missing {missing_psf_dims}."
        )

    residual_ordered = residual_da.transpose(*residual_dims)
    psf_ordered = psf_da.transpose(*psf_dims)
    n_taylor_terms = residual_ordered.sizes["taylor_term"]
    expected_psf_terms = 2 * n_taylor_terms - 1
    if psf_ordered.sizes["psf_taylor_order"] != expected_psf_terms:
        raise ValueError(
            "MT-MFS restoration requires 2 * n_taylor_terms - 1 PSF Taylor "
            f"orders; received {psf_ordered.sizes['psf_taylor_order']} for "
            f"n_taylor_terms={n_taylor_terms}."
        )

    principal_values = np.array(residual_ordered.values, copy=True, order="C")
    psf_values = psf_ordered.values
    for time_index in range(residual_ordered.sizes["time"]):
        for polarization_index in range(residual_ordered.sizes["polarization"]):
            psf_stack = np.ascontiguousarray(
                psf_values[time_index, :, polarization_index]
            )
            inverse_hessian = mtmfs.hessian(psf_stack)["inverse_hessian"][0]
            residual_stack = np.ascontiguousarray(
                principal_values[time_index, :, polarization_index]
            )
            finite_pixels = np.all(np.isfinite(residual_stack), axis=0)
            residual_stack[:, ~finite_pixels] = 0
            mtmfs.principal_solution(residual_stack, inverse_hessian)
            residual_stack[:, ~finite_pixels] = np.nan
            principal_values[time_index, :, polarization_index] = residual_stack

    principal = residual_ordered.copy(data=principal_values)
    return principal.transpose(*residual_da.dims)


###############################################################################
# Imaging setup and PSF beam fitting
###############################################################################


@shares_param_docs
def imaging_preparation_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    continuum_mode="mfs",
    processing_set_data_group_name="corrected",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    task_id=0,
):
    """Prepare static products once per continuum map task.

    Require existing imaging weights and delegate metadata, PSF sampling-grid,
    and primary-beam setup to :func:`imaging_setup_continuum_single_field`.

    Parameters
    ----------
    continuum_mode : {"mfs", "mvc"}
        Select Taylor UV gridding or frequency-resolved MVC processing.
    ps_xdt : xarray.DataTree
        Visibility data for this frequency chunk.

    img_xds : xarray.Dataset
        Empty image dataset carrying the geometry and coordinates for this chunk.

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

    imaging_weights_params : dict
        Weighting configuration: ``weighting`` is ``"natural"``, ``"uniform"``,
        ``"briggs"`` or ``"briggs_abs"``; ``robust`` controls the Briggs
        trade-off. ``weighting_scope`` is ``"local"`` for a density per
        frequency channel or ``"global"`` for one density across all selected
        channels in distributed continuum imaging. Natural weighting is
        independent of this scope.

    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).

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

    task_id : int, optional
        Identifier of the parallel chunk being processed.

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
        continuum_mode=continuum_mode,
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

    cube_psf_da = _continuum_plane_to_frequency(img_xds, psf_da, "psf_taylor_order")

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


###############################################################################
# Model prediction: MFS and MVC
###############################################################################


@shares_param_docs
def prepare_model_uv_continuum_single_field(
    model_xds,
    image_params,
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_group_in_name="model",
    image_data_group_out_name="model",
):
    """Prepare the Fourier-domain continuum model for the next imaging cycle.

    This function is called once after each continuum model update. It converts the
    updated image-domain continuum model from the Stokes basis into the
    instrumental correlation basis and Fourier-transforms every Taylor term to
    produce the corresponding model visibility grids.

    The resulting Fourier-domain model is shared by all frequency-chunk map
    workers during the subsequent imaging cycle. Each worker reconstructs the model
    visibilities at its local frequencies and degrids them directly, avoiding
    repeated polarization transformations and FFTs on every worker.

    Parameters
    ----------
    model_xds : xarray.Dataset
        Image-domain continuum model dataset. The data group selected by
        ``image_data_group_in_name`` must contain a ``sky`` variable with a
        ``taylor_term`` dimension.

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

    instrument_polarization_basis : {"linear", "circular"}, optional
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

    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    image_data_group_out_name : str, optional
        Data group receiving the Fourier-domain model visibility grids.

    Returns
    -------
    model_uv_xds : xarray.Dataset
        Continuum model dataset in the instrumental correlation basis containing
        the Fourier-domain Taylor coefficients (``VISIBILITY_MODEL``), indexed by
        ``taylor_term``.

    Notes
    -----
    This function performs the expensive polarization transformation and Fourier
    transform only once per imaging cycle. The resulting Fourier-domain model is
    reused by every distributed map task, substantially reducing the computational
    cost of continuum degridding.
    """
    import copy

    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder import (
        fft_norm_continuum_img_xds,
    )

    if not isinstance(model_xds, xr.Dataset):
        raise TypeError(
            f"model_xds must be an xarray.Dataset; received {type(model_xds).__name__}."
        )

    if instrument_polarization_basis not in ("linear", "circular"):
        raise ValueError(
            "instrument_polarization_basis must be either 'linear' or "
            f"'circular'; received {instrument_polarization_basis!r}."
        )

    n_taylor_terms = int(image_params.get("n_taylor_terms", 0))

    if n_taylor_terms < 1:
        raise ValueError("image_params['n_taylor_terms'] must be at least 1.")

    data_groups = model_xds.attrs.get("data_groups", {})

    if image_data_group_in_name not in data_groups:
        raise KeyError(
            f"Model data group {image_data_group_in_name!r} is not present in "
            "model_xds.attrs['data_groups']."
        )

    model_data_group = data_groups[image_data_group_in_name]
    model_sky_name = model_data_group.get("sky")

    if model_sky_name is None:
        raise KeyError(
            f"Model data group {image_data_group_in_name!r} does not define "
            "a 'sky' variable."
        )

    if model_sky_name not in model_xds:
        raise KeyError(
            f"Model sky variable {model_sky_name!r} is not present in model_xds."
        )

    model_sky = model_xds[model_sky_name]

    if "taylor_term" not in model_sky.dims:
        raise ValueError(
            f"Model sky variable {model_sky_name!r} must contain a "
            "'taylor_term' dimension."
        )

    if model_sky.sizes["taylor_term"] != n_taylor_terms:
        raise ValueError(
            "The number of model Taylor terms does not match "
            "image_params['n_taylor_terms']: "
            f"{model_sky.sizes['taylor_term']} != {n_taylor_terms}."
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
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
        image_data_group_out_modified={
            "visibility": "VISIBILITY_MODEL",
        },
        # The accumulated image-domain model is returned separately. Keeping
        # it here would double the large state copied to each map task.
        image_data_variables_keep=[],
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )

    output_data_groups = model_uv_xds.attrs.get("data_groups", {})

    if image_data_group_out_name not in output_data_groups:
        raise RuntimeError(
            "The model FFT removed or failed to create data group "
            f"{image_data_group_out_name!r}."
        )

    output_model_group = output_data_groups[image_data_group_out_name]
    model_visibility_name = output_model_group.get("visibility")

    if model_visibility_name is None:
        raise RuntimeError(
            "fft_norm_continuum_img_xds did not register a model visibility variable."
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

    if model_visibility.sizes["taylor_term"] != n_taylor_terms:
        raise RuntimeError(
            "The Fourier-domain model has the wrong number of Taylor terms: "
            f"{model_visibility.sizes['taylor_term']} != {n_taylor_terms}."
        )

    model_visibility.attrs.update(
        {
            "description": (
                "Global Fourier-domain continuum Taylor model used for "
                "distributed degridding."
            ),
            "n_taylor_terms": n_taylor_terms,
            "instrument_polarization_basis": (instrument_polarization_basis),
        }
    )

    return model_uv_xds


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
    image_data_group_in_name="model",
    image_data_group_out_name="model",
):
    """Construct a local MVC model UV cube in the channel-PB convention.

    The Taylor model uses the common effective primary-beam convention.  Before
    prediction this function evaluates the Taylor polynomial at each channel and
    applies ``A_nu / Abar``, matching CASA's MVC imaging-cycle convention.
    """

    import copy

    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder import (
        fft_norm_img_xds,
    )
    from astroviper.utils.data_group_tools import modify_data_groups_xds

    frequency_values = np.asarray(
        frequency_values,
        dtype=np.float64,
    )

    reference_frequency = float(image_params["reference_frequency"])
    n_taylor_terms = int(image_params["n_taylor_terms"])

    model_name = model_xds.attrs["data_groups"][image_data_group_in_name]["sky"]

    model_taylor = model_xds[model_name].isel(taylor_term=slice(0, n_taylor_terms))

    x = (frequency_values - reference_frequency) / reference_frequency

    basis = xr.DataArray(
        x[:, None] ** np.arange(n_taylor_terms)[None, :],
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
            "The cached MVC PB cube does not match the local model frequency axis."
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
    # Xarray's shallow dataset operations also share the nested attrs mapping.
    # Isolate it before registering the temporary MVC sky and visibility roles.
    mvc_xds.attrs = copy.deepcopy(model_xds.attrs)

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
        data_group_out_name=image_data_group_out_name,
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
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        image_data_group_out_modified={
            "visibility": "VISIBILITY_MODEL",
        },
        image_data_variables_keep=["sky"],
        processing_function_threads=(processing_function_threads),
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )

    return mvc_xds


###############################################################################
# MVC Taylor contributions and normalization
###############################################################################


def make_mvc_taylor_normal_equation_contributions(
    residual_cube,
    psf_cube,
    primary_beam_cube,
    residual_normalization,
    psf_normalization,
    *,
    n_taylor_terms,
    reference_frequency,
    primary_beam_limit=0.2,
):
    """Form additive MVC Taylor products for one map-task frequency chunk.

    The returned dataset has no frequency dimension and can therefore be
    combined by element-wise addition in the distributed reduce stage.  The
    primary-beam weighting and Taylor moment construction happen here; only the
    normalization by globally accumulated weights, the per-pixel spectral fit
    and the construction of the effective beam remain for the append node.

    Channel residuals are multiplied by their frequency-dependent primary beam
    before the weighted Taylor moments are formed, and the squared beam is
    summed into the ``2 * n_taylor_terms - 1`` moments of the per-pixel spectral fit.
    The PSF cube is converted with the same spectral basis, without a
    primary-beam factor, into ``2 * n_taylor_terms - 1`` Hessian numerators.

    Parameters
    ----------
    residual_cube : xarray.DataArray
        Channel residual images with dimensions
        ``(time, frequency, polarization, l, m)``.
    psf_cube : xarray.DataArray or None
        Channel PSF images with the same dimensions as ``residual_cube``. Use
        ``None`` after the first imaging cycle.
    primary_beam_cube : xarray.DataArray
        Frequency-dependent primary beam with the same dimensions.
    residual_normalization : xarray.DataArray
        Per-channel residual sum of imaging weights with dimensions
        ``(time, frequency, polarization)``.
    psf_normalization : xarray.DataArray or None
        Per-channel PSF sum of imaging weights with dimensions
        ``(time, frequency, polarization)``. Required with ``psf_cube``.
    n_taylor_terms : int
        Number of sky-model Taylor coefficients.
    reference_frequency : float
        Taylor reference frequency in Hz.
    primary_beam_limit : float, default 0.2
        Primary-beam cutoff. Channel beams at or below it do not contribute to
        the effective beam, and the residual is zero where the effective beam
        is at or below it. Where fewer than ``n_taylor_terms`` channel beams exceed it,
        only Taylor term zero of the spectrum is fitted.

    Returns
    -------
    xarray.Dataset
        Additive residual Taylor numerators, squared-beam moments,
        residual-weight moments, residual-weight sum, and the number of
        channels whose beam exceeds ``primary_beam_limit``.  When a PSF cube is supplied,
        also contains PSF Taylor numerators, the PSF weight sum, and the
        PSF-weighted primary-beam sum.

    Notes
    -----
    With ``x_nu = (nu - nu0) / nu0``, channel beams ``A_nu`` and the common
    beam ``Abar = sum(w_nu * A_nu) / sum(w_nu)`` (channel beams at or below
    ``primary_beam_limit`` count as zero), the PSF planes are the plain weighted moments
    of the channel PSFs,

    ``B_k = sum(w_nu * x_nu**k * B_nu) / sum(w_nu)``,

    and the Taylor residuals are

    ``b = Abar * H @ inv(G) @ u``,

    with ``u_k = sum(w_nu * x_nu**k * A_nu * R_nu)``,
    ``G_tk = sum(w_nu * x_nu**(t+k) * A_nu**2)`` at each pixel, and
    ``H_tk = sum(w_nu * x_nu**(t+k)) / sum(w_nu)``, the PSF planes at their
    peak.  ``inv(G) @ u`` is the least-squares fit of a Taylor polynomial to
    the primary-beam corrected channel residuals ``R_nu / A_nu`` with weights
    ``w_nu * A_nu**2``, so a channel enters with a factor ``A_nu`` and a
    channel whose beam goes to zero drops out.  The model is predicted as
    ``M_nu * A_nu / Abar`` (:func:`apply_mvc_primary_beam_convention`), so a
    model change ``dM_nu = sum_k x_nu**k dm_k`` at a pixel changes ``u`` there
    by ``-G @ dm / Abar`` and ``b`` by ``-H @ dm``: the Hessian the
    deconvolver inverts, ``B_(t+k)`` at the PSF peak, is the response of the
    residual at every position, not only at the pointing centre.  Away from
    the peak the response follows the channel PSFs weighted by
    ``w_nu * A_nu**2`` instead of ``w_nu``.  Where fewer than ``n_taylor_terms``
    channel beams exceed ``primary_beam_limit``, the fit is not determined by them;
    there only Taylor term zero is fitted, ``u_0 / G_00``.

    The sums over all chunks and ``Abar`` are known only after the reduce
    stage; :func:`finalize_mvc_taylor_normal_equations` solves for ``b`` in
    the append node.
    """
    import numpy as np
    import xarray as xr

    required_cube_dims = ("time", "frequency", "polarization", "l", "m")
    required_normalization_dims = ("time", "frequency", "polarization")

    cubes = {
        "residual_cube": residual_cube,
        "primary_beam_cube": primary_beam_cube,
    }
    if psf_cube is not None:
        cubes["psf_cube"] = psf_cube
    for name, cube in cubes.items():
        if cube.dims != required_cube_dims:
            raise ValueError(
                f"{name} has dimensions {cube.dims}; expected {required_cube_dims}."
            )

    normalizations = {"residual_normalization": residual_normalization}
    if psf_cube is not None:
        if psf_normalization is None:
            raise ValueError("psf_normalization is required when psf_cube is supplied.")
        normalizations["psf_normalization"] = psf_normalization
    for name, normalization in normalizations.items():
        if normalization.dims != required_normalization_dims:
            raise ValueError(
                f"{name} has dimensions {normalization.dims}; expected "
                f"{required_normalization_dims}."
            )

    try:
        xr.align(residual_cube, primary_beam_cube, join="exact", copy=False)
        xr.align(
            residual_normalization,
            residual_cube,
            join="exact",
            copy=False,
            exclude={"l", "m"},
        )
        if psf_cube is not None:
            xr.align(residual_cube, psf_cube, join="exact", copy=False)
            xr.align(
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

    n_taylor_terms = int(n_taylor_terms)
    reference_frequency = float(reference_frequency)
    primary_beam_limit = float(primary_beam_limit)

    if n_taylor_terms < 1:
        raise ValueError(f"n_taylor_terms must be positive; received {n_taylor_terms}.")
    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError("reference_frequency must be finite and positive.")
    if not np.isfinite(primary_beam_limit) or primary_beam_limit < 0.0:
        raise ValueError("primary_beam_limit must be finite and non-negative.")

    frequency = np.asarray(residual_cube.frequency.values, dtype=np.float64)
    if frequency.size == 0:
        raise ValueError("An MVC map contribution requires at least one channel.")

    x_frequency = xr.DataArray(
        (frequency - reference_frequency) / reference_frequency,
        dims=("frequency",),
        coords={"frequency": residual_cube.frequency},
    )

    # Accumulate weights in float64 for numerical stability, then return image
    # products at the precision requested by the input cube.
    output_dtype = (
        np.float32
        if all(
            np.dtype(cube.dtype).itemsize <= np.dtype(np.float32).itemsize
            for cube in cubes.values()
        )
        else np.float64
    )

    def _clean_weights(normalization):
        return xr.where(
            np.isfinite(normalization) & (normalization > 0.0),
            normalization.astype(np.float64),
            0.0,
        )

    residual_weight = _clean_weights(residual_normalization)
    residual_weight_sum = residual_weight.sum(dim="frequency")

    valid_channel_pb = np.isfinite(primary_beam_cube) & (
        primary_beam_cube > primary_beam_limit
    )
    # CASA clips each channel PB before forming the effective PB. Keep all
    # channel weights in the denominator, including clipped ones.
    finite_primary_beam = xr.where(valid_channel_pb, primary_beam_cube, 0.0)
    # The spectral fit weights each channel by its beam as the prediction
    # applies it, without the clip.
    channel_primary_beam = xr.where(
        np.isfinite(primary_beam_cube), primary_beam_cube, 0.0
    )
    beam_weighted_residual = channel_primary_beam * residual_cube
    channel_primary_beam_squared = channel_primary_beam**2

    residual_terms = []
    for order in range(n_taylor_terms):
        numerator = (residual_weight * x_frequency**order * beam_weighted_residual).sum(
            dim="frequency"
        )
        residual_terms.append(numerator)

    residual_numerator = xr.concat(
        residual_terms,
        dim=xr.IndexVariable("taylor_term", np.arange(n_taylor_terms, dtype=np.int64)),
    ).transpose("time", "taylor_term", "polarization", "l", "m")
    residual_numerator.attrs = {
        "description": "Additive MVC Taylor residual numerators.",
        "continuum_mode": "mvc",
        "reference_frequency": reference_frequency,
        "n_taylor_terms": n_taylor_terms,
        "primary_beam_limit": primary_beam_limit,
    }

    n_psf_taylor_terms = 2 * n_taylor_terms - 1
    psf_taylor_order = xr.IndexVariable(
        "psf_taylor_order", np.arange(n_psf_taylor_terms, dtype=np.int64)
    )
    beam_squared_numerator = xr.concat(
        [
            (residual_weight * x_frequency**order * channel_primary_beam_squared).sum(
                dim="frequency"
            )
            for order in range(n_psf_taylor_terms)
        ],
        dim=psf_taylor_order,
    ).transpose("time", "psf_taylor_order", "polarization", "l", "m")
    beam_squared_numerator.attrs = {
        "description": "Additive MVC squared-beam moments of the spectral fit.",
        "continuum_mode": "mvc",
        "reference_frequency": reference_frequency,
        "n_taylor_terms": n_taylor_terms,
        "primary_beam_limit": primary_beam_limit,
    }
    residual_weight_moments = xr.concat(
        [
            (residual_weight * x_frequency**order).sum(dim="frequency")
            for order in range(n_psf_taylor_terms)
        ],
        dim=psf_taylor_order,
    ).transpose("time", "psf_taylor_order", "polarization")

    contributions = xr.Dataset(
        {
            "MVC_RESIDUAL_TAYLOR_NUMERATOR": residual_numerator,
            "MVC_RESIDUAL_WEIGHT_SUM": residual_weight_sum,
            "MVC_RESIDUAL_BEAM_SQUARED_TAYLOR_SUM": beam_squared_numerator,
            "MVC_RESIDUAL_WEIGHT_TAYLOR_SUM": residual_weight_moments,
            "MVC_RESIDUAL_BEAM_CHANNEL_COUNT": (
                (valid_channel_pb & (residual_weight > 0.0))
                .sum(dim="frequency")
                .transpose("time", "polarization", "l", "m")
            ),
        },
        attrs={
            "continuum_mode": "mvc",
            "n_taylor_terms": n_taylor_terms,
            "reference_frequency": reference_frequency,
            "primary_beam_limit": primary_beam_limit,
            "mvc_output_dtype": np.dtype(output_dtype).name,
        },
    )

    if psf_cube is not None:
        psf_weight = _clean_weights(psf_normalization)
        psf_weight_sum = psf_weight.sum(dim="frequency")

        finite_psf = xr.where(np.isfinite(psf_cube), psf_cube, 0.0)
        psf_terms = [
            (psf_weight * x_frequency**order * finite_psf).sum(dim="frequency")
            for order in range(n_psf_taylor_terms)
        ]
        psf_numerator = xr.concat(
            psf_terms,
            dim=xr.IndexVariable(
                "psf_taylor_order",
                np.arange(n_psf_taylor_terms, dtype=np.int64),
            ),
        ).transpose("time", "psf_taylor_order", "polarization", "l", "m")
        psf_numerator.attrs = {
            "description": "Additive MVC Taylor PSF numerators.",
            "continuum_mode": "mvc",
            "reference_frequency": reference_frequency,
            "n_taylor_terms": n_taylor_terms,
            "n_psf_taylor_terms": n_psf_taylor_terms,
            "primary_beam_limit": primary_beam_limit,
        }
        contributions["MVC_PSF_TAYLOR_NUMERATOR"] = psf_numerator
        contributions["MVC_PSF_WEIGHT_SUM"] = psf_weight_sum
        contributions["MVC_PRIMARY_BEAM_WEIGHTED_SUM"] = (
            (finite_primary_beam * psf_weight)
            .sum(dim="frequency")
            .transpose("time", "polarization", "l", "m")
        )

    return contributions


def finalize_mvc_taylor_normal_equations(
    contributions,
    *,
    effective_primary_beam=None,
):
    """Normalize globally reduced MVC map contributions in the append node."""
    import numpy as np
    import xarray as xr

    residual_sum = contributions["MVC_RESIDUAL_WEIGHT_SUM"]
    if bool((residual_sum <= 0.0).any()):
        raise ValueError(
            "MVC residual normalization has an empty image plane: "
            f"values={np.asarray(residual_sum.values)!r}."
        )

    if effective_primary_beam is None:
        psf_sum = contributions["MVC_PSF_WEIGHT_SUM"]
        if bool((psf_sum <= 0.0).any()):
            raise ValueError("MVC PSF normalization has an empty image plane.")
        effective_primary_beam = (
            contributions["MVC_PRIMARY_BEAM_WEIGHTED_SUM"] / psf_sum
        ).transpose("time", "polarization", "l", "m")

    primary_beam_limit = float(contributions.attrs.get("primary_beam_limit", 0.2))
    valid_effective_pb = np.isfinite(effective_primary_beam) & (
        effective_primary_beam > primary_beam_limit
    )

    # Fit the Taylor coefficients at each pixel, inv(G) @ u, and return them
    # to the PSF moments and the common beam, Abar * H @ inv(G) @ u. Where
    # fewer than n_taylor_terms channel beams exceed primary_beam_limit, fit Taylor term zero only.
    pixel_dims = ("time", "polarization", "l", "m")
    numerator = contributions["MVC_RESIDUAL_TAYLOR_NUMERATOR"].transpose(
        *pixel_dims, "taylor_term"
    )
    n_taylor_terms = numerator.sizes["taylor_term"]
    hankel = np.add.outer(np.arange(n_taylor_terms), np.arange(n_taylor_terms))
    beam_squared = (
        contributions["MVC_RESIDUAL_BEAM_SQUARED_TAYLOR_SUM"]
        .transpose(*pixel_dims, "psf_taylor_order")
        .values
    )
    supported = valid_effective_pb.transpose(*pixel_dims).values
    determined = supported & (
        contributions["MVC_RESIDUAL_BEAM_CHANNEL_COUNT"].transpose(*pixel_dims).values
        >= n_taylor_terms
    )
    gram = beam_squared[..., hankel]
    gram[~determined] = np.eye(n_taylor_terms)
    coefficients = np.linalg.solve(
        gram, np.where(determined[..., None], numerator.values, 0.0)[..., None]
    )[..., 0]
    flat = supported & ~determined & (beam_squared[..., 0] > 0.0)
    coefficients[flat, 0] = numerator.values[flat, 0] / beam_squared[flat, 0]
    hessian = (
        (contributions["MVC_RESIDUAL_WEIGHT_TAYLOR_SUM"] / residual_sum)
        .transpose("time", "polarization", "psf_taylor_order")
        .values[..., hankel]
    )
    residual_taylor = xr.where(
        valid_effective_pb,
        effective_primary_beam
        * contributions["MVC_RESIDUAL_TAYLOR_NUMERATOR"].copy(
            data=np.einsum("tpij,tplmj->tiplm", hessian, coefficients)
        ),
        0.0,
    )

    psf_taylor = None
    if "MVC_PSF_TAYLOR_NUMERATOR" in contributions:
        psf_sum = contributions["MVC_PSF_WEIGHT_SUM"]
        psf_taylor = contributions["MVC_PSF_TAYLOR_NUMERATOR"] / psf_sum

    output_dtype = np.dtype(contributions.attrs.get("mvc_output_dtype", "float64"))
    residual_taylor = residual_taylor.astype(output_dtype, copy=False)
    effective_primary_beam = effective_primary_beam.astype(output_dtype, copy=False)
    if psf_taylor is not None:
        psf_taylor = psf_taylor.astype(output_dtype, copy=False)

    residual_taylor.attrs = {
        "description": "MVC Taylor residual normal-equation right-hand sides.",
        "continuum_mode": "mvc",
        "reference_frequency": contributions.attrs.get("reference_frequency"),
        "n_taylor_terms": contributions.attrs.get("n_taylor_terms"),
        "primary_beam_limit": primary_beam_limit,
    }
    effective_primary_beam.attrs = {
        "description": "Imaging-weighted effective primary beam for MVC.",
        "continuum_mode": "mvc",
        "primary_beam_limit": primary_beam_limit,
    }
    if psf_taylor is not None:
        psf_taylor.attrs = {
            "description": "MVC Taylor PSF/Hessian planes formed in map tasks.",
            "type": "point_spread_function",
            "continuum_mode": "mvc",
            "reference_frequency": contributions.attrs.get("reference_frequency"),
            "n_taylor_terms": contributions.attrs.get("n_taylor_terms"),
            "n_psf_taylor_terms": psf_taylor.sizes["psf_taylor_order"],
            "primary_beam_limit": primary_beam_limit,
        }

    return residual_taylor, psf_taylor, effective_primary_beam


def convert_mvc_cubes_to_taylor_normal_equations(
    residual_cube,
    psf_cube,
    primary_beam_cube,
    residual_normalization,
    psf_normalization,
    *,
    n_taylor_terms,
    reference_frequency,
    primary_beam_limit=0.2,
):
    """Convert complete MVC channel cubes into Taylor normal equations.

    This compatibility entry point composes the map-contribution and append
    finalization helpers.  Distributed imaging calls those two stages
    separately so channel cubes never enter the reduce stage.
    """
    if residual_cube.sizes.get("frequency", 0) < int(n_taylor_terms):
        raise ValueError(
            f"MVC requires at least n_taylor_terms={int(n_taylor_terms)} channels; received "
            f"{residual_cube.sizes.get('frequency', 0)}."
        )
    contributions = make_mvc_taylor_normal_equation_contributions(
        residual_cube,
        psf_cube,
        primary_beam_cube,
        residual_normalization,
        psf_normalization,
        n_taylor_terms=n_taylor_terms,
        reference_frequency=reference_frequency,
        primary_beam_limit=primary_beam_limit,
    )
    return finalize_mvc_taylor_normal_equations(contributions)


###############################################################################
# Residual updates and observed-grid caching
###############################################################################


def form_residual_grid_from_cache(
    observed_grid_xds,
    model_grid_xds,
    image_data_group_in_observed="residual",
    image_data_group_in_model="residual",
    grid_kind="continuum",
):
    """Form a residual UV grid from cached observed and model grids.

    The inputs may be globally reduced MFS Taylor grids or task-local,
    frequency-resolved MVC grids. Their visibility grids are subtracted, while
    the observed-data normalization is retained
    unchanged because it represents the sum of imaging weights rather than a
    visibility contribution. The model-grid normalization is intentionally
    ignored: the gridder skips exactly zero model samples, so that auxiliary
    sum need not equal the observed-data sum even though the model numerator is
    correct.

    Parameters
    ----------
    observed_grid_xds : xarray.Dataset
        Cached observed-data UV grid.
    model_grid_xds : xarray.Dataset
        Predicted-model UV grid for the current cycle.
    image_data_group_in_observed : str, optional
        Observed-grid data group registering ``visibility`` and
        ``visibility_normalization``.
    image_data_group_in_model : str, optional
        Model-grid data group registering ``visibility`` and
        ``visibility_normalization``.
    grid_kind : str, optional
        Human-readable grid type used in diagnostics and metadata.

    Returns
    -------
    xarray.Dataset
        A copy of ``model_grid_xds`` whose visibility grid is
        ``observed - model`` and whose normalization is the cached observed
        normalization.
    """
    import copy

    import numpy as np
    import xarray as xr

    def _registered_arrays(dataset, label, data_group_name):
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(
                f"{label} must be an xarray.Dataset; received {type(dataset).__name__}."
            )
        data_groups = dataset.attrs.get("data_groups", {})
        if data_group_name not in data_groups:
            raise KeyError(
                f"{label} does not contain image data group {data_group_name!r}."
            )
        data_group = data_groups[data_group_name]
        variable_names = {
            role: data_group.get(role)
            for role in ("visibility", "visibility_normalization")
        }
        missing_roles = [role for role, name in variable_names.items() if name is None]
        if missing_roles:
            raise KeyError(
                f"{label} data group {data_group_name!r} does not register "
                f"roles {missing_roles}."
            )
        missing_variables = [
            name for name in variable_names.values() if name not in dataset
        ]
        if missing_variables:
            raise KeyError(f"{label} is missing variables {missing_variables}.")
        return variable_names

    observed_names = _registered_arrays(
        observed_grid_xds,
        "observed_grid_xds",
        image_data_group_in_observed,
    )
    model_names = _registered_arrays(
        model_grid_xds,
        "model_grid_xds",
        image_data_group_in_model,
    )

    observed_grid = observed_grid_xds[observed_names["visibility"]]
    model_grid = model_grid_xds[model_names["visibility"]]
    observed_normalization = observed_grid_xds[
        observed_names["visibility_normalization"]
    ]

    try:
        observed_grid, model_grid = xr.align(
            observed_grid,
            model_grid,
            join="exact",
            copy=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"Cached observed and predicted-model {grid_kind} grids have "
            "incompatible "
            "coordinates."
        ) from exc

    if observed_grid.dims != model_grid.dims or observed_grid.shape != model_grid.shape:
        raise ValueError(
            f"Cached observed and predicted-model {grid_kind} visibility grids "
            "must have "
            "identical dimensions and shapes."
        )

    residual_xds = model_grid_xds.copy(deep=False)
    residual_xds.attrs = copy.deepcopy(model_grid_xds.attrs)
    residual_values = np.subtract(
        observed_grid.values,
        model_grid.values,
    )
    residual_xds[model_names["visibility"]] = xr.DataArray(
        residual_values,
        dims=model_grid.dims,
        coords=model_grid.coords,
        attrs={
            **model_grid.attrs,
            "description": (
                f"{grid_kind.upper()} residual UV grid formed from cached observed-data "
                "and reduced predicted-model grids."
            ),
        },
    )
    residual_xds[model_names["visibility_normalization"]] = observed_normalization.copy(
        deep=True
    )
    residual_xds.attrs["visibility_grid_source"] = "cached_observed_minus_model"
    return residual_xds


def form_mfs_residual_grid_from_cache(
    observed_grid_xds,
    model_grid_xds,
    image_data_group_in_observed="residual",
    image_data_group_in_model="residual",
):
    """Form an MFS Taylor residual grid using the generic cache operation."""
    return form_residual_grid_from_cache(
        observed_grid_xds,
        model_grid_xds,
        image_data_group_in_observed=image_data_group_in_observed,
        image_data_group_in_model=image_data_group_in_model,
        grid_kind="MFS",
    )


@shares_param_docs
def residual_update_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    continuum_mode="mfs",
    primary_beam_xds=None,
    processing_set_data_group_name="corrected",
    deconvolver="hogbom",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    visibility_memory_mode="recompute",
    observed_visibility_grid_xds=None,
    is_first_cycle=True,
    model_xds=None,
    model_uv_xds=None,
    task_id=0,
    primary_beam_limit=0.2,
):
    """Perform one continuum imaging-cycle update for a single frequency chunk.

    This function is the primary processing entry point executed by the continuum
    map node task. It performs the operations required to compute the chunk-local
    continuum products that are later accumulated by the GraphViper reduce stage.

    During the first imaging cycle, the function first executes the one-time imaging
    preparation for the frequency chunk before computing the initial residual
    products. During subsequent imaging cycles, it reuses the existing imaging
    geometry, updates the imaging weights when required, and computes a new
    residual using the globally prepared Fourier-domain continuum model.

    The residual-update stage predicts the model visibilities (except during the
    first imaging cycle), subtracts them from the observed visibilities, grids the
    resulting residual visibilities into Taylor-weighted UV-domain products, and
    returns those products for global reduction. For MVC, the channel-resolved
    residual and first-cycle PSF grids are normalized and inverse transformed
    locally before being returned. MFS retains the globally reduced inverse-FFT
    path. No model update or restoration is performed here.

    Parameters
    ----------
    continuum_mode : {"mfs", "mvc"}
        Select Taylor UV gridding or frequency-resolved MVC processing.
    primary_beam_xds : xarray.Dataset, optional
        Cached primary beam for this MVC partition.
    observed_visibility_grid_xds : xarray.Dataset, optional
        Task-local cached observed visibility grid used by later residual-update
        cycles. The distributed application supplies this only for in-memory MVC
        caching; other modes and first cycles leave it unset.
    model_xds : xarray.Dataset, optional
        Accumulated image-domain Taylor model used for MVC prediction.
    ps_xdt : xarray.DataTree
        Visibility data for this frequency chunk.

    img_xds : xarray.Dataset
        Chunk-local continuum image dataset used to accumulate the UV-domain
        products.

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

    imaging_weights_params : dict
        Weighting configuration: ``weighting`` is ``"natural"``, ``"uniform"``,
        ``"briggs"`` or ``"briggs_abs"``; ``robust`` controls the Briggs
        trade-off. ``weighting_scope`` is ``"local"`` for a density per
        frequency channel or ``"global"`` for one density across all selected
        channels in distributed continuum imaging. Natural weighting is
        independent of this scope.

    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).

    deconvolver : str, optional
        Deconvolution algorithm for the model update. ``"hogbom"`` (C++,
        threaded across planes) cleans Taylor term zero only. ``"mtmfs"``
        (C++, multi-term multi-frequency synthesis) cleans the Taylor terms
        together and needs ``image_params["n_taylor_terms"]`` from 2 to 4; with
        ``n_taylor_terms=1`` it falls back to ``"hogbom"`` with a warning. Other values
        are refused before anything is written.

    instrument_polarization_basis : {"linear", "circular"}, optional
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
    is_first_cycle : bool, optional
        Indicates whether this is the first imaging cycle.

    model_uv_xds : xarray.Dataset, optional
        Globally prepared Fourier-domain Taylor model used for degridding during
        all imaging cycles after the first.

    task_id : int, optional
        Identifier of the parallel chunk being processed.

    primary_beam_limit : float, optional
        Channel primary-beam cutoff used by MVC map-local Taylor construction.

    Returns
    -------
    img_xds : xarray.Dataset
        Chunk-local MFS UV products or MVC frequency images for subsequent
        map-local Taylor conversion.

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

    if is_first_cycle:
        (
            img_xds,
            setup_return_df,
            T_setup,
        ) = imaging_preparation_continuum_single_field(
            ps_xdt,
            img_xds,
            image_params,
            imaging_weights_params,
            continuum_mode=continuum_mode,
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
        is_first_cycle,
        continuum_mode=continuum_mode,
        model_xds=model_xds,
        primary_beam_xds=primary_beam_xds,
        processing_set_data_group_name=processing_set_data_group_name,
        instrument_polarization_basis=instrument_polarization_basis,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
        visibility_memory_mode=visibility_memory_mode,
        observed_visibility_grid_xds=observed_visibility_grid_xds,
    )

    timing["T_residual_cycle"] = time.time() - start
    accumulate_timing(timing, residual_return_df)

    # MVC channels are exclusively owned by this map task. Normalize and
    # inverse-transform them here so the distributed reducer carries cropped
    # image cubes rather than padded UV grids. The normalization arrays remain
    # in the dataset because the global Taylor conversion reuses them as
    # spectral weights.
    timing["T_local_ifft"] = 0.0

    mvc_observed_grid_xds = None
    if (
        str(continuum_mode).lower() == "mvc"
        and is_first_cycle
        and visibility_memory_mode != "recompute"
    ):
        residual_group = img_xds.attrs["data_groups"]["residual"]
        visibility_name = residual_group["visibility"]
        normalization_name = residual_group["visibility_normalization"]
        mvc_observed_grid_xds = img_xds[[visibility_name, normalization_name]].copy(
            deep=True
        )
        mvc_observed_grid_xds.attrs = img_xds.attrs.copy()
        mvc_observed_grid_xds.attrs["visibility_grid_source"] = "observed_data"

    if str(continuum_mode).lower() == "mvc":
        from astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder import (
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

        if is_first_cycle:
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
                    "The first MVC map task did not create its channel PSF image cube."
                )

        residual_group = img_xds.attrs["data_groups"]["residual"]
        primary_beam_name = residual_group.get("primary_beam")
        if primary_beam_name is not None and primary_beam_name in img_xds:
            primary_beam = img_xds[primary_beam_name]
        elif primary_beam_xds is not None:
            primary_beam_name = primary_beam_xds.attrs.get(
                "primary_beam_name",
                "PRIMARY_BEAM",
            )
            primary_beam = primary_beam_xds[primary_beam_name]
        else:
            raise RuntimeError(
                "MVC map-local Taylor construction requires its channel primary beam."
            )

        contributions = make_mvc_taylor_normal_equation_contributions(
            img_xds["SKY_RESIDUAL_MVC_CUBE"],
            (img_xds["POINT_SPREAD_FUNCTION_MVC_CUBE"] if is_first_cycle else None),
            primary_beam,
            img_xds["VISIBILITY_NORMALIZATION"],
            img_xds["UV_SAMPLING_NORMALIZATION"] if is_first_cycle else None,
            n_taylor_terms=image_params.get("n_taylor_terms", 2),
            reference_frequency=image_params["reference_frequency"],
            primary_beam_limit=primary_beam_limit,
        )
        for variable_name, data_array in contributions.data_vars.items():
            img_xds[variable_name] = data_array
        img_xds.attrs.update(contributions.attrs)

        if mvc_observed_grid_xds is not None:
            residual_group = mvc_observed_grid_xds.attrs["data_groups"]["residual"]
            img_xds[_MVC_OBSERVED_VISIBILITY_CACHE] = mvc_observed_grid_xds[
                residual_group["visibility"]
            ]
            img_xds[_MVC_OBSERVED_NORMALIZATION_CACHE] = mvc_observed_grid_xds[
                residual_group["visibility_normalization"]
            ]

        timing["T_local_ifft"] = time.time() - start

    timing["n_channels"] = img_xds.sizes.get(
        "frequency",
        len(img_xds.coords.get("frequency", [])),
    )
    timing["n_taylor_terms"] = image_params.get("n_taylor_terms", 2)

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df


###############################################################################
# Convergence statistics and model updates
###############################################################################


def _continuum_clean_support(
    img_xds, image_data_group_in_name, residual_plane, control_params
):
    """The pixels of Taylor term zero that may be cleaned, or ``None``.

    One support for the stop test (:func:`build_continuum_residual_imaging_dict`)
    and the native MT-MFS model update, so that a plane is never kept going
    for a peak its model update may not clean. It is the clean mask of the
    data group (values above 0.5) or, without one, the pixels where the
    primary beam is at least ``primary_beam_limit`` times its peak and not 0.
    When both are present, their intersection is used so an explicit mask
    cannot re-enable pixels below the primary-beam cutoff.
    ``primary_beam_limit`` defaults to 0, every pixel where the primary beam
    is above 0, as for the continuum Högbom model update and cube imaging,
    which make the same mask with the same default
    (:func:`~astroviper.processing_functions.image_analysis.make_mask.make_mask`):
    the residual holds the sky seen through the primary beam wherever that
    is above 0, and a source there that is not cleaned leaves its sidelobes
    across the image.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Continuum image dataset holding the residual data group.
    image_data_group_in_name : str
        Residual data group whose ``mask`` or ``primary_beam`` role gives the
        support.
    residual_plane : xarray.DataArray
        Taylor term zero of the residual, dimensions
        ``(time, polarization, l, m)``; the support is broadcast to it.
    control_params : dict
        Cycle control or model update parameters; ``primary_beam_limit``
        (default 0) is read from it.

    Returns
    -------
    numpy.ndarray or None
        Boolean support, shape ``(time, polarization, l, m)``; ``None`` when
        the data group has neither a mask nor a primary beam.
    """
    import numpy as np

    group = img_xds.attrs["data_groups"][image_data_group_in_name]

    def plane_zero(array):
        selection = {
            dim: 0
            for dim in ("taylor_term", "psf_taylor_order", "frequency")
            if dim in array.dims
        }
        return array.isel(selection, drop=True)

    mask_name = group.get("mask")
    primary_beam_name = group.get("primary_beam")
    support = None
    if mask_name is not None:
        support = plane_zero(img_xds[mask_name]).broadcast_like(residual_plane) > 0.5
    if primary_beam_name is not None:
        primary_beam = plane_zero(img_xds[primary_beam_name])
        primary_beam_limit = float(control_params.get("primary_beam_limit", 0.0))
        primary_beam_support = (
            (primary_beam >= primary_beam_limit * primary_beam.max(skipna=True))
            & (primary_beam != 0)
        ).broadcast_like(residual_plane)
        support = (
            primary_beam_support if support is None else support & primary_beam_support
        )
    if support is None:
        return None
    return np.ascontiguousarray(
        support.transpose("time", "polarization", "l", "m").values,
        dtype=bool,
    )


def build_continuum_residual_imaging_dict(
    img_xds, image_data_group_in_name, cycle_control_params
):
    """Present Taylor-zero residual statistics as one plane to the controller.

    Continuum Högbom updates only Taylor term zero. Its fitted sidelobe has
    dimensions (time, polarization), whereas the shared controller expects
    (time, frequency, polarization). Build a temporary view with one frequency
    plane, broadcasting masks and sidelobes by dimension name. Peaks are
    measured on finite pixels in the explicit CLEAN mask, or the same PB
    support as model_update when no explicit mask exists. No counters or
    history entries are advanced. When entropy stopping is enabled, residual
    entropy and SNR are measured on Taylor term zero using that same clean
    support. The cube workflow is unchanged.
    """
    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.image_analysis import (
        image_statistics as imgstats,
    )
    from astroviper.processing_functions.imaging.utils import (
        ImagingDict,
        validate_entropy_params,
    )

    entropy_stop, entropy_max_snr, entropy_spatial_bins, entropy_flux_bins = (
        validate_entropy_params(cycle_control_params)
    )

    group = img_xds.attrs["data_groups"][image_data_group_in_name]

    def plane_zero(array):
        selection = {
            dim: 0
            for dim in ("taylor_term", "psf_taylor_order", "frequency")
            if dim in array.dims
        }
        return array.isel(selection, drop=True)

    residual = plane_zero(img_xds[group["sky"]]).transpose(
        "time", "polarization", "l", "m"
    )
    support = _continuum_clean_support(
        img_xds, image_data_group_in_name, residual, cycle_control_params
    )
    if support is None:
        support = np.ones(residual.shape, dtype=bool)
    absolute = np.abs(residual.values)
    sidelobe = plane_zero(
        img_xds[
            group.get(
                "max_sidelobe_point_spread_function",
                "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
            )
        ]
    )
    sidelobe = sidelobe.broadcast_like(residual.isel(l=0, m=0, drop=True))
    sidelobe = sidelobe.transpose("time", "polarization").values
    if entropy_stop:
        entropy_residual = residual.expand_dims(frequency=[0], axis=1)
        entropy_image = entropy_residual.to_dataset(name="SKY_RESIDUAL")
        entropy_image["MASK"] = xr.DataArray(
            support[:, np.newaxis],
            dims=entropy_residual.dims,
            coords=entropy_residual.coords,
        )
        entropy_image.attrs["data_groups"] = {
            "residual": {"sky": "SKY_RESIDUAL", "mask": "MASK"}
        }
        entropy, residual_snr = imgstats.image_residual_entropy(
            entropy_image,
            data_group_name="residual",
            spatial_bins=entropy_spatial_bins,
            flux_bins=entropy_flux_bins,
            max_snr=entropy_max_snr,
        )
    statistics = ImagingDict()
    for tt in range(residual.sizes["time"]):
        for pp in range(residual.sizes["polarization"]):
            finite = np.isfinite(absolute[tt, pp])
            selected = support[tt, pp] & finite
            fields = {
                "peakres": float(np.max(absolute[tt, pp][selected], initial=0.0)),
                "peakres_nomask": float(np.max(absolute[tt, pp][finite], initial=0.0)),
                "masksum": int(np.count_nonzero(selected)),
                "iter_done": 0,
                "max_psf_sidelobe": float(sidelobe[tt, pp]),
                "gain": cycle_control_params.get("gain", 0.1),
            }
            if entropy_stop:
                fields["entropy"] = float(entropy[tt, 0, pp])
                fields["residual_snr"] = float(residual_snr[tt, 0, pp])
            statistics.add(
                fields,
                time=tt,
                chan=0,
                pol=pp,
            )
    return statistics


def _model_update_native_mtmfs(
    img_xds,
    deconvolve_params,
    residual_name,
    psf_name,
    max_sidelobe_name,
    image_data_group_in_name,
    image_data_group_out_name,
):
    """Run the native MT-MFS kernel on each time/polarization Taylor stack."""
    import time

    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.deconvolution import (
        _divergence_factors,
    )
    from astroviper.processing_functions.imaging.deconvolvers import mtmfs
    from astroviper.processing_functions.imaging.utils import ImagingDict, StopCode
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_CONTINUE,
        MODEL_UPDATE_STOP_DESCRIPTIONS,
    )

    stored_residual_dims = img_xds[residual_name].dims
    residual_da = img_xds[residual_name].transpose(
        "time", "taylor_term", "polarization", "l", "m"
    )
    psf_da = img_xds[psf_name].transpose(
        "time", "psf_taylor_order", "polarization", "l", "m"
    )
    residual = residual_da.values
    psf = psf_da.values
    ntime, n_taylor_terms, npol, nl, nm = residual.shape

    if not 2 <= n_taylor_terms <= 4:
        raise ValueError(
            "Native MT-MFS continuum deconvolution requires 2 to 4 Taylor "
            f"terms; received {n_taylor_terms}."
        )
    expected_psf_terms = 2 * n_taylor_terms - 1
    if psf.shape != (ntime, expected_psf_terms, npol, nl, nm):
        raise ValueError(
            "POINT_SPREAD_FUNCTION must have shape "
            f"(time, {expected_psf_terms}, polarization, l, m) for "
            f"n_taylor_terms={n_taylor_terms}; received {psf.shape}."
        )
    if residual.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(
            "Native MT-MFS residuals must use float32 or float64; "
            f"received {residual.dtype}."
        )
    if psf.dtype != residual.dtype:
        raise TypeError(
            "Native MT-MFS residual and PSF dtypes must match; "
            f"received {residual.dtype} and {psf.dtype}."
        )

    # A read-only source cannot receive the native kernel's residual update.
    # Give the dataset writable storage while preserving its current layout;
    # individual non-canonical Taylor stacks are handled below.
    if not residual.flags.writeable:
        residual = np.array(residual, copy=True, order="K")
        residual_da = residual_da.copy(data=residual)
        img_xds[residual_name] = residual_da.transpose(*stored_residual_dims)

    reference_plane = residual_da.isel(taylor_term=0, drop=True)
    support = _continuum_clean_support(
        img_xds, image_data_group_in_name, reference_plane, deconvolve_params
    )
    if support is None:
        support = reference_plane.notnull().values

    max_sidelobe = img_xds[max_sidelobe_name]
    selection = {
        dim: 0
        for dim in ("taylor_term", "frequency", "psf_taylor_order")
        if dim in max_sidelobe.dims
    }
    max_sidelobe = (
        max_sidelobe.isel(selection, drop=True)
        .broadcast_like(reference_plane.isel(l=0, m=0, drop=True))
        .transpose("time", "polarization")
        .values
    )

    def plane_control(name, tt, pp, default):
        value = deconvolve_params.get(name, default)
        if value is None:
            value = default
        if np.ndim(value) == 0:
            return value
        values = np.asarray(value)
        if values.shape != (ntime, 1, npol):
            raise ValueError(
                f"{name} must be scalar or have shape {(ntime, 1, npol)}; "
                f"received {values.shape}."
            )
        return values[tt, 0, pp]

    # A call without the per-cycle limits (outside the imaging cycle loop) is
    # one model update with all of max_iter left, as in deconvolve():
    # min(max_iter_per_cycle, max_iter), a negative cap meaning no cap, and
    # the absolute threshold.
    max_iter_total = int(deconvolve_params.get("max_iter", 0))
    cap = deconvolve_params.get("max_iter_per_cycle")
    cap = max_iter_total if cap is None or int(cap) < 0 else int(cap)
    default_max_iter = min(cap, max_iter_total)
    default_threshold = deconvolve_params.get("threshold")
    default_threshold = 0.0 if default_threshold is None else default_threshold
    # The divergence test of the model update, made by the kernel on the
    # Taylor-zero residual: a plane is stopped as diverged when the RMS of
    # that residual has been above (1 + gain) times the lowest RMS of the
    # model update for max_iter_divergence consecutive iterations, or at once
    # when its peak exceeds (1 + gain) times its peak at the start of the
    # model update. max_iter_divergence defaults to 1 as in deconvolve(), and
    # -1 turns the test off. The RMS limit is that of Asp, not Hogbom's
    # (1 + gain / 10): in healthy MT-MFS model updates the Taylor-zero RMS
    # can rise by more than gain / 10 above its lowest value (a component
    # subtracted where that residual is not shaped like the PSF raises it),
    # and the tighter limit stopped such updates after a few components.
    # The peak limit is Hogbom's; it stops a runaway. gain and
    # max_iter_divergence are checked as deconvolve() checks them: the
    # limits need 0 < gain <= 1, so the kernel's adaptive gain (gain <= 0)
    # is refused as on the Hogbom path.
    gain = deconvolve_params.get("gain", 0.1)
    max_iter_divergence = deconvolve_params.get("max_iter_divergence", 1)
    if not 0 < gain <= 1:
        raise ValueError("CLEAN gain must be between 0 and 1.")
    if (
        isinstance(max_iter_divergence, bool)
        or not isinstance(max_iter_divergence, int | np.integer)
        or not (max_iter_divergence == -1 or max_iter_divergence >= 1)
    ):
        raise ValueError(
            "max_iter_divergence must be a positive integer, or -1 to "
            f"disable the divergence test; got {max_iter_divergence!r}."
        )
    _, limit = _divergence_factors(gain)
    divergence = {
        "max_iter_divergence": int(max_iter_divergence),
        "divergence_rms_factor": limit,
        "divergence_peak_factor": limit,
    }

    model = np.zeros_like(residual)
    imaging_dict = ImagingDict()
    start = time.time()
    for tt in range(ntime):
        for pp in range(npol):
            residual_view = residual[tt, :, pp, :, :]
            itemsize = residual_view.dtype.itemsize
            plane_bytes = nl * nm * itemsize
            native_layout = (
                residual_view.flags.writeable
                and residual_view.strides[2] == itemsize
                and residual_view.strides[1] == nm * itemsize
                and residual_view.strides[0] >= plane_bytes
                and residual_view.strides[0] % itemsize == 0
            )
            # Preserve the zero-copy path for canonical image storage. MVC and
            # external callers may instead place Taylor terms after the image
            # axes; copy those stacks into the native layout and write them back.
            residual_work = (
                residual_view
                if native_layout
                else np.array(residual_view, copy=True, order="C")
            )
            psf_work = np.ascontiguousarray(psf[tt, :, pp, :, :])
            model_work = np.zeros_like(residual_work)
            mask_work = np.ascontiguousarray(support[tt, pp], dtype=residual_work.dtype)
            selected = support[tt, pp] & np.isfinite(residual_work[0])
            start_peak = float(np.max(np.abs(residual_work[0][selected]), initial=0.0))
            start_peak_nomask = float(np.max(np.abs(residual_work[0]), initial=0.0))
            threshold_per_cycle = float(
                plane_control("threshold_per_cycle", tt, pp, default_threshold)
            )
            max_iter = int(
                plane_control(
                    "max_iter_per_cycle_updated",
                    tt,
                    pp,
                    default_max_iter,
                )
            )
            result = mtmfs.clean(
                residual_work,
                psf_work,
                model_work,
                mask=mask_work,
                scales=deconvolve_params.get("scales", []),
                small_scale_bias=deconvolve_params.get("small_scale_bias", 0.0),
                max_iter_per_cycle_updated=max_iter,
                gain=deconvolve_params.get("gain", 0.1),
                threshold_per_cycle=threshold_per_cycle,
                mask_threshold=deconvolve_params.get("mask_threshold", 0.9),
                **divergence,
            )
            if not native_layout:
                residual_view[...] = residual_work
            model[tt, :, pp, :, :] = model_work
            final_selected = support[tt, pp] & np.isfinite(residual_work[0])
            model_update_stop_code = int(result["stop_code"])
            fields = {
                "max_iter": deconvolve_params.get("max_iter"),
                "threshold_per_cycle": threshold_per_cycle,
                "iter_done": int(result["iterations_performed"]),
                "gain": deconvolve_params.get("gain", 0.1),
                "max_psf_sidelobe": float(max_sidelobe[tt, pp]),
                "stop_code": StopCode(
                    imaging=IMAGING_CONTINUE,
                    model_update=model_update_stop_code,
                ),
                "stop_description": MODEL_UPDATE_STOP_DESCRIPTIONS[
                    model_update_stop_code
                ],
                "stokes": img_xds.coords["polarization"].values[pp],
                "frequency": img_xds.attrs.get("reference_frequency", 0.0),
                "time": img_xds.coords["time"].values[tt],
                "start_model_flux": 0.0,
                "model_flux": float(result["model_flux"]),
                "start_peakres": start_peak,
                "start_peakres_nomask": start_peak_nomask,
                "peakres": float(result["peak_residual"]),
                "peakres_nomask": float(np.max(np.abs(residual_work[0]), initial=0.0)),
                "masksum": int(np.count_nonzero(final_selected)),
                "scales": result["scales"],
                "hessian_condition_number": result["hessian_condition_number"],
            }
            imaging_dict.add(fields, time=tt, pol=pp, chan=0)

    img_xds["SKY_MODEL"] = residual_da.copy(data=model)
    img_xds.attrs.setdefault("data_groups", {})[image_data_group_out_name] = {
        "sky": "SKY_MODEL"
    }
    elapsed = time.time() - start
    return imaging_dict, pd.DataFrame(
        {"T_mtmfs": [elapsed], "T_model_update_cycle": [elapsed]}
    )


@shares_param_docs
def model_update_mtmfs_single_field(
    img_xds,
    deconvolver,
    deconvolve_params,
    is_first_cycle=True,
    processing_function_threads=1,
    image_data_group_in_name="residual",
    image_data_group_out_name="model",
):
    """Perform one continuum model update.

    Dispatch the continuum model update to native MT-MFS or to the compatibility
    Högbom path. Native MT-MFS receives each complete Taylor stack directly as
    ``(n_taylor_terms, l, m)``. Högbom temporarily projects Taylor term zero onto a
    single-frequency cube representation.

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

    Only the zeroth Taylor coefficient is modified during the model update.
    Higher-order model terms remain unchanged; subsequent residual updates
    recalculate the residual Taylor terms.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Globally reduced continuum image dataset. The residual and model images
        are expected to use the ``taylor_term`` dimension, while the point-spread
        function uses ``psf_taylor_order``.

    deconvolver : str
        Deconvolution algorithm for the model update. ``"hogbom"`` (C++,
        threaded across planes) cleans Taylor term zero only. ``"mtmfs"``
        (C++, multi-term multi-frequency synthesis) cleans the Taylor terms
        together and needs ``image_params["n_taylor_terms"]`` from 2 to 4; with
        ``n_taylor_terms=1`` it falls back to ``"hogbom"`` with a warning. Other values
        are refused before anything is written.

    deconvolve_params : dict
        Model-update control parameters. These typically include entries such as
        ``max_iter``, ``max_iter_per_cycle_updated``, ``threshold_per_cycle``,
        ``gain``, ``scales``, ``small_scale_bias``, and ``mask_threshold``.
        The per-cycle limits and thresholds accept scalars or per-plane arrays,
        using the same keys as the cube deconvolver. Without them the limit is
        ``min(max_iter_per_cycle, max_iter)`` (a negative
        ``max_iter_per_cycle`` means no cap) and ``threshold_per_cycle`` is
        ``threshold``, as in the cube deconvolver. ``max_iter_divergence``
        (default 1, ``-1`` turns the test off) sets the divergence test of
        the model update, made on the Taylor-zero residual: a plane stops
        when its RMS has been above ``(1 + gain)`` times the lowest RMS of
        the model update for that many iterations in a row (the limit of
        Asp), or at once when its peak exceeds ``(1 + gain)`` times the
        peak at the start of the model update. ``gain``
        (``0 < gain <= 1``) and ``max_iter_divergence`` are checked as in
        the cube deconvolver. Both deconvolvers restrict cleaning to pixels
        where the primary beam is at least ``primary_beam_limit`` (default 0)
        times its peak and not 0. When a clean mask is present, its intersection
        with that PB support is used; the stop test measures the same pixels.

    is_first_cycle : bool, optional
        Indicates whether this is the first model update.

    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.

    image_data_group_in_name : str, optional
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.

    image_data_group_out_name : str, optional
        Name of the output model data group.

    Returns
    -------
    deconvolve_dict : ImagingDict
        Per-plane deconvolution statistics returned by the selected backend.

    return_df : pandas.DataFrame
        Timing information for the continuum model update.

    Notes
    -----
    Native MT-MFS updates the selected residual Taylor stacks in ``img_xds``
    in place. Their spatial planes are contiguous while the binding explicitly
    honors the Taylor-term stride introduced by the polarization axis, so no
    residual copy or reshape is made. The model buffer remains a zero-initialized
    increment for the current model update.
    """
    import copy
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    deconvolver = deconvolver.lower()
    if deconvolver not in {"hogbom", "mtmfs"}:
        raise NotImplementedError(
            "Continuum deconvolution supports 'hogbom' and 'mtmfs'; "
            f"received {deconvolver!r}."
        )

    if "SKY_RESIDUAL" not in img_xds:
        raise KeyError("The continuum image dataset does not contain SKY_RESIDUAL.")

    if "POINT_SPREAD_FUNCTION" not in img_xds:
        raise KeyError(
            "The continuum image dataset does not contain POINT_SPREAD_FUNCTION."
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
            "POINT_SPREAD_FUNCTION must contain a 'psf_taylor_order' dimension."
        )

    if residual.sizes["taylor_term"] < 1:
        raise ValueError("SKY_RESIDUAL contains no Taylor terms.")

    if psf.sizes["psf_taylor_order"] < 1:
        raise ValueError("POINT_SPREAD_FUNCTION contains no PSF Taylor terms.")

    # make a cube-compatible copy of the per-plane controls
    cube_deconvolve_params = copy.deepcopy(deconvolve_params)

    for parameter_name in (
        "max_iter_per_cycle_updated",
        "threshold_per_cycle",
    ):
        parameter_value = cube_deconvolve_params.get(parameter_name)

        if parameter_value is None or np.ndim(parameter_value) == 0:
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
            "continuum model update. It must be created by "
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

    if deconvolver == "mtmfs":
        return _model_update_native_mtmfs(
            img_xds,
            cube_deconvolve_params,
            residual_name,
            psf_name,
            max_sidelobe_name,
            image_data_group_in_name,
            image_data_group_out_name,
        )

    # ------------------------------------------------------------------
    # Build a temporary cube-like image dataset.
    #
    # The existing Högbom backend expects
    #
    #     time x frequency x polarization x l x m
    #
    # whereas the continuum products use Taylor dimensions.
    # ------------------------------------------------------------------

    frequency_coord = _continuum_frequency_coord(img_xds)

    hogbom_xds = xr.Dataset(attrs=copy.deepcopy(img_xds.attrs))

    # Preserve coordinates that do not depend on a Taylor-order dimension.
    for coord_name, coord in img_xds.coords.items():
        if (
            "taylor_term" not in coord.dims
            and "psf_taylor_order" not in coord.dims
            and coord_name != "frequency"
        ):
            hogbom_xds = hogbom_xds.assign_coords({coord_name: coord.copy(deep=False)})

    hogbom_xds = hogbom_xds.assign_coords(frequency=("frequency", frequency_coord))

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
                hogbom_xds[variable_name] = _continuum_plane_to_frequency(
                    img_xds, data_array, "psf_taylor_order"
                )

            elif "frequency" in data_array.dims:
                hogbom_xds[variable_name] = data_array.isel(
                    frequency=slice(0, 1)
                ).assign_coords(frequency=frequency_coord)

            else:
                raise ValueError(
                    f"{variable_name} has unsupported dimensions {data_array.dims}."
                )

        elif "taylor_term" in data_array.dims:
            hogbom_xds[variable_name] = _continuum_plane_to_frequency(
                img_xds, data_array, "taylor_term"
            )

        elif "psf_taylor_order" in data_array.dims:
            hogbom_xds[variable_name] = _continuum_plane_to_frequency(
                img_xds, data_array, "psf_taylor_order"
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

    # The Taylor-plane selection above can leave a strided view. Högbom's C++
    # binding deliberately accepts only a C-contiguous bool mask, so make the
    # one-plane adapter buffer explicit. The > 0.5 rule is the same mask
    # interpretation used by continuum starting and stopping statistics; NaNs
    # therefore remain outside the CLEAN support rather than becoming True.
    hogbom_mask_name = hogbom_xds.attrs["data_groups"][image_data_group_in_name].get(
        "mask"
    )
    hogbom_data_group = hogbom_xds.attrs["data_groups"][image_data_group_in_name]
    primary_beam_name = hogbom_data_group.get("primary_beam")
    if hogbom_mask_name is None and primary_beam_name in hogbom_xds:
        from astroviper.processing_functions.image_analysis.make_mask import make_mask

        make_mask(
            hogbom_xds,
            primary_beam_limit=cube_deconvolve_params.get("primary_beam_limit", 0.0),
            image_data_group_in_name=image_data_group_in_name,
            image_data_group_out_name=image_data_group_in_name,
            combine_mask=False,
            overwrite=False,
        )
        hogbom_mask_name = hogbom_xds.attrs["data_groups"][image_data_group_in_name][
            "mask"
        ]

    if hogbom_mask_name in hogbom_xds:
        hogbom_mask = hogbom_xds[hogbom_mask_name].transpose(
            "time", "frequency", "polarization", "l", "m", missing_dims="ignore"
        )
        if "frequency" not in hogbom_mask.dims:
            hogbom_mask = hogbom_mask.expand_dims(
                frequency=frequency_coord,
                axis=1,
            )
        mask_values = np.asarray(hogbom_mask.values)
        hogbom_xds[hogbom_mask_name] = hogbom_mask.copy(
            data=np.ascontiguousarray(
                np.isfinite(mask_values) & (mask_values > 0.5),
                dtype=np.bool_,
            )
        )

    # Register the temporary model through the data-group interface expected by
    # deconvolve() and starting_statistics().
    _, hogbom_model_group = create_data_groups_in_and_out(
        hogbom_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified={"sky": "SKY_MODEL"},
        overwrite=True,
    )
    modify_data_groups_xds(
        hogbom_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=hogbom_model_group,
        description="Temporary single-plane Högbom continuum model.",
    )
    model_name = hogbom_model_group["sky"]

    T_make_hogbom_view = time.time() - start

    # Högbom operates on real arrays but supports both image precisions. Match
    # the residual precision throughout and avoid copying arrays already in the
    # required real dtype.
    float_dtype = hogbom_xds[residual_name].real.dtype
    hogbom_xds[residual_name] = hogbom_xds[residual_name].real.astype(
        float_dtype, copy=False
    )
    psf = hogbom_xds[psf_name].real
    hogbom_xds[psf_name] = psf.copy(
        data=np.ascontiguousarray(psf.values, dtype=float_dtype)
    )
    hogbom_xds[model_name] = hogbom_xds[model_name].real.astype(float_dtype, copy=False)

    # ------------------------------------------------------------------
    # Run the existing single-frequency Högbom model-update backend.
    # ------------------------------------------------------------------
    start = time.time()

    from astroviper.processing_functions.imaging.model_update import (
        model_update_cube_single_field,
    )

    deconvolve_dict, hogbom_return_df = model_update_cube_single_field(
        hogbom_xds,
        deconvolver,
        cube_deconvolve_params,
        model_exists=not is_first_cycle,
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
            "The continuum SKY_MODEL variable must contain a 'taylor_term' dimension."
        )

    # Use positional assignment because the Taylor coordinate need not
    # literally contain the value zero.
    img_xds["SKY_MODEL"].data[:, 0, ...] = model_zero.data

    _, continuum_model_group = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified={"sky": "SKY_MODEL"},
        overwrite=True,
    )
    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=continuum_model_group,
        description="Continuum model increment produced by Högbom CLEAN.",
    )

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


def accumulate_continuum_model(
    model_increment_xds,
    previous_model_xds=None,
    continuum_mode="mfs",
):
    """Create the accumulated continuum model for the next residual update.

    The model-update backend returns the initial model during the first cycle
    and a model increment during every later cycle.  This helper normalizes
    those two cases into one persistent model-state dataset.  It deliberately
    performs positional array addition, matching the historical driver-side
    accumulation after validating dimensions and shapes.

    Parameters
    ----------
    model_increment_xds : xarray.Dataset
        Image dataset returned by the current model-update cycle. It must
        contain ``SKY_MODEL``. During the first MVC cycle it must also contain
        the singleton effective ``PRIMARY_BEAM``.
    previous_model_xds : xarray.Dataset, optional
        Accumulated model returned by the preceding append node. If omitted,
        the current model is copied as the initial accumulated state.
    continuum_mode : {"mfs", "mvc"}, default ``"mfs"``
        Continuum execution mode. MVC retains the effective primary beam with
        the accumulated Taylor model.

    Returns
    -------
    xarray.Dataset
        Independent, in-memory accumulated model state containing
        ``SKY_MODEL`` and, for MVC, ``PRIMARY_BEAM``.

    Raises
    ------
    TypeError
        If either supplied model object is not an xarray dataset.
    KeyError
        If a required model or MVC primary-beam variable is absent.
    ValueError
        If the mode is unsupported or the increment layout differs from the
        previous accumulated model.
    """
    import xarray as xr

    if not isinstance(model_increment_xds, xr.Dataset):
        raise TypeError(
            "model_increment_xds must be an xarray.Dataset; received "
            f"{type(model_increment_xds).__name__}."
        )

    continuum_mode = str(continuum_mode).lower()
    if continuum_mode not in ("mfs", "mvc"):
        raise ValueError(
            f"continuum_mode must be either 'mfs' or 'mvc'; received {continuum_mode!r}."
        )

    if "SKY_MODEL" not in model_increment_xds:
        raise KeyError("model_increment_xds does not contain 'SKY_MODEL'.")

    if previous_model_xds is None:
        model_variable_names = ["SKY_MODEL"]
        if continuum_mode == "mvc":
            model_variable_names.append("PRIMARY_BEAM")

        missing = [
            name for name in model_variable_names if name not in model_increment_xds
        ]
        if missing:
            raise KeyError(
                "The initial continuum model state is missing required "
                f"variables: {missing}."
            )

        return model_increment_xds[model_variable_names].copy(deep=True)

    if not isinstance(previous_model_xds, xr.Dataset):
        raise TypeError(
            "previous_model_xds must be an xarray.Dataset; received "
            f"{type(previous_model_xds).__name__}."
        )
    if "SKY_MODEL" not in previous_model_xds:
        raise KeyError("previous_model_xds does not contain 'SKY_MODEL'.")

    model_increment = model_increment_xds["SKY_MODEL"]
    previous_model = previous_model_xds["SKY_MODEL"]

    if model_increment.dims != previous_model.dims:
        raise ValueError(
            "The continuum model increment dimensions do not match the "
            "accumulated model: "
            f"{model_increment.dims} != {previous_model.dims}."
        )
    if model_increment.shape != previous_model.shape:
        raise ValueError(
            "The continuum model increment shape does not match the "
            "accumulated model: "
            f"{model_increment.shape} != {previous_model.shape}."
        )

    accumulated_model_xds = previous_model_xds.copy(deep=True)
    accumulated_model = accumulated_model_xds["SKY_MODEL"]
    accumulated_model.data = accumulated_model.data + model_increment.data
    accumulated_model.attrs = previous_model.attrs.copy()

    return accumulated_model_xds


###############################################################################
# Restoration and primary-beam correction
###############################################################################


def restore_continuum_image(
    img_xds,
    image_data_group_in_residual_name="residual",
    image_data_group_in_model_name="model",
    image_data_group_out_restore_name="restored",
    processing_function_threads=1,
    primary_beam_correction=False,
    primary_beam_limit=0.2,
    primary_beam_key="primary_beam",
    deconvolver="hogbom",
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
        ├── optionally divide model and residual by the primary beam
        │   before restoring the model
        └── produce the restored image and PB-corrected image
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

    primary_beam_correction : bool, optional
        If ``True``, also produce the primary-beam-corrected Taylor-zero image
        using the shared cube-restoration convention
        ``(model / primary_beam) * clean_beam + residual / primary_beam``.

    primary_beam_limit : float, optional
        Primary-beam cutoff below which the corrected image is blanked.

    primary_beam_key : str, optional
        Role key of the primary beam in the residual data group.

    deconvolver : {"hogbom", "mtmfs"}, optional
        Deconvolver that produced the model. MT-MFS restoration applies the
        Taylor Hessian's principal solution to a private residual copy before
        selecting Taylor term zero. The stored residual remains unchanged.

    Returns
    -------
    img_xds : xarray.Dataset
        Continuum dataset with the restored Taylor-zero image and, when
        requested, its primary-beam-corrected counterpart added.

    timing_df : pandas.DataFrame
        Timing information returned by the restoration backend.

    Notes
    -----
    This function is a compatibility layer around the existing cube restoration
    routine. Although the surrounding imaging algorithm is MT-MFS, the current
    restoration operates only on the zeroth Taylor coefficient and therefore
    produces the restored reference-frequency continuum image. Primary-beam
    correction follows the shared cube restoration convention from ``main``;
    it is not a post-restoration division."""

    import copy

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

    if primary_beam_correction and not 0.0 <= float(primary_beam_limit) < 1.0:
        raise ValueError(
            "primary_beam_limit must satisfy 0 <= primary_beam_limit < 1; "
            f"received {primary_beam_limit}."
        )

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

    restoration_residual_da = residual_da
    if str(deconvolver).lower() == "mtmfs":
        restoration_residual_da = _mtmfs_principal_residual_for_restoration(
            residual_da, psf_da
        )

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

    frequency_value = _continuum_frequency_coord(img_xds)
    cube_xds = cube_xds.assign_coords(frequency=("frequency", frequency_value))

    cube_xds[residual_name] = _continuum_plane_to_frequency(
        img_xds, restoration_residual_da, "taylor_term"
    )
    cube_xds[model_name] = _continuum_plane_to_frequency(
        img_xds, model_da, "taylor_term"
    )

    if "psf_taylor_order" in psf_da.dims:
        if psf_da.sizes["psf_taylor_order"] < 1:
            raise ValueError(f"PSF variable {psf_name!r} contains no Taylor orders.")

        cube_xds[psf_name] = _continuum_plane_to_frequency(
            img_xds, psf_da, "psf_taylor_order"
        )

    elif "taylor_term" in psf_da.dims:
        cube_xds[psf_name] = _continuum_plane_to_frequency(
            img_xds, psf_da, "taylor_term"
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

    # The continuum primary beam is already reduced to the reference-frequency
    # plane. Present it to the shared restoration backend as a one-channel cube.
    if primary_beam_correction:
        primary_beam_name = residual_group.get(primary_beam_key)
        if primary_beam_name is None:
            raise KeyError(
                f"Residual data group {image_data_group_in_residual_name!r} "
                f"does not define {primary_beam_key!r}."
            )
        if primary_beam_name not in img_xds:
            raise KeyError(
                f"Primary-beam variable {primary_beam_name!r} is missing from img_xds."
            )

        primary_beam_da = img_xds[primary_beam_name]
        if "frequency" in primary_beam_da.dims:
            if primary_beam_da.sizes["frequency"] != 1:
                raise ValueError(
                    "Continuum restoration requires one averaged primary-beam "
                    "plane, not a multi-channel primary-beam cube."
                )
            cube_primary_beam_da = primary_beam_da.isel(
                frequency=slice(0, 1)
            ).assign_coords(frequency=cube_xds.coords["frequency"])
        elif "taylor_term" in primary_beam_da.dims:
            cube_primary_beam_da = (
                primary_beam_da.isel(taylor_term=slice(0, 1))
                .rename({"taylor_term": "frequency"})
                .assign_coords(frequency=cube_xds.coords["frequency"])
            )
        else:
            frequency_axis = (
                primary_beam_da.dims.index("time") + 1
                if "time" in primary_beam_da.dims
                else 0
            )
            cube_primary_beam_da = primary_beam_da.expand_dims(
                frequency=cube_xds.coords["frequency"].values,
                axis=frequency_axis,
            )

        expected_primary_beam_dims = (
            "time",
            "frequency",
            "polarization",
            "l",
            "m",
        )
        missing_primary_beam_dims = [
            dim
            for dim in expected_primary_beam_dims
            if dim not in cube_primary_beam_da.dims
        ]
        if missing_primary_beam_dims:
            raise ValueError(
                f"Primary-beam variable {primary_beam_name!r} cannot be converted "
                f"to the expected cube dimensions {expected_primary_beam_dims}. "
                f"Missing dimensions: {missing_primary_beam_dims}; "
                f"found {cube_primary_beam_da.dims}."
            )
        cube_xds[primary_beam_name] = cube_primary_beam_da.transpose(
            *expected_primary_beam_dims
        )

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
    if primary_beam_correction:
        cube_residual_group[primary_beam_key] = primary_beam_name

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
        primary_beam_correction=primary_beam_correction,
        primary_beam_limit=primary_beam_limit,
        primary_beam_key=primary_beam_key,
    )

    restored_group = cube_xds.attrs["data_groups"][image_data_group_out_restore_name]
    restored_name = restored_group["sky"]

    if restored_name not in cube_xds:
        raise RuntimeError(
            "The cube restoration routine registered restored variable "
            f"{restored_name!r}, but did not create it."
        )

    # ------------------------------------------------------------------
    # Store the restored reference-frequency plane without a spectral/Taylor
    # axis. There is only one restored continuum plane; aligning it to the
    # multi-term coordinate would otherwise create NaN-padded fake products.
    # ------------------------------------------------------------------
    restored_da = cube_xds[restored_name].isel(frequency=0, drop=True)

    img_xds[restored_name] = restored_da

    continuum_restored_group = copy.deepcopy(restored_group)
    if primary_beam_correction:
        corrected_name = restored_group["sky_primary_beam_corrected"]
        if corrected_name not in cube_xds:
            raise RuntimeError(
                "The cube restoration routine registered primary-beam-corrected "
                f"variable {corrected_name!r}, but did not create it."
            )
        continuum_corrected_name = "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
        img_xds[continuum_corrected_name] = cube_xds[corrected_name].isel(
            frequency=0, drop=True
        )
        continuum_restored_group["sky_primary_beam_corrected"] = (
            continuum_corrected_name
        )

    img_xds.attrs.setdefault("data_groups", {})[image_data_group_out_restore_name] = (
        continuum_restored_group
    )

    return img_xds, timing_df
