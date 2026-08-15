import copy

import numpy as np
import toolviper.utils.logger as logger
import xarray as xr

from astroviper.utils.param_docs import shares_param_docs


def _equalize_parallel_hand_weights(
    data_weight: np.ndarray, casa_weighting_implementation: bool
) -> np.ndarray:
    """Collapse parallel-hand correlation weights to a single equalized weight.

    Stokes images are formed from sums and differences of correlation pairs (see
    Moellenbrock 2025, "Weights Equalization Analysis of Correlation Combinations
    for Stokes Parameters"). To ensure each member of a correlation pair contributes
    to its grid at the correct level so that the Stokes formation identities are
    satisfied per visibility pair, the per-correlation weights must be equalized
    *before* gridding. For the parallel-hand pair forming Stokes I, the formal
    error-propagation analysis gives the equalized per-correlation weight as

        w_I / 2 = 2 * w_xx * w_yy / (w_xx + w_yy)

    The legacy CASA implementation instead uses the arithmetic mean
    ``(w_xx + w_yy) / 2``. This is strictly incorrect except in the special case
    ``w_xx == w_yy``; it tends to overweight less-reliable visibilities and yields
    an over-optimistic sensitivity estimate. The arithmetic form is retained as an
    option for compatibility with CASA reference results.

    Parameters
    ----------
    data_weight : numpy.ndarray
        Weight array with polarization on the last axis. For 2 polarizations, axes
        0 and 1 are treated as the parallel-hand pair (XX/YY or RR/LL). For 4
        polarizations, axes 0 and 3 are treated as the parallel-hand pair and the
        cross-hands are dropped (the current implementation propagates only the
        parallel-hand equalization to the cross-hands; see Moellenbrock 2025
        "A Small Complication"). For any other polarization count, the array is
        returned unchanged.
    casa_weighting_implementation : bool
        If True, use the legacy CASA arithmetic-mean equalization. If False, use the
        formally correct error-propagation equalization (default in callers).

    Returns
    -------
    numpy.ndarray
        Weight array with the polarization axis reduced to length 1, or the input
        array unchanged if its polarization count is not 2 or 4.
    """
    n_pol = data_weight.shape[-1]
    if n_pol == 2:
        w0, w1 = data_weight[..., 0], data_weight[..., 1]
    elif n_pol == 4:
        w0, w1 = data_weight[..., 0], data_weight[..., 3]
    else:
        return data_weight
    if casa_weighting_implementation:
        equalized = (w0 + w1) / 2
    else:
        equalized = (2 * w0 * w1) / (w0 + w1)
    return equalized[..., np.newaxis]


def normalize_imaging_weight_params(imaging_weights_params):
    """Validate and normalize the imaging-weight configuration."""
    import copy

    from astroviper.processing_functions.imaging.check_imaging_parameters import (
        check_imaging_weights_params,
    )

    params = copy.deepcopy(imaging_weights_params)

    if not check_imaging_weights_params(params):
        raise ValueError("Invalid imaging-weight parameters.")

    if params["weighting"] == "uniform":
        params["weighting"] = "briggs"
        params["robust"] = -2.0

    return params


def prepare_local_data_weights(
    ms_xds,
    ms_data_group,
    *,
    casa_weighting_implementation,
):
    """Return flagged and parallel-hand-equalized data weights."""
    import numpy as np

    data_weight = np.array(
        ms_xds[ms_data_group["weight"]].values,
        copy=True,
    )

    flag = ms_xds[ms_data_group["flag"]].values
    data_weight[flag == 1] = np.nan

    return _equalize_parallel_hand_weights(
        data_weight,
        casa_weighting_implementation,
    )


def collapse_continuum_weight_density(weight_density_xds: xr.Dataset) -> xr.Dataset:
    """Collapse channel-dependent density contributions for continuum Briggs.

    CASA continuum weighting forms one common UV-density plane from every
    selected channel before calculating a single Briggs factor. This function
    performs that frequency collapse after distributed partitions have been
    aligned and reduced.

    Parameters
    ----------
    weight_density_xds : xarray.Dataset
        Reduced dataset containing ``WEIGHT_DENSITY_GRID`` with dimensions
        ``(frequency, weight_polarization, u, v)`` and ``SUM_WEIGHT`` with
        dimensions ``(frequency, weight_polarization)``.

    Returns
    -------
    xarray.Dataset
        A copy containing one stored continuum density plane and one summed
        weight plane. The ``continuum_frequency_collapsed`` attribute marks
        that the singleton frequency coordinate represents all input channels.
    """
    required = ("WEIGHT_DENSITY_GRID", "SUM_WEIGHT")
    missing = [name for name in required if name not in weight_density_xds]
    if missing:
        raise KeyError(f"Weight-density dataset is missing variables {missing}.")

    density = weight_density_xds["WEIGHT_DENSITY_GRID"]
    sum_weight = weight_density_xds["SUM_WEIGHT"]
    expected_density_dims = ("frequency", "weight_polarization", "u", "v")
    expected_sum_dims = ("frequency", "weight_polarization")
    if density.dims != expected_density_dims:
        raise ValueError(
            f"WEIGHT_DENSITY_GRID dimensions are {density.dims}; expected "
            f"{expected_density_dims}."
        )
    if sum_weight.dims != expected_sum_dims:
        raise ValueError(
            f"SUM_WEIGHT dimensions are {sum_weight.dims}; expected "
            f"{expected_sum_dims}."
        )
    if density.sizes["frequency"] == 0:
        raise ValueError("Cannot collapse an empty continuum frequency axis.")

    n_input_frequency_channels = int(density.sizes["frequency"])
    representative_frequency = float(
        np.mean(np.asarray(weight_density_xds.frequency.values, dtype=np.float64))
    )
    collapsed_density = (
        density.sum(dim="frequency")
        .expand_dims(frequency=[representative_frequency])
        .transpose(*expected_density_dims)
    )
    collapsed_sum_weight = (
        sum_weight.sum(dim="frequency")
        .expand_dims(frequency=[representative_frequency])
        .transpose(*expected_sum_dims)
    )

    result = weight_density_xds.drop_vars(required).drop_dims("frequency")
    result["WEIGHT_DENSITY_GRID"] = collapsed_density
    result["SUM_WEIGHT"] = collapsed_sum_weight
    result.attrs = weight_density_xds.attrs.copy()
    result.attrs["continuum_frequency_collapsed"] = True
    result.attrs["n_input_frequency_channels"] = n_input_frequency_channels
    return result


# @shares_param_docs
def grid_imaging_weight_density_continuum(
    ps_xdt: xr.DataTree,
    img_xds: xr.Dataset,
    imaging_weights_params: dict,
    ms_data_group_in_name: str = "base",
    single_precision_gridding: bool = False,
    processing_function_threads: int = 1,
) -> xr.Dataset:
    """Grid one visibility partition's contribution to the weight density.

    This function implements the first, partition-local stage of distributed
    Briggs or uniform weighting. It masks flagged data weights, equalizes the
    parallel-hand correlation weights, and grids the resulting weights onto a
    channel-dependent UV-density grid.

    No Briggs factors are calculated and no weights are degridded back to the
    visibility samples. The returned density and sum-of-weight products are
    intended to be accumulated across visibility partitions by a distributed
    reduction stage.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing-set partition containing one or more MeasurementSet-like
        datasets.

    img_xds : xarray.Dataset
        Image dataset supplying the local frequency coordinate, image size, and
        angular cell size.

    imaging_weights_params : dict
        Imaging-weight configuration. Only Briggs and uniform weighting require
        this function. Uniform weighting is treated as Briggs weighting with
        ``robust=-2``.

        The entry ``casa_weighting_implementation`` selects the parallel-hand
        equalization convention.

    ms_data_group_in_name : str, optional
        Processing-set data group containing the raw weights, flags, and UVW
        coordinates.

    single_precision_gridding : bool, optional
        If true, use ``float32`` for the weight-density grid. Otherwise use
        ``float64``. ``SUM_WEIGHT`` is always accumulated in ``float64``.

    processing_function_threads : int, optional
        Number of threads supplied to the weight-density gridder.

    Returns
    -------
    xarray.Dataset
        Dataset containing

        ``WEIGHT_DENSITY_GRID``
            Partition-local UV-density contribution with dimensions
            ``(frequency, weight_polarization, u, v)``.

        ``SUM_WEIGHT``
            Partition-local sum of equalized data weights with dimensions
            ``(frequency, weight_polarization)``.

        The physical frequency coordinate is retained so that a later reducer
        can align and sum contributions from arbitrary time, baseline, or
        frequency partitions.

    Notes
    -----
    This function does not modify ``ps_xdt``. In particular, flags are applied
    to a copy of the raw data-weight array rather than replacing flagged
    elements in the processing-set weight variable.

    Natural weighting does not require a UV-density grid and is therefore
    rejected by this function.
    """
    import copy

    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.imaging.check_imaging_parameters import (
        check_imaging_weights_params,
    )
    from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights import (
        grid_imaging_weights,
    )

    # ------------------------------------------------------------------
    # Validate and normalize the weighting configuration.
    # ------------------------------------------------------------------
    if not isinstance(imaging_weights_params, dict):
        raise TypeError(
            "imaging_weights_params must be a dictionary; received "
            f"{type(imaging_weights_params).__name__}."
        )

    weight_params = copy.deepcopy(imaging_weights_params)

    if not check_imaging_weights_params(weight_params):
        raise ValueError("imaging_weights_params validation failed.")

    weighting = str(weight_params["weighting"]).lower()

    if weighting == "natural":
        raise ValueError("Natural weighting does not require a weight-density grid.")

    if weighting == "uniform":
        # The robust value is not used during density gridding, but normalize
        # the configuration here so that all stages use the same convention.
        weight_params["weighting"] = "briggs"
        weight_params["robust"] = -2.0

    elif weighting != "briggs":
        raise ValueError(
            "grid_imaging_weight_density_continuum supports only "
            f"'briggs' and 'uniform'; received {weighting!r}."
        )

    casa_weighting_implementation = bool(weight_params["casa_weighting_implementation"])

    # ------------------------------------------------------------------
    # Resolve the common local output layout.
    # ------------------------------------------------------------------
    required_image_dimensions = ("frequency", "l", "m")

    missing_image_dimensions = [
        dim for dim in required_image_dimensions if dim not in img_xds.sizes
    ]

    if missing_image_dimensions:
        raise ValueError(
            "img_xds is missing dimensions required for weight-density "
            f"gridding: {missing_image_dimensions}."
        )

    frequency = np.asarray(
        img_xds.coords["frequency"].values,
        dtype=np.float64,
    )

    if frequency.ndim != 1:
        raise ValueError(
            "The image frequency coordinate must be one-dimensional; "
            f"received shape {frequency.shape}."
        )

    if frequency.size == 0:
        raise ValueError(
            "The image frequency coordinate must contain at least one channel."
        )

    if not np.all(np.isfinite(frequency)):
        raise ValueError("The image frequency coordinate contains non-finite values.")

    n_uv = np.asarray(
        [
            img_xds.sizes["l"],
            img_xds.sizes["m"],
        ],
        dtype=np.int64,
    )

    delta_lm = np.asarray(
        img_xds.xr_img.get_lm_cell_size(),
        dtype=np.float64,
    )

    if delta_lm.shape != (2,):
        raise ValueError(
            "The image cell-size accessor must return two angular cell sizes; "
            f"received shape {delta_lm.shape}."
        )

    density_dtype = np.float32 if single_precision_gridding else np.float64

    # The current weighting kernels operate on one equalized polarization
    # plane. The final imaging weights are tiled over the requested image
    # correlations during the later degrid stage.
    n_weight_polarization = 1

    weight_density_grid = np.zeros(
        (
            frequency.size,
            n_weight_polarization,
            n_uv[0],
            n_uv[1],
        ),
        dtype=density_dtype,
    )

    sum_weight = np.zeros(
        (
            frequency.size,
            n_weight_polarization,
        ),
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Accumulate every MS child belonging to this visibility partition.
    # ------------------------------------------------------------------
    datasets_gridded = 0

    for ms_name, ms_xds in ps_xdt.items():
        data_groups = ms_xds.attrs.get("data_groups", {})

        if ms_data_group_in_name not in data_groups:
            raise KeyError(
                f"Processing-set child {ms_name!r} does not contain data "
                f"group {ms_data_group_in_name!r}."
            )

        data_group = data_groups[ms_data_group_in_name]

        required_roles = (
            "uvw",
            "weight",
            "flag",
        )

        missing_roles = [role for role in required_roles if role not in data_group]

        if missing_roles:
            raise KeyError(
                f"Processing-set data group {ms_data_group_in_name!r} in "
                f"child {ms_name!r} is missing roles {missing_roles}."
            )

        required_variables = {role: data_group[role] for role in required_roles}

        missing_variables = [
            variable_name
            for variable_name in required_variables.values()
            if variable_name not in ms_xds
        ]

        if missing_variables:
            raise KeyError(
                f"Processing-set child {ms_name!r} is missing variables "
                f"{missing_variables}."
            )

        if "frequency" not in ms_xds.coords:
            raise KeyError(
                f"Processing-set child {ms_name!r} does not contain a "
                "'frequency' coordinate."
            )

        ms_frequency = np.asarray(
            ms_xds.coords["frequency"].values,
            dtype=np.float64,
        )

        if ms_frequency.ndim != 1:
            raise ValueError(
                f"The frequency coordinate for child {ms_name!r} must be "
                f"one-dimensional; received shape {ms_frequency.shape}."
            )

        frequency_matches = np.isclose(
            ms_frequency[:, np.newaxis],
            frequency[np.newaxis, :],
            rtol=1.0e-12,
            atol=0.0,
        )
        match_counts = frequency_matches.sum(axis=1)
        if np.any(match_counts != 1):
            raise ValueError(
                f"The frequencies in processing-set child {ms_name!r} do not "
                "each match exactly one channel in the local image frequency "
                "axis. "
                f"MS frequencies={ms_frequency}; "
                f"image frequencies={frequency}."
            )
        image_frequency_indices = np.argmax(frequency_matches, axis=1)
        if np.unique(image_frequency_indices).size != ms_frequency.size:
            raise ValueError(
                f"The frequencies in processing-set child {ms_name!r} do not "
                "map one-to-one onto the local image frequency axis. "
                f"MS frequencies={ms_frequency}; "
                f"image frequencies={frequency}."
            )

        uvw = np.asarray(ms_xds[data_group["uvw"]].values)

        # Make a copy because applying the flags must not mutate the raw
        # processing-set WEIGHT variable.
        data_weight = np.array(
            ms_xds[data_group["weight"]].values,
            copy=True,
        )

        flag = np.asarray(ms_xds[data_group["flag"]].values)

        if flag.shape != data_weight.shape:
            raise ValueError(
                f"FLAG and WEIGHT have different shapes for child "
                f"{ms_name!r}: {flag.shape} != {data_weight.shape}."
            )

        data_weight[flag == 1] = np.nan

        data_weight = _equalize_parallel_hand_weights(
            data_weight,
            casa_weighting_implementation,
        )

        if data_weight.shape[-1] != n_weight_polarization:
            raise ValueError(
                "Parallel-hand equalization must produce one weight "
                f"polarization; received shape {data_weight.shape} for "
                f"child {ms_name!r}."
            )

        uses_full_frequency_axis = np.array_equal(
            image_frequency_indices,
            np.arange(frequency.size),
        )
        if uses_full_frequency_axis:
            child_weight_density_grid = weight_density_grid
            child_sum_weight = sum_weight
        else:
            child_weight_density_grid = np.zeros(
                (
                    ms_frequency.size,
                    n_weight_polarization,
                    n_uv[0],
                    n_uv[1],
                ),
                dtype=density_dtype,
            )
            child_sum_weight = np.zeros(
                (ms_frequency.size, n_weight_polarization),
                dtype=np.float64,
            )

        grid_imaging_weights(
            child_weight_density_grid,
            child_sum_weight,
            uvw,
            data_weight,
            ms_frequency,
            n_uv,
            delta_lm,
            processing_function_threads=processing_function_threads,
            truncate_uv_cells=True,
        )

        if not uses_full_frequency_axis:
            weight_density_grid[
                image_frequency_indices, ...
            ] += child_weight_density_grid
            sum_weight[image_frequency_indices, ...] += child_sum_weight

        datasets_gridded += 1

    if datasets_gridded == 0:
        raise RuntimeError(
            "No processing-set datasets were gridded into the local "
            "weight-density contribution."
        )

    # ------------------------------------------------------------------
    # Package the numerical output with coordinates and metadata.
    # ------------------------------------------------------------------
    result_xds = xr.Dataset(
        data_vars={
            "WEIGHT_DENSITY_GRID": xr.DataArray(
                weight_density_grid,
                dims=(
                    "frequency",
                    "weight_polarization",
                    "u",
                    "v",
                ),
                coords={
                    "frequency": frequency,
                    "weight_polarization": np.arange(
                        n_weight_polarization,
                        dtype=np.int64,
                    ),
                    "u": np.arange(n_uv[0], dtype=np.int64),
                    "v": np.arange(n_uv[1], dtype=np.int64),
                },
                attrs={
                    "description": (
                        "Partition-local contribution to the global "
                        "imaging-weight density grid."
                    ),
                },
            ),
            "SUM_WEIGHT": xr.DataArray(
                sum_weight,
                dims=(
                    "frequency",
                    "weight_polarization",
                ),
                coords={
                    "frequency": frequency,
                    "weight_polarization": np.arange(
                        n_weight_polarization,
                        dtype=np.int64,
                    ),
                },
                attrs={
                    "description": (
                        "Partition-local sum of flagged and equalized "
                        "visibility data weights."
                    ),
                },
            ),
        },
        attrs={
            "weighting": weight_params["weighting"],
            "robust": weight_params.get("robust"),
            "casa_weighting_implementation": (casa_weighting_implementation),
            "n_processing_set_datasets_gridded": datasets_gridded,
            "cell_size_l": float(delta_lm[0]),
            "cell_size_m": float(delta_lm[1]),
        },
    )

    return result_xds


# @shares_param_docs
def degrid_imaging_weights_continuum(
    ps_xdt: xr.DataTree,
    img_xds: xr.Dataset,
    global_weighting_xds: xr.Dataset,
    imaging_weights_params: dict,
    ms_data_group_in_name: str = "base",
    ms_data_group_out_name: str = "imaging",
    ms_data_group_out_modified: dict | None = None,
    overwrite: bool = False,
    processing_function_threads: int = 1,
) -> xr.DataTree:
    """Create per-visibility imaging weights from the global density grid.

    This function implements the partition-local numerical stage of the second
    distributed continuum weighting graph. It receives the globally accumulated
    weight-density grid and globally calculated Briggs factors, selects the
    frequency planes required by each local processing-set dataset, and degrids
    the final imaging weights onto the corresponding visibility samples.

    Parallel-hand data weights are flagged and equalized locally before
    degridding. The resulting single-polarization imaging weights are tiled over
    the original correlation axis and registered under the logical
    ``"weight_imaging"`` role in the requested output data group.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Local processing-set partition for which imaging weights are created.

    img_xds : xarray.Dataset
        Image dataset providing image size and angular cell size. Its frequency
        coordinate is not required to equal the full global frequency axis.

    global_weighting_xds : xarray.Dataset
        Globally reduced weighting dataset containing

        ``WEIGHT_DENSITY_GRID``
            Dimensions ``(frequency, weight_polarization, u, v)``.

        ``SUM_WEIGHT``
            Dimensions ``(frequency, weight_polarization)``.

        ``BRIGGS_FACTORS``
            Dimensions
            ``(briggs_parameter, frequency, weight_polarization)``.

    imaging_weights_params : dict
        Imaging-weight configuration. Uniform weighting is normalized to Briggs
        weighting with ``robust=-2``.

    ms_data_group_in_name : str, optional
        Name of the processing-set input data group containing UVW, raw weights,
        and flags.

    ms_data_group_out_name : str, optional
        Name of the processing-set data group receiving the final imaging
        weights.

    ms_data_group_out_modified : dict, optional
        Mapping describing the output imaging-weight variable. Defaults to

        ``{"weight_imaging": "WEIGHT_IMAGING"}``.

    overwrite : bool, optional
        Whether an existing output variable may be replaced.

    processing_function_threads : int, optional
        Number of threads supplied to the degridding kernel.

    Returns
    -------
    xarray.DataTree
        The input processing-set partition with ``WEIGHT_IMAGING`` created and
        registered for every contained MeasurementSet-like dataset.

    Notes
    -----
    This function does not calculate a density grid or Briggs factors. Those
    products must be calculated globally before this function is called.

    Frequency planes are matched by physical frequency coordinate rather than
    by local positional index. This allows the global weighting dataset to
    contain channels contributed by multiple distributed partitions.
    """
    import copy

    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights import (
        degrid_imaging_weights,
    )
    from astroviper.utils.data_group_tools import (
        create_ps_xdt_data_groups_in_and_out,
        modify_data_groups_ps_xdt,
    )

    # ------------------------------------------------------------------
    # Validate and normalize configuration.
    # ------------------------------------------------------------------
    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {
            "weight_imaging": "WEIGHT_IMAGING",
        }

    if not isinstance(global_weighting_xds, xr.Dataset):
        raise TypeError(
            "global_weighting_xds must be an xarray.Dataset; received "
            f"{type(global_weighting_xds).__name__}."
        )

    weight_params = normalize_imaging_weight_params(imaging_weights_params)

    weighting = str(weight_params["weighting"]).lower()

    if weighting != "briggs":
        raise ValueError(
            "degrid_imaging_weights_continuum requires Briggs or uniform "
            f"weighting; normalized weighting is {weighting!r}."
        )

    required_global_variables = (
        "WEIGHT_DENSITY_GRID",
        "SUM_WEIGHT",
        "BRIGGS_FACTORS",
    )

    missing_global_variables = [
        variable_name
        for variable_name in required_global_variables
        if variable_name not in global_weighting_xds
    ]

    if missing_global_variables:
        raise KeyError(
            "global_weighting_xds is missing required variables "
            f"{missing_global_variables}."
        )

    if "frequency" not in global_weighting_xds.coords:
        raise KeyError("global_weighting_xds does not contain a frequency coordinate.")

    global_density_da = global_weighting_xds["WEIGHT_DENSITY_GRID"]
    global_sum_weight_da = global_weighting_xds["SUM_WEIGHT"]
    global_briggs_da = global_weighting_xds["BRIGGS_FACTORS"]

    expected_density_dims = (
        "frequency",
        "weight_polarization",
        "u",
        "v",
    )
    expected_sum_weight_dims = (
        "frequency",
        "weight_polarization",
    )
    expected_briggs_dims = (
        "briggs_parameter",
        "frequency",
        "weight_polarization",
    )

    if global_density_da.dims != expected_density_dims:
        raise ValueError(
            "WEIGHT_DENSITY_GRID has dimensions "
            f"{global_density_da.dims}; expected "
            f"{expected_density_dims}."
        )

    if global_sum_weight_da.dims != expected_sum_weight_dims:
        raise ValueError(
            "SUM_WEIGHT has dimensions "
            f"{global_sum_weight_da.dims}; expected "
            f"{expected_sum_weight_dims}."
        )

    if global_briggs_da.dims != expected_briggs_dims:
        raise ValueError(
            "BRIGGS_FACTORS has dimensions "
            f"{global_briggs_da.dims}; expected "
            f"{expected_briggs_dims}."
        )

    if global_briggs_da.sizes["briggs_parameter"] != 2:
        raise ValueError(
            "BRIGGS_FACTORS must contain exactly two Briggs parameters; "
            f"received "
            f"{global_briggs_da.sizes['briggs_parameter']}."
        )

    global_frequency = np.asarray(
        global_weighting_xds.coords["frequency"].values,
        dtype=np.float64,
    )
    continuum_frequency_collapsed = bool(
        global_weighting_xds.attrs.get("continuum_frequency_collapsed", False)
    )

    if global_frequency.ndim != 1:
        raise ValueError(
            "The global frequency coordinate must be one-dimensional; "
            f"received shape {global_frequency.shape}."
        )

    if global_frequency.size == 0:
        raise ValueError("The global frequency coordinate contains no channels.")

    if not np.all(np.isfinite(global_frequency)):
        raise ValueError("The global frequency coordinate contains non-finite values.")

    if np.unique(global_frequency).size != global_frequency.size:
        raise ValueError("The global frequency coordinate contains duplicate values.")

    # ------------------------------------------------------------------
    # Resolve the common image geometry used by the degridder.
    # ------------------------------------------------------------------
    for dimension_name in ("l", "m"):
        if dimension_name not in img_xds.sizes:
            raise ValueError(
                f"img_xds does not contain dimension " f"{dimension_name!r}."
            )

    n_uv = np.asarray(
        [
            img_xds.sizes["l"],
            img_xds.sizes["m"],
        ],
        dtype=np.int64,
    )

    if (
        global_density_da.sizes["u"] != n_uv[0]
        or global_density_da.sizes["v"] != n_uv[1]
    ):
        raise ValueError(
            "The global density-grid geometry does not match img_xds: "
            f"density={(global_density_da.sizes['u'], global_density_da.sizes['v'])}, "
            f"image={(n_uv[0], n_uv[1])}."
        )

    delta_lm = np.asarray(
        img_xds.xr_img.get_lm_cell_size(),
        dtype=np.float64,
    )

    if delta_lm.shape != (2,):
        raise ValueError(
            "The image cell-size accessor must return two values; "
            f"received shape {delta_lm.shape}."
        )

    # Validate stored grid geometry when available.
    for attr_name, cell_size in (
        ("cell_size_l", delta_lm[0]),
        ("cell_size_m", delta_lm[1]),
    ):
        stored_value = global_weighting_xds.attrs.get(attr_name)

        if stored_value is not None and not np.isclose(
            float(stored_value),
            float(cell_size),
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                f"Global weighting metadata {attr_name!r}="
                f"{stored_value} does not match img_xds value "
                f"{cell_size}."
            )

    # ------------------------------------------------------------------
    # Create the input/output processing-set data groups.
    # ------------------------------------------------------------------
    ms_data_group_in, ms_data_group_out = create_ps_xdt_data_groups_in_and_out(
        ps_xdt,
        data_group_in_name=ms_data_group_in_name,
        data_group_out_name=ms_data_group_out_name,
        data_group_out_modified=copy.deepcopy(ms_data_group_out_modified),
        overwrite=overwrite,
    )

    casa_weighting_implementation = bool(weight_params["casa_weighting_implementation"])

    datasets_processed = 0

    # ------------------------------------------------------------------
    # Degrid the global density onto every local visibility partition.
    # ------------------------------------------------------------------
    for ms_name, ms_xds in ps_xdt.items():
        if "frequency" not in ms_xds.coords:
            raise KeyError(
                f"Processing-set child {ms_name!r} does not contain a "
                "'frequency' coordinate."
            )

        local_frequency = np.asarray(
            ms_xds.coords["frequency"].values,
            dtype=np.float64,
        )

        if local_frequency.ndim != 1:
            raise ValueError(
                f"The frequency coordinate for child {ms_name!r} must "
                f"be one-dimensional; received "
                f"shape {local_frequency.shape}."
            )

        if not np.all(np.isfinite(local_frequency)):
            raise ValueError(
                f"The frequency coordinate for child {ms_name!r} "
                "contains non-finite values."
            )

        if continuum_frequency_collapsed:
            if global_frequency.size != 1:
                raise ValueError(
                    "A frequency-collapsed continuum density must contain "
                    "exactly one stored density plane."
                )
            local_density_grid = np.repeat(
                np.asarray(global_density_da.values),
                local_frequency.size,
                axis=0,
            )
            local_briggs_factors = np.repeat(
                np.asarray(global_briggs_da.values),
                local_frequency.size,
                axis=1,
            )
        else:
            # Match local frequencies to global planes. Using nearest matching
            # with a tight tolerance avoids depending on exact floating-point
            # identity.
            global_indices = []

            for local_value in local_frequency:
                close_indices = np.flatnonzero(
                    np.isclose(
                        global_frequency,
                        local_value,
                        rtol=1.0e-12,
                        atol=0.0,
                    )
                )

                if close_indices.size == 0:
                    raise KeyError(
                        f"Frequency {local_value} Hz from child "
                        f"{ms_name!r} is absent from the global "
                        "weight-density grid."
                    )

                if close_indices.size > 1:
                    raise ValueError(
                        f"Frequency {local_value} Hz from child "
                        f"{ms_name!r} matches multiple global planes."
                    )

                global_indices.append(int(close_indices[0]))

            global_indices = np.asarray(
                global_indices,
                dtype=np.int64,
            )

            # Preserve the local channel order expected by the degridding kernel.
            local_density_grid = np.asarray(
                global_density_da.isel(frequency=global_indices).values
            )

            local_briggs_factors = np.asarray(
                global_briggs_da.isel(frequency=global_indices).values
            )

        expected_local_density_shape = (
            local_frequency.size,
            global_density_da.sizes["weight_polarization"],
            n_uv[0],
            n_uv[1],
        )

        if local_density_grid.shape != expected_local_density_shape:
            raise ValueError(
                f"Selected density grid for child {ms_name!r} has shape "
                f"{local_density_grid.shape}; expected "
                f"{expected_local_density_shape}."
            )

        expected_local_briggs_shape = (
            2,
            local_frequency.size,
            global_density_da.sizes["weight_polarization"],
        )

        if local_briggs_factors.shape != expected_local_briggs_shape:
            raise ValueError(
                f"Selected Briggs factors for child {ms_name!r} have "
                f"shape {local_briggs_factors.shape}; expected "
                f"{expected_local_briggs_shape}."
            )

        required_input_variables = (
            ms_data_group_in["uvw"],
            ms_data_group_in["weight"],
            ms_data_group_in["flag"],
        )

        missing_input_variables = [
            variable_name
            for variable_name in required_input_variables
            if variable_name not in ms_xds
        ]

        if missing_input_variables:
            raise KeyError(
                f"Processing-set child {ms_name!r} is missing input "
                f"variables {missing_input_variables}."
            )

        uvw = np.asarray(ms_xds[ms_data_group_in["uvw"]].values)

        data_weight = prepare_local_data_weights(
            ms_xds,
            ms_data_group_in,
            casa_weighting_implementation=(casa_weighting_implementation),
        )

        if data_weight.shape[-1] != (global_density_da.sizes["weight_polarization"]):
            raise ValueError(
                f"Equalized raw weights for child {ms_name!r} contain "
                f"{data_weight.shape[-1]} polarization planes, but the "
                "global density grid contains "
                f"{global_density_da.sizes['weight_polarization']}."
            )

        imaging_weights = degrid_imaging_weights(
            local_density_grid,
            uvw,
            data_weight,
            local_briggs_factors,
            local_frequency,
            n_uv,
            delta_lm,
            processing_function_threads=(processing_function_threads),
            truncate_uv_cells=True,
        )

        imaging_weights = np.asarray(imaging_weights)

        expected_equalized_shape = data_weight.shape

        if imaging_weights.shape != expected_equalized_shape:
            raise ValueError(
                f"Degridded imaging weights for child {ms_name!r} have "
                f"shape {imaging_weights.shape}; expected "
                f"{expected_equalized_shape}."
            )

        if "polarization" not in ms_xds.sizes:
            raise ValueError(
                f"Processing-set child {ms_name!r} does not contain "
                "a polarization dimension."
            )

        n_polarization = int(ms_xds.sizes["polarization"])

        if imaging_weights.shape[-1] != 1:
            raise ValueError(
                "The current continuum weighting implementation expects "
                "one equalized weight-polarization plane before tiling; "
                f"received shape {imaging_weights.shape} for child "
                f"{ms_name!r}."
            )

        output_weights = np.repeat(
            imaging_weights,
            n_polarization,
            axis=-1,
        )

        output_weight_name = ms_data_group_out["weight_imaging"]

        reference_weight = ms_xds[ms_data_group_in["weight"]]

        if output_weights.shape != reference_weight.shape:
            raise ValueError(
                f"Final imaging weights for child {ms_name!r} have "
                f"shape {output_weights.shape}; expected the raw weight "
                f"shape {reference_weight.shape}."
            )

        ms_xds[output_weight_name] = xr.DataArray(
            output_weights,
            dims=reference_weight.dims,
            coords={
                dimension_name: reference_weight.coords[dimension_name]
                for dimension_name in reference_weight.dims
                if dimension_name in reference_weight.coords
            },
            attrs={
                "description": (
                    "Per-visibility continuum imaging weights derived "
                    "from the globally reduced UV-density grid."
                ),
                "weighting": weight_params["weighting"],
                "robust": weight_params.get("robust"),
            },
        )

        datasets_processed += 1

    if datasets_processed == 0:
        raise RuntimeError("No processing-set datasets were assigned imaging weights.")

    modify_data_groups_ps_xdt(
        ps_xdt,
        data_group_out_name=ms_data_group_out_name,
        data_group_out=ms_data_group_out,
        description=(
            "Continuum imaging weights derived from the globally reduced "
            "weight-density grid and global Briggs factors."
        ),
    )

    return ps_xdt


@shares_param_docs
def calculate_imaging_weights(
    ps_xdt: xr.DataTree,
    img_xds: xr.Dataset,
    imaging_weights_params: dict,
    ms_data_group_in_name: str = "base",
    ms_data_group_out_name: str = "imaging",
    ms_data_group_out_modified: dict = {"weight_imaging": "WEIGHT_IMAGING"},
    overwrite: bool = False,
    single_precision_gridding: bool = False,
    return_weight_density_grid: bool = False,
    processing_function_threads: int = 1,
    truncate_uv_cells: bool = False,
) -> None | np.ndarray:
    """
    Calculate imaging weights for interferometric data.

    Grids per-visibility data weights from a Processing Set (``ps_xdt`` as an
    xarray ``DataTree``), applies the chosen weighting scheme, and degrids the
    weights back onto the constituent MeasurementSet-like datasets in the tree.
    Prior to gridding, parallel-hand correlation weights are *equalized* into a
    single weight per visibility — see Notes and
    :func:`_equalize_parallel_hand_weights` for the formal derivation.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing Set DataTree containing one or more MeasurementSet-like xarray
        Datasets. Each Dataset must include the fields referenced by the data
        group parameters and ``grid_params`` (e.g., UVW, WEIGHT, FLAG, frequency).
    img_xds : xarray.Dataset
        Image xarray Dataset containing image parameters (e.g., image size, cell
        size).
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    ms_data_group_in_name : str, default ``"base"``
        Name of the input data group.
    ms_data_group_out_name : str, default ``"imaging"``
        Name of the output data group.
    ms_data_group_out_modified : dict, optional
        Mapping of output variable names; the ``"weight_imaging"`` key sets the
        name of the output imaging-weight variable. Defaults to
        ``{"weight_imaging": "WEIGHT_IMAGING"}``.
    overwrite : bool, default ``False``
        If True, an existing data variable may be overwritten.
    single_precision_gridding : bool, default ``False``
        If True, use single precision for the weight-density grid (Briggs path
        only; ignored for natural weighting).
    return_weight_density_grid : bool, default ``False``
        If True *and* the weighting scheme requires gridding (Briggs/uniform),
        also return the 2D weight-density grid (useful for debugging). Ignored
        for natural weighting, which always returns ``None``.
    processing_function_threads : int, default ``1``
        Number of threads handed to the per-processing-function (C++ / Numba /
        FFT) kernels.
    truncate_uv_cells : bool, default ``False``
        If True, assign shifted UV coordinates to density cells by integer
        truncation, matching CASA continuum weighting. If False, use the
        nearest-cell convention used by cube weighting.

    Returns
    -------
    weight_density_grid : numpy.ndarray, optional
        Only returned if ``return_weight_density_grid=True`` and the weighting
        scheme is Briggs-like. Array of shape ``(n_chan, 1, n_u, n_v)`` containing
        the weight-density grid.

    Notes
    -----
    - **Natural weighting**: No gridding is performed. Per-visibility weights are
      only equalized across the parallel-hand correlation pair (see
      :func:`_equalize_parallel_hand_weights`) and written to the output data
      group. The values themselves are not rescaled.
    - **Uniform weighting**: Implemented as Briggs with ``robust = -2.0``.
    - **Briggs weighting**: Equalized weights are scaled by robust-dependent
      factors computed from the weight-density grid and the channel-wise
      ``sum_weight``.
    - Flagged visibilities (``flag == 1``) are set to ``NaN`` *in place* on the
      input weight array prior to equalization (memory-constrained design — the
      input weights are mutated by this function).
    - Polarization handling:
        * 2 polarizations (XX, YY or RR, LL): parallel-hand pair is equalized.
        * 4 polarizations (XX, XY, YX, YY): parallel-hand pair (XX, YY) is
          equalized and the resulting weight is applied to all four
          polarizations. Cross-hand weights are not separately equalized in this
          implementation; see Moellenbrock 2025 "A Small Complication" for the
          rationale and caveats.
    - The equalization formula is selected by
      ``imaging_weights_params["casa_weighting_implementation"]`` (see Parameters
      above).

    See Also
    --------
    _equalize_parallel_hand_weights : Parallel-hand weight equalization.
    grid_imaging_weights : Grid per-visibility weights onto a UV grid.
    degrid_imaging_weights : Interpolate imaging weights from the UV grid back to
        visibilities.
    calculate_briggs_params : Compute robust scaling factors for Briggs weighting.

    References
    ----------
    Moellenbrock, G. (2025). "Weights Equalization Analysis of Correlation
    Combinations for Stokes Parameters."

    Examples
    --------
    >>> calculate_imaging_weights(
    ...     ps_xdt,
    ...     img_xds,
    ...     imaging_weights_params={"weighting": "briggs", "robust": 0.5},
    ...     ms_data_group_in_name="base",
    ... )
    """

    from astroviper.processing_functions.imaging.check_imaging_parameters import (
        check_imaging_weights_params,
    )
    from astroviper.processing_functions.imaging.imaging_weighting.briggs_weighting import (
        calculate_briggs_params,
    )
    from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights import (
        degrid_imaging_weights,
        grid_imaging_weights,
    )
    from astroviper.utils.data_group_tools import (
        create_ps_xdt_data_groups_in_and_out,
        modify_data_groups_ps_xdt,
    )

    _imaging_weights_params = copy.deepcopy(imaging_weights_params)
    _ms_data_group_out_modified = copy.deepcopy(ms_data_group_out_modified)
    assert check_imaging_weights_params(
        _imaging_weights_params
    ), "######### ERROR: imaging_weights_params checking failed"

    # Uniform weighting is implemented as Briggs with robust = -2.0.
    if _imaging_weights_params["weighting"] == "uniform":
        _imaging_weights_params["weighting"] = "briggs"
        _imaging_weights_params["robust"] = -2.0

    casa_weighting_implementation = _imaging_weights_params[
        "casa_weighting_implementation"
    ]

    ms_data_group_in, ms_data_group_out = create_ps_xdt_data_groups_in_and_out(
        ps_xdt,
        data_group_in_name=ms_data_group_in_name,
        data_group_out_name=ms_data_group_out_name,
        data_group_out_modified=_ms_data_group_out_modified,
        overwrite=overwrite,
    )

    # for natural weighting, we can skip the gridding and degridding operations
    if _imaging_weights_params["weighting"] == "natural":
        logger.debug(
            "Calculating natural imaging weights (parallel-hand equalization only; "
            "no rescaling of data weights)."
        )

        # fill visibility column
        for ms_name, ms_xdt in ps_xdt.items():
            data_weight = ms_xdt[ms_data_group_in["weight"]].values
            data_weight[ms_xdt[ms_data_group_in["flag"]] == 1] = np.nan
            data_weight = _equalize_parallel_hand_weights(
                data_weight, casa_weighting_implementation
            )

            n_pol = ms_xdt.sizes["polarization"]
            ms_xdt[ms_data_group_out["weight_imaging"]] = xr.DataArray(
                np.tile(data_weight, (1, 1, 1, n_pol)),
                dims=ms_xdt[ms_data_group_out["weight"]].dims,
            )

        modify_data_groups_ps_xdt(
            ps_xdt,
            data_group_out_name=ms_data_group_out_name,
            data_group_out=ms_data_group_out,
            description=(
                "Natural imaging weights; parallel-hand correlation weights "
                "equalized, values otherwise not rescaled."
            ),
        )

        return

    # Briggs (and uniform, routed above) requires the weight-density grid and
    # robust factors, so we grid and degrid here.
    n_uv = np.array([img_xds.sizes["l"], img_xds.sizes["m"]])
    delta_lm = img_xds.xr_img.get_lm_cell_size()
    n_imag_chan = img_xds.sizes["frequency"]
    if single_precision_gridding:
        dtype = np.float32
    else:
        dtype = np.float64
    weight_density_grid = np.zeros((n_imag_chan, 1, n_uv[0], n_uv[1]), dtype=dtype)
    sum_weight = np.zeros((n_imag_chan, 1), dtype=np.double)

    # Grid the weights.
    for ms_name, ms_xdt in ps_xdt.items():
        uvw = ms_xdt[ms_data_group_in["uvw"]].values
        data_weight = ms_xdt[ms_data_group_in["weight"]].values
        data_weight[ms_xdt[ms_data_group_in["flag"]] == 1] = np.nan
        data_weight = _equalize_parallel_hand_weights(
            data_weight, casa_weighting_implementation
        )

        freq_chan = ms_xdt.frequency.values

        grid_imaging_weights(
            weight_density_grid,
            sum_weight,
            uvw,
            data_weight,
            freq_chan,
            n_uv,
            delta_lm,
            processing_function_threads=processing_function_threads,
            truncate_uv_cells=truncate_uv_cells,
        )

    briggs_factors = calculate_briggs_params(
        weight_density_grid, sum_weight, _imaging_weights_params
    )  # 2 x chan x pol

    # Degrid the weights.
    for ms_name, ms_xdt in ps_xdt.items():
        uvw = ms_xdt[ms_data_group_in["uvw"]].values
        data_weight = ms_xdt[ms_data_group_in["weight"]].values
        data_weight[ms_xdt[ms_data_group_in["flag"]] == 1] = np.nan
        data_weight = _equalize_parallel_hand_weights(
            data_weight, casa_weighting_implementation
        )

        freq_chan = ms_xdt.frequency.values

        imaging_weights = degrid_imaging_weights(
            weight_density_grid,
            uvw,
            data_weight,
            briggs_factors,
            freq_chan,
            n_uv,
            delta_lm,
            processing_function_threads=processing_function_threads,
            truncate_uv_cells=truncate_uv_cells,
        )

        n_pol = ms_xdt.sizes["polarization"]
        ms_xdt[ms_data_group_out["weight_imaging"]] = xr.DataArray(
            np.tile(imaging_weights, (1, 1, 1, n_pol)),
            dims=ms_xdt[ms_data_group_out["weight"]].dims,
        )

    modify_data_groups_ps_xdt(
        ps_xdt,
        data_group_out_name=ms_data_group_out_name,
        data_group_out=ms_data_group_out,
        description=(
            f"Briggs imaging weights with robust={_imaging_weights_params['robust']}; "
            "parallel-hand correlation weights equalized, then rescaled by "
            "robust-dependent factors computed from the weight-density grid and "
            "channel-wise sum of data weights."
        ),
    )

    if return_weight_density_grid:
        return weight_density_grid
    else:
        return
