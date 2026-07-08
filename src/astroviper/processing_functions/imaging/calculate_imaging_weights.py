import copy
from typing import Dict, Union

import numpy as np
import toolviper.utils.logger as logger
import xarray as xr

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
    num_threads: int = 1,
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
    num_threads : int, default ``1``
        Thread count forwarded from ``processing_function_threads`` for the
        weight grid/degrid step. The weight-density grid/degrid kernels are
        currently serial Numba kernels (the grid accumulation is kept serial for
        bit-reproducible weight sums), so this value is propagated for API
        consistency and future parallelization but does not yet change runtime.

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

    if _imaging_weights_params["weighting"] == "natural":
        logger.debug(
            "Calculating natural imaging weights (parallel-hand equalization only; "
            "no rescaling of data weights)."
        )

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
            num_threads=num_threads,
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
            num_threads=num_threads,
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
