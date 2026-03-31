import numpy as np
from typing import Union, Dict
import xarray as xr
from astroviper.core.imaging.check_imaging_parameters import (
    check_imaging_weights_params,
)
from astroviper.core.imaging.imaging_weighting.grid_imaging_weights import (
    grid_imaging_weights,
    degrid_imaging_weights,
)

from astroviper.core.imaging.imaging_weighting.briggs_weighting import (
    calculate_briggs_params,
)

# from graphviper.parameter_checking.check_params import check_sel_params
from astroviper.utils.data_group_tools import (
    create_ps_xdt_data_groups_in_and_out,
    modify_data_groups_ps_xdt,
)
import copy
import toolviper.utils.logger as logger


def calculate_imaging_weights(
    ps_xdt: xr.DataTree,
    img_xds: xr.Dataset,
    imaging_weights_params: Dict,
    ms_data_group_in_name: str = "base",
    ms_data_group_out_name: str = "imaging",
    ms_data_group_out_modified: dict = {"weight_imaging": "WEIGHT_IMAGING"},
    overwrite: bool = False,
    single_precision_gridding: bool = False,
    return_weight_density_grid: bool = False,
) -> Union[
    None,
    np.ndarray,
]:
    """
    Calculate imaging weights for interferometric data using natural or Briggs weighting.

    This function grids per-visibility data weights from a Processing Set (``ps_iter`` as
    an xarray ``DataTree``), applies the chosen weighting scheme (natural or Briggs with
    a specified robust parameter), and degrids the weights back onto the constituent
    MeasurementSet-like datasets in the tree.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing Set DataTree containing one or more MeasurementSet-like xarray Datasets.
        Each Dataset must include the fields referenced by the data group parameters and
        ``grid_params`` (e.g., UVW, WEIGHT, FLAG, frequency).
    img_xds : xarray.Dataset
        Image xarray Dataset containing image parameters (e.g., image size, cell size).
    imaging_weights_params : dict
        Weighting scheme configuration. Must include:
            - ``weighting`` : {"natural", "briggs"}
                Type of weighting to apply.
            - ``robust`` : float, optional
                Briggs robust parameter (ignored if ``"natural"``).
    ms_data_group_in_name : str, default ``"base"``
        Name of the input data group.
    ms_data_group_out_name : str, default ``"imaging"``
        Name of the output data group.
    ms_data_group_out_modified : dict, optional
        Mapping of output variable names; the ``"weight_imaging"`` key sets
        the name of the output imaging-weight variable. Defaults to
        ``{"weight_imaging": "WEIGHT_IMAGING"}`` for Briggs weighting.
        **Natural weighting uses the same variable name as the input data weights, so this parameter is ignored in that case.**
    overwrite : bool, default ``False``
        If True, an existing data variable may be overwritten.
    return_weight_density_grid : bool, default False
        If True, also return the 2D weight-density grid used for Briggs weighting
        (useful for debugging).
    single_precision_gridding : bool, default False
        If True, use single precision for gridding operations.

    Returns
    -------
    weight_density_grid : numpy.ndarray, optional
        Only returned if ``return_weight_density_grid=True``. Array of shape
        ``(n_chan, 1, n_u, n_v)`` containing the weight-density grid.

    Notes
    -----
    - **Natural weighting**: Imaging weights are identical to the input data weights;
      a new data group is created for bookkeeping, but values are not rescaled.
    - **Briggs weighting**: Imaging weights are scaled by robust-dependent factors
      computed from the weight-density grid and the channel-wise ``sum_weight``.
    - Flagged visibilities (``flag == 1``) are set to ``NaN`` during gridding.
    - Polarization handling:
        * If there are 2 polarizations (XX, YY): parallel hands are averaged.
        * If there are 4 polarizations (XX, XY, YX, YY): XX and YY are averaged and
          the resulting weights are applied to all four polarizations.

    See Also
    --------
    grid_imaging_weights : Grid per-visibility weights onto a UV grid.
    degrid_imaging_weights : Interpolate imaging weights from the UV grid back to visibilities.
    calculate_briggs_params : Compute robust scaling factors for Briggs weighting.

    Examples
    --------
    >>> calculate_imaging_weights(
    ...     ps_xdt,
    ...     imaging_weights_params={"weighting": "briggs", "robust": 0.5},
    ...     ms_data_group_in_name="base",
    ... )
    """
    _imaging_weights_params = copy.deepcopy(imaging_weights_params)
    _ms_data_group_out_modified = copy.deepcopy(ms_data_group_out_modified)
    assert check_imaging_weights_params(
        _imaging_weights_params
    ), "######### ERROR: imaging_weights_params checking failed"

    ms_data_group_in, ms_data_group_out = create_ps_xdt_data_groups_in_and_out(
        ps_xdt,
        data_group_in_name=ms_data_group_in_name,
        data_group_out_name=ms_data_group_out_name,
        data_group_out_modified=_ms_data_group_out_modified,
        overwrite=overwrite,
    )

    if _imaging_weights_params["weighting"] == "natural":
        ms_data_group_out["weight_imaging"] = ms_data_group_in["weight"]
        logger.debug(
            "Calculating natural imaging weights (no rescaling of data weights)."
        )
        modify_data_groups_ps_xdt(
            ps_xdt,
            data_group_out_name=ms_data_group_out_name,
            data_group_out=ms_data_group_out,
            description="Natural imaging weights; data weights used directly with no rescaling.",
        )
        return

    # Briggs weighting requires calculating the weight-density grid and robust factors, so we proceed with gridding and degridding.
    # Grid Weights
    n_uv = np.array([img_xds.sizes["l"], img_xds.sizes["m"]])
    delta_lm = img_xds.xr_img.get_lm_cell_size()
    n_imag_chan = img_xds.sizes["frequency"]
    if single_precision_gridding:
        dtype = np.float32
    else:
        dtype = np.float64
    weight_density_grid = np.zeros((n_imag_chan, 1, n_uv[0], n_uv[1]), dtype=dtype)
    sum_weight = np.zeros((n_imag_chan, 1), dtype=np.double)

    # Grid the Weights
    for ms_name, ms_xdt in ps_xdt.items():
        uvw = ms_xdt[ms_data_group_in["uvw"]].values
        data_weight = ms_xdt[ms_data_group_in["weight"]].values
        data_weight[ms_xdt[ms_data_group_in["flag"]] == 1] = (
            np.nan
        )  # Set flagged data to NaN for weighting.

        if data_weight.shape[3] == 2:
            data_weight = ((data_weight[..., 0] + data_weight[..., 1]) / 2)[
                ..., np.newaxis
            ]

        if data_weight.shape[3] == 4:
            data_weight = ((data_weight[..., 0] + data_weight[..., 3]) / 2)[
                ..., np.newaxis
            ]

        freq_chan = ms_xdt.frequency.values

        grid_imaging_weights(
            weight_density_grid, sum_weight, uvw, data_weight, freq_chan, n_uv, delta_lm
        )

    # Calculate Briggs
    briggs_factors = calculate_briggs_params(
        weight_density_grid, sum_weight, _imaging_weights_params
    )  # 2 x chan x pol
    # print("sum_weight", sum_weight)
    # print("briggs_factors", briggs_factors)
    # print("4 sum of data weights ", np.nansum(data_weight))

    # Degrid the Weights
    for ms_name, ms_xdt in ps_xdt.items():
        uvw = ms_xdt[ms_data_group_in["uvw"]].values
        data_weight = ms_xdt[ms_data_group_in["weight"]].values
        data_weight[ms_xdt[ms_data_group_in["flag"]] == 1] = (
            np.nan
        )  # Set flagged data to NaN for weighting.
        if data_weight.shape[3] == 2:
            data_weight = ((data_weight[..., 0] + data_weight[..., 1]) / 2)[
                ..., np.newaxis
            ]

        if data_weight.shape[3] == 4:
            data_weight = ((data_weight[..., 0] + data_weight[..., 3]) / 2)[
                ..., np.newaxis
            ]

        freq_chan = ms_xdt.frequency.values

        imaging_weights = degrid_imaging_weights(
            weight_density_grid,
            uvw,
            data_weight,
            briggs_factors,
            freq_chan,
            n_uv,
            delta_lm,
        )

        # # Flag data
        # flags = np.any(ms_xdt[data_group_out["flag"]], axis=-1)  #
        # data_weight[flags == 1] = np.nan

        # ms_xdt[data_group_out["weight_imaging"]] = xr.DataArray(
        #     imaging_weights[..., 0], dims=ms_xdt[data_group_out["weight"]].dims[:-1]
        # )
        # print("imaging_weights.shape", imaging_weights.shape)
        # print("imaging_weights.shape", np.tile(imaging_weights, (1, 2, 1, 1)).shape)

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
            f"Briggs imaging weights with robust={imaging_weights_params['robust']}; "
            "data weights rescaled by robust-dependent factors calculated from the "
            "weight-density grid and channel-wise sum of data weights."
        ),
    )

    if return_weight_density_grid:
        return weight_density_grid
    else:
        return
