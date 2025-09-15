from time import time
import numpy as np
from typing import Tuple, Union, Dict
import numpy as np
import xarray as xr
from xarray import DataTree
import xarray as xr
from astroviper.core.imaging.check_imaging_parameters import (
    check_imaging_weights_params,
    check_grid_params,
)
from astroviper.core.imaging.imaging_weighting.grid_imaging_weights import (
    grid_imaging_weights,
    degrid_imaging_weights,
)

from astroviper.core.imaging.imaging_weighting.briggs_weighting import (
    calculate_briggs_params,
)

# from graphviper.parameter_checking.check_params import check_sel_params
from astroviper.utils.check_params import (
    check_sel_params_ps_xdt,
)
import copy


def calculate_imaging_weights(
    ps_xdt: DataTree,
    grid_params: Dict,
    imaging_weights_params: Dict,
    sel_params: Dict,
    return_weight_density_grid: bool = False,
) -> Union[
    Tuple[DataTree, dict],
    Tuple[DataTree, dict, np.ndarray],
]:
    """
    Calculate imaging weights for interferometric data using natural or Briggs weighting.

    This function grids per-visibility data weights from a Processing Set (``ps_xdt`` as
    an xarray ``DataTree``), applies the chosen weighting scheme (natural or Briggs with
    a specified robust parameter), and degrids the weights back onto the constituent
    MeasurementSet-like datasets in the tree.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing Set containing one or more MeasurementSet-like xarray Datasets.
        Each Dataset must include the fields referenced by ``sel_params`` and
        ``grid_params`` (e.g., UVW, WEIGHT, FLAG, frequency).
    grid_params : dict
        Gridder parameters. Must include:
            - ``image_size`` : tuple of int
                UV grid size as ``(n_u, n_v)``.
            - ``cell_size`` : array-like of float
                Angular cell size (radians) per UV pixel, typically length 2.
    imaging_weights_params : dict
        Weighting scheme configuration. Must include:
            - ``weighting`` : {"natural", "briggs"}
                Type of weighting to apply.
            - ``robust`` : float, optional
                Briggs robust parameter (ignored if ``"natural"``).
    sel_params : dict
        Selection parameters for input/output data groups. Defines which columns
        (e.g., weight, uvw, flag) are read and where output imaging weights are stored.
        Common keys include:
            - ``data_group_in_name`` : str, default ``"base"``
                Name of the input data group.
            - ``overwrite`` : bool, default ``False``
                If True, an existing data variable may be overwritten.
            - ``data_group_out`` : dict, default ``{"weight_imaging": "WEIGHT"}``
                Mapping of output variable names; the ``"weight_imaging"`` key sets
                the name of the output imaging-weight variable.
            - ``data_group_out_name`` : str, default ``"imaging"``
                Name of the output data group.
    return_weight_density_grid : bool, default False
        If True, also return the 2D weight-density grid used for Briggs weighting
        (useful for debugging).

    Returns
    -------
    ps_xdt : xarray.DataTree
        The input Processing Set updated with imaging weights written to each leaf Dataset.
    data_group_out : dict
        Metadata describing the output data group (e.g., names, descriptions, timestamps).
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
    >>> ps_xdt, data_group_out = calculate_imaging_weights(
    ...     ps_xdt,
    ...     grid_params={
    ...         "image_size": (256, 256),
    ...         "cell_size": np.array([-0.1, 0.1]) * np.pi / (180 * 3600),
    ...         "fft_padding": 1.0,
    ...     },
    ...     imaging_weights_params={"weighting": "briggs", "robust": 0.5},
    ...     sel_params={"data_group_in_name": "base"},
    ... )
    """
    _sel_params = copy.deepcopy(sel_params)
    _imaging_weights_params = copy.deepcopy(imaging_weights_params)
    assert check_imaging_weights_params(
        _imaging_weights_params
    ), "######### ERROR: imaging_weights_params checking failed"

    if _imaging_weights_params["weighting"] == "natural":
        _sel_params["overwrite"] = True  # No actual overwrite is occuring.
        data_group_in, data_group_out = check_sel_params_ps_xdt(
            ps_xdt,
            _sel_params,
            default_data_group_in_name="base",
            default_data_group_out_name="imaging",
            default_data_group_out_modified={"weight_imaging": "WEIGHT"},
        )
        description = "Data group created for natural imaging weights with ."

        data_group_out_name = data_group_out["data_group_out_name"]
        del data_group_out["data_group_out_name"]
        from datetime import datetime, timezone

        for ms_xdt in ps_xdt.values():
            now = datetime.now(timezone.utc)
            ms_xdt.data_groups[data_group_out_name] = data_group_out
            ms_xdt.data_groups[data_group_out_name]["date"] = now.isoformat()
            ms_xdt.data_groups[data_group_out_name]["description"] = description

            data_group_out["data_group_out_name"] = data_group_out_name

        return ps_xdt, data_group_out
    else:
        data_group_in, data_group_out = check_sel_params_ps_xdt(
            ps_xdt,
            _sel_params,
            default_data_group_in_name="base",
            default_data_group_out_name="imaging",
            default_data_group_out_modified={"weight_imaging": "WEIGHT_IMAGING"},
        )
        description = (
            "Data group created for briggs imaging weights with robust value "
            + str(_imaging_weights_params["robust"])
        )

    _grid_params = copy.deepcopy(grid_params)
    assert check_grid_params(
        _grid_params
    ), "######### ERROR: grid_params checking failed"
    _grid_params["image_size_padded"] = _grid_params[
        "image_size"
    ]  # do not need to pad since no fft

    _grid_params["n_imag_chan"] = ps_xdt.xr_ps.get_freq_axis().size

    # Grid Weights
    n_uv = _grid_params["image_size_padded"]
    n_imag_chan = _grid_params["n_imag_chan"]
    weight_density_grid = np.zeros((n_imag_chan, 1, n_uv[0], n_uv[1]), dtype=np.double)
    # weight_density_grid = np.zeros((n_imag_chan, 1, n_uv[0], n_uv[1]), dtype=np.float32)
    sum_weight = np.zeros((n_imag_chan, 1), dtype=np.double)

    # Grid the Weights
    for ms_xdt in ps_xdt.values():
        uvw = ms_xdt[data_group_out["uvw"]].values
        data_weight = ms_xdt[data_group_out["weight"]].values
        data_weight[ms_xdt[data_group_out["flag"]] == 1] = (
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
            weight_density_grid, sum_weight, uvw, data_weight, freq_chan, _grid_params
        )

    # Calculate Briggs
    briggs_factors = calculate_briggs_params(
        weight_density_grid, sum_weight, _imaging_weights_params
    )  # 2 x chan x pol
    # print("sum_weight", sum_weight)
    # print("briggs_factors", briggs_factors)
    # print("4 sum of data weights ", np.nansum(data_weight))

    # Degrid the Weights

    data_group_out_name = data_group_out["data_group_out_name"]
    del data_group_out["data_group_out_name"]

    for ms_xdt in ps_xdt.values():
        uvw = ms_xdt[data_group_out["uvw"]].values
        data_weight = ms_xdt[data_group_out["weight"]].values
        data_weight[ms_xdt[data_group_out["flag"]] == 1] = (
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
            _grid_params,
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

        ms_xdt[data_group_out["weight_imaging"]] = xr.DataArray(
            np.tile(imaging_weights, (1, 1, 1, n_pol)),
            dims=ms_xdt[data_group_out["weight"]].dims,
        )

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        ms_xdt.data_groups[data_group_out_name] = data_group_out
        ms_xdt.data_groups[data_group_out_name]["date"] = now.isoformat()
        ms_xdt.data_groups[data_group_out_name]["description"] = description

        data_group_out["data_group_out_name"] = data_group_out_name

    if return_weight_density_grid:
        return ps_xdt, data_group_out, weight_density_grid
    else:
        return ps_xdt, data_group_out
