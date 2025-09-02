from time import time
import numpy as np
import scipy

# import cngi._utils._constants as const
from scipy import constants
from numba import jit
import numba
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
from astroviper.utils.check_params import check_params, check_sel_params
import copy


def calculate_imaging_weights(ms_xds, grid_params, imaging_weights_params, sel_params):
    _sel_params = copy.deepcopy(sel_params)
    _imaging_weights_params = copy.deepcopy(imaging_weights_params)
    assert check_imaging_weights_params(
        _imaging_weights_params
    ), "######### ERROR: imaging_weights_params checking failed"

    if _imaging_weights_params["weighting"] == "natural":
        _sel_params["overwrite"] = True  # No actual overwrite is occuring.
        data_group_in, data_group_out = check_sel_params(
            ms_xds,
            _sel_params,
            default_data_group_in_name="base",
            default_data_group_out_name="imaging",
            default_data_group_out_modified={"weight_imaging": "WEIGHT"},
        )
        description = "Data group created for natural imaging weights with ."
        return _sel_params["data_group_out"]
    else:
        data_group_in, data_group_out = check_sel_params(
            ms_xds,
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

    uvw = ms_xds[data_group_out["uvw"]].values
    data_weight = ms_xds[data_group_out["weight"]].values

    # * (
    #     1 - ms_xds[data_group_out["flag"]].values
    # )
    data_weight[ms_xds[data_group_out["flag"]] == 1] = (
        np.nan
    )  # Set flagged data to NaN for weighting.
    freq_chan = ms_xds.frequency.values

    if data_weight.shape[3] == 2:
        data_weight = ((data_weight[..., 0] + data_weight[..., 1]) / 2)[..., np.newaxis]

    if data_weight.shape[3] == 4:
        data_weight = ((data_weight[..., 0] + data_weight[..., 3]) / 2)[..., np.newaxis]

    # Grid Weights
    n_uv = _grid_params["image_size_padded"]
    n_imag_chan = data_weight.shape[2]
    weight_density_grid = np.zeros((n_imag_chan, 1, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, 1), dtype=np.double)

    grid_imaging_weights(
        weight_density_grid, sum_weight, uvw, data_weight, freq_chan, _grid_params
    )

    # Calculate Briggs
    briggs_factors = calculate_briggs_params(
        weight_density_grid, sum_weight, _imaging_weights_params
    )  # 2 x chan x pol
    print("sum_weight", sum_weight)
    print("briggs_factors", briggs_factors)

    imaging_weights = degrid_imaging_weights(
        weight_density_grid, uvw, data_weight, briggs_factors, freq_chan, _grid_params
    )

    # Flag data
    flags = np.any(ms_xds[data_group_out["flag"]], axis=-1)  #
    data_weight[flags == 1] = np.nan

    ms_xds[data_group_out["weight_imaging"]] = xr.DataArray(
        imaging_weights[..., 0], dims=ms_xds[data_group_out["weight"]].dims[:-1]
    )

    data_group_out_name = _sel_params["data_group_out"]["data_group_out_name"]
    del data_group_out["data_group_out_name"]
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    ms_xds.data_groups[data_group_out_name] = data_group_out
    ms_xds.data_groups[data_group_out_name]["date"] = now.isoformat()
    ms_xds.data_groups[data_group_out_name]["description"] = description

    return ms_xds, data_group_out_name
