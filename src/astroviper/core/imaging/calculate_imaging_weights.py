import numpy as np
import scipy

# import cngi._utils._constants as const
from scipy import constants
from numba import jit
import numba
import xarray as xr
from astroviper.core.imaging.check_imaging_parameters import (
    check_imaging_weights_parms,
    check_grid_parms,
)
from astroviper.core.imaging.imaging_weighting.grid_imaging_weights import (
    grid_imaging_weights,
    degrid_imaging_weights
)

from astroviper.core.imaging.imaging_weighting.briggs_weighting import calculate_briggs_params

# from graphviper.parameter_checking.check_parms import check_sel_parms
from astroviper.utils.check_parms import check_parms, check_sel_parms
import copy


def calculate_imaging_weights(ms_xds, grid_parms, imaging_weights_parms, sel_parms):
    _sel_parms = copy.deepcopy(sel_parms)
    _imaging_weights_parms = copy.deepcopy(imaging_weights_parms)
    assert check_imaging_weights_parms(
        _imaging_weights_parms
    ), "######### ERROR: imaging_weights_parms checking failed"

    if _imaging_weights_parms["weighting"] == "natural":
        _sel_parms["overwrite"] = True  # No actual overwrite is occuring.
        data_group_in, data_group_out = check_sel_parms(
            ms_xds,
            _sel_parms,
            default_data_group_out={"imaging": {"weight_imaging": "WEIGHT"}},
        )
        return _sel_parms["data_group_out"]
    else:
        data_group_in, data_group_out = check_sel_parms(
            ms_xds,
            _sel_parms,
            default_data_group_out={"imaging": {"weight_imaging": "WEIGHT_IMAGING"}},
        )

    # print(data_group_in, data_group_out)

    _grid_parms = copy.deepcopy(grid_parms)
    assert check_grid_parms(_grid_parms), "######### ERROR: grid_parms checking failed"
    _grid_parms["image_size_padded"] = _grid_parms[
        "image_size"
    ]  # do not need to pad since no fft

    uvw = ms_xds[data_group_out["uvw"]].values
    data_weight = ms_xds[data_group_out["weight"]].values * (
        1 - ms_xds[data_group_out["flag"]].values
    )
    freq_chan = ms_xds.frequency.values

    # Grid Weights
    weight_density_grid, sum_weight = grid_imaging_weights(
        uvw, data_weight, freq_chan, _grid_parms
    )

    # Calculate Briggs
    # print('weight_density_grid',weight_density_grid)
    briggs_factors = calculate_briggs_params(
        weight_density_grid, sum_weight, _imaging_weights_parms
    )  # 2 x chan x pol
    
    print("briggs_factors", briggs_factors.shape, briggs_factors)

    imaging_weights = degrid_imaging_weights(
        weight_density_grid, uvw, data_weight, briggs_factors, freq_chan, _grid_parms
    )

    ms_xds[data_group_out["weight_imaging"]] = xr.DataArray(
        imaging_weights, dims=ms_xds[data_group_out["weight"]].dims
    )

    return _sel_parms["data_group_out"]

