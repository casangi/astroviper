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
from astroviper.core.imaging.imaging_utils.standard_grid import (
    standard_imaging_weight_degrid_numpy_wrap,
    standard_grid_psf_numpy_wrap,
)

# from graphviper.parameter_checking.check_parms import check_sel_parms
from astroviper.utils.check_parms import check_parms, check_sel_parms
import copy


def make_imaging_weights(ms_xds, grid_parms, imaging_weights_parms, sel_parms):
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
    _grid_parms["oversampling"] = 0
    _grid_parms["support"] = 1
    _grid_parms["do_psf"] = True
    _grid_parms["complex_grid"] = False
    _grid_parms["do_imaging_weight"] = True

    cgk_1D = np.ones((1))

    uvw = ms_xds[data_group_out["uvw"]].values
    data_weight = ms_xds[data_group_out["weight"]].values
    freq_chan = ms_xds.frequency.values

    # Grid Weights
    weight_density_grid, sum_weight = standard_grid_psf_numpy_wrap(
        uvw, data_weight, freq_chan, cgk_1D, _grid_parms
    )

    # Calculate Briggs
    # print('weight_density_grid',weight_density_grid)
    briggs_factors = calculate_briggs_parms(
        weight_density_grid, sum_weight, _imaging_weights_parms
    )  # 2 x chan x pol

    imaging_weights = standard_imaging_weight_degrid_numpy_wrap(
        weight_density_grid, uvw, data_weight, briggs_factors, freq_chan, _grid_parms
    )

    ms_xds[data_group_out["weight_imaging"]] = xr.DataArray(
        imaging_weights, dims=ms_xds[data_group_out["weight"]].dims
    )

    return _sel_parms["data_group_out"]


def calculate_briggs_parms(grid_of_imaging_weights, sum_weight, imaging_weights_parms):
    if imaging_weights_parms["weighting"] == "briggs":
        robust = imaging_weights_parms["robust"]
        briggs_factors = np.ones((2,) + sum_weight.shape)

        squared_sum_weight = np.sum((grid_of_imaging_weights) ** 2, axis=(2, 3))
        briggs_factors[0, :, :] = (
            np.square(5.0 * 10.0 ** (-robust)) / (squared_sum_weight / sum_weight)
        )[None, None, :, :]
    elif imaging_weights_parms["weighting"] == "briggs_abs":
        robust = imaging_weights_parms["robust"]
        briggs_factors = np.ones((2,) + sum_weight.shape)
        briggs_factors[0, :, :] = briggs_factors[0, 0, 0, :, :] * np.square(robust)
        briggs_factors[1, :, :] = (
            briggs_factors[1, 0, 0, :, :]
            * 2.0
            * np.square(imaging_weights_parms["briggs_abs_noise"])
        )
    else:
        briggs_factors = np.zeros((2, 1, 1) + sum_weight.shape)
        briggs_factors[0, :, :] = np.ones((1, 1, 1) + sum_weight.shape)

    return briggs_factors
