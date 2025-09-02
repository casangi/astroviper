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
from astroviper.core.imaging.imaging_utils.standard_grid import (
    standard_imaging_weight_degrid_numpy_wrap,
    standard_grid_psf_numpy_wrap,
)

from astroviper.utils.check_params import check_params, check_sel_params
import copy


def make_imaging_weights(ms_xds, grid_params, imaging_weights_params, sel_params):
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
        data_group_out["description"] = description
        ms_xds.attrs["data_groups"][
            data_group_out["data_group_out_name"]
        ] = data_group_out
        return data_group_out
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

    # print(data_group_in, data_group_out)

    _grid_params = copy.deepcopy(grid_params)
    assert check_grid_params(
        _grid_params
    ), "######### ERROR: grid_params checking failed"
    _grid_params["image_size_padded"] = _grid_params[
        "image_size"
    ]  # do not need to pad since no fft
    _grid_params["oversampling"] = 0
    _grid_params["support"] = 1
    _grid_params["do_psf"] = True
    _grid_params["complex_grid"] = False
    _grid_params["do_imaging_weight"] = True

    cgk_1D = np.ones((1))

    uvw = ms_xds[data_group_out["uvw"]].values
    data_weight = ms_xds[data_group_out["weight"]].values * (
        1 - ms_xds[data_group_out["flag"]].values
    )
    freq_chan = ms_xds.frequency.values

    # Grid Weights
    weight_density_grid, sum_weight = standard_grid_psf_numpy_wrap(
        uvw, data_weight, freq_chan, cgk_1D, _grid_params
    )

    # Calculate Briggs
    # print('weight_density_grid',weight_density_grid)
    briggs_factors = calculate_briggs_params(
        weight_density_grid, sum_weight, _imaging_weights_params
    )  # 2 x chan x pol

    imaging_weights = standard_imaging_weight_degrid_numpy_wrap(
        weight_density_grid, uvw, data_weight, briggs_factors, freq_chan, _grid_params
    )

    ms_xds[data_group_out["weight_imaging"]] = xr.DataArray(
        imaging_weights, dims=ms_xds[data_group_out["weight"]].dims
    )

    # print("weight data_group_out", data_group_out)

    ms_xds.attrs["data_groups"][data_group_out["data_group_out_name"]] = data_group_out

    return data_group_out


def calculate_briggs_params(
    grid_of_imaging_weights, sum_weight, imaging_weights_params
):
    if imaging_weights_params["weighting"] == "briggs":
        robust = imaging_weights_params["robust"]
        briggs_factors = np.ones((2,) + sum_weight.shape)

        squared_sum_weight = np.sum((grid_of_imaging_weights) ** 2, axis=(2, 3))

        # print("squared_sum_weight", squared_sum_weight.shape, squared_sum_weight)
        # print("sum_weight", sum_weight.shape, sum_weight)
        briggs_factors[0, :, :] = (
            np.square(5.0 * 10.0 ** (-robust)) / (squared_sum_weight / sum_weight)
        )[None, None, :, :]
    elif imaging_weights_params["weighting"] == "briggs_abs":
        robust = imaging_weights_params["robust"]
        briggs_factors = np.ones((2,) + sum_weight.shape)
        briggs_factors[0, :, :] = briggs_factors[0, :, :] * np.square(robust)
        briggs_factors[1, :, :] = (
            briggs_factors[1, :, :]
            * 2.0
            * np.square(imaging_weights_params["briggs_abs_noise"])
        )
    else:
        briggs_factors = np.zeros((2, 1, 1) + sum_weight.shape)
        briggs_factors[0, :, :] = np.ones((1, 1, 1) + sum_weight.shape)

    return briggs_factors
