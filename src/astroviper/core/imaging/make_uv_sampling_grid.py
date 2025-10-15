import numpy as np
import scipy
from scipy import constants
from numba import jit
import numba
import xarray as xr
from astroviper.core.imaging.check_imaging_parameters import check_grid_params
from astroviper.core.imaging.imaging_utils.mosaic_grid import mosaic_grid_jit
from astroviper.core.imaging.imaging_utils.standard_grid import standard_grid_jit

# from graphviper.parameter_checking.check_params import check_sel_params
from astroviper.utils.check_params import check_params, check_sel_params
import copy


def make_uv_sampling_grid(
    ms_xds, gcf_xds, img_xds, vis_sel_params, img_sel_params, grid_params
):
    # Deep copy so that inputs are not modified
    _vis_sel_params = copy.deepcopy(vis_sel_params)
    _img_sel_params = copy.deepcopy(img_sel_params)
    _grid_params = copy.deepcopy(grid_params)

    _img_sel_params["overwrite"] = True
    assert check_grid_params(
        _grid_params
    ), "######### ERROR: grid_params checking failed"

    ms_data_group_in, ms_data_group_out = check_sel_params(
        ms_xds, _vis_sel_params, default_data_group_in_name="base"
    )
    img_data_group_in, img_data_group_out = check_sel_params(
        img_xds,
        _img_sel_params,
        default_data_group_in_name="mosaic",
        default_data_group_out_name="mosaic",
        default_data_group_out_modified={
            "uv_sampling": "UV_SAMPLING",
            "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
        },
    )

    # print(ms_data_group_in, ms_data_group_out)

    weight_imaging = ms_xds[ms_data_group_in["weight_imaging"]].values
    n_chan = weight_imaging.shape[2]

    if _grid_params["chan_mode"] == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight_imaging.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = _grid_params["image_size_padded"]
    delta_lm = _grid_params["cell_size"]
    oversampling = gcf_xds.attrs["oversampling"]

    _grid_params["complex_grid"] = True
    if img_data_group_out["uv_sampling"] not in img_xds:
        img_xds[img_data_group_out["uv_sampling"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128),
            dims=["frequency", "polarization", "u", "v"],
        )
        img_xds[img_data_group_out["uv_sampling_normalization"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol), dtype=np.double),
            dims=["frequency", "polarization"],
        )

    grid = img_xds[img_data_group_out["uv_sampling"]].values
    sum_weight = img_xds[img_data_group_out["uv_sampling_normalization"]].values

    vis_data = vis_data = np.zeros((1, 1, 1, 1), dtype=bool)
    uvw = ms_xds[ms_data_group_in["uvw"]].values
    freq_chan = ms_xds.frequency.values
    imaging_weight = ms_xds[ms_data_group_in["weight_imaging"]].values

    do_psf = True

    # print(img_xds)
    cf_baseline_map = gcf_xds["CF_BASELINE_MAP"].values
    cf_chan_map = gcf_xds["CF_CHAN_MAP"].values
    cf_pol_map = gcf_xds["CF_POL_MAP"].values
    conv_kernel = gcf_xds["CONV_KERNEL"].values
    weight_support = gcf_xds["SUPPORT"].values
    phase_gradient = gcf_xds["PHASE_GRADIENT"].values

    mosaic_grid_jit(
        grid,
        sum_weight,
        do_psf,
        vis_data,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        cf_baseline_map,
        cf_chan_map,
        cf_pol_map,
        imaging_weight,
        conv_kernel,
        n_uv,
        delta_lm,
        weight_support,
        oversampling,
        phase_gradient,
    )

    img_xds.attrs["data_groups"][
        img_data_group_out["data_group_out_name"]
    ] = img_data_group_out


def make_uv_sampling_grid_single_field(
    ms_xds, cgk_1D, img_xds, vis_sel_params, img_sel_params, grid_params
):
    # Deep copy so that inputs are not modified
    _vis_sel_params = copy.deepcopy(vis_sel_params)
    _img_sel_params = copy.deepcopy(img_sel_params)
    _grid_params = copy.deepcopy(grid_params)

    _img_sel_params["overwrite"] = True
    assert check_grid_params(
        _grid_params
    ), "######### ERROR: grid_params checking failed"

    ms_data_group_in, ms_data_group_out = check_sel_params(
        ms_xds, _vis_sel_params, skip_data_group_out=True
    )
    img_data_group_in, img_data_group_out = check_sel_params(
        img_xds,
        _img_sel_params,
        default_data_group_out={
            "mosaic": {
                "uv_sampling": "UV_SAMPLING",
                "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
            }
        },
    )

    # print(ms_data_group_in, ms_data_group_out)

    weight_imaging = ms_xds[ms_data_group_in["weight_imaging"]].values
    n_chan = weight_imaging.shape[2]

    if _grid_params["chan_mode"] == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight_imaging.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = _grid_params["image_size_padded"]
    delta_lm = _grid_params["cell_size"]

    _grid_params["complex_grid"] = True
    if img_data_group_out["uv_sampling"] not in img_xds:
        img_xds[img_data_group_out["uv_sampling"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128),
            dims=["frequency", "polarization", "u", "v"],
        )
        img_xds[img_data_group_out["uv_sampling_normalization"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol), dtype=np.double),
            dims=["frequency", "polarization"],
        )

    grid = img_xds[img_data_group_out["uv_sampling"]].values
    sum_weight = img_xds[img_data_group_out["uv_sampling_normalization"]].values

    vis_data = vis_data = np.zeros((1, 1, 1, 1), dtype=bool)
    uvw = ms_xds[ms_data_group_in["uvw"]].values
    freq_chan = ms_xds.frequency.values
    imaging_weight = ms_xds[ms_data_group_in["weight_imaging"]].values

    do_psf = True
    do_imaging_weight = False

    standard_grid_jit(
        grid,
        sum_weight,
        do_psf,
        do_imaging_weight,
        vis_data,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        imaging_weight,
        cgk_1D,
        n_uv,
        delta_lm,
        support=7,
        oversampling=100,
    )
