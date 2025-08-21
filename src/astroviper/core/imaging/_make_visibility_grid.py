import numpy as np
import scipy
from scipy import constants
from numba import jit
import numba
import xarray as xr
from astroviper.core._imaging._check_imaging_parameters import _check_grid_parms
from astroviper.core._imaging._imaging_utils._mosaic_grid import _mosaic_grid_jit
from astroviper.core._imaging._imaging_utils._standard_grid import _standard_grid_jit

# from graphviper.parameter_checking.check_parms import check_sel_parms
from astroviper.utils.check_parms import check_parms, check_sel_parms
import copy


def _make_visibility_grid(
    ms_xds, gcf_xds, img_xds, vis_sel_parms, img_sel_parms, grid_parms
):
    # Deep copy so that inputs are not modified
    _vis_sel_parms = copy.deepcopy(vis_sel_parms)
    _img_sel_parms = copy.deepcopy(img_sel_parms)
    _grid_parms = copy.deepcopy(grid_parms)

    _img_sel_parms["overwrite"] = True
    assert _check_grid_parms(_grid_parms), "######### ERROR: grid_parms checking failed"

    ms_data_group_in, ms_data_group_out = check_sel_parms(
        ms_xds, _vis_sel_parms, skip_data_group_out=True
    )
    img_data_group_in, img_data_group_out = check_sel_parms(
        img_xds,
        _img_sel_parms,
        default_data_group_out={
            "mosaic": {
                "visibility": "VISIBILITY",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
            }
        },
    )

    # print(ms_data_group_in, ms_data_group_out)

    weight_imaging = ms_xds[ms_data_group_in["weight_imaging"]].values
    n_chan = weight_imaging.shape[2]

    if _grid_parms["chan_mode"] == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight_imaging.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = _grid_parms["image_size_padded"]
    delta_lm = _grid_parms["cell_size"]
    oversampling = gcf_xds.attrs["oversampling"]

    _grid_parms["complex_grid"] = True
    if img_data_group_out["visibility"] not in img_xds:
        if _grid_parms["complex_grid"]:
            img_xds[img_data_group_out["visibility"]] = xr.DataArray(
                np.zeros(
                    (n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128
                ),
                dims=["frequency", "polarization", "u", "v"],
            )

        else:
            img_xds[img_data_group_out["visibility"]] = xr.DataArray(
                np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double),
                dims=["frequency", "polarization", "u", "v"],
            )
        img_xds[img_data_group_out["visibility_normalization"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol), dtype=np.double),
            dims=["frequency", "polarization"],
        )

    grid = img_xds[img_data_group_out["visibility"]].values
    sum_weight = img_xds[img_data_group_out["visibility_normalization"]].values

    vis_data = ms_xds[ms_data_group_in["correlated_data"]].values
    uvw = ms_xds[ms_data_group_in["uvw"]].values
    freq_chan = ms_xds.frequency.values
    imaging_weight = ms_xds[ms_data_group_in["weight_imaging"]].values

    do_psf = False

    # print(img_xds)
    cf_baseline_map = gcf_xds["CF_BASELINE_MAP"].values
    cf_chan_map = gcf_xds["CF_CHAN_MAP"].values
    cf_pol_map = gcf_xds["CF_POL_MAP"].values
    conv_kernel = gcf_xds["CONV_KERNEL"].values
    weight_support = gcf_xds["SUPPORT"].values
    phase_gradient = gcf_xds["PHASE_GRADIENT"].values

    _mosaic_grid_jit(
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


def _make_visibility_grid_single_field(
    ms_xds, cgk_1D, img_xds, vis_sel_parms, img_sel_parms, grid_parms
):
    # Deep copy so that inputs are not modified
    _vis_sel_parms = copy.deepcopy(vis_sel_parms)
    _img_sel_parms = copy.deepcopy(img_sel_parms)
    _grid_parms = copy.deepcopy(grid_parms)

    _img_sel_parms["overwrite"] = True
    assert _check_grid_parms(_grid_parms), "######### ERROR: grid_parms checking failed"

    ms_data_group_in, ms_data_group_out = check_sel_parms(
        ms_xds, _vis_sel_parms, skip_data_group_out=True
    )
    img_data_group_in, img_data_group_out = check_sel_parms(
        img_xds,
        _img_sel_parms,
        default_data_group_out={
            "mosaic": {
                "visibility": "VISIBILITY",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
            }
        },
    )

    # print(ms_data_group_in, ms_data_group_out)

    weight_imaging = ms_xds[ms_data_group_in["weight_imaging"]].values
    n_chan = weight_imaging.shape[2]

    if _grid_parms["chan_mode"] == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight_imaging.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = _grid_parms["image_size_padded"]
    delta_lm = _grid_parms["cell_size"]

    _grid_parms["complex_grid"] = True
    if img_data_group_out["visibility"] not in img_xds:
        if _grid_parms["complex_grid"]:
            img_xds[img_data_group_out["visibility"]] = xr.DataArray(
                np.zeros(
                    (n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128
                ),
                dims=["frequency", "polarization", "u", "v"],
            )

        else:
            img_xds[img_data_group_out["visibility"]] = xr.DataArray(
                np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double),
                dims=["frequency", "polarization", "u", "v"],
            )
        img_xds[img_data_group_out["visibility_normalization"]] = xr.DataArray(
            np.zeros((n_imag_chan, n_imag_pol), dtype=np.double),
            dims=["frequency", "polarization"],
        )

    grid = img_xds[img_data_group_out["visibility"]].values
    sum_weight = img_xds[img_data_group_out["visibility_normalization"]].values

    vis_data = ms_xds[ms_data_group_in["correlated_data"]].values
    uvw = ms_xds[ms_data_group_in["uvw"]].values
    freq_chan = ms_xds.frequency.values
    imaging_weight = ms_xds[ms_data_group_in["weight_imaging"]].values

    do_psf = False
    do_imaging_weight = False

    _standard_grid_jit(
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
