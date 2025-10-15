import numpy as np
import scipy

# import cngi._utils._constants as const
from scipy import constants
from numba import jit
import numba
import xarray as xr
from astroviper.core.imaging.check_imaging_parameters import (
    check_grid_params,
    check_gcf_params,
)
from astroviper.core.imaging.imaging_utils.standard_grid import (
    standard_imaging_weight_degrid_numpy_wrap,
    standard_grid_psf_numpy_wrap,
)

# from graphviper.parameter_checking.check_params import check_sel_params
from astroviper.utils.check_params import check_sel_params
import copy
import time


def make_gridding_convolution_function(
    gcf_xds, ms_xds, gcf_params, grid_params, sel_params
):
    start_0 = time.time()
    _gcf_params = copy.deepcopy(gcf_params)
    _grid_params = copy.deepcopy(grid_params)
    _sel_params = copy.deepcopy(sel_params)

    # print(sel_params)
    data_group_in, _ = check_sel_params(
        ms_xds, _sel_params, default_data_group_in_name="base"
    )

    _gcf_params["field_phase_dir"] = ms_xds.xr_ms.get_field_and_source_xds(
        data_group_in["data_group_in_name"]
    ).FIELD_PHASE_CENTER_DIRECTION.isel(field_name=0)

    # _gcf_params["basline_ant"] = np.array(
    #     [ms_xds.baseline_antenna1_id.values, ms_xds.baseline_antenna2_id.values]
    # ).T

    # MSv4 schema no longer has ids. This is a temporary fix and not very robust.
    baseline_ant1_id = np.where(
        ms_xds.antenna_xds.antenna_name == ms_xds.baseline_antenna1_name
    )[0]
    baseline_ant2_id = np.where(
        ms_xds.antenna_xds.antenna_name == ms_xds.baseline_antenna2_name
    )[0]
    _gcf_params["basline_ant"] = np.array([baseline_ant1_id, baseline_ant2_id]).T

    _gcf_params["freq_chan"] = ms_xds.frequency.values
    _gcf_params["pol"] = ms_xds.polarization.values

    assert check_gcf_params(_gcf_params), "######### ERROR: gcf_params checking failed"
    assert check_grid_params(
        _grid_params
    ), "######### ERROR: grid_params checking failed"

    _gcf_params["resize_conv_size"] = (_gcf_params["max_support"] + 1) * _gcf_params[
        "oversampling"
    ]

    # print('^^^^^', len(gcf_xds))
    # print('&&&&&&&&&&')
    # print(gcf_xds)
    # print('&&&&&&&&&&')
    # if len(gcf_xds) > 1:
    #     print(gcf_xds.baseline.size, len(_gcf_params["basline_ant"]))
    # else:
    #     print('No data vars')
    if (len(gcf_xds) < 1) or (gcf_xds.baseline.size != len(_gcf_params["basline_ant"])):
        gcf_xds = xr.Dataset()
        if _gcf_params["function"] == "airy":
            from .imaging_utils.make_pb_symmetric import airy_disk_rorder

            pb_func = airy_disk_rorder
        elif _gcf_params["function"] == "casa_airy":
            from .imaging_utils.make_pb_symmetric import casa_airy_disk_rorder

            pb_func = casa_airy_disk_rorder
        else:
            assert (
                False
            ), "######### ERROR: Only airy and casa_airy function has been implemented"

        n_unique_ant = len(_gcf_params["list_dish_diameters"])
        cf_baseline_map, pb_ant_pairs = create_cf_baseline_map(
            _gcf_params["unique_ant_indx"], _gcf_params["basline_ant"], n_unique_ant
        )
        cf_chan_map, pb_freq = create_cf_chan_map(
            _gcf_params["freq_chan"], _gcf_params["chan_tolerance_factor"]
        )

        cf_pol_map = np.zeros(
            (len(_gcf_params["pol"]),), dtype=int
        )  # create_cf_pol_map(), currently treating all pols the same
        pb_pol = np.array([0])

        _gcf_params["ipower"] = 1
        baseline_pb = make_baseline_patterns(
            pb_freq, pb_pol, pb_ant_pairs, pb_func, _gcf_params, _grid_params
        )

        _gcf_params["ipower"] = 2
        baseline_pb_sqrd = make_baseline_patterns(
            pb_freq, pb_pol, pb_ant_pairs, pb_func, _gcf_params, _grid_params
        )

        # print("%%% make_baseline_patterns ",time.time()-start_0)

        start_1 = time.time()
        conv_kernel = np.real(
            np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(baseline_pb, axes=(3, 4)), axes=(3, 4)),
                axes=(3, 4),
            )
        )
        conv_kernel_convolved = np.real(
            np.fft.fftshift(
                np.fft.fft2(
                    np.fft.ifftshift(baseline_pb_sqrd, axes=(3, 4)), axes=(3, 4)
                ),
                axes=(3, 4),
            )
        )

        # print("%%% fft ",time.time()-start_1)

        start_2 = time.time()
        (
            resized_conv_kernel,
            resized_conv_kernel_convolved,
            conv_support,
        ) = resize_and_calc_support(
            conv_kernel, conv_kernel_convolved, _gcf_params, _grid_params
        )

        # print("%%% resize ",time.time()-start_2)

        # gcf_xds = xr.Dataset()
        gcf_xds["SUPPORT"] = xr.DataArray(
            conv_support, dims=["conv_baseline", "conv_chan", "conv_pol", "xy"]
        )
        gcf_xds["CONV_KERNEL_CONVOLVED"] = xr.DataArray(
            resized_conv_kernel_convolved,
            dims=["conv_baseline", "conv_chan", "conv_pol", "u", "v"],
        )
        gcf_xds["CONV_KERNEL"] = xr.DataArray(
            resized_conv_kernel,
            dims=("conv_baseline", "conv_chan", "conv_pol", "u", "v"),
        )
        gcf_xds["PS_CORR_IMAGE"] = xr.DataArray(
            np.ones(_grid_params["image_size"]), dims=["l", "m"]
        )

        coords = {
            "u": np.arange(_gcf_params["resize_conv_size"][0]),
            "v": np.arange(_gcf_params["resize_conv_size"][1]),
            "xy": np.arange(2),
            "field_id": [0],
            "l": np.arange(_grid_params["image_size"][0]),
            "m": np.arange(_grid_params["image_size"][1]),
        }
        gcf_xds["CF_BASELINE_MAP"] = xr.DataArray(cf_baseline_map, dims=("baseline"))
        gcf_xds["CF_CHAN_MAP"] = xr.DataArray(cf_chan_map, dims=("chan"))
        gcf_xds["CF_POL_MAP"] = xr.DataArray(cf_pol_map, dims=("pol"))
        gcf_xds.attrs["cell_uv"] = 1 / (
            _grid_params["image_size_padded"]
            * _grid_params["cell_size"]
            * _gcf_params["oversampling"]
        )
        gcf_xds.attrs["oversampling"] = _gcf_params["oversampling"]

        gcf_xds.assign_coords(coords)

    start_3 = time.time()
    field_phase_dir = np.array(_gcf_params["field_phase_dir"].data)
    phase_gradient = make_phase_gradient(
        field_phase_dir[None, :], _gcf_params, _grid_params
    )

    gcf_xds["PHASE_GRADIENT"] = xr.DataArray(
        phase_gradient, dims=("field_id", "u", "v")
    )
    # print("%%% Phase gradient ",time.time()-start_3)

    # list_xarray_data_variables = [gcf_dataset['A_TERM'],gcf_dataset['WEIGHT_A_TERM'],gcf_dataset['A_SUPPORT'],gcf_dataset['WEIGHT_A_SUPPORT'],gcf_dataset['PHASE_GRADIENT']]
    # return _store(gcf_dataset,list_xarray_data_variables,_storage_params)

    return gcf_xds


def make_baseline_patterns(
    pb_freq, pb_pol, pb_ant_pairs, pb_func, gcf_params, grid_params
):
    import copy

    pb_grid_params = copy.deepcopy(grid_params)
    pb_grid_params["cell_size"] = grid_params["cell_size"] * gcf_params["oversampling"]
    pb_grid_params["image_size"] = pb_grid_params["image_size_padded"]
    pb_grid_params["image_center"] = pb_grid_params["image_size"] // 2

    # print("grid_params",grid_params)
    # print("pb_grid_params",pb_grid_params)
    import time

    start = time.time()
    patterns = pb_func(pb_freq, pb_pol, gcf_params, pb_grid_params)
    # print('@@@The core',time.time()-start)
    baseline_pattern = np.zeros(
        (
            len(pb_ant_pairs),
            len(pb_freq),
            len(pb_pol),
            grid_params["image_size_padded"][0],
            grid_params["image_size_padded"][1],
        ),
        dtype=np.double,
    )

    for ant_pair_indx, ant_pair in enumerate(pb_ant_pairs):
        for freq_indx in range(len(pb_freq)):
            baseline_pattern[ant_pair_indx, freq_indx, 0, :, :] = (
                patterns[ant_pair[0], freq_indx, 0, :, :]
                * patterns[ant_pair[1], freq_indx, 0, :, :]
            )

    return baseline_pattern  # , conv_support_array


def create_cf_baseline_map(unique_ant_indx, basline_ant, n_unique_ant):
    n_unique_ant_pairs = int((n_unique_ant**2 + n_unique_ant) / 2)

    pb_ant_pairs = np.zeros((n_unique_ant_pairs, 2), dtype=int)
    k = 0
    for i in range(n_unique_ant):
        for j in range(i, n_unique_ant):
            pb_ant_pairs[k, :] = [i, j]
            k = k + 1

    cf_baseline_map = np.zeros((basline_ant.shape[0],), dtype=int)
    basline_ant_unique_ant_indx = np.concatenate(
        (
            unique_ant_indx[basline_ant[:, 0]][:, None],
            unique_ant_indx[basline_ant[:, 1]][:, None],
        ),
        axis=1,
    )

    for k, ij in enumerate(pb_ant_pairs):
        cf_baseline_map[
            (basline_ant_unique_ant_indx[:, 0] == ij[0])
            & (basline_ant_unique_ant_indx[:, 1] == ij[1])
        ] = k

    return cf_baseline_map, pb_ant_pairs


def create_cf_chan_map(freq_chan, chan_tolerance_factor):
    n_chan = len(freq_chan)
    cf_chan_map = np.zeros((n_chan,), dtype=int)

    orig_width = (np.max(freq_chan) - np.min(freq_chan)) / len(freq_chan)

    tol = np.max(freq_chan) * chan_tolerance_factor
    n_pb_chan = int(np.floor((np.max(freq_chan) - np.min(freq_chan)) / tol) + 0.5)

    # Create PB's for each channel
    if n_pb_chan == 0:
        n_pb_chan = 1

    if n_pb_chan >= n_chan:
        cf_chan_map = np.arange(n_chan)
        pb_freq = freq_chan
        return cf_chan_map, pb_freq

    pb_delta_bandwdith = (np.max(freq_chan) - np.min(freq_chan)) / n_pb_chan
    pb_freq = (
        np.arange(n_pb_chan) * pb_delta_bandwdith
        + np.min(freq_chan)
        + pb_delta_bandwdith / 2
    )

    cf_chan_map = np.zeros((n_chan,), dtype=int)
    for i in range(n_chan):
        cf_chan_map[i], _ = find_nearest(pb_freq, freq_chan[i])

    return cf_chan_map, pb_freq


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def resize_and_calc_support(
    conv_kernel, conv_kernel_convolved, gcf_params, grid_params
):
    # print("$$"*10,conv_kernel.shape,conv_kernel_convolved.shape)
    import itertools

    conv_shape = conv_kernel.shape[0:3]
    conv_support = np.zeros(conv_shape + (2,), dtype=int)  # 2 is to enable x,y support

    resized_conv_size = tuple(gcf_params["resize_conv_size"])
    start_indx = (
        grid_params["image_size_padded"] // 2 - gcf_params["resize_conv_size"] // 2
    )
    end_indx = start_indx + gcf_params["resize_conv_size"]

    resized_conv_kernel = np.zeros(conv_shape + resized_conv_size, dtype=np.double)
    resized_conv_kernel_convolved = np.zeros(
        conv_shape + resized_conv_size, dtype=np.double
    )

    for idx in itertools.product(*[range(s) for s in conv_shape]):
        conv_support[idx] = calc_conv_size(
            conv_kernel_convolved[idx],
            grid_params["image_size_padded"],
            gcf_params["support_cut_level"],
            gcf_params["oversampling"],
            gcf_params["max_support"],
        )

        embed_conv_size = (conv_support[idx] + 1) * gcf_params["oversampling"]
        embed_start_indx = gcf_params["resize_conv_size"] // 2 - embed_conv_size // 2
        embed_end_indx = embed_start_indx + embed_conv_size

        resized_conv_kernel[idx] = conv_kernel[
            idx[0],
            idx[1],
            idx[2],
            start_indx[0] : end_indx[0],
            start_indx[1] : end_indx[1],
        ]
        normalize_factor = np.real(
            np.sum(
                resized_conv_kernel[
                    idx[0],
                    idx[1],
                    idx[2],
                    embed_start_indx[0] : embed_end_indx[0],
                    embed_start_indx[1] : embed_end_indx[1],
                ]
            )
            / (gcf_params["oversampling"][0] * gcf_params["oversampling"][1])
        )
        resized_conv_kernel[idx] = resized_conv_kernel[idx] / normalize_factor

        resized_conv_kernel_convolved[idx] = conv_kernel_convolved[
            idx[0],
            idx[1],
            idx[2],
            start_indx[0] : end_indx[0],
            start_indx[1] : end_indx[1],
        ]
        normalize_factor = np.real(
            np.sum(
                resized_conv_kernel_convolved[
                    idx[0],
                    idx[1],
                    idx[2],
                    embed_start_indx[0] : embed_end_indx[0],
                    embed_start_indx[1] : embed_end_indx[1],
                ]
            )
            / (gcf_params["oversampling"][0] * gcf_params["oversampling"][1])
        )
        resized_conv_kernel_convolved[idx] = (
            resized_conv_kernel_convolved[idx] / normalize_factor
        )

    return resized_conv_kernel, resized_conv_kernel_convolved, conv_support


def calc_conv_size(sub_a_term, imsize, support_cut_level, oversampling, max_support):
    abs_sub_a_term = np.abs(sub_a_term)

    min_amplitude = np.min(abs_sub_a_term)
    max_indx = np.argmax(abs_sub_a_term)
    max_indx = np.unravel_index(max_indx, np.abs(sub_a_term).shape)
    max_amplitude = abs_sub_a_term[max_indx]
    cut_level_amplitude = support_cut_level * max_amplitude

    assert (
        min_amplitude < cut_level_amplitude
    ), "######### ERROR: support_cut_level too small or imsize too small."

    # x axis support
    indx_x = imsize[0] // 2
    indx_y = imsize[1] // 2
    while sub_a_term[indx_x, indx_y] > cut_level_amplitude:
        indx_x = indx_x + 1
        assert (
            indx_x < imsize[0]
        ), "######### ERROR: support_cut_level too small or imsize too small."
    approx_conv_size_x = indx_x - imsize[0] // 2
    support_x = (int(0.5 + approx_conv_size_x / oversampling[0]) + 1) * 2 + 1
    # support_x = int((approx_conv_size_x/oversampling[0])-1)
    # support_x = support_x if (support_x % 2) else support_x+1 #Support must be odd, to ensure symmetry

    # y axis support
    indx_x = imsize[0] // 2
    indx_y = imsize[1] // 2
    while sub_a_term[indx_x, indx_y] > cut_level_amplitude:
        indx_y = indx_y + 1
        assert (
            indx_y < imsize[1]
        ), "######### ERROR: support_cut_level too small or imsize too small."
    approx_conv_size_y = indx_y - imsize[1] // 2
    support_y = (int(0.5 + approx_conv_size_y / oversampling[1]) + 1) * 2 + 1
    # approx_conv_size_y = (indx_y-imsize[1]//2)*2
    # support_y = ((approx_conv_size_y/oversampling[1])-1).astype(int)
    # support_y = support_y if (support_y % 2) else support_y+1 #Support must be odd, to ensure symmetry

    assert support_x < max_support[0], (
        "######### ERROR: support_cut_level too small or imsize too small."
        + str(support_x)
        + ",*,"
        + str(max_support[0])
    )
    assert support_y < max_support[1], (
        "######### ERROR: support_cut_level too small or imsize too small."
        + str(support_y)
        + ",*,"
        + str(max_support[1])
    )

    # print('approx_conv_size_x,approx_conv_size_y',approx_conv_size_x,approx_conv_size_y,support_x,support_y,max_support)
    # print('support_x, support_y',support_x, support_y)
    if support_x > support_y:
        support_y = support_x
    else:
        support_x = support_y
    return [support_x, support_y]


def make_phase_gradient(field_phase_dir, gcf_params, grid_params):
    from astropy.wcs import WCS
    import math

    rad_to_deg = 180 / np.pi

    # print(' make_phase_gradient ',field_phase_dir,gcf_params,grid_params)

    phase_center = gcf_params["phase_direction"].values
    w = WCS(naxis=2)
    w.wcs.crpix = grid_params["image_size_padded"] // 2
    w.wcs.cdelt = grid_params["cell_size"] * rad_to_deg
    # w.wcs.cdelt = [grid_params['cell_size'][0]*rad_to_deg, grid_params['cell_size'][1]*rad_to_deg]
    w.wcs.crval = np.array(phase_center) * rad_to_deg
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    # print('field_phase_dir ',field_phase_dir,'phase_center',phase_center)
    # print(w.all_world2pix(field_phase_dir*rad_to_deg, 1))
    pix_dist = (
        np.array(w.all_world2pix(field_phase_dir * rad_to_deg, 1))
        - grid_params["image_size_padded"] // 2
    )
    pix = (
        -(pix_dist)
        * 2
        * np.pi
        / (grid_params["image_size_padded"] * gcf_params["oversampling"])
    )
    # print('pix_dist',pix_dist, pix, gcf_params['resize_conv_size'])
    # print('%%%%%%%%%%%%%%%%')

    image_size = gcf_params["resize_conv_size"]
    center_indx = image_size // 2
    x = np.arange(-center_indx[0], image_size[0] - center_indx[0])
    y = np.arange(-center_indx[1], image_size[1] - center_indx[1])
    # y_grid, x_grid = np.meshgrid(y,x)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    phase_gradient = np.moveaxis(
        np.exp(1j * (x_grid[:, :, None] * pix[:, 0] + y_grid[:, :, None] * pix[:, 1])),
        2,
        0,
    )
    return phase_gradient
