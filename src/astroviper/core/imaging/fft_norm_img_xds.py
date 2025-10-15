import numpy as np
import scipy
from scipy import constants
from numba import jit
import numba
import xarray as xr

# from graphviper.parameter_checking.check_params import check_sel_params
from astroviper.utils.check_params import check_params, check_sel_params
from astroviper.core.imaging.check_imaging_parameters import (
    check_grid_params,
    check_norm_params,
)
import copy


def fft_norm_img_xds(img_xds, gcf_xds, grid_params, norm_params, sel_params):
    _sel_params = copy.deepcopy(sel_params)
    _grid_params = copy.deepcopy(grid_params)
    assert check_grid_params(
        _grid_params
    ), "######### ERROR: grid_params checking failed"

    _norm_params = copy.deepcopy(norm_params)
    assert check_norm_params(
        _norm_params
    ), "######### ERROR: norm_params checking failed"

    if "PRIMARY_BEAM" in img_xds:
        data_group_in, data_group_out = check_sel_params(
            img_xds,
            _sel_params,
            default_data_group_out_name="imaging",
            default_data_group_out_modified={
                "point_spread_function": "POINT_SPREAD_FUNCTION",
                "sky": "SKY",
            },
        )
        data_group_out["primary_beam"] = "PRIMARY_BEAM"
    else:
        data_group_in, data_group_out = check_sel_params(
            img_xds,
            _sel_params,
            default_data_group_out_name="imaging",
            default_data_group_out_modified={
                "primary_beam": "PRIMARY_BEAM",
                "point_spread_function": "POINT_SPREAD_FUNCTION",
                "sky": "SKY",
            },
        )
    fft_pair = {
        "aperture": "primary_beam",
        "uv_sampling": "point_spread_function",
        "visibility": "sky",
    }
    sum_of_weight_pair = {
        "aperture": "aperture_grid_normalization",
        "uv_sampling": "uv_sampling_normalization",
        "visibility": "visibility_normalization",
    }

    # print(data_group_in, data_group_out)

    for data_variable in ["aperture", "uv_sampling", "visibility"]:
        if data_variable in data_group_in.keys():

            if (data_variable == "aperture") and ("APERTURE" in img_xds):
                # print('1.',data_group_out["aperture_grid_sum_weight"],img_xds[data_group_in[data_variable]].shape,_grid_params['image_size'])
                image = remove_padding(
                    ifft_uv_to_lm(img_xds[data_group_in[data_variable]].values),
                    _grid_params["image_size"],
                ).real
                sum_weight_copy = copy.deepcopy(
                    img_xds[data_group_out["aperture_normalization"]].values
                )
                # print("%%%%%%%%%%%%%Sum of weight", sum_weight_copy,data_group_out[sum_of_weight_pair[data_variable]])
                ##Don't mutate inputs, therefore do deep copy (https://docs.dask.org/en/latest/delayed-best-practices.html).
                sum_weight_copy[sum_weight_copy == 0] = 1
                image = np.sqrt(np.abs(image / sum_weight_copy[:, :, None, None]))
                if _norm_params["pb_limit"] > 0:
                    image[image < _norm_params["pb_limit"]] = np.nan  # 0.0
            else:
                image = remove_padding(
                    ifft_uv_to_lm(img_xds[data_group_in[data_variable]].values),
                    _grid_params["image_size"],
                )
                sum_weight = img_xds[
                    data_group_out[sum_of_weight_pair[data_variable]]
                ].values
                # print("%%%%%%%%%%%%%Sum of weight", sum_weight,data_group_out[sum_of_weight_pair[data_variable]])
                primary_beam = img_xds[data_group_out["primary_beam"]].values
                oversampling = gcf_xds.oversampling
                ps_correcting_image = gcf_xds.PS_CORR_IMAGE.values
                direction = "forward"

                if data_variable == "uv_sampling":
                    norm_params["norm_type"] = "flat_noise"
                if data_variable == "visibility":
                    norm_params["norm_type"] = "flat_sky"
                image = normalize(
                    image,
                    sum_weight,
                    primary_beam,
                    oversampling,
                    ps_correcting_image,
                    direction,
                    norm_params=_norm_params,
                )
            img_xds[data_group_out[fft_pair[data_variable]]] = xr.DataArray(
                image, dims=("frequency", "polarization", "l", "m")
            )


def ifft_uv_to_lm(image):
    image_shape = image.shape
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(image, axes=(2, 3)), axes=(2, 3)), axes=(2, 3)
    ).real * (image_shape[-1] * image_shape[-2])

    # dafft.fftshift(dafft.ifft2(dafft.ifftshift(grids_and_sum_weights[0], axes=(0, 1)), axes=(0, 1)), axes=(0, 1))


def fft_lm_to_uv(image):
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(2, 3)), axes=(2, 3)), axes=(2, 3)
    ).real


def remove_padding(image_dask_array, image_size):
    # Check that image_size < image_size
    # Check parameters

    import numpy as np

    image_size_padded = np.array(image_dask_array.shape[-2:])
    start_xy = image_size_padded // 2 - image_size // 2
    end_xy = start_xy + image_size

    # print("start end",image_size_padded,start_xy,end_xy)

    image_dask_array = image_dask_array[
        ..., start_xy[0] : end_xy[0], start_xy[1] : end_xy[1]
    ]
    return image_dask_array


def normalize(
    image,
    sum_weight,
    primary_beam,
    oversampling,
    ps_correcting_image,
    direction,
    norm_params,
):
    """
    PB normalization on the cubes

    direction : 'forward''reverse'
    norm_type : 'flatnoise','flatsky','common','pbsquare'

    Multiply and/or divide by PB models, accounting for masks/regions.

    #See https://casa.nrao.edu/casadocs/casa-5.6.0/imaging/synthesis-imaging/data-weighting Normalizationm Steps
    #Standard gridder (only ps term) devide by sum of weight and ps correcting image.
    #https://library.nrao.edu/public/memos/evla/EVLAM_198.pdf

    #

    """
    import dask.array as da
    import numpy as np
    import copy

    norm_type = norm_params["norm_type"]

    def normalize_image(
        image, sum_weights, normalizing_image, oversampling, correct_oversampling
    ):
        sum_weights_copy = copy.deepcopy(
            sum_weights
        )  ##Don't mutate inputs, therefore do deep copy (https://docs.dask.org/en/latest/delayed-best-practices.html).
        sum_weights_copy[sum_weights_copy == 0] = 1

        if correct_oversampling:
            image_size = np.array(image.shape[2:])
            image_center = image_size // 2
            sincx = np.sinc(
                np.arange(-image_center[0], image_size[0] - image_center[0])
                / (image_size[0] * oversampling[0])
            )
            sincy = np.sinc(
                np.arange(-image_center[1], image_size[1] - image_center[1])
                / (image_size[1] * oversampling[1])
            )

            oversampling_correcting_func = np.dot(
                sincx[:, None], sincy[None, :]
            )  # Last section for sinc correcting function https://library.nrao.edu/public/memos/evla/EVLAM_198.pdf

            # print(image.shape,sum_weights_copy.shape,oversampling_correcting_func.shape,oversampling_correcting_func[None,None,:,:].shape,normalizing_image.shape)

            normalized_image = (image / sum_weights_copy) / (
                oversampling_correcting_func[None, None, :, :] * normalizing_image
            )

            # print(sum_weights_copy,oversampling_correcting_func[500,360,None,None])
            # normalized_image = (image / sum_weights_copy ) / (oversampling_correcting_func[:,:,None,None])
        else:
            normalized_image = (image / sum_weights_copy) / normalizing_image
        return normalized_image

    if direction == "forward":
        correct_oversampling = True
        if norm_type == "flat_noise":
            # Divide the raw image by sqrt(.weight) so that the input to the minor cycle represents the product of the sky and PB. The noise is 'flat' across the region covered by each PB.
            normalizing_image = ps_correcting_image[None, None, :, :] * primary_beam
            normalized_image = normalize_image(
                image,
                sum_weight[:, :, None, None],
                normalizing_image,
                oversampling,
                correct_oversampling,
            )
        elif norm_type == "flat_sky":
            #  Divide the raw image by .weight so that the input to the minor cycle represents only the sky. The noise is higher in the outer regions of the primary beam where the sensitivity is low.
            normalizing_image = (
                ps_correcting_image[None, None, :, :] * primary_beam * primary_beam
            )

            # print(sel_params['data_group_in']['weight_pb'])
            # print('$%$%',img_dataset[sel_params['data_group_in']['weight_pb']].data.compute())

            normalized_image = normalize_image(
                image,
                sum_weight[:, :, None, None],
                normalizing_image,
                oversampling,
                correct_oversampling,
            )

            # print(normalized_image.compute())
        elif norm_type == "none":
            pass
            # print('in normalize none ')
            # No normalization after gridding and FFT. The minor cycle sees the sky times pb square
        #            normalizing_image = ps_correcting_image[None, None, :, :]
        #            normalized_image = da.map_blocks(
        #                normalize_image,
        #                image,
        #                sum_weight[:, :, None, None],
        #                normalizing_image,
        #                oversampling,
        #                correct_oversampling,
        #            )
        # normalized_image = image

        if norm_params["pb_limit"] > 0:
            normalized_image[primary_beam < norm_params["pb_limit"]] = np.nan  # 0.0

        if norm_params["single_precision"]:
            normalized_image = (normalized_image.astype(np.float32)).astype(np.float64)

        return normalized_image
    elif direction == "reverse":
        print("reverse operation not yet implemented not yet implemented")
