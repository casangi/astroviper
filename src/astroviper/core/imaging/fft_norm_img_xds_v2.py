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


def _get_oversampling_correcting_func(image_shape, oversampling):
    image_center = np.array(image_shape) // 2
    sincx = np.sinc(
        np.arange(-image_center[0], image_shape[0] - image_center[0])
        / (image_shape[0] * oversampling[0])
    )
    sincy = np.sinc(
        np.arange(-image_center[1], image_shape[1] - image_center[1])
        / (image_shape[1] * oversampling[1])
    )
    return np.dot(sincx[:, None], sincy[None, :])


def _ifft_remove_padding(raw_grid, image_size):
    """Memory-efficient IFFT + crop, processing one 2D (l,m) slice at a time.

    Instead of FFT-ing the full 5D padded array (which transiently allocates a
    second padded-size complex array), we process each (time, freq, pol) plane
    individually:
      - only one 2D complex padded plane is ever in memory alongside the output
      - the output is pre-allocated at the final (unpadded) size

    This trades a small amount of extra compute (Python loop overhead) for a
    large reduction in peak memory.
    """
    import time
    import toolviper.utils.logger as logger

    start = time.time()

    padded_h, padded_w = raw_grid.shape[-2], raw_grid.shape[-1]
    scale = padded_h * padded_w

    image_size = np.asarray(image_size)
    start_l = padded_h // 2 - image_size[0] // 2
    end_l = start_l + image_size[0]
    start_m = padded_w // 2 - image_size[1] // 2
    end_m = start_m + image_size[1]

    T, F, P = raw_grid.shape[0], raw_grid.shape[1], raw_grid.shape[2]
    output = np.empty((T, F, P, image_size[0], image_size[1]), dtype=np.float64)

    for t in range(T):
        for f in range(F):
            for p in range(P):
                # All temporaries live only for this iteration (one 2D padded plane).
                tmp = np.fft.fftshift(
                    np.fft.ifft2(np.fft.ifftshift(raw_grid[t, f, p]))
                ).real
                output[t, f, p] = tmp[start_l:end_l, start_m:end_m] * scale

    logger.debug("Time for ifft_uv_to_lm: " + str(time.time() - start))
    return output


def fft_norm_img_xds(img_xds, oversampling, grid_params, norm_params, sel_params):
    import toolviper.utils.logger as logger

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

            grid_var_name = data_group_in[data_variable]
            raw_grid = img_xds[grid_var_name].values
            # Free the large complex grid from img_xds before FFT so it can be
            # collected as soon as del raw_grid is called below.
            img_xds[grid_var_name] = xr.DataArray(np.empty(0))

            if (data_variable == "aperture") and ("APERTURE" in img_xds):
                # _ifft_remove_padding processes slice-by-slice to avoid a
                # second full-size padded array in memory.
                image = _ifft_remove_padding(raw_grid, _grid_params["image_size"]).real
                del raw_grid
                sum_weight_copy = copy.deepcopy(
                    img_xds[data_group_out["aperture_normalization"]].values
                )
                ##Don't mutate inputs, therefore do deep copy (https://docs.dask.org/en/latest/delayed-best-practices.html).
                sum_weight_copy[sum_weight_copy == 0] = 1
                image = np.sqrt(np.abs(image / sum_weight_copy[:, :, None, None]))
                if _norm_params["pb_limit"] > 0:
                    image[image < _norm_params["pb_limit"]] = np.nan  # 0.0
            else:
                # Slice-by-slice FFT + crop avoids holding the full padded
                # 5D FFT result in memory alongside raw_grid.
                image = _ifft_remove_padding(raw_grid, _grid_params["image_size"])
                del raw_grid
                sum_weight = img_xds[
                    data_group_out[sum_of_weight_pair[data_variable]]
                ].values
                primary_beam = img_xds[data_group_out["primary_beam"]].values
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
                    direction,
                    norm_params=_norm_params,
                )
            img_xds[data_group_out[fft_pair[data_variable]]] = xr.DataArray(
                image, dims=("time", "frequency", "polarization", "l", "m")
            )


def ifft_uv_to_lm(image):
    import time
    import toolviper.utils.logger as logger

    start = time.time()
    image_shape = image.shape
    image_sky = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(image, axes=(2, 3)), axes=(2, 3)), axes=(2, 3)
    ).real * (image_shape[-1] * image_shape[-2])
    logger.debug("Time for ifft_uv_to_lm: " + str(time.time() - start))
    return image_sky


def fft_lm_to_uv(image):
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(2, 3)), axes=(2, 3)), axes=(2, 3)
    ).real


def remove_padding(image_dask_array, image_size):
    import numpy as np

    image_size_padded = np.array(image_dask_array.shape[-2:])
    start_xy = image_size_padded // 2 - image_size // 2
    end_xy = start_xy + image_size

    image_dask_array = image_dask_array[
        ..., start_xy[0] : end_xy[0], start_xy[1] : end_xy[1]
    ]
    return image_dask_array


def normalize(
    image,
    sum_weight,
    primary_beam,
    oversampling,
    direction,
    norm_params,
):
    """
    PB normalization on the cubes.

    ps_correcting_image and oversampling_correcting_func are both cheap to
    compute (~0.3 s and ~0.5 s respectively) and are now generated on-the-fly
    inside this function rather than being passed as pre-allocated arrays.
    They are combined into a single 2D factor before touching the large image
    arrays so no extra (T, F, P, l, m) temporaries are created for them.

    direction : 'forward' | 'reverse'
    norm_type : 'flat_noise' | 'flat_sky' | 'none'
    """
    import numpy as np
    import copy
    import toolviper.utils.logger as logger
    import time
    from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel,
    )

    norm_type = norm_params["norm_type"]

    def normalize_image(
        image, sum_weights, normalizing_image, oversampling, correct_oversampling
    ):
        sum_weights_copy = copy.deepcopy(sum_weights)
        sum_weights_copy[sum_weights_copy == 0] = 1

        if correct_oversampling:
            image_size = tuple(image.shape[2:])

            start = time.time()
            oversampling_correcting_func = _get_oversampling_correcting_func(
                image_size, oversampling
            )
            logger.debug(
                "Time to get oversampling correcting func: " + str(time.time() - start)
            )

            start = time.time()
            _, ps_corr_image = create_prolate_spheroidal_kernel(
                100, 7, n_uv=list(image_size)
            )
            logger.debug("Calculate ps correcting image: " + str(time.time() - start))

            # Combine both 2D correction factors into a single 2D array before
            # broadcasting against the large (T, F, P, l, m) normalizing_image.
            # This avoids allocating an extra (T, F, P, l, m) temporary.
            combined_2d_correction = oversampling_correcting_func * ps_corr_image

            # Apply combined correction in-place on normalizing_image.
            # normalizing_image is already a local copy (primary_beam or pb^2),
            # so mutation is safe.
            normalizing_image *= combined_2d_correction

            normalized_image = (image / sum_weights_copy) / normalizing_image
        else:
            normalized_image = (image / sum_weights_copy) / normalizing_image
        return normalized_image

    if direction == "forward":
        correct_oversampling = True
        if norm_type == "flat_noise":
            # Divide the raw image by sqrt(.weight) so that the input to the
            # minor cycle represents the product of the sky and PB. The noise
            # is 'flat' across the region covered by each PB.
            # primary_beam is read-only; make a copy so in-place *= is safe.
            normalizing_image = primary_beam.copy()
            normalized_image = normalize_image(
                image,
                sum_weight[:, :, None, None],
                normalizing_image,
                oversampling,
                correct_oversampling,
            )
        elif norm_type == "flat_sky":
            # Divide the raw image by .weight so that the input to the minor
            # cycle represents only the sky. Noise is higher in the outer
            # regions of the primary beam where sensitivity is low.
            normalizing_image = primary_beam * primary_beam
            normalized_image = normalize_image(
                image,
                sum_weight[:, :, None, None],
                normalizing_image,
                oversampling,
                correct_oversampling,
            )
        elif norm_type == "none":
            pass

        if norm_params["pb_limit"] > 0:
            normalized_image[primary_beam < norm_params["pb_limit"]] = np.nan

        return normalized_image
    elif direction == "reverse":
        print("reverse operation not yet implemented not yet implemented")
