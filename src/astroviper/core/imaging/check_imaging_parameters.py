import numpy as np

# from graphviper.parameter_checking.check_params import check_params
from astroviper.utils.check_params import check_params


def check_grid_params(grid_params):
    import numbers

    params_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    if not (
        check_params(
            grid_params,
            "image_size",
            [list],
            list_acceptable_data_types=[int],
            list_len=2,
        )
    ):
        params_passed = False
    if not (
        check_params(
            grid_params,
            "image_center",
            [list],
            list_acceptable_data_types=[int],
            list_len=2,
            default=np.array(grid_params["image_size"]) // 2,
        )
    ):
        params_passed = False
    if not (
        check_params(
            grid_params,
            "cell_size",
            [list],
            list_acceptable_data_types=[numbers.Number],
            list_len=2,
        )
    ):
        params_passed = False
    if not (
        check_params(
            grid_params,
            "fft_padding",
            [numbers.Number],
            default=1.2,
            acceptable_range=[1, 10],
        )
    ):
        params_passed = False
    if not (
        check_params(
            grid_params,
            "chan_mode",
            [str],
            acceptable_data=["cube", "continuum"],
            default="cube",
        )
    ):
        params_passed = False

    if params_passed == True:
        grid_params["image_size"] = np.array(grid_params["image_size"]).astype(int)
        grid_params["image_size_padded"] = (
            grid_params["fft_padding"] * grid_params["image_size"]
        ).astype(int)
        grid_params["image_center"] = np.array(grid_params["image_center"])
        # grid_params['cell_size'] = arc_sec_to_rad * np.array(grid_params['cell_size'])
        # grid_params['cell_size'][0] = -grid_params['cell_size'][0]

    return params_passed


##################### Function Specific Parms #####################
def check_gcf_params(gcf_params):
    import numbers

    params_passed = True

    if not (
        check_params(
            gcf_params,
            "function",
            [str],
            acceptable_data=["casa_airy", "airy"],
            default="casa_airy",
        )
    ):
        params_passed = False
    if not (
        check_params(
            gcf_params,
            "freq_chan",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        params_passed = False
    if not (
        check_params(
            gcf_params,
            "list_dish_diameters",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        params_passed = False
    if not (
        check_params(
            gcf_params,
            "list_blockage_diameters",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        params_passed = False
    if not (
        check_params(
            gcf_params,
            "unique_ant_indx",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        params_passed = False

    if not (
        check_params(
            gcf_params,
            "pol",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number, str],
            list_len=-1,
        )
    ):
        params_passed = False
    if not (
        check_params(
            gcf_params, "chan_tolerance_factor", [numbers.Number], default=0.005
        )
    ):
        params_passed = False
    if not (
        check_params(
            gcf_params,
            "oversampling",
            [list, np.array],
            list_acceptable_data_types=[int],
            list_len=2,
            default=[10, 10],
        )
    ):
        params_passed = False
    if not (
        check_params(
            gcf_params,
            "max_support",
            [list, np.array],
            list_acceptable_data_types=[int],
            list_len=2,
            default=[15, 15],
        )
    ):
        params_passed = False
    # if not(check_params(gcf_params, 'image_phase_center', [list,np.array], list_acceptable_data_types=[numbers.Number], list_len=2)): params_passed = False
    if not (
        check_params(
            gcf_params, "support_cut_level", [numbers.Number], default=2.5 * 10**-2
        )
    ):
        params_passed = False
    if not (check_params(gcf_params, "a_chan_num_chunk", [int], default=3)):
        params_passed = False

    if gcf_params["function"] == "airy" or gcf_params["function"] == "casa_airy":
        if not (
            check_params(
                gcf_params,
                "list_dish_diameters",
                [list, np.array],
                list_acceptable_data_types=[numbers.Number],
                list_len=-1,
            )
        ):
            params_passed = False
        if not (
            check_params(
                gcf_params,
                "list_blockage_diameters",
                [list, np.array],
                list_acceptable_data_types=[numbers.Number],
                list_len=-1,
            )
        ):
            params_passed = False

        if len(gcf_params["list_dish_diameters"]) != len(
            gcf_params["list_blockage_diameters"]
        ):
            print(
                "######### ERROR:Parameter ",
                "list_dish_diameters and list_blockage_diameters must be the same length.",
            )
            params_passed = False

    if params_passed == True:
        gcf_params["oversampling"] = np.array(gcf_params["oversampling"]).astype(int)
        gcf_params["max_support"] = np.array(gcf_params["max_support"]).astype(int)
        # gcf_params['image_phase_center'] =  np.array(gcf_params['image_phase_center'])
        gcf_params["freq_chan"] = np.array(gcf_params["freq_chan"])
        gcf_params["list_dish_diameters"] = np.array(gcf_params["list_dish_diameters"])
        gcf_params["list_blockage_diameters"] = np.array(
            gcf_params["list_blockage_diameters"]
        )
        gcf_params["unique_ant_indx"] = np.array(gcf_params["unique_ant_indx"])
        # gcf_params['basline_ant'] =  np.array(gcf_params['basline_ant'])
        gcf_params["pol"] = np.array(gcf_params["pol"])

    return params_passed


def check_mosaic_pb_params(pb_mosaic_params):
    params_passed = True

    if not (check_params(pb_mosaic_params, "pb_name", [str], default="PB")):
        params_passed = False

    if not (check_params(pb_mosaic_params, "weight_name", [str], default="WEIGHT_PB")):
        params_passed = False

    return params_passed


def check_rotation_params(rotation_params):
    import numbers

    params_passed = True

    if not (
        check_params(
            rotation_params,
            "new_phase_center",
            [list],
            list_acceptable_data_types=[numbers.Number],
            list_len=2,
        )
    ):
        params_passed = False

    if not (
        check_params(
            rotation_params, "common_tangent_reprojection", [bool], default=True
        )
    ):
        params_passed = False

    if not (check_params(rotation_params, "single_precision", [bool], default=True)):
        params_passed = False

    return params_passed


def check_norm_params(norm_params):
    import numbers

    params_passed = True

    if not (
        check_params(
            norm_params,
            "norm_type",
            [str],
            default="flat_sky",
            acceptable_data=["flat_noise", "flat_sky", "none"],
        )
    ):
        params_passed = False

    if not (check_params(norm_params, "single_precision", [bool], default=True)):
        params_passed = False

    if not (check_params(norm_params, "pb_limit", [numbers.Number], default=0.2)):
        params_passed = False

    return params_passed


def check_imaging_weights_params(imaging_weights_params):
    import numbers

    params_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    if not (
        check_params(
            imaging_weights_params,
            "weighting",
            [str],
            acceptable_data=["natural", "uniform", "briggs", "briggs_abs"],
            default="natural",
        )
    ):
        params_passed = False

    if imaging_weights_params["weighting"] == "briggs_abs":
        if not (
            check_params(
                imaging_weights_params,
                "briggs_abs_noise",
                [numbers.Number],
                default=1.0,
            )
        ):
            params_passed = False

    if (imaging_weights_params["weighting"] == "briggs") or (
        imaging_weights_params["weighting"] == "briggs_abs"
    ):
        if not (
            check_params(
                imaging_weights_params,
                "robust",
                [numbers.Number],
                default=0.5,
                acceptable_range=[-2, 2],
            )
        ):
            params_passed = False

    return params_passed


def check_pb_params(img_dataset, pb_params):
    import numbers

    params_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    # if not(check_params(pb_params, 'pb_name', [str], default='PB')): params_passed = False

    if not (check_params(pb_params, "function", [str], default="casa_airy")):
        params_passed = False

    if not (
        check_params(
            pb_params,
            "list_dish_diameters",
            [list],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        params_passed = False
    if not (
        check_params(
            pb_params,
            "list_blockage_diameters",
            [list],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        params_passed = False

    if len(pb_params["list_dish_diameters"]) != len(
        pb_params["list_blockage_diameters"]
    ):
        print(
            "######### ERROR:Parameter ",
            "list_dish_diameters and list_blockage_diameters must be the same length.",
        )
        params_passed = False

    return params_passed
