import numpy as np
from astroviper._utils._parm_utils._check_parms import _check_parms


def _check_grid_parms(grid_parms):
    import numbers

    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    if not (
        _check_parms(
            grid_parms,
            "image_size",
            [list],
            list_acceptable_data_types=[int],
            list_len=2,
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            grid_parms,
            "image_center",
            [list],
            list_acceptable_data_types=[int],
            list_len=2,
            default=np.array(grid_parms["image_size"]) // 2,
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            grid_parms,
            "cell_size",
            [list],
            list_acceptable_data_types=[numbers.Number],
            list_len=2,
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            grid_parms,
            "fft_padding",
            [numbers.Number],
            default=1.2,
            acceptable_range=[1, 10],
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            grid_parms,
            "chan_mode",
            [str],
            acceptable_data=["cube", "continuum"],
            default="cube",
        )
    ):
        parms_passed = False

    if parms_passed == True:
        grid_parms["image_size"] = np.array(grid_parms["image_size"]).astype(int)
        grid_parms["image_size_padded"] = (
            grid_parms["fft_padding"] * grid_parms["image_size"]
        ).astype(int)
        grid_parms["image_center"] = np.array(grid_parms["image_center"])
        # grid_parms['cell_size'] = arc_sec_to_rad * np.array(grid_parms['cell_size'])
        # grid_parms['cell_size'][0] = -grid_parms['cell_size'][0]

    return parms_passed


##################### Function Specific Parms #####################
def _check_gcf_parms(gcf_parms):
    import numbers

    parms_passed = True

    if not (
        _check_parms(
            gcf_parms,
            "function",
            [str],
            acceptable_data=["casa_airy", "airy"],
            default="casa_airy",
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            gcf_parms,
            "freq_chan",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            gcf_parms,
            "list_dish_diameters",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            gcf_parms,
            "list_blockage_diameters",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            gcf_parms,
            "unique_ant_indx",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        parms_passed = False

    if not (
        _check_parms(
            gcf_parms,
            "pol",
            [list, np.array],
            list_acceptable_data_types=[numbers.Number, str],
            list_len=-1,
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            gcf_parms, "chan_tolerance_factor", [numbers.Number], default=0.005
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            gcf_parms,
            "oversampling",
            [list, np.array],
            list_acceptable_data_types=[int],
            list_len=2,
            default=[10, 10],
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            gcf_parms,
            "max_support",
            [list, np.array],
            list_acceptable_data_types=[int],
            list_len=2,
            default=[15, 15],
        )
    ):
        parms_passed = False
    # if not(_check_parms(gcf_parms, 'image_phase_center', [list,np.array], list_acceptable_data_types=[numbers.Number], list_len=2)): parms_passed = False
    if not (
        _check_parms(
            gcf_parms, "support_cut_level", [numbers.Number], default=2.5 * 10**-2
        )
    ):
        parms_passed = False
    if not (_check_parms(gcf_parms, "a_chan_num_chunk", [int], default=3)):
        parms_passed = False

    if gcf_parms["function"] == "airy" or gcf_parms["function"] == "casa_airy":
        if not (
            _check_parms(
                gcf_parms,
                "list_dish_diameters",
                [list, np.array],
                list_acceptable_data_types=[numbers.Number],
                list_len=-1,
            )
        ):
            parms_passed = False
        if not (
            _check_parms(
                gcf_parms,
                "list_blockage_diameters",
                [list, np.array],
                list_acceptable_data_types=[numbers.Number],
                list_len=-1,
            )
        ):
            parms_passed = False

        if len(gcf_parms["list_dish_diameters"]) != len(
            gcf_parms["list_blockage_diameters"]
        ):
            print(
                "######### ERROR:Parameter ",
                "list_dish_diameters and list_blockage_diameters must be the same length.",
            )
            parms_passed = False

    if parms_passed == True:
        gcf_parms["oversampling"] = np.array(gcf_parms["oversampling"]).astype(int)
        gcf_parms["max_support"] = np.array(gcf_parms["max_support"]).astype(int)
        # gcf_parms['image_phase_center'] =  np.array(gcf_parms['image_phase_center'])
        gcf_parms["freq_chan"] = np.array(gcf_parms["freq_chan"])
        gcf_parms["list_dish_diameters"] = np.array(gcf_parms["list_dish_diameters"])
        gcf_parms["list_blockage_diameters"] = np.array(
            gcf_parms["list_blockage_diameters"]
        )
        gcf_parms["unique_ant_indx"] = np.array(gcf_parms["unique_ant_indx"])
        # gcf_parms['basline_ant'] =  np.array(gcf_parms['basline_ant'])
        gcf_parms["pol"] = np.array(gcf_parms["pol"])

    return parms_passed


def _check_mosaic_pb_parms(pb_mosaic_parms):
    parms_passed = True

    if not (_check_parms(pb_mosaic_parms, "pb_name", [str], default="PB")):
        parms_passed = False

    if not (_check_parms(pb_mosaic_parms, "weight_name", [str], default="WEIGHT_PB")):
        parms_passed = False

    return parms_passed


def _check_rotation_parms(rotation_parms):
    import numbers

    parms_passed = True

    if not (
        _check_parms(
            rotation_parms,
            "new_phase_center",
            [list],
            list_acceptable_data_types=[numbers.Number],
            list_len=2,
        )
    ):
        parms_passed = False

    if not (
        _check_parms(
            rotation_parms, "common_tangent_reprojection", [bool], default=True
        )
    ):
        parms_passed = False

    if not (_check_parms(rotation_parms, "single_precision", [bool], default=True)):
        parms_passed = False

    return parms_passed


def _check_norm_parms(norm_parms):
    import numbers

    parms_passed = True

    if not (
        _check_parms(
            norm_parms,
            "norm_type",
            [str],
            default="flat_sky",
            acceptable_data=["flat_noise", "flat_sky", "none"],
        )
    ):
        parms_passed = False

    if not (_check_parms(norm_parms, "single_precision", [bool], default=True)):
        parms_passed = False

    if not (_check_parms(norm_parms, "pb_limit", [numbers.Number], default=0.2)):
        parms_passed = False

    return parms_passed


def _check_imaging_weights_parms(imaging_weights_parms):
    import numbers

    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    if not (
        _check_parms(
            imaging_weights_parms,
            "weighting",
            [str],
            acceptable_data=["natural", "uniform", "briggs", "briggs_abs"],
            default="natural",
        )
    ):
        parms_passed = False

    if imaging_weights_parms["weighting"] == "briggs_abs":
        if not (
            _check_parms(
                imaging_weights_parms, "briggs_abs_noise", [numbers.Number], default=1.0
            )
        ):
            parms_passed = False

    if (imaging_weights_parms["weighting"] == "briggs") or (
        imaging_weights_parms["weighting"] == "briggs_abs"
    ):
        if not (
            _check_parms(
                imaging_weights_parms,
                "robust",
                [numbers.Number],
                default=0.5,
                acceptable_range=[-2, 2],
            )
        ):
            parms_passed = False

    return parms_passed


def _check_pb_parms(img_dataset, pb_parms):
    import numbers

    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    # if not(_check_parms(pb_parms, 'pb_name', [str], default='PB')): parms_passed = False

    if not (_check_parms(pb_parms, "function", [str], default="casa_airy")):
        parms_passed = False

    if not (
        _check_parms(
            pb_parms,
            "list_dish_diameters",
            [list],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        parms_passed = False
    if not (
        _check_parms(
            pb_parms,
            "list_blockage_diameters",
            [list],
            list_acceptable_data_types=[numbers.Number],
            list_len=-1,
        )
    ):
        parms_passed = False

    if len(pb_parms["list_dish_diameters"]) != len(pb_parms["list_blockage_diameters"]):
        print(
            "######### ERROR:Parameter ",
            "list_dish_diameters and list_blockage_diameters must be the same length.",
        )
        parms_passed = False

    return parms_passed
