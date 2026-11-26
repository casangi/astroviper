from astroviper.utils.check_params import check_params


def check_mosaic_pb_params(pb_mosaic_params):
    params_passed = True

    if not (check_params(pb_mosaic_params, "pb_name", [str], default="PB")):
        params_passed = False

    if not (check_params(pb_mosaic_params, "weight_name", [str], default="WEIGHT_PB")):
        params_passed = False

    return params_passed
