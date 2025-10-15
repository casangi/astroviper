# ducting - code is complex and might fail after some time if parameters is wrong. Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has 14 parameters.

import numpy as np
from .check_params import check_params, _check_dataset


def check_logger_params(logger_params):
    import numbers

    params_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    if not (check_params(logger_params, "log_to_term", [bool], default=True)):
        params_passed = False
    if not (check_params(logger_params, "log_to_file", [bool], default=False)):
        params_passed = False
    if not (check_params(logger_params, "log_file", [str], default="viper_")):
        params_passed = False
    if not (
        check_params(
            logger_params,
            "log_level",
            [str],
            default="INFO",
            acceptable_data=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
    ):
        params_passed = False

    return params_passed


def check_worker_logger_params(logger_params):
    import numbers

    params_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    if not (check_params(logger_params, "log_to_term", [bool], default=False)):
        params_passed = False
    if not (check_params(logger_params, "log_to_file", [bool], default=False)):
        params_passed = False
    if not (check_params(logger_params, "log_file", [str], default="viper_")):
        params_passed = False
    if not (
        check_params(
            logger_params,
            "log_level",
            [str],
            default="INFO",
            acceptable_data=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
    ):
        params_passed = False

    return params_passed
