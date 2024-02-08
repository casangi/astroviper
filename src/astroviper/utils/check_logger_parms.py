# ducting - code is complex and might fail after some time if parameters is wrong. Sensable values are also checked. Gives printout of all wrong parameters. Dirty images alone has 14 parameters.

import numpy as np
from .check_parms import check_parms, _check_dataset


def check_logger_parms(logger_parms):
    import numbers

    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    if not (check_parms(logger_parms, "log_to_term", [bool], default=True)):
        parms_passed = False
    if not (check_parms(logger_parms, "log_to_file", [bool], default=False)):
        parms_passed = False
    if not (check_parms(logger_parms, "log_file", [str], default="viper_")):
        parms_passed = False
    if not (
        check_parms(
            logger_parms,
            "log_level",
            [str],
            default="INFO",
            acceptable_data=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
    ):
        parms_passed = False

    return parms_passed


def check_worker_logger_parms(logger_parms):
    import numbers

    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)

    if not (check_parms(logger_parms, "log_to_term", [bool], default=False)):
        parms_passed = False
    if not (check_parms(logger_parms, "log_to_file", [bool], default=False)):
        parms_passed = False
    if not (check_parms(logger_parms, "log_file", [str], default="viper_")):
        parms_passed = False
    if not (
        check_parms(
            logger_parms,
            "log_level",
            [str],
            default="INFO",
            acceptable_data=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
    ):
        parms_passed = False

    return parms_passed
