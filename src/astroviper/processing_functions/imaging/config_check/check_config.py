"""Coordinate the three independent imaging configuration validators."""

from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
    check_driver_workflow,
)
from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_model_update,
)
from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    check_residual_update,
)
from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def check_config(config, *, mode, require_complete=False):
    """Run all three configuration checks without changing the inputs.

    Parameters
    ----------
    config : mapping
        Configuration passed unchanged to every validator. Distributed drivers
        call this coordinator before I/O or graph construction. Direct calls
        to complete local cube workflows use the same coordinator; individual
        residual/model entry points call only their corresponding validator.
    mode : ImagingMode
        Explicit imaging kind and field layout, shared by all three checks.

    require_complete : bool, optional
        Require all coordinates and cycle controls needed by a distributed driver.
        Lower-level entry points check only the configuration they receive.

    Raises
    ------
    TypeError, ValueError, KeyError, NotImplementedError
        Propagated immediately from the first failing validator. Later checks
        are not run after a failure.
    """
    check_residual_update(config, mode=mode, require_complete=require_complete)
    check_model_update(config, mode=mode, require_complete=require_complete)
    check_driver_workflow(config, mode=mode, require_complete=require_complete)
