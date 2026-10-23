"""Non-mutating imaging configuration checks shared by all three layers.

Only a first subset of issue #319 is implemented. Runtime array/native checks
remain independent of configuration validation and must never be bypassed.
"""

from ._common import CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD, ImagingMode
from .check_config import check_config
from .param_check_driver_workflow import check_driver_workflow
from .param_check_model_update import check_model_update
from .param_check_residual_update import check_residual_update

__all__ = [
    "ImagingMode",
    "CUBE_SINGLE_FIELD",
    "CONTINUUM_SINGLE_FIELD",
    "check_config",
    "check_driver_workflow",
    "check_model_update",
    "check_residual_update",
]
