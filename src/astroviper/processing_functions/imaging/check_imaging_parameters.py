"""Compatibility entry point for the active imaging-weight parameter checker.

The unused grid/GCF, mosaic beam, rotation, normalization, and primary-beam
checkers have been retired. They mixed validation with mutation and conversion;
new callers use the non-mutating ``imaging.config_check`` validators instead.
"""


def check_imaging_weights_params(imaging_weights_params):
    """Validate weighting parameters without modifying their mapping."""
    from astroviper.processing_functions.imaging.config_check import (
        CONTINUUM_SINGLE_FIELD,
        check_residual_update,
    )

    check_residual_update(
        {"imaging_weights_params": imaging_weights_params},
        mode=CONTINUUM_SINGLE_FIELD,
    )
    return True
