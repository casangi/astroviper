"""Validate explicit primary-beam prescriptions without resolving user inputs."""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
    require_integer,
    require_mapping,
)


def _check_aperture(dish, blockage, *, name):
    """Require physical aperture dimensions, preserving the supplied values."""
    dish = require_finite_real(f"{name} dish diameter", dish, positive=True)
    blockage = require_finite_real(f"{name} blockage diameter", blockage)
    if dish.shape != blockage.shape or np.any(blockage < 0) or np.any(blockage >= dish):
        raise ValueError(f"{name} needs matching diameters with 0 <= blockage < dish.")


def check_continuum_primary_beam(image_params):
    """Check explicit continuum beam settings; inferred apertures are separate.

    Parameters
    ----------
    image_params : mapping
        Image configuration with optional continuum beam parameters.
    """
    # --- Beam prescription ---
    # An unsupported explicit model must not survive until beam evaluation.
    model = image_params.get("primary_beam_model", "auto")
    if model not in ("auto", "airy", "casa_airy"):
        raise ValueError("primary_beam_model must be 'auto', 'airy', or 'casa_airy'.")

    # --- Explicit aperture lists ---
    # Empty/mismatched lists can otherwise truncate a zip or leave a zero beam.
    for name in ("list_dish_diameters", "list_blockage_diameters"):
        if image_params.get(name) is not None:
            values = require_finite_real(name, image_params[name])
            if values.ndim != 1 or np.any(values < 0):
                raise ValueError(f"{name} must be a nonempty non-negative vector.")
    dish = image_params.get("list_dish_diameters")
    blockage = image_params.get("list_blockage_diameters")
    if dish is not None:
        require_finite_real("list_dish_diameters", dish, positive=True)
        if blockage is not None:
            _check_aperture(dish, blockage, name="continuum primary beam")

    # --- Explicit support ---
    # A nonphysical cut radius cannot define the CASA beam support.
    if "primary_beam_max_radius_1ghz" in image_params:
        require_finite_real(
            "primary_beam_max_radius_1ghz",
            image_params["primary_beam_max_radius_1ghz"],
            positive=True,
            scalar=True,
        )


def check_cube_primary_beam(params):
    """Check cube beam schema and physical values before constructing a beam.

    Parameters
    ----------
    params : mapping or None
        ``primary_beam_params`` supplied to the cube workflow.
    """
    from astroviper.utils.beam_models import AIRY_DISK_MODELS, airy_disk_model

    if params is None:
        return
    require_mapping("primary_beam_params", params)

    # --- Recognized prescription ---
    # A self-contained model cannot be combined with ignored sibling overrides.
    allowed = {
        "telescope",
        "effective_dish_diameter",
        "blockage_diameter",
        "max_rad_1GHz",
        "beam_model",
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unknown primary_beam_params keys: {sorted(unknown)}.")
    if "beam_model" in params:
        if len(params) != 1:
            raise ValueError(
                "beam_model cannot be combined with other primary_beam_params."
            )
        _check_explicit_beam_model(params["beam_model"])
        return

    # --- Telescope and applicable overrides ---
    # VLA uses polynomial coefficients, so dish overrides would be ineffective.
    telescope = params.get("telescope", "alma")
    if not isinstance(telescope, str) or telescope.lower() not in (
        *AIRY_DISK_MODELS,
        "vla",
    ):
        raise ValueError("primary_beam_params has an unsupported telescope.")
    if telescope.lower() == "vla":
        if set(params) & {"effective_dish_diameter", "blockage_diameter"}:
            raise ValueError("VLA primary beams do not accept dish/blockage overrides.")
        if "max_rad_1GHz" in params:
            if params["max_rad_1GHz"] is None:
                raise ValueError("max_rad_1GHz must be strictly positive.")
            require_finite_real(
                "max_rad_1GHz", params["max_rad_1GHz"], positive=True, scalar=True
            )
        return

    # --- Physical aperture ---
    # Read defaults into local variables without filling or rewriting params.
    defaults = airy_disk_model(telescope)
    dish = params.get("effective_dish_diameter", defaults["dish_diameter"])
    blockage = params.get("blockage_diameter", defaults["blockage_diameter"])
    require_finite_real("effective_dish_diameter", dish, scalar=True)
    require_finite_real("blockage_diameter", blockage, scalar=True)
    _check_aperture(dish, blockage, name="primary_beam_params")
    if params.get("max_rad_1GHz") is not None:
        require_finite_real(
            "max_rad_1GHz", params["max_rad_1GHz"], positive=True, scalar=True
        )


def _check_explicit_beam_model(model):
    """Check small analytic/polynomial model metadata without evaluating pixels."""
    from collections.abc import Mapping

    import xarray as xr

    # --- Analytic model ---
    # Required fields and legacy aliases follow the existing beam normalizer.
    if isinstance(model, Mapping) and not isinstance(model, xr.Dataset):
        from astroviper.utils.beam_models import normalize_beam_model_dict

        values = normalize_beam_model_dict(model)
        if values["func"] not in ("airy", "casa_airy", "none"):
            raise ValueError("beam_model func must be 'airy', 'casa_airy', or 'none'.")
        require_finite_real(
            "beam_model dish_diameter", values["dish_diameter"], scalar=True
        )
        require_finite_real(
            "beam_model blockage_diameter", values["blockage_diameter"], scalar=True
        )
        _check_aperture(
            values["dish_diameter"], values["blockage_diameter"], name="beam_model"
        )
        if values["max_rad_1GHz"] is not None:
            require_finite_real(
                "beam_model max_rad_1GHz",
                values["max_rad_1GHz"],
                positive=True,
                scalar=True,
            )
        return

    # --- Polynomial model ---
    # Imaging supports symmetric power beams, not Jones/aperture models.
    if not isinstance(model, xr.Dataset):
        raise TypeError(
            "Unsupported beam model: expected an analytic mapping or beam-polynomial dataset."
        )
    if "BPC" not in model:
        raise ValueError(
            "beam_model must be an analytic mapping or beam-polynomial dataset."
        )
    from astroviper.utils.beam_models import normalize_beam_model_xds

    model = normalize_beam_model_xds(model)
    if model.BPC.dims != ("frequency", "polarization", "coefficient"):
        raise ValueError(
            "beam_model BPC needs (frequency, polarization, coefficient) dimensions."
        )
    if not all(model.BPC.sizes.values()):
        raise ValueError("beam_model BPC dimensions must be nonempty.")
    # Coefficients are a small explicit model table, not visibility/image data.
    require_finite_real("beam_model BPC coefficients", model.BPC.values)
    frequencies = require_finite_real(
        "beam_model frequency", model.frequency.values, positive=True
    )
    if frequencies.ndim != 1:
        raise ValueError("beam_model frequency must be one-dimensional.")
    for name in ("dish_diameter", "max_rad_1GHz"):
        if name not in model.attrs:
            raise ValueError(f"beam_model requires {name} metadata.")
        require_finite_real(
            f"beam_model {name}", model.attrs[name], positive=True, scalar=True
        )
    if model.attrs.get("n_sample") is not None:
        require_integer("beam_model n_sample", model.attrs["n_sample"], minimum=2)


def check_primary_beam_limit(value):
    """Reject cutoffs that cannot retain a finite, nonempty beam region."""
    limit = require_finite_real("primary_beam_limit", value, scalar=True)
    if not 0 <= limit < 1:
        raise ValueError("primary_beam_limit must lie in [0, 1).")
