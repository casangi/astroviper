"""Helpers to load the legacy SIRIUS reference fixtures (see data/generate_legacy_fixtures.py)."""

from __future__ import annotations

import os

import numpy as np

from astroviper.utils.beam_models import (
    airy_disk_model,
    read_aperture_polynomial_coefficients,
    read_beam_polynomial_coefficients,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# beam models used by each legacy scenario (same order as in the generator)
LEGACY_BEAM_MODELS = {
    "vla_airy": lambda: [airy_disk_model("vla")],
    "vla_airy_pointing": lambda: [airy_disk_model("vla")],
    "alma_het_mosaic_noise": lambda: [airy_disk_model("aca"), airy_disk_model("alma")],
    "evla_polynomial_beam": lambda: [read_beam_polynomial_coefficients("EVLA_")],
    "evla_zernike_beam": lambda: [
        read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup")
    ],
    "evla_mixed_beams": lambda: [
        read_aperture_polynomial_coefficients("EVLA_avg_zcoeffs_SBand_lookup"),
        airy_disk_model("vla"),
    ],
}
LEGACY_BEAM_IMAGE_SIZE = {
    "evla_zernike_beam": [500, 500],
    "evla_mixed_beams": [500, 500],
}


def load_legacy(name: str) -> dict:
    """Load ``legacy_<name>.npz`` into a dict (pointing ``None`` when absent)."""
    with np.load(os.path.join(DATA_DIR, f"legacy_{name}.npz"), allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    d["pointing_ra_dec"] = d["pointing_ra_dec"] if d["pointing_ra_dec"].size else None
    d["beam_params"] = {
        "mueller_selection": d["mueller_selection"],
        "image_size": LEGACY_BEAM_IMAGE_SIZE.get(name, [1000, 1000]),
        "fov_scaling": 4.0,
        "pa_radius": 0.2,
    }
    d["beam_models"] = LEGACY_BEAM_MODELS[name]()
    return d
