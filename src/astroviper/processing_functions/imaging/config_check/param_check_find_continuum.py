"""Pure checks for the find_continuum configuration and its opened image.

The work directory and file access belong to the workflow. These checks
neither open files nor insert configuration defaults.
"""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
    require_integer,
    require_mapping,
)

# Data group roles of the cubes findContinuum reads.
FIND_CONTINUUM_ROLES = (
    "sky",
    "primary_beam",
    "point_spread_function",
    "beam_fit_params_point_spread_function",
)

# ========================================================================
# Supplied configuration, checked before opening the image
# ========================================================================


def check_find_continuum_config(n_bin, primary_beam_limit, n_mapping_parallelism):
    """Validate the channel binning, the primary beam limit and the frequency chunking.

    Parameters
    ----------
    n_bin : int
        Channels averaged in the mean spectrum, at least 1.
    primary_beam_limit : float or None
        In ``[0, 1)``, or ``None``.
    n_mapping_parallelism : mapping or None
        ``{"frequency": n}`` with at least one chunk, or ``None``.
    """
    require_integer("n_bin", n_bin, minimum=1)
    if primary_beam_limit is not None:
        require_finite_real("primary_beam_limit", primary_beam_limit, scalar=True)
        if not 0.0 <= primary_beam_limit < 1.0:
            raise ValueError("primary_beam_limit must be in [0, 1).")
    if n_mapping_parallelism is None:
        return
    require_mapping("n_mapping_parallelism", n_mapping_parallelism)
    if set(n_mapping_parallelism) != {"frequency"}:
        raise ValueError('n_mapping_parallelism maps only "frequency".')
    require_integer(
        'n_mapping_parallelism["frequency"]',
        n_mapping_parallelism["frequency"],
        minimum=1,
    )


# ========================================================================
# Opened image metadata, always checked even for prechecked configuration
# ========================================================================


def check_find_continuum_inputs(
    img_xds,
    image_data_group_in_name,
    atmospheric_transmission,
    atmospheric_sky_temperature,
):
    """Require the cubes of the data group, one time and Stokes I, and one
    atmospheric value per channel.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Opened image.
    image_data_group_in_name : str
        Data group with the roles of ``FIND_CONTINUUM_ROLES``.
    atmospheric_transmission, atmospheric_sky_temperature : array_like
        One value per channel of ``img_xds``.

    Raises
    ------
    ValueError
        The data group is missing or lacks a role, the image has more than one
        time or no Stokes I plane, or a spectrum has another length.

    Notes
    -----
    Only metadata is inspected: no image pixels are loaded.
    """
    data_groups = img_xds.attrs.get("data_groups", {})
    if image_data_group_in_name not in data_groups:
        raise ValueError(f"The image has no data group {image_data_group_in_name!r}.")
    data_group = data_groups[image_data_group_in_name]
    missing = [role for role in FIND_CONTINUUM_ROLES if role not in data_group]
    if missing:
        raise ValueError(
            f"Data group {image_data_group_in_name!r} lacks the roles {missing}."
        )
    if img_xds.sizes["time"] != 1:
        raise ValueError("find_continuum needs an image with one time.")
    if "I" not in img_xds.polarization.values:
        raise ValueError("find_continuum needs a Stokes I plane.")
    n_channels = img_xds.sizes["frequency"]
    for name, spectrum in (
        ("atmospheric_transmission", atmospheric_transmission),
        ("atmospheric_sky_temperature", atmospheric_sky_temperature),
    ):
        if np.shape(spectrum) != (n_channels,):
            raise ValueError(f"{name} must have one value per channel ({n_channels}).")
