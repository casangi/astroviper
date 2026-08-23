"""Per-plane (``l``, ``m``) statistics of image cubes.

Every statistic is reduced over the two sky axes of each
``(time, frequency, polarization)`` plane and ignores NaN pixels, so the result
for a cube of shape ``(time, frequency, polarization, l, m)`` has shape
``(time, frequency, polarization)``.
"""

# Statistics computed for every plane over its valid (non-NaN) pixels, in the
# order ``_plane_statistics_1d`` returns them. ``peak`` is the signed value of
# the pixel with the largest absolute value (the CLEAN "peak residual"),
# ``rms`` is ``sqrt(mean(x**2))``, ``std`` the standard deviation about the
# mean, ``mad_sigma`` the median absolute deviation scaled to a Gaussian sigma
# (``1.4826 * MAD`` -- a noise estimate robust to sources) and ``n_pixels``
# the number of valid pixels that went into the statistics.
PLANE_STATISTIC_NAMES = (
    "mean",
    "median",
    "max",
    "min",
    "peak",
    "sum",
    "rms",
    "std",
    "mad_sigma",
    "n_pixels",
)

# Image-variable keys (as used by ``image_data_variables_keep`` /
# ``astroviper.utils.io``) whose statistics are collected by default, in the
# order they are reported. Only the ones present on the dataset are used.
DEFAULT_STATISTICS_VARIABLES = (
    "sky_residual",
    "sky_restored",
    # Written by correct_sky_by_primary_beam (primary_beam_correction=True;
    # branch 265-port-sirius) -- resolved by upper-casing the key when the
    # io table of this branch does not list it yet.
    "sky_restored_primary_beam_corrected",
    "sky_model",
    "sky_dirty",
    "sky_deconvolved",
)


def _plane_statistics_1d(values):
    """Return the :data:`PLANE_STATISTIC_NAMES` values of one plane's valid pixels.

    Parameters
    ----------
    values : numpy.ndarray
        1-D float64 array of the plane's non-NaN pixels (after any masking).

    Returns
    -------
    list of float
        One value per entry of :data:`PLANE_STATISTIC_NAMES`; all NaN (and
        ``n_pixels = 0``) when ``values`` is empty.
    """
    import numpy as np

    n = values.size
    if n == 0:
        out = [np.nan] * len(PLANE_STATISTIC_NAMES)
        out[PLANE_STATISTIC_NAMES.index("n_pixels")] = 0.0
        return out

    mean = float(values.mean())
    median = float(np.median(values))
    peak_index = int(np.abs(values).argmax())
    return [
        mean,
        median,
        float(values.max()),
        float(values.min()),
        float(values[peak_index]),
        float(values.sum()),
        float(np.sqrt(np.mean(values * values))),
        float(values.std()),
        float(1.4826 * np.median(np.abs(values - median))),
        float(n),
    ]


def calculate_plane_statistics(
    img_xds,
    image_data_variables=DEFAULT_STATISTICS_VARIABLES,
    mask_name=None,
):
    """Compute NaN-ignoring per-plane statistics of the image-domain variables.

    For every requested image variable present on ``img_xds`` each statistic in
    :data:`PLANE_STATISTIC_NAMES` is reduced over the ``(l, m)`` axes of every
    ``(time, frequency, polarization)`` plane, both over all pixels and --
    with the suffix ``_masked`` -- over the pixels where the mask variable is
    ``True`` (inside the clean/primary-beam mask). NaN pixels are ignored
    throughout. Each plane is processed on its own so no full-cube copy is made.

    Parameters
    ----------
    img_xds : xarray.Dataset
        In-memory image dataset with dims ``(time, frequency, polarization, l,
        m)`` (all data variables must be materialised, not dask-backed).
    image_data_variables : iterable of str, optional
        Image-variable keys (``"sky_residual"``, ``"sky_restored"``, ...) as in
        :data:`astroviper.utils.io.imaging_data_variable_data_group_roles`;
        each is resolved to its uppercase data-variable name (``SKY_RESIDUAL``)
        and skipped silently when absent. Default
        :data:`DEFAULT_STATISTICS_VARIABLES`.
    mask_name : str, optional
        Name of the boolean mask data variable (same shape as the image
        variables, ``True`` = inside the mask). ``None`` (default) resolves the
        ``"mask"`` role of the ``"residual"`` data group if registered. When no
        mask is available every ``*_masked`` statistic is NaN (and
        ``n_pixels_masked`` is 0).

    Returns
    -------
    dict of str -> xarray.Dataset
        ``{image_variable_key: xarray.Dataset}``, one entry per variable found.
        Each dataset has dims ``(time, frequency, polarization)`` with the
        coordinates of ``img_xds``, one float64 data variable per statistic in
        :data:`PLANE_STATISTIC_NAMES` plus its ``_masked`` twin, and the attrs
        ``data_variable`` (the source name, e.g. ``"SKY_RESIDUAL"``) and
        ``mask_present``.

    Examples
    --------
    >>> stats = calculate_plane_statistics(img_xds)
    >>> stats["sky_residual"]["peak_masked"].sel(polarization="I").values
    """
    import numpy as np
    import xarray as xr

    from astroviper.utils.io import imaging_data_variables_and_dims_double_precision

    if mask_name is None:
        mask_name = img_xds.attrs.get("data_groups", {}).get("residual", {}).get("mask")
    mask_present = mask_name is not None and mask_name in img_xds
    mask_values = img_xds[mask_name].values if mask_present else None

    plane_dims = ("time", "frequency", "polarization")
    coords = {dim: img_xds.coords[dim] for dim in plane_dims}
    stat_names = list(PLANE_STATISTIC_NAMES) + [
        name + "_masked" for name in PLANE_STATISTIC_NAMES
    ]

    results = {}
    for key in image_data_variables:
        variable_name = imaging_data_variables_and_dims_double_precision.get(
            key, {}
        ).get("name", key.upper())
        if variable_name not in img_xds:
            continue
        data = img_xds[variable_name].transpose(*plane_dims, "l", "m").values
        n_time, n_freq, n_pol = data.shape[:3]
        out = np.full((len(stat_names), n_time, n_freq, n_pol), np.nan)

        for t in range(n_time):
            for f in range(n_freq):
                for p in range(n_pol):
                    plane = data[t, f, p]
                    valid = ~np.isnan(plane)
                    # Flatten to float64 once; a single-precision image stays
                    # single on the dataset, only this plane's copy is promoted.
                    all_values = plane[valid].astype(np.float64, copy=False)
                    out[: len(PLANE_STATISTIC_NAMES), t, f, p] = _plane_statistics_1d(
                        all_values
                    )
                    if mask_present:
                        masked_values = plane[valid & mask_values[t, f, p]].astype(
                            np.float64, copy=False
                        )
                        out[len(PLANE_STATISTIC_NAMES) :, t, f, p] = (
                            _plane_statistics_1d(masked_values)
                        )
        if not mask_present:
            out[
                len(PLANE_STATISTIC_NAMES) + PLANE_STATISTIC_NAMES.index("n_pixels")
            ] = 0.0

        results[key] = xr.Dataset(
            {name: (plane_dims, out[i]) for i, name in enumerate(stat_names)},
            coords=coords,
            attrs={"data_variable": variable_name, "mask_present": mask_present},
        )
    return results


def concatenate_plane_statistics(statistics_list):
    """Merge per-chunk plane statistics along ``frequency`` (the reduce step).

    Parameters
    ----------
    statistics_list : list of dict
        Outputs of :func:`calculate_plane_statistics` (or of this function) for
        disjoint frequency chunks; empty dicts (failed chunks) are skipped.

    Returns
    -------
    dict of str -> xarray.Dataset
        Per image variable the datasets concatenated along ``frequency`` and
        sorted by it, so the result has dims ``(time, frequency, polarization)``
        in global frequency order. Empty when every input is empty.
    """
    import xarray as xr

    per_variable = {}
    for statistics in statistics_list:
        for key, ds in (statistics or {}).items():
            per_variable.setdefault(key, []).append(ds)
    return {
        key: xr.concat(parts, dim="frequency").sortby("frequency")
        if len(parts) > 1
        else parts[0]
        for key, parts in per_variable.items()
    }


def plane_statistics_to_dataframe(statistics):
    """Flatten plane statistics to a long-format :class:`pandas.DataFrame`.

    Parameters
    ----------
    statistics : dict of str -> xarray.Dataset
        Output of :func:`calculate_plane_statistics` /
        :func:`concatenate_plane_statistics`.

    Returns
    -------
    pandas.DataFrame
        Columns ``image_variable``, ``statistic``, ``time``, ``frequency``,
        ``polarization``, ``value`` -- one row per (variable, statistic, plane).
        Suitable for feather/parquet (which cannot hold N-D arrays).
    """
    import pandas as pd

    frames = []
    for key, ds in statistics.items():
        long = ds.to_dataframe().reset_index()
        long = long.melt(
            id_vars=["time", "frequency", "polarization"],
            var_name="statistic",
            value_name="value",
        )
        long.insert(0, "image_variable", key)
        frames.append(
            long[
                [
                    "image_variable",
                    "statistic",
                    "time",
                    "frequency",
                    "polarization",
                    "value",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "image_variable",
                "statistic",
                "time",
                "frequency",
                "polarization",
                "value",
            ]
        )
    return pd.concat(frames, ignore_index=True)
