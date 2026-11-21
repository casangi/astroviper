"""Statistics of an image in an annulus of its primary beam.

The noise measurement of the CASA pipeline's image quality assessment
[pipeline ``image_rms``]: the restored image, not divided by the primary beam,
where the primary beam lies strictly between two levels and outside the clean
mask, after Chauvenet clipping, plane by plane; then the median over the
channels the caller selects. The levels and the channels are the caller's
choice: nothing here is a telescope default, and nothing scores the result.
``IMAGE_QUALITY_METRICS.md`` at the repository root maps these functions to the
pipeline and lists the departures.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from astroviper.utils.param_docs import shares_param_docs

# Statistics of the pixels Chauvenet's criterion keeps, as CASA ``ia.statistics``
# reports them: ``n_pixels`` [``npts``], ``rms`` = ``sqrt(mean(x**2))``,
# ``standard_deviation`` about the mean with ``ddof=1`` [``sigma``], ``median``
# and ``median_absolute_deviation`` = ``median(|x - median|)``, unscaled
# [``medabsdevmed``].
ANNULUS_STATISTIC_NAMES = (
    "n_pixels",
    "rms",
    "standard_deviation",
    "median",
    "median_absolute_deviation",
)
# Below this median number of kept pixels per channel the summary takes the
# whole annulus instead of the annulus outside the clean mask [pipeline PIPE-541].
MIN_MEDIAN_N_PIXELS = 10
_IMAGE_DIMS = ("time", "frequency", "polarization", "l", "m")
_PLANE_DIMS = _IMAGE_DIMS[:3]


def _data_group_variables(img_xds, image_data_group_in_name, roles, optional_roles=()):
    """Data variables of the ``roles``, and of the ``optional_roles`` present."""
    data_groups = img_xds.attrs.get("data_groups", {})
    if image_data_group_in_name not in data_groups:
        raise ValueError(f"The image has no data group {image_data_group_in_name!r}.")
    data_group = data_groups[image_data_group_in_name]
    missing = [role for role in roles if role not in data_group]
    if missing:
        raise ValueError(
            f"Data group {image_data_group_in_name!r} lacks the roles {missing}."
        )
    names = {role: data_group[role] for role in roles}
    for role in optional_roles:
        if role in data_group and data_group[role] in img_xds:
            names[role] = data_group[role]
    for name in names.values():
        if name not in img_xds:
            raise ValueError(f"The image has no data variable {name!r}.")
        dims = img_xds[name].dims
        if dims != _IMAGE_DIMS:
            raise ValueError(f"{name} has the dimensions {dims}, not {_IMAGE_DIMS}.")
    return names


def _clipped_statistics(values):
    """:data:`ANNULUS_STATISTIC_NAMES` of the values Chauvenet's criterion keeps."""
    from astroviper.processing_functions.imaging.find_continuum import (
        image_statistics,
    )

    statistics = image_statistics(values, algorithm="chauvenet", maxiter=5)
    return (
        int(statistics["npts"]),
        statistics["rms"],
        statistics["sigma"],
        statistics["median"],
        statistics["medabsdevmed"],
    )


@shares_param_docs
def primary_beam_annulus_limits(
    img_xds: xr.Dataset,
    nominal_annulus_limits: Sequence[float],
    image_data_group_in_name: str = "restored",
    *,
    input_checked: bool = False,
) -> tuple[float, float]:
    """Primary beam levels of the noise annulus of an image [pipeline ``pblimits``].

    The nominal levels, unless the primary beam at the edge of the image is
    above the nominal lower level: then its values ``n_m // 20`` and
    ``2 * (n_m // 20)`` pixels from the edge. The primary beam is read along
    ``m`` from index 0 at ``l`` index ``n_l // 2``, time 0, polarization 0 and
    frequency index ``n_f // 2`` of ``img_xds``, the middle channel of the
    dataset given: a block of channels gives the levels of its own middle
    channel, so callers pass the whole cube or its middle channel.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image on ``(time, frequency, polarization, l, m)``; only the one
        column of its primary beam is read.
    nominal_annulus_limits : sequence of float
        Nominal primary beam levels ``(lower, upper)``, with ``0 <= lower <=
        upper <= 1``. The CASA pipeline passes ``(0.2, 0.3)``.
    image_data_group_in_name : str, optional
        Data group whose ``"primary_beam"`` role names the primary beam.
    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` [pipeline ``pblimit_image``, ``pblimit_cleanmask``],
        equal when they come from the primary beam of an image with
        ``n_m < 20``.

    Raises
    ------
    TypeError, ValueError
        If ``nominal_annulus_limits`` is malformed, unless ``input_checked``.
    ValueError
        If the data group is missing or lacks the ``primary_beam`` role, or the
        primary beam is not on the image dimensions.
    TypeError
        If the primary beam is dask-backed.

    Notes
    -----
    The pipeline compares the edge value with 0.2, its nominal lower level, in
    double precision. It also keeps its nominal levels when the edge pixel is
    masked and otherwise skips zero pixels; ``tclean`` masks and zeroes the
    primary beam at and below ``|pblimit|``, at most 0.2, so neither changes
    the levels. Here a zero or NaN edge keeps the nominal levels.
    """
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check._predicates import (
            check_annulus_limits,
        )

        check_annulus_limits(nominal_annulus_limits, name="nominal_annulus_limits")

    names = _data_group_variables(img_xds, image_data_group_in_name, ("primary_beam",))
    primary_beam = img_xds[names["primary_beam"]]
    if primary_beam.chunks is not None:
        raise TypeError(
            f"{names['primary_beam']} is dask-backed; open the image without dask."
        )
    column = primary_beam.isel(
        time=0,
        frequency=primary_beam.sizes["frequency"] // 2,
        polarization=0,
        l=primary_beam.sizes["l"] // 2,
    ).values.astype(np.float64)
    lower, upper = (float(limit) for limit in nominal_annulus_limits)
    if column[0] > lower:
        offset = column.size // 20
        return float(column[offset]), float(column[2 * offset])
    return lower, upper


@shares_param_docs
def calculate_primary_beam_annulus_statistics(
    img_xds: xr.Dataset,
    annulus_limits: Sequence[float],
    mask_union: bool = False,
    image_data_group_in_name: str = "restored",
    *,
    input_checked: bool = False,
) -> xr.Dataset:
    """Chauvenet-clipped statistics of every image plane in a primary beam annulus.

    In every ``(time, frequency, polarization)`` plane, the finite pixels
    where the plane's primary beam lies strictly between the two levels are
    clipped with Chauvenet's criterion
    (:func:`~astroviper.processing_functions.image_analysis.statistics.chauvenet.chauvenet_sample`)
    and reduced to :data:`ANNULUS_STATISTIC_NAMES`, as CASA
    ``ia.statistics(algorithm="chauvenet", maxiter=5)`` reduces them, once for
    the whole annulus and once for the annulus outside the clean mask
    [pipeline ``cleanbox.analyse_clean_result``]. The primary beam is compared
    with the levels in double precision, as in CASA's mask expressions.

    Parameters
    ----------
    img_xds : xarray.Dataset
        In-memory image on ``(time, frequency, polarization, l, m)``. The
        pipeline measures the restored image of its last ``tclean`` run, not
        divided by the primary beam; the cube imager's ``"restored"`` group
        holds it unless ``restoration_output="primary_beam_corrected"``.
    annulus_limits : sequence of float
        Primary beam levels ``(lower, upper)``, with ``0 <= lower <= upper <=
        1``, as :func:`primary_beam_annulus_limits` gives them. The pipeline
        writes them into its mask expressions with six decimals: pass
        ``round(level, 6)`` to reproduce it.
    mask_union : bool, optional
        Exclude from every plane the union of the clean masks of all planes of
        ``img_xds`` [pipeline flattened clean mask], as the pipeline does for
        a whole cube. Default ``False``: every plane excludes its own mask.
    image_data_group_in_name : str, optional
        Data group whose ``"sky"`` and ``"primary_beam"`` roles name the image
        and its primary beam, and whose optional ``"mask"`` role names the
        clean mask (``True`` inside), not a primary beam mask.
    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    xarray.Dataset
        On ``(time, frequency, polarization)`` with the coordinates of
        ``img_xds``: the :data:`ANNULUS_STATISTIC_NAMES` of the whole annulus
        and, with the suffix ``_outside_mask``, of the annulus outside the
        clean mask, which without a mask is the whole annulus. ``n_pixels`` is
        int64 and the other statistics are float64, NaN for a plane without
        kept pixels.
        The attrs ``data_variable`` and ``mask_source`` name the image and the
        mask (``None`` without one); ``mask_union`` and ``annulus_limits``
        repeat the arguments.

    Raises
    ------
    TypeError, ValueError
        If ``annulus_limits`` or ``mask_union`` is malformed, unless
        ``input_checked``.
    ValueError
        If the data group is missing or lacks a role, or a variable is not on
        the image dimensions.
    TypeError
        If a variable is dask-backed or the mask is not bool.
    """
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check._common import (
            require_boolean,
        )
        from astroviper.processing_functions.imaging.config_check._predicates import (
            check_annulus_limits,
        )

        check_annulus_limits(annulus_limits)
        require_boolean("mask_union", mask_union)

    names = _data_group_variables(
        img_xds, image_data_group_in_name, ("sky", "primary_beam"), ("mask",)
    )
    for name in names.values():
        if img_xds[name].chunks is not None:
            raise TypeError(
                f"{name} is dask-backed; load the image in the node-task layer."
            )
    mask_name = names.get("mask")
    if mask_name is not None and img_xds[mask_name].dtype != bool:
        raise TypeError(f"{mask_name} must be bool, not {img_xds[mask_name].dtype}.")

    sky = img_xds[names["sky"]].values
    primary_beam = img_xds[names["primary_beam"]].values
    mask = None if mask_name is None else img_xds[mask_name].values
    if mask is not None and mask_union:
        mask = np.broadcast_to(mask.any(axis=(0, 1, 2)), mask.shape)
    lower, upper = (float(limit) for limit in annulus_limits)

    planes_shape = sky.shape[:3]
    statistics = {}
    for suffix in ("", "_outside_mask"):
        statistics[f"n_pixels{suffix}"] = np.zeros(planes_shape, dtype=np.int64)
        for name in ANNULUS_STATISTIC_NAMES[1:]:
            statistics[name + suffix] = np.full(planes_shape, np.nan)
    for index in np.ndindex(planes_shape):
        plane = sky[index]
        beam = primary_beam[index].astype(np.float64)
        annulus = np.isfinite(plane) & (beam > lower) & (beam < upper)
        whole = _clipped_statistics(plane[annulus])
        outside = (
            whole
            if mask is None
            else _clipped_statistics(plane[annulus & ~mask[index]])
        )
        for name, value, value_outside in zip(
            ANNULUS_STATISTIC_NAMES, whole, outside, strict=True
        ):
            statistics[name][index] = value
            statistics[name + "_outside_mask"][index] = value_outside

    return xr.Dataset(
        {name: (_PLANE_DIMS, values) for name, values in statistics.items()},
        coords={dim: img_xds.coords[dim] for dim in _PLANE_DIMS},
        attrs={
            "data_variable": names["sky"],
            "mask_source": mask_name,
            "mask_union": mask_union,
            "annulus_limits": [lower, upper],
        },
    )


def summarize_primary_beam_annulus_rms(
    annulus_statistics: xr.Dataset,
    channel_indices: Sequence[int] | None = None,
) -> xr.Dataset:
    """Median, minimum and maximum over channels of the annulus RMS.

    For every time and polarization [pipeline ``image_rms``,
    ``image_rms_min``, ``image_rms_max``]: the RMS outside the clean mask, or
    that of the whole annulus when the median over all channels of the pixels
    kept outside the mask is below :data:`MIN_MEDIAN_N_PIXELS`. A channel
    without kept pixels counts as an RMS of 0, as CASA reports it.

    Parameters
    ----------
    annulus_statistics : xarray.Dataset
        Output of :func:`calculate_primary_beam_annulus_statistics` for every
        channel of the image, with ``mask_union=True`` to reproduce the
        pipeline on a cube.
    channel_indices : sequence of int, optional
        Indices along ``frequency`` of the channels the median, minimum and
        maximum are taken over; a repeated index counts repeatedly, as in the
        pipeline. Default: every channel.

    Returns
    -------
    xarray.Dataset
        On ``(time, polarization)``: float64 ``rms_median``, ``rms_min`` and
        ``rms_max``, NaN when no channel keeps a pixel even in the whole
        annulus, and bool ``full_annulus``, whether the whole annulus replaced
        the annulus outside the mask, never for statistics without a mask
        (attr ``mask_source`` ``None``).

    Raises
    ------
    ValueError
        If ``channel_indices`` is empty or holds a value that is not the index
        of a channel.

    Notes
    -----
    The pipeline makes this choice once for the image, from the pixels kept
    in all polarizations together; with more than one polarization, this
    function makes it for each polarization on its own.
    """
    statistics = annulus_statistics.transpose(*_PLANE_DIMS)
    n_channels = statistics.sizes["frequency"]
    if channel_indices is None:
        channels = np.arange(n_channels)
    else:
        channels = np.asarray(channel_indices)
        if (
            channels.ndim != 1
            or channels.size == 0
            or channels.dtype.kind not in "iu"
            or channels.min() < 0
            or channels.max() >= n_channels
        ):
            raise ValueError(
                f"channel_indices must hold indices of the {n_channels} channels."
            )

    n_outside = statistics["n_pixels_outside_mask"].values
    masked = annulus_statistics.attrs.get("mask_source", "") is not None
    full_annulus = masked & (np.median(n_outside, axis=1) < MIN_MEDIAN_N_PIXELS)
    use_whole = full_annulus[:, np.newaxis, :]
    n_pixels = np.where(use_whole, statistics["n_pixels"].values, n_outside)
    rms = np.where(
        use_whole, statistics["rms"].values, statistics["rms_outside_mask"].values
    )
    rms = np.where(n_pixels > 0, rms, 0.0)[:, channels, :]
    empty = ~(n_pixels > 0).any(axis=1)
    summary = {
        "rms_median": np.median(rms, axis=1),
        "rms_min": rms.min(axis=1),
        "rms_max": rms.max(axis=1),
    }
    for values in summary.values():
        values[empty] = np.nan
    dims = ("time", "polarization")
    return xr.Dataset(
        {
            **{name: (dims, values) for name, values in summary.items()},
            "full_annulus": (dims, full_annulus),
        },
        coords={dim: statistics.coords[dim] for dim in dims},
    )
