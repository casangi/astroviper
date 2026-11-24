"""Pure checks for the find_continuum configuration and its opened image.

The work directory and file access belong to the workflow. These checks
neither open files nor insert configuration defaults.
"""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_boolean,
    require_finite_real,
    require_integer,
    require_mapping,
)

# Data group roles of the cubes findContinuum reads; it reads the point spread
# function only through its beams.
FIND_CONTINUUM_ROLES = (
    "sky",
    "primary_beam",
    "beam_fit_params_point_spread_function",
)
# Format version of the first-round store made next to the cube; 3 weights
# the moments of each window of a cube of several by its own rest frequency.
FIND_CONTINUUM_FIRST_ROUND_VERSION = 3
# Telescopes of findContinuum's time-division-mode rules (fC 4449-4463).
FIND_CONTINUUM_ULTRA_TELESCOPES = ("ALMA", "VLA")
# Fewest channels of a spectral window: findContinuum's selection needs two
# baseline channels at 19 % of the window (fC 5092-5102).
FIND_CONTINUUM_ULTRA_MIN_CHANNELS = 8
_IMAGE_DIMS = ("time", "frequency", "polarization", "l", "m")

# ========================================================================
# Supplied configuration, checked before opening the image
# ========================================================================


def check_find_continuum_config(n_bin, primary_beam_limit, n_mapping_parallelism):
    """Validate the channel binning, the primary beam limit and the frequency
    chunking.

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
    _check_frequency_chunks(n_mapping_parallelism)


def check_find_continuum_ultra_config(
    n_bin,
    spectral_dynamic_range_bandwidth,
    telescope_name,
    single_continuum,
    primary_beam_limit,
    first_round_store,
    n_mapping_parallelism,
    thread_info,
    memory_budget_gb,
    monitor_resources_seconds,
):
    """Validate the configuration of find_continuum_ultra.

    Parameters
    ----------
    n_bin : int
        Channels averaged in the mean spectrum, at least 1 (findContinuum
        divides by it, fC 1009).
    spectral_dynamic_range_bandwidth : float or None
        Finite and positive (Hz), or ``None``.
    telescope_name : str
        One of ``FIND_CONTINUUM_ULTRA_TELESCOPES``.
    single_continuum : bool
        Whether the window is a single-continuum setup.
    primary_beam_limit : float
        Finite, in ``(0, 1)``: findContinuum's annulus levels need a
        positive level (fC 9603-9605).
    first_round_store : str or None
        Path of a first-round store, or ``None``.
    n_mapping_parallelism : mapping or None
        ``{"frequency": n}`` with at least one chunk, or ``None``.
    thread_info : mapping or None
        ``n_threads`` (at least 1) and ``memory_per_thread`` (finite and
        positive, GiB), or ``None``.
    memory_budget_gb, monitor_resources_seconds : float or None
        Finite and positive, or ``None``.

    Raises
    ------
    TypeError
        A value of another type.
    ValueError
        A value outside its range.
    """
    require_integer("n_bin", n_bin, minimum=1)
    if spectral_dynamic_range_bandwidth is not None:
        require_finite_real(
            "spectral_dynamic_range_bandwidth",
            spectral_dynamic_range_bandwidth,
            positive=True,
            scalar=True,
        )
    if telescope_name not in FIND_CONTINUUM_ULTRA_TELESCOPES:
        raise ValueError(
            f"telescope_name must be one of {FIND_CONTINUUM_ULTRA_TELESCOPES}, not "
            f"{telescope_name!r}."
        )
    require_boolean("single_continuum", single_continuum)
    require_finite_real("primary_beam_limit", primary_beam_limit, scalar=True)
    if not 0.0 < primary_beam_limit < 1.0:
        raise ValueError("primary_beam_limit must be in (0, 1).")
    if first_round_store is not None and not isinstance(first_round_store, str):
        raise TypeError("first_round_store must be a path (str) or None.")
    _check_frequency_chunks(n_mapping_parallelism)
    if thread_info is not None:
        require_mapping("thread_info", thread_info)
        require_integer(
            'thread_info["n_threads"]', thread_info.get("n_threads"), minimum=1
        )
        require_finite_real(
            'thread_info["memory_per_thread"]',
            thread_info.get("memory_per_thread"),
            positive=True,
            scalar=True,
        )
    for name, value in (
        ("memory_budget_gb", memory_budget_gb),
        ("monitor_resources_seconds", monitor_resources_seconds),
    ):
        if value is not None:
            require_finite_real(name, value, positive=True, scalar=True)


def _check_frequency_chunks(n_mapping_parallelism):
    """``{"frequency": n}`` with at least one chunk, or ``None``."""
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


def check_find_continuum_first_round(
    image_params, output_image_format, *, image_data_variables_keep=None
):
    """Require the Stokes I plane of which the imager makes the first round,
    in a Zarr image.

    Parameters
    ----------
    image_params : mapping
        Image parameters of ``image_cube_single_field``.
    output_image_format : str
        Output format of ``image_cube_single_field``; find_continuum reads
        Zarr image stores only.
    image_data_variables_keep : sequence of str or None, optional
        Explicit retained products. When supplied, must include primary_beam
        because find_continuum reads the saved beam.
    """
    if (
        image_data_variables_keep is not None
        and "primary_beam" not in image_data_variables_keep
    ):
        raise ValueError(
            "find_continuum_first_round requires primary_beam in image_data_variables_keep."
        )
    if "I" not in list(image_params["polarization_coords"]):
        raise ValueError(
            "find_continuum_first_round needs Stokes I in "
            "image_params['polarization_coords']."
        )
    if output_image_format != "zarr":
        raise ValueError(
            "find_continuum_first_round needs output_image_format 'zarr', not "
            f"{output_image_format!r}: find_continuum reads Zarr image stores."
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


def check_find_continuum_ultra_inputs(
    img_xds,
    image_data_group_in_name,
    channel_range,
    atmospheric_transmission,
    atmospheric_sky_temperature,
):
    """Require of the opened image what find_continuum_ultra reads: the cubes
    of the data group on the image dimensions with one time and Stokes I, a
    spectral window of increasing frequencies and at least
    ``FIND_CONTINUUM_ULTRA_MIN_CHANNELS`` channels, finite atmosphere spectra
    without a zero, finite positive beams of Stokes I, the rest frequency and
    two pixels along ``l`` and ``m``.

    Parameters
    ----------
    img_xds : xarray.Dataset
        The opened image, every channel of the cube.
    image_data_group_in_name : str
        Data group with the roles of ``FIND_CONTINUUM_ROLES``.
    channel_range : sequence of int or None
        ``[start, stop)`` of the spectral window in channels of the cube
        (:func:`check_find_continuum_channel_range`); ``None``: every
        channel.
    atmospheric_transmission, atmospheric_sky_temperature : array_like
        One value per channel of the window.

    Returns
    -------
    list of int
        ``[start, stop)`` of the window.

    Raises
    ------
    TypeError
        An atmosphere spectrum of values that are not real.
    ValueError
        The data group is missing or lacks a role, a cube is not on
        ``(time, frequency, polarization, l, m)``, the image has more than one
        time or no Stokes I, the window is refused by
        :func:`check_find_continuum_channel_range`, has too few channels or
        frequencies that do not increase (the bandwidth warning divides by
        the signed increment, fC 3750-3752), an atmosphere spectrum has
        another length, a non-finite value or a zero (its fit divides by its
        values, fC 4465-4488), a beam axis of Stokes I in the window is not
        finite and positive, the rest frequency is missing, or ``l`` or ``m``
        has fewer than two pixels (fC 7410-7417).

    Notes
    -----
    Reads the beam fits of the window besides metadata; no plane of a cube.
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
    for role in ("sky", "primary_beam"):
        dims = img_xds[data_group[role]].dims
        if dims != _IMAGE_DIMS:
            raise ValueError(
                f"{data_group[role]} has the dimensions {dims}, not {_IMAGE_DIMS}."
            )
    if img_xds.sizes["time"] != 1:
        raise ValueError("find_continuum_ultra needs an image with one time.")
    if "I" not in img_xds.polarization.values:
        raise ValueError("find_continuum_ultra needs a Stokes I plane.")
    start, stop = check_find_continuum_channel_range(
        channel_range, img_xds.frequency.values
    )
    n_channels = stop - start
    if n_channels < FIND_CONTINUUM_ULTRA_MIN_CHANNELS:
        raise ValueError(
            f"The spectral window [{start}, {stop}) has {n_channels} channels: "
            f"findContinuum needs at least {FIND_CONTINUUM_ULTRA_MIN_CHANNELS}, for "
            "two baseline channels at each edge."
        )
    if not np.all(np.diff(img_xds.frequency.values[start:stop]) > 0):
        raise ValueError(
            f"The frequencies of the spectral window [{start}, {stop}) must increase."
        )
    for name, spectrum in (
        ("atmospheric_transmission", atmospheric_transmission),
        ("atmospheric_sky_temperature", atmospheric_sky_temperature),
    ):
        if np.shape(spectrum) != (n_channels,):
            raise ValueError(
                f"{name} must have one value per channel of the spectral window "
                f"({n_channels})."
            )
        if np.any(require_finite_real(name, spectrum) == 0):
            raise ValueError(f"{name} must have no zero.")
    beams = (
        img_xds[data_group["beam_fit_params_point_spread_function"]]
        .sel(polarization="I")
        .isel(frequency=slice(start, stop))
        .values[..., :2]
    )
    if not np.all(np.isfinite(beams) & (beams > 0)):
        raise ValueError(
            "The beam axes of Stokes I must be finite and positive in every channel "
            f"of the spectral window [{start}, {stop})."
        )
    if "rest_frequency" not in img_xds.frequency.attrs:
        raise ValueError("The frequency coordinate has no rest_frequency attribute.")
    if img_xds.sizes["l"] < 2 or img_xds.sizes["m"] < 2:
        raise ValueError("find_continuum_ultra needs two pixels along l and m.")
    return [start, stop]


def check_find_continuum_channel_range(channel_range, frequency):
    """The channels of the cube that findContinuum sees, on one linear axis.

    findContinuum describes its cube by a linear frequency axis, so it runs on
    one spectral window of the cube. The check also runs without a range, so
    that a cube of several windows is refused instead of being analysed on a
    wrong frequency axis.

    Parameters
    ----------
    channel_range : sequence of int or None
        ``[start, stop)``, channels of the cube; ``None``: every channel.
    frequency : array_like
        Frequency of every channel of the cube.

    Returns
    -------
    list of int
        ``[start, stop)``.

    Raises
    ------
    ValueError
        The range is empty or outside the cube, or its frequencies are not
        linear by
        :func:`~astroviper.processing_functions.imaging.utils.spectral_windows.linear_channel_ranges`
        (the message lists the linear channel ranges of the cube).
    """
    from astroviper.processing_functions.imaging.utils.spectral_windows import (
        linear_channel_ranges,
    )

    frequency = np.asarray(frequency)
    if channel_range is None:
        start, stop = 0, frequency.size
    else:
        if len(channel_range) != 2:
            raise ValueError("channel_range must be [start, stop].")
        start, stop = channel_range
        require_integer("channel_range[0]", start)
        require_integer("channel_range[1]", stop)
    if not 0 <= start < stop <= frequency.size:
        raise ValueError(
            f"channel_range [{start}, {stop}) is empty or outside the "
            f"{frequency.size} channels of the cube."
        )
    if len(linear_channel_ranges(frequency[start:stop])) != 1:
        raise ValueError(
            f"The frequencies of channels [{start}, {stop}) of the cube are not "
            f"linear. Its linear channel ranges are "
            f"{linear_channel_ranges(frequency)}: pass one of them as channel_range."
        )
    return [int(start), int(stop)]


def check_find_continuum_first_round_store(
    attrs,
    image_binding,
    primary_beam_limit,
    polarization_index,
    tile_shape,
    variables,
    channel_range,
):
    """Require a first-round store complete for a spectral window and made from
    the cube as it is now.

    Parameters
    ----------
    attrs : mapping or None
        Attributes of the first-round store (its ``find_continuum_first_round``
        and ``image_binding`` entries are checked), ``None`` if the store does
        not exist.
    image_binding : mapping
        The binding of the cube now: ``image_store_zarr_json_mtime_ns``,
        ``frequency_sha1``, ``l_sha1``, ``m_sha1`` and ``shape``.
    primary_beam_limit : float
        Primary beam limit of the find_continuum call.
    polarization_index : int
        Index of Stokes I in the cube.
    tile_shape : sequence of int
        Inner chunk shape of the cube along ``l`` and ``m``.
    variables : mapping
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the
        find_continuum call.
    channel_range : sequence of int
        ``[start, stop)``, channels of the cube that findContinuum sees.

    Raises
    ------
    ValueError
        The store is missing or incomplete, has another version (a store of an
        earlier version must be made again), was made with another primary beam
        limit, Stokes index, tile shape or data variables, holds no window
        with every channel of ``channel_range``, or a field of its binding
        differs from the cube's: the cube was rewritten, or is another one.
    """
    first_round = None if attrs is None else attrs.get("find_continuum_first_round")
    if first_round is None:
        raise ValueError("The find_continuum first-round store is missing.")
    if first_round.get("version") != FIND_CONTINUUM_FIRST_ROUND_VERSION:
        raise ValueError(
            f"The first-round store has version {first_round.get('version')}, "
            f"not {FIND_CONTINUUM_FIRST_ROUND_VERSION}: make the first round again."
        )
    if not first_round.get("complete"):
        raise ValueError("The first-round store is incomplete.")
    for name, value in (
        ("primary_beam_limit", primary_beam_limit),
        ("polarization_index", polarization_index),
        ("tile_shape", list(tile_shape)),
        ("variables", dict(variables)),
    ):
        if _plain(first_round.get(name)) != value:
            raise ValueError(
                f"The first-round store was made with {name} "
                f"{first_round.get(name)}, not {value}."
            )
    start, stop = channel_range
    ranges = [window["channel_range"] for window in first_round["windows"]]
    if not any(first <= start and stop <= last for first, last in ranges):
        raise ValueError(
            f"The first-round store holds channels {ranges} of the cube, not "
            f"[{start}, {stop})."
        )
    binding = attrs.get("image_binding") or {}
    for name in sorted(set(binding) | set(image_binding)):
        if _plain(binding.get(name)) != _plain(image_binding.get(name)):
            raise ValueError(
                "The first-round store was not made from the cube as it is now "
                f"({name} differs): make the first round again."
            )


def _plain(value):
    return list(value) if isinstance(value, list | tuple) else value


# ========================================================================
# Planned node tasks
# ========================================================================


def check_find_continuum_ultra_task_config(
    planes_per_batch, polarization_index, channel_range
):
    """Validate the configuration a find_continuum_ultra frequency task gets.

    Parameters
    ----------
    planes_per_batch : int
        Planes a read spans at most, at least 1.
    polarization_index : int
        Index of Stokes I along ``polarization``, at least 0.
    channel_range : sequence of int
        ``[start, stop)`` of the spectral window in channels of the cube,
        ``0 <= start < stop``.

    Raises
    ------
    TypeError
        A value that is not an integer.
    ValueError
        A value outside its range.
    """
    require_integer("planes_per_batch", planes_per_batch, minimum=1)
    require_integer("polarization_index", polarization_index)
    if len(channel_range) != 2:
        raise ValueError("channel_range must be [start, stop].")
    start, stop = channel_range
    require_integer("channel_range[0]", start)
    require_integer("channel_range[1]", stop)
    if not start < stop:
        raise ValueError(f"channel_range [{start}, {stop}) is empty.")


def check_find_continuum_ultra_task_memory(task_bytes, memory_per_thread_bytes):
    """Require that the smallest find_continuum_ultra node task of a reduction
    fits the memory of a thread.

    A task between its budget and the memory of a thread runs, at one plane
    per batch; only a task that cannot fit a thread at all is refused.

    Parameters
    ----------
    task_bytes : int
        Memory of a node task that computes one requested reduction (or the
        first round), one plane per batch.
    memory_per_thread_bytes : int
        Memory of a thread of the cluster.

    Raises
    ------
    ValueError
        ``task_bytes`` exceeds ``memory_per_thread_bytes``.
    """
    if task_bytes > memory_per_thread_bytes:
        raise ValueError(
            f"A find_continuum_ultra node task needs {task_bytes / 2**30:.2f} GiB "
            "for one reduction at one plane per batch, more than the "
            f"{memory_per_thread_bytes / 2**30:.2f} GiB of memory per thread: "
            "run it with fewer threads per worker or more memory per worker "
            "(thread_info['memory_per_thread'])."
        )


def check_find_continuum_cluster_threads(thread_info):
    """Require threads with a memory limit of the cluster find_continuum_ultra
    runs on, when it takes them from the cluster.

    Parameters
    ----------
    thread_info : mapping
        ``n_threads`` and ``memory_per_thread`` (GiB) as
        :func:`~astroviper.utils.data_partitioning.get_thread_info` returns
        them.

    Raises
    ------
    ValueError
        No thread, or no finite positive memory per thread.
    """
    n_threads = thread_info.get("n_threads")
    memory_per_thread = thread_info.get("memory_per_thread")
    if not (
        isinstance(n_threads, int | np.integer)
        and n_threads >= 1
        and isinstance(memory_per_thread, int | float | np.number)
        and np.isfinite(memory_per_thread)
        and memory_per_thread > 0
    ):
        raise ValueError(
            f"The cluster reports {n_threads} threads of {memory_per_thread} GiB "
            "(no worker connected yet, or workers without a memory limit): pass "
            "thread_info."
        )
