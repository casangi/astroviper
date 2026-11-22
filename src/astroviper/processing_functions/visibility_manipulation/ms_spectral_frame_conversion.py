import warnings

import astropy.units as u
import numpy as np
import xarray as xr
import xradio.measurement_set  # noqa: F401 -- registers the xr_ms accessor
from astropy.coordinates import (
    ICRS,
    LSRD,
    LSRK,
    CartesianDifferential,
    CartesianRepresentation,
    EarthLocation,
    SkyCoord,
    SpectralCoord,
)
from astropy.coordinates.spectral_coordinate import NoDistanceWarning, NoVelocityWarning
from astropy.time import Time

# Output frames by their XRADIO name (``frequency.attrs["observer"]``).
_OUTPUT_FRAMES = {"lsrk": LSRK, "lsrd": LSRD, "BARY": ICRS}


def ms_spectral_frame_conversion(
    ms: xr.DataTree,
    freqrange: list[float] | None = None,
    outframe: str = "lsrk",
    *,
    output_frequency: np.ndarray | None = None,
) -> xr.DataTree:
    """Regrid a measurement set from TOPO onto a fixed frequency grid in another frame.

    At each time the frequency of every channel in ``outframe`` is its TOPO
    frequency times one Doppler factor (:func:`doppler_factor`). Each variable
    with a frequency dimension is interpolated linearly onto the output grid
    from the two input channels that bracket an output channel, as CASA
    ``mstransform`` does (``regridms=True``, ``interpolation="linear"``):

    - correlated data: ``v0 + w (v1 - v0)``;
    - weights: interpolated the same way. CASA also multiplies them by the
      ratio of the output to the input channel width, which is 1 on its
      default grid; here the channels are interpolated, not averaged, so
      the weights are not scaled;
    - flags: an output sample is flagged when an input sample it is
      interpolated from is flagged.

    Output channels outside the input frequencies of a time are flagged, with
    zero data and weight. An output channel depends only on the input channels
    that bracket it, so blocks of output channels can be computed separately
    from the input channels they need, given one output grid for the whole
    measurement set.

    Parameters
    ----------
    ms : xarray.DataTree
        Measurement set (MSv4) with a TOPO frequency axis in Hz, strictly
        increasing or decreasing. The variables with a frequency dimension
        must have the ``correlated_data``, ``flag`` or ``weight`` role in a
        data group.
    freqrange : list of float, optional
        Not implemented. Select channels with ``ms.sel(frequency=...)`` or
        pass ``output_frequency``.
    outframe : {"lsrk", "lsrd", "BARY"}, default "lsrk"
        Spectral frame of the output, as XRADIO names it.
    output_frequency : numpy.ndarray, optional
        Frequencies of the output channels in ``outframe`` (Hz). Default:
        :func:`outframe_freq`.

    Returns
    -------
    xarray.DataTree
        The measurement set on the output grid, with ``observer`` and
        ``channel_width`` of the frequency coordinate updated. The variables
        without a frequency dimension and the child nodes share their data
        with ``ms``.

    Raises
    ------
    NotImplementedError
        If ``freqrange`` is given.
    ValueError
        If the frequency axis is not TOPO, not in Hz, has fewer than two
        channels or is not strictly monotonic; if ``outframe`` is not
        supported; if ``output_frequency`` is not a 1-D array of finite
        values; or if a variable with a frequency dimension has none of the
        roles above.

    See Also
    --------
    doppler_factor : Doppler factor TOPO -> ``outframe`` at every time.
    outframe_freq : Default output grid.
    """
    if freqrange:
        raise NotImplementedError(
            "freqrange is not implemented; select channels with "
            "ms.sel(frequency=...) or pass output_frequency."
        )
    frequency_attrs = ms.frequency.attrs
    if frequency_attrs.get("observer") != "TOPO":
        raise ValueError(
            f"The frequency axis must be TOPO, not {frequency_attrs.get('observer')!r}."
        )
    if frequency_attrs.get("units") != "Hz":
        raise ValueError(
            f"The frequency axis must be in Hz, not {frequency_attrs.get('units')!r}."
        )
    input_frequency = ms.frequency.values
    if input_frequency.size < 2:
        raise ValueError("Linear interpolation needs at least two channels.")
    steps = np.diff(input_frequency)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError(
            "The frequency axis must be strictly increasing or decreasing."
        )
    dataset = ms.to_dataset()
    roles = _frequency_roles(dataset)

    doppler = doppler_factor(ms, outframe)
    if output_frequency is None:
        output_frequency = _uniform_grid(input_frequency, doppler)
    else:
        output_frequency = np.asarray(output_frequency, dtype=float)
        if (
            output_frequency.ndim != 1
            or output_frequency.size == 0
            or not np.all(np.isfinite(output_frequency))
        ):
            raise ValueError(
                "output_frequency must be a non-empty 1-D array of finite frequencies."
            )

    if output_frequency.size > 1:
        output_channel_width = np.abs(np.diff(output_frequency)).mean()
    else:
        output_channel_width = (
            abs(frequency_attrs["channel_width"]["data"]) * doppler.mean()
        )

    regridded = _regrid(
        {name: dataset[name] for name in roles},
        roles,
        input_frequency,
        doppler,
        output_frequency,
    )

    frequency = xr.Variable(
        "frequency",
        output_frequency,
        {
            **frequency_attrs,
            "observer": outframe,
            "channel_width": {
                **frequency_attrs["channel_width"],
                "data": float(output_channel_width),
            },
        },
    )
    converted = ms.copy()
    converted.dataset = (
        dataset.drop_vars([*roles, "frequency"])
        .assign_coords(frequency=frequency)
        .assign(regridded)
    )
    return converted


def doppler_factor(ms: xr.DataTree, outframe: str = "lsrk") -> np.ndarray:
    """Ratio of the frequency in ``outframe`` to the TOPO frequency at every time.

    The ratio does not depend on the frequency (to 1e-16), so one astropy
    call gives it for all channels and times. The observer is the telescope
    site at rest on the Earth and the target is the phase center of the
    first field, which is also what CASA ``mstransform`` uses by default
    (``phasecenter=""``).

    Parameters
    ----------
    ms : xarray.DataTree
        Measurement set (MSv4).
    outframe : {"lsrk", "lsrd", "BARY"}, default "lsrk"
        Spectral frame, as XRADIO names it.

    Returns
    -------
    numpy.ndarray
        Doppler factor, shape ``(time,)``.

    Raises
    ------
    ValueError
        If ``outframe`` is not supported.
    """
    rest_frame = frame_from_str(outframe)
    observer = get_all_itrs_loc(ms)
    n_time = observer.shape[0]
    at_rest = CartesianDifferential(np.zeros((3, n_time)) * u.m / u.s)
    observer = observer.realize_frame(observer.cartesian.with_differentials(at_rest))
    # with_observer_stationary_relative_to cannot broadcast a scalar frame
    # against an array observer, so the frame gets the shape of the observer.
    rest = rest_frame(
        CartesianRepresentation(np.zeros((3, n_time)) * u.m).with_differentials(at_rest)
    )
    with warnings.catch_warnings():
        # The phase center has no distance and no velocity: SpectralCoord puts
        # it far away at rest, which is the intent.
        warnings.simplefilter("ignore", NoDistanceWarning)
        warnings.simplefilter("ignore", NoVelocityWarning)
        topo = SpectralCoord(
            np.ones(n_time) * u.Hz,
            observer=observer,
            target=get_phase_center(ms, ms.field_name.values[0]),
        )
        return topo.with_observer_stationary_relative_to(rest).to_value(u.Hz)


def outframe_freq(ms: xr.DataTree, outframe: str = "lsrk") -> np.ndarray:
    """Default output grid: uniform, covering the input band in ``outframe`` at all times.

    The grid has as many channels as the input, from the lowest frequency of
    the band in ``outframe`` over all times, with a width of the converted
    band divided by the number of channels.

    Parameters
    ----------
    ms : xarray.DataTree
        Measurement set (MSv4) with a TOPO frequency axis in Hz.
    outframe : {"lsrk", "lsrd", "BARY"}, default "lsrk"
        Spectral frame, as XRADIO names it.

    Returns
    -------
    numpy.ndarray
        Output channel frequencies in Hz.
    """
    return _uniform_grid(ms.frequency.values, doppler_factor(ms, outframe))


def get_phase_center(ms: xr.DataTree, fieldname: str) -> SkyCoord:
    """Phase center of a field of the measurement set.

    Parameters
    ----------
    ms : xarray.DataTree
        Measurement set (MSv4).
    fieldname : str
        Field name.

    Returns
    -------
    astropy.coordinates.SkyCoord
        Phase center direction.
    """
    pc = ms.xr_ms.get_field_and_source_xds().FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=fieldname
    )
    return SkyCoord(
        pc.sel(sky_dir_label="ra").data * u.Unit(pc.units),
        pc.sel(sky_dir_label="dec").data * u.Unit(pc.units),
        frame=pc.attrs["frame"],
    )


def frame_from_str(framestr: str) -> type:
    """Astropy frame of an XRADIO spectral frame name.

    Parameters
    ----------
    framestr : {"lsrk", "lsrd", "BARY"}
        Spectral frame, as XRADIO names it.

    Returns
    -------
    type
        Astropy coordinate frame class whose origin is at rest in that frame.

    Raises
    ------
    ValueError
        If the frame is not supported.
    """
    if framestr not in _OUTPUT_FRAMES:
        raise ValueError(
            f"Unsupported spectral frame {framestr!r}; use one of {list(_OUTPUT_FRAMES)}."
        )
    return _OUTPUT_FRAMES[framestr]


def get_all_itrs_loc(ms: xr.DataTree):
    """Telescope site in ITRS at every time of the measurement set.

    Parameters
    ----------
    ms : xarray.DataTree
        Measurement set (MSv4).

    Returns
    -------
    astropy.coordinates.ITRS
        Site position, shape ``(time,)``.
    """
    obsstr = ms["antenna_xds"].attrs["overall_telescope_name"]
    if obsstr == "EVLA":
        obsstr = "VLA"
    elif obsstr == "ATA":
        obsstr = "ALMA"

    obs = EarthLocation.of_site(obsstr)
    obs_t = ms["time"].data * u.Unit(ms["time"].attrs["units"])
    t = Time(
        obs_t,
        format=ms["time"].attrs["format"],
        scale=ms["time"].attrs["scale"],
    )
    return obs.get_itrs(obstime=t)


def _uniform_grid(input_frequency: np.ndarray, doppler: np.ndarray) -> np.ndarray:
    lowest = input_frequency.min() * doppler.min()
    highest = input_frequency.max() * doppler.max()
    n_channel = input_frequency.size
    return np.arange(n_channel) * ((highest - lowest) / n_channel) + lowest


def _frequency_roles(dataset: xr.Dataset) -> dict[str, str]:
    """Data-group role of every data variable with a frequency dimension."""
    role_of = {}
    for data_group in dataset.attrs.get("data_groups", {}).values():
        for role in ("correlated_data", "flag", "weight"):
            if role in data_group:
                role_of[data_group[role]] = role
    roles = {}
    for name, variable in dataset.data_vars.items():
        if "frequency" not in variable.dims:
            continue
        if name not in role_of:
            raise ValueError(
                f"{name} has a frequency dimension but no correlated_data, flag or "
                "weight role in a data group, so it cannot be regridded."
            )
        if variable.dims[0] != "time":
            raise ValueError(f"{name} must have time as its first dimension.")
        roles[name] = role_of[name]
    for name, coordinate in dataset.coords.items():
        if name != "frequency" and "frequency" in coordinate.dims:
            raise ValueError(
                f"Coordinate {name} on the frequency axis is not supported."
            )
    return roles


def _bracketing_channels(
    ascending_frequency: np.ndarray,
    increasing: bool,
    doppler: float,
    output_frequency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Input channels bracketing each output channel at one time.

    Returns the channels just below and above each output channel (indices
    into the input axis), the interpolation weight of the channel above, and
    whether the output channel lies within the input frequencies.
    """
    n_channel = ascending_frequency.size
    position = output_frequency / doppler
    lower = np.searchsorted(ascending_frequency, position, side="right") - 1
    np.clip(lower, 0, n_channel - 2, out=lower)
    below = ascending_frequency[lower]
    weight = (position - below) / (ascending_frequency[lower + 1] - below)
    inside = (position >= ascending_frequency[0]) & (
        position <= ascending_frequency[-1]
    )
    if increasing:
        return lower, lower + 1, weight, inside
    return n_channel - 1 - lower, n_channel - 2 - lower, weight, inside


def _regrid(
    variables: dict[str, xr.DataArray],
    roles: dict[str, str],
    input_frequency: np.ndarray,
    doppler: np.ndarray,
    output_frequency: np.ndarray,
    block_bytes: int = 2**28,
) -> dict[str, xr.Variable]:
    """Interpolate every variable onto the output grid, one time at a time.

    The input is read in blocks of times of about ``block_bytes``, so a lazily
    opened measurement set is never loaded whole.
    """
    increasing = bool(input_frequency[-1] > input_frequency[0])
    ascending_frequency = input_frequency if increasing else input_frequency[::-1]
    n_output = output_frequency.size
    outputs, axes, shapes, buffers = {}, {}, {}, {}
    bytes_per_time, time_chunk = 0, 1
    for name, variable in variables.items():
        frequency_axis = variable.get_axis_num("frequency")
        shape = list(variable.shape)
        shape[frequency_axis] = n_output
        outputs[name] = np.empty(shape, dtype=variable.dtype)
        buffers[name] = [np.empty(shape[1:], dtype=variable.dtype) for _ in range(2)]
        axes[name] = frequency_axis - 1
        broadcast = [1] * (variable.ndim - 1)
        broadcast[frequency_axis - 1] = n_output
        shapes[name] = broadcast
        bytes_per_time += variable.nbytes // variable.shape[0]
        time_chunk = max(
            time_chunk, variable.encoding.get("preferred_chunks", {}).get("time", 1)
        )
    block = max(time_chunk, block_bytes // bytes_per_time // time_chunk * time_chunk)

    for start in range(0, doppler.size, block):
        inputs = {
            name: variable[start : start + block].values
            for name, variable in variables.items()
        }
        for t in range(start, min(start + block, doppler.size)):
            lower, upper, weight, inside = _bracketing_channels(
                ascending_frequency, increasing, doppler[t], output_frequency
            )
            # Usually empty or a few edge channels.
            outside = np.flatnonzero(~inside)
            only_lower = np.flatnonzero(weight == 0)
            only_upper = np.flatnonzero(weight == 1)
            for name, role in roles.items():
                values, out = inputs[name][t - start], outputs[name][t]
                axis, (below, above) = axes[name], buffers[name]
                np.take(values, lower, axis=axis, out=below, mode="clip")
                np.take(values, upper, axis=axis, out=above, mode="clip")
                index = [slice(None)] * out.ndim
                if role == "flag":
                    np.logical_or(below, above, out=out)
                    for channels, source in ((only_lower, below), (only_upper, above)):
                        index[axis] = channels
                        out[tuple(index)] = source[tuple(index)]
                    index[axis] = outside
                    out[tuple(index)] = True
                    continue
                above -= below
                above *= weight.astype(out.real.dtype).reshape(shapes[name])
                np.add(below, above, out=out)
                index[axis] = outside
                out[tuple(index)] = 0

    return {
        name: xr.Variable(variables[name].dims, outputs[name], variables[name].attrs)
        for name in roles
    }
