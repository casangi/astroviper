from astroviper.utils.param_docs import shares_param_docs
from astroviper.utils.task_logging import (
    write_task_kill_switch_log as _write_task_kill_switch_log,
)


def _log_task_io_failure(phase, exc, task_id, image_store, data_selection, task_coords):
    """Log a NON-FATAL node-task I/O failure; return its timing-row marker columns.

    A chunk whose data cannot be read or whose result cannot be written (e.g. a
    Lustre client eviction that outlives the reader/writer retry schedules, or a
    corrupt input shard) must not abort the whole multi-node run: the caller
    logs it here, marks the task's row in the timing frame with these columns
    (``task_failed_phase`` / ``task_error`` / ``failed_channel_start`` /
    ``failed_n_channels``), and the run continues with the failed channels left
    at the image store's fill value. ``task_coords`` is the whole task's for a
    load failure and one on-disk chunk's (see :func:`_chunk_task_coords`) for a
    write failure. Failures stay queryable per run from the saved node-task
    frame.
    """
    import socket

    import toolviper.utils.logger as logger

    hostname = socket.gethostname()
    frequency = task_coords["frequency"]
    frequency_slice = frequency.get("slice")
    if isinstance(frequency_slice, slice) and frequency_slice.start is not None:
        chan_start = int(frequency_slice.start)
    else:
        chan_start = _global_channel_offset(data_selection)
    n_channels = len(frequency["data"])
    logger.error(
        f"node task {task_id} on {hostname}: {phase} FAILED for channels "
        f"[{chan_start}, {chan_start + n_channels}) of {image_store}; skipping "
        "them and continuing the run. Error: "
        f"{exc!r}"
    )
    return {
        "hostname": hostname,
        "task_id": task_id,
        "n_channels": n_channels,
        "task_failed_phase": phase,
        "task_error": repr(exc)[:500],
        "failed_channel_start": chan_start,
        "failed_n_channels": n_channels,
    }


def _global_channel_offset(data_selection):
    """Global channel number of this task's first channel: the start of the
    ``frequency`` slice in ``data_selection`` (frequency and channel are the
    same axis), e.g. ``{'ms_name': {'frequency': slice(2, 4)}}`` -> 2; 0 when
    no frequency slice is present."""
    for sel in (data_selection or {}).values():
        freq_sel = sel.get("frequency") if isinstance(sel, dict) else None
        if isinstance(freq_sel, slice) and freq_sel.start is not None:
            return int(freq_sel.start)
    return 0


def _remap_imaging_dict_to_global_channels(combined_imaging_dict, data_selection):
    """Shift a chunk-local deconvolve ImagingDict onto global channel numbers.

    The per-chunk ``image_cube_single_field`` labels channels ``0..N-1`` within
    the chunk. The global channel offset for this chunk is the start of the
    ``frequency`` slice in ``data_selection`` (frequency and channel are the
    same axis), e.g. ``{'ms_name': {'frequency': slice(2, 4)}}`` -> offset 2.

    Returns a new ImagingDict whose ``Key.chan`` values are global channel
    numbers. A no-op returning the input unchanged when the offset is 0 (e.g. a
    single chunk starting at channel 0) or no frequency slice is present.
    """

    return _shift_imaging_dict_channels(
        combined_imaging_dict, _global_channel_offset(data_selection)
    )


def _shift_imaging_dict_channels(imaging_dict, chan_offset):
    """Return ``imaging_dict`` with every ``Key.chan`` shifted by ``chan_offset``.

    The science function labels the channels of the cube it is handed
    ``0..N-1``; the node task images one channel at a time, so each
    per-channel dict comes back with ``chan == 0`` and is shifted onto its
    chunk-local channel here (and onto the global channel number afterwards by
    :func:`_remap_imaging_dict_to_global_channels`). Returns the input itself
    for a zero offset.
    """
    from astroviper.processing_functions.imaging.utils.imaging_dict import (
        ImagingDict,
        Key,
    )

    if chan_offset == 0:
        return imaging_dict

    shifted = ImagingDict()
    for key, value in imaging_dict.data.items():
        shifted.data[Key(time=key.time, pol=key.pol, chan=key.chan + chan_offset)] = (
            value
        )
    return shifted


def _visibility_to_image_frequency_maps(ps_xdt, img_xds):
    """Map every measurement set's visibility channels onto the chunk's image
    channels (``{ms_name: int array of image channel indices}``), with the same
    nearest-channel rule the gridders apply, so the per-channel loop hands the
    science function exactly the visibility channels it would grid onto that
    image channel.
    """
    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        map_visibility_frequencies_to_image,
    )

    return {
        ms_name: map_visibility_frequencies_to_image(
            ms_xdt.frequency.values, img_xds.frequency.values
        )
        for ms_name, ms_xdt in ps_xdt.items()
    }


def _select_processing_set_channel(ps_xdt, frequency_maps, chan_index):
    """Slice the loaded chunk down to the visibility channels that map onto
    image channel ``chan_index``.

    Returns ``{ms_name: measurement-set node}`` holding zero-copy views of the
    loaded arrays (the mapped visibility channels of a measurement set are a
    contiguous run, so the frequency selection is a basic slice), each with its
    own deep-copied ``attrs`` so the data groups and variables the processing
    functions register (``WEIGHT_IMAGING``, ``VISIBILITY_MODEL``,
    ``VISIBILITY_RESIDUAL`` and their data groups) never leak between channels
    or back into the loaded chunk. Measurement sets without a visibility
    channel on this image channel are left out; ``None`` when none remain.
    """
    import copy

    import numpy as np

    selected = {}
    for ms_name, ms_xdt in ps_xdt.items():
        vis_chans = np.flatnonzero(frequency_maps[ms_name] == chan_index)
        if vis_chans.size == 0:
            continue
        first, last = int(vis_chans[0]), int(vis_chans[-1])
        if last - first + 1 == vis_chans.size:
            indexer = slice(first, last + 1)  # contiguous run: a view, no copy
        else:
            indexer = vis_chans  # not expected; fancy indexing copies
        ms_chan = ms_xdt.isel(frequency=indexer)
        ms_chan.attrs = copy.deepcopy(ms_xdt.attrs)
        selected[ms_name] = ms_chan
    return selected or None


def _whole_processing_set_channel(ps_xdt):
    """The whole loaded chunk, for an image channel no visibility channel maps
    onto with the chunk's image axis.

    The science function then maps the chunk's visibility channels onto that
    one image channel with its own rule (see
    :func:`~astroviper.processing_functions.imaging.utils.frequency_mapping.map_visibility_frequencies_to_image`):
    a visibility channel within half a visibility channel spacing of it, or
    the only visibility channel of the chunk at any distance, is gridded onto
    it, and one farther away raises ``ValueError``. So, unlike in a call with
    the whole image cube, where such a channel stays empty, a chunk of one
    visibility channel is imaged onto it.

    Like :func:`_select_processing_set_channel`, returns ``{ms_name:
    measurement-set node}`` of zero-copy views of the loaded arrays, each with
    its own deep-copied ``attrs``, so the variables and data groups the
    processing functions register (``WEIGHT_IMAGING``, ``VISIBILITY_MODEL``,
    ``VISIBILITY_RESIDUAL`` and their data groups) stay off the loaded chunk
    and die with the channel. Handing over the loaded chunk itself would
    register them on it, and the next channel's views would carry them into
    the science function's no-overwrite checks (``AssertionError: Output data
    variable WEIGHT_IMAGING already exists``).
    """
    import copy

    selected = {}
    for ms_name, ms_xdt in ps_xdt.items():
        ms_chan = ms_xdt.isel(frequency=slice(None))
        ms_chan.attrs = copy.deepcopy(ms_xdt.attrs)
        selected[ms_name] = ms_chan
    return selected


def _select_image_channel(img_xds, chan_index):
    """One-channel slice of the empty chunk image with its own ``attrs`` copy
    (the science function registers data groups on it in place)."""
    import copy

    img_chan = img_xds.isel(frequency=slice(chan_index, chan_index + 1))
    img_chan.attrs = copy.deepcopy(img_xds.attrs)
    return img_chan


class _ImageChunkAccumulator:
    """Gather consecutive per-channel science results into one on-disk
    frequency chunk of ``n_channels`` channels starting at chunk-local channel
    ``start``.

    A one-channel chunk *is* the science result: nothing is allocated or
    copied, so a single-channel task, or ``image_chunking={"frequency": 1}``,
    adds no memory at all. A wider chunk gets one buffer per frequency-
    dependent variable (data variables and non-index coordinates such as
    ``velocity``), allocated at the chunk's channel count when the first result
    arrives, into which every channel is copied straight away; of the first
    result only the coordinates, attrs and static (frequency-independent)
    variables are kept as the template, so at most one channel's arrays are
    alive next to the chunk buffer. The ``frequency`` coordinate is the task's
    own.
    """

    def __init__(self, chunk_img_xds, start, n_channels):
        self.start = int(start)
        self.stop = self.start + int(n_channels)
        self._n = int(n_channels)
        frequency = chunk_img_xds.coords["frequency"].isel(
            frequency=slice(self.start, self.stop)
        )
        self._frequency = (("frequency",), frequency.values, dict(frequency.attrs))
        self._filled = 0
        self._single = None  # the result itself, for a one-channel chunk
        self._buffers = None  # {name: (dims, ndarray, attrs)}
        self._coord_names = None
        self._template = None  # first result minus its frequency-dependent arrays

    @property
    def complete(self):
        return self._filled == self._n

    def insert(self, channel_xds, chan_index):
        """Take chunk-local channel ``chan_index`` (consecutive) from ``channel_xds``."""
        import numpy as np

        if channel_xds.sizes.get("frequency") != 1:
            raise ValueError(
                "expected a one-channel science result, got "
                f"{channel_xds.sizes.get('frequency')} channels"
            )
        expected = self.start + self._filled
        if chan_index != expected:
            raise ValueError(
                f"channel {chan_index} arrived out of order; expected {expected}"
            )
        if self._n == 1:
            self._single = channel_xds
            self._filled = 1
            return
        if self._buffers is None:
            self._buffers = {}
            for name, var in channel_xds.variables.items():
                if name == "frequency" or "frequency" not in var.dims:
                    continue
                shape = tuple(
                    self._n if dim == "frequency" else size
                    for dim, size in zip(var.dims, var.shape, strict=True)
                )
                self._buffers[name] = (
                    var.dims,
                    np.empty(shape, dtype=var.dtype),
                    dict(var.attrs),
                )
            self._coord_names = set(channel_xds.coords)
            self._template = channel_xds.drop_vars(list(self._buffers) + ["frequency"])
        local = self._filled
        for name, (dims, buffer, _attrs) in self._buffers.items():
            if name not in channel_xds.variables:
                raise RuntimeError(
                    f"channel {chan_index} result is missing variable {name!r} "
                    "present for the chunk's first channel"
                )
            var = channel_xds.variables[name]
            if var.dims != dims:
                raise RuntimeError(
                    f"channel {chan_index} variable {name!r} has dims {var.dims}, "
                    f"the chunk's first channel had {dims}"
                )
            index = [slice(None)] * len(dims)
            index[dims.index("frequency")] = slice(local, local + 1)
            buffer[tuple(index)] = var.values
        self._filled += 1

    def assemble(self):
        """The finished chunk as one dataset (the result itself for one channel)."""
        import copy

        import xarray as xr

        if not self.complete:
            raise RuntimeError(
                f"chunk [{self.start}, {self.stop}) has {self._filled} of "
                f"{self._n} channels"
            )
        if self._n == 1:
            return self._single
        coords = {"frequency": self._frequency}
        coords.update(self._template.coords)
        data_vars = dict(self._template.data_vars)
        for name, (dims, buffer, attrs) in self._buffers.items():
            target = coords if name in self._coord_names else data_vars
            target[name] = (dims, buffer, attrs)
        return xr.Dataset(
            data_vars, coords=coords, attrs=copy.deepcopy(self._template.attrs)
        )


def _chunk_task_coords(task_coords, data_selection, local_start, local_stop):
    """``task_coords`` narrowed to chunk-local channels ``[local_start,
    local_stop)``: the ``frequency`` entry carries that range's coordinate
    values and its global ``slice`` (from the task's own slice, or the
    ``data_selection`` offset when the task carries none), which is what every
    writer uses to place the chunk in the image store."""
    import numpy as np

    frequency = dict(task_coords["frequency"])
    task_slice = frequency.get("slice")
    if isinstance(task_slice, slice) and task_slice.start is not None:
        global_start = int(task_slice.start)
    else:
        global_start = _global_channel_offset(data_selection)
    frequency["data"] = np.asarray(task_coords["frequency"]["data"])[
        local_start:local_stop
    ]
    frequency["slice"] = slice(global_start + local_start, global_start + local_stop)
    chunk_coords = dict(task_coords)
    chunk_coords["frequency"] = frequency
    return chunk_coords


def _channel_cycle_control_params(
    cycle_control_params, task_coords, data_selection, chan_index
):
    """``cycle_control_params`` of chunk-local channel ``chan_index``: a per
    channel ``threshold`` (one row of Stokes plane values per image channel)
    is replaced by the row of the channel's global number (see
    :func:`_chunk_task_coords`); any other ``threshold`` leaves the dict as
    it is."""
    import numpy as np

    threshold = cycle_control_params.get("threshold")
    if np.ndim(threshold) != 2:
        return cycle_control_params
    channel = _chunk_task_coords(
        task_coords, data_selection, chan_index, chan_index + 1
    )["frequency"]["slice"].start
    return {
        **cycle_control_params,
        "threshold": [float(value) for value in threshold[channel]],
    }


def _select_chunk_writer(
    graph_mode,
    skunk_works,
    image_sharding,
    output_image_format,
    image_store,
    image_data_variables_keep,
    processing_function_threads,
):
    """The write path for this task's finished chunks, as a callable
    ``write(chunk_xds, chunk_task_coords)``.

    A variable of ``image_data_variables_keep`` that the run does not make
    (``sky_model`` or ``mask`` of a run without deconvolution, for example)
    is skipped: only the kept variables a chunk holds are written, and the
    others keep the fill value of the image store.
    """
    if graph_mode and output_image_format == "fits":
        # FITS performance path: pwrite the chunk's contiguous channel block
        # (and its BEAMS-table rows) directly into the pre-created
        # XRADIO-conformant FITS files -- disjoint byte ranges across tasks,
        # no locking, no file creation.
        from astroviper.node_tasks.imaging.utils import (
            write_result_chunk_to_fits_skunk_works as writer,
        )
    elif graph_mode and skunk_works and image_sharding:
        # Sharded performance path: write the chunk's inner-chunk blob(s) into
        # shared, pre-created Zarr v3 shard files (far fewer files ->
        # metadata-server relief; the "single parallel file" pattern).
        from astroviper.node_tasks.imaging.utils import (
            write_result_chunk_to_disk_sharded_skunk_works as writer,
        )
    elif graph_mode and skunk_works:
        # Experimental performance path: encode and write only the chunk's
        # blob(s) directly to the pre-created Zarr image store (no open_group).
        from astroviper.node_tasks.imaging.utils import (
            write_result_chunk_to_disk_using_zarr_skunk_works as writer,
        )
    elif graph_mode:
        from astroviper.utils.io import write_result_chunk_to_disk_using_zarr

        def write(chunk_xds, chunk_task_coords):
            write_result_chunk_to_disk_using_zarr(
                image_store,
                _variables_made(image_data_variables_keep, chunk_xds),
                chunk_task_coords,
                chunk_xds,
            )

        return write
    else:

        def write(chunk_xds, chunk_task_coords):
            chunk_xds.to_zarr(image_store, consolidated=True)

        return write

    def write(chunk_xds, chunk_task_coords):
        writer(
            image_store,
            _variables_made(image_data_variables_keep, chunk_xds),
            chunk_task_coords,
            chunk_xds,
            processing_function_threads=processing_function_threads,
        )

    return write


def _variables_made(image_data_variables_keep, chunk_xds):
    """The keys of ``image_data_variables_keep`` whose variable the chunk holds."""
    return [
        key for key in image_data_variables_keep if key.upper() in chunk_xds.data_vars
    ]


def _concat_image_statistics(statistics_chunks):
    """Concatenate the per-chunk ``{image_variable_key: Dataset}`` statistics
    along ``frequency`` into the task's."""
    import xarray as xr

    if not statistics_chunks:
        return {}
    if len(statistics_chunks) == 1:
        return statistics_chunks[0]
    keys = list(statistics_chunks[0])
    return {
        key: xr.concat(
            [chunk[key] for chunk in statistics_chunks if key in chunk],
            dim="frequency",
        )
        for key in keys
    }


def _add_residual_entropy(statistics, chunk_xds, cycle_control_params):
    """Add ``entropy``, the entropy of the residual of every plane, to the
    image statistics of the residual (``report_statistics``)."""
    from astroviper.processing_functions.image_analysis.image_statistics import (
        image_residual_entropy,
    )
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        validate_entropy_params,
    )

    if "sky_residual" not in statistics:
        return
    _, max_snr, spatial_bins, flux_bins = validate_entropy_params(cycle_control_params)
    entropy, _ = image_residual_entropy(
        chunk_xds,
        data_group_name="residual",
        spatial_bins=spatial_bins,
        flux_bins=flux_bins,
        max_snr=max_snr,
    )
    statistics["sky_residual"]["entropy"] = (
        ("time", "frequency", "polarization"),
        entropy,
    )


def _combine_channel_timing_frames(timing_frames):
    """Fold the per-channel one-row timing frames into the chunk's one row.

    ``T_*`` columns and per-channel counts (``n_channels``) are summed;
    ``task_id`` is common to all channels; ``n_cycles`` becomes the largest
    number of imaging cycles any channel of the chunk ran (what one shared loop
    would have run for every channel) and ``n_cycles_total`` their sum (the
    imaging cycles actually run).
    """
    import pandas as pd

    columns = []
    for frame in timing_frames:
        for column in frame.columns:
            if column not in columns:
                columns.append(column)
    combined = {}
    for column in columns:
        values = [
            frame[column].iloc[0] for frame in timing_frames if column in frame.columns
        ]
        if column == "task_id":
            combined[column] = values[0]
        elif column == "n_cycles":
            combined[column] = max(values)
        else:
            combined[column] = sum(values)
    combined["n_cycles_total"] = sum(
        frame["n_cycles"].iloc[0] for frame in timing_frames if "n_cycles" in frame
    )
    return pd.DataFrame({key: [value] for key, value in combined.items()})


@shares_param_docs
def image_cube_single_field(
    image_params,
    imaging_weights_params,
    cycle_control_params,
    task_coords,
    data_selection,
    image_store,
    input_data_store,
    processing_set_data_group_name="corrected",
    deconvolver="hogbom",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    restore=False,
    primary_beam_correction=False,
    psf_fitting_method="astroviper",
    gridding_kernel_params=None,
    primary_beam_params=None,
    memory_mode="in_memory",
    skunk_works=False,
    data_group=None,
    task_id=0,
    input_data=None,
    lazy_input_data=None,
    graph_mode=True,
    image_chunking=None,
    image_sharding=None,
    output_image_format="zarr",
    task_time_kill_switch_seconds=None,
    report_statistics=False,
):
    """Image one frequency chunk of a single-field cube and write it to disk.

    Thin node task: builds the empty per-chunk
    image in the correlation (instrument) polarization basis, loads (or receives)
    this chunk's visibilities, runs the science
    :func:`~astroviper.processing_functions.imaging.image_cube_single_field.image_cube_single_field`
    **once per frequency channel**, writes every finished on-disk frequency
    chunk (``image_chunking["frequency"]`` channels; by default the whole
    task) to the image store as soon as its channels are imaged, and returns
    the timing and deconvolution metadata.

    Imaging one channel at a time gives every channel its own imaging cycle
    loop: a channel that has converged stops cycling -- no more degridding,
    gridding or FFTs for it -- while the others carry on, and ``max_cycles``
    counts per channel. Each call receives the visibility channels that map
    onto that image channel (zero-copy views of the loaded chunk) and a
    one-channel slice of the empty image; the science function itself handles
    full ``(time, frequency, polarization, l, m)`` cubes and is unchanged.
    Writing chunk by chunk keeps at most one chunk in memory next to the
    channel being imaged, and a one-channel chunk -- a single-channel task or
    ``image_chunking={"frequency": 1}`` -- is written without any copy.

    This function has a fully spelled-out signature so it can be called directly
    (standalone) outside of a graph.  When driven by
    :func:`graphviper.graph_tools.map.map`, graphviper adapts it automatically
    (via :func:`graphviper.graph_tools.map.make_graph_node_task`), expanding the
    single ``input_params`` dict it passes into these keyword arguments.

    Parameters
    ----------
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    cycle_control_params : dict
        CLEAN cycle controls. An **imaging cycle** (below simply a cycle)
        is one **residual update** (degrid the model, form residual
        visibilities, grid and inverse FFT them into the residual image)
        followed by one **model update** (deconvolve the residual image into
        the sky model). Every limit and threshold is applied independently to
        each ``(time, frequency, polarization)`` plane: a plane stops when it
        meets its own criterion. The imaging cycle loop runs separately for
        every frequency channel in cube imaging (one channel per node task),
        so a channel's cycles continue until all of its (time, polarization)
        planes have stopped, and a channel that has stopped does no further
        residual updates while the others carry on. **Cycle control** is the
        stop test between the residual update and the model update, which
        decides whether a plane gets another cycle and sets the limits of its
        next model update. It is made after every residual update, the first
        included, on the residual that update has just made, never on the
        residual a model update ends with, and the residual and restored
        images written are those of the last residual update. **Iteration
        control** is the stop test inside the model update, made by the
        deconvolver at every iteration against those limits. The CASA
        ``tclean`` equivalent is given in brackets.
        Keys:

        - ``max_iter`` [CASA ``niter``] : Maximum number of deconvolution
          iterations (flux components) per plane, summed over all cycles. A
          plane stops once it has spent this budget. ``max_iter = 0`` makes
          only the dirty image (no deconvolution). *Differs from CASA*: CASA's
          ``niter`` is one budget for the whole image; here every plane gets
          the full value, and no budget is shared or split between planes.
        - ``max_cycles`` [CASA ``nmajor``] : Maximum number of cycles.
          ``max_cycles = N`` performs at most ``N`` model updates and
          ``N + 1`` residual updates: the first makes the dirty image, and
          every model update is followed by a residual update, whose residual
          the stop test checks; the last one's residual is written.
          ``max_cycles = 0`` makes only the dirty image; ``max_cycles = -1``
          removes the limit. Counted per
          frequency channel: a channel that converges early stops cycling while
          the others continue. ``max_cycles = "auto"`` predicts the limit of a
          channel from its dirty image: every plane takes ``P0``, its peak
          residual at the start of the first cycle (the ``peak_residual``
          below), ``f``, its PSF fraction (below), and ``L``, its threshold or,
          when that is 0, its theoretical point source sensitivity, worked out
          from the imaging weights as the ``SENSITIVITY`` of
          ``point_spread_function_cube_single_field``;
          ``n = ceil(ln(P0 / L) / ln(1 / f))`` cycles (at least 1) bring the
          peak down to ``L``, and the plane's limit is ``n + 3`` against a
          threshold and ``3 * n`` against the sensitivity (``EXTRA_CYCLES``
          and ``NOISY_CYCLE_SAFETY`` of
          :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
          The sensitivity is the thermal noise of the plane when the weights
          are absolute (``WEIGHT = 1 / sigma**2``). With unit weights, as a
          simulation without noise writes them, it is not a noise level but
          about 1 over the square root of the number of visibilities (natural
          weighting; 1e-3 for a million) in the units of the data; the limit
          grows only with ``ln(P0 / L)``, and on the same prediction ``3 * n``
          cycles bring the peak down to ``P0 * (L / P0)**3``. The channel's
          ``max_cycles`` is the largest limit of its planes (at least 1; a
          plane without a pixel in the clean mask, with a peak of 0 or without
          data is left out) and then counts down like a number. Every plane
          records its limit as ``max_cycles_auto`` in the returned convergence
          record. Needs ``max_psf_fraction < 1``.
        - ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, as a
          float in Jy, or a list of one float per Stokes plane in the order of
          ``polarization_coords`` (the floors of the Stokes planes of noise free
          data scale with their fluxes). The distributed application and the
          node task also take a nested list of shape (n_frequency,
          n_polarization), one row per image channel in the order of the image
          frequency coordinate; each channel gets its row. A plane stops when the
          peak of its residual inside the clean mask, as the latest residual
          update made it, falls to or below its threshold times
          ``1 + threshold_tolerance``; the threshold is
          also a hard floor on ``threshold_per_cycle``. ``0`` disables the
          absolute stop. *Differs from CASA*: floats in Jy only, no ``'1mJy'``
          strings.
        - ``threshold_tolerance`` : Relative tolerance of the threshold stop,
          a finite number ``>= 0``: a plane whose threshold is above 0 stops
          with imaging stop code 2, and gets no iterations while other planes of
          its channel go on, once the peak of its residual inside the clean mask
          is at most ``threshold * (1 + threshold_tolerance)``; the model update
          of a plane that goes on still cleans down to ``threshold_per_cycle``,
          floored at the threshold itself. Default 0.02 (``THRESHOLD_TOLERANCE`` of
          :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
          No CASA equivalent.
        - ``threshold_sigma`` [CASA ``nsigma``] : Noise based stopping threshold
          per plane, as a multiple of the plane's robust residual rms
          (``1.4826 * MAD``). The effective threshold of a plane is
          ``max(threshold, threshold_sigma * rms)`` and it floors
          ``threshold_per_cycle`` in the same way. ``0`` disables it. Reserved:
          accepted but not yet implemented.
        - ``primary_beam_limit`` [CASA ``pblimit`` / ``pbmask``] : Primary beam
          mask cutoff as a fraction of the peak primary beam, in ``[0, 1]``.
          Pixels where the primary beam is below this fraction are excluded
          from cleaning. A masking cutoff, distinct from ``threshold``.
          Default 0: without a clean mask, the model update cleans and the
          stop test measures every pixel where the primary beam is above 0.
        - ``gain`` [CASA ``gain``] : CLEAN loop gain, the fraction of the
          selected peak flux subtracted from the residual image at each
          deconvolution iteration (``0 < gain <= 1``).
        - ``psf_sidelobe_factor`` [CASA ``cyclefactor``] : Multiplier applied to
          the measured peak PSF sidelobe level (``max_psf_sidelobe``) when
          setting how deep one model update cleans (see
          ``threshold_per_cycle``). Larger values trigger the next residual
          update sooner; smaller values clean deeper before each residual
          update.
        - ``max_iter_per_cycle`` [CASA ``cycleniter``] : Maximum number of
          deconvolution iterations a plane may run in one cycle's model update
          before the next residual update is triggered. ``max_iter_per_cycle =
          -1`` lets the adaptive ``threshold_per_cycle`` govern the depth
          instead; otherwise the count is clamped to never exceed the plane's
          remaining ``max_iter``.
        - ``min_psf_fraction`` [CASA ``minpsffraction``] : Lower clamp on the PSF
          fraction defined below. Raising it limits how deep a single model
          update cleans.
        - ``max_psf_fraction`` [CASA ``maxpsffraction``] : Upper clamp on the
          same PSF fraction; it guarantees a minimum amount of cleaning per
          model update even when the PSF sidelobe level is high.
        - ``max_iter_divergence`` : Divergence test of the model update.
          Number of consecutive deconvolution iterations the RMS of
          a plane's residual, taken over the clean mask, may be above
          ``(1 + gain / 10)`` times the lowest RMS it has reached in the model
          update before that model update is stopped as diverged. For Hogbom
          a peak above ``(1 + gain)`` times the peak at the start of the
          model update stops it at once. Asp, whose least-squares components
          can raise the peak of a healthy residual, has no peak limit and
          allows ``(1 + gain)`` times the lowest RMS instead. An infinite peak
          always stops the model update. The next residual update then
          recomputes the true residual, the stop test is made on it, and the
          cycles go on (see ``max_cycles_divergence``). Default 1, the first
          such iteration; ``-1`` disables the two limits.
        - ``max_cycles_divergence`` : Divergence stop of the cycle control. A
          plane whose last ``max_cycles_divergence`` model updates all ended
          diverged (see ``max_iter_divergence``) gets no more iterations and
          is stopped with imaging stop code 5 by the stop test after the next
          residual update (unless a stop that takes precedence applies there:
          an empty mask, ``max_iter`` spent, the threshold or the entropy
          stop, codes 7, 1, 2 and 10), while the other planes of its channel
          carry on.
          Default ``-1``, off.
        - ``entropy_stop`` : If ``True``, a plane stops once the entropy of
          its residual has passed its maximum. The entropy (Homan, Roth and
          Pushkarev 2024, AJ 167, 11) measures how much the residual looks
          like noise everywhere. It rises while the clean removes emission
          and falls once the clean fits noise. It is worked out after every
          residual update, and the plane stops when it is lower than in an
          earlier cycle. The fall is noticed one model update after the
          maximum and the model of that model update is kept, so a small
          ``max_iter_per_cycle`` makes the stop sharper. Default ``False``.
          No CASA equivalent.
        - ``entropy_max_snr`` : The entropy of a plane is followed once the
          peak of its residual inside the clean mask is at most this many
          times the RMS of the residual. Above it the residual is dominated
          by the pattern of the point spread function. Default 6.
        - ``entropy_spatial_bins`` : Number of spatial bins along each of the
          two image axes used for the entropy. Default 7.
        - ``entropy_flux_bins`` : Number of flux bins per unit of RMS used for
          the entropy. Default 10.
        - ``record_iteration_history`` : If ``True``, every model update
          records, for every plane and every deconvolution iteration, the peak
          residual the iteration cleaned (signed, in the units of the residual
          image) and the RMS of the residual inside the clean mask after its
          subtraction, to follow the deconvolution in detail. The returned
          convergence record holds them as ``iteration_peak_residual`` and
          ``iteration_residual_rms``: per plane one array per cycle, with one
          value per iteration of that cycle's model update (empty for a cycle
          without one), so the record grows by two float64 values per
          iteration done. A debugging and tuning tool: the images are the same
          to the bit with and without it, and the RMS is the sum of squares
          the peak search of the next iteration makes anyway, so it costs no
          extra pass over the image per iteration. Hogbom deconvolvers only:
          ``deconvolver="asp"`` raises ``ValueError``. Default ``False``.
          No CASA equivalent.
        - ``exact_psf_subtraction`` : If ``True``, a Hogbom model update
          subtracts the PSF of every component it finds from the whole residual
          image, with a PSF of twice the image size made once per channel from
          the same imaging weights, cell size and gridding kernel. With
          ``False`` it subtracts the image-sized PSF, which reaches about
          ``n/2`` pixels either side of the component along each axis
          (``n`` the image size in pixels), so farther pixels keep the
          sidelobes of the flux cleaned until the next residual update. The
          beam fit, ``max_psf_sidelobe`` and the PSF written use the
          image-sized PSF either way. Costs a second PSF per channel (four
          times the pixels, kept through the channel's cycles) and a
          subtraction over the whole image at every iteration. Hogbom
          deconvolvers only: ``deconvolver="asp"`` raises ``ValueError``.
          Default ``False``. No CASA equivalent.
        - ``asp_stop_after_hogbom_run`` : If ``True``, an Asp model update ends
          after each of its Hogbom runs (51 or 510 point components) unless
          the RMS of the residual has fallen, by less than a factor 1.5, since
          the model update started, as in CASA and LibRA. Their residual drifts
          from the true one during a model update; the Asp residual of
          AstroVIPER is the exact one, so with ``False`` the model update goes
          on to its ``threshold_per_cycle``, its iteration limit or the
          divergence test, which takes fewer imaging cycles. Asp only.
          Default ``False``.
        - ``asp_residual_components`` : If ``True``, each Asp model update
          starts with components shaped by the residual (Ma, Zhang & Zhang
          2026, A&A 713, A164) down to ``threshold_per_cycle``, and the Asp
          iterations go on from there. They remove the ring residuals Asp
          leaves on resolved emission in far fewer iterations, and take no
          residual below ``threshold_per_cycle``. Further keys
          (``asp_rc_*``) are described in
          :func:`~astroviper.processing_functions.imaging.deconvolution.asp_clean`.
          Asp only. Default ``True``. No CASA equivalent.
        - ``asp_scale_significance`` : If ``True``, Asp measures the noise of
          the residual smoothed with each of its initial scales and, once the
          peak is below ``threshold_per_cycle``, goes on cleaning the most
          significant smoothed residual while it is as significant as the
          threshold is at the smallest scale. The noise is measured on the
          pixels outside the clean mask, without the outermost row and column
          of the image, sampling one in every ``max(1, npix // 400000)`` of
          the ``npix`` pixels of the image (one in 2 at 1024 x 1024, one in 10
          at 2048 x 2048). With fewer than 100 sampled pixels (a clean mask
          that covers the image, for example), or a noise of 0 there (a
          residual that is mostly 0 outside the mask), nothing is cleaned
          below ``threshold_per_cycle``. Asp only. Default ``True``. No CASA
          equivalent.

        A plane whose model updates do no iteration any more (two in a row)
        is stopped with the no progress stop code at the next stop test, so
        an all-zero plane cannot keep its channel cycling.

        Derived per plane before each model update (not set by the caller):
        ``psf_fraction = clamp(max_psf_sidelobe * psf_sidelobe_factor,
        min_psf_fraction, max_psf_fraction)`` is the fraction of the current
        peak residual down to which one model update cleans, and
        ``threshold_per_cycle = max(psf_fraction * peak_residual, threshold)``
        is the stopping threshold of that model update, where
        ``peak_residual`` is the plane's peak residual inside the clean mask
        after the residual update of the same cycle (the peak of the stop
        test), and
        ``max_iter_per_cycle_updated = min(max_iter_per_cycle,
        max_iter_remaining)`` is the iteration limit of that model update (0
        for a plane the entropy or divergence stop has ended), where
        ``max_iter_remaining`` is what is left of ``max_iter``.
    task_coords : dict
        Per-chunk coordinate mapping; ``task_coords[<parallel dim>]`` supplies
        this chunk's parallel coordinate values (``"data"``) and its
        ``"slice"`` into the full output array (for cube imaging the
        parallel dim is ``frequency``).
    data_selection : dict
        Per-chunk ``{ms_name: {dim: slice}}`` selection injected by graphviper;
        used to load this chunk's data and to remap chunk-local channel numbers
        to global ones.
    image_store : str
        Path/URL of the on-disk Zarr image cube.
    input_data_store : str
        Path/URL of the processing-set Zarr store to load this chunk's
        visibilities from (used only when ``input_data`` and
        ``lazy_input_data`` are ``None``).
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    deconvolver : str, optional
        Deconvolution algorithm for the model update. One of ``"hogbom"`` (C++, threaded across planes), ``"hogbom_many_threads"``
        (C++, threaded across *and* within planes -- faster when there are
        few planes, e.g. single-channel imaging) or ``"asp"``.
    instrument_polarization_basis : str, optional
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` or ``"circular"``. The residual update grids and degrids the
        correlations of this basis and the model update deconvolves in the
        Stokes basis, in which the image is written. The Stokes planes requested
        in ``image_params["polarization_coords"]`` fix the correlations that are
        loaded and gridded: the two parallel hands give ``I, Q`` (linear) or
        ``I, V`` (circular), all four correlations give ``I, Q, U, V``. A sample
        is used only if none of its loaded correlations is flagged.
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the model
        update runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    restore : bool, optional
        If ``True`` produce a restored image after deconvolution: the model
        convolved with the clean beam (the Gaussian fit to the PSF) plus the
        residual, written to the ``sky_restored`` (``SKY_RESTORED``) variable.
    primary_beam_correction : bool, optional
        If ``True`` write the primary beam corrected restored sky to the
        ``sky_restored_primary_beam_corrected``
        (``SKY_RESTORED_PRIMARY_BEAM_CORRECTED``) variable: the model divided
        by the (power) primary beam and convolved with the clean beam, plus
        the residual divided by the primary beam; pixels below the primary
        beam cutoff (``primary_beam_limit``) are blanked with NaN.  Requires
        ``restore``.
    psf_fitting_method : str, optional
        Beam-fit algorithm for the PSF: ``"astroviper"`` (default) or
        ``"casa"``, the C++ port of CASA's ``StokesImageUtil::FitGaussianPSF``
        (the fit behind ``tclean``'s restoring beam).
    gridding_kernel_params : dict, optional
        The gridding convolution kernel the gridder and degridder use:
        ``type`` (``"prolate_spheroidal_casa"``, Schwab's rational
        approximation of the ``m = 6``, ``alpha = 1`` function with its
        analytic correction, the kernel CASA uses; ``"prolate_spheroidal"``,
        the generalised prolate spheroidal function of Schwab 1984 with its
        bandwidth and weight exponent chosen for the image region the
        ``fft_padding`` keeps; ``"exponential_semicircle"``, the modified
        exponential of a semicircle of Barnett et al. 2019 and Arras et al.
        2021 shaped for that region; or
        ``"least_misfit"``, the table that minimises the map error over that
        region for the lookup used, after Ye et al. 2020), ``support`` (the
        odd number of grid points the kernel spans along one axis; 7 for the
        ``prolate_spheroidal_casa`` kernel, any odd number from 3 for the others),
        ``oversampling`` (samples of the kernel table per uv cell) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, so the lookup error,
        and with it the floor the imaging cycle converges to, falls as 1 over
        the oversampling; ``"linear"`` interpolates the two neighbouring
        samples and the error falls as 1 over its square). Missing keys take
        the defaults ``{"type": "prolate_spheroidal_casa", "support": 7,
        "oversampling": 100, "interpolation": "nearest"}``; ``None`` is all
        defaults. A ``prolate_spheroidal``, ``exponential_semicircle`` or
        ``least_misfit`` kernel whose image plane correction falls below 1e-6
        of its centre value at the edge of the image kept (a support too large
        for the ``fft_padding``, e.g. 17 at 1.05) is refused. The kernel of the
        former ``gridder="prolate_spheroidal"`` is ``"prolate_spheroidal_casa"``,
        not ``"prolate_spheroidal"``.
    primary_beam_params : dict, optional
        The primary beam of the imager, the power pattern that gives the clean
        mask and the primary beam correction: ``telescope`` (``"alma"``,
        ``"aca"`` or ``"ngvla"``, the textbook obscured Airy disk with the
        diameters of :func:`~astroviper.utils.beam_models.airy_disk_model`, no
        cut; ``"vla"``, the EVLA beam polynomial of CASA's ``PBMath1DEVLA``,
        with the coefficients CASA picks for each channel (of its two 10052 MHz
        rows, the first), at the exact radius and zero beyond 58 arcmin at
        1 GHz, past the 5 percent power level its polynomials were fitted
        to), ``effective_dish_diameter`` and
        ``blockage_diameter`` (metres; refused for ``"vla"``) and
        ``max_rad_1GHz`` (radians at 1 GHz, scaled as 1 / frequency, beyond
        which the beam is zero; ``None`` is no cut for an Airy disk). Numbers
        given with a telescope override that telescope's values;
        no telescope means ``"alma"``. ``None`` is the ALMA 12 m beam (10.7 m,
        0.75 m, no cut). Or ``beam_model`` alone: an analytic dict
        or a beam-polynomial dataset as ``simulate_processing_set`` takes it in
        ``beam_models`` (``"vla"`` is
        ``read_beam_polynomial_coefficients("EVLA_")``). The beam is the square
        of the voltage pattern the simulator applies for the model, computed by
        the same function
        (:func:`~astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response`).
        The CASA variants, to compare with the ``.pb`` of ``tclean``, are such
        models: ``airy_disk_model(telescope, func="casa_airy")`` and
        ``read_beam_polynomial_coefficients("EVLA_", n_sample=10000)``.
        Zernike aperture models and Jones images are not supported: they are
        not azimuthally symmetric and depend on the parallactic angle.
    memory_mode : str, optional
        Only ``"in_memory"`` is implemented.  Default ``"in_memory"``.
    skunk_works : bool, optional
        If ``True`` use the experimental performance I/O path: load only the
        data group's data variables straight from the Zarr chunk blobs with
        :func:`~astroviper.node_tasks.imaging.utils.load_processing_set_skunk_works`
        (reconstructing coordinates from the inputs) and write each result chunk
        with
        :func:`~astroviper.node_tasks.imaging.utils.write_result_chunk_to_disk_using_zarr_skunk_works`.
        Both the skunk-works load and write spread their per-array / per-variable
        I/O and (de)compression concurrently across ``processing_function_threads``
        threads.  Requires ``data_group``.  Default ``False`` (production I/O).
    data_group : dict, optional
        Resolved role->variable mapping for ``processing_set_data_group_name``
        (e.g. ``{"correlated_data": "VISIBILITY", "uvw": "UVW", ...}``), supplied
        by the distributed graph; only used when ``skunk_works`` is ``True``.
    task_id : int, optional
        Identifier of the parallel chunk being processed.
    input_data : dict, optional
        Pre-loaded data for this chunk (supplied by the data-loading layer); when
        ``None`` (default) the data is loaded from ``input_data_store``.
    lazy_input_data : dict, optional
        ``{ms_name: xarray.Dataset}``: this chunk's data, not yet read. For a
        Measurement Set v2 ``ps_store`` the distributed application opens the
        Measurement Set with XRADIO's ``xradio_msv2`` engine and gives every
        task, per MSv4 of its ``data_selection``, the data group's variables
        already restricted to that selection (lazily indexed arrays; see
        :func:`~astroviper.node_tasks.imaging.utils.add_lazy_input_data`).
        The task reads them with
        :func:`~astroviper.node_tasks.imaging.utils.load_processing_set_skunk_works_msv2`
        (used only when ``input_data`` is ``None``; ``skunk_works`` and
        ``data_group`` are not consulted for the read). Any read error skips
        the chunk, as for the other inputs, except
        ``xradio.measurement_set.MSv2ChangedError``: the Measurement Set
        changed after the distributed application opened it, so the graph no
        longer describes it and the error is raised, aborting the run. ``None``
        (default): the data is loaded from ``input_data_store``.
    graph_mode : bool, optional
        If ``True`` (default) each kept variable's slice is written into the
        pre-allocated Zarr store with
        :func:`~astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.  If
        ``False`` the whole chunk image is written with ``to_zarr``.
    image_chunking : dict, optional
        On-disk chunk shape of the image store as ``{dimension_name:
        chunk_size}`` (see the distributed application). Only its
        ``"frequency"`` entry matters here: finished channels are gathered into
        chunks of that many channels and every complete chunk is written -- and
        freed -- right away, so the task holds at most one chunk plus the
        channel being imaged (a one-channel chunk is written without any copy).
        ``None`` (default) writes the whole task as one chunk.
    image_sharding : dict, optional
        Shard shape of the (Zarr v3 sharded) image store as ``{dimension_name:
        shard_size}``; when set together with ``skunk_works`` the chunks are
        written with
        :func:`~astroviper.node_tasks.imaging.utils.write_result_chunk_to_disk_sharded_skunk_works`.
        ``None`` (default) selects the unsharded writer.
    output_image_format : str, optional
        On-disk format of the image store this task writes into: ``"zarr"``
        (default) or ``"fits"``.  With ``"fits"`` the chunk is ``pwrite``-en
        directly into the pre-created XRADIO-conformant FITS files
        (:func:`~astroviper.node_tasks.imaging.utils.write_result_chunk_to_fits_skunk_works`;
        the driver created them with
        :func:`~astroviper.node_tasks.imaging.utils.create_empty_fits_images`).
    report_statistics : bool, optional
        The statistics only a report of the run needs; the distributed
        application sets it when it makes one. With ``True`` the image
        statistics of the residual also hold ``entropy``, the entropy of the
        residual of every plane as the entropy stop works it out
        (:func:`~astroviper.processing_functions.image_analysis.image_statistics.image_residual_entropy`
        with the ``entropy_*`` settings of ``cycle_control_params``; not a
        number for a plane whose peak is above ``entropy_max_snr`` times its
        RMS). Default ``False``.

    Returns
    -------
    dict
        Single dict with two keys:

        * ``"timing_node_tasks"`` : one-row :class:`pandas.DataFrame` with a
          ``T_*`` column per processing function (load, image build, weights,
          PSF, primary beam, gridding, FFT normalization, degridding,
          deconvolution, write, ...) summed over the chunk's channels, plus
          ``task_id``, ``n_channels``, ``n_cycles`` (the largest number of
          imaging cycles any channel of the chunk ran), ``n_cycles_total``
          (imaging cycles summed over the channels), ``T_channel_bookkeeping``
          (slicing the chunk per channel and gathering the results into
          on-disk chunks) and the total ``T_image_cube_task``. A write failure
          marks the row with ``task_failed_phase``, ``task_error``,
          ``failed_channel_start``, ``failed_n_channels`` (the first failed
          chunk) and ``n_failed_chunks``.
        * ``"deconvolution"`` : the per-plane deconvolution
          :class:`~astroviper.processing_functions.imaging.utils.imaging_dict.ImagingDict`,
          with channels remapped to global channel numbers.
        * ``"image_statistics"`` : ``{image_variable_key: xarray.Dataset}`` of
          NaN-ignoring per-plane statistics of the image-domain variables
          present in memory (``sky_residual``, ``sky_restored``, ``sky_model``,
          ...), computed over ``(l, m)`` *before* the chunk is written. Each
          dataset has dims ``(time, frequency, polarization)`` for this chunk's
          frequencies and one variable per statistic (``mean``, ``median``,
          ``max``, ``min``, ``peak``, ``sum``, ``rms``, ``std``, ``mad_sigma``,
          ``n_pixels`` and their ``_masked`` twins restricted to the clean
          mask, or to ``PRIMARY_BEAM > primary_beam_limit`` when no mask
          exists, e.g. ``max_iter=0``); see
          :func:`~astroviper.processing_functions.image_analysis.plane_statistics.calculate_plane_statistics`.
          The reduce concatenates the chunks along ``frequency``.
    """
    import time

    import toolviper.utils.logger as logger
    from toolviper.utils.memory_management import get_rss_gb
    from xradio.image import make_empty_sky_image

    import astroviper.processing_functions as pf

    task_start = time.time()

    logger.debug(
        "Memory usage at start of image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    assert memory_mode == "in_memory", (
        "Currently only memory_mode='in_memory' is implemented."
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = [
            "sky_residual",
            "point_spread_function",
            "primary_beam",
        ]

    # Build the empty per-chunk image in the correlation (instrument)
    # polarization basis the gridder works in. The requested Stokes planes
    # (image_params["polarization_coords"]) fix the correlations: the two
    # parallel hands for I, Q (linear) or I, V (circular), all four for
    # I, Q, U, V. The science function transforms between the two bases.
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        correlations_for_stokes,
    )

    correlation_pol_coords = correlations_for_stokes(
        image_params["polarization_coords"], instrument_polarization_basis
    )
    start = time.time()
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=task_coords["frequency"]["data"],
        pol_coords=correlation_pol_coords,
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )
    T_make_empty_image = time.time() - start

    start = time.time()
    try:
        if input_data is not None:
            # Data was pre-loaded by the data loading layer (disk-chunk granularity
            # I/O coalescing). The framework has already applied the task-level
            # sub-selection, so use the dict directly.
            ps_xdt = input_data
        elif lazy_input_data is not None:
            # Measurement Set v2: read the lazily indexed selection that the
            # distributed application attached to this task (the values are
            # read by XRADIO's xradio_msv2 engine).
            from astroviper.node_tasks.imaging.utils import (
                load_processing_set_skunk_works_msv2,
            )

            ps_xdt = load_processing_set_skunk_works_msv2(lazy_input_data)
        elif skunk_works:
            # Experimental performance path: read only this chunk's data-group
            # variables straight from the Zarr chunk blobs and reconstruct the
            # coordinates from the inputs (no datatree/coords/sub-datasets open).
            from astroviper.node_tasks.imaging.utils import (
                load_processing_set_skunk_works,
            )

            ps_xdt = load_processing_set_skunk_works(
                input_data_store,
                sel_parms=data_selection,
                data_group=data_group,
                processing_set_data_group_name=processing_set_data_group_name,
                frequency_coords=task_coords["frequency"]["data"],
                instrument_polarization_basis=instrument_polarization_basis,
                processing_function_threads=processing_function_threads,
            )
        else:
            from xradio.measurement_set.load_processing_set import load_processing_set

            ps_xdt = load_processing_set(
                input_data_store,
                sel_parms=data_selection,
                data_group_name=processing_set_data_group_name,
                load_sub_datasets=False,
            )
    except Exception as exc:
        # A Measurement Set v2 that changed after the distributed application
        # opened it no longer matches the graph: every remaining task would be
        # inconsistent, which a skipped chunk would hide, so the run aborts.
        from astroviper.node_tasks.imaging.utils import is_fatal_load_error

        if is_fatal_load_error(exc):
            raise
        # A chunk whose data cannot be read is skipped -- logged + marked in the
        # timing frame -- instead of aborting the whole run (dask/MPI would
        # otherwise tear down every node after this task exhausts its retries).
        import pandas as pd

        from astroviper.processing_functions.imaging.utils.imaging_dict import (
            ImagingDict,
        )

        row = _log_task_io_failure(
            "load", exc, task_id, image_store, data_selection, task_coords
        )
        row.update(
            {
                "T_make_empty_image": T_make_empty_image,
                "T_load": time.time() - start,
                "T_image_cube_task": time.time() - task_start,
                "start_unixtime": task_start,
            }
        )
        return {
            "timing_node_tasks": pd.DataFrame({k: [v] for k, v in row.items()}),
            "deconvolution": ImagingDict(),
            "image_statistics": {},
        }
    T_load = time.time() - start

    # ---- Imaging cycle loops, one frequency channel at a time ----
    # The science function is called once per image channel, with the
    # visibility channels that map onto it (zero-copy views of the loaded
    # chunk) and a one-channel slice of the empty image, so every channel runs
    # its own imaging cycle loop: a channel that has converged stops cycling
    # (no further degridding, gridding or FFTs for it) while the others carry
    # on, and ``max_cycles`` counts per channel. Finished channels are gathered
    # into on-disk frequency chunks (``image_chunking["frequency"]`` channels;
    # by default the whole task) and every complete chunk has its plane
    # statistics taken, is written and is freed right away, so the task never
    # holds more than one chunk plus the channel being imaged -- and with
    # one-channel chunks nothing is ever copied. The science function itself
    # handles full cubes -- looping over a dimension belongs here (AGENTS.md
    # section 3), never inside a processing function.
    from astroviper.processing_functions.image_analysis.plane_statistics import (
        calculate_plane_statistics,
    )
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        merge_imaging_dicts,
    )
    from astroviper.utils.data_tree import clear_cached_accessors, release_data_tree

    n_chan = img_xds.sizes["frequency"]
    # Channels per written chunk: the on-disk frequency chunk, clipped to this
    # task (the cube's last task may be shorter). The whole-image ``to_zarr``
    # of graph_mode=False is not chunk-aware, so there the task is one chunk.
    if graph_mode and image_chunking and image_chunking.get("frequency"):
        chunk_channels = min(int(image_chunking["frequency"]), n_chan)
    else:
        chunk_channels = n_chan
    write_chunk = _select_chunk_writer(
        graph_mode,
        skunk_works,
        image_sharding,
        output_image_format,
        image_store,
        image_data_variables_keep,
        processing_function_threads,
    )
    # Masked statistics use the clean MASK when present; a max_iter=0 run has
    # none, so the fallback mask PRIMARY_BEAM > primary_beam_limit (the same
    # valid-sky cutoff the deconvolver would use) applies.
    primary_beam_limit = cycle_control_params.get("primary_beam_limit", 0.2)
    frequency_maps = _visibility_to_image_frequency_maps(ps_xdt, img_xds)
    timing_frames = []
    imaging_dicts = []
    statistics_chunks = []
    write_failures = []  # (exception, chunk task_coords) per failed chunk write
    accumulator = None
    T_channel_bookkeeping = 0.0
    T_image_statistics = 0.0
    T_write = 0.0
    for chan_index in range(n_chan):
        start = time.time()
        ps_chan = _select_processing_set_channel(ps_xdt, frequency_maps, chan_index)
        if ps_chan is None:
            # No visibility channel maps onto this image channel: hand over
            # views of the whole chunk (see _whole_processing_set_channel for
            # what the science function grids onto it).
            logger.debug(
                f"Image channel {chan_index} of task {task_id} has no visibility "
                "channels; imaging it from the full chunk."
            )
            ps_chan = _whole_processing_set_channel(ps_xdt)
        img_chan = _select_image_channel(img_xds, chan_index)
        if accumulator is None:
            accumulator = _ImageChunkAccumulator(
                img_xds, chan_index, min(chunk_channels, n_chan - chan_index)
            )
        T_channel_bookkeeping += time.time() - start

        img_chan, timing_chan, imaging_dict_chan = pf.imaging.image_cube_single_field(
            ps_chan,
            img_chan,
            image_params,
            imaging_weights_params,
            _channel_cycle_control_params(
                cycle_control_params, task_coords, data_selection, chan_index
            ),
            processing_set_data_group_name=processing_set_data_group_name,
            deconvolver=deconvolver,
            instrument_polarization_basis=instrument_polarization_basis,
            single_precision_image=single_precision_image,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            image_data_variables_keep=image_data_variables_keep,
            restore=restore,
            primary_beam_correction=primary_beam_correction,
            psf_fitting_method=psf_fitting_method,
            gridding_kernel_params=gridding_kernel_params,
            primary_beam_params=primary_beam_params,
            task_id=task_id,
        )

        start = time.time()
        timing_frames.append(timing_chan)
        imaging_dicts.append(
            _shift_imaging_dict_channels(imaging_dict_chan, chan_index)
        )
        accumulator.insert(img_chan, chan_index)
        # Drop this channel's objects right away: cached accessors would
        # otherwise pin its arrays until a full garbage-collection pass.
        clear_cached_accessors(img_chan)
        for ms_chan in ps_chan.values():
            clear_cached_accessors(ms_chan)
        # The loop variable would otherwise keep this channel's last
        # measurement set, with its model and residual visibilities and
        # imaging weights, alive through the statistics and the write.
        ms_chan = None
        img_chan = None
        ps_chan = None
        if chan_index == n_chan - 1:
            # Every channel is imaged: free the loaded chunk (visibilities,
            # weights, flags, uvw) before the last chunk's statistics and
            # write instead of after them. Severing its parent<->child links
            # lets it die by reference counting here.
            release_data_tree(ps_xdt)
            ps_xdt = None
        if not accumulator.complete:
            T_channel_bookkeeping += time.time() - start
            continue

        # ---- A whole on-disk frequency chunk is done: statistics, write, free ----
        chunk_xds = accumulator.assemble()
        chunk_task_coords = _chunk_task_coords(
            task_coords, data_selection, accumulator.start, accumulator.stop
        )
        accumulator = None
        T_channel_bookkeeping += time.time() - start

        # Per-plane (l, m) statistics of every image-domain variable in memory,
        # taken BEFORE the write so they describe exactly what goes to disk
        # (and survive a skipped write). Channels carry their global frequency
        # values, so the chunks -- and the reduce -- concatenate along
        # ``frequency``.
        start = time.time()
        statistics = calculate_plane_statistics(
            chunk_xds, primary_beam_limit=primary_beam_limit
        )
        if report_statistics:
            _add_residual_entropy(statistics, chunk_xds, cycle_control_params)
        statistics_chunks.append(statistics)
        T_image_statistics += time.time() - start

        start = time.time()
        try:
            write_chunk(chunk_xds, chunk_task_coords)
        except Exception as exc:
            # A chunk whose result cannot be written is skipped -- logged +
            # marked in the timing row below -- instead of aborting the whole
            # run; its channels keep the image store's fill value, and the
            # task's other chunks are still written.
            write_failures.append((exc, chunk_task_coords))
        T_write += time.time() - start
        clear_cached_accessors(chunk_xds)
        chunk_xds = None

    start = time.time()
    timing_df = _combine_channel_timing_frames(timing_frames)
    combined_imaging_dict = merge_imaging_dicts(imaging_dicts)
    image_statistics = _concat_image_statistics(statistics_chunks)
    timing_df["T_channel_bookkeeping"] = T_channel_bookkeeping + (time.time() - start)

    # The deconvolve dict's channels are chunk-local (0-based); remap them to
    # global channel numbers so the reduce can merge chunks correctly.
    combined_imaging_dict = _remap_imaging_dict_to_global_channels(
        combined_imaging_dict, data_selection
    )

    # Two reference-cycle classes pin this task's gigabytes past `= None`
    # (2026-08-12 findings; each survives until a full gc pass otherwise):
    # 1. DataTree parent<->child links (the loaded chunk's tree, released
    #    after the last channel's science call; released here only when the
    #    loop did not run), and
    # 2. the xarray cached-accessor cycle on the image dataset
    #    (_cache['xr_img'] <-> xradio ImageXds._xds, created by the
    #    img_xds.xr_img.* calls in the processing functions).
    # Sever both so everything dies by refcount right here. Both helpers are
    # no-ops on None, the load-layer dict path and cache-less datasets.
    release_data_tree(ps_xdt)
    clear_cached_accessors(img_xds)
    img_xds = None
    ps_xdt = None

    logger.debug(
        "Memory usage after image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    # Fold the node-task timings (image build, load, write, total) into the
    # per-chunk timing frame produced by the science function.
    task_total_time = time.time() - task_start
    timing_df["T_make_empty_image"] = T_make_empty_image
    timing_df["T_load"] = T_load
    timing_df["T_image_statistics"] = T_image_statistics
    timing_df["T_write"] = T_write
    timing_df["T_image_cube_task"] = task_total_time
    # Wall-clock anchor so the task-stream analysis can place this task on the
    # run's common timeline without needing the resource monitor (whose own
    # anchor, recorded a hair earlier around the whole task, overwrites this
    # column when monitor_resources_seconds is set).
    timing_df["start_unixtime"] = task_start
    # Record which node ran this task so the per-chunk timing frame can be grouped
    # by host (identify stragglers / a slow node in the sweep), plus the exact
    # execution slot (process + thread + Dask worker name) so the task-stream
    # analysis can reconstruct TRUE per-worker lanes -- and place reduce nodes
    # (which record the same identity) on the lane they actually ran on --
    # instead of inferring lanes by interval packing. worker_name is None
    # outside a Dask worker (the MPI ranks).
    import os
    import socket
    import threading

    hostname = socket.gethostname()
    timing_df["hostname"] = hostname
    timing_df["process_pid"] = os.getpid()
    timing_df["thread_native_id"] = threading.get_native_id()
    try:
        from distributed import get_worker

        timing_df["worker_name"] = str(get_worker().name)
    except Exception:
        timing_df["worker_name"] = None

    if write_failures:
        # Log every failed chunk; the timing row records the first one plus
        # the count, so failures stay queryable per run.
        markers = [
            _log_task_io_failure(
                "write", exc, task_id, image_store, data_selection, chunk_coords
            )
            for exc, chunk_coords in write_failures
        ]
        for key in (
            "task_failed_phase",
            "task_error",
            "failed_channel_start",
            "failed_n_channels",
        ):
            timing_df[key] = markers[0][key]
        timing_df["n_failed_chunks"] = len(write_failures)

    # Timing kill switch: if this task overran the watchdog threshold, dump its
    # full timing breakdown to an error log and raise -- aborting the whole
    # distributed computation (fail fast on a pathological node/I-O stall rather
    # than hang the run). A task already skipped for a write failure is exempt:
    # its (long) retry schedule must not re-escalate into the abort this
    # skip-and-log path exists to avoid.
    if (
        not write_failures
        and task_time_kill_switch_seconds is not None
        and task_total_time > task_time_kill_switch_seconds
    ):
        log_path = _write_task_kill_switch_log(
            timing_df,
            task_total_time,
            task_time_kill_switch_seconds,
            image_store,
            task_id,
            hostname,
        )
        msg = (
            f"task_time_kill_switch tripped: task {task_id} on {hostname} took "
            f"{task_total_time:.1f}s > {task_time_kill_switch_seconds}s threshold. "
            f"Aborting the run. Timing log written to: {log_path}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Debug: phase-grouped timing breakdown for this chunk. The generic
    # formatter lives in the top-level utils; the imaging phase layout
    # parameterizes it.
    from astroviper.processing_functions.imaging.utils import (
        IMAGING_TIMING_PHASES,
        IMAGING_TIMING_TOTAL_KEY,
    )
    from astroviper.utils.timing import print_timing_summary

    print_timing_summary(
        timing_df,
        IMAGING_TIMING_PHASES,
        total_key=IMAGING_TIMING_TOTAL_KEY,
        printer=logger.debug,
    )

    return {
        "timing_node_tasks": timing_df,
        "deconvolution": combined_imaging_dict,
        "image_statistics": image_statistics,
    }
