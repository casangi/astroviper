from astroviper.utils.param_docs import shares_param_docs

# Image variables the node task writes when the distributed application gives
# an image store (keys of the io registry, ``astroviper.utils.io``).
POINT_SPREAD_FUNCTION_IMAGE_VARIABLES = [
    "point_spread_function",
    "beam_fit_params_point_spread_function",
]


@shares_param_docs
def point_spread_function_cube_single_field(
    image_params,
    imaging_weights_params,
    task_coords,
    data_selection,
    input_data_store,
    image_store=None,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    psf_fitting_method="astroviper",
    gridding_kernel_params=None,
    primary_beam_params=None,
    load_data_variables=None,
    task_id=0,
    input_data=None,
):
    """Make the point spread function of one frequency chunk of a single-field
    cube, with its beam fit and the weight statistics of every channel.

    Thin node task: loads this chunk's weights, flags and uvw (``VISIBILITY``
    is never read when ``load_data_variables`` leaves it out), then, **once
    per frequency channel**, runs
    :func:`~astroviper.processing_functions.imaging.residual_update.imaging_setup_single_field`
    (imaging weights, point spread function, primary beam, Stokes basis and
    Gaussian beam fit, exactly as the imager's setup) followed by
    :func:`~astroviper.processing_functions.imaging.weight_statistics.calculate_weight_statistics_single_field`
    on the same weights. Each call receives the visibility channels that map
    onto that image channel, as in
    :func:`~astroviper.node_tasks.imaging.image_cube_single_field.image_cube_single_field`.

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
    task_coords : dict
        Per-chunk coordinate mapping; ``task_coords[<parallel dim>]`` supplies
        this chunk's parallel coordinate values (``"data"``) and its
        ``"slice"`` into the full output array (for cube imaging the
        parallel dim is ``frequency``).
    data_selection : dict
        ``{ms_name: isel dict}`` selecting this task's visibilities.
    input_data_store : str
        Path of the processing set.
    image_store : str, optional
        Path/URL of the on-disk Zarr image cube.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
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
    load_data_variables : list of str, optional
        Data variables to load (XRADIO ``include_variables``), for example the
        data group's weight, flag and uvw variables; ``None`` (default) loads
        every variable of the data group, ``VISIBILITY`` included.
    task_id : int, optional
        Identifier of the parallel chunk being processed.
    input_data : dict, optional
        Pre-loaded ``{ms_name: node}`` visibilities; when given, nothing is
        loaded.

    Returns
    -------
    dict
        * ``"timing_node_tasks"`` : one-row :class:`pandas.DataFrame` with the
          ``T_*`` columns of the setup summed over the chunk's channels, plus
          ``task_id``, ``n_channels``, ``T_load``, ``T_make_empty_image``,
          ``T_weight_statistics``, ``T_write`` and the total
          ``T_point_spread_function_task``.
        * ``"point_spread_function_properties"`` : :class:`xarray.Dataset` over
          the chunk's ``frequency`` and the Stokes ``polarization`` with
          ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION``,
          ``MAX_SIDELOBE_POINT_SPREAD_FUNCTION`` and the variables of
          :func:`~astroviper.processing_functions.imaging.weight_statistics.calculate_weight_statistics_single_field`.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger
    import xarray as xr
    from xradio.image import make_empty_sky_image

    from astroviper.node_tasks.imaging.image_cube_single_field import (
        _chunk_task_coords,
        _select_image_channel,
        _select_processing_set_channel,
        _visibility_to_image_frequency_maps,
        _whole_processing_set_channel,
    )
    from astroviper.processing_functions.imaging.residual_update import (
        imaging_setup_single_field,
    )
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        correlations_for_stokes,
    )
    from astroviper.processing_functions.imaging.weight_statistics import (
        calculate_weight_statistics_single_field,
    )
    from astroviper.utils.data_tree import clear_cached_accessors, release_data_tree

    task_start = time.time()

    # The empty per-chunk image is in the correlation (instrument) basis the
    # gridder works in; the setup transforms it to the Stokes planes requested.
    start = time.time()
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=task_coords["frequency"]["data"],
        pol_coords=correlations_for_stokes(
            image_params["polarization_coords"], instrument_polarization_basis
        ),
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )
    T_make_empty_image = time.time() - start

    start = time.time()
    if input_data is not None:
        ps_xdt = input_data
    else:
        from xradio.measurement_set.load_processing_set import load_processing_set

        ps_xdt = load_processing_set(
            input_data_store,
            sel_parms=data_selection,
            data_group_name=processing_set_data_group_name,
            include_variables=load_data_variables,
            load_sub_datasets=False,
        )
    T_load = time.time() - start

    n_chan = img_xds.sizes["frequency"]
    frequency_maps = _visibility_to_image_frequency_maps(ps_xdt, img_xds)
    psf_channels = []
    statistics_channels = []
    timing_frames = []
    T_weight_statistics = 0.0
    for chan_index in range(n_chan):
        ps_chan = _select_processing_set_channel(ps_xdt, frequency_maps, chan_index)
        if ps_chan is None:
            logger.debug(
                f"Image channel {chan_index} of task {task_id} has no visibility "
                "channels; making its point spread function from the full chunk."
            )
            ps_chan = _whole_processing_set_channel(ps_xdt)
        img_chan = _select_image_channel(img_xds, chan_index)

        img_chan, timing_chan = imaging_setup_single_field(
            ps_chan,
            img_chan,
            image_params,
            imaging_weights_params,
            processing_set_data_group_name=processing_set_data_group_name,
            single_precision_image=single_precision_image,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            image_data_variables_keep=["point_spread_function"],
            psf_fitting_method=psf_fitting_method,
            gridding_kernel_params=gridding_kernel_params,
            primary_beam_params=primary_beam_params,
        )
        timing_frames.append(timing_chan)

        # The imaging weights are still in memory: take their statistics on
        # the channel's Stokes planes.
        start = time.time()
        statistics_channels.append(
            calculate_weight_statistics_single_field(
                ps_chan, img_chan, processing_set_data_group_name
            )
        )
        T_weight_statistics += time.time() - start

        # The beam fit labels its parameters with the beam_params_label
        # coordinate but gives the variable the dimension beam_params; use the
        # labelled dimension, as in the image store. The image is kept only to
        # be written.
        psf_chan = img_chan[
            (["POINT_SPREAD_FUNCTION"] if image_store is not None else [])
            + [
                "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
                "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
            ]
        ]
        if "beam_params" in psf_chan.dims:
            psf_chan = psf_chan.rename({"beam_params": "beam_params_label"})
        psf_channels.append(
            psf_chan.assign_coords(
                beam_params_label=img_chan["beam_params_label"].values
            )
        )
        psf_chan = None
        clear_cached_accessors(img_chan)
        for ms_chan in ps_chan.values():
            clear_cached_accessors(ms_chan)
        ms_chan = None
        img_chan = None
        ps_chan = None

    release_data_tree(ps_xdt)
    ps_xdt = None
    clear_cached_accessors(img_xds)
    img_xds = None

    psf_xds = xr.concat(psf_channels, dim="frequency")
    psf_channels = None

    start = time.time()
    if image_store is not None:
        from astroviper.utils.io import write_result_chunk_to_disk_using_zarr

        write_result_chunk_to_disk_using_zarr(
            image_store,
            POINT_SPREAD_FUNCTION_IMAGE_VARIABLES,
            _chunk_task_coords(task_coords, data_selection, 0, n_chan),
            psf_xds,
        )
    T_write = time.time() - start

    # The images have a single time plane; the properties are per channel and
    # Stokes plane.
    properties = xr.merge(
        [
            psf_xds[
                [
                    "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
                    "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
                ]
            ]
            .isel(time=0, drop=True)
            .reset_coords(drop=True),
            xr.concat(statistics_channels, dim="frequency"),
        ],
        join="exact",
        compat="override",
        combine_attrs="drop_conflicts",
    )

    timing = pd.concat(timing_frames, ignore_index=True).sum(numeric_only=True)
    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})
    timing_df["task_id"] = task_id
    timing_df["n_channels"] = n_chan
    timing_df["T_make_empty_image"] = T_make_empty_image
    timing_df["T_load"] = T_load
    timing_df["T_weight_statistics"] = T_weight_statistics
    timing_df["T_write"] = T_write
    timing_df["T_point_spread_function_task"] = time.time() - task_start

    return {
        "timing_node_tasks": timing_df,
        "point_spread_function_properties": properties,
    }
