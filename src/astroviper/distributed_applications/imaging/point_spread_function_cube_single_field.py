import os
from typing import Any

import toolviper.utils.parameter

from astroviper.utils.param_docs import shares_param_docs

# The toolviper parameter-check schema lives next to this module (rather than in
# a central config/ directory) so it is easy to find; point the validator at it.
_PARAM_CONFIG_DIR = os.path.dirname(__file__)


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def point_spread_function_cube_single_field(
    ps_store: str,
    image_params: dict[str, Any],
    imaging_weights_params: dict[str, Any],
    image_store: str | None = None,
    instrument_polarization_basis: str = "linear",
    scan_intents: list[str] | None = ["OBSERVE_TARGET#ON_SOURCE"],  # noqa: B006 - None means every intent, so the default stays a list; never mutated
    field_name: str | None = None,
    processing_set_data_group_name: str = "corrected",
    single_precision_image: bool = True,
    gridding_kernel_params: dict[str, Any] | None = None,
    primary_beam_params: dict[str, Any] | None = None,
    psf_fitting_method: str = "astroviper",
    n_mapping_parallelism: dict[str, int | None] | None = None,
    thread_info: dict | None = None,
    processing_function_threads: int = 1,
    overwrite: bool = False,
    compute_backend: str = "dask",
    mpi_cluster_setup: dict[str, Any] | None = None,
):
    """
    Make the point spread function of a spectral cube, with its beam fit and
    the theoretical sensitivity of the imaging weights.

    Only the point spread function is made: no sky, residual or model image
    and no deconvolution, and ``VISIBILITY`` is not loaded (the node tasks
    load only the data group's weight, flag and uvw variables). Every image
    channel goes through the same setup as in
    :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`
    (imaging weights, point spread function, primary beam, Stokes basis,
    Gaussian beam fit), so the beam fit and the largest sidelobe equal what
    the imager records for the same parameters. The cube
    (``POINT_SPREAD_FUNCTION`` and ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION``)
    is written only when ``image_store`` is given.

    Parameters
    ----------
    ps_store : str
        Path of the input processing set (Zarr store). A Measurement Set v2 is
        not supported.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
    imaging_weights_params : dict
        Weighting configuration: ``weighting`` is ``"natural"``, ``"uniform"``,
        ``"briggs"`` or ``"briggs_abs"``; ``robust`` controls the Briggs
        trade-off. ``weighting_scope`` is ``"local"`` for a density per
        frequency channel or ``"global"`` for one density across all selected
        channels in distributed continuum imaging. Natural weighting is
        independent of this scope.
    image_store : str, optional
        Path/URL of the on-disk Zarr image cube.
    instrument_polarization_basis : str
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` or ``"circular"``. The residual update grids and degrids the
        correlations of this basis and the model update deconvolves in the
        Stokes basis, in which the image is written. The Stokes planes requested
        in ``image_params["polarization_coords"]`` fix the correlations that are
        loaded and gridded: the two parallel hands give ``I, Q`` (linear) or
        ``I, V`` (circular), all four correlations give ``I, Q, U, V``. A sample
        is used only if none of its loaded correlations is flagged.
    scan_intents : list[str] or None
        The scan intents to image. ``None`` images every scan intent (a
        Measurement Set v2 from CASA often lacks the default
        ``"OBSERVE_TARGET#ON_SOURCE"``).
    field_name : str, optional
        Not used yet (as in the imager); kept for the interface.
    processing_set_data_group_name : str
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    single_precision_image : bool
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the model
        update runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
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
    psf_fitting_method : str, optional
        Beam-fit algorithm for the PSF: ``"astroviper"`` (default) or
        ``"casa"``, the C++ port of CASA's ``StokesImageUtil::FitGaussianPSF``
        (the fit behind ``tclean``'s restoring beam).
    n_mapping_parallelism : dict, optional
        Mapping parallelism of the distributed graph, as ``{parallel_axis:
        n_chunks}``: the named axis is split into ``n_chunks`` chunks, with one
        node task mapped over each chunk. Cube imaging always maps over
        ``frequency``, so the only allowed key is ``"frequency"`` (e.g.
        ``{"frequency": 500}``). An entry value of ``None``, or omitting the
        parameter entirely (default), auto-determines the chunk count from the
        image size, memory constraints, and available parallelism.
    thread_info : dict, optional
        Thread information as returned by
        :func:`~astroviper.utils.data_partitioning.get_thread_info`. Queried
        automatically when ``None``.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    overwrite : bool
        Whether to overwrite an existing ``image_store``. Default is False.
    compute_backend : str
        Engine used to execute the map/reduce graph. ``"dask"`` (default) builds
        a ``dask.delayed`` graph with
        :func:`graphviper.graph_tools.generate_dask_workflow` and runs
        ``dask.compute``. ``"mpi"`` instead executes the same graph with
        :func:`graphviper.graph_tools.processes_with_mpi` on an
        ``mpi4py.futures`` manager-worker pool; the program must then be launched
        in the static model (``... python -m mpi4py.futures <script>``), where the
        rank running this function is the manager and the other ranks are the
        worker pool. The Dask client is unused in MPI mode.
    mpi_cluster_setup : dict, optional
        Options forwarded to :func:`graphviper.graph_tools.processes_with_mpi`
        when ``compute_backend="mpi"`` (e.g. ``max_workers``, ``chunksize``,
        ``reduce_in_pool``, ``progress_every``). Ignored for the Dask backend.

    Returns
    -------
    xarray.Dataset
        Dims ``frequency`` (the image channels, the same frequencies as the
        imager's image), ``polarization`` (the Stokes planes of
        ``image_params["polarization_coords"]``) and ``beam_params_label``
        (``major``, ``minor``, ``pa``), with the data variables

        - ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION`` (frequency, polarization,
          beam_params_label): the Gaussian fit of the point spread function
          (radians), as in the imager.
        - ``MAX_SIDELOBE_POINT_SPREAD_FUNCTION`` (frequency, polarization): the
          largest sidelobe, as the imager computes it.
        - ``SENSITIVITY`` and ``SENSITIVITY_NATURAL`` (frequency,
          polarization): the theoretical point source noise (Jy/beam) for the
          imaging weights and for natural weights. Per correlation ``c``,
          ``sigma_c = sqrt(sum_k W_ck**2 / w_ck) / sum_k W_ck`` over the
          unflagged cross-correlation samples ``k`` (unique visibilities, no
          conjugate doubling), with ``w`` the natural weight ``WEIGHT =
          1 / sigma_vis**2`` of one real or imaginary component and ``W`` the
          imaging weight (``W = w`` for the natural sensitivity); per Stokes
          plane ``S``, ``sigma_S**2 = sum_c |M_Sc|**2 sigma_c**2`` with ``M``
          the imager's instrument basis to Stokes matrix (``I = (XX + YY) / 2``
          gives ``sigma_I**2 = (sigma_XX**2 + sigma_YY**2) / 4``). An absolute
          noise only when the weights are absolute (a simulation with noise:
          yes; a simulation without noise writes ``WEIGHT = 1``).
        - ``SUM_OF_WEIGHTS`` (frequency, polarization): the sum of the imaging
          weights of the correlations forming the plane, the mean per
          correlation.
        - ``EFFECTIVE_NUMBER_OF_VISIBILITIES`` (frequency, polarization):
          ``(sum W)**2 / sum W**2`` per correlation, the mean over the
          correlations forming the plane.
        - ``UV_DISTANCE_MAX``, ``UV_DISTANCE_MIN`` and ``ABS_W_MAX``
          (frequency): the uv extent of the unflagged cross correlations, in
          wavelengths.
        - ``FLAGGED_FRACTION`` (frequency): the fraction of cross-correlation
          samples excluded (flagged, or a non-finite or non-positive weight).

        Attributes: ``image_size``, ``cell_size`` (radians), ``fft_padding``,
        ``imaging_weights_params``, ``gridding_kernel_params`` (defaults filled
        in) and ``psf_fitting_method``.
    """

    import time

    import dask
    import numpy as np
    import toolviper.utils.logger as logger
    import zarr
    from graphviper.graph_tools import (
        generate_dask_workflow,
        map,
        processes_with_mpi,
        reduce,
    )
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from numcodecs import Blosc
    from xradio.image import make_empty_sky_image
    from xradio.measurement_set import open_processing_set

    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        _validate_n_mapping_parallelism,
        calculate_mapping_parallelism_for_cube_imaging,
    )
    from astroviper.node_tasks.imaging.point_spread_function_cube_single_field import (
        POINT_SPREAD_FUNCTION_IMAGE_VARIABLES,
    )
    from astroviper.node_tasks.imaging.point_spread_function_cube_single_field import (
        point_spread_function_cube_single_field as point_spread_function_node_task,
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        resolve_gridding_kernel_params,
    )
    from astroviper.processing_functions.imaging.utils.imaging_polarization import (
        correlation_selection,
        correlations_for_stokes,
    )
    from astroviper.utils.data_group_tools import modify_data_groups_xds
    from astroviper.utils.data_tree import release_data_tree
    from astroviper.utils.io import (
        create_empty_data_variables_on_disk,
        image_data_groups_for_kept_variables,
        write_zarr_image_store,
    )
    from astroviper.utils.measurement_set_v2 import is_measurement_set_v2

    application_start = time.time()

    # The node tasks load through XRADIO's processing set loader, which can
    # leave VISIBILITY out; a Measurement Set v2 is read only by the imager's
    # skunk works path.
    if is_measurement_set_v2(ps_store):
        raise ValueError(
            f"{ps_store} is a Measurement Set v2; convert it with "
            "xradio.measurement_set.convert_msv2_to_processing_set and use the "
            "processing set."
        )
    if n_mapping_parallelism is not None:
        _validate_n_mapping_parallelism(n_mapping_parallelism)
    # The requested Stokes planes fix the correlations that are loaded and
    # gridded. Checked here, before anything is created.
    correlation_coords = correlations_for_stokes(
        image_params["polarization_coords"], instrument_polarization_basis
    )

    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=image_params["frequency_coords"],
        pol_coords=image_params["polarization_coords"],
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )

    n_frequency_chunks = int(
        calculate_mapping_parallelism_for_cube_imaging(
            img_xds,
            single_precision_image,
            None
            if n_mapping_parallelism is None
            else n_mapping_parallelism["frequency"],
            thread_info,
        )
    )
    parallel_coords = {
        "frequency": make_parallel_coord(
            coord=img_xds.frequency, n_chunks=n_frequency_chunks
        )
    }
    logger.info(
        "Number of frequency chunks ... : "
        + str(len(parallel_coords["frequency"]["data_chunks"]))
    )

    # The point spread function cube is written like the imager's image store:
    # an empty store with the kept variables' data groups, filled by the node
    # tasks one frequency chunk each.
    if image_store is not None:
        img_xds.attrs.get("data_groups", {}).pop("base", None)
        for data_group_name, data_group in image_data_groups_for_kept_variables(
            POINT_SPREAD_FUNCTION_IMAGE_VARIABLES
        ).items():
            modify_data_groups_xds(
                img_xds,
                data_group_out_name=data_group_name,
                data_group_out=data_group,
                description="Created by the point_spread_function_cube_single_field "
                "driver; populated by its node tasks.",
            )
        image_store = write_zarr_image_store(img_xds, image_store, overwrite=overwrite)
        create_empty_data_variables_on_disk(
            image_store,
            POINT_SPREAD_FUNCTION_IMAGE_VARIABLES,
            shape_dict=img_xds.sizes,
            parallel_coords=parallel_coords,
            compressor=Blosc(cname="lz4", clevel=5),
            double_precision=not single_precision_image,
            data_variable_definitions="imaging",
        )

    ps_xdt = open_processing_set(ps_store, scan_intents=scan_intents)
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps_xdt
    )

    # Load only the correlations that are needed (as in the imager).
    for ms_name, ms_xdt in ps_xdt.items():
        selection = correlation_selection(
            ms_xdt.polarization.values, correlation_coords, ms_name
        )
        if selection is None:
            continue
        for task in node_task_data_mapping.values():
            if task["data_selection"].get(ms_name) is not None:
                task["data_selection"][ms_name]["polarization"] = selection

    # The point spread function and the weight statistics read only the data
    # group's weight, flag and uvw variables: VISIBILITY is never loaded.
    load_data_variables = sorted(
        {
            ms_xdt.ds.attrs["data_groups"][processing_set_data_group_name][role]
            for ms_xdt in ps_xdt.values()
            for role in ("weight", "flag", "uvw")
        }
    )

    input_params = {}
    # frequency_coords is not used by the node tasks (they use
    # task_coords["frequency"]["data"]).
    input_params["image_params"] = {
        k: v for k, v in image_params.items() if k != "frequency_coords"
    }
    input_params["imaging_weights_params"] = imaging_weights_params
    input_params["input_data_store"] = ps_store
    input_params["image_store"] = image_store
    input_params["processing_set_data_group_name"] = processing_set_data_group_name
    input_params["instrument_polarization_basis"] = instrument_polarization_basis
    input_params["single_precision_image"] = single_precision_image
    input_params["processing_function_threads"] = processing_function_threads
    input_params["psf_fitting_method"] = psf_fitting_method
    input_params["gridding_kernel_params"] = gridding_kernel_params
    input_params["primary_beam_params"] = primary_beam_params
    input_params["load_data_variables"] = load_data_variables

    viper_graph = map(
        input_data=ps_xdt,
        node_task_data_mapping=node_task_data_mapping,
        node_task=point_spread_function_node_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    viper_graph = reduce(
        viper_graph, combine_point_spread_function_results, {}, mode="tree"
    )

    if compute_backend == "mpi":
        result = processes_with_mpi(viper_graph, mpi_cluster_setup)
    elif compute_backend == "dask":
        result = dask.compute(generate_dask_workflow(viper_graph))[0]
    else:
        raise ValueError(
            f"Unknown compute_backend {compute_backend!r}; expected 'dask' or 'mpi'."
        )

    release_data_tree(ps_xdt)
    ps_xdt = None
    if image_store is not None:
        zarr.consolidate_metadata(image_store)

    # The reduce concatenates the chunks in graph order; put the channels in
    # the image's order.
    properties = result["point_spread_function_properties"].sel(
        frequency=img_xds.frequency.values
    )
    properties.attrs = {
        "image_size": [int(size) for size in image_params["image_size"]],
        "cell_size": [float(cell) for cell in np.asarray(image_params["cell_size"])],
        "fft_padding": float(image_params["fft_padding"]),
        "imaging_weights_params": dict(imaging_weights_params),
        "gridding_kernel_params": resolve_gridding_kernel_params(
            gridding_kernel_params
        ),
        "psf_fitting_method": psf_fitting_method,
    }

    timing_node_tasks = result["timing_node_tasks"]
    logger.info(
        "point_spread_function_cube_single_field: "
        f"{len(timing_node_tasks)} node tasks, slowest "
        f"{timing_node_tasks['T_point_spread_function_task'].max():.2f} s, "
        f"total {time.time() - application_start:.2f} s."
    )
    return properties


def combine_point_spread_function_results(input_data, input_params):
    """Reduce per-chunk point spread function results into one dict.

    Concatenates the one-row timing frames and the per-chunk
    ``point_spread_function_properties`` datasets (disjoint channel ranges)
    along ``frequency``. Returns the same keys, so it composes under tree
    reduction.

    Parameters
    ----------
    input_data : list of dict
        Per-chunk (or partially reduced) results with ``"timing_node_tasks"``
        and ``"point_spread_function_properties"`` keys.
    input_params : dict
        Unused; present for the GraphVIPER reduce signature.

    Returns
    -------
    dict
        ``{"timing_node_tasks": pandas.DataFrame,
        "point_spread_function_properties": xarray.Dataset}``.
    """
    import pandas as pd
    import xarray as xr

    return {
        "timing_node_tasks": pd.concat(
            [result["timing_node_tasks"] for result in input_data],
            ignore_index=True,
        ),
        "point_spread_function_properties": xr.concat(
            [result["point_spread_function_properties"] for result in input_data],
            dim="frequency",
        ),
    }
