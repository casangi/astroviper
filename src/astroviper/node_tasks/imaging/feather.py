"""Node task for the feather workflow.

Thin orchestration layer (one ``input_params`` dict in, one value out): it loads
this frequency chunk's single-dish and interferometer images from disk, calls
the science function
:func:`astroviper.processing_functions.imaging.feather.feather_core`, and writes
the resulting ``SKY`` slice into the pre-allocated Zarr image store using the
parallel chunk-writer
:func:`astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.  This mirrors
the ``image_cube_single_field`` node task.
"""


def _rename_beam_to_convention(xds):
    """Normalize older XRADIO beam naming to the BEAM_FIT_PARAMS_SKY convention.

    Some on-disk images use the legacy ``BEAM`` data variable and ``beam_param``
    coordinate; rename them to ``BEAM_FIT_PARAMS_SKY`` / ``beam_params_label``.
    """
    if "BEAM" in xds:
        xds = xds.rename(
            {"BEAM": "BEAM_FIT_PARAMS_SKY", "beam_param": "beam_params_label"}
        )
    if "BEAM_FIT_PARAMS" in xds:
        xds = xds.rename({"BEAM_FIT_PARAMS": "BEAM_FIT_PARAMS_SKY"})
    return xds


def feather(input_params, graph_mode=True):
    """Feather one frequency chunk and write it to the Zarr image store.

    Parameters
    ----------
    input_params : dict
        Parameters injected by the graph framework (``task_coords``,
        ``data_selection``, ``task_id``, ...) merged with the feather parameters
        set by the driver: ``input_data_store`` (``{"sd": path, "int": path}``),
        ``image_store``, ``image_data_variables_keep`` (``["sky"]``),
        ``sdfactor``, ``axes``, ``selection``, ``fft_backend`` and
        ``processing_function_threads``.
    graph_mode : bool, optional
        If ``True`` (default) the ``SKY`` slice is written into the pre-allocated
        Zarr store with
        :func:`~astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.  If
        ``False`` the whole chunk image is written with ``to_zarr``.

    Returns
    -------
    pandas.DataFrame
        One-row timing frame with ``T_load``, ``T_feather``, ``T_write`` and the
        total ``T_feather_task`` (plus ``task_id``).
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger
    from toolviper.utils.memory_management import free_memory, get_rss_gb
    from xradio.image import load_image

    import astroviper.processing_functions as pf

    task_start = time.time()
    # No per-task allocator tuning (mirrors the image_cube node task): mallopt
    # would disable glibc's dynamic mmap-threshold adaptation; the allocator
    # environment is set once per worker instead.

    logger.debug("Memory at start of feather node task: " + str(get_rss_gb()) + " GB")

    selection = input_params.get("selection", {})

    # Load this chunk's single-dish ("sd") and interferometer ("int") images.
    # load_image with block_des reads only this task's frequency slice; .load()
    # materializes it to NumPy (processing functions operate on NumPy, not Dask).
    start = time.time()
    xdses = {}
    for key in ("sd", "int"):
        block_des = {**selection, **input_params["data_selection"][key]}
        xds = load_image(input_params["input_data_store"][key], block_des=block_des)
        xdses[key] = _rename_beam_to_convention(xds).load()
    T_load = time.time() - start

    start = time.time()
    featherd_img_xds = pf.imaging.feather_core(
        xdses["sd"],
        xdses["int"],
        sdfactor=input_params["sdfactor"],
        axes=input_params["axes"],
        fft_backend=input_params.get("fft_backend", "scipy"),
        processing_function_threads=input_params.get("processing_function_threads", 1),
    )
    T_feather = time.time() - start

    start = time.time()
    if graph_mode:
        from astroviper.utils.io import write_result_chunk_to_disk_using_zarr

        write_result_chunk_to_disk_using_zarr(
            input_params["image_store"],
            input_params["image_data_variables_keep"],
            input_params["task_coords"],
            featherd_img_xds,
        )
    else:
        featherd_img_xds.to_zarr(input_params["image_store"], consolidated=True)
    T_write = time.time() - start

    featherd_img_xds = None
    xdses = None
    free_memory()

    logger.debug("Memory after feather node task: " + str(get_rss_gb()) + " GB")

    return pd.DataFrame(
        [
            {
                "task_id": input_params.get("task_id"),
                "T_load": T_load,
                "T_feather": T_feather,
                "T_write": T_write,
                "T_feather_task": time.time() - task_start,
            }
        ]
    )
