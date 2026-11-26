"""Node task: continuum subtraction of one block of times of one MSv4."""

from __future__ import annotations

import os
import socket
import time

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def uv_continuum_subtraction(
    ps_store: str,
    ms_name: str,
    continuum_frequency_ranges,
    continuum_frequency_frame: str,
    fit_order: int,
    ms_data_group_in_name: str,
    ms_data_group_out_name: str,
    ms_data_group_out_modified: dict | None = None,
    doppler_factors=None,
    task_coords: dict | None = None,
    data_selection: dict | None = None,
    task_id: int = 0,
    *,
    input_checked: bool = False,
) -> dict:
    """Subtract the continuum of one block of times of one MSv4 and write it.

    Loads the input data group of the task's times (all baselines, channels
    and polarizations) with XRADIO, subtracts the continuum
    (:func:`~astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction.uv_continuum_subtraction`)
    and region-writes the output correlated data and flags into the variables
    the distributed application created in the MSv4.

    Parameters
    ----------
    ps_store : str
        Path of the processing set (Zarr) with the output variables created.
    ms_name : str
        Name of the MSv4 in the processing set.
    continuum_frequency_ranges : list or numpy.ndarray
        The continuum ranges, ``[low, high]`` pairs of frequencies (Hz) in
        ``continuum_frequency_frame``; a channel is a continuum channel at the
        times its frequency, converted to that frame, lies in a range, edges
        included (the ``continuum_frequency_ranges`` of ``find_continuum_ultra``
        are the outer edges of its continuum channels).
    continuum_frequency_frame : str
        Spectral frame of ``continuum_frequency_ranges``, by its XRADIO
        observer name: the frame of the measurement set, or, for a ``"TOPO"``
        measurement set, ``"lsrk"``, ``"lsrd"`` or ``"BARY"``, to which the
        channel frequencies are converted at every time with the Doppler factor
        of ``ms_spectral_frame_conversion.doppler_factor`` (the telescope site
        at rest on the Earth, the phase centre of the first field).
    fit_order : int
        Order of the polynomial fitted to the continuum channels of every
        spectrum [CASA ``fitorder``]; lowered to ``n - 1`` for a spectrum with
        ``n <= fit_order`` valid continuum channels.
    ms_data_group_in_name : str
        Measurement-set data group whose correlated data are continuum
        subtracted, with its flags and weights.
    ms_data_group_out_name : str
        Measurement-set data group of the continuum-subtracted visibilities.
    ms_data_group_out_modified : dict, optional
        The output variables of the roles ``correlated_data`` and ``flag``;
        ``None`` gives ``{"correlated_data": "VISIBILITY_CONTINUUM_SUBTRACTED",
        "flag": "FLAG_CONTINUUM_SUBTRACTED"}``. Neither may be a variable of
        another data group, the input group included.
    doppler_factors : numpy.ndarray, optional
        Doppler factors of all times of the MSv4 from the frame of the
        measurement set to the frame of the ranges, computed once by the
        distributed application; ``None`` when the frames are equal.
    task_coords : dict
        GraphVIPER task coordinates; ``task_coords["time"]["slice"]`` is the
        task's block of times of the MSv4.
    data_selection : dict
        ``{ms_name: {"time": slice}}``, the task's selection.
    task_id : int
        Identifier of the parallel chunk being processed.
    input_checked : bool, default False
        Skip the configuration checks done by the calling layer. The checks of
        the measurement sets always run.

    Returns
    -------
    dict
        ``timing_node_tasks`` (``pandas.DataFrame``, one row: ``task_id``,
        ``ms_name``, ``n_time``, ``T_load``, ``T_continuum_channel_mask``,
        ``T_fit_and_subtract``, ``T_spectrum_sums``, ``T_write``,
        ``T_uv_continuum_subtraction_task``, the wall clock anchor
        ``start_unixtime`` and the execution identity ``hostname``,
        ``process_pid``, ``thread_native_id``, ``worker_name``);
        ``fit_statistics`` (``pandas.DataFrame``, one row per polarization:
        ``ms_name``, ``task_id``, ``polarization`` and the statistics of the
        fits); ``spectrum_sums`` (``{ms_name: dict}`` with ``sum_weight_in``,
        ``sum_weighted_visibility_in``, ``sum_weight_out`` and
        ``sum_weighted_visibility_out``, ``(frequency, polarization)``, see
        :func:`~astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction.weighted_spectrum_sums`).
    """
    import threading

    import pandas as pd
    import toolviper.utils.logger as logger
    from xradio.measurement_set import load_processing_set

    from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
        DEFAULT_MS_DATA_GROUP_OUT_MODIFIED,
        weighted_spectrum_sums,
    )
    from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
        uv_continuum_subtraction as uv_continuum_subtraction_pf,
    )
    from astroviper.utils.measurement_set_tools import write_visibility_chunk_to_disk
    from astroviper.utils.node_task_io import _stored_data_group

    task_start = time.time()
    out_modified = dict(
        ms_data_group_out_modified or DEFAULT_MS_DATA_GROUP_OUT_MODIFIED
    )
    time_slice = task_coords["time"]["slice"]
    # Only the variables the fit reads: the output variables are in no data
    # group until the run ends, so the data group selection would keep them.
    stored_group = _stored_data_group(ps_store, ms_name, ms_data_group_in_name)
    ms_xdt = load_processing_set(
        ps_store,
        sel_parms={ms_name: data_selection[ms_name]},
        data_group_name=ms_data_group_in_name,
        include_variables=[
            stored_group[role] for role in ("correlated_data", "flag", "weight")
        ],
        load_sub_datasets=False,
    )[ms_name]
    timing = {"T_load": time.time() - task_start}
    logger.debug(
        f"uv_continuum_subtraction task {task_id}: {ms_name}, "
        f"{ms_xdt.sizes['time']} times"
    )

    result = uv_continuum_subtraction_pf(
        ms_xdt,
        continuum_frequency_ranges,
        continuum_frequency_frame,
        fit_order,
        ms_data_group_in_name=ms_data_group_in_name,
        ms_data_group_out_name=ms_data_group_out_name,
        ms_data_group_out_modified=out_modified,
        overwrite=True,  # the distributed application checked the outputs
        doppler_factors=None
        if doppler_factors is None
        else doppler_factors[time_slice],
        input_checked=input_checked,
    )
    timing["T_continuum_channel_mask"] = result["T_continuum_channel_mask"]
    timing["T_fit_and_subtract"] = result["T_fit_and_subtract"]

    start = time.time()
    data_group_in = ms_xdt.attrs["data_groups"][ms_data_group_in_name]
    data_group_out = ms_xdt.attrs["data_groups"][ms_data_group_out_name]
    weight = ms_xdt[data_group_in["weight"]].values
    sum_weighted_in, sum_weight_in = weighted_spectrum_sums(
        ms_xdt[data_group_in["correlated_data"]].values,
        ms_xdt[data_group_in["flag"]].values,
        weight,
    )
    sum_weighted_out, sum_weight_out = weighted_spectrum_sums(
        ms_xdt[data_group_out["correlated_data"]].values,
        ms_xdt[data_group_out["flag"]].values,
        weight,
    )
    timing["T_spectrum_sums"] = time.time() - start

    start = time.time()
    output = [data_group_out["correlated_data"], data_group_out["flag"]]
    write_visibility_chunk_to_disk(
        os.path.join(ps_store, ms_name), task_coords, ms_xdt.ds[output]
    )
    timing["T_write"] = time.time() - start
    timing["T_uv_continuum_subtraction_task"] = time.time() - task_start

    # The Dask worker this task ran on (None outside a Dask worker, e.g. MPI).
    try:
        from distributed import get_worker

        worker_name = str(get_worker().name)
    except Exception:
        worker_name = None
    row = {
        "task_id": task_id,
        "ms_name": ms_name,
        "n_time": int(ms_xdt.sizes["time"]),
        **timing,
        "start_unixtime": task_start,
        "hostname": socket.gethostname(),
        "process_pid": os.getpid(),
        "thread_native_id": threading.get_native_id(),
        "worker_name": worker_name,
    }
    fit_statistics = pd.DataFrame(result["fit_statistics"])
    fit_statistics.insert(0, "task_id", task_id)
    fit_statistics.insert(0, "ms_name", ms_name)
    return {
        "timing_node_tasks": pd.DataFrame([row]),
        "fit_statistics": fit_statistics,
        "spectrum_sums": {
            ms_name: {
                "sum_weight_in": sum_weight_in,
                "sum_weighted_visibility_in": sum_weighted_in,
                "sum_weight_out": sum_weight_out,
                "sum_weighted_visibility_out": sum_weighted_out,
            }
        },
    }
