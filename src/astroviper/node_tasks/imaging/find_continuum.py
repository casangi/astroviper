"""Node tasks of the find_continuum distributed application.

The map task loads one chunk of channels of the dirty cube (and of its primary
beam when a reduction uses it) and returns the partial reductions of
:func:`astroviper.processing_functions.imaging.find_continuum.find_continuum_cube_reductions`;
the reduce task merges them. The two-dimensional images that masks refer to
are the small Zarr images find_continuum writes (:func:`write_small_image`).
"""

import json

import numpy as np


def write_small_image(path, values, mask, metadata):
    """Write a small image (CASA axis order) as a Zarr store.

    Parameters
    ----------
    path : str
        Store path.
    values : numpy.ndarray
        Pixel values, shape ``(l, m, polarization, frequency)``.
    mask : numpy.ndarray
        Pixel mask of the same shape (``True`` = valid).
    metadata : dict
        CASA image metadata (coordinates, beam, units), stored as JSON.
    """
    import xarray as xr

    dims = ["time", "frequency", "polarization", "l", "m"]
    xds = xr.Dataset(
        {
            "SKY": (dims, np.transpose(values, (3, 2, 0, 1))[None]),
            "MASK": (dims, np.transpose(mask, (3, 2, 0, 1))[None]),
        },
        attrs={"casa_metadata": json.dumps(metadata)},
    )
    xds.to_zarr(path, mode="w", consolidated=False)


def read_small_image(path):
    """Read a small image written by :func:`write_small_image`.

    Parameters
    ----------
    path : str
        Store path.

    Returns
    -------
    values : numpy.ndarray
        Pixel values in CASA axis order ``(l, m, polarization, frequency)``.
    mask : numpy.ndarray
        Pixel mask (``True`` = valid).
    metadata : dict
        CASA image metadata.
    """
    import xarray as xr

    xds = xr.open_zarr(path, chunks=None, consolidated=False)
    values = np.transpose(xds["SKY"].values[0], (2, 3, 1, 0))
    mask = np.transpose(xds["MASK"].values[0], (2, 3, 1, 0)).astype(bool)
    return values, mask, json.loads(xds.attrs["casa_metadata"])


def find_continuum_cube_reductions(
    input_image_store,
    variables,
    image_paths,
    requests,
    data_selection=None,
    task_id=None,
):
    """Partial reductions of one chunk of channels of the dirty cube.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store holding the cube.
    variables : dict
        Cube name (as the lattice expressions write it) to the data variable of
        the store that holds it.
    image_paths : dict
        Name of every small image the masks refer to (as the lattice
        expressions write it) to its absolute path.
    requests : list of dict
        See :func:`astroviper.processing_functions.imaging.find_continuum.find_continuum_cube_reductions`.
    data_selection : dict, optional
        ``{"img": {"frequency": slice}}``, the chunk of this task.
    task_id : int, optional
        Graph node id.

    Returns
    -------
    dict
        ``reductions`` (the partial results) and ``timing_node_tasks`` (a
        one-row ``pandas.DataFrame``).
    """
    import os
    import socket
    import threading
    import time

    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.imaging.find_continuum import (
        find_continuum_cube_reductions as reduce_chunk,
    )

    start_unixtime = time.time()
    selection = (data_selection or {"img": {}})["img"]
    lazy_xds = xr.open_zarr(input_image_store, chunks=None)
    img_xds = (
        lazy_xds[sorted(set(variables.values()))]
        .isel(selection)
        .sel(polarization=["I"])
        .load()
    )
    frequency_selection = selection.get("frequency", slice(None))
    channel_index = np.arange(lazy_xds.sizes["frequency"])[frequency_selection]
    images = {}
    for name, path in image_paths.items():
        values, mask, _ = read_small_image(path)
        images[name] = np.where(mask, values, np.nan)
    T_load = time.time() - start_unixtime

    start = time.time()
    partials = reduce_chunk(img_xds, requests, variables, images, channel_index)
    T_reduce = time.time() - start

    try:
        from dask.distributed import get_worker

        worker_name = str(get_worker().name)
    except Exception:
        worker_name = None
    timing_df = pd.DataFrame(
        [
            {
                "task_id": task_id,
                "start_unixtime": start_unixtime,
                "n_channels": channel_index.size,
                "T_load": T_load,
                "T_reduce": T_reduce,
                "T_find_continuum_task": time.time() - start_unixtime,
                "hostname": socket.gethostname(),
                "process_pid": os.getpid(),
                "thread_native_id": threading.get_native_id(),
                "worker_name": worker_name,
            }
        ]
    )
    return {"reductions": partials, "timing_node_tasks": timing_df}


def merge_find_continuum_cube_reductions(input_data, input_params):
    """Reduce node task: merge the outputs of map or reduce nodes.

    Parameters
    ----------
    input_data : list of dict
        Outputs of :func:`find_continuum_cube_reductions` (or of this task).
    input_params : dict
        Unused.

    Returns
    -------
    dict
        Merged ``reductions`` and concatenated ``timing_node_tasks``.
    """
    import pandas as pd

    from astroviper.processing_functions.imaging.find_continuum import (
        merge_find_continuum_reductions,
    )

    merged = input_data[0]["reductions"]
    for other in input_data[1:]:
        merged = merge_find_continuum_reductions(merged, other["reductions"])
    return {
        "reductions": merged,
        "timing_node_tasks": pd.concat(
            [item["timing_node_tasks"] for item in input_data], ignore_index=True
        ),
    }
