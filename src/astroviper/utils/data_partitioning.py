import toolviper.utils.logger as logger

bytes_in_dtype = {
    "float32": 4,
    "float64": 8,
    "double": 8,
    "complex64": 8,
    "complex128": 16,
}


def get_thread_info(client=None):
    """Query available threads and memory from the Dask cluster or local machine.

    If no client is provided, attempts to connect to the current distributed
    ``Client``. Falls back to ``psutil`` when no distributed scheduler is running.

    Parameters
    ----------
    client : distributed.Client, optional
        An active Dask distributed ``Client``. If ``None``, the current client
        is retrieved automatically; if no distributed client exists, ``psutil``
        is used to query the local machine.

    Returns
    -------
    dict
        Dictionary with keys:

        ``n_threads`` : int
            Total number of worker threads available.
        ``memory_per_thread`` : float
            Memory available per thread in GiB (minimum across all workers).
    """

    if client is None:
        try:
            from distributed import Client

            client = Client.current()
        except Exception:  # Using default Dask schedular.
            import psutil

            cpu_cores = psutil.cpu_count()
            total_memory = psutil.virtual_memory().total / (1024**3)
            # print(cpu_cores,total_memory)
            thread_info = {
                "n_threads": cpu_cores,
                "memory_per_thread": total_memory / cpu_cores,
            }
            return thread_info

    memory_per_thread = -1
    n_threads = 0

    # client.cluster only exists for LocalCluster
    if client.cluster is None:
        worker_items = client.scheduler_info()["workers"].items()
    else:
        worker_items = client.cluster.scheduler_info["workers"].items()

    for _worker_name, worker in worker_items:
        temp_memory_per_thread = (worker["memory_limit"] / worker["nthreads"]) / (
            1024**3
        )
        n_threads = n_threads + worker["nthreads"]
        if (memory_per_thread == -1) or (memory_per_thread > temp_memory_per_thread):
            memory_per_thread = temp_memory_per_thread

    thread_info = {"n_threads": n_threads, "memory_per_thread": memory_per_thread}
    return thread_info


def prime_factors(n):
    """Return the prime factorization of a positive integer.

    Parameters
    ----------
    n : int
        Positive integer to factorize.

    Returns
    -------
    list of int
        Prime factors of ``n`` in non-decreasing order.
        Returns an empty list for ``n == 1``.

    Examples
    --------
    >>> prime_factors(12)
    [2, 2, 3]
    >>> prime_factors(7)
    [7]
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def calculate_data_chunking(
    memory_singleton_chunk,
    chunking_dims_sizes,
    thread_info,
    constant_memory=0,
    tasks_per_thread=4,
):
    """Determine the number of chunks per dimension for parallel processing.

    The chunk count satisfies two constraints:

    * **Memory constraint** – the number of chunks must be large enough that
      each chunk fits within the memory available per thread.
    * **Graph constraint** – the number of chunks should be large enough to
      keep all threads busy (``n_threads * tasks_per_thread``).

    Parameters
    ----------
    memory_singleton_chunk : float
        Memory in GiB required to process a single chunk when all parallelized
        dimensions have size 1.
    chunking_dims_sizes : dict
        Mapping of dimension name to the total number of elements along that
        dimension (e.g. ``{"frequency": 5}``).
    thread_info : dict
        Thread information as returned by :func:`get_thread_info`.
        Expected keys: ``"n_threads"`` (int), ``"memory_per_thread"`` (float, GiB).
    constant_memory : float, optional
        Fixed memory overhead in GiB consumed regardless of chunk size.
        Default is ``0``.
    tasks_per_thread : int, optional
        Target number of tasks queued per thread to keep the scheduler busy.
        Default is ``4``.

    Returns
    -------
    dict
        Mapping of dimension name to the recommended number of chunks along
        that dimension (e.g. ``{"frequency": 5}``). Each count is between 1
        and the size of its dimension, and the counts multiply to at least the
        number of chunks the two constraints require.

    Raises
    ------
    AssertionError
        If the cluster does not have enough memory per thread to process even
        a single chunk.
    """

    import numpy as np

    n_total_dims = np.prod(np.array(list(chunking_dims_sizes.values())))

    n_mem_chunks = int(
        np.ceil(
            n_total_dims
            / (
                (thread_info["memory_per_thread"] - constant_memory)
                / memory_singleton_chunk
            )
        )
    )
    assert n_mem_chunks <= n_total_dims, (
        "Not enough cluster memory per thread. Need at least "
        + str(memory_singleton_chunk)
        + " GiB but only "
        + str(thread_info["memory_per_thread"])
        + " GiB available."
    )

    n_graph_chunks = int(thread_info["n_threads"] * tasks_per_thread)
    if n_graph_chunks > n_total_dims:
        n_graph_chunks = n_total_dims

    if n_mem_chunks > n_graph_chunks:
        n_chunks = n_mem_chunks
    else:
        n_chunks = n_graph_chunks

    logger.debug(
        "Suggest n_chunks: "
        + str(n_chunks)
        + ", n_mem_chunks: "
        + str(n_mem_chunks)
        + ", n_graph_chunks: "
        + str(n_graph_chunks)
    )

    dims_sizes = []
    dims_names = []

    for dim_name in sorted(chunking_dims_sizes, key=chunking_dims_sizes.get):
        dims_sizes.append(chunking_dims_sizes[dim_name])
        dims_names.append(dim_name)

    dims_sizes_arr = np.array(dims_sizes)
    dims_names_arr = np.array(dims_names)

    if len(dims_names_arr) == 1:
        n_chunks_dict = dict(zip([dims_names_arr[0]], [n_chunks], strict=False))
        return n_chunks_dict
    else:
        # Place the prime factors of the target, largest first, on the axis with
        # the longest chunks, preferring axes the new count divides evenly. An axis
        # can take several factors but not more chunks than its size; if a factor
        # fits on no axis, try the next target, up to one chunk per sample.
        for n_target in range(n_chunks, n_total_dims + 1):
            dims_n_chunks_arr = np.ones(len(dims_sizes_arr), dtype=int)
            for factor in reversed(prime_factors(n_target)):
                new_n_chunks_arr = dims_n_chunks_arr * factor
                has_room = new_n_chunks_arr <= dims_sizes_arr
                if not has_room.any():
                    break
                divides = dims_sizes_arr % new_n_chunks_arr == 0
                chunk_lengths = np.where(
                    divides if divides.any() else has_room,
                    dims_sizes_arr / dims_n_chunks_arr,
                    0,
                )
                dims_n_chunks_arr[np.argmax(chunk_lengths)] *= factor
            if np.prod(dims_n_chunks_arr) == n_target:
                break
        else:
            dims_n_chunks_arr = dims_sizes_arr

        n_chunks_dict = dict(
            zip(dims_names_arr, dims_n_chunks_arr.tolist(), strict=False)
        )
        return n_chunks_dict
