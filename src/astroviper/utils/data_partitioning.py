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


#: Keys of a distributed application's ``return_dict["compute_resources"]``.
COMPUTE_RESOURCES_KEYS = (
    "compute_backend",
    "scheduler",
    "processing_function_threads",
    "mpi_world_size",
    "n_hosts",
    "n_workers",
    "n_worker_slots",
    "worker_memory_limit_bytes",
    "n_logical_cpus",
    "memory_total_bytes",
    "hosts",
)
#: Keys of one entry of ``compute_resources["hosts"]``.
COMPUTE_RESOURCES_HOST_KEYS = (
    "n_workers",
    "n_worker_slots",
    "worker_memory_limit_bytes",
    "n_logical_cpus",
    "memory_total_bytes",
)
#: Values of ``compute_resources["scheduler"]``.
COMPUTE_RESOURCES_SCHEDULERS = (
    "distributed",
    "threads",
    "processes",
    "synchronous",
    "mpi4py.futures",
)


def _host_compute_resources():
    """The hostname, logical CPUs and memory of the host this runs on."""
    import os
    import socket

    import psutil

    return {
        "hostname": socket.gethostname(),
        "n_logical_cpus": os.cpu_count(),
        "memory_total_bytes": int(psutil.virtual_memory().total),
    }


def _total(values):
    """The sum of ``values``; None when there are none or any is None."""
    values = list(values)
    if not values or any(value is None for value in values):
        return None
    return int(sum(values))


def get_compute_resources(compute_backend, processing_function_threads):
    """What the graphs of a distributed application are computed on.

    Parameters
    ----------
    compute_backend : str
        ``"dask"`` or ``"mpi"``, the engine that computes the graphs.
    processing_function_threads : int or None
        The threads of each node task's processing functions, as the
        application resolved them with
        :func:`resolve_processing_function_threads`; ``None`` for an
        application without that argument.

    Returns
    -------
    dict
        The keys of :data:`COMPUTE_RESOURCES_KEYS`; every value is an int, a
        str, None (not known) or, for ``"hosts"``, a dict:

        ``compute_backend``
            ``"dask"`` or ``"mpi"``.
        ``scheduler``
            Dask: ``"distributed"``, ``"threads"``, ``"processes"`` or
            ``"synchronous"``, None for a scheduler not recognised; MPI:
            ``"mpi4py.futures"``.
        ``processing_function_threads``
            As passed.
        ``mpi_world_size``
            ``MPI.COMM_WORLD.Get_size()``; None for Dask.
        ``n_hosts``
            Hosts with workers; None for MPI.
        ``n_workers``
            Worker processes (MPI: every rank but the manager).
        ``n_worker_slots``
            The tasks that can run at once: the threads of the distributed
            workers, the threads of the local thread pool, the local
            processes, 1 for the synchronous scheduler, the MPI workers.
        ``worker_memory_limit_bytes``
            The sum of the distributed workers' memory limits; None when a
            worker has none, for the local schedulers and for MPI.
        ``n_logical_cpus``, ``memory_total_bytes``
            Logical CPUs and memory summed over the hosts.
        ``hosts``
            ``{hostname: {key: value}}`` with the keys of
            :data:`COMPUTE_RESOURCES_HOST_KEYS`, keyed by
            ``socket.gethostname()`` on the workers (the ``hostname`` of the
            node tasks' timing); the totals above are its sums. None for MPI
            and for a scheduler not recognised.

    Raises
    ------
    ValueError
        For a ``compute_backend`` other than ``"dask"`` or ``"mpi"``.

    Notes
    -----
    An application calls this right before its first compute, so it
    describes the workers at that moment; workers that join or leave later
    are not in it. With a distributed client it reads
    ``Client.scheduler_info`` and runs one function on every worker for its
    host; a worker that fails it raises before the compute. MPI is described
    for the static ``mpi4py.futures`` model, where the pool is every rank but
    the manager. A dask.config ``"pool"``, or the threaded scheduler called
    from a thread other than the main thread without ``num_workers``, gives
    ``n_worker_slots`` None.
    """
    if compute_backend == "mpi":
        from mpi4py import MPI

        mpi_world_size = int(MPI.COMM_WORLD.Get_size())
        n_workers = mpi_world_size - 1 if mpi_world_size > 1 else None
        return {
            "compute_backend": "mpi",
            "scheduler": "mpi4py.futures",
            "processing_function_threads": processing_function_threads,
            "mpi_world_size": mpi_world_size,
            "n_hosts": None,
            "n_workers": n_workers,
            "n_worker_slots": n_workers,
            "worker_memory_limit_bytes": None,
            "n_logical_cpus": None,
            "memory_total_bytes": None,
            "hosts": None,
        }
    if compute_backend != "dask":
        raise ValueError(
            f"Unknown compute_backend {compute_backend!r}; expected 'dask' or 'mpi'."
        )
    import threading

    import dask
    import dask.local
    import dask.multiprocessing
    import dask.threaded
    from dask.base import get_scheduler
    from dask.system import CPU_COUNT
    from distributed import Client

    scheduler_function = get_scheduler()
    client = getattr(scheduler_function, "__self__", None)
    custom_pool = dask.config.get("pool", None) is not None
    num_workers = dask.config.get("num_workers", None)
    hosts = None
    if isinstance(client, Client):
        scheduler = "distributed"
        workers = client.scheduler_info(n_workers=-1)["workers"]
        by_host = {}
        if workers:
            host_of = client.run(_host_compute_resources, workers=list(workers))
            for address, worker in workers.items():
                by_host.setdefault(host_of[address]["hostname"], []).append(
                    (worker, host_of[address])
                )
        hosts = {
            hostname: {
                "n_workers": len(records),
                "n_worker_slots": _total(int(w["nthreads"]) for w, _ in records),
                "worker_memory_limit_bytes": _total(
                    int(w["memory_limit"]) if w.get("memory_limit") else None
                    for w, _ in records
                ),
                "n_logical_cpus": records[0][1]["n_logical_cpus"],
                "memory_total_bytes": records[0][1]["memory_total_bytes"],
            }
            for hostname, records in by_host.items()
        } or None
    else:
        if scheduler_function in (None, dask.threaded.get):
            scheduler, n_workers = "threads", 1
            if custom_pool:
                n_worker_slots = None
            elif num_workers:
                n_worker_slots = int(num_workers)
            elif threading.current_thread() is threading.main_thread():
                n_worker_slots = CPU_COUNT
            else:
                n_worker_slots = None
        elif scheduler_function is dask.multiprocessing.get:
            scheduler = "processes"
            n_workers = n_worker_slots = (
                None if custom_pool else int(num_workers or CPU_COUNT)
            )
        elif scheduler_function is dask.local.get_sync:
            scheduler, n_workers, n_worker_slots = "synchronous", 1, 1
        else:
            scheduler = None
        if scheduler is not None:
            host = _host_compute_resources()
            hosts = {
                host.pop("hostname"): {
                    "n_workers": n_workers,
                    "n_worker_slots": n_worker_slots,
                    "worker_memory_limit_bytes": None,
                    **host,
                }
            }
    values = list(hosts.values()) if hosts else []
    return {
        "compute_backend": "dask",
        "scheduler": scheduler,
        "processing_function_threads": processing_function_threads,
        "mpi_world_size": None,
        "n_hosts": len(hosts) if hosts else None,
        **{key: _total(h[key] for h in values) for key in COMPUTE_RESOURCES_HOST_KEYS},
        "hosts": hosts,
    }


def _host_cpu_count():
    """The hostname and ``dask.system.cpu_count()`` of the host this runs on."""
    import socket

    from dask.system import cpu_count

    return socket.gethostname(), cpu_count()


def resolve_processing_function_threads(processing_function_threads, compute_backend):
    """The ``processing_function_threads`` that a distributed application passes down.

    A distributed application calls this first, before it checks its
    configuration, and passes the count it returns to its node tasks.

    Parameters
    ----------
    processing_function_threads : int
        The count passed to the application; ``0`` asks for auto. Any other
        value, and a value that is not an integer, is returned unchanged.
    compute_backend : str
        ``"dask"`` or ``"mpi"``, the engine that computes the graphs.

    Returns
    -------
    int
        For ``0``: over the hosts of :func:`get_compute_resources`, the
        smallest number of CPUs per worker thread (a host's CPUs divided by
        its ``n_worker_slots``, rounded down), at least 1. A host's CPUs are
        ``dask.system.cpu_count()`` where its tasks run, which counts only
        the CPUs that CPU affinity and cgroup quotas allow: run on the
        workers of a distributed client (the smallest count of a host's
        workers), else ``dask.system.CPU_COUNT``. 1 when there is no host or
        the CPUs or worker threads of a host are not known: MPI, a scheduler
        not recognised, a pool of unknown size, a client without workers.

    Notes
    -----
    Logs the count it resolves for ``0``, and one warning that lists every
    host whose worker threads times the returned count exceed its CPUs; it
    never refuses a count. With a distributed client it calls
    :func:`get_compute_resources` and runs one function on every worker.
    """
    import numbers

    if isinstance(processing_function_threads, bool) or not isinstance(
        processing_function_threads, numbers.Integral
    ):
        return processing_function_threads
    hosts = {}
    if compute_backend == "dask":
        resources = get_compute_resources("dask", processing_function_threads)
        slots = {
            hostname: host["n_worker_slots"]
            for hostname, host in (resources["hosts"] or {}).items()
        }
        if resources["scheduler"] == "distributed":
            from dask.base import get_scheduler

            cpus = {}
            for hostname, n_cpus in (
                get_scheduler().__self__.run(_host_cpu_count).values()
            ):
                cpus[hostname] = min(n_cpus, cpus.get(hostname, n_cpus))
        else:
            from dask.system import CPU_COUNT

            cpus = dict.fromkeys(slots, CPU_COUNT)
        hosts = {hostname: (slots[hostname], cpus.get(hostname)) for hostname in slots}
    known = {
        hostname: (n_slots, n_cpus)
        for hostname, (n_slots, n_cpus) in hosts.items()
        if n_slots is not None and n_cpus is not None
    }
    count = processing_function_threads
    if count == 0:
        if known and len(known) == len(hosts):
            count = max(1, min(n_cpus // n_slots for n_slots, n_cpus in known.values()))
            logger.info(
                f"processing_function_threads=0 resolved to {count}, the CPUs per "
                "Dask worker thread of the host with the fewest."
            )
        else:
            count = 1
            logger.info(
                "processing_function_threads=0 resolved to 1: the CPUs per Dask "
                "worker thread are not known."
            )
    oversubscribed = [
        f"{hostname} ({n_slots} worker threads, {n_cpus} CPUs)"
        for hostname, (n_slots, n_cpus) in known.items()
        if n_slots * count > n_cpus
    ]
    if oversubscribed:
        logger.warning(
            f"processing_function_threads={count} times the Dask worker threads "
            f"exceeds the CPUs of {len(oversubscribed)} "
            f"host{'s' if len(oversubscribed) > 1 else ''}: "
            f"{', '.join(oversubscribed)}."
        )
    return count


def check_compute_resources(compute_resources):
    """Refuse keys and values that :func:`get_compute_resources` does not write.

    Absent keys are allowed: they are not known.

    Parameters
    ----------
    compute_resources : dict
        A ``return_dict["compute_resources"]``, or a part of one.

    Raises
    ------
    ValueError
        For a key outside :data:`COMPUTE_RESOURCES_KEYS`, a ``compute_backend``
        other than ``"dask"``, ``"mpi"`` or None, a ``scheduler`` outside
        :data:`COMPUTE_RESOURCES_SCHEDULERS` and None, or a ``hosts`` entry
        with a key outside :data:`COMPUTE_RESOURCES_HOST_KEYS`.
    """
    unknown = sorted(set(compute_resources) - set(COMPUTE_RESOURCES_KEYS))
    if unknown:
        raise ValueError(
            f"compute_resources has unknown keys {unknown}; the keys are "
            f"{list(COMPUTE_RESOURCES_KEYS)}."
        )
    if compute_resources.get("compute_backend") not in (None, "dask", "mpi"):
        raise ValueError(
            "compute_resources['compute_backend'] must be 'dask', 'mpi' or None; "
            f"got {compute_resources['compute_backend']!r}."
        )
    if compute_resources.get("scheduler") not in (None, *COMPUTE_RESOURCES_SCHEDULERS):
        raise ValueError(
            f"compute_resources['scheduler'] must be one of "
            f"{list(COMPUTE_RESOURCES_SCHEDULERS)} or None; got "
            f"{compute_resources['scheduler']!r}."
        )
    for hostname, host in (compute_resources.get("hosts") or {}).items():
        unknown = sorted(set(host) - set(COMPUTE_RESOURCES_HOST_KEYS))
        if unknown:
            raise ValueError(
                f"compute_resources['hosts'][{hostname!r}] has unknown keys "
                f"{unknown}; the keys are {list(COMPUTE_RESOURCES_HOST_KEYS)}."
            )


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
    assert thread_info["memory_per_thread"] >= (
        constant_memory + memory_singleton_chunk
    ), (
        "Not enough cluster memory per thread. Need at least "
        + str(constant_memory + memory_singleton_chunk)
        + " GiB but only "
        + str(thread_info["memory_per_thread"])
        + " GiB available."
    )

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


def calculate_frequency_chunking(
    n_frequency,
    memory_per_frequency,
    constant_memory,
    n_chunks,
    thread_info=None,
    tasks_per_thread=4,
):
    """Choose frequency mapping parallelism from a shared memory model.

    Parameters
    ----------
    n_frequency : int
        Total number of input frequency channels.
    memory_per_frequency : float
        GiB whose allocation scales with the channels assigned to a task.
    constant_memory : float
        GiB allocated once per task, independent of its channel count.
    n_chunks : int or None
        Explicit task count, returned unchanged when supplied.
    thread_info : dict or None
        Thread count and memory per thread. Queried when omitted.
    tasks_per_thread : int, optional
        Target number of queued tasks per worker thread.

    Returns
    -------
    int
        Number of frequency chunks.
    """
    if n_chunks is not None:
        return n_chunks
    if thread_info is None:
        thread_info = get_thread_info()
    return calculate_data_chunking(
        memory_per_frequency,
        {"frequency": int(n_frequency)},
        thread_info,
        constant_memory=constant_memory,
        tasks_per_thread=tasks_per_thread,
    )["frequency"]
