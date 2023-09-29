import warnings, time, os, psutil, multiprocessing, re
import dask
import copy
import os
import logging
import astroviper
import distributed
from astroviper._utils._parm_utils._check_logger_parms import (
    _check_logger_parms,
    _check_worker_logger_parms,
)
from astroviper._utils._logger import (
    _setup_logger,
    _get_logger,
    _setup_logger,
)
from astroviper._concurrency._dask._worker import (
    _worker,
)  # _worker_logger_plugin


def local_client(
    cores=None,
    memory_limit=None,
    autorestrictor=False,
    dask_local_dir=None,
    local_dir=None,
    wait_for_workers=True,
    log_parms={},
    worker_log_parms={},
):
    """
    local_dir setting is only useful for testing since this function creates a local cluster. slurm_cluster_client should be used for a multinode cluster.

    https://github.com/dask/dask/issues/5577
    log_parms['log_to_term'] = True/False
    log_parms['log_file'] = True/False
    log_parms['log_level'] =
    """

    _log_parms = copy.deepcopy(log_parms)
    _worker_log_parms = copy.deepcopy(worker_log_parms)

    assert _check_logger_parms(
        _log_parms
    ), "######### ERROR: initialize_processing log_parms checking failed."
    
    if _worker_log_parms is not None:
        assert _check_worker_logger_parms(
            _worker_log_parms
        ), "######### ERROR: initialize_processing log_parms checking failed."

    if local_dir:
        os.environ["VIPER_LOCAL_DIR"] = local_dir
        local_cache = True
    else:
        local_cache = False

    print(_log_parms)
    _setup_logger(**_log_parms)
    logger = _get_logger()

    _set_up_dask(dask_local_dir)

    #viper_path = astroviper.__path__.__dict__["_path"][0]
    viper_path = astroviper.__path__[0]
    if local_cache or autorestrictor:
        dask.config.set(
            {
                "distributed.scheduler.preload": os.path.join(
                    viper_path, "_concurrency/_dask/_scheduler.py"
                )
            }
        )
        dask.config.set(
            {
                "distributed.scheduler.preload-argv": [
                    "--local_cache",
                    local_cache,
                    "--autorestrictor",
                    autorestrictor,
                ]
            }
        )

    """ This method of assigning a worker plugin does not seem to work when using dask_jobqueue. Consequently using client.register_worker_plugin so that the method of assigning a worker plugin is the same for local_client and slurm_cluster_client.
    if local_cache or _worker_log_parms:
        dask.config.set({"distributed.worker.preload": os.path.join(viper_path,'_utils/_worker.py')})
        dask.config.set({"distributed.worker.preload-argv": ["--local_cache",local_cache,"--log_to_term",_worker_log_parms['log_to_term'],"--log_to_file",_worker_log_parms['log_to_file'],"--log_file",_worker_log_parms['log_file'],"--log_level",_worker_log_parms['log_level']]})
    """
    # setup distributed based multiprocessing environment
    if cores is None:
        cores = multiprocessing.cpu_count()
    if memory_limit is None:
        memory_limit = (
            str(round(((psutil.virtual_memory().available / (1024**2))) / cores))
            + "MB"
        )
    cluster = distributed.LocalCluster(
        n_workers=cores, threads_per_worker=1, processes=True, memory_limit=memory_limit
    )  # , silence_logs=logging.ERROR #,resources={'GPU': 2}
    client = distributed.Client(cluster)
    client.get_versions(check=True)

    """
    When constructing a graph that has local cache enabled all workers need to be up and running.
    """
    if local_cache or wait_for_workers:
        client.wait_for_workers(n_workers=cores)

    if local_cache or _worker_log_parms:
        plugin = _worker(local_cache, _worker_log_parms)
        client.register_worker_plugin(plugin, name="viper_worker")

    logger.info("Created client " + str(client))

    return client


def slurm_cluster_client(
    workers_per_node,
    cores_per_node,
    memory_per_node,
    number_of_nodes,
    queue,
    interface,
    python_env_dir,
    dask_local_dir,
    dask_log_dir,
    exclude_nodes="nmpost090",
    dashboard_port=9000,
    local_dir=None,
    autorestrictor=False,
    wait_for_workers=True,
    log_parms={},
    worker_log_parms={},
):
    """
    local_cache setting is only useful for testing since this function creates a local cluster. slurm_cluster_client should be used for a multinode cluster.

    https://github.com/dask/dask/issues/5577
    log_parms['log_to_term'] = True/False
    log_parms['log_file'] = True/False
    log_parms['log_level'] =

    interface eth0, ib0
    python "/mnt/condor/jsteeb/viper_py/bin/python"
    dask_local_dir "/mnt/condor/jsteeb"
    dask_log_dir "/.lustre/aoc/projects/ngvla/viper/ngvla_sim",
    """

    from dask_jobqueue import SLURMCluster
    from distributed import Client, config, performance_report

    _log_parms = copy.deepcopy(log_parms)
    _worker_log_parms = copy.deepcopy(worker_log_parms)

    assert _check_logger_parms(
        _log_parms
    ), "######### ERROR: initialize_processing log_parms checking failed."
    assert _check_worker_logger_parms(
        _worker_log_parms
    ), "######### ERROR: initialize_processing log_parms checking failed."

    if local_dir:
        os.environ["VIPER_LOCAL_DIR"] = local_dir
        local_cache = True
    else:
        local_cache = False

    # Viper logger for code that is not part of the Dask graph. The worker logger is setup in the _viper_worker plugin.
    from viper._utils._logger import _setup_logger

    _setup_logger(**_log_parms)
    logger = _get_logger()

    _set_up_dask(dask_local_dir)

    viper_path = astroviper.__path__.__dict__["_path"][0]
    if local_cache or autorestrictor:
        dask.config.set(
            {
                "distributed.scheduler.preload": os.path.join(
                    viper_path, "_concurrency/_dask/_scheduler.py"
                )
            }
        )
        dask.config.set(
            {
                "distributed.scheduler.preload-argv": [
                    "--local_cache",
                    local_cache,
                    "--autorestrictor",
                    autorestrictor,
                ]
            }
        )

    """ This method of assigning a worker plugin does not seem to work when using dask_jobqueue. Consequently using client.register_worker_plugin so that the method of assigning a worker plugin is the same for local_client and slurm_cluster_client.
    if local_cache or _worker_log_parms:
        dask.config.set({"distributed.worker.preload": os.path.join(viper_path,'_utils/_worker.py')})
        dask.config.set({"distributed.worker.preload-argv": ["--local_cache",local_cache,"--log_to_term",_worker_log_parms['log_to_term'],"--log_to_file",_worker_log_parms['log_to_file'],"--log_file",_worker_log_parms['log_file'],"--log_level",_worker_log_parms['log_level']]})
    """

    cluster = SLURMCluster(
        processes=workers_per_node,
        cores=cores_per_node,
        interface=interface,
        memory=memory_per_node,
        walltime="24:00:00",
        queue=queue,
        name="viper",
        python=python_env_dir,  # "/mnt/condor/jsteeb/viper_py/bin/python", #"/.lustre/aoc/projects/ngvla/viper/viper_py_env/bin/python",
        local_directory=dask_local_dir,  # "/mnt/condor/jsteeb",
        log_directory=dask_log_dir,
        job_extra_directives=["--exclude=" + exclude_nodes],
        # job_extra_directives=["--exclude=nmpost087,nmpost089,nmpost088"],
        scheduler_options={"dashboard_address": ":" + str(dashboard_port)},
    )  # interface='ib0'

    client = Client(cluster)

    cluster.scale(workers_per_node * number_of_nodes)

    """
    When constructing a graph that has local cache enabled all workers need to be up and running.
    """
    if local_cache or wait_for_workers:
        client.wait_for_workers(n_workers=workers_per_node * number_of_nodes)

    if local_cache or _worker_log_parms:
        plugin = _worker(local_cache, _worker_log_parms)
        client.register_worker_plugin(plugin, name="viper_worker")

    logger.info("Created client " + str(client))

    return client


def _set_up_dask(local_directory):
    if local_directory:
        dask.config.set({"temporary_directory": local_directory})
    dask.config.set({"distributed.scheduler.allowed-failures": 10})
    dask.config.set({"distributed.scheduler.work-stealing": True})
    dask.config.set({"distributed.scheduler.unknown-task-duration": "99m"})
    dask.config.set({"distributed.worker.memory.pause": False})
    dask.config.set({"distributed.worker.memory.terminate": False})
    # dask.config.set({"distributed.worker.memory.recent-to-old-time": '999s'})
    dask.config.set({"distributed.comm.timeouts.connect": "3600s"})
    dask.config.set({"distributed.comm.timeouts.tcp": "3600s"})
    dask.config.set({"distributed.nanny.environ.OMP_NUM_THREADS": 1})
    dask.config.set({"distributed.nanny.environ.MKL_NUM_THREADS": 1})
    # https://docs.dask.org/en/stable/how-to/customize-initialization.html
