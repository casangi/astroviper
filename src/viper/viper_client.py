import warnings, time, os, psutil, multiprocessing, re
import dask
import copy
import os
import logging
from viper._utils._parm_utils._check_logger_parms import _check_logger_parms, _check_worker_logger_parms
from viper._utils._viper_logger import  _setup_viper_logger

def viper_local_client(cores=None, memory_limit=None,autorestrictor=False,local_cache=False,wait_for_workers=True, log_parms={}, worker_log_parms={},local_directory=None):

    '''
    local_cache setting is only useful for testing since this function creates a local cluster. viper_slurm_cluster_client should be used for a multinode cluster.

    https://github.com/dask/dask/issues/5577
    log_parms['log_to_term'] = True/False
    log_parms['log_file'] = True/False
    log_parms['level'] =
    '''
    
    _log_parms = copy.deepcopy(log_parms)
    _worker_log_parms = copy.deepcopy(worker_log_parms)
    
    assert(_check_logger_parms(_log_parms)), "######### ERROR: initialize_processing log_parms checking failed."
    assert(_check_worker_logger_parms(_worker_log_parms)), "######### ERROR: initialize_processing log_parms checking failed."
    
    # setup dask.distributed based multiprocessing environment
    if cores is None: cores = multiprocessing.cpu_count()
    if memory_limit is None: memory_limit = str(round(((psutil.virtual_memory().available / (1024 ** 2))) / cores)) + 'MB'
    
    if local_directory: dask.config.set({"temporary_directory": local_directory})
    dask.config.set({"distributed.scheduler.allowed-failures": 10})
    dask.config.set({"distributed.scheduler.work-stealing": True})
    dask.config.set({"distributed.scheduler.unknown-task-duration": '99m'})
    dask.config.set({"distributed.worker.memory.pause": False})
    dask.config.set({"distributed.worker.memory.terminate": False})
    #dask.config.set({"distributed.worker.memory.recent-to-old-time": '999s'})
    dask.config.set({"distributed.comm.timeouts.connect": '3600s'})
    dask.config.set({"distributed.comm.timeouts.tcp": '3600s'})
    dask.config.set({"distributed.nanny.environ.OMP_NUM_THREADS": 1})
    dask.config.set({"distributed.nanny.environ.MKL_NUM_THREADS": 1})
    #https://docs.dask.org/en/stable/how-to/customize-initialization.html
 

    import viper
    viper_path = viper.__path__.__dict__["_path"][0]
    print(viper_path)
    

    
    if local_cache or autorestrictor:
        dask.config.set({"distributed.scheduler.preload": os.path.join(viper_path,'_utils/_viper_scheduler.py')})
        dask.config.set({"distributed.scheduler.preload-argv": ["--local_cache",local_cache,"--autorestrictor",autorestrictor]})
    
    if local_cache:
        dask.config.set({"distributed.worker.preload": os.path.join(viper_path,'_utils/_viper_worker.py')})
        dask.config.set({"distributed.worker.preload-argv": ["--local_cache",local_cache]}) #,"--log_parms",_worker_log_parms
    
    cluster = dask.distributed.LocalCluster(n_workers=cores, threads_per_worker=1, processes=True, memory_limit=memory_limit) #, silence_logs=logging.ERROR #,resources={'GPU': 2}
    client = dask.distributed.Client(cluster)
    client.get_versions(check=True)

    '''
    When constructing a graph that has local cache enabled all workers need to be up and running.
    '''
    if local_cache or wait_for_workers:
        client.wait_for_workers(n_workers=cores)
        
        
        
    _setup_viper_logger(log_to_term=_log_parms['log_to_term'],log_to_file=_log_parms['log_to_file'],log_file=_log_parms['log_file'], level=_log_parms['log_level'])
    #worker_logger = _viper_worker_logger_plugin(_worker_log_parms)
    #client.register_worker_plugin(plugin=worker_logger, name='viper_worker_logger')
    
    return client


def viper_slurm_cluster_client(cores=None, memory_limit=None,autorestrictor=False,local_cache=False):
    #print('Scaling cluster to ', nodes, 'nodes.')
    #cluster.scale(nodes)


    return None
