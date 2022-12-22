import warnings, time, os, psutil, multiprocessing, re
import dask
import copy
import os
import logging

def viper_local_client(cores=None, memory_limit=None,autorestrictor=False,local_cache=False):

    '''
    local_cache setting is only useful for testing.
    '''

    # setup dask.distributed based multiprocessing environment
    if cores is None: cores = multiprocessing.cpu_count()
    if memory_limit is None: memory_limit = str(round(((psutil.virtual_memory().available / (1024 ** 2))) / cores)) + 'MB'
    
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
        dask.config.set({"distributed.worker.preload-argv": ["--local_cache",local_cache]})
    
    cluster = dask.distributed.LocalCluster(n_workers=cores, threads_per_worker=1, processes=True, memory_limit=memory_limit) #, silence_logs=logging.ERROR #,resources={'GPU': 2}
    client = dask.distributed.Client(cluster)
    client.get_versions(check=True)
    
    '''
    To ensure that number of workers is correct, scale and then wait (use dask jobque script)
    '''
    
    return client


def viper_slurm_cluster_client(cores=None, memory_limit=None,autorestrictor=False,local_cache=False):

    return None
