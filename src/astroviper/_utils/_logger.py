logger_name = "viper"
import sys
import logging
from datetime import datetime

formatter = logging.Formatter(
    "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)
from dask.distributed import get_worker
from dask.distributed import WorkerPlugin
import dask


def _get_logger(name=logger_name):
    """
    Will first try to get worker logger. If this fails graph construction logger is returned.
    """
    from dask.distributed import get_worker

    try:
        worker = get_worker()
    except:
        return logging.getLogger(name)

    try:
        logger = worker.plugins["viper_worker"].get_logger()
        return logger
    except:
        return logging.getLogger()


def _setup_logger(
    log_to_term=False,
    log_to_file=True,
    log_file="viper_",
    log_level="INFO",
    name=logger_name,
):
    """To setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(log_level))

    if log_to_term:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_to_file:
        log_file = log_file + datetime.today().strftime("%Y%m%d_%H%M%S") + ".log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def _get_worker_logger_name(name=logger_name):
    worker_log_name = name + "_" + str(get_worker().id)
    return worker_log_name


"""
class _viper_worker_logger_plugin(WorkerPlugin):
    def __init__(self,log_parms):
        self.log_to_term=log_parms['log_to_term']
        self.log_to_file=log_parms['log_to_file']
        self.log_file=log_parms['log_file']
        self.level=log_parms['log_level']
        self.logger = None
        print(self.log_to_term,self.log_to_file,self.log_file,self.log_level)
        
    def get_logger(self):
        return self.logger
        
    def setup(self, worker: dask.distributed.Worker):
        "Run when the plugin is attached to a worker. This happens when the plugin is registered and attached to existing workers, or when a worker is created after the plugin has been registered."
        self.logger = _setup_viper_worker_logger(self.log_to_term,self.log_to_file,self.log_file,self.level)
"""


def _setup_worker_logger(log_to_term, log_to_file, log_file, log_level):
    parallel_logger_name = _get_worker_logger_name()

    logger = logging.getLogger(parallel_logger_name)
    logger.setLevel(logging.getLevelName(log_level))

    if log_to_term:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_to_file:
        log_file = (
            log_file
            + "_"
            + str(get_worker().id)
            + "_"
            + datetime.today().strftime("%Y%m%d_%H%M%S")
            + ".log"
        )
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
