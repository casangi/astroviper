import click

from astroviper._utils._logger import _setup_worker_logger


class _worker:
    def __init__(self, local_cache, log_parms):
        print("init local cache")
        self.local_cache = local_cache

        print("log_parms", log_parms)
        # /.lustre/aoc/projects/ngvla/viper/ngvla_sim/viper_
        self.log_to_term = log_parms["log_to_term"]
        self.log_to_file = log_parms["log_to_file"]
        self.log_file = log_parms["log_file"]
        self.log_level = log_parms["log_level"]

    def get_logger(self):
        return self.logger

    def setup(self, worker):
        """
        Run when the plugin is attached to a worker. This happens when the plugin is registered
        and attached to existing workers, or when a worker is created after the plugin has been
        registered.
        """

        self.logger = _setup_worker_logger(
            self.log_to_term, self.log_to_file, self.log_file, self.log_level
        )
        self.logger.debug(
            "Logger created on worker " + str(worker.id) + ",*," + str(worker.address)
        )

        # Documentation https://distributed.dask.org/en/stable/worker.html#distributed.worker.Worker
        self.worker = worker

        if self.local_cache:
            ip = worker.address[
                worker.address.rfind("/") + 1 : worker.address.rfind(":")
            ]
            self.logger.debug(str(worker.id) + ",*," + ip)
            worker.state.available_resources = {
                **worker.state.available_resources,
                **{ip: 1},
            }
            # print(worker.state.available_resources)


# https://github.com/dask/distributed/issues/4169
@click.command()
@click.option("--local_cache", default=False)
# @click.option("--log_parms", default={'log_to_term':True,'log_to_file':False,'log_file':'viper_', 'log_level':'DEBUG'})
@click.option("--log_to_term", default=True)
@click.option("--log_to_file", default=False)
@click.option("--log_file", default="viper_")
@click.option("--log_level", default="INFO")
async def dask_setup(
    worker, local_cache, log_to_term, log_to_file, log_file, log_level
):
    log_parms = {
        "log_to_term": log_to_term,
        "log_to_file": log_to_file,
        "log_file": log_file,
        "log_level": log_level,
    }
    plugin = _worker(local_cache, log_parms)
    await worker.client.register_worker_plugin(plugin, name="viper_worker")
