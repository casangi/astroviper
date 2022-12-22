import click

class viper_worker():
    def __init__(self,local_cache):
        print('init local cache')
        self.local_cache = True

    def setup(self, worker):
        """
        Run when the plugin is attached to a worker. This happens when the plugin is registered
        and attached to existing workers, or when a worker is created after the plugin has been
        registered.
        """
        #Documentation https://distributed.dask.org/en/stable/worker.html#distributed.worker.Worker
        self.worker = worker
        
        if self.local_cache:
            ip = worker.address[worker.address.rfind('/')+1:worker.address.rfind(':')]
            worker.state.available_resources = {**worker.state.available_resources, **{ip:1}}
            #print(worker.state.available_resources)
            


# https://github.com/dask/distributed/issues/4169
@click.command()
@click.option("--local_cache", default=False)
async def dask_setup(worker,local_cache):
    plugin = viper_worker(local_cache)
    await worker.client.register_worker_plugin(plugin)
