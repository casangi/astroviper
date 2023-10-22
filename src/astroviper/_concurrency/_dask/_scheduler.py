'''
MIT License

Copyright (c) 2022 Jonathan Simon Kenyon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from collections import defaultdict
from distributed import SchedulerPlugin
from dask.core import reverse_dict
from dask.base import tokenize
from dask.order import graph_metrics, ndependencies
import click
from distributed.diagnostics.plugin import SchedulerPlugin
import numpy as np


def unravel_deps(hlg_deps, name, unravelled_deps=None):
    """Recursively construct a set of all dependencies for a specific task."""

    if unravelled_deps is None:
        unravelled_deps = set()

    for dep in hlg_deps[name]:
        unravelled_deps |= {dep}
        unravel_deps(hlg_deps, dep, unravelled_deps)

    return unravelled_deps


def get_node_depths(dependencies, root_nodes, metrics):
    node_depths = {}

    for k in dependencies.keys():
        # Get dependencies per node.
        deps = unravel_deps(dependencies, k)
        # Associate nodes with root nodes.
        roots = root_nodes & deps
        offset = metrics[k][-1]
        node_depths[k] = max(metrics[r][-1] - offset for r in roots) if roots else 0

    return node_depths


class schedular(SchedulerPlugin):
    def __init__(self, autorestrictor, local_cache):
        self.autorestrictor = autorestrictor
        self.local_cache = local_cache
        super().__init__()

    def add_worker(self, scheduler, worker):
        if self.local_cache:
            # Set the resource label to the ip of the node that the worker is on, so that tasks that require a specific node can be assigned to the correct worker.
            ip = worker[worker.rfind("/") + 1 : worker.rfind(":")]
            scheduler.add_resources(worker=worker, resources={ip: 1})

    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None, **kw):
        if self.autorestrictor:
            print("Using autorestrictor")
            """Processes dependencies to assign tasks to specific workers."""
            workers = list(scheduler.workers.keys())
            n_worker = len(workers)

            tasks = scheduler.tasks
            dependencies = kw["dependencies"]

            # print('In update_graph :', scheduler, ',*,', dsk, ',*,', keys , ',*,', restrictions , ',*,', kw)
            if dependencies:
                dependents = reverse_dict(dependencies)

                # print('reversed dict:', dependents)

                _, total_dependencies = ndependencies(dependencies, dependents)
                # TODO: Avoid calling graph metrics.
                metrics = graph_metrics(dependencies, dependents, total_dependencies)

                # Terminal nodes have no dependents, root nodes have no dependencies.
                # Horizontal partition nodes are initialized as the terminal nodes.
                part_nodes = {k for (k, v) in dependents.items() if not v}
                root_nodes = {k for (k, v) in dependencies.items() if not v}

                # Figure out the depth of every task. Depth is defined as maximum
                # distance from a root node. TODO: Optimize get_node_depths.

                node_depths = get_node_depths(dependencies, root_nodes, metrics)
                # try:
                max_depth = max(node_depths.values())
                # except:
                #        print('&&&&& dependencies, root_nodes, metrics',node_depths,',*,',dependencies, root_nodes, metrics)

                # If we have fewer partition nodes than workers, we cannot utilise all
                # the workers and are likely dealing with a reduction. We work our way
                # back through the graph, starting at the deepest terminal nodes, and
                # try to find a depth at which there was enough work to utilise all
                # workers.
                while (len(part_nodes) < n_worker) & (max_depth > 0):
                    _part_nodes = part_nodes.copy()
                    for pn in _part_nodes:
                        if node_depths[pn] == max_depth:
                            part_nodes ^= set((pn,))
                            part_nodes |= dependencies[pn]
                    max_depth -= 1
                    if max_depth <= 0:
                        return  # In this case, there in nothing we can do - fall back.

                part_roots = {}
                part_dependencies = {}
                part_dependents = {}

                for pn in part_nodes:
                    # Get dependencies per partition node.
                    part_dependencies[pn] = unravel_deps(dependencies, pn)
                    # Get dependents per partition node.
                    part_dependents[pn] = unravel_deps(dependents, pn)
                    # Associate partition nodes with root nodes.
                    part_roots[pn] = root_nodes & part_dependencies[pn]

                # Create a unique token for each set of partition roots. TODO: This is
                # very strict. What about nodes with very similar roots? Tokenization
                # may be overkill too.
                root_tokens = {tokenize(*sorted(v)): v for v in part_roots.values()}

                hash_map = defaultdict(set)
                group_offset = 0

                # Associate partition roots with a specific group if they are not a
                # subset of another, larger root set.
                for k, v in root_tokens.items():
                    if any(v < vv for vv in root_tokens.values()):  # Strict subset.
                        continue
                    else:
                        hash_map[k] |= set([group_offset])
                        group_offset += 1

                # If roots were a subset, they should share the group of their
                # superset/s.
                for k, v in root_tokens.items():
                    if not v:  # Special case - no dependencies. Handled below.
                        continue
                    shared_roots = {
                        kk: None for kk, vv in root_tokens.items() if v < vv
                    }
                    if shared_roots:
                        hash_map[k] = set().union(
                            *[hash_map[kk] for kk in shared_roots.keys()]
                        )

                task_groups = defaultdict(set)

                for pn in part_nodes:
                    pdp = part_dependencies[pn]
                    pdn = part_dependents[pn]

                    if pdp:
                        groups = hash_map[tokenize(*sorted(part_roots[pn]))]
                    else:  # Special case - no dependencies.
                        groups = {group_offset}
                        group_offset += 1

                    for g in groups:
                        task_groups[g] |= pdp | pdn | {pn}

                worker_loads = {wkr: 0 for wkr in workers}

                for task_group in task_groups.values():
                    assignee = min(worker_loads, key=worker_loads.get)
                    worker_loads[assignee] += len(task_group)

                    for task_name in task_group:
                        try:
                            task = tasks[task_name]
                        except KeyError:  # Keys may not have an assosciated task.
                            continue

                        # print('^^^^^^',dir(task))
                        # if task._worker_restrictions is None:
                        #    task._worker_restrictions = set()
                        # task._worker_restrictions |= {assignee}
                        # task._loose_restrictions = False

                        if task.worker_restrictions is None:
                            task.worker_restrictions = set()
                        task.worker_restrictions |= {assignee}
                        task.loose_restrictions = False


@click.command()
@click.option("--autorestrictor", default=False)
@click.option("--local_cache", default=False)
def dask_setup(scheduler, autorestrictor, local_cache):
    plugin = schedular(autorestrictor, local_cache)
    scheduler.add_plugin(plugin)
