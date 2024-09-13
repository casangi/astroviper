from xradio.vis.read_processing_set import read_processing_set
from graphviper.graph_tools.coordinate_utils import (interpolate_data_coords_onto_parallel_coords,
                                                     make_parallel_coord)
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from graphviper.graph_tools.coordinate_utils import make_time_coord
from graphviper.graph_tools.coordinate_utils import make_frequency_coord

from astroviper.calibration.fringefit import fringefit_single
import dask
import xarray as xa

ps = read_processing_set('n14c3.zarr')
ps.keys()

xds = ps['n14c3_000']

# 
meas = make_time_coord(time_start='2014-10-22 13:18:00', time_delta=120, n_samples=2)
parallel_coords = {}
parallel_coords['baseline_id'] = make_parallel_coord(
    coord=xds.baseline_id, n_chunks=1)
parallel_coords['time'] = make_parallel_coord(meas, n_chunks=1)
node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords,
                                                                      ps, ps_partition=['spectral_window_name'])
subsel = {'polarization': 'LL'} 
res = fringefit_single(ps, node_task_data_mapping, subsel)

# print(res)

