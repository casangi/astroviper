import numpy as np
import xarray as xr
import dask
from scipy.optimize import minimize

# Types
from collections.abc import Mapping, Iterable
from typing import Any, Union
from xarray.core.datatree import DataTree # As type

# xradio, graphviper and astroviper imports
from xradio.measurement_set.open_processing_set import (
    open_processing_set,
)
from xradio.measurement_set import ProcessingSetXdt 

from graphviper.graph_tools.map import map
from graphviper.graph_tools.reduce import reduce
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
)

from astroviper.calibration.fringe_cal_xds import make_empty_cal_xds

from .utils import (unique,
                    get_ps_polarization_coord,
                    make_cal_map
)

