import dask
import time
import os
import json

import numpy as np
import xarray as xr
import dask.array as da
import zarr
import copy

import astropy
import astropy.units as u
import astropy.coordinates as coord

from numba import njit
from numba.core import types
from numba.typed import Dict

DIMENSION_KEY = "_ARRAY_DIMENSIONS"
jit_cache =  False

def _get_attrs(zarr_obj):
    '''
    get attributes of zarr obj (groups or arrays)
    '''
    return {
        k: v
        for k, v in zarr_obj.attrs.asdict().items()
        if not k.startswith("_NC")
    }

def _open_no_dask_zarr(zarr_name,slice_dict={}):
    '''
        Alternative to xarray open_zarr where the arrays are not Dask Arrays.
        
        slice_dict: A dictionary of slice objects for which values to read form a dimension.
                    For example silce_dict={'time':slice(0,10)} would select the first 10 elements in the time dimension.
                    If a dim is not specified all values are retruned.
        return:
            xarray.Dataset()
    '''
    
    zarr_group = zarr.open_group(store=zarr_name,mode='r')
    group_attrs = _get_attrs(zarr_group)
    
    slice_dict_complete = copy.deepcopy(slice_dict)
    coords = {}
    xds = xr.Dataset()
    for var_name, var in zarr_group.arrays():
        var_attrs = _get_attrs(var)
        
        for dim in var_attrs[DIMENSION_KEY]:
            if dim not in slice_dict_complete:
                slice_dict_complete[dim] = slice(None) #No slicing.
                
        if (var_attrs[DIMENSION_KEY][0] == var_name) and (len(var_attrs[DIMENSION_KEY]) == 1):
            coords[var_name] = var[slice_dict_complete[var_attrs[DIMENSION_KEY][0]]] #Dimension coordinates.
        else:
            #Construct slicing
            slicing_list = []
            for dim in var_attrs[DIMENSION_KEY]:
                slicing_list.append(slice_dict_complete[dim])
            slicing_tuple = tuple(slicing_list)
            xds[var_name] = xr.DataArray(var[slicing_tuple],dims=var_attrs[DIMENSION_KEY])
            
    xds = xds.assign_coords(coords)
    
    xds.attrs = group_attrs
    return xds
