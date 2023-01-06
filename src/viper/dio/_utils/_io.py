import xarray as xr
import zarr
import copy
import os

DIMENSION_KEY = "_ARRAY_DIMENSIONS"  # Used by xarray to store array labeling info in zarr meta data.
from viper._utils._viper_logger import _get_viper_logger


def _get_attrs(zarr_obj):
    """
    get attributes of zarr obj (groups or arrays)
    """
    return {k: v for k, v in zarr_obj.attrs.asdict().items() if not k.startswith("_NC")}
    
    
    
    
def _load_chunk(zarr_name, slice_dict={},viper_local_dir=None,chunk_id=None,date_time=None):
    logger = _get_viper_logger()
    if viper_local_dir:
        viper_local_xds = os.path.join(viper_local_dir,*os.path.split(zarr_name)[-2:]) + '_' + str(chunk_id) + '_' + date_time

        #Check if already chached:
        try:
            logger.debug(zarr_name + ' chunk ' + str(slice_dict) + ' was found in viper cache: ' + viper_local_xds)
            return _load_no_dask_zarr(zarr_name=viper_local_xds, slice_dict={})
        except:
            logger.debug(zarr_name + ' chunk ' + str(slice_dict) + ' was not found in cache or failed to load. Retrieving chunk from ' + zarr_name + ' .')
            xds = _load_no_dask_zarr(zarr_name=zarr_name, slice_dict=slice_dict)
            xr.Dataset.to_zarr(xds,viper_local_xds,consolidated=True)
            
            return xds 
    else:
        return  _open_no_dask_zarr(zarr_name, slice_dict=slice_dict)


def _load_no_dask_zarr(zarr_name, slice_dict={}):
    """
    Alternative to xarray open_zarr where the arrays are not Dask Arrays.

    slice_dict: A dictionary of slice objects for which values to read form a dimension.
                For example silce_dict={'time':slice(0,10)} would select the first 10 elements in the time dimension.
                If a dim is not specified all values are retruned.
    return:
        xarray.Dataset()
    """
    
    logger = _get_viper_logger()

    zarr_group = zarr.open_group(store=zarr_name, mode="r")
    group_attrs = _get_attrs(zarr_group)

    slice_dict_complete = copy.deepcopy(slice_dict)
    coords = {}
    xds = xr.Dataset()
    for var_name, var in zarr_group.arrays():
        var_attrs = _get_attrs(var)

        for dim in var_attrs[DIMENSION_KEY]:
            if dim not in slice_dict_complete:
                slice_dict_complete[dim] = slice(None)  # No slicing.

        if (var_attrs[DIMENSION_KEY][0] == var_name) and (
            len(var_attrs[DIMENSION_KEY]) == 1
        ):
            coords[var_name] = var[
                slice_dict_complete[var_attrs[DIMENSION_KEY][0]]
            ]  # Dimension coordinates.
        else:
            # Construct slicing
            slicing_list = []
            for dim in var_attrs[DIMENSION_KEY]:
                slicing_list.append(slice_dict_complete[dim])
            slicing_tuple = tuple(slicing_list)
            xds[var_name] = xr.DataArray(
                var[slicing_tuple], dims=var_attrs[DIMENSION_KEY]
            )

    xds = xds.assign_coords(coords)

    xds.attrs = group_attrs
    return xds


def _load_mxds(mxds_name,sel_parms):
    mxds = {}
    for xds_id in sel_parms['xds_id']:
        mxds[xds_id] = xr.open_zarr(os.path.join(mxds_name,'xds'+str(xds_id)))
    return mxds
