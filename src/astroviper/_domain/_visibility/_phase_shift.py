import numpy as np
import scipy
from scipy import constants
from numba import jit
import numba
import xarray as xr
import copy

# silence NumbaPerformanceWarning
# import warnings
# from numba.errors import NumbaPerformanceWarning
# warnings.filterwarnings("ignore", category=NumbaPerformanceWarning) #Suppress  NumbaPerformanceWarning: '@' is faster on contiguous arrays warning. This happens for phasor_loop and apply_rotation_matrix functions.

# Based on CASACORE measures/Measures/UVWMachine.cc and CASA code/synthesis/TransformMachines2/FTMachine.cc::girarUVW


def _phase_shift_ms_xds(ms_xds, shift_params, sel_params):
    """
    Rotate uvw coordinates and phase rotate visibilities. For a joint mosaics rotation_params['common_tangent_reprojection'] must be true.
    The specified phasedirection and field phase directions are assumed to be in the same frame.
    East-west arrays, emphemeris objects or objects within the nearfield are not supported.

    Parameters
    ----------
    ms_xds : xarray.core.dataset.Dataset
        Input vis.zarr multi dataset.
    shift_params : dictionary
    shift_params['new_phase_direction'] : list of number, length = 2, units = radians
       The phase direction to rotate to (right ascension and declination).
    shift_params['common_tangent_reprojection']  : bool, default = True
       If true common tangent reprojection is used (should be true if a joint mosaic image is being created).
    shift_params['single_precision'] : bool, default = True
       If shift_params['single_precision'] is true then the output visibilities are cast from 128 bit complex to 64 bit complex. Mathematical operations are always done in double precision.
    sel_params : dict
    sel_params['data_group_in'] : dict, default = vis_dataset.data_group[0][0]
        Only the id has to be specified
        The names of the input data and uvw data variables.
        For example sel_params['data_group_in'] = {'id':'1', 'data':'DATA','uvw':'UVW'}.
    sel_params['data_group_out'] : dict, default = {**_vis_dataset.data_group[0],**{'id':str(new_data_group_id),'uvw':'UVW_ROT','data':'DATA_ROT','field_id':shift_params['new_phase_direction']}}
        The names of the new data and uvw data variables that have been direction rotated.
        For example sel_params['data_group_out'] = {'id':2,'data':'DATA_ROT','uvw':'UVW_ROT'}.
        The new_data_group_id is the next available id.
    sel_params['overwrite'] : bool, default = False
    Returns
    -------
    psf_dataset : xarray.core.dataset.Dataset
    """

    # from graphviper.parameter_checking.check_params import check_sel_params
    from astroviper.utils.check_params import check_params, check_sel_params

    _sel_params = copy.deepcopy(sel_params)
    _shift_params = copy.deepcopy(shift_params)

    assert _check_shift_params(
        _shift_params
    ), "######### ERROR: shift_params checking failed"

    data_group_in_name, data_group_in, data_group_out_name, data_group_out = check_sel_params(
        ms_xds,
        _sel_params,
        default_data_group_out={
            "phase_shift": {"correlated_data": "VISIBILITY_SHIFT", "uvw": "UVW_SHIFT"}
        },
    )

    #print(data_group_in, data_group_out)

    if isinstance(_shift_params["new_phase_direction"], dict):
        _shift_params["new_phase_direction"] = xr.DataArray.from_dict(
            _shift_params["new_phase_direction"]
        )

    field_phase_xda = ms_xds[data_group_in["correlated_data"]].attrs[
        "field_and_source_xds"
    ]["FIELD_PHASE_CENTER"]

    # Add check to make sure frames and units are the same.
    # Add check for correctly formated measures.

    unique_values, unique_indices = np.unique(
        field_phase_xda.field_name, return_index=True
    )

    uvw_rotmat_xda, phase_rotation_xda = calc_rotation_matrices(field_phase_xda, _shift_params) # (field_name, 3, 3), (field_name, 3)
    uvw_rotmat_xda = uvw_rotmat_xda.sel(field_name=ms_xds.field_name) # (time, 3, 3)
    phase_rotation_xda = phase_rotation_xda.sel(field_name=ms_xds.field_name) # (time, 3)
    
    if _shift_params["mosaic_facetting"]:
        end_slice = 2
    else:
        end_slice = 3
        
        
    import time
    start = time.time()
    # ms_xds[data_group_out["uvw"]] = xr.DataArray(
    #      np.einsum("ijk, ilk -> ilj ", uvw_rotmat_xda.values, ms_xds[data_group_in["uvw"]].values),
    #     dims=ms_xds[data_group_in["uvw"]].dims,
    # ) # (time, 3 , 3), (time, baseline, uvw:3)  -> (time, baseline, 3)
    #print('Time taken for einsum',time.time()-start)
    
    start = time.time()
    ms_xds[data_group_out["uvw"]] = xr.DataArray(
        (uvw_rotmat_xda.values[:, np.newaxis, ...] @ ms_xds[data_group_in["uvw"]].values[..., np.newaxis])[...,0],
        dims=ms_xds[data_group_in["uvw"]].dims,
    ) # (time, [1], 3 , 3), (time, baseline, uvw:3, [1])  -> (time, baseline, 3, [1]) -> (time, baseline, 3)
    #print('Time taken for @',time.time()-start)
    
    print("ms_xds[data_group_out[uvw]] ", ms_xds[data_group_out["uvw"]] )

    phase_direction = np.einsum(
        " ijk, ik -> ij",
        ms_xds[data_group_out["uvw"]][..., 0:end_slice].values, phase_rotation_xda[..., 0:end_slice].values
    )# (time, baseline, 3), (time, 3) -> (time, baseline)
    print('end_slice',end_slice)    
    print('phase_direction',phase_direction)
    
    
    # print('phase_rotation_xda',phase_rotation_xda)
    # print('phase_direction',phase_direction)

    phasor = np.exp(
        2.0
        * 1j
        * np.pi
        * phase_direction[:, :, None, None]
        * ms_xds.frequency.values[None, None, :, None]
        / constants.c
    )  # phasor_ngcasa = - phasor_casa. Sign flip is due to CASA

    if _shift_params["single_precision"]:
        # print("single precision")
        ms_xds[data_group_out["correlated_data"]] = (
            (phasor * ms_xds[data_group_in["correlated_data"]]).astype(np.complex64)
        ).astype(np.complex64)
    else:
        ms_xds[data_group_out["correlated_data"]] = (
            phasor * ms_xds[data_group_in["correlated_data"]]
        )

    from xradio.measurement_set import MeasurementSetXds
    new_field_and_source_xds = xr.Dataset()
    new_field_and_source_xds["FIELD_PHASE_CENTER"] = _shift_params["new_phase_direction"]
    ms_xds[data_group_out["correlated_data"]].attrs["field_and_source_xds"] = new_field_and_source_xds
    
    ms_xds.attrs['active_data_group'] = data_group_out_name

    return ms_xds



def calc_rotation_matrices(field_phase_xda, shift_params):
    from scipy.spatial.transform import Rotation as R
    
    new_phase_direction_xda = shift_params["new_phase_direction"]
    
    # Original code
    # R_current_to_XYZ = R.from_euler('XZ',  np.array([
    #         np.pi/2 - field_phase_xda.sel(sky_dir_label="dec"),
    #         - field_phase_xda.sel(sky_dir_label="ra"),
    #     ]).T).as_matrix()
    
    # R_XYZ_to_new = R.from_euler('ZX', [
    #         new_phase_direction_xda.sel(sky_dir_label="ra"),
    #         - np.pi/2 + new_phase_direction_xda.sel(sky_dir_label="dec"),
    #     ]).as_matrix()
    
    
    #Works
    # R_current_to_XYZ = R.from_euler('XZ',  np.array([
    #         - np.pi/2 + field_phase_xda.sel(sky_dir_label="dec"),
    #         - field_phase_xda.sel(sky_dir_label="ra"),
    #     ]).T).as_matrix()
    
    # R_XYZ_to_new = R.from_euler('ZX', [
    #         new_phase_direction_xda.sel(sky_dir_label="ra"),
    #         np.pi/2 - new_phase_direction_xda.sel(sky_dir_label="dec"),
    #     ]).as_matrix()

    # uvw_rotmat = np.matmul(R_XYZ_to_new.T,R_current_to_XYZ.transpose([0,2,1])) 
    
    
    ########### 
    #Right handed coordinate system XYZ.
    #Rotate from current phase direction to XYZ system.
    R_current_to_XYZ = R.from_euler('ZX',  np.array([
            field_phase_xda.sel(sky_dir_label="ra"), # -(-ra): ra is defined in the opposite direction to the right handed system consequently the first negative. The second negative is to rotate to the XYZ system.
            np.pi/2 - field_phase_xda.sel(sky_dir_label="dec"), 
        ]).T).as_matrix() # (field_name, ra_dec:2) -> (field_name, 3, 3)
    
    R_XYZ_to_new = R.from_euler('XZ', [
            - np.pi/2 + new_phase_direction_xda.sel(sky_dir_label="dec"),
            - new_phase_direction_xda.sel(sky_dir_label="ra"),
        ]).as_matrix() # (ra_dec:2) -> (3, 3)
    
    p = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    print('R_XYZ_to_new',R_XYZ_to_new@p)
    

    
    uvw_rotmat = R_XYZ_to_new @ R_current_to_XYZ #(3,3) @ (field_name, 3, 3) -> (field_name x 3 x 3)

    
    # uvw_rotmat_truth = np.array([[ 1.00000e+00, -1.00712e-05, -2.94701e-05],
    #                             [ 1.00624e-05,  1.00000e+00, -2.99609e-04],
    #                             [ 2.94732e-05,  2.99609e-04,  1.00000e+00]])
    
    # print(np.sum(np.abs(uvw_rotmat[0,:,:] - uvw_rotmat_truth)))
    # print('**********')
    # print(uvw_rotmat[0,:,:]-uvw_rotmat_truth)
    # print('**********')
    # print(uvw_rotmat[0,:,:])
    
    new_phase_direction_cosine_xda = _directional_cosine_xda(
        new_phase_direction_xda
    )  # (radec:2) -> (lmn:3)
    
    field_phase_direction_cosine_xda = _directional_cosine_xda(
        field_phase_xda
    )  # field_name x lmn[3]

    # phase_rotation = np.einsum(
    #     "ij,kj->ki",
    #     R_XYZ_to_new,
    #     (new_phase_direction_cosine_xda.data - field_phase_direction_cosine_xda.data),
    # )  # (3,3),(field_name, 3) -> (field_name, 3)

    #Faster than using einsum.
    temp = R.from_euler(
        "XZ", [[np.pi / 2 - new_phase_direction_xda.sel(sky_dir_label="dec"), -new_phase_direction_xda.sel(sky_dir_label="ra") + np.pi / 2]]
    ).as_matrix()[0]
    
    phase_rotation = (temp @ ((new_phase_direction_cosine_xda.data - field_phase_direction_cosine_xda.data)[:,:,np.newaxis]))[...,0] # (3,3),(field_name,3,[1]) -> (field_name, 3, 1) -> (field_name, 3)
    
    # print("new_phase_direction_cosine_xda.data ",new_phase_direction_cosine_xda.data )
    # print("field_phase_direction_cosine_xda.data ",field_phase_direction_cosine_xda.data )
    # print("R_XYZ_to_new ",R_XYZ_to_new )
    # print("(new_phase_direction_cosine_xda.data - field_phase_direction_cosine_xda.data)",(new_phase_direction_cosine_xda.data - field_phase_direction_cosine_xda.data))
    # print("phase_rotation ",phase_rotation )
    print('temp',temp)
    print(phase_rotation)
    print(phase_rotation.shape)
    print(temp.shape,((new_phase_direction_cosine_xda.data - field_phase_direction_cosine_xda.data)[:,:,np.newaxis]).shape)
    ta = (new_phase_direction_cosine_xda.data - field_phase_direction_cosine_xda.data)
    print(ta.shape)
    print(temp@ta[0,:])
    
    print('new',new_phase_direction_xda.data)
    print('field',field_phase_xda.data)

    #Create xarray DataArray and expand to 
    uvw_rotmat_xda = xr.DataArray(
        uvw_rotmat,
        dims=("field_name", "uvw_row_label", "uvw_col_label"),
        coords={
            "uvw_row_label": ["u", "v", "w"],
            "uvw_col_label": ["u", "v", "w"],
            "field_name": field_phase_xda.field_name,
        },
        attrs={"type": "rotation_matrix"},
    )

    phase_rotation_xda = xr.DataArray(
        phase_rotation,
        dims=("field_name", "uvw_label"),
        coords={
            "uvw_label": ["u", "v", "w"],
            "field_name": field_phase_xda.field_name,
        },
    )
    
    
    #Rotate with facetting style rephasing..for multifield mosaic
    if shift_params["mosaic_facetting"]:
        uvw_rotmat_xda[:,2,0:2]= 0.0
    
    return uvw_rotmat_xda, phase_rotation_xda



# def calc_rotation_matrices(field_phase_xda, shift_params):
#     from scipy.spatial.transform import Rotation as R
    
#     new_phase_direction_xda = shift_params["new_phase_direction"]
#     rotmat_new_phase_direction = R.from_euler(
#         "XZ",
#         [
#             [
#                 np.pi / 2 - new_phase_direction_xda.sel(sky_dir_label="dec"),
#                 -new_phase_direction_xda.sel(sky_dir_label="ra") + np.pi / 2,
#             ]
#         ],
#     ).as_matrix()[
#         0
#     ]  #  (1, radec:2) -> (3, 3)

#     new_phase_direction_cosine_xda = _directional_cosine_xda(
#         new_phase_direction_xda
#     )  # (radec:2) -> (lmn:3)
    
#     rotmat_field_phase_direction = R.from_euler(
#         "ZX",
#         (field_phase_xda + [-np.pi / 2, -np.pi / 2]),
#     ).as_matrix()  # (field_name, radec) -> (field_name, 3, 3)

#     uvw_rotmat = np.matmul(rotmat_new_phase_direction, rotmat_field_phase_direction)#.transpose([0,2,1]) # (3,3) @ (field_name,3) -> [field_name x 3 x 3]
#     # uvw_rotmat = np.einsum('ij,tjk->tik', rotmat_new_phase_direction, rotmat_field_phase_direction).transpose([0,2,1]) # field_name x 3 x 3
#     # uvw_rotmat = np.einsum(
#     #     "ij,tjk->tki", rotmat_new_phase_direction, rotmat_field_phase_direction
#     # )  # (3,3),(field_name, 3, 3) -> (field_name, 3, 3), matrix multiplication and transpose per field_name step.

#     field_phase_direction_cosine_xda = _directional_cosine_xda(
#         field_phase_xda
#     )  # field_name x lmn[3]

#     phase_rotation = np.einsum(
#         "ij,kj->ki",
#         rotmat_new_phase_direction,
#         (new_phase_direction_cosine_xda.data - field_phase_direction_cosine_xda.data),
#     )  # (3,3),(field_name, 3) -> (field_name, 3)
    
#     #Create xarray DataArray and expand to 
#     uvw_rotmat_xda = xr.DataArray(
#         uvw_rotmat,
#         dims=("field_name", "uvw_row_label", "uvw_col_label"),
#         coords={
#             "uvw_row_label": ["u", "v", "w"],
#             "uvw_col_label": ["u", "v", "w"],
#             "field_name": field_phase_xda.field_name,
#         },
#         attrs={"type": "rotation_matrix"},
#     )

#     phase_rotation_xda = xr.DataArray(
#         phase_rotation,
#         dims=("field_name", "uvw_label"),
#         coords={
#             "uvw_label": ["u", "v", "w"],
#             "field_name": field_phase_xda.field_name,
#         },
#     )
    
#     #print(uvw_rotmat_xda)
    
#     if shift_params["common_tangent_reprojection"]:
#         uvw_rotmat_xda[:,2,0:2]= 0.0
    
#     return uvw_rotmat_xda, phase_rotation_xda


# @jit(nopython=True, cache=True, nogil=True)
# def _directional_cosine(phase_direction_in_radians):
#     """
#     # In https://arxiv.org/pdf/astro-ph/0207413.pdf see equation 160
#     phase_direction_in_radians (RA,DEC)
#     """

#     phase_direction_cosine = np.zeros((3,), dtype=numba.f8)
#     # phase_direction_cosine = np.zeros((3,))
#     phase_direction_cosine[0] = np.cos(phase_direction_in_radians[0]) * np.cos(
#         phase_direction_in_radians[1]
#     )
#     phase_direction_cosine[1] = np.sin(phase_direction_in_radians[0]) * np.cos(
#         phase_direction_in_radians[1]
#     )
#     phase_direction_cosine[2] = np.sin(phase_direction_in_radians[1])
#     return phase_direction_cosine


def _directional_cosine_xda(phase_direction_xda):
    """
    Converts a data variable Sky Coord with units in radians from ra and dec to directional cosines (l, m, n).

    Parameters
    ----------
    phase_direction_xda : xarray.DataArray
        The input DataArray in xda format containing ra and dec sky direction labels.

    Returns
    -------
    xarray.DataArray
        A new DataArray containing the directional cosines (l, m, n) along the lmn_label dimension.

    See Also
    --------
    https://arxiv.org/pdf/astro-ph/0207413.pdf (Equation 160)
    """

    ra = phase_direction_xda.sel(sky_dir_label="ra")
    dec = phase_direction_xda.sel(sky_dir_label="dec")

    # Calculate directional cosines
    l = np.cos(ra) * np.cos(dec)
    m = np.sin(ra) * np.cos(dec)
    n = np.sin(dec)
    
    # Create a new DataArray with the lmn_label dimension
    phase_direction_cosine_xda = xr.concat([l, m, n], dim="lmn_label").assign_coords(
        lmn_label=["l", "m", "n"]
    )

    # Ensure that dimension order is maintained.
    reordered_dims = list(phase_direction_xda.dims)
    sky_dir_label_index = np.where(np.array(reordered_dims) == "sky_dir_label")[0][0]
    reordered_dims[sky_dir_label_index] = "lmn_label"
    phase_direction_cosine_xda = phase_direction_cosine_xda.transpose(*reordered_dims)

    return phase_direction_cosine_xda


def _check_shift_params(shift_params):
    # from graphviper.parameter_checking.check_params import check_params
    from astroviper.utils.check_params import check_params, check_sel_params
    import numbers

    params_passed = True

    # if not(_check_params(shift_params, 'new_phase_direction', [dict])): params_passed = False

    if not (
        check_params(shift_params, "common_tangent_reprojection", [bool], default=True)
    ):
        params_passed = False

    if not (check_params(shift_params, "single_precision", [bool], default=True)):
        params_passed = False

    return params_passed
