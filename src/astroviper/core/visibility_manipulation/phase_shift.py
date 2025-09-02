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


def phase_shift_vis_ds(ms_xds, shift_params, sel_params):
    """
    Rotate uvw coordinates and phase rotate visibilities. For a joint mosaics rotation_params['common_tangent_reprojection'] must be true.
    The specified phasedirection and field phase directions are assumed to be in the same frame.
    East-west arrays, emphemeris objects or objects within the nearfield are not supported.

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
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

    assert check_shift_params(
        _shift_params
    ), "######### ERROR: shift_params checking failed"

    data_group_in, data_group_out = check_sel_params(
        ms_xds,
        _sel_params,
        default_data_group_in_name="base",
        default_data_group_out_name="phase_shift",
        default_data_group_out_modified={
            "correlated_data": "VISIBILITY_SHIFT",
            "uvw": "UVW_SHIFT",
        },
    )

    field_phase_direction = ms_xds.xr_ms.get_field_and_source_xds(
        data_group_in["data_group_in_name"]
    ).FIELD_PHASE_CENTER_DIRECTION.isel(field_name=0)
    uvw_rotmat, phase_rotation = calc_rotation_matrices(
        field_phase_direction, _shift_params
    )

    # print("uvw_rotmat, phase_rotation",uvw_rotmat, phase_rotation)

    ms_xds[data_group_out["uvw"]] = xr.DataArray(
        ms_xds[data_group_in["uvw"]].values @ uvw_rotmat,
        dims=ms_xds[data_group_in["uvw"]].dims,
    )

    if _shift_params["common_tangent_reprojection"]:
        end_slice = 2
    else:
        end_slice = 3

    phase_direction = (
        ms_xds[data_group_out["uvw"]][:, :, 0:end_slice].values
        @ phase_rotation[0:end_slice]
    )

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
        ).astype(np.complex128)
    else:
        ms_xds[data_group_out["correlated_data"]] = (
            phasor * ms_xds[data_group_in["correlated_data"]]
        )

    ms_xds[data_group_out["correlated_data"]].attrs["phase_direction"] = shift_params[
        "new_phase_direction"
    ]

    ms_xds.attrs["data_groups"][data_group_out["data_group_out_name"]] = data_group_out

    return data_group_out


def calc_rotation_matrices(field_phase_direction, shift_params):
    from scipy.spatial.transform import Rotation as R

    ra_image = shift_params["new_phase_direction"].values[0]
    dec_image = shift_params["new_phase_direction"].values[1]

    rotmat_new_phase_direction = R.from_euler(
        "XZ", [[np.pi / 2 - dec_image, -ra_image + np.pi / 2]]
    ).as_matrix()[0]
    new_phase_direction_cosine = directional_cosine(np.array([ra_image, dec_image]))

    uvw_rotmat = np.zeros((3, 3), np.double)
    phase_rotation = np.zeros((3,), np.double)

    rotmat_field_phase_direction = R.from_euler(
        "ZX",
        [
            [
                -np.pi / 2 + field_phase_direction.values[0],
                field_phase_direction.values[1] - np.pi / 2,
            ]
        ],
    ).as_matrix()[0]
    uvw_rotmat = np.matmul(rotmat_new_phase_direction, rotmat_field_phase_direction).T

    field_phase_direction_cosine = directional_cosine(
        np.array(field_phase_direction.values)
    )
    phase_rotation = np.matmul(
        rotmat_new_phase_direction,
        (new_phase_direction_cosine - field_phase_direction_cosine),
    )

    return uvw_rotmat, phase_rotation


@jit(nopython=True, cache=True, nogil=True)
def directional_cosine(phase_direction_in_radians):
    """
    # In https://arxiv.org/pdf/astro-ph/0207413.pdf see equation 160
    phase_direction_in_radians (RA,DEC)
    """

    phase_direction_cosine = np.zeros((3,), dtype=numba.f8)
    # phase_direction_cosine = np.zeros((3,))
    phase_direction_cosine[0] = np.cos(phase_direction_in_radians[0]) * np.cos(
        phase_direction_in_radians[1]
    )
    phase_direction_cosine[1] = np.sin(phase_direction_in_radians[0]) * np.cos(
        phase_direction_in_radians[1]
    )
    phase_direction_cosine[2] = np.sin(phase_direction_in_radians[1])
    return phase_direction_cosine


def check_shift_params(shift_params):
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
