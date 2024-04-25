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


def _phase_shift_vis_ds(ms_xds, shift_parms, sel_parms):
    """
    Rotate uvw coordinates and phase rotate visibilities. For a joint mosaics rotation_parms['common_tangent_reprojection'] must be true.
    The specified phasedirection and field phase directions are assumed to be in the same frame.
    East-west arrays, emphemeris objects or objects within the nearfield are not supported.

    Parameters
    ----------
    mxds : xarray.core.dataset.Dataset
        Input vis.zarr multi dataset.
    shift_parms : dictionary
    shift_parms['new_phase_direction'] : list of number, length = 2, units = radians
       The phase direction to rotate to (right ascension and declination).
    shift_parms['common_tangent_reprojection']  : bool, default = True
       If true common tangent reprojection is used (should be true if a joint mosaic image is being created).
    shift_parms['single_precision'] : bool, default = True
       If shift_parms['single_precision'] is true then the output visibilities are cast from 128 bit complex to 64 bit complex. Mathematical operations are always done in double precision.
    sel_parms : dict
    sel_parms['data_group_in'] : dict, default = vis_dataset.data_group[0][0]
        Only the id has to be specified
        The names of the input data and uvw data variables.
        For example sel_parms['data_group_in'] = {'id':'1', 'data':'DATA','uvw':'UVW'}.
    sel_parms['data_group_out'] : dict, default = {**_vis_dataset.data_group[0],**{'id':str(new_data_group_id),'uvw':'UVW_ROT','data':'DATA_ROT','field_id':shift_parms['new_phase_direction']}}
        The names of the new data and uvw data variables that have been direction rotated.
        For example sel_parms['data_group_out'] = {'id':2,'data':'DATA_ROT','uvw':'UVW_ROT'}.
        The new_data_group_id is the next available id.
    sel_parms['overwrite'] : bool, default = False
    Returns
    -------
    psf_dataset : xarray.core.dataset.Dataset
    """

    #from graphviper.parameter_checking.check_parms import check_sel_parms
    from astroviper.utils.check_parms import check_parms, check_sel_parms

    _sel_parms = copy.deepcopy(sel_parms)
    _shift_parms = copy.deepcopy(shift_parms)

    assert _check_shift_parms(
        _shift_parms
    ), "######### ERROR: shift_parms checking failed"

    data_group_in, data_group_out = check_sel_parms(
        ms_xds,
        _sel_parms,
        default_data_group_out={
            "phase_shift": {"visibility": "VISIBILITY_SHIFT", "uvw": "UVW_SHIFT"}
        },
    )

    field_phase_direction = ms_xds[data_group_in['visibility']].attrs["field_info"][
            "phase_direction"
        ]
    uvw_rotmat, phase_rotation = calc_rotation_matrices(field_phase_direction, _shift_parms)

    # print("uvw_rotmat, phase_rotation",uvw_rotmat, phase_rotation)

    ms_xds[data_group_out["uvw"]] = xr.DataArray(
        ms_xds[data_group_in["uvw"]].values @ uvw_rotmat,
        dims=ms_xds[data_group_in["uvw"]].dims,
    )

    if _shift_parms["common_tangent_reprojection"]:
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

    if _shift_parms["single_precision"]:
        # print("single precision")
        ms_xds[data_group_out["visibility"]] = (
            (phasor * ms_xds[data_group_in["visibility"]]).astype(np.complex64)
        ).astype(np.complex128)
    else:
        ms_xds[data_group_out["visibility"]] = (
            phasor * ms_xds[data_group_in["visibility"]]
        )

    ms_xds[data_group_out["visibility"]].attrs["phase_direction"] = shift_parms[
        "new_phase_direction"
    ]

    return _sel_parms["data_group_out"]


def calc_rotation_matrices(field_phase_direction, shift_parms):
    from scipy.spatial.transform import Rotation as R

    ra_image = shift_parms["new_phase_direction"]["data"][0]
    dec_image = shift_parms["new_phase_direction"]["data"][1]

    rotmat_new_phase_direction = R.from_euler(
        "XZ", [[np.pi / 2 - dec_image, -ra_image + np.pi / 2]]
    ).as_matrix()[0]
    new_phase_direction_cosine = _directional_cosine(np.array([ra_image, dec_image]))

    uvw_rotmat = np.zeros((3, 3), np.double)
    phase_rotation = np.zeros((3,), np.double)

    rotmat_field_phase_direction = R.from_euler(
        "ZX",
        [[-np.pi / 2 + field_phase_direction['data'][0], field_phase_direction['data'][1] - np.pi / 2]],
    ).as_matrix()[0]
    uvw_rotmat = np.matmul(rotmat_new_phase_direction, rotmat_field_phase_direction).T

    field_phase_direction_cosine = _directional_cosine(np.array(field_phase_direction['data']))
    phase_rotation = np.matmul(
        rotmat_new_phase_direction,
        (new_phase_direction_cosine - field_phase_direction_cosine),
    )

    return uvw_rotmat, phase_rotation


@jit(nopython=True, cache=True, nogil=True)
def _directional_cosine(phase_direction_in_radians):
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


def _check_shift_parms(shift_parms):
    #from graphviper.parameter_checking.check_parms import check_parms
    from astroviper.utils.check_parms import check_parms, check_sel_parms
    import numbers

    parms_passed = True

    # if not(_check_parms(shift_parms, 'new_phase_direction', [dict])): parms_passed = False

    if not (
        check_parms(shift_parms, "common_tangent_reprojection", [bool], default=True)
    ):
        parms_passed = False

    if not (check_parms(shift_parms, "single_precision", [bool], default=True)):
        parms_passed = False

    return parms_passed
