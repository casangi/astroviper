# -*- coding: utf-8 -*-
import xradio
import xarray
import copy
from typing import List
from astropy.coordinates import SpectralCoord, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="astropy.coordinates.spectral_coordinate",
)
np.set_printoptions(precision=12)


def _ms_spectral_frame_conversion(
    ms: xarray.core.datatree.DataTree,
    freqrange: List[float] = [],
    outframe: str = "LSRK",
) -> xarray.core.datatree.DataTree:
    """


    Parameters
    ----------
    ms_xds : xarray.core.datatree.DataTree
        DESCRIPTION. This is the input ms v4 xds that is to be transformed to
        a new spectral frame
    freqrange : List(float, float)
        DESCRIPTION. Selection of subset of channels falling in the
        freqrange to transform

    Returns
    -------
    ms v4 xds

    """

    # outms = ms.copy(inherit=False, deep=True)
    outms = copy.deepcopy(ms)
    # need to do selection over frequency here
    pc = ms.ms.get_field_and_source_xds().FIELD_PHASE_CENTER
    pcdir = [d * u.Unit(e) for d, e in zip(pc.data[0], pc.units)]
    phcen = SkyCoord(pcdir[0], pcdir[1], frame=pc.frame)
    locATt = _get_all_itrs_loc(ms)
    obsfreq = ms.frequency.data * u.Unit(ms.frequency.attrs["units"][0])
    newFreqInFrame = _outframe_freq(outms, outframe)
    ##outMSFreq = outms.frequency.assign_coords(frequency=newFreqInFrame) buggy
    outMSFreq = xarray.DataArray(
        data=newFreqInFrame,
        dims=outms.frequency.dims,
        coords=outms.frequency.coords,
        attrs=outms.frequency.attrs,
    )
    outMSFreq.attrs["observer"] = outframe
    # cannot assign the values or array directly
    # for k in range(len(newFreqInFrame)):
    #    outMSFreq.values[k] = newFreqInFrame[k]  # wopuld rather assign by coordinate value
    outms.frequency = outMSFreq
    print(obsfreq.value - outms.frequency.data, newFreqInFrame, obsfreq.value)
    # for aTLoc in locATt:
    _interpolate_data_weight_from_TOPO(outms, obsfreq)

    return outms


def _outframe_freq(ms: xarray.core.datatree.DataTree, outframe: str = "lsrk"):
    """
    Function to get a set of uniformly spaced frequencies that covers the range of frequencies
    in the ms over time in the outframe requested
    Parameters
    ----------
    ms : xarray.core.datatree.DataTree
        DESCRIPTION.
    outframe : str, optional
        For now accepts lsrk, lsrd, bary, galacto(centric).
        The default is "lsrk".
    Returns
    -------
    outfreqs : numpy array
        The nchannels in the ms frequencies in the outframe requested

    """
    # loc_itrs = obs.get_itrs(obstime=t)
    obsframe = ms["frequency"].attrs["observer"]
    if "TOPO" not in obsframe:
        raise Exception("This function works with TOPO only for now")
    obsfreq = ms["frequency"].data * u.Unit(ms.frequency.attrs["units"][0])
    # print("_outframe_freq originfreq", obsfreq)
    nchan = len(obsfreq)

    locATt = _get_all_itrs_loc(ms)
    # There should be only one field name in each ms
    # should we test for uniqueness ?
    fldname = ms.field_name.data[0]
    phcen = _get_phase_center(ms, fldname)
    maxFreq = -1
    minFreq = 1e12
    for a_loc in locATt:
        frameSpec = SpectralCoord(
            obsfreq, observer=a_loc, target=phcen
        ).with_observer_stationary_relative_to(_frame_from_str(outframe))
        freqATt = frameSpec.quantity.value
        minFreq = min(minFreq, np.min(freqATt))
        maxFreq = max(maxFreq, np.max(freqATt))
    outfreqs = np.zeros([nchan])

    if nchan > 1:
        width = maxFreq - minFreq
        outfreqs = np.vectorize(lambda k: k * width + minFreq)(
            np.arange(nchan)
        )
    else:
        outfreqs = np.array([minFreq])
    # print("minfreq ", minFreq, " maxfreq ", maxFreq)
    # print("difference obsfreq  ...new frame freq ", obsfreq.value - outfreqs)
    return outfreqs


def _get_phase_center(
    ms: xarray.core.datatree.DataTree, fieldname: str
) -> SkyCoord:
    pc = ms.ms.get_field_and_source_xds().FIELD_PHASE_CENTER.sel(
        field_name=fieldname
    )
    pcdir = [d * u.Unit(e) for d, e in zip(pc.data, pc.attrs["units"])]
    phcen = SkyCoord(pcdir[0], pcdir[1], frame=pc.attrs["frame"])
    return phcen


def _frame_from_str(framestr: str):
    """
    Tries to interprete string frame a la casa definition and return the
    appropriate astropy frame
    """
    fr = ""
    if framestr.lower() == "lsrk":
        from astropy.coordinates import LSRK

        fr = LSRK
    elif framestr.lower() == "lsrd":
        from astropy.coordinates import LSRD

        fr = LSRD
    elif "galacto" in framestr.lower():
        from astropy.coordinates import Galactocentric

        fr = Galactocentric
    elif "bary" in framestr.lower():
        from astropy.coordinates import BarycentricTrueEcliptic

        fr = BarycentricTrueEcliptic
    else:
        raise (f"Don't know the frame {framestr}")
    return fr


def _get_all_itrs_loc(ms: xarray.core.datatree.DataTree):
    """This returns an astropy itrs location at all unique times
    in the ms
    """

    obsstr = ms["antenna_xds"].attrs["overall_telescope_name"]
    if obsstr == "EVLA":
        obsstr = "VLA"
    elif obsstr == "ATA":
        obsstr = "ALMA"

    obs = EarthLocation.of_site(obsstr)
    obs_t = ms["time"].data * u.Unit(ms["time"].attrs["units"][0])
    t = Time(
        obs_t,
        format=ms["time"].attrs["format"],
        scale=ms["time"].attrs["scale"],
    )
    locATt = obs.get_itrs(obstime=t)
    return locATt


def _interpolate_data_weight_from_TOPO(
    ms: xarray.core.datatree.DataTree, origfreq: u.quantity.Quantity
):
    """
    interpolate the visibility (and weights)  for every time stamp in the frame of the ms
    The outframe is assumed to have already been assigned to the
    ms.frequency attributes

    """
    infreq = origfreq.to(u.Hz).value
    interpfreq = ms.frequency.data
    locATt = _get_all_itrs_loc(ms)
    fldname = ms.field_name.data[0]
    phcen = _get_phase_center(ms, fldname)
    outframe = ms.frequency.attrs["observer"]
    for a_loc in locATt:
        frameSpec = SpectralCoord(
            origfreq, observer=a_loc, target=phcen
        ).with_observer_stationary_relative_to(_frame_from_str(outframe))
        freqATt = frameSpec.quantity.to(u.Hz).value
        elvis = ms.VISIBILITY[a_loc.obstime.value]
        elwgt = ms.WEIGHT[a_loc.obstime.value]
        elflg = ms.FLAG[a_loc.obstime.value]
        elwgt = elwgt * np.logical_not(elflg)
        _interp_channels(elvis, elwgt, freqATt, interpfreq)


def _interp_channels(data, weights, datafreq, interpfreq):
    wgtdata = data.data * weights.data
    for b in range(data.shape[0]):
        for p in range(data.shape[2]):
            fintd = interp1d(
                datafreq,
                wgtdata[b, :, p],
                kind="linear",
                fill_value="extrapolate",
            )
            wgtdata[b, :, p] = fintd(interpfreq)
            fintw = interp1d(
                datafreq,
                weights[b, :, p],
                kind="linear",
                fill_value="extrapolate",
            )
            weights[b, :, p] = fintw(interpfreq)
    data[weights != 0] = wgtdata[weights != 0] / weights[weights != 0]
    data[weights == 0] = 0.0
