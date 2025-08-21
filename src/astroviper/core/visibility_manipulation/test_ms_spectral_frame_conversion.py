# -*- coding: utf-8 -*-
from ms_spectral_frame_conversion import ms_spectral_frame_conversion
import xradio
from xradio.measurement_set import open_processing_set
from astropy.time import Time
import numpy as np
import xarray as xr
import matplotlib.pylab as pl


def test_ms_spectral_frame_conversion():
    """
    unit tests data sets are
    casa based cvel data for comparison Antennae_fld1_casa_lsrk
    topo data Antennae_fld1_topo Antennae_fld1_topo


    Returns
    -------
    None.

    """
    ps_xdt = xr.open_datatree("Antennae_fld1_topo.ps.zarr")
    origms = ps_xdt["Antennae_fld1_topo_0"]
    t = origms.time.data
    tas = Time(t, format="unix", scale="utc")
    origfreq = origms.frequency.data
    lsrkms = ms_spectral_frame_conversion(origms)
    finalfreq = lsrkms.frequency.data
    # 4 times in the data
    t1 = Time(tas[0].datetime)
    t2 = Time(tas[20].datetime)
    t3 = Time(tas[40].datetime)
    t4 = Time(tas[50].datetime)
    for ta in (t1, t2, t3, t4):
        if not np.all(
            (
                lsrkms.VISIBILITY.sel(
                    time=ta.unix, baseline_id=1, method="nearest"
                ).argmax(dim="frequency")
                == 74
            ).data
        ):
            raise Exception("Simulated line is not in expected channel")
    # Some visual
    pl.figure()
    pl.imshow(
        np.abs(origms.VISIBILITY.data[20:60, 0, :, 0]),
        extent=[origfreq[0], origfreq[-1], tas[-1].datetime, tas[0].datetime],
        aspect="auto",
        interpolation="none",
    )
    pl.title("topo data")
    pl.figure()
    finalfreq = lsrkms.frequency.data
    pl.imshow(
        np.abs(lsrkms.VISIBILITY.data[20:60, 0, :, 0]),
        extent=[
            finalfreq[0],
            finalfreq[-1],
            tas[-1].datetime,
            tas[0].datetime,
        ],
        aspect="auto",
        interpolation="none",
    )
    pl.title("astroviper cvel")
    pl.figure()
    ps_casalsrk_xdt = xr.open_datatree("Antennae_fld1_casa_lsrk.ps.zarr")
    casams = ps_casalsrk_xdt["Antennae_fld1_casa_lsrk_0"]
    casafreq = casams.frequency.data
    pl.imshow(
        np.abs(casams.VISIBILITY.data[20:60, 0, :, 0]),
        extent=[casafreq[0], casafreq[-1], tas[-1].datetime, tas[0].datetime],
        aspect="auto",
        interpolation="none",
    )
    pl.title("casa cvel")
    pl.figure()
    pl.plot(
        casams.frequency,
        casams.VISIBILITY[40, 0, :, 0],
        label="casa cvel output",
    )
    pl.plot(
        lsrkms.frequency,
        lsrkms.VISIBILITY[40, 0, :, 0],
        label="astroviper cvel",
    )
    pl.legend()
    return True
