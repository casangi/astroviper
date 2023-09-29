import numpy as np
import xarray as xr

def _make_time_coord(time_start="2019-10-03T19:00:00.000", time_delta=3600, n_samples=10, time_scale='utc'):
    from astropy.timeseries import TimeSeries
    from astropy.time import Time
    from astropy import units as u
    ts = np.array(
        TimeSeries(
            time_start=time_start, time_delta=time_delta * u.s, n_samples=n_samples,
        ).time.unix
    )

    return {'dims':'time','data':ts,'attrs':{"units":"s","type":"time","format":"unix","time_scale":time_scale}}
        
def _make_frequency_coord(
    freq_start=3 * 10**9,
    freq_delta=0.4 * 10**9,
    n_channels=50,
    velocity_frame='lsrk'
):
    freq_chan = (np.arange(0, n_channels) * freq_delta + freq_start).astype(float)
    return {'dims':'frequency','data':freq_chan,'attrs':{"units":"Hz","type":"spectral_coord","velocity_frame":velocity_frame}}

def _make_parallel_coord(coord, n_chunks=None):

    if isinstance(coord,xr.core.dataarray.DataArray):
        coord = coord.copy(deep=True).to_dict()

    n_samples = len(coord['data'])
    parallel_coord = {}
    parallel_coord['data'] = coord['data']
    parallel_coord['data_chunks'] = dict(zip(np.arange(n_chunks),np.array_split(coord['data'], n_chunks)))
    parallel_coord['dims'] = coord['dims']
    parallel_coord['attrs'] = coord['attrs']
    return parallel_coord
