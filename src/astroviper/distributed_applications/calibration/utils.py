"""
Some utility functions I found myself wanting while writing calibration code.
"""

import xarray as xr
import numpy as np



def make_scalar(val, unit):
    return {'attrs' : {'type': 'quantity',
                       'units': unit},
            'data' : value,
            'dims' : []}


def multiply_scalar_in_place(m, s):
    s['data'] *= s['data']
    return None # For clarity; we mod in place and return None

def unique(s):
    "Get the unique characters in a string in the order they occur in the original"
    return list(set(s))

def power_of_two_grid(freqs, df=None, eps=1e-7):
    """Uniformize a mostly uniform PS frequency grid.

The combined frequency grid of EVN measurement sets (and
possibly others) are almost, but not quite, uniform; the DC edges of
upper and lower sidebands appears in both spectral windows, and there
is a compensating gap at the high end of the upper sideband. This does
not spark joy, but we can code around it.
    """
    if df == None:
        df = freqs[1]-freqs[0]
    dfs = freqs - freqs[0]
    ichans = dfs //  df
    offsets = np.abs(((dfs/df + 0.5) % 1) - 0.5)
    unevenness = np.max(np.abs(offsets))
    if unevenness > eps:
        raise ValueError("Uneven grid")
    power_of_two = np.pow(2, int(np.ceil(np.log2(np.max(ichans)))))
    freqs2 = np.linspace(freqs[0], freqs[0]+power_of_two*df, power_of_two, False)
    return freqs2


def get_ps_polarization_coord(ps):
    """There is no processing-set-level aggregator for polarization axis

This is a stub! I should make an aggregator!"""
    pol_coords = []
    children_names = list(ps.children)
    x = ps[children_names[0]]
    pol_coords = x.polarization 
    for n in children_names[1:]:
        pol_coords2 = ps[n].polarization
        if not pol_coords.coords.equals(pol_coords2.coords):
            raise ValueError("Polarization mismatch")
    return pol_coords
        
def make_cal_map(cal_tree):
    spw_to_cal_map = {}
    for s in cal_tree.children:
        spw_to_cal_map.setdefault(cal_tree[s].spectral_window_name, []).append(cal_tree[s])
    assert max([len(v) for v in spw_to_cal_map.values()])==1
    return {k:v[0] for k,v in spw_to_cal_map.items()}

def make_uniform_freq_grid(ps):
    fs0 = ps.xr_ps.get_freq_axis().values
    fs1 = power_of_two_grid(fs0)
    return fs1

def ds_children_identical(xdts, k):
    """
    Is a given child xds (really an xdt) identical for an iterable collection of xdt's?

    Ignoring inherited axes.
    """
    child_dses = [x[k].to_dataset(inherit=False) for x in xdts]
    return datasets_identical(child_dses)

def datasets_identical(ds):
    if len(ds) < 2:
        return True
    else:
        return all([ds[0].identical(d) for d in ds[1:]])
                    
def combine_xdses(ps):
    """Combine a set of xds datasets along the frequency axis.

For now, we only support the case where the metadata is otherwise compatible.
"""
    # We only use the ps for this call:
    big_freq_coord = make_uniform_freq_grid(ps)
    xdses = [ps[x] for x in ps.children if ps[x].VISIBILITY.size > 0]
    if not ds_children_identical(xdses, 'antenna_xds'):
        raise ValueError("Discrepency in antenna_xds")

    # We are going to check that coordinates for each xds are the same except for frequency and uvw_label, which we discard
    x = xdses[0]
    new_coords = set(x.coords) - set(['frequency', 'uwv_label'])
    empty_datasets = [xr.Dataset({c:x.coords[c] for c in new_coords}) for x in xdses]
    if not datasets_identical(empty_datasets):
        raise valueError("Coordinate mismatch in datasets")

    # We use this for coordinates below:
    ds = empty_datasets[0]
    big_coords = xr.Coordinates({
        # 'baseline_id': ds.baseline_id,
        'frequency' : big_freq_coord,
        # 'polarization' : ds.polarization,
        'time' : ds.time,
        })

    interpolation_coords = ('baseline_id', 'frequency', 'polarization', 'time')
    # Amusingly, the order in arrays is (time, baseline, frequency, polarization)
    # FIXME: Handle flags!
    data_vars = [v for v in x.data_vars
                 if tuple(x[v].indexes) == interpolation_coords and v!='FLAG']
    # Can't interpolate flags!
    var_dict = {}
    for var_name in data_vars:
        all_var = []
        for xds in xdses:
            # Since we can't interpolate flags, we use them to nan values instead
            var = xds[var_name].where(np.logical_not(xds['FLAG']), np.nan)
            # Upper and lower bands share a DC edge in the visibilities for EVN
            # data, but the value there is of no use to anyone. The other edge
            # isn't either, and we need to wipe them so our merge onto a big
            # grid works nicely.
            var[{'frequency' : 0}] = np.nan
            var[{'frequency': -1}] = np.nan 
            all_var.append(var)
        # Interpolate each spectral window onto the unified grid.
        big_var_subs = [v.interp_like(big_coords) for v in all_var]
        # breakpoint()
        big_dataset = xr.merge(big_var_subs, compat='no_conflicts', combine_attrs='identical')
        # We only want the variable we had, so we extract it
        var_dict[var_name] = big_dataset[var_name]
        # and we add back a dummy flag array so no one has to get sad
        var_dict['FLAG'] = xr.DataArray(coords=ds.coords, data=False)
    combo_xds = xr.Dataset(data_vars=var_dict, coords=ds.coords)\
    # The antenna_xds member is already a DataTree but we can't use it
    # as-is because the interited coordinates it doesn't even use don't
    # match the new grid.
    antenna_xds = x['antenna_xds'].to_dataset(inherit=False)
    antenna_xdt = xr.DataTree(dataset=antenna_xds)
    combo_xdt = xr.DataTree(dataset=combo_xds, children=dict(antenna_xds=antenna_xdt))
    # We still have to set the frequency metadata, such fun!
    # We do the traditional CASA thing and adopt the metadata of the lowest-frequency spectral window:
    freq_attrs = sorted([x.frequency.attrs for x in xdses], key=lambda x: x['reference_frequency']['data'])[0]
    # If someone has a mix of frequency units this will fail and we will all point at someone and laugh.
    # Maybe me, maybe them, maybe both.
    combo_xdt.frequency.attrs = freq_attrs
    return combo_xdt

