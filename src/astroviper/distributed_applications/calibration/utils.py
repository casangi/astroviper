"""
Some utility functions I found myself wanting while writing calibration code.
"""

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

