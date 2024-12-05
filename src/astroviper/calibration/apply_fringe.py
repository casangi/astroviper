from xradio.vis.read_processing_set import read_processing_set

import dask
import numpy as np
import xarray as xa
import pandas as pd
import datetime

# I am surprised this is not some kind of standard function, but the
# pandas version scolds me for using it on a string.
#
# This implementation is O(n^2) but if that is ever an issue we are
# doing something else very wrong.
def unique(s):
    "Get the unique characters in a string in the order they occur in the original"
    u = []
    for c in s:
        if c not in u:
            u.append(c)
    return u

def nanCount(j):
    return np.sum(np.isnan(j))

def numberCount(j):
    return np.sum(~np.isnan(j))


def makeCalTable(xds):
    "An attempt to make a calibration table out of coordinates"
    pols_ant = unique(''.join([c for c in ''.join(xds.polarization.values)]))
    coords = xa.Coordinates(coords={'time' : xa.Coordinates(coords = {'time' : 0.5*(xds.time[0]+xds.time[-1])}),
                                    'antenna_name' : xds.antenna_xds.antenna_name,
                                    'polarization' : pols_ant,
                                    'parameter' : ['one', 'two', 'three', 'ah-ha']
                                    })
    cds = xa.Dataset(data_vars = dict(cals=(coords.sizes.keys(), np.zeros(tuple(coords.sizes.values()), complex))),
                     coords=coords)
    return cds

#############################################################################
# How to make a (time, frequency) grid:
#
# np.expand_dims(dt, 1) + np.expand_dims(df, 0) => array of shape (nt, nf)
#############################################################################

class GridJonesCalculator(object):
    def __init__(self, xds):
        """
"""
        # We scrape all the metadata from an xds. Maybe this is wise, maybe not.
        self.xds = xds
        self.frequency = xds.frequency
        self.time = xds.time
        self.baseline_id = xds.baseline_id
        self.VISIBILITY = xds.VISIBILITY
        self.n_ants = xds.antenna_xds.antenna_name.size
        # We expand out a copy of the baseline_antenna1_id array to have shape
        # (1, n_baselines, 1, 1, 1) 
        self.ant1_mask = np.expand_dims(xds.baseline_antenna1_name.values, (0, 2, 3, 4))
        self.ant2_mask = np.expand_dims(xds.baseline_antenna2_name.values, (0, 2, 3, 4))
        self.makeAccumulatedJoneses()

    def makeAccumulatedJoneses(self):
        vcs = self.VISIBILITY.shape
        # We are going to use 2x2 matrices for our Jones matrices because that's how they multiply
        assert vcs[-1] == 4 # We'll figure other cases out later
        new_shape = vcs[:-1] + (2, 2)
        self.j_a1_composed = np.zeros(new_shape, complex)
        self.j_a1_composed = np.identity(2)
        self.j_a2_composed = np.ones(new_shape, complex)
        self.j_a2_composed = np.identity(2)

    def insertBaselineDimension(self, j, nbaselines):
        """We add a baseline dimension, but we also broadcast to it"""
        # Insert a baseline dimension to a calibration matrix.
        j_shape = j.shape
        new_shape = j_shape[:1] + (nbaselines,) + j_shape[1:]
        j = np.expand_dims(j, 1)
        j = np.broadcast_to(j, new_shape)
        return j

    def calcGridJonesAnt(self, fp, df, dt):
        """Return Jones matrices for all grid points of an xds for a single set of fringefit parameters (which come in pairs one for each polarization)"""
        # We now assume phi0, tau and r are 2-vectors
        phi0, tau, r = fp
        # We upscale the dimensions so that things broadcast nicely:
        df_shaped = np.expand_dims(df, (0, 2))
        dt_shaped = np.expand_dims(dt, (1, 2))
        phi_shaped = np.expand_dims(phi0, (0, 1))
        # Calculate phases:
        phi = (phi_shaped +
               2*np.pi*tau.values*df_shaped +
               2*np.pi*r.values*dt_shaped)
        # And then phasors.
        # (I spent a long time trying to express this in a neat numpy way.
        # I did not succeed, so I do it the ugly stupid way for now.)
        many_jones_diags = np.exp(1J*phi, dtype=complex)
        many_jones = np.zeros(many_jones_diags.shape + (2,), complex)
        many_jones[:, :, 0, 0] = many_jones_diags[:, :, 0]
        many_jones[:, :, 1, 1] = many_jones_diags[:, :, 1]
        return many_jones

    def calcGridJones(self, cal_quantum):
        dt = (self.time -  cal_quantum.t_ref).values
        df = (self.frequency - cal_quantum.f_ref).values
        # This needs fixed too
        for iant, ant in enumerate(cal_quantum.coords['antenna_name'].values):
            fp = cal_quantum.sel(antenna_name=ant)
            params = np.sum(~np.isnan(fp.values))
            print(f"{ant=} {params=}")
            if params == 0:
                continue
            print(f"{fp.values}")
            j = self.calcGridJonesAnt(fp, df, dt)
            count = np.sum(~np.isnan(j))
            print(f"{count=}")
            print(f"{np.max(np.abs(j.flatten()))=}")
            j = self.insertBaselineDimension(j, self.xds.baseline_id.size)
            # Then we can make a version of our baseline jones matrices that only affects a specific ant1:
            j_1_mask = np.where(self.ant1_mask != ant, j, 1)
            j_2_mask = np.where(self.ant2_mask != ant, j, 1)
            # The second array needs to be hermitianized on the last two axes, which we have to do by hand
            j_a2 = j.transpose(0, 1, 2, 4, 3).conj()
            # Then we can apply those entries to our corrected data array by multiplication:
            # First antenna corrected by multiplication from the left:
            print(f"{np.max(np.abs(self.j_a1_composed.flatten()))=}")
            self.j_a1_composed = np.matmul(j, self.j_a1_composed)
            # Second antenna in baseline corrected (by Hermitian matrix) from the right
            self.j_a2_composed = np.matmul(self.j_a2_composed, j)
            if False:
                print(f"{numberCount(self.j_a1_composed)=}")
                print(f"{nanCount(self.j_a1_composed)=}")
                print(f"{numberCount(self.j_a2_composed)=}")
                print(f"{nanCount(self.j_a2_composed)=}")
                print(f"{np.max(np.abs(self.j_a1_composed.flatten()))=}")


# We need to consult the data for polarizations now.
ps = read_processing_set('n14c3.zarr')
ps.keys()

# Current version of this ps is split by SPW and not by field
xds = ps['n14c3_099']

# In fact, all xdses have the same polarization setup here, but whomst can say if that is always true?
# Actually, I think maybe we could?
pols_ant = unique(''.join([c for c in ''.join(xds.polarization.values)]))
quantumCoords = xa.Coordinates(coords={'antenna_name' : xds.antenna_xds.antenna_name,
                                       'parameter' : range(3),
                                       'polarization' : pols_ant
                                       })
q = xa.DataArray(coords=quantumCoords)

q.attrs['f_ref'] = xds.frequency[0]
q.attrs['t_ref'] = xds.time[0]

# Note that the f_ref attr copies over a lot of metadata.
# Which I think is a good thing?
q.attrs['f_ref'].attrs['spectral_window_name']


# You can't assign to DataArrays by name, only by integer index.
q[0] = [[0.0, 0.0], [1.0e-9,-1.0e-9], [0, 0]]
 
gjc = GridJonesCalculator(xds)
gjc.calcGridJones(q)

def squareUpLastDimension(v):
    s = v.shape[:-1] + (2,2)
    v2 = np.reshape(v, s)
    return v2




v = squareUpLastDimension(xds.VISIBILITY.values)
# And this is the payoff I guess
v2 = gjc.j_a1_composed @ v @ gjc.j_a2_composed

# This works, although we should do better along the polarization axis.
xds.assign({'FROBBED' : xa.DataArray( coords=(xds.time, xds.baseline_id, xds.frequency, pols_ant, pols_ant), data=v2)})

# Isn't there meant to be a nice way to get spw now?
#>>> xds.partition_info['spectral_window_name']

# We can also now do this at ps level:
ps2 = ps.sel(spw_name='spw_0')
