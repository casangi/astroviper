import numpy as np
from astropy import constants as const
from astroviper.core.imaging.imaging_utils.standard_grid import *
from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import *
from astroviper.core.imaging.fft import fft_lm_to_uv
from astroviper.core.imaging.ifft import ifft_uv_to_lm

import unittest


class TestStandardGridNumpyWrap(unittest.TestCase):
    def setup_vis_uvw(self):
        nant = 200
        ntime = 1
        nfreq = 1
        npol = 1
        self.freq_chan = np.array([1.4e9])
        self.grid_size = np.array([nant, nant])
        maxUV = 2e3
        cell = 0.5 * const.c.to("m/s").value / self.freq_chan[0] / maxUV
        self.cell_size = np.array([cell, cell])
        sources = [
            np.array([110, 87]),
            np.array([115, 92]),
            np.array([1.0, 2.0]),
        ]
        mod_im = np.zeros((nant, nant), dtype=np.float64)
        mod_im[sources[0], sources[1]] = sources[2]
        ft_mod = np.conj(fft_lm_to_uv(mod_im, axes=[0, 1]))
        uv_axis = np.linspace(-maxUV, maxUV, nant)
        self.vis_data = np.zeros((ntime, nant * nant, nfreq, npol), dtype=np.complex128)
        self.weight = np.ones(self.vis_data.shape, dtype=np.float64)
        self.uvw = np.zeros((ntime, nant * nant, 3), dtype=np.float64)
        for u_idx, u in enumerate(uv_axis):
            for v_idx, v in enumerate(uv_axis):
                b_idx = u_idx * nant + v_idx
                self.uvw[0, b_idx, :] = np.array([u, v, 0.0])
                self.vis_data[0, b_idx, 0, 0] = ft_mod[u_idx, v_idx]

        # vis_data, uvw, weight, freq_chan, cgk_1D, params

    def test_standard_grid_numpy_wrap_input_checked(self):
        """Test standard_grid_numpy_wrap_input_checked
        We test making a complex grid and FFTing that
        along testing for making a psf

        """
        self.setup_vis_uvw()
        oversampling = 100
        support = 7
        cgk_1D = create_prolate_spheroidal_kernel_1D(oversampling, support)
        grid, sumwt = standard_grid_numpy_wrap_input_checked(
            vis_data=self.vis_data,
            uvw=self.uvw,
            weight=self.weight,
            freq_chan=self.freq_chan,
            cgk_1D=cgk_1D,
            image_size=self.grid_size,
            cell_size=self.cell_size,
            oversampling=oversampling,
            support=support,
            complex_grid=True,
            do_psf=False,
            chan_mode="continuum",
        )
        kernel, corrTerm = create_prolate_spheroidal_kernel(
            oversampling, support, self.grid_size
        )
        dirty_im = (
            ifft_uv_to_lm(grid, axes=[2, 3])
            / corrTerm
            * self.grid_size[0]
            * self.grid_size[1]
            / sumwt
        )
        print(dirty_im.shape, dirty_im[0, 0, 87, 92], dirty_im[0, 0, 110, 115])
        self.assertGreater(dirty_im[0, 0, 87, 92], 1.8)
        self.assertGreater(dirty_im[0, 0, 110, 115], 0.9)
        grid, sumwt = standard_grid_numpy_wrap_input_checked(
            vis_data=self.vis_data,
            uvw=self.uvw,
            weight=self.weight,
            freq_chan=self.freq_chan,
            cgk_1D=cgk_1D,
            image_size=self.grid_size,
            cell_size=self.cell_size,
            oversampling=oversampling,
            support=support,
            complex_grid=True,
            do_psf=True,
            chan_mode="continuum",
        )
        dirty_psf = (
            ifft_uv_to_lm(grid, axes=[2, 3])
            / corrTerm
            * self.grid_size[0]
            * self.grid_size[1]
            / sumwt
        )
        self.assertGreater(0.01, np.abs(dirty_psf[0, 0, 100, 100] - 1.0))
