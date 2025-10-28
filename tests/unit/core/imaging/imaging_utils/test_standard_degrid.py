# -*- coding: utf-8 -*-
import numpy as np
import random
from astropy import constants as const

from astroviper.core.imaging.fft import fft_lm_to_uv
from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel_1D,
)
import unittest
from astroviper.core.imaging.imaging_utils.standard_degrid import (
    sgrid,
    sgrid_numba,
    dgrid,
    dgrid_numba,
    dgrid_optimized,
)


class TestStandardDegrid(unittest.TestCase):
    def setup_vis_uvw(self):
        npix = 200
        nant = 64
        ntime = 1
        nfreq = 1
        npol = 1
        self.freq_chan = np.array([1.4e9])
        self.grid_size = np.array([npix, npix])
        maxUV = 2e3
        cell = 0.5 * const.c.to("m/s").value / self.freq_chan[0] / maxUV
        self.cell_size = np.array([cell, cell])
        sources = [
            np.array([100]),
            np.array([100]),
            np.array([2.0]),
        ]
        mod_im = np.zeros((npix, npix), dtype=np.float64)
        mod_im[sources[0], sources[1]] = sources[2]
        self.grid = np.zeros([nfreq, npol, npix, npix], dtype=complex)
        self.grid[0, 0, :, :] = np.conj(fft_lm_to_uv(mod_im, axes=[0, 1]))
        self.vis_data = np.zeros((ntime, nant * nant, nfreq, npol), dtype=np.complex128)
        self.uvw = np.zeros((ntime, nant * nant, 3), dtype=np.float64)
        self.uvw[0, :, 0] = [random.uniform(-maxUV, maxUV) for i in range(nant * nant)]
        self.uvw[0, :, 1] = [random.uniform(-maxUV, maxUV) for i in range(nant * nant)]

    def test_sgrid_basic(self):
        """Test sgrid with basic valid inputs."""
        uvw = np.array([1.0, 2.0, 3.0])
        dphase = 0.0
        freq = 1.0e9  # 1 GHz
        c = const.c.to("m/s").value  # Speed of light
        scale = np.array([1.0e-4, 1.0e-4, 0.0])
        offset = np.array([500.0, 500.0, 0.0])
        sampling = 1
        pos = np.zeros(3)
        loc = np.zeros(3, dtype=int)
        off = np.zeros(3, dtype=int)

        phasor = sgrid(uvw, dphase, freq, c, scale, offset, sampling, pos, loc, off)

        # Expected values (calculate based on the sgrid logic)
        expected_pos_x = scale[0] * uvw[0] * freq / c + offset[0]
        expected_pos_y = scale[1] * uvw[1] * freq / c + offset[1]
        expected_loc_x = int(round(expected_pos_x))
        expected_loc_y = int(round(expected_pos_y))
        expected_off_x = int(round((expected_loc_x - expected_pos_x) * sampling))
        expected_off_y = int(round((expected_loc_y - expected_pos_y) * sampling))
        expected_phase = -2.0 * np.pi * dphase * freq / c
        expected_phasor = complex(np.cos(expected_phase), np.sin(expected_phase))

        # Assertions
        self.assertAlmostEqual(phasor.real, expected_phasor.real)
        self.assertAlmostEqual(phasor.imag, expected_phasor.imag)
        self.assertTrue(
            np.allclose(pos[:2], np.array([expected_pos_x, expected_pos_y]))
        )
        self.assertTrue(
            np.array_equal(loc[:2], np.array([expected_loc_x, expected_loc_y]))
        )
        self.assertTrue(
            np.array_equal(off[:2], np.array([expected_off_x, expected_off_y]))
        )

    def test_sgrid_numba(self):
        """Test sgrid with numba valid inputs."""
        uvw = np.array([1.0, 2.0, 3.0])
        dphase = 0.0
        freq = 1.0e9  # 1 GHz
        c = 299792458.0  # Speed of light
        scale = np.array([1.0e-4, 1.0e-4, 0.0])
        offset = np.array([500.0, 500.0, 0.0])
        sampling = 1

        phasor, loc, off = sgrid_numba(uvw, dphase, freq, c, scale, offset, sampling)

        # Expected values (calculate based on the sgrid logic)
        expected_pos_x = scale[0] * uvw[0] * freq / c + offset[0]
        expected_pos_y = scale[1] * uvw[1] * freq / c + offset[1]
        expected_loc_x = int(round(expected_pos_x))
        expected_loc_y = int(round(expected_pos_y))
        expected_off_x = int(round((expected_loc_x - expected_pos_x) * sampling))
        expected_off_y = int(round((expected_loc_y - expected_pos_y) * sampling))
        expected_phase = -2.0 * np.pi * dphase * freq / c
        expected_phasor = complex(np.cos(expected_phase), np.sin(expected_phase))

        # Assertions
        self.assertAlmostEqual(phasor.real, expected_phasor.real)
        self.assertAlmostEqual(phasor.imag, expected_phasor.imag)

        self.assertTrue(
            np.array_equal(loc[:2], np.array([expected_loc_x, expected_loc_y]))
        )
        self.assertTrue(
            np.array_equal(off[:2], np.array([expected_off_x, expected_off_y]))
        )

    def test_dgrid(self):
        self.setup_vis_uvw()
        offset = self.grid_size // 2
        flag = np.empty(self.vis_data.shape, dtype=bool)
        flag.fill(False)
        nt = self.vis_data.shape[0]
        nb = self.vis_data.shape[1]
        dphase = np.zeros([nt, nb], dtype=float)
        c = const.c.to("m/s").value
        chanmap = np.zeros(self.freq_chan.shape, dtype=int)
        polmap = np.zeros([1], dtype=int)
        convFunc = create_prolate_spheroidal_kernel_1D(100, 7)
        dgrid(
            self.uvw,
            dphase,
            self.vis_data,
            flag,
            self.cell_size,
            offset,
            self.grid,
            self.freq_chan,
            c,
            7,
            100,
            convFunc,
            chanmap,
            polmap,
        )
        self.assertEqual(np.max(np.abs(self.vis_data)), 2.0)
        dgrid_numba(
            self.uvw,
            dphase,
            self.vis_data,
            flag,
            self.cell_size,
            offset,
            self.grid,
            self.freq_chan,
            c,
            7,
            100,
            convFunc,
            chanmap,
            polmap,
        )
        self.assertEqual(np.max(np.abs(self.vis_data)), 4.0)
        dgrid_optimized(
            self.uvw,
            dphase,
            self.vis_data,
            flag,
            self.cell_size,
            offset,
            self.grid,
            self.freq_chan,
            c,
            7,
            100,
            convFunc,
            chanmap,
            polmap,
        )
        self.assertEqual(np.max(np.abs(self.vis_data)), 6.0)
