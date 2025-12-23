import numpy as np
import xarray
from xradio.image.image import make_empty_sky_image
from astroviper.core.imaging.imaging_utils.standard_image_grid_preparation import (
    _mult_div,
    remove_padding,
    apply_pb,
    make_empty_padded_uv_image,
)
import unittest


class TestStandardImGridPrep(unittest.TestCase):
    def make_image_xds(self, npol=1, nchan=1, which="SKY"):
        """

        Parameters
        ----------
        npol : TYPE, optional
            DESCRIPTION. The default is 1.
        nchan : TYPE, optional
            DESCRIPTION. The default is 1.
        which : TYPE, optional
            which data variable to add. The default is "SKY".

        Returns
        -------
        a : TYPE
            DESCRIPTION.

        """
        pc = [1.0, 0.0]  # phasecenter
        imsize = [200, 200]  # image size
        cellsize = [4.8e-6, 4.8e-6]  # close to 1arcsec
        freq = np.arange(1, nchan + 1) * 1e8 + 1e9  # frequency
        allpol = np.array(["RR", "LL", "LR", "RL"])
        pol = allpol[:npol]
        epoch = [5.7e4]  # times in MJD
        a = make_empty_sky_image(
            phase_center=pc,
            image_size=imsize,
            cell_size=cellsize,
            frequency_coords=freq,
            pol_coords=pol,
            time_coords=epoch,
        )

        sky_data_dims = ("time", "frequency", "polarization", "l", "m")
        sky_data_shape = (
            len(epoch),
            len(freq),
            len(pol),
            imsize[0],
            imsize[1],
        )
        sky_coords = {dim: a.coords[dim] for dim in sky_data_dims}
        a[which] = xarray.DataArray(
            np.ones(sky_data_shape, dtype=np.float32),
            coords=sky_coords,
            dims=sky_data_dims,
        )
        return a

    def test_remove_padding(self):
        a = self.make_image_xds()
        ra_o, dec_o = (
            a.right_ascension[99, 99].data,
            a.declination[99, 99].data,
        )
        a = remove_padding(a, [180, 180])
        ra_f, dec_f = (
            a.right_ascension[89, 89].data,
            a.declination[89, 89].data,
        )
        self.assertEqual(ra_f, ra_o)
        self.assertEqual(dec_f, dec_o)

    ###################################
    def test_make_empty_padded_uv_image(self):
        a = self.make_image_xds(which="RESIDUAL")
        b = make_empty_padded_uv_image(a, [250, 250], "VISIBILITY_RESIDUAL")
        self.assertEqual(b.sizes["u"], 250)
        # print(
        #    "SHAPES",
        #    b["VISIBILITY_RESIDUAL"].shape,
        #    b["VISIBILITY_RESIDUAL_NORMALIZATION"].shape,
        # )
        # print(b.attrs)
        self.assertTrue("residual" in b.attrs["data_groups"]["base"])
        self.assertEqual(b["VISIBILITY_RESIDUAL"].shape[3], 250)

    ##########################
    def test_mult_div(self):

        a = self.make_image_xds(npol=1, nchan=2)
        sky_dims = a["SKY"].dims
        sky_coords = a["SKY"].coords
        sky_shape = a["SKY"].shape
        a["PRIMARY_BEAM"] = xarray.DataArray(
            np.ones(sky_shape, dtype=np.float32),
            coords=sky_coords,
            dims=sky_dims,
        )
        a["PRIMARY_BEAM"].data[0, 1, 0, :, :] = 0
        _mult_div(a["SKY"], a["PRIMARY_BEAM"], multiply=True)
        self.assertEqual(np.max(a["SKY"].data[0, 0, 0, :, :]), 1.0)

    ######
    def test_apply_pb(self):

        a = self.make_image_xds(npol=1, nchan=2)
        sky_dims = a["SKY"].dims
        sky_coords = a["SKY"].coords
        sky_shape = a["SKY"].shape
        a["PRIMARY_BEAM"] = xarray.DataArray(
            np.ones(sky_shape, dtype=np.float32),
            coords=sky_coords,
            dims=sky_dims,
        )
        a["PRIMARY_BEAM"].data[0, 1, 0, :, :] = 0
        apply_pb(a, data_vars="SKY", multiply=True)
        self.assertEqual(np.max(a["SKY"].data[0, 0, 0, :, :]), 1.0)
