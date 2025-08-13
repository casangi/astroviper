# run using eg
# python -m pytest ../astroviper/tests/imaging/test_feather.py

from astroviper.imaging import feather
import copy
import dask.array as da
import numpy as np
import os
import shutil
from toolviper.dask.client import local_client
from toolviper.utils.data import download
import unittest
import xarray as xr
from xradio.image.image import load_image, make_empty_sky_image, read_image, write_image


class FeatherShared:
    """Shared artifacts and helpers for feather tests across classes."""
    int_image = "feather_sim_vla_c1_pI.im"
    sd_image = "feather_sim_sd_c1_pI.im"
    feather_out = "feather.zarr"
    int_zarr = "int.zarr"
    sd_zarr = "sd.zarr"

    @staticmethod
    def _rm(path: str) -> None:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    @classmethod
    def _ensure_inputs(cls) -> None:
        """Create int/sd zarr inputs once; reuse across tests/classes."""
        if os.path.exists(cls.int_zarr) and os.path.exists(cls.sd_zarr):
            return
        # downloads (idempotent)
        download(cls.sd_image)
        download(cls.int_image)
        # build skeleton and select center window if needed
        imsize = [1024, 1024]
        nchan = 16
        rad_per_arcsec = np.pi / 180 / 3600
        skel_xds = make_empty_sky_image(
            phase_center=[0.6, -0.2],
            image_size=imsize,
            cell_size=[15 * rad_per_arcsec, 15 * rad_per_arcsec],
            chan_coords=np.linspace(1.4e9, 1.5e9, nchan),
            pol_coords=["I"],
            time_coords=[0],
        )
        sel_dict = {}
        if imsize[0] < 4096:
            blc = 2048 - imsize[0] // 2
            l_slice = slice(blc, blc + imsize[0])
            sel_dict["l"] = l_slice
        if imsize[1] < 4096:
            blc = 2048 - imsize[1] // 2
            m_slice = slice(blc, blc + imsize[1])
            sel_dict["m"] = m_slice

        xds_sd_temp = read_image(cls.sd_image).isel(sel_dict)
        xds_int_temp = read_image(cls.int_image).isel(sel_dict)

        dm = skel_xds.sizes
        sky_da_zeros = da.zeros(
            [dm["time"], dm["frequency"], dm["polarization"], dm["l"], dm["m"]],
            dtype=np.float32,
        )
        sky_dims = list(skel_xds.dims)
        if "beam_param" in sky_dims:
            sky_dims.remove("beam_param")
        coords = ["time", "frequency", "polarization", "l", "m"]
        sky_coords = {c: skel_xds[c] for c in coords}
        sky_xa_zeros = xr.DataArray(data=sky_da_zeros, coords=sky_coords, dims=sky_dims)

        beam_da_zeros = da.zeros(
            [dm["time"], dm["frequency"], dm["polarization"], dm["beam_param"]],
            dtype=np.float32,
        )
        beam_dims = ["time", "frequency", "polarization", "beam_param"]
        beam_xa_zeros = xr.DataArray(
            beam_da_zeros.copy(),
            dims=beam_dims,
            coords={k: v for k, v in skel_xds.coords.items() if k in beam_dims + ["velocity"]},
        )

        exp_fds = read_image("feather.im")
        for i in (0, 1):
            xds = copy.deepcopy(skel_xds)
            xds["SKY"] = sky_xa_zeros.copy()
            xds["BEAM"] = beam_xa_zeros.copy()
            for j in range(0, nchan, 16):
                min_chan = j
                max_chan = min(j + 16, nchan)
                fx = xds_sd_temp if i == 0 else xds_int_temp
                xds["SKY"][{"frequency": slice(min_chan, max_chan)}] = fx["SKY"].values
                xds["SKY"].attrs = {"units": "Jy/beam"}
                xds["BEAM"][{"frequency": slice(min_chan, max_chan)}] = fx["BEAM"].values
                xds["BEAM"].attrs = {"units": "rad"}
            if i == 0:
                xds_sd = xds
            else:
                xds_int = xds

        for xds, outfile in zip([xds_sd, xds_int], [cls.sd_zarr, cls.int_zarr]):
            cls._rm(outfile)
            write_image(xds, outfile, "zarr")


class FeatherTest(FeatherShared, unittest.TestCase):

    int_image = "feather_sim_vla_c1_pI.im"
    sd_image = "feather_sim_sd_c1_pI.im"
    feather_out = "feather.zarr"
    int_zarr = "int.zarr"
    sd_zarr = "sd.zarr"

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _rm(path: str) -> None:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    @classmethod
    def _ensure_inputs(cls) -> None:
        """Create the int/sd zarr inputs once; reuse across tests/classes.
        Minimal change: this is the original generation logic lifted out of
        `test_feather`, executed only when needed.
        """
        if os.path.exists(cls.int_zarr) and os.path.exists(cls.sd_zarr):
            return  # already generated

        # downloads (idempotent)
        download(cls.sd_image)
        download(cls.int_image)

        # build skeleton and select center window if needed
        imsize = [1024, 1024]
        nchan = 16
        rad_per_arcsec = np.pi / 180 / 3600
        skel_xds = make_empty_sky_image(
            phase_center=[0.6, -0.2],
            image_size=imsize,
            cell_size=[15 * rad_per_arcsec, 15 * rad_per_arcsec],
            chan_coords=np.linspace(1.4e9, 1.5e9, nchan),
            pol_coords=["I"],
            time_coords=[0],
        )
        sel_dict = {}
        if imsize[0] < 4096:
            blc = 2048 - imsize[0] // 2
            l_slice = slice(blc, blc + imsize[0])
            sel_dict["l"] = l_slice
        if imsize[1] < 4096:
            blc = 2048 - imsize[1] // 2
            m_slice = slice(blc, blc + imsize[1])
            sel_dict["m"] = m_slice

        xds_sd_temp = read_image(cls.sd_image).isel(sel_dict)
        xds_int_temp = read_image(cls.int_image).isel(sel_dict)

        dm = skel_xds.sizes
        sky_da_zeros = da.zeros(
            [dm["time"], dm["frequency"], dm["polarization"], dm["l"], dm["m"]],
            dtype=np.float32,
        )
        sky_dims = list(skel_xds.dims)
        if "beam_param" in sky_dims:
            sky_dims.remove("beam_param")  # CI compatibility
        coords = ["time", "frequency", "polarization", "l", "m"]
        sky_coords = {c: skel_xds[c] for c in coords}
        sky_xa_zeros = xr.DataArray(data=sky_da_zeros, coords=sky_coords, dims=sky_dims)

        beam_da_zeros = da.zeros(
            [dm["time"], dm["frequency"], dm["polarization"], dm["beam_param"]],
            dtype=np.float32,
        )
        beam_dims = ["time", "frequency", "polarization", "beam_param"]
        beam_xa_zeros = xr.DataArray(
            beam_da_zeros.copy(),
            dims=beam_dims,
            coords={k: v for k, v in skel_xds.coords.items() if k in beam_dims + ["velocity"]},
        )

        exp_fds = read_image("feather.im")  # downloaded in test_feather if needed
        for i in (0, 1):
            xds = copy.deepcopy(skel_xds)
            xds["SKY"] = sky_xa_zeros.copy()
            xds["BEAM"] = beam_xa_zeros.copy()
            for j in range(0, nchan, 16):
                min_chan = j
                max_chan = min(j + 16, nchan)
                fx = xds_sd_temp if i == 0 else xds_int_temp
                xds["SKY"][{"frequency": slice(min_chan, max_chan)}] = fx["SKY"].values
                xds["SKY"].attrs = {"units": "Jy/beam"}
                xds["BEAM"][{"frequency": slice(min_chan, max_chan)}] = fx["BEAM"].values
                xds["BEAM"].attrs = {"units": "rad"}
            if i == 0:
                xds_sd = xds
            else:
                xds_int = xds

        # write inputs once (idempotent behavior)
        for xds, outfile in zip([xds_sd, xds_int], [cls.sd_zarr, cls.int_zarr]):
            cls._rm(outfile)
            write_image(xds, outfile, "zarr")

    # ------------------------------------------------------------------------
    def setUp(self):
        pass

    # ------------------------------------------------------------------------
    def test_feather(self):
        # ensure inputs exist once, reused across tests/classes
        self._ensure_inputs()
        download("feather.im")  # expected result file used for comparison

        exp_fds = read_image("feather.im")
        xds_sd = load_image(self.sd_zarr)
        xds_int = load_image(self.int_zarr)

        log_params = {"log_level": "DEBUG"}
        worker_log_params = {"log_level": "DEBUG"}
        for cores in (1, 4):
            # clean output safely (file or dir)
            self._rm(self.feather_out)
            viper_client = local_client(
                cores=cores,
                memory_limit="8.0GiB",
                log_params=log_params,
                worker_log_params=worker_log_params,
            )
            feather(
                outim={"name": self.feather_out, "overwrite": True},
                highres=self.int_zarr,
                lowres=self.sd_zarr,
                sdfactor=1,
            )
            viper_client.close()
            feather_xds = load_image(self.feather_out)
            self.assertEqual(
                feather_xds["SKY"].shape, xds_sd["SKY"].shape, "Incorrect sky shape"
            )
            self.assertTrue(
                (feather_xds["BEAM"].values == xds_int["BEAM"].values).all(),
                "Incorrect beam values",
            )
            self.assertTrue(
                np.isclose(
                    feather_xds["SKY"].values, exp_fds["SKY"].values, atol=2e-7
                ).all(),
                "Incorrect sky values",
            )

    def test_overwrite(self):
        """Test overwrite option using prebuilt int/sd zarr inputs"""
        # ensure inputs exist even if this test runs first
        self._ensure_inputs()
        # ensure output path isn't a leftover directory from prior test run
        self._rm(self.feather_out)

        open(self.feather_out, "w").close()  # create a temp file, not a dir
        self.assertTrue(
            os.path.exists(self.feather_out), "Feather output file not created"
        )
        # test overwrite not present defaults to False by testing for exception
        try:
            feather(
                outim={
                    "name": self.feather_out,
                },
                highres=self.int_zarr,
                lowres=self.sd_zarr,
                sdfactor=1,
            )
        except RuntimeError:
            print("RuntimeError raised as expected")
        else:
            self.fail("Feather should have failed to overwrite")
        # test if overwrite specified but not bool
        try:
            feather(
                outim={"name": self.feather_out, "overwrite": 1},
                highres=self.int_zarr,
                lowres=self.sd_zarr,
                sdfactor=1,
            )
        except TypeError:
            print("TypeError raised as expected")
        else:
            self.fail("Feather should have failed to run because overwrite is not bool")


# Module-level cleanup after all tests in this file finish

def tearDownModule(module=None):
    for f in [
        FeatherShared.int_image,
        FeatherShared.sd_image,
        FeatherShared.feather_out,
        FeatherShared.int_zarr,
        FeatherShared.sd_zarr,
    ]:
        FeatherShared._rm(f)

