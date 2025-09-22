# run using eg
# python -m pytest ../astroviper/tests/imaging/test_feather.py

from astroviper.distributed.imaging import feather
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
    feather_expected = "feather.im"  # expected output for comparison
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
            coords={
                k: v
                for k, v in skel_xds.coords.items()
                if k in beam_dims + ["velocity"]
            },
        )

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
                xds["BEAM"][{"frequency": slice(min_chan, max_chan)}] = fx[
                    "BEAM"
                ].values
                xds["BEAM"].attrs = {"units": "rad"}
            if i == 0:
                xds_sd = xds
            else:
                xds_int = xds

        for xds, outfile in zip([xds_sd, xds_int], [cls.sd_zarr, cls.int_zarr]):
            cls._rm(outfile)
            write_image(xds, outfile, "zarr")

    @classmethod
    def _feather(cls, cores: int = 1, overwrite: bool = True) -> None:
        viper_client = local_client(
            cores=cores,
            memory_limit="2.0GiB",
            log_params={"log_level": "DEBUG"},
            worker_log_params={"log_level": "DEBUG"},
        )
        feather(
            outim={"name": cls.feather_out, "overwrite": overwrite},
            highres=cls.int_zarr,
            lowres=cls.sd_zarr,
            sdfactor=1,
        )
        viper_client.close()

    @classmethod
    def _ensure_feather_output(
        cls, regenerate: bool = False, cores: int = 1, overwrite: bool = True
    ) -> None:
        cls._ensure_inputs()
        # If a leftover regular file exists at the zarr path, remove it
        if os.path.isfile(cls.feather_out):
            cls._rm(cls.feather_out)
        # Generate only if the zarr directory doesn't exist
        if regenerate or not os.path.isdir(cls.feather_out):
            cls._feather(cores=cores, overwrite=overwrite)
        # Final sanity: must be a directory now
        if not os.path.isdir(cls.feather_out):
            cls.fail(
                f"Expected zarr directory at {self.feather_out}, but it was not created."
            )


class FeatherTest(FeatherShared, unittest.TestCase):

    # ------------------------------------------------------------------------
    def setUp(self):
        pass

    # ------------------------------------------------------------------------
    def test_feather(self):
        # ensure inputs exist once, reused across tests/classes
        self._ensure_inputs()
        download(self.feather_expected)  # expected result file used for comparison

        exp_fds = read_image("feather.im")
        xds_sd = load_image(self.sd_zarr)
        xds_int = load_image(self.int_zarr)

        for cores in (1, 4):
            self._feather(cores=cores, overwrite=True)
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
        self._rm(self.feather_expected)  # cleanup after test

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
            self._feather(cores=4, overwrite=False)
        except RuntimeError:
            pass
        else:
            self.fail("Feather should have failed to overwrite")
        # test if overwrite specified but not bool
        try:
            self._feather(cores=4, overwrite="true")  # overwrite not bool

        except TypeError:
            print("TypeError raised as expected")
        else:
            self.fail("Feather should have failed to run because overwrite is not bool")


class FeatherModelComparison(FeatherShared, unittest.TestCase):
    """Comparisons between feathered image and model; uses artifacts built once.

    This class does **not** regenerate inputs; it uses the shared zarrs and
    triggers a single feather run only if the output is missing.
    """

    model_key = "feather_model_convolved"
    model_image = "feather_model_convolved.im"

    @classmethod
    def setUpClass(cls):
        # Use the key, not the filename
        try:
            download(cls.model_key)
        except Exception:
            # fallback: refresh index, then retry
            try:
                from toolviper.utils.data import (
                    update,
                )  # local import to avoid global change

                update()
                download(cls.model_key)
            except Exception as e:
                raise unittest.SkipTest(f"Could not download '{cls.model_key}': {e}")

        # Make the failure obvious if the download location differs
        if not os.path.exists(cls.model_image):
            raise unittest.SkipTest(
                f"Model image not found after download: {os.path.abspath(cls.model_image)}"
            )
        # ensure feathered output exists (built by FeatherTest or here once)
        cls._ensure_feather_output(regenerate=False, cores=4, overwrite=True)

    @classmethod
    def tearDownClass(cls):
        cls._rm(cls.model_image)

    def test_basic_stats_and_positions(self):

        feather_xds = load_image(self.feather_out)
        # expected shape from prior runs
        self.assertEqual(
            feather_xds.SKY.shape,
            (1, 16, 1, 1024, 1024),
            f"Unexpected feathered SKY shape: {feather_xds.SKY.shape}",
        )

        # load model for comparison
        model_xds = read_image(self.model_image)
        self.assertEqual(
            feather_xds.SKY.shape,
            model_xds.SKY.shape,
            f"feather/model shape mismatch: {feather_xds.SKY.shape} vs {model_xds.SKY.shape}",
        )

        # sums
        feather_plane = feather_xds.SKY.isel(frequency=0)
        model_plane = model_xds.SKY.isel(frequency=0)
        fsum = float(feather_plane.sum().compute().values)
        msum = float(model_plane.sum().compute().values)
        rel = fsum / msum - 1.0
        self.assertAlmostEqual(
            fsum, 21276.0859375, delta=1e-3, msg=f"feather sum got {fsum}"
        )
        self.assertAlmostEqual(msum, 21275.71, delta=1e-2, msg=f"model sum got {msum}")

        # global max positions and values
        f_vals = feather_plane.compute().values
        m_vals = model_plane.compute().values
        f_pos = list(np.unravel_index(int(np.argmax(f_vals)), f_vals.shape))
        m_pos = list(np.unravel_index(int(np.argmax(m_vals)), m_vals.shape))
        self.assertEqual(f_pos, [0, 0, 452, 488], f"feather max pos {f_pos}")
        self.assertEqual(m_pos, [0, 0, 452, 488], f"model max pos {m_pos}")
        self.assertEqual(f_pos, m_pos, f"max positions differ: {f_pos} vs {m_pos}")

        f_max = float(feather_plane.max().compute().values)
        m_max = float(model_plane.max().compute().values)
        self.assertAlmostEqual(f_max, 1.3862089, delta=5e-7, msg=f"feather max {f_max}")
        self.assertAlmostEqual(m_max, 1.4138035, delta=5e-7, msg=f"model max {m_max}")

    def test_center_region_and_width_inference(self):
        self._ensure_feather_output(regenerate=False, cores=4, overwrite=True)
        feather_xds = load_image(self.feather_out)
        model_xds = read_image(self.model_image)

        feather_plane = feather_xds.SKY.isel(frequency=0)
        model_plane = model_xds.SKY.isel(frequency=0)

        center = feather_xds.SKY.shape[3] // 2
        pslice = slice(center - 20, center + 20)
        f_center = feather_plane.isel(l=pslice, m=pslice)
        m_center = model_plane.isel(l=pslice, m=pslice)

        f_pos = list(
            np.unravel_index(int(np.argmax(f_center.compute().values)), f_center.shape)
        )
        m_pos = list(
            np.unravel_index(int(np.argmax(m_center.compute().values)), m_center.shape)
        )
        self.assertEqual(f_pos, [0, 0, 19, 19], f"feather center max pos {f_pos}")
        self.assertEqual(m_pos, [0, 0, 20, 20], f"model center max pos {m_pos}")
        delta = (np.array(f_pos) - np.array(m_pos)).tolist()
        self.assertEqual(delta, [0, 0, -1, -1], f"center offset {delta}")

        f_peak = float(f_center[tuple(f_pos)].compute().item())
        m_peak = float(m_center[tuple(f_pos)].compute().item())
        self.assertAlmostEqual(
            f_peak, 0.5338305830955505, delta=5e-10, msg=f"feather center {f_peak}"
        )
        self.assertAlmostEqual(
            m_peak, 0.5857419371604919, delta=5e-10, msg=f"model center {m_peak}"
        )

        rel = f_peak / m_peak
        width_pct = (np.sqrt(1.0 / rel) - 1.0) * 100.0
        self.assertAlmostEqual(
            width_pct, 4.7, delta=0.2, msg=f"width pct {width_pct:.3f}%"
        )


# Module-level cleanup after all tests in this file finish


def tearDownModule(module=None):
    for f in [
        FeatherShared.int_image,
        FeatherShared.sd_image,
        FeatherShared.feather_out,
        FeatherShared.int_zarr,
        FeatherShared.sd_zarr,
        FeatherModelComparison.model_image,
    ]:
        FeatherShared._rm(f)
