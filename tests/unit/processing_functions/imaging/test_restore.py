"""Unit tests for :mod:`astroviper.processing_functions.imaging.restore`.

The tests build synthetic image datasets (no downloaded data) and check that
``restore_image``:

- restores ``model * clean_beam + residual`` with a unit-peak beam,
- builds the clean beam with the same orientation/size convention as the PSF
  Gaussian fit (validated by a round-trip through
  :func:`point_spread_function_gaussian_fit`),
- preserves dtype, registers the restored data group, and does not mutate the
  input residual/model variables,
- falls back to the residual where the beam is undefined or the model is empty.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.restore import (
    _clean_beam_kernel_ft,
    _convolve_with_clean_beam,
    _elliptical_gaussian_kernel,
    _inverse_rfft2,
    _multiply_by_correcting_transform,
    _restore_image_arrays,
    primary_beam_corrected_plane,
    restore_image,
)


def _make_restore_xds(
    nt=1,
    nf=1,
    npol=2,
    ny=64,
    nx=64,
    beams=None,
    residual=None,
    model=None,
    dtype=np.float32,
    delta=1.0e-3,
):
    """Build a minimal image Dataset with residual + model data groups.

    ``beams`` maps ``(t, f)`` -> ``(major_fwhm_rad, minor_fwhm_rad, pa_rad)`` and
    is broadcast across polarizations.  Defaults: an 8x4 pixel beam at pa=0,
    a single unit point source at the centre of the model, and a zero residual.
    """
    if residual is None:
        residual = np.zeros((nt, nf, npol, ny, nx), dtype=dtype)
    if model is None:
        model = np.zeros((nt, nf, npol, ny, nx), dtype=dtype)
        model[:, :, :, ny // 2, nx // 2] = 1.0

    if beams is None:
        beams = {
            (t, f): (8.0 * delta, 4.0 * delta, 0.0)
            for t in range(nt)
            for f in range(nf)
        }

    beam_arr = np.full((nt, nf, npol, 3), np.nan, dtype=np.float64)
    for (t, f), (maj, mn, pa) in beams.items():
        beam_arr[t, f, :, :] = [maj, mn, pa]

    coords = {
        "time": np.arange(nt, dtype=np.float64),
        "frequency": 1.0e9 + 1.0e6 * np.arange(nf),
        "polarization": ["I", "Q", "U", "V"][:npol],
        "l": (np.arange(ny) - ny // 2) * (-delta),  # negative increment (RA-like)
        "m": (np.arange(nx) - nx // 2) * delta,
        "beam_params_label": ["major", "minor", "pa"],
    }
    dims_lm = ["time", "frequency", "polarization", "l", "m"]
    data_vars = {
        "SKY_RESIDUAL": (dims_lm, residual),
        "SKY_MODEL": (dims_lm, model),
        "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": (
            ["time", "frequency", "polarization", "beam_params"],
            beam_arr,
        ),
    }
    xds = xr.Dataset(data_vars, coords=coords)
    xds.attrs["data_groups"] = {
        "residual": {
            "sky": "SKY_RESIDUAL",
            "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        },
        "model": {"sky": "SKY_MODEL"},
    }
    return xds


# ---------------------------------------------------------------------------
# _elliptical_gaussian_kernel
# ---------------------------------------------------------------------------


class TestEllipticalGaussianKernel:
    def test_unit_peak_at_centre(self):
        k = _elliptical_gaussian_kernel(64, 64, 8.0, 4.0, 0.0, np.float64)
        assert k[32, 32] == pytest.approx(1.0)
        assert k.max() == pytest.approx(1.0)
        assert np.unravel_index(np.argmax(k), k.shape) == (32, 32)

    def test_fwhm_along_axes_pa_zero(self):
        # The fit measures pa from m (axis 1) towards l (axis 0), so pa=0 puts
        # the major axis along m (axis 1) and the minor along l (axis 0).
        major_pix, minor_pix = 8.0, 4.0
        k = _elliptical_gaussian_kernel(80, 80, major_pix, minor_pix, 0.0, np.float64)
        c = 40
        # Half-power point is at +FWHM/2 from the centre along each axis.
        assert k[c, c + 4] == pytest.approx(0.5, abs=1e-6)  # along m (major, 8 px)
        assert k[c + 2, c] == pytest.approx(0.5, abs=1e-6)  # along l (minor, 4 px)

    def test_orientation_pa_rotates_major_axis(self):
        # pa = pi/2 -> major axis rotates onto l (axis 0).
        k = _elliptical_gaussian_kernel(80, 80, 8.0, 4.0, np.pi / 2, np.float64)
        c = 40
        assert k[c + 4, c] == pytest.approx(0.5, abs=1e-6)  # major now along l
        assert k[c, c + 2] == pytest.approx(0.5, abs=1e-6)  # minor now along m

    def test_dtype(self):
        k = _elliptical_gaussian_kernel(16, 16, 4.0, 4.0, 0.0, np.float32)
        assert k.dtype == np.float32


# ---------------------------------------------------------------------------
# restore_image
# ---------------------------------------------------------------------------


class TestRestoreImage:
    def test_point_source_restored_to_unit_peak_beam(self):
        # Unit point source at centre, zero residual -> restored == the
        # centred unit-peak clean beam.
        delta = 1.0e-3
        xds = _make_restore_xds(beams={(0, 0): (8.0 * delta, 4.0 * delta, 0.0)})
        out, _ = restore_image(xds)
        restored = out["SKY_RESTORED"].values
        expected = _elliptical_gaussian_kernel(64, 64, 8.0, 4.0, 0.0, np.float32)
        for pp in range(restored.shape[2]):
            np.testing.assert_allclose(restored[0, 0, pp], expected, atol=1e-5)

    def test_additivity_restored_is_convolved_model_plus_residual(self):
        delta = 1.0e-3
        rng = np.random.default_rng(0)
        residual = rng.standard_normal((1, 1, 2, 64, 64)).astype(np.float32) * 0.01
        model = np.zeros((1, 1, 2, 64, 64), dtype=np.float32)
        model[0, 0, :, 32, 32] = 3.0  # flux-3 point source
        xds = _make_restore_xds(
            beams={(0, 0): (8.0 * delta, 4.0 * delta, 0.0)},
            residual=residual.copy(),
            model=model.copy(),
        )
        out, _ = restore_image(xds)
        restored = out["SKY_RESTORED"].values
        beam = _elliptical_gaussian_kernel(64, 64, 8.0, 4.0, 0.0, np.float32)
        for pp in range(2):
            expected = 3.0 * beam + residual[0, 0, pp]
            np.testing.assert_allclose(restored[0, 0, pp], expected, atol=1e-5)
            # Peak of the convolved model is the source flux (unit-peak beam).
            assert (restored[0, 0, pp] - residual[0, 0, pp]).max() == pytest.approx(
                3.0, abs=1e-5
            )

    def test_round_trips_through_psf_gaussian_fit(self):
        # Build a clean beam, restore a unit point source (zero residual) so the
        # restored image *is* that beam, then fit it back and check the recovered
        # [major, minor, pa] match -- this validates the construction convention
        # against point_spread_function_gaussian_fit.
        from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
            point_spread_function_gaussian_fit,
        )

        delta = 1.0e-3
        major_pix, minor_pix, pa = 9.0, 5.0, 0.6
        xds = _make_restore_xds(
            npol=1,
            ny=128,
            nx=128,
            beams={(0, 0): (major_pix * delta, minor_pix * delta, pa)},
            dtype=np.float64,
            delta=delta,
        )
        out, _ = restore_image(xds)
        beam_img = out["SKY_RESTORED"].values  # the centred clean beam

        # Feed the restored beam back in as a PSF and fit it.
        psf_xds = xr.Dataset(
            {
                "POINT_SPREAD_FUNCTION": (
                    ["time", "frequency", "polarization", "l", "m"],
                    beam_img,
                )
            },
            coords={
                k: out.coords[k]
                for k in ("time", "frequency", "polarization", "l", "m")
            },
        )
        psf_xds.attrs["data_groups"] = {
            "image": {"point_spread_function": "POINT_SPREAD_FUNCTION"}
        }
        fitted = point_spread_function_gaussian_fit(
            psf_xds,
            image_data_group_in_name="image",
            image_data_group_out_name="image",
        )
        params = fitted["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].values[0, 0, 0]
        rec_major, rec_minor, rec_pa = params
        assert rec_major == pytest.approx(major_pix * delta, rel=0.05)
        assert rec_minor == pytest.approx(minor_pix * delta, rel=0.05)
        assert rec_pa == pytest.approx(pa, abs=0.05)

    def test_returns_timing_frame(self):
        # Self-times like the other imaging processing functions: a one-row
        # frame with a non-negative T_restore (seconds).
        xds = _make_restore_xds()
        out, return_df = restore_image(xds)
        assert list(return_df.columns) == ["T_restore"]
        assert len(return_df) == 1
        assert float(return_df["T_restore"].iloc[0]) >= 0.0

    def test_registers_restored_data_group_and_preserves_dtype(self):
        xds = _make_restore_xds(dtype=np.float32)
        out, _ = restore_image(xds)
        assert "restored" in out.attrs["data_groups"]
        assert out.attrs["data_groups"]["restored"]["sky"] == "SKY_RESTORED"
        # Inherited the residual group's beam-fit role.
        assert (
            out.attrs["data_groups"]["restored"][
                "beam_fit_params_point_spread_function"
            ]
            == "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
        )
        assert out["SKY_RESTORED"].dtype == np.float32

    def test_does_not_mutate_inputs(self):
        xds = _make_restore_xds()
        residual0 = xds["SKY_RESIDUAL"].values.copy()
        model0 = xds["SKY_MODEL"].values.copy()
        restore_image(xds)
        np.testing.assert_array_equal(xds["SKY_RESIDUAL"].values, residual0)
        np.testing.assert_array_equal(xds["SKY_MODEL"].values, model0)

    def test_consume_model_matches_default_and_drops_model(self):
        # consume_model=True must give a bit-identical restored image while
        # reusing the model cube's buffer and removing the model variable.
        delta = 1.0e-3
        rng = np.random.default_rng(3)
        residual = rng.standard_normal((1, 1, 2, 64, 64)).astype(np.float32) * 0.01
        model = np.zeros((1, 1, 2, 64, 64), dtype=np.float32)
        model[0, 0, :, 32, 32] = 3.0
        beams = {(0, 0): (8.0 * delta, 4.0 * delta, 0.3)}

        xds_default = _make_restore_xds(
            beams=beams, residual=residual.copy(), model=model.copy()
        )
        out_default, _ = restore_image(xds_default)

        xds_consume = _make_restore_xds(
            beams=beams, residual=residual.copy(), model=model.copy()
        )
        model_buffer = xds_consume["SKY_MODEL"].values
        out_consume, _ = restore_image(xds_consume, consume_model=True)

        np.testing.assert_array_equal(
            out_consume["SKY_RESTORED"].values, out_default["SKY_RESTORED"].values
        )
        assert np.shares_memory(out_consume["SKY_RESTORED"].values, model_buffer)
        assert "SKY_MODEL" not in out_consume
        np.testing.assert_array_equal(out_consume["SKY_RESIDUAL"].values, residual)

    def test_consume_model_dtype_mismatch_falls_back(self):
        # Model/residual dtype mismatch: the buffer cannot be reused, so the
        # restore must fall back to allocating and preserve the model.
        delta = 1.0e-3
        rng = np.random.default_rng(4)
        residual = rng.standard_normal((1, 1, 1, 32, 32)).astype(np.float32) * 0.01
        model = np.zeros((1, 1, 1, 32, 32), dtype=np.float64)
        model[0, 0, :, 16, 16] = 2.0
        beams = {(0, 0): (6.0 * delta, 3.0 * delta, 0.0)}

        xds_default = _make_restore_xds(
            npol=1,
            ny=32,
            nx=32,
            beams=beams,
            residual=residual.copy(),
            model=model.copy(),
        )
        out_default, _ = restore_image(xds_default)

        xds_consume = _make_restore_xds(
            npol=1,
            ny=32,
            nx=32,
            beams=beams,
            residual=residual.copy(),
            model=model.copy(),
        )
        out_consume, _ = restore_image(xds_consume, consume_model=True)

        np.testing.assert_array_equal(
            out_consume["SKY_RESTORED"].values, out_default["SKY_RESTORED"].values
        )
        assert "SKY_MODEL" in out_consume
        np.testing.assert_array_equal(out_consume["SKY_MODEL"].values, model)

    def test_nan_beam_falls_back_to_residual(self):
        rng = np.random.default_rng(1)
        residual = rng.standard_normal((1, 1, 2, 32, 32)).astype(np.float32)
        model = np.zeros((1, 1, 2, 32, 32), dtype=np.float32)
        model[0, 0, :, 16, 16] = 5.0
        # Beam left as NaN (no entry in beams -> NaN row).
        xds = _make_restore_xds(
            ny=32, nx=32, beams={}, residual=residual.copy(), model=model.copy()
        )
        out, _ = restore_image(xds)
        # Undefined beam -> restored == residual (model not restored).
        np.testing.assert_array_equal(out["SKY_RESTORED"].values, residual)

    def test_empty_model_plane_gives_residual(self):
        delta = 1.0e-3
        rng = np.random.default_rng(2)
        residual = rng.standard_normal((1, 1, 1, 32, 32)).astype(np.float32)
        model = np.zeros((1, 1, 1, 32, 32), dtype=np.float32)  # nothing cleaned
        xds = _make_restore_xds(
            npol=1,
            ny=32,
            nx=32,
            beams={(0, 0): (6.0 * delta, 6.0 * delta, 0.0)},
            residual=residual.copy(),
            model=model.copy(),
        )
        out, _ = restore_image(xds)
        np.testing.assert_array_equal(out["SKY_RESTORED"].values, residual)


class TestEllipticalGaussianUvTaper:
    """The analytic uv taper is the Fourier transform of the image-plane kernel."""

    def test_matches_fft_of_kernel(self):
        from astroviper.processing_functions.imaging.restore import (
            elliptical_gaussian_uv_taper,
        )

        ny = nx = 128
        major, minor, pa = 9.0, 4.0, 0.6  # FWHM in pixels, pa in radians
        kernel = _elliptical_gaussian_kernel(ny, nx, major, minor, pa, np.float64)
        kernel_ft = np.fft.fftshift(np.abs(np.fft.fft2(np.fft.ifftshift(kernel))))
        kernel_ft /= kernel_ft.max()
        # Pixel-index frequencies; the sky l axis runs opposite to the l pixel
        # index, so u = -f_l while v = +f_m.
        u = -np.fft.fftshift(np.fft.fftfreq(ny))[:, None]
        v = np.fft.fftshift(np.fft.fftfreq(nx))[None, :]
        taper = elliptical_gaussian_uv_taper(u, v, major, minor, pa)
        np.testing.assert_allclose(kernel_ft, taper, atol=1e-8)

    def test_unit_at_origin_and_even(self):
        from astroviper.processing_functions.imaging.restore import (
            elliptical_gaussian_uv_taper,
        )

        assert elliptical_gaussian_uv_taper(0.0, 0.0, 1e-5, 5e-6, 0.3) == 1.0
        u = np.array([100.0, -3000.0])
        v = np.array([-200.0, 1500.0])
        np.testing.assert_allclose(
            elliptical_gaussian_uv_taper(u, v, 1e-5, 5e-6, 0.3),
            elliptical_gaussian_uv_taper(-u, -v, 1e-5, 5e-6, 0.3),
        )


# ---------------------------------------------------------------------------
# primary beam correction in the restore pass
# ---------------------------------------------------------------------------


def _with_primary_beam(xds, primary_beam_limit=0.2):
    """Add a primary beam that falls off across the image to the residual group.

    The beam power falls linearly from 1 at the centre to 0 at the corner, so
    it crosses ``primary_beam_limit`` inside the image and varies across the
    clean beam.
    """
    ny, nx = xds["SKY_RESIDUAL"].shape[-2:]
    radius = np.hypot(
        *np.meshgrid(np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing="ij")
    )
    power = np.clip(1.0 - radius / (0.75 * max(ny, nx)), 0.0, None)
    shape = xds["SKY_RESIDUAL"].shape
    xds["PRIMARY_BEAM"] = xr.DataArray(
        np.broadcast_to(power, shape).astype(xds["SKY_RESIDUAL"].dtype).copy(),
        dims=xds["SKY_RESIDUAL"].dims,
    )
    xds.attrs["data_groups"]["residual"]["primary_beam"] = "PRIMARY_BEAM"
    return xds


class TestPrimaryBeamCorrectionInRestore:
    delta = 1.0e-3
    beams = {(0, 0): (8.0 * delta, 4.0 * delta, 0.3)}

    def _inputs(self, dtype=np.float32, seed=5):
        rng = np.random.default_rng(seed)
        residual = rng.standard_normal((1, 1, 2, 64, 64)).astype(dtype) * 0.01
        model = np.zeros((1, 1, 2, 64, 64), dtype=dtype)
        model[0, 0, :, 20, 44] = (
            3.0  # off centre, where the primary beam has a gradient
        )
        return residual, model

    def test_exact_for_the_model_part(self):
        # A point source of flux F: the corrected image is F x B(x - x0)
        # plus R / P, whatever the primary beam does across the clean beam.
        residual, model = self._inputs()
        xds = _with_primary_beam(
            _make_restore_xds(beams=self.beams, residual=residual, model=model)
        )
        primary_beam = xds["PRIMARY_BEAM"].values
        out, return_df = restore_image(xds, restoration_output="both")
        corrected = out["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].values
        inside = primary_beam >= 0.2
        # the true sky convolved with the beam: the restored image of the
        # model divided by the primary beam at the source, with no residual
        true_flux = model / np.where(inside, primary_beam, np.nan)
        reference = restore_image(
            _make_restore_xds(
                beams=self.beams,
                residual=np.zeros_like(residual),
                model=np.nan_to_num(true_flux).astype(model.dtype),
            )
        )[0]["SKY_RESTORED"].values
        expected = reference + residual / primary_beam
        np.testing.assert_allclose(
            corrected[inside], expected[inside], rtol=1e-5, atol=1e-6
        )
        assert np.isnan(corrected[~inside]).all()
        assert out.attrs["data_groups"]["restored"]["sky_primary_beam_corrected"] == (
            "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
        )
        assert list(return_df.columns) == ["T_restore", "T_correct_sky_by_primary_beam"]
        # the restored image itself is unchanged by the correction
        plain = restore_image(
            _make_restore_xds(beams=self.beams, residual=residual, model=model)
        )[0]["SKY_RESTORED"].values
        np.testing.assert_array_equal(out["SKY_RESTORED"].values, plain)

    def test_differs_from_dividing_the_restored_image_beside_a_source(self):
        # Dividing the restored image (the CASA convention) carries the
        # gradient of the primary beam across the clean beam; the two agree
        # at the source pixel only.
        residual, model = self._inputs()
        residual[...] = 0.0
        xds = _with_primary_beam(
            _make_restore_xds(beams=self.beams, residual=residual, model=model)
        )
        primary_beam = xds["PRIMARY_BEAM"].values
        out, _ = restore_image(xds, restoration_output="both")
        corrected = out["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].values
        divided = out["SKY_RESTORED"].values / primary_beam
        np.testing.assert_allclose(
            corrected[0, 0, 0, 20, 44], divided[0, 0, 0, 20, 44], rtol=1e-6
        )
        assert not np.allclose(
            corrected[0, 0, 0, 16:25, 40:49], divided[0, 0, 0, 16:25, 40:49], rtol=1e-3
        )

    def test_consume_model_gives_the_same_corrected_image(self):
        residual, model = self._inputs()
        xds = _with_primary_beam(
            _make_restore_xds(
                beams=self.beams, residual=residual.copy(), model=model.copy()
            )
        )
        reference = restore_image(xds, restoration_output="both")[0]
        xds = _with_primary_beam(
            _make_restore_xds(
                beams=self.beams, residual=residual.copy(), model=model.copy()
            )
        )
        consumed = restore_image(xds, consume_model=True, restoration_output="both")[0]
        np.testing.assert_array_equal(
            consumed["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].values,
            reference["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].values,
        )
        assert "SKY_MODEL" not in consumed

    def test_corrected_only_core_matches_cube_primary_beam_output(self):
        """Selecting only the corrected core output preserves cube numerics."""
        residual, model = self._inputs()
        xds = _with_primary_beam(
            _make_restore_xds(beams=self.beams, residual=residual, model=model)
        )
        reference = restore_image(xds, restoration_output="both")[0]

        products, _ = _restore_image_arrays(
            residual,
            model,
            xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
            self.delta,
            primary_beam=xds.PRIMARY_BEAM.values,
            primary_beam_limit=0.2,
            output="primary_beam_corrected",
        )

        assert set(products) == {"primary_beam_corrected"}
        np.testing.assert_array_equal(
            products["primary_beam_corrected"],
            reference.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.values,
        )

    def test_zero_limit_excludes_zero_and_nonfinite_primary_beam_pixels(self):
        """A zero cutoff cannot feed zero-beam NaNs through the model FFT."""
        residual, model = self._inputs()
        xds = _with_primary_beam(
            _make_restore_xds(beams=self.beams, residual=residual, model=model)
        )
        primary_beam = xds.PRIMARY_BEAM.values
        primary_beam[..., 0, 0] = np.nan

        out, _ = restore_image(
            xds,
            restoration_output="both",
            primary_beam_limit=0.0,
        )
        corrected = out.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.values
        supported = np.isfinite(primary_beam) & (primary_beam > 0.0)

        assert np.isfinite(corrected[supported]).all()
        assert np.isnan(corrected[~supported]).all()

    def test_limit_dtype_and_missing_beam(self):
        residual, model = self._inputs(dtype=np.float64)
        xds = _with_primary_beam(
            _make_restore_xds(
                beams=self.beams, residual=residual, model=model, dtype=np.float64
            )
        )
        out, _ = restore_image(xds, restoration_output="both", primary_beam_limit=0.6)
        corrected = out["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"]
        assert corrected.dtype == np.float64
        assert np.isnan(corrected.values[xds["PRIMARY_BEAM"].values < 0.6]).all()
        # no clean beam: the model is not restored and not convolved
        residual, model = self._inputs()
        xds = _with_primary_beam(
            _make_restore_xds(
                beams={(0, 0): (np.nan, np.nan, 0.0)}, residual=residual, model=model
            )
        )
        out, _ = restore_image(xds, restoration_output="both")
        primary_beam = xds["PRIMARY_BEAM"].values
        inside = primary_beam >= 0.2
        expected = (model + residual) / primary_beam
        np.testing.assert_allclose(
            out["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"].values[inside],
            expected[inside],
            rtol=1e-6,
        )

    def test_needs_the_primary_beam(self):
        residual, model = self._inputs()
        xds = _make_restore_xds(beams=self.beams, residual=residual, model=model)
        with pytest.raises(KeyError, match="primary_beam"):
            restore_image(xds, restoration_output="both")


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),
        (63, 50),
        (50, 63),
        (500, 480),
        (127, 131),  # primes: Bluestein plans on both axes
        (1009, 1013),
        (6000, 1),
    ],
)
def test_inverse_rfft2_is_irfft2_bit_for_bit(shape, dtype, workers):
    """restore_image's two-step inverse real FFT (no hidden copy of the half
    spectrum) gives exactly scipy.fft.irfft2's result, for even, odd and prime
    sizes, both image dtypes and more than one worker."""
    import scipy.fft

    rng = np.random.default_rng(sum(shape))
    plane = rng.standard_normal(shape).astype(dtype)
    beam = np.exp(-rng.uniform(0.0, 3.0, size=(shape[0], shape[1] // 2 + 1)))
    spectrum = scipy.fft.rfft2(plane)
    spectrum *= beam.astype(dtype)
    expected = scipy.fft.irfft2(spectrum, s=shape, workers=workers)
    result = _inverse_rfft2(spectrum.copy(), shape, workers)
    assert result.dtype == expected.dtype == dtype
    assert result.shape == expected.shape == shape
    assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize("output", ["apparent", "primary_beam_corrected", "both"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("consume_model", [False, True])
@pytest.mark.parametrize("restoring_beam", [None, [10.0e-3, 6.0e-3, 0.0]])
def test_cube_restoration_selector_matches_both_products(
    output, dtype, consume_model, restoring_beam
):
    image = _make_restore_xds(ny=16, nx=16)
    image["SKY_MODEL"][:] = 0
    image["SKY_MODEL"].values[..., 8, 8] = 2
    image["SKY_RESIDUAL"][:] = 0.1
    image = image.astype(dtype)
    image["PRIMARY_BEAM"] = xr.full_like(image.SKY_RESIDUAL, 0.5)
    image.PRIMARY_BEAM.values[..., 0, :] = 0
    image.attrs["data_groups"]["residual"]["primary_beam"] = "PRIMARY_BEAM"
    reference, _ = restore_image(
        image.copy(deep=True), restoration_output="both", restoring_beam=restoring_beam
    )
    model = image.SKY_MODEL.copy(deep=True)
    result, _ = restore_image(
        image,
        restoration_output=output,
        consume_model=consume_model,
        restoring_beam=restoring_beam,
    )
    apparent = output in ("apparent", "both")
    corrected = output in ("primary_beam_corrected", "both")
    assert ("SKY_RESTORED" in result) == apparent
    assert ("SKY_RESTORED_PRIMARY_BEAM_CORRECTED" in result) == corrected
    for name in ("SKY_RESTORED", "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"):
        if name in result:
            if restoring_beam is not None and output == "apparent":
                # One inverse transform of both terms; "both" has one per term.
                atol = 10 * np.finfo(dtype).eps * float(np.abs(reference[name]).max())
                np.testing.assert_allclose(
                    result[name], reference[name], rtol=0, atol=atol
                )
            else:
                np.testing.assert_array_equal(result[name], reference[name])
            assert result[name].dtype == dtype
    assert result.attrs["data_groups"]["restored"]["sky"] == (
        "SKY_RESTORED" if apparent else "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"
    )
    if not apparent or not consume_model:
        xr.testing.assert_identical(result.SKY_MODEL, model)


@pytest.mark.parametrize("output", [True, False, "unknown", None])
def test_cube_restoration_rejects_invalid_selector_before_product_access(output):
    with pytest.raises(ValueError, match="restoration_output"):
        restore_image(None, restoration_output=output)


# ---------------------------------------------------------------------------
# restoring_beam: every plane restored to one target beam
# ---------------------------------------------------------------------------

DELTA = 1.0e-3
PSF_PIXELS = (8.0, 4.0, 0.3)
TARGET_PIXELS = (10.0, 6.0, 0.7)
TARGET = [TARGET_PIXELS[0] * DELTA, TARGET_PIXELS[1] * DELTA, TARGET_PIXELS[2]]
OUTPUTS = ["apparent", "primary_beam_corrected", "both"]


def _radians(beam_pixels):
    return (beam_pixels[0] * DELTA, beam_pixels[1] * DELTA, beam_pixels[2])


def _target_inputs(dtype, nf=1, beams=None, seed=11, shape=(64, 64)):
    """Two polarizations of noise, off-centre point sources and a primary beam
    that crosses its cutoff; every plane has the PSF beam ``PSF_PIXELS``."""
    rng = np.random.default_rng(seed)
    residual = (0.01 * rng.standard_normal((1, nf, 2, *shape))).astype(dtype)
    model = np.zeros((1, nf, 2, *shape), dtype=dtype)
    model[0, :, :, 20, 44] = 3.0
    model[0, :, 1, 40, 25] = -1.5
    if beams is None:
        beams = {(0, f): _radians(PSF_PIXELS) for f in range(nf)}
    return _with_primary_beam(
        _make_restore_xds(
            nf=nf,
            ny=shape[0],
            nx=shape[1],
            beams=beams,
            residual=residual,
            model=model,
            dtype=dtype,
        )
    )


def _pixel_size(xds):
    """The pixel size ``restore_image`` takes from the ``l`` coordinate."""
    return abs(float(xds.l.values[1] - xds.l.values[0]))


def _covariance_pixels(major, minor, pa):
    """Covariance (pixels^2) of a beam in pixels; the major axis lies along
    ``(sin pa, -cos pa)`` in ``(l index, m index)``."""
    sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    e = np.array([np.sin(pa), -np.cos(pa)])
    f = np.array([np.cos(pa), np.sin(pa)])
    return (major * sigma) ** 2 * np.outer(e, e) + (minor * sigma) ** 2 * np.outer(f, f)


def _correcting_transform(ny, nx, psf, target):
    """H on the ``rfft2`` grid in float64, from beams in pixels."""
    d = _covariance_pixels(*target) - _covariance_pixels(*psf)
    f0 = np.fft.fftfreq(ny)[:, None]
    f1 = np.fft.rfftfreq(nx)[None, :]
    q = d[0, 0] * f0**2 + 2.0 * d[0, 1] * f0 * f1 + d[1, 1] * f1**2
    ratio = (target[0] * target[1]) / (psf[0] * psf[1])
    return ratio * np.exp(-2.0 * np.pi**2 * q)


def _divided(plane, primary_beam_plane, outside, dtype=np.float64):
    """``plane / P`` where the primary beam is not below the 0.2 cutoff,
    ``outside`` elsewhere, at ``dtype``."""
    return np.divide(
        plane,
        primary_beam_plane,
        out=np.full(plane.shape, outside, dtype=dtype),
        where=primary_beam_plane >= 0.2,
    )


def _restored(xds, output, consume_model, restoring_beam):
    out, _ = restore_image(
        xds,
        restoration_output=output,
        consume_model=consume_model,
        restoring_beam=restoring_beam,
    )
    return {
        name: out[name].values
        for name in ("SKY_RESTORED", "SKY_RESTORED_PRIMARY_BEAM_CORRECTED")
        if name in out
    }


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("consume_model", [False, True])
def test_none_restores_each_plane_with_its_own_beam(output, dtype, consume_model):
    """``restoring_beam=None`` is today's per-plane restore, bit for bit."""
    beams = {(0, 0): _radians(PSF_PIXELS), (0, 1): (6.0e-3, 5.0e-3, 1.2)}
    xds = _target_inputs(dtype, nf=2, beams=beams)
    residual = xds.SKY_RESIDUAL.values.copy()
    model = xds.SKY_MODEL.values.copy()
    primary_beam = xds.PRIMARY_BEAM.values.copy()
    pixel_size = _pixel_size(xds)
    restored = _restored(xds, output, consume_model, None)
    for f, q in np.ndindex(2, 2):
        kernel_ft = _clean_beam_kernel_ft(beams[(0, f)], 64, 64, pixel_size, dtype, 1)
        if output != "primary_beam_corrected":
            expected = _convolve_with_clean_beam(model[0, f, q], kernel_ft, 1)
            expected += residual[0, f, q]
            np.testing.assert_array_equal(restored["SKY_RESTORED"][0, f, q], expected)
        if output != "apparent":
            expected = primary_beam_corrected_plane(
                primary_beam[0, f, q],
                0.2,
                model[0, f, q],
                residual[0, f, q],
                kernel_ft=kernel_ft,
            )
            np.testing.assert_array_equal(
                restored["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"][0, f, q], expected
            )


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("consume_model", [False, True])
def test_target_equal_to_the_psf_beam_gives_the_none_result(
    output, dtype, consume_model
):
    reference = _restored(_target_inputs(dtype, nf=2), output, consume_model, None)
    xds = _target_inputs(dtype, nf=2)
    residual = xds.SKY_RESIDUAL.copy(deep=True)
    restored = _restored(xds, output, consume_model, list(_radians(PSF_PIXELS)))
    assert restored.keys() == reference.keys()
    for name, value in restored.items():
        np.testing.assert_array_equal(value, reference[name])
    xr.testing.assert_identical(xds.SKY_RESIDUAL, residual)


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("consume_model", [False, True])
@pytest.mark.parametrize("nan_pixel", [False, True])
@pytest.mark.parametrize("shape", [(64, 64), (64, 51)])
def test_target_restore_matches_the_float64_construction(
    output, dtype, consume_model, nan_pixel, shape
):
    """The model convolved with the sampled target beam plus the residual
    multiplied by H. The corrected plane is ``(M / P) * B_T + (R_0 / P) * K``:
    the residual, like the model, is divided by the primary beam before it is
    convolved, with ``R_0`` zero outside the cutoff and at a NaN residual
    pixel, where the planes are NaN. CASA's order for the residual,
    ``(R_0 * K) / P``, is off by more than 100 times the tolerance."""
    xds = _target_inputs(dtype, shape=shape)
    if nan_pixel:
        xds.SKY_RESIDUAL.values[0, 0, 0, 10, 50] = np.nan
    residual = xds.SKY_RESIDUAL.values.astype(np.float64)
    model = xds.SKY_MODEL.values.astype(np.float64)
    primary_beam = xds.PRIMARY_BEAM.values.astype(np.float64)
    restored = _restored(xds, output, consume_model, TARGET)
    kernel_ft = _clean_beam_kernel_ft(TARGET, *shape, _pixel_size(xds), np.float64, 1)
    transform = _correcting_transform(*shape, PSF_PIXELS, TARGET_PIXELS)
    tolerance = 1e-6 if dtype == np.float32 else 1e-12

    def convolved(plane, spectrum):
        return np.fft.irfft2(np.fft.rfft2(plane) * spectrum, s=shape)

    for q in range(2):
        finite = np.isfinite(residual[0, 0, q])
        residual_zero = np.where(finite, residual[0, 0, q], 0.0)
        primary_beam_plane = primary_beam[0, 0, q]
        model_part = convolved(
            _divided(model[0, 0, q], primary_beam_plane, 0.0), kernel_ft
        )
        expected = {
            "SKY_RESTORED": np.where(
                finite,
                convolved(model[0, 0, q], kernel_ft)
                + convolved(residual_zero, transform),
                np.nan,
            ),
            "SKY_RESTORED_PRIMARY_BEAM_CORRECTED": np.where(
                (primary_beam_plane >= 0.2) & finite,
                model_part
                + convolved(
                    _divided(residual_zero, primary_beam_plane, 0.0), transform
                ),
                np.nan,
            ),
        }
        for name, value in restored.items():
            assert value.dtype == dtype
            np.testing.assert_array_equal(
                np.isnan(value[0, 0, q]), np.isnan(expected[name])
            )
            error = np.nanmax(np.abs(value[0, 0, q] - expected[name]))
            assert error <= tolerance * np.nanmax(np.abs(expected[name]))
        corrected = expected["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"]
        casa_order = model_part + _divided(
            convolved(residual_zero, transform), primary_beam_plane, 0.0
        )
        assert np.nanmax(np.abs(casa_order - corrected)) > 100 * tolerance * np.nanmax(
            np.abs(corrected)
        )


def test_correcting_transform_takes_the_psf_beam_to_the_target():
    """A sky made with the sampled PSF beam, times H, is the sky made with the
    sampled target beam; the opposite sign of the cross term is not. 288 rows
    span two row blocks, and the odd number of columns needs ``nx``."""
    ny, nx = 288, 95
    psf, target = (10.0, 6.0, 0.4), (14.0, 9.0, 0.9)
    sky = np.zeros((ny, nx))
    sky[100, 40], sky[200, 61], sky[31, 18] = 1.0, -0.5, 2.0

    def made_with(beam):
        kernel = _elliptical_gaussian_kernel(ny, nx, *beam, np.float64)
        return np.fft.irfft2(
            np.fft.rfft2(sky) * np.fft.rfft2(np.fft.ifftshift(kernel)), s=(ny, nx)
        )

    residual, expected = made_with(psf), made_with(target)
    got = _restore_image_arrays(
        residual[None, None, None],
        np.zeros((1, 1, 1, ny, nx)),
        np.array(psf)[None, None, None],
        1.0,
        restoring_beam=target,
    )[0]["apparent"][0, 0, 0]
    assert np.abs(got - expected).max() <= 1e-12 * np.abs(expected).max()

    flipped = (_covariance_pixels(*target) - _covariance_pixels(*psf)) * [
        [1.0, -1.0],
        [-1.0, 1.0],
    ]
    spectrum = np.fft.rfft2(residual)
    _multiply_by_correcting_transform(
        spectrum, flipped, (target[0] * target[1]) / (psf[0] * psf[1]), nx
    )
    wrong = np.fft.irfft2(spectrum, s=(ny, nx))
    assert np.abs(wrong - expected).max() > 1e-2 * np.abs(expected).max()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_correcting_transform_makes_no_subnormal_values(dtype):
    """A correcting Gaussian of 40 x 30 px takes H down to exp(-1e4) at the
    highest frequencies, below the smallest normal number of either dtype."""
    covariance = _covariance_pixels(40.0, 30.0, 0.3)
    spectrum = np.ones((256, 129), dtype=dtype)
    _multiply_by_correcting_transform(spectrum, covariance, 1.5, 256)
    magnitude = np.abs(spectrum)
    tiny = np.finfo(dtype).tiny
    assert not ((magnitude > 0) & (magnitude < tiny / np.finfo(dtype).eps ** 2)).any()
    assert magnitude[0, 0] == np.float32(1.5) and (magnitude == 0).any()


@pytest.mark.parametrize("output", ["apparent", "both"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_point_source_in_the_residual_keeps_its_peak(output, dtype):
    """CASA Memo 10, Fig. 2: with the target axes doubled a point source left
    in the residual keeps its peak (Jy per target beam), not 0.25. The
    primary beam is 1 across the source, which the corrected plane divides
    before the convolution."""
    residual = _elliptical_gaussian_kernel(64, 64, *PSF_PIXELS, dtype)
    xds = _with_primary_beam(
        _make_restore_xds(
            npol=1,
            beams={(0, 0): _radians(PSF_PIXELS)},
            residual=residual[None, None, None].copy(),
            model=np.zeros((1, 1, 1, 64, 64), dtype=dtype),
            dtype=dtype,
        )
    )
    xds.PRIMARY_BEAM.values[...] = 1.0
    doubled = [2 * PSF_PIXELS[0] * DELTA, 2 * PSF_PIXELS[1] * DELTA, PSF_PIXELS[2]]
    restored = _restored(xds, output, False, doubled)
    apparent = restored["SKY_RESTORED"]
    assert np.unravel_index(np.argmax(apparent), apparent.shape) == (0, 0, 0, 32, 32)
    for value in restored.values():
        assert value[0, 0, 0, 32, 32] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("consume_model", [False, True])
def test_a_nan_residual_pixel_is_the_only_nonfinite_pixel(output, dtype, consume_model):
    xds = _target_inputs(dtype)
    xds.SKY_RESIDUAL.values[0, 0, 0, 10, 50] = np.nan
    inside = xds.PRIMARY_BEAM.values >= 0.2
    restored = _restored(xds, output, consume_model, TARGET)
    expected = np.zeros(inside.shape, dtype=bool)
    expected[0, 0, 0, 10, 50] = True
    for name, value in restored.items():
        if name == "SKY_RESTORED":
            np.testing.assert_array_equal(~np.isfinite(value), expected)
        else:
            np.testing.assert_array_equal(~np.isfinite(value) & inside, expected)
            assert np.isnan(value[~inside]).all()


@pytest.mark.parametrize("output", ["primary_beam_corrected", "both"])
def test_an_overflowing_quotient_is_the_only_nonfinite_pixel(output):
    """At ``primary_beam_limit=0`` a residual of 10 over a primary beam of
    1e-38 overflows ``R / P`` in single precision. That pixel is left out of
    the transform and NaN, as a non-finite residual pixel; the rest of the
    corrected plane stays finite."""
    xds = _target_inputs(np.float32)
    xds.PRIMARY_BEAM.values[0, 0, 0, 30, 30] = 1e-38
    xds.SKY_RESIDUAL.values[0, 0, 0, 30, 30] = 10.0
    expected = ~(xds.PRIMARY_BEAM.values > 0)
    expected[0, 0, 0, 30, 30] = True
    with np.errstate(over="ignore"):
        out, _ = restore_image(
            xds,
            restoration_output=output,
            restoring_beam=TARGET,
            primary_beam_limit=0.0,
        )
    corrected = out.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.values
    np.testing.assert_array_equal(np.isnan(corrected), expected)
    assert np.isfinite(corrected[~expected]).all()


@pytest.mark.parametrize("output", ["primary_beam_corrected", "both"])
def test_corrected_restore_to_a_target_keeps_one_spectrum_besides_the_beam(
    output, monkeypatch
):
    """Each term of the corrected plane, and of the apparent plane made with
    it, has its own inverse transform: as without a target, at most the clean
    beam's transform and one more spectrum are live. One inverse transform of
    both terms kept a third spectrum, which took one plane more physical
    memory than without a target on macOS."""
    import weakref

    import scipy.fft

    live, counts = [], []
    rfft2 = scipy.fft.rfft2

    def counted(*args, **kwargs):
        spectrum = rfft2(*args, **kwargs)
        live.append(weakref.ref(spectrum))
        counts.append(sum(ref() is not None for ref in live))
        return spectrum

    monkeypatch.setattr(scipy.fft, "rfft2", counted)
    for restoring_beam in (None, TARGET):
        live.clear()
        counts.clear()
        _restored(_target_inputs(np.float32), output, False, restoring_beam)
        assert max(counts) == 2


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("consume_model", [False, True])
def test_planes_not_contained_or_without_a_fit_keep_their_residual(
    output, dtype, consume_model, monkeypatch
):
    """Their model is restored with the target and their residual added as it
    is; one warning names them, with the growth factor."""
    import toolviper.utils.logger as logger

    messages = []
    monkeypatch.setattr(logger, "warning", messages.append)
    beams = {
        (0, 0): _radians(PSF_PIXELS),
        (0, 1): _radians((12.0, 4.0, TARGET_PIXELS[2])),
        (0, 2): (np.nan, np.nan, np.nan),
    }
    xds = _target_inputs(dtype, nf=3, beams=beams)
    residual = xds.SKY_RESIDUAL.values.copy()
    model = xds.SKY_MODEL.values.copy()
    primary_beam = xds.PRIMARY_BEAM.values.copy()
    restored = _restored(xds, output, consume_model, TARGET)
    kernel_ft = _clean_beam_kernel_ft(TARGET, 64, 64, _pixel_size(xds), dtype, 1)
    for f, q in np.ndindex(3, 2):
        if f == 0:
            continue
        if output != "primary_beam_corrected":
            expected = _convolve_with_clean_beam(model[0, f, q], kernel_ft, 1)
            expected += residual[0, f, q]
            np.testing.assert_array_equal(restored["SKY_RESTORED"][0, f, q], expected)
        if output != "apparent":
            expected = primary_beam_corrected_plane(
                primary_beam[0, f, q],
                0.2,
                model[0, f, q],
                residual[0, f, q],
                kernel_ft=kernel_ft,
            )
            np.testing.assert_array_equal(
                restored["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"][0, f, q], expected
            )
    frequency = xds.frequency.values.tolist()
    assert len(messages) == 1
    assert "(time index, frequency in Hz)" in messages[0]
    assert f"[((0, {frequency[1]}), 1.2" in messages[0]
    assert f"Without a PSF fit: [(0, {frequency[2]})]" in messages[0]


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_target_equal_within_the_tolerance_scales_the_residual(
    output, dtype, monkeypatch
):
    """D is zero after clipping: H is the area ratio, applied without a
    transform, to the residual and to the residual divided by the primary
    beam."""
    import astroviper.processing_functions.imaging.restore as restore

    def no_transform(*args):
        raise AssertionError("the correcting transform was evaluated")

    monkeypatch.setattr(restore, "_multiply_by_correcting_transform", no_transform)
    major, minor, pa = _radians(PSF_PIXELS)
    target = [major * (1 - 1e-9), minor * (1 - 1e-9), pa]
    xds = _target_inputs(dtype)
    residual = xds.SKY_RESIDUAL.values.copy()
    model = xds.SKY_MODEL.values.copy()
    primary_beam = xds.PRIMARY_BEAM.values.copy()
    restored = _restored(xds, output, False, target)
    kernel_ft = _clean_beam_kernel_ft(target, 64, 64, _pixel_size(xds), dtype, 1)
    ratio = dtype((target[0] * target[1]) / (major * minor))
    for q in range(2):
        if output != "apparent":
            primary_beam_plane = primary_beam[0, 0, q]
            expected = _convolve_with_clean_beam(
                _divided(model[0, 0, q], primary_beam_plane, 0, dtype), kernel_ft, 1
            )
            expected += (
                _divided(residual[0, 0, q], primary_beam_plane, np.nan, dtype) * ratio
            )
            np.testing.assert_array_equal(
                restored["SKY_RESTORED_PRIMARY_BEAM_CORRECTED"][0, 0, q], expected
            )
            assert np.isfinite(expected[primary_beam_plane >= 0.2]).all()
        if output != "primary_beam_corrected":
            expected = _convolve_with_clean_beam(model[0, 0, q], kernel_ft, 1)
            expected += residual[0, 0, q] * ratio
            np.testing.assert_array_equal(restored["SKY_RESTORED"][0, 0, q], expected)


def test_restored_beam_variable_holds_the_target_in_every_plane():
    beams = {
        (0, 0): _radians(PSF_PIXELS),
        (0, 1): (0.0, 0.0, 0.0),  # a flagged channel
        (0, 2): (np.nan, np.nan, np.nan),  # a failed fit
    }
    xds = _target_inputs(np.float32, nf=3, beams=beams)
    residual = xds.SKY_RESIDUAL.copy(deep=True)
    out, _ = restore_image(xds, restoration_output="both", restoring_beam=TARGET)
    restored_beam = out.BEAM_FIT_PARAMS_SKY_RESTORED
    psf_beam = out.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION
    assert restored_beam.dtype == np.float64
    # Labelled as xradio's BeamFitParamsArray, whose writers require it.
    assert restored_beam.dims == (*psf_beam.dims[:-1], "beam_params_label")
    assert restored_beam.beam_params_label.values.tolist() == ["major", "minor", "pa"]
    xr.testing.assert_identical(restored_beam.frequency, psf_beam.frequency)
    np.testing.assert_array_equal(
        restored_beam.values, np.broadcast_to(TARGET, psf_beam.shape)
    )
    assert restored_beam.attrs == {"type": "beam_fit_params_sky", "units": "rad"}
    restored_group = out.attrs["data_groups"]["restored"]
    assert restored_group["beam_fit_params_sky"] == "BEAM_FIT_PARAMS_SKY_RESTORED"
    assert "restoring beam" in restored_group["description"]
    xr.testing.assert_identical(out.SKY_RESIDUAL, residual)

    out, _ = restore_image(_target_inputs(np.float32), restoration_output="both")
    assert "BEAM_FIT_PARAMS_SKY_RESTORED" not in out
    assert "beam_fit_params_sky" not in out.attrs["data_groups"]["restored"]


def test_restores_to_two_targets_keep_the_beam_of_each_group():
    """A second restore into another group refuses to replace the beam the
    first group records unless it is given its own variable or overwrite."""
    xds = _target_inputs(np.float32)
    wide = [TARGET[0] * 1.5, TARGET[1] * 1.5, TARGET[2]]
    restore_image(
        xds,
        image_data_group_out_restore_name="restored_small",
        image_data_group_out_modified={"sky": "SKY_RESTORED_SMALL"},
        restoring_beam=TARGET,
        overwrite=False,
    )
    with pytest.raises(AssertionError, match="BEAM_FIT_PARAMS_SKY_RESTORED"):
        restore_image(
            xds,
            image_data_group_out_restore_name="restored_large",
            image_data_group_out_modified={"sky": "SKY_RESTORED_LARGE"},
            restoring_beam=wide,
            overwrite=False,
        )
    restore_image(
        xds,
        image_data_group_out_restore_name="restored_large",
        image_data_group_out_modified={
            "sky": "SKY_RESTORED_LARGE",
            "beam_fit_params_sky": "BEAM_FIT_PARAMS_SKY_RESTORED_LARGE",
        },
        restoring_beam=wide,
        overwrite=False,
    )
    groups = xds.attrs["data_groups"]
    for group, target in (("restored_small", TARGET), ("restored_large", wide)):
        beam = xds[groups[group]["beam_fit_params_sky"]].values
        np.testing.assert_array_equal(beam, np.broadcast_to(target, beam.shape))
    assert (
        groups["restored_large"]["beam_fit_params_sky"]
        == "BEAM_FIT_PARAMS_SKY_RESTORED_LARGE"
    )


def test_restoring_beam_is_checked_on_a_direct_call():
    with pytest.raises(ValueError, match="common_beam"):
        restore_image(_target_inputs(np.float32), restoring_beam="common")


def test_restoring_beam_in_arcseconds_is_refused_on_a_direct_call():
    """[1.2, 1.0, 0.5] meant as arcseconds, on 64 x 64 pixels of 1 mrad: the
    target would contain every PSF beam and pass without a warning."""
    with pytest.raises(ValueError, match=r"not smaller than the image \(0.064"):
        restore_image(_target_inputs(np.float32), restoring_beam=[1.2, 1.0, 0.5])
