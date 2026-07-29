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
    _elliptical_gaussian_kernel,
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
        "l": (np.arange(nx) - nx // 2) * (-delta),  # negative increment (RA-like)
        "m": (np.arange(ny) - ny // 2) * delta,
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
