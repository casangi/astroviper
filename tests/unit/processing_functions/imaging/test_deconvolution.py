"""
Unit tests for :mod:`astroviper.processing_functions.imaging.deconvolution`.

These tests exercise the Python-level orchestration of the CLEAN
pipeline:

- :func:`_validate_deconvolve_params`: parameter validation and defaults.
- :func:`_plane_peak_abs_signed`: signed peak-residual helper.
- :func:`progress_callback`: logging cadence.
- :func:`get_phase_center`: phase-center extraction from coordinates.
- :func:`hogbom_clean`: 5-D cube wrapper around ``hogbom.clean_cube``.
- :func:`deconvolve`: the end-to-end xarray-driven entry point.

All tests construct synthetic inputs so they run without downloading
test data. The cube-driven tests require the compiled Hogbom extension;
the pure-Python ones do not.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.deconvolution import (
    _plane_peak_abs_signed,
    _validate_deconvolve_params,
    asp_clean,
    deconvolve,
    get_phase_center,
    hogbom_clean,
    progress_callback,
)
from astroviper.processing_functions.imaging.utils.return_dict import ReturnDict

try:
    from astroviper.processing_functions.imaging.deconvolvers import (
        hogbom,
    )  # noqa: F401

    HOGBOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    HOGBOM_AVAILABLE = False

try:
    from astroviper.processing_functions.imaging.deconvolvers import (
        aspclean,
    )  # noqa: F401

    ASP_AVAILABLE = True
except ImportError:  # pragma: no cover
    ASP_AVAILABLE = False


requires_hogbom = pytest.mark.skipif(
    not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
)

requires_asp = pytest.mark.skipif(
    not ASP_AVAILABLE, reason="Asp extension not compiled/available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _delta_psf_cube(nt, nf, npol, ny, nx, dtype=np.float32):
    psf = np.zeros((nt, nf, npol, ny, nx), dtype=dtype)
    psf[..., ny // 2, nx // 2] = 1.0
    return psf


def _make_img_xds(
    nt=1,
    nf=1,
    npol=1,
    ny=32,
    nx=32,
    sources=None,
    psf_npol=None,
    with_mask=False,
    dtype=np.float32,
):
    """
    Build a minimal image Dataset matching what :func:`deconvolve`
    expects.

    The dataset has two data variables — ``RESIDUAL`` and
    ``POINT_SPREAD_FUNCTION`` — wired up as a ``"residual"`` data
    group. The residual contains point sources (via ``sources``) and
    the PSF is a delta at the centre pixel.

    Parameters
    ----------
    sources : dict, optional
        Maps ``(tt, nn, pp)`` -> list of ``(iy, ix, amp)``. Defaults to
        a single source at the centre of every plane.
    psf_npol : int, optional
        PSF polarization count (defaults to ``npol``). Set to ``1`` to
        exercise the broadcast path.
    with_mask : bool
        If True, attach a ``MASK_RESIDUAL`` data variable.
    """
    if psf_npol is None:
        psf_npol = npol

    resid = np.zeros((nt, nf, npol, ny, nx), dtype=dtype)
    if sources is None:
        sources = {
            (t, f, p): [(ny // 2, nx // 2, 1.0 + 0.1 * (t * nf * npol + f * npol + p))]
            for t in range(nt)
            for f in range(nf)
            for p in range(npol)
        }
    for (tt, nn, pp), src_list in sources.items():
        for iy, ix, amp in src_list:
            resid[tt, nn, pp, iy, ix] = amp

    psf = _delta_psf_cube(nt, nf, psf_npol, ny, nx, dtype=dtype)

    coords = {
        "time": np.arange(nt, dtype=np.float64),
        "frequency": 1.0e9 + 1.0e6 * np.arange(nf),
        "polarization": ["I", "Q", "U", "V"][:npol],
        "l": np.linspace(-0.01, 0.01, nx),
        "m": np.linspace(-0.01, 0.01, ny),
    }

    data_vars = {
        "RESIDUAL": (["time", "frequency", "polarization", "l", "m"], resid),
        "POINT_SPREAD_FUNCTION": (
            ["time", "frequency", "polarization", "l", "m"],
            psf,
        ),
        # The deconvolve return-dict builder records the per-plane peak PSF
        # sidelobe, so the data group must carry a
        # MAX_SIDELOBE_POINT_SPREAD_FUNCTION (time, frequency, polarization)
        # variable (produced in production by point_spread_function_gaussian_fit).
        "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
            ["time", "frequency", "polarization"],
            np.full((nt, nf, npol), 0.2, dtype=dtype),
        ),
    }

    if with_mask:
        # Boolean mask, matching the in-memory MASK produced by make_mask in
        # production and required by the hogbom binding.
        mask = np.ones((nt, nf, npol, ny, nx), dtype=bool)
        data_vars["MASK_RESIDUAL"] = (
            ["time", "frequency", "polarization", "l", "m"],
            mask,
        )

    xds = xr.Dataset(data_vars, coords=coords)

    # The (y, x) dimensions are named ``l``/``m`` in astroviper image
    # datasets. ``deconvolve`` / ``imgstats`` operate on the trailing
    # two axes of the values array, so dimension naming is not
    # inspected; we only need the data_groups attribute.
    residual_group = {
        "sky": "RESIDUAL",
        "point_spread_function": "POINT_SPREAD_FUNCTION",
        "max_sidelobe_point_spread_function": "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
    }
    if with_mask:
        residual_group["mask"] = "MASK_RESIDUAL"
    xds.attrs["data_groups"] = {"residual": residual_group}
    return xds


# ---------------------------------------------------------------------------
# _validate_deconvolve_params
# ---------------------------------------------------------------------------


class TestValidateDeconvolveParams:
    def test_none_returns_defaults(self):
        result = _validate_deconvolve_params(None)
        assert result == {
            "gain": 0.1,
            "niter": 1000,
            "threshold": 0.0,
            "primary_beam_limit": 0.0,
            "clean_box": (-1, -1, -1, -1),
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        }

    def test_empty_dict_returns_defaults(self):
        assert _validate_deconvolve_params({}) == {
            "gain": 0.1,
            "niter": 1000,
            "threshold": 0.0,
            "primary_beam_limit": 0.0,
            "clean_box": (-1, -1, -1, -1),
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        }

    def test_partial_params_filled(self):
        result = _validate_deconvolve_params({"gain": 0.05})
        assert result["gain"] == 0.05
        assert result["niter"] == 1000
        assert result["threshold"] == 0.0
        assert result["clean_box"] == (-1, -1, -1, -1)
        assert result["minpsffraction"] == 0.05
        assert result["maxpsffraction"] == 0.8

    def test_full_valid_params_preserved(self):
        params = {
            "gain": 0.3,
            "niter": 42,
            "threshold": 1e-6,
            "clean_box": (1, 2, 3, 4),
            "minpsffraction": 0.1,
            "maxpsffraction": 0.7,
        }
        result = _validate_deconvolve_params(params)
        assert result == params

    @pytest.mark.parametrize("gain", [0.01, 0.1, 0.5, 1.0])
    def test_gain_accepts_valid(self, gain):
        result = _validate_deconvolve_params({"gain": gain})
        assert result["gain"] == gain

    @pytest.mark.parametrize("gain", [0.0, -0.1, 1.1, 2.0])
    def test_gain_rejects_invalid(self, gain):
        with pytest.raises(ValueError, match="CLEAN gain"):
            _validate_deconvolve_params({"gain": gain})

    @pytest.mark.parametrize("niter", [1, 100, 1000])
    def test_niter_accepts_valid(self, niter):
        result = _validate_deconvolve_params({"niter": niter})
        assert result["niter"] == niter

    @pytest.mark.parametrize("niter", [0, -1, 1.5, "100", None])
    def test_niter_rejects_invalid(self, niter):
        with pytest.raises(ValueError, match="positive integer"):
            _validate_deconvolve_params({"niter": niter})

    @pytest.mark.parametrize("threshold", [None, 0.0, 1e-6, 1.0])
    def test_threshold_accepts_valid(self, threshold):
        result = _validate_deconvolve_params({"threshold": threshold})
        assert result["threshold"] == threshold

    def test_threshold_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_deconvolve_params({"threshold": -0.1})

    @pytest.mark.parametrize(
        "box",
        [None, (-1, -1, -1, -1), (0, 10, 0, 10), (5, 15, 10, 20)],
    )
    def test_clean_box_accepts_valid(self, box):
        result = _validate_deconvolve_params({"clean_box": box})
        assert result["clean_box"] == box

    @pytest.mark.parametrize(
        "box",
        [(0, 10), (0, 10, 0), (0, 10, 0, 10, 0), "invalid", []],
    )
    def test_clean_box_rejects_invalid(self, box):
        with pytest.raises(ValueError, match="4-tuple"):
            _validate_deconvolve_params({"clean_box": box})

    def test_does_not_mutate_other_keys(self):
        params = {"gain": 0.2, "extra": "ignored"}
        result = _validate_deconvolve_params(params)
        assert result["extra"] == "ignored"


# ---------------------------------------------------------------------------
# _plane_peak_abs_signed
# ---------------------------------------------------------------------------


class TestPlanePeakAbsSigned:
    def test_returns_signed_peak(self):
        arr = np.zeros((8, 8), dtype=np.float32)
        arr[3, 3] = 5.0
        arr[4, 4] = -7.0
        assert _plane_peak_abs_signed(arr) == pytest.approx(-7.0)

    def test_no_mask_when_all_positive(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert _plane_peak_abs_signed(arr) == pytest.approx(4.0)

    def test_with_mask_excludes_pixels(self):
        arr = np.array([[1.0, 2.0], [3.0, -10.0]], dtype=np.float32)
        mask = np.ones_like(arr)
        mask[1, 1] = 0.0  # exclude the largest magnitude
        assert _plane_peak_abs_signed(arr, mask=mask) == pytest.approx(3.0)

    def test_mask_all_false_returns_nan(self):
        arr = np.ones((4, 4), dtype=np.float32)
        mask = np.zeros_like(arr)
        result = _plane_peak_abs_signed(arr, mask=mask)
        assert np.isnan(result)

    def test_mask_threshold_is_half(self):
        """Mask values must exceed 0.5 to count as valid."""
        arr = np.array([[0.0, 5.0], [0.0, 0.0]], dtype=np.float32)
        mask = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=np.float32)
        result = _plane_peak_abs_signed(arr, mask=mask)
        # 0.5 is NOT > 0.5, so the 5.0 pixel is excluded.
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# get_phase_center
# ---------------------------------------------------------------------------


class TestGetPhaseCenter:
    def test_reads_centre_pixel(self):
        ny, nx = 4, 4
        ra = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
        dec = -ra

        xds = xr.Dataset(
            coords={
                "right_ascension": (["l", "m"], ra),
                "declination": (["l", "m"], dec),
            }
        )
        result = get_phase_center(xds)
        # Centre pixel is (ny//2, nx//2).
        expected_ra = ra[ny // 2, nx // 2]
        expected_dec = dec[ny // 2, nx // 2]
        assert result == f"{expected_ra},{expected_dec}"


# ---------------------------------------------------------------------------
# hogbom_clean (5D cube wrapper)
# ---------------------------------------------------------------------------


@requires_hogbom
class TestHogbomCleanCube:
    def test_basic_cube_cleaned(self):
        nt, nf, npol, ny, nx = 1, 1, 1, 16, 16
        resid = np.zeros((nt, nf, npol, ny, nx), dtype=np.float32)
        resid[0, 0, 0, 8, 8] = 2.5
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        result = hogbom_clean(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            deconvolve_params={"gain": 1.0, "niter": 10, "threshold": 0.1},
        )

        assert result["iterations_performed"].shape == (nt, nf, npol)
        assert model[0, 0, 0, 8, 8] == pytest.approx(2.5, abs=1e-6)
        assert np.allclose(resid, 0.0, atol=1e-6)

    def test_multiplane_threaded_matches_serial(self):
        nt, nf, npol, ny, nx = 2, 2, 2, 16, 16
        rng = np.random.default_rng(7)
        base = rng.standard_normal((nt, nf, npol, ny, nx)).astype(np.float32)
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)

        resid_a = base.copy()
        model_a = np.zeros_like(resid_a)
        hogbom_clean(
            resid_a,
            psf,
            model_a,
            {"gain": 0.2, "niter": 20, "threshold": 0.05},
            num_threads=1,
        )

        resid_b = base.copy()
        model_b = np.zeros_like(resid_b)
        hogbom_clean(
            resid_b,
            psf,
            model_b,
            {"gain": 0.2, "niter": 20, "threshold": 0.05},
            num_threads=4,
        )

        assert np.array_equal(model_a, model_b)
        assert np.array_equal(resid_a, resid_b)

    def test_broadcast_psf(self):
        nt, nf, npol, ny, nx = 1, 1, 3, 16, 16
        resid = np.zeros((nt, nf, npol, ny, nx), dtype=np.float32)
        for p in range(npol):
            resid[0, 0, p, 8, 8] = 1.0 + p
        psf = _delta_psf_cube(nt, nf, 1, ny, nx)
        model = np.zeros_like(resid)

        hogbom_clean(resid, psf, model, {"gain": 1.0, "niter": 5, "threshold": 0.1})

        for p in range(npol):
            assert model[0, 0, p, 8, 8] == pytest.approx(1.0 + p, abs=1e-6)

    def test_mask_cube_respected(self):
        nt, nf, npol, ny, nx = 1, 1, 1, 16, 16
        resid = np.zeros((nt, nf, npol, ny, nx), dtype=np.float32)
        resid[0, 0, 0, 5, 5] = 10.0
        resid[0, 0, 0, 10, 10] = 3.0
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        # The hogbom binding requires a boolean mask (matching the in-memory
        # MASK produced by make_mask in production).
        mask = np.ones((nt, nf, npol, ny, nx), dtype=bool)
        mask[0, 0, 0, 5, 5] = False

        hogbom_clean(
            resid,
            psf,
            model,
            {"gain": 1.0, "niter": 5, "threshold": 0.1},
            mask_cube=mask,
        )

        # Masked source should survive; other source should be cleaned.
        assert resid[0, 0, 0, 5, 5] == pytest.approx(10.0, abs=1e-6)
        assert model[0, 0, 0, 10, 10] == pytest.approx(3.0, abs=1e-6)

    def test_rejects_non_5d(self):
        arr = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="5D"):
            hogbom_clean(arr, arr, arr)

    def test_rejects_shape_mismatch(self):
        resid = np.zeros((1, 1, 1, 16, 16), dtype=np.float32)
        psf = np.zeros((1, 1, 1, 16, 16), dtype=np.float32)
        model = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="same shape"):
            hogbom_clean(resid, psf, model)

    def test_rejects_psf_spatial_mismatch(self):
        resid = np.zeros((1, 1, 1, 16, 16), dtype=np.float32)
        psf = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
        model = np.zeros_like(resid)
        with pytest.raises(ValueError, match="time, frequency, y, x"):
            hogbom_clean(resid, psf, model)

    def test_rejects_bad_psf_pol(self):
        resid = np.zeros((1, 1, 3, 16, 16), dtype=np.float32)
        psf = np.zeros((1, 1, 2, 16, 16), dtype=np.float32)
        model = np.zeros_like(resid)
        with pytest.raises(ValueError, match="polarization"):
            hogbom_clean(resid, psf, model)

    def test_rejects_bad_mask_shape(self):
        resid = np.zeros((1, 1, 1, 16, 16), dtype=np.float32)
        psf = _delta_psf_cube(1, 1, 1, 16, 16)
        model = np.zeros_like(resid)
        bad_mask = np.ones((1, 1, 1, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="mask_cube"):
            hogbom_clean(resid, psf, model, mask_cube=bad_mask)


# ---------------------------------------------------------------------------
# deconvolve (end-to-end)
# ---------------------------------------------------------------------------


@requires_hogbom
class TestDeconvolve:
    def test_basic_single_plane(self):
        xds = _make_img_xds()
        returndict = deconvolve(
            img_xds=xds,
            deconvolve_params={"gain": 1.0, "niter": 5, "threshold": 0.05},
        )

        assert isinstance(returndict, ReturnDict)
        assert len(returndict.data) == 1  # 1 time * 1 freq * 1 pol
        entry = list(returndict.data.values())[0]
        for field in (
            "iter_done",
            "niter",
            "cyclethreshold",
            "loop_gain",
            "peakres",
            "peakres_nomask",
            "masksum",
            "model_flux",
            "start_model_flux",
            "start_peakres",
            "start_peakres_nomask",
            "stokes",
            "frequency",
            "time",
        ):
            assert field in entry

        # Model variable created in place.
        assert "SKY_MODEL" in xds.data_vars
        # Source cleaned into model.
        model = xds["SKY_MODEL"].values
        assert model[0, 0, 0, 16, 16] == pytest.approx(1.0, abs=1e-5)

    def test_multi_pol_returndict_entries(self):
        nt, nf, npol = 1, 1, 4
        xds = _make_img_xds(nt=nt, nf=nf, npol=npol)
        returndict = deconvolve(
            img_xds=xds,
            deconvolve_params={"gain": 1.0, "niter": 5, "threshold": 0.05},
        )
        assert len(returndict.data) == nt * nf * npol
        for p in range(npol):
            entry = returndict.sel(pol=p)
            assert isinstance(entry, dict)
            assert entry["stokes"] == ["I", "Q", "U", "V"][p]

    def test_multi_chan_returndict_frequencies(self):
        nt, nf, npol = 1, 3, 1
        xds = _make_img_xds(nt=nt, nf=nf, npol=npol)
        returndict = deconvolve(
            img_xds=xds,
            deconvolve_params={"gain": 1.0, "niter": 5, "threshold": 0.05},
        )
        assert len(returndict.data) == nt * nf * npol
        freqs = [float(returndict.sel(chan=c)["frequency"]) for c in range(nf)]
        assert len(set(freqs)) == nf

    def test_rejects_psf_pol_mismatch(self):
        """Residual polarization != PSF polarization raises (unless PSF has
        exactly one polarization, which ``deconvolve`` handles by
        broadcasting). Here we force npol_psf == 2 with npol == 3 to
        trigger the error path."""
        xds = _make_img_xds(npol=3, psf_npol=3)
        # Swap in a PSF with mismatched polarization dim size by
        # constructing a new array on a separate pol coord.
        psf_arr = np.zeros(
            (
                xds.sizes["time"],
                xds.sizes["frequency"],
                2,
                xds.sizes["l"],
                xds.sizes["m"],
            ),
            dtype=np.float32,
        )
        psf_arr[..., xds.sizes["l"] // 2, xds.sizes["m"] // 2] = 1.0
        # Build a second dataset carrying only the bad PSF, then merge in
        # by deleting+re-adding so xarray accepts the new size.
        del xds["POINT_SPREAD_FUNCTION"]
        xds = xds.assign_coords({"psf_pol": ["I", "Q"]})
        xds["POINT_SPREAD_FUNCTION"] = (
            ["time", "frequency", "psf_pol", "l", "m"],
            psf_arr,
        )
        with pytest.raises((ValueError, KeyError)):
            deconvolve(img_xds=xds)

    def test_rejects_unknown_algorithm(self):
        xds = _make_img_xds()
        with pytest.raises(ValueError, match="not recognized"):
            deconvolve(img_xds=xds, algorithm="clark")

    def test_num_threads_parameter_passes_through(self):
        """num_threads > 1 must not change the result for a deterministic
        input (same-input-same-output contract)."""
        xds_a = _make_img_xds(nt=1, nf=2, npol=2, ny=16, nx=16)
        xds_b = _make_img_xds(nt=1, nf=2, npol=2, ny=16, nx=16)

        deconvolve(
            img_xds=xds_a,
            deconvolve_params={"gain": 0.2, "niter": 10, "threshold": 0.05},
            num_threads=1,
        )
        deconvolve(
            img_xds=xds_b,
            deconvolve_params={"gain": 0.2, "niter": 10, "threshold": 0.05},
            num_threads=4,
        )

        assert np.array_equal(xds_a["RESIDUAL"].values, xds_b["RESIDUAL"].values)
        assert np.array_equal(xds_a["SKY_MODEL"].values, xds_b["SKY_MODEL"].values)

    def test_returndict_consistency(self):
        xds = _make_img_xds(nt=1, nf=1, npol=1)
        returndict = deconvolve(
            img_xds=xds,
            deconvolve_params={"gain": 0.5, "niter": 20, "threshold": 0.001},
        )
        entry = list(returndict.data.values())[0]

        # start_peakres must be >= peakres (peak should decrease).
        start = (
            entry["start_peakres"][-1]
            if isinstance(entry["start_peakres"], list)
            else entry["start_peakres"]
        )
        end = (
            entry["peakres"][-1]
            if isinstance(entry["peakres"], list)
            else entry["peakres"]
        )
        assert abs(end) <= abs(start) + 1e-6

        # iter_done within bounds.
        iter_done = (
            entry["iter_done"][-1]
            if isinstance(entry["iter_done"], list)
            else entry["iter_done"]
        )
        assert 0 <= iter_done <= entry["niter"]

    def test_deconvolve_with_mask(self):
        xds = _make_img_xds(nt=1, nf=1, npol=1, with_mask=True)
        # Zero the mask at the source location; CLEAN should not touch it.
        xds["MASK_RESIDUAL"].values[0, 0, 0, 16, 16] = 0.0

        returndict = deconvolve(
            img_xds=xds,
            deconvolve_params={"gain": 1.0, "niter": 5, "threshold": 0.05},
        )
        # Source should survive since the pixel is masked out.
        assert xds["RESIDUAL"].values[0, 0, 0, 16, 16] == pytest.approx(1.0, abs=1e-6)
        assert xds["SKY_MODEL"].values[0, 0, 0, 16, 16] == pytest.approx(0.0, abs=1e-6)
        assert isinstance(returndict, ReturnDict)


# ---------------------------------------------------------------------------
# asp_clean (5-D cube wrapper around aspclean.clean_cube)
# ---------------------------------------------------------------------------


@requires_asp
class TestAspClean:
    @staticmethod
    def _gauss_psf_cube(nt, nf, npol, ny, nx, sig=2.0):
        from astroviper.processing_functions.imaging.deconvolvers import aspclean

        yy, xx = np.mgrid[0:ny, 0:nx]
        g = np.exp(-(((xx - nx // 2) ** 2 + (yy - ny // 2) ** 2) / (2 * sig**2)))
        psf = np.zeros((nt, nf, npol, ny, nx), dtype=np.float64)
        psf[...] = g
        return psf, aspclean

    def test_basic_cube_point_source(self):
        nt, nf, npol, ny, nx = 1, 1, 1, 32, 32
        resid = np.zeros((nt, nf, npol, ny, nx), dtype=np.float32)
        resid[0, 0, 0, 16, 16] = 5.0
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        result = asp_clean(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            deconvolve_params={
                "gain": 0.5,
                "niter": 50,
                "threshold": 0.01,
                "fusedthreshold": 0.5,
                "psf_width": 1.0,
            },
        )
        assert result["iterations_performed"].shape == (nt, nf, npol)
        assert result["peak_residual"].shape == (nt, nf, npol)
        # point source essentially cleaned into the model at its pixel
        assert np.max(np.abs(resid)) < 0.05
        assert model[0, 0, 0, 16, 16] == pytest.approx(5.0, abs=0.1)

    def test_extended_source_flux_recovered_and_invariant(self):
        nt, nf, npol, ny, nx = 1, 1, 1, 64, 64
        psf, aspclean = self._gauss_psf_cube(nt, nf, npol, ny, nx, sig=2.0)
        yy, xx = np.mgrid[0:ny, 0:nx]
        sky = 3.0 * np.exp(-(((xx - 36) ** 2 + (yy - 30) ** 2) / (2 * 4.0**2)))
        dirty = aspclean.convolve_centered(sky, psf[0, 0, 0])
        # Independent C-contiguous copy so the reference `dirty` is not the
        # same buffer that asp_clean mutates in place.
        resid = dirty[None, None, None].copy()
        model = np.zeros_like(resid)

        asp_clean(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            deconvolve_params={
                "gain": 0.1,
                "niter": 200,
                "threshold": 0.01,
                "fusedthreshold": 0.1,
            },
        )
        # residual reduced and most flux recovered
        assert np.max(np.abs(resid[0, 0, 0])) < 0.2 * np.max(np.abs(dirty))
        assert 0.6 < model.sum() / sky.sum() < 1.2
        # Reconstruction invariant: residual == dirty - model (*) psf. This is
        # exact except for the per-step box clamping near the image edges (a
        # faithful property of the CASA algorithm), so a tight-but-not-machine
        # tolerance is used.
        recon = aspclean.convolve_centered(model[0, 0, 0], psf[0, 0, 0])
        err = np.max(np.abs(resid[0, 0, 0] + recon - dirty))
        assert err / np.max(np.abs(dirty)) < 1e-5

    def test_threaded_matches_serial(self):
        nt, nf, npol, ny, nx = 1, 2, 2, 32, 32
        rng = np.random.default_rng(11)
        base = rng.standard_normal((nt, nf, npol, ny, nx)).astype(np.float64)
        psf, _ = self._gauss_psf_cube(nt, nf, npol, ny, nx, sig=2.0)

        ra, ma = base.copy(), np.zeros_like(base)
        asp_clean(
            ra,
            psf,
            ma,
            {"gain": 0.2, "niter": 40, "threshold": 0.05, "fusedthreshold": 0.1},
            num_threads=1,
        )
        rb, mb = base.copy(), np.zeros_like(base)
        asp_clean(
            rb,
            psf,
            mb,
            {"gain": 0.2, "niter": 40, "threshold": 0.05, "fusedthreshold": 0.1},
            num_threads=4,
        )

        assert np.array_equal(ra, rb)
        assert np.array_equal(ma, mb)

    def test_broadcast_psf(self):
        nt, nf, npol, ny, nx = 1, 1, 3, 32, 32
        resid = np.zeros((nt, nf, npol, ny, nx), dtype=np.float32)
        for p in range(npol):
            resid[0, 0, p, 16, 16] = 1.0 + p
        psf = _delta_psf_cube(nt, nf, 1, ny, nx)
        model = np.zeros_like(resid)
        out = asp_clean(
            resid,
            psf,
            model,
            {
                "gain": 0.5,
                "niter": 40,
                "threshold": 0.01,
                "fusedthreshold": 0.5,
                "psf_width": 1.0,
            },
        )
        assert out["model_flux"].shape == (nt, nf, npol)
        assert np.all(np.asarray(out["model_flux"]) > 0)

    def test_bool_mask_converted_and_respected(self):
        nt, nf, npol, ny, nx = 1, 1, 1, 32, 32
        resid = np.zeros((nt, nf, npol, ny, nx), dtype=np.float32)
        resid[0, 0, 0, 6, 6] = 10.0
        resid[0, 0, 0, 22, 22] = 4.0
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        # bool mask zeroing a box around the strong source (Asp wants a
        # numeric mask; asp_clean must convert without mutating the input).
        mask = np.ones((nt, nf, npol, ny, nx), dtype=bool)
        mask[0, 0, 0, 0:12, 0:12] = False

        # Restrict to small scales so a large-scale component just outside the
        # masked box cannot leak flux into it.
        asp_clean(
            resid,
            psf,
            model,
            {
                "gain": 0.5,
                "niter": 60,
                "threshold": 0.01,
                "fusedthreshold": 0.5,
                "psf_width": 1.0,
                "largestscale": 2,
            },
            mask_cube=mask,
        )

        assert mask.dtype == np.bool_  # input mask untouched
        # No component centres are placed in the masked box, so the masked
        # strong source is preserved and the model deep inside the box is
        # negligible (only faint Gaussian tails from edge components bleed in).
        assert resid[0, 0, 0, 6, 6] == pytest.approx(10.0, abs=0.05)
        assert np.max(np.abs(model[0, 0, 0, 0:6, 0:6])) < 0.05
        # the unmasked source is essentially fully cleaned
        assert abs(resid[0, 0, 0, 22, 22]) < 0.5

    def test_rejects_non_5d(self):
        arr = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="5D"):
            asp_clean(arr, arr, arr)

    def test_rejects_shape_mismatch(self):
        resid = np.zeros((1, 1, 1, 16, 16), dtype=np.float32)
        psf = np.zeros((1, 1, 1, 16, 16), dtype=np.float32)
        model = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="same shape"):
            asp_clean(resid, psf, model)

    def test_rejects_bad_psf_pol(self):
        resid = np.zeros((1, 1, 3, 16, 16), dtype=np.float32)
        psf = np.zeros((1, 1, 2, 16, 16), dtype=np.float32)
        model = np.zeros_like(resid)
        with pytest.raises(ValueError, match="polarization"):
            asp_clean(resid, psf, model)


# ---------------------------------------------------------------------------
# deconvolve end-to-end with algorithm="asp"
# ---------------------------------------------------------------------------


@requires_asp
class TestDeconvolveAsp:
    def test_basic_single_plane(self):
        xds = _make_img_xds()
        returndict = deconvolve(
            img_xds=xds,
            algorithm="asp",
            deconvolve_params={
                "gain": 0.5,
                "niter": 50,
                "threshold": 0.01,
                "fusedthreshold": 0.5,
                "psf_width": 1.0,
            },
        )
        assert isinstance(returndict, ReturnDict)
        assert "SKY_MODEL" in xds.data_vars
        # the point source is cleaned into the model
        assert xds["SKY_MODEL"].values[0, 0, 0, 16, 16] == pytest.approx(1.0, abs=0.1)
        assert np.max(np.abs(xds["RESIDUAL"].values)) < 0.1

    @pytest.mark.parametrize("alias", ["asp", "aspclean", "asp_clean", "ASP"])
    def test_algorithm_aliases(self, alias):
        xds = _make_img_xds(nt=1, nf=1, npol=1)
        rd = deconvolve(
            img_xds=xds,
            algorithm=alias,
            deconvolve_params={"niter": 10, "psf_width": 1.0, "fusedthreshold": 0.5},
        )
        assert isinstance(rd, ReturnDict)

    def test_multi_plane_returndict_entries(self):
        nt, nf, npol = 1, 2, 2
        xds = _make_img_xds(nt=nt, nf=nf, npol=npol)
        rd = deconvolve(
            img_xds=xds,
            algorithm="asp",
            deconvolve_params={
                "gain": 0.5,
                "niter": 30,
                "threshold": 0.01,
                "fusedthreshold": 0.5,
                "psf_width": 1.0,
            },
        )
        assert len(rd.data) == nt * nf * npol


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
