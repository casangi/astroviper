"""Tests for astroviper.distributed_applications.image_analysis.imfit."""

from __future__ import annotations

import warnings

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import Angle

from astroviper.distributed_applications.image_analysis.imfit import imfit
from astroviper.distributed_applications.model.component_models import make_gauss2d
from astroviper.utils._gaussian_math import FWHM2SIG, SIG2FWHM

# ---------------------------------------------------------------------------
# Synthetic xradio-style Dataset builder
# ---------------------------------------------------------------------------


def _make_xradio_image(
    nl=128,
    nm=128,
    cellsize=1e-5,  # radians per pixel (~2 arcsec)
    cellsize_m=None,
    components=None,
    offset=0.05,
    noise_sigma=0.02,
    beam_fwhm_maj=None,
    beam_fwhm_min=None,
    beam_pa=0.0,
    phase_center=(1.0, 0.5),  # RA, Dec in radians
    frame="icrs",
    projection="SIN",
    add_sky_coord_grids=False,
    seed=42,
):
    """Build a synthetic xradio Dataset with known Gaussian sources.

    Uses ``make_empty_sky_image`` for the base Dataset structure and
    ``make_gauss2d`` for source construction.  The l-axis is **decreasing**
    (positive → negative), so the pixel index for a source at l0 = k*cellsize
    is ``nl//2 - k``.  The m-axis is increasing, so the pixel index for
    m0 = k*cellsize is ``nm//2 + k``.

    Parameters
    ----------
    components : list of dict
        Each dict has keys: amp, l0, m0, fwhm_maj, fwhm_min, pa (all in
        radians for positions/widths, radians for PA).
    """
    from xradio.image import make_empty_sky_image

    rng = np.random.default_rng(seed)
    cellsize_l = float(cellsize)
    cellsize_m = cellsize_l if cellsize_m is None else float(cellsize_m)

    xds = make_empty_sky_image(
        phase_center=list(phase_center),
        image_size=[nl, nm],
        cell_size=[cellsize_l, cellsize_m],
        frequency_coords=np.array([1.0e9]),
        pol_coords=["I"],
        time_coords=np.array([0.0]),
        direction_reference=frame,
        projection=projection,
    )

    # Build image on a 2D DataArray; make_gauss2d accumulates components
    plane = xr.DataArray(
        np.full((nl, nm), offset, dtype=float),
        dims=("l", "m"),
        coords={"l": xds.coords["l"], "m": xds.coords["m"]},
    )

    for comp in components or []:
        plane = make_gauss2d(
            plane,
            a=comp["fwhm_maj"],
            b=comp["fwhm_min"],
            theta=comp["pa"],
            x0=comp["l0"],
            y0=comp["m0"],
            peak=comp["amp"],
            x_coord="l",
            y_coord="m",
            angle="pa",
        )

    # Expand to (time, frequency, polarization, l, m) and add noise
    img = plane.values[np.newaxis, np.newaxis, np.newaxis, :, :]
    img = img + rng.normal(0, noise_sigma, img.shape)

    xds["SKY"] = xr.DataArray(
        img,
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            "time": xds.coords["time"],
            "frequency": xds.coords["frequency"],
            "polarization": xds.coords["polarization"],
            "l": xds.coords["l"],
            "m": xds.coords["m"],
        },
    )

    if beam_fwhm_maj is not None:
        beam_data = np.array([[[[beam_fwhm_maj, beam_fwhm_min, beam_pa]]]])
        xds["BEAM_FIT_PARAMS_SKY"] = xr.DataArray(
            beam_data,
            dims=("time", "frequency", "polarization", "beam_params_label"),
            coords={
                "time": xds.coords["time"],
                "frequency": xds.coords["frequency"],
                "polarization": xds.coords["polarization"],
                "beam_params_label": xds.coords["beam_params_label"],
            },
        )

    if add_sky_coord_grids:
        # Simple tangent-plane approximation for RA/Dec grids
        ra0, dec0 = phase_center
        cos_dec = np.cos(dec0)
        L, M = np.meshgrid(
            xds.coords["l"].values, xds.coords["m"].values, indexing="ij"
        )
        RA = ra0 + L / cos_dec
        DEC = dec0 + M
        xds["right_ascension"] = xr.DataArray(
            RA, dims=("l", "m"), coords={"l": xds.coords["l"], "m": xds.coords["m"]}
        )
        xds["declination"] = xr.DataArray(
            DEC, dims=("l", "m"), coords={"l": xds.coords["l"], "m": xds.coords["m"]}
        )

    return xds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImfitBasic:
    """Basic functionality tests."""

    def test_single_circular_source(self):
        """Fit a single circular Gaussian and verify recovery."""
        cellsize = 1e-5
        nl = 128
        fwhm = 10 * cellsize
        fwhm_pix = 10.0  # in pixels
        comp = {
            "amp": 1.0,
            "l0": 3 * cellsize,
            "m0": -2 * cellsize,
            "fwhm_maj": fwhm,
            "fwhm_min": fwhm,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl, nm=nl, components=[comp], noise_sigma=0.01, cellsize=cellsize
        )
        # Initial guesses in pixel coordinates (imfit uses coord_type="pixel")
        # l-axis is decreasing: pixel for l0 = k*cellsize is nl//2 - k
        # m-axis is increasing: pixel for m0 = k*cellsize is nm//2 + k
        center_l_pix = nl // 2 - 3  # pixel index for l0 = 3*cellsize
        center_m_pix = nl // 2 - 2  # pixel index for m0 = -2*cellsize
        init = [
            {
                "amp": 0.9,
                "x0": center_l_pix,
                "y0": center_m_pix,
                "fwhm_major": fwhm_pix,
                "fwhm_minor": fwhm_pix,
                "theta": 0.0,
            }
        ]
        ds = imfit(xds, n_components=1, beam_var=None, initial_guesses=init)

        assert ds["success"].values
        np.testing.assert_allclose(ds["amplitude"].values, 1.0, atol=0.05)
        np.testing.assert_allclose(ds["x0_world"].values, comp["l0"], atol=cellsize)
        np.testing.assert_allclose(ds["y0_world"].values, comp["m0"], atol=cellsize)
        np.testing.assert_allclose(ds["fwhm_major_world"].values, fwhm, rtol=0.1)
        np.testing.assert_allclose(ds["fwhm_minor_world"].values, fwhm, rtol=0.1)

    def test_accepts_native_world_initial_guesses(self):
        """Native ``l``/``m`` initial guesses should be converted to pixel space."""
        cellsize = 1e-5
        nl = 128
        fwhm = 10 * cellsize
        comp = {
            "amp": 1.0,
            "l0": 3 * cellsize,
            "m0": -2 * cellsize,
            "fwhm_maj": fwhm,
            "fwhm_min": fwhm,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl, nm=nl, components=[comp], noise_sigma=0.01, cellsize=cellsize
        )
        init = [
            {
                "amp": 0.9,
                "l0": comp["l0"],
                "m0": comp["m0"],
                "fwhm_major": fwhm,
                "fwhm_minor": fwhm,
                "theta": 0.0,
            }
        ]

        ds = imfit(xds, n_components=1, beam_var=None, initial_guesses=init)

        assert ds["success"].values
        np.testing.assert_allclose(ds["x0_world"].values, comp["l0"], atol=cellsize)
        np.testing.assert_allclose(ds["y0_world"].values, comp["m0"], atol=cellsize)
        np.testing.assert_allclose(ds["x0_pixel"].values, nl // 2 - 3, atol=0.5)
        np.testing.assert_allclose(ds["y0_pixel"].values, nl // 2 - 2, atol=0.5)

    def test_no_math_angles_in_output(self):
        """Verify math-convention angles are excluded from imfit output."""
        xds = _make_xradio_image(
            components=[
                {
                    "amp": 1.0,
                    "l0": 0,
                    "m0": 0,
                    "fwhm_maj": 5e-5,
                    "fwhm_min": 3e-5,
                    "pa": 0.3,
                }
            ],
            noise_sigma=0.01,
        )
        ds = imfit(xds, n_components=1, beam_var=None)

        math_vars = [v for v in ds.data_vars if "_math" in v]
        assert math_vars == [], f"Math-angle vars found: {math_vars}"

        pixel_pa_vars = [v for v in ds.data_vars if v.startswith("theta_pixel")]
        assert pixel_pa_vars == [], f"Pixel PA vars found: {pixel_pa_vars}"

    def test_pa_variable_exists(self):
        """Verify the output has 'pa' (not 'theta_world_pa')."""
        xds = _make_xradio_image(
            components=[
                {
                    "amp": 1.0,
                    "l0": 0,
                    "m0": 0,
                    "fwhm_maj": 5e-5,
                    "fwhm_min": 3e-5,
                    "pa": 0.3,
                }
            ],
            noise_sigma=0.01,
        )
        ds = imfit(xds, n_components=1, beam_var=None)
        assert "pa" in ds.data_vars
        assert "pa_err" in ds.data_vars
        assert "theta_world" not in ds.data_vars
        assert "theta_world_pa" not in ds.data_vars


class TestImfitSkyCoords:
    """Tests for sky coordinate translation."""

    def test_accepts_sexagesimal_sky_initial_guesses(self):
        """Sexagesimal sky positions should seed the pixel-space fit correctly."""
        cellsize = 1e-5
        nl = 128
        fwhm = 10 * cellsize
        comp = {
            "amp": 1.0,
            "l0": 0.0,
            "m0": 0.0,
            "fwhm_maj": fwhm,
            "fwhm_min": fwhm,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[comp],
            noise_sigma=0.01,
            cellsize=cellsize,
            add_sky_coord_grids=False,
        )
        init = [
            {
                "amp": 0.9,
                "ra": Angle(1.0, unit="rad").to_string(unit="hourangle", sep=":"),
                "dec": Angle(0.5, unit="rad").to_string(unit="deg", sep=":"),
                "fwhm_major": fwhm,
                "fwhm_minor": fwhm,
                "theta": 0.0,
            }
        ]

        ds = imfit(xds, n_components=1, beam_var=None, initial_guesses=init)

        assert ds["success"].values
        np.testing.assert_allclose(ds["x0_world"].values, 0.0, atol=cellsize)
        np.testing.assert_allclose(ds["y0_world"].values, 0.0, atol=cellsize)

    def test_sky_coords_from_grids(self):
        """When RA/Dec grids exist, interpolate fitted centers."""
        cellsize = 1e-5
        nl = 128
        fwhm = 10 * cellsize
        comp = {
            "amp": 1.0,
            "l0": 2 * cellsize,
            "m0": -1 * cellsize,
            "fwhm_maj": fwhm,
            "fwhm_min": fwhm,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[comp],
            noise_sigma=0.01,
            cellsize=cellsize,
            add_sky_coord_grids=True,
        )
        # Pixel-coordinate initial guesses
        init = [
            {
                "amp": 0.9,
                "x0": nl // 2 - 2,  # l-axis decreasing: pixel for l0=2*cs is nl//2-2
                "y0": nl // 2 - 1,  # m-axis increasing: pixel for m0=-1*cs is nm//2-1
                "fwhm_major": 10.0,
                "fwhm_minor": 10.0,
                "theta": 0.0,
            }
        ]
        ds = imfit(xds, n_components=1, beam_var=None, initial_guesses=init)

        assert "right_ascension" in ds.data_vars
        assert "declination" in ds.data_vars
        # Verify the RA/Dec are close to expected (tangent-plane approx)
        ra0, dec0 = 1.0, 0.5
        expected_ra = ra0 + comp["l0"] / np.cos(dec0)
        expected_dec = dec0 + comp["m0"]
        np.testing.assert_allclose(
            ds["right_ascension"].values.flat[0], expected_ra, atol=cellsize
        )
        np.testing.assert_allclose(
            ds["declination"].values.flat[0], expected_dec, atol=cellsize
        )

    def test_sky_coords_from_descending_grids_propagates_errors(self):
        """Descending l/m sky grids should interpolate centers and errors."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _attach_sky_coordinates,
        )

        l_coord = np.array([2.0, 1.0, 0.0])
        m_coord = np.array([5.0, 3.0, 1.0])
        L, M = np.meshgrid(l_coord, m_coord, indexing="ij")
        xds = xr.Dataset(
            data_vars={
                "right_ascension": xr.DataArray(
                    10.0 * L + 2.0 * M,
                    dims=("l", "m"),
                    coords={"l": l_coord, "m": m_coord},
                ),
                "declination": xr.DataArray(
                    -3.0 * L + 4.0 * M,
                    dims=("l", "m"),
                    coords={"l": l_coord, "m": m_coord},
                ),
            },
            coords={"l": l_coord, "m": m_coord},
        )
        ds = xr.Dataset(
            {
                "x0_world": xr.DataArray([1.25], dims=("component",)),
                "y0_world": xr.DataArray([2.5], dims=("component",)),
                "x0_world_err": xr.DataArray([0.2], dims=("component",)),
                "y0_world_err": xr.DataArray([0.5], dims=("component",)),
            }
        )

        out = _attach_sky_coordinates(ds, xds)

        np.testing.assert_allclose(out["right_ascension"].values, [17.5])
        np.testing.assert_allclose(out["declination"].values, [6.25])
        np.testing.assert_allclose(
            out["right_ascension_err"].values,
            [np.sqrt((10.0 * 0.2) ** 2 + (2.0 * 0.5) ** 2)],
        )
        np.testing.assert_allclose(
            out["declination_err"].values,
            [np.sqrt((-3.0 * 0.2) ** 2 + (4.0 * 0.5) ** 2)],
        )
        assert out["right_ascension"].attrs["description"]
        assert out["declination_err"].attrs["description"]

    @pytest.mark.parametrize(
        "l_coord,m_coord,match",
        [
            (np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0]), "l_coord"),
            (np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0]), "l_coord"),
            (np.array([0.0, 1.0]), np.array([0.0, 1.0, 1.0]), "m_coord"),
            (np.array([0.0, 1.0]), np.array([0.0, 2.0, 1.0]), "m_coord"),
        ],
    )
    def test_lm_radec_interpolation_rejects_non_monotonic_axes(
        self, l_coord, m_coord, match
    ):
        """RA/Dec grid interpolation should reject repeated or unsorted axes."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _prepare_lm_radec_interpolation_grids,
        )

        ra_grid = xr.DataArray(
            np.zeros((l_coord.size, m_coord.size)),
            dims=("l", "m"),
            coords={"l": l_coord, "m": m_coord},
        )
        dec_grid = xr.DataArray(
            np.zeros((l_coord.size, m_coord.size)),
            dims=("l", "m"),
            coords={"l": l_coord, "m": m_coord},
        )

        with pytest.raises(ValueError, match=match):
            _prepare_lm_radec_interpolation_grids(l_coord, m_coord, ra_grid, dec_grid)

    def test_sky_coords_from_wcs(self):
        """When no grids exist, use WCS projection from coordinate_system_info."""
        cellsize = 1e-5
        nl = 128
        fwhm = 10 * cellsize
        comp = {
            "amp": 1.0,
            "l0": 0.0,
            "m0": 0.0,
            "fwhm_maj": fwhm,
            "fwhm_min": fwhm,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[comp],
            noise_sigma=0.01,
            cellsize=cellsize,
            add_sky_coord_grids=False,
        )
        # Source at center → pixel center
        init = [
            {
                "amp": 0.9,
                "x0": nl // 2,
                "y0": nl // 2,
                "fwhm_major": 10.0,
                "fwhm_minor": 10.0,
                "theta": 0.0,
            }
        ]
        ds = imfit(xds, n_components=1, beam_var=None, initial_guesses=init)

        assert "Right Ascension" in ds.data_vars
        assert "Declination" in ds.data_vars
        assert "Right Ascension_err" in ds.data_vars
        assert "Declination_err" in ds.data_vars
        assert "Right Ascension err" not in ds.data_vars
        assert "Declination err" not in ds.data_vars
        # Source at phase center → RA, Dec ≈ phase center
        np.testing.assert_allclose(
            ds["Right Ascension"].values.flat[0], 1.0, atol=cellsize
        )
        np.testing.assert_allclose(ds["Declination"].values.flat[0], 0.5, atol=cellsize)
        assert ds["Right Ascension_err"].attrs["description"]
        assert ds["Declination_err"].attrs["description"]
        assert ds["Right Ascension"].attrs.get("frame") == "icrs"


class TestImfitDeconvolution:
    """Tests for beam deconvolution."""

    def test_resolved_source_deconvolution(self):
        """A source larger than the beam should be deconvolved correctly."""
        cellsize = 1e-5
        nl = 128
        src_fwhm_maj = 15 * cellsize
        src_fwhm_min = 10 * cellsize
        beam_fwhm = 5 * cellsize
        # Convolved sizes (circular beam, same PA) add in quadrature
        conv_fwhm_maj = np.sqrt(src_fwhm_maj**2 + beam_fwhm**2)
        conv_fwhm_min = np.sqrt(src_fwhm_min**2 + beam_fwhm**2)
        pa = 0.5
        comp = {
            "amp": 1.0,
            "l0": 0.0,
            "m0": 0.0,
            "fwhm_maj": conv_fwhm_maj,
            "fwhm_min": conv_fwhm_min,
            "pa": pa,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[comp],
            noise_sigma=0.005,
            cellsize=cellsize,
            beam_fwhm_maj=beam_fwhm,
            beam_fwhm_min=beam_fwhm,
            beam_pa=0.0,
        )
        conv_fwhm_maj_pix = conv_fwhm_maj / cellsize
        conv_fwhm_min_pix = conv_fwhm_min / cellsize
        init = [
            {
                "amp": 0.9,
                "x0": nl // 2,
                "y0": nl // 2,
                "fwhm_major": conv_fwhm_maj_pix,
                "fwhm_minor": conv_fwhm_min_pix,
                "theta": pa,
            }
        ]
        ds = imfit(xds, n_components=1, initial_guesses=init)

        assert "fwhm_major_deconv" in ds.data_vars
        assert "is_unresolved" in ds.data_vars
        assert not ds["is_unresolved"].values.flat[0]
        np.testing.assert_allclose(
            ds["fwhm_major_deconv"].values.flat[0], src_fwhm_maj, rtol=0.15
        )
        np.testing.assert_allclose(
            ds["fwhm_minor_deconv"].values.flat[0], src_fwhm_min, rtol=0.15
        )

    def test_unresolved_source(self):
        """A source smaller than the beam → unresolved with upper limit."""
        cellsize = 1e-5
        nl = 128
        beam_fwhm = 15 * cellsize
        src_fwhm_maj = beam_fwhm * 0.7
        src_fwhm_min = beam_fwhm * 0.5
        comp = {
            "amp": 1.0,
            "l0": 0.0,
            "m0": 0.0,
            "fwhm_maj": src_fwhm_maj,
            "fwhm_min": src_fwhm_min,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[comp],
            noise_sigma=0.005,
            cellsize=cellsize,
            beam_fwhm_maj=beam_fwhm,
            beam_fwhm_min=beam_fwhm,
            beam_pa=0.0,
        )
        init = [
            {
                "amp": 0.9,
                "x0": nl // 2,
                "y0": nl // 2,
                "fwhm_major": src_fwhm_maj / cellsize,
                "fwhm_minor": src_fwhm_min / cellsize,
                "theta": 0.0,
            }
        ]
        ds = imfit(xds, n_components=1, initial_guesses=init)

        assert ds["is_unresolved"].values.flat[0]
        assert np.isnan(ds["fwhm_major_deconv"].values.flat[0])
        # Upper limit should equal beam major FWHM
        np.testing.assert_allclose(
            ds["fwhm_upper_limit"].values.flat[0], beam_fwhm, rtol=1e-10
        )

    def test_deconvolved_pixel_sizes_present(self):
        """Deconvolved pixel-frame sizes should be present when resolved."""
        cellsize = 1e-5
        nl = 128
        beam_fwhm = 5 * cellsize
        fwhm_maj = 15 * cellsize
        fwhm_min = 10 * cellsize
        comp = {
            "amp": 1.0,
            "l0": 0.0,
            "m0": 0.0,
            "fwhm_maj": fwhm_maj,
            "fwhm_min": fwhm_min,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[comp],
            noise_sigma=0.005,
            cellsize=cellsize,
            beam_fwhm_maj=beam_fwhm,
            beam_fwhm_min=beam_fwhm,
            beam_pa=0.0,
        )
        init = [
            {
                "amp": 0.9,
                "x0": nl // 2,
                "y0": nl // 2,
                "fwhm_major": fwhm_maj / cellsize,
                "fwhm_minor": fwhm_min / cellsize,
                "theta": 0.0,
            }
        ]
        ds = imfit(xds, n_components=1, initial_guesses=init)

        assert "fwhm_major_deconv_pixel" in ds.data_vars
        assert "fwhm_minor_deconv_pixel" in ds.data_vars


class TestImfitInputValidation:
    """Tests for input validation and edge cases."""

    def test_rejects_world_width_guesses_for_non_square_pixels(self):
        """World-frame width guesses should fail when the image pixels are not square."""
        cellsize_l = 1e-5
        xds = _make_xradio_image(
            nl=128,
            nm=128,
            cellsize=cellsize_l,
            cellsize_m=2e-5,
            components=[
                {
                    "amp": 1.0,
                    "l0": 0.0,
                    "m0": 0.0,
                    "fwhm_maj": 6 * cellsize_l,
                    "fwhm_min": 4 * cellsize_l,
                    "pa": 0.0,
                }
            ],
            noise_sigma=0.01,
        )

        with pytest.raises(ValueError, match="square pixels"):
            imfit(
                xds,
                n_components=1,
                beam_var=None,
                initial_guesses=[
                    {
                        "amp": 0.9,
                        "l0": 0.0,
                        "m0": 0.0,
                        "fwhm_major": 6 * cellsize_l,
                        "fwhm_minor": 4 * cellsize_l,
                        "theta": 0.0,
                    }
                ],
            )

    def test_missing_data_var_raises(self):
        """Missing data variable should raise KeyError."""
        xds = xr.Dataset()
        with pytest.raises(KeyError, match="NONEXISTENT"):
            imfit(xds, n_components=1, data_var="NONEXISTENT", beam_var=None)

    def test_missing_lm_dims_raises(self):
        """Data variable without l/m dims should raise ValueError."""
        xds = xr.Dataset({"SKY": xr.DataArray(np.zeros((10, 10)), dims=("x", "y"))})
        with pytest.raises(ValueError, match="'l' and 'm' dimensions"):
            imfit(xds, n_components=1, beam_var=None)

    def test_missing_mask_warns(self):
        """Missing mask variable should warn, not error."""
        cellsize = 1e-5
        xds = _make_xradio_image(
            components=[
                {
                    "amp": 1.0,
                    "l0": 0,
                    "m0": 0,
                    "fwhm_maj": 5 * cellsize,
                    "fwhm_min": 5 * cellsize,
                    "pa": 0.0,
                }
            ],
            noise_sigma=0.01,
            cellsize=cellsize,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = imfit(xds, n_components=1, mask_var="NONEXISTENT", beam_var=None)
            mask_warnings = [x for x in w if "NONEXISTENT" in str(x.message)]
            assert len(mask_warnings) >= 1

    def test_missing_beam_warns(self):
        """Missing beam variable should warn and skip deconvolution."""
        cellsize = 1e-5
        xds = _make_xradio_image(
            components=[
                {
                    "amp": 1.0,
                    "l0": 0,
                    "m0": 0,
                    "fwhm_maj": 5 * cellsize,
                    "fwhm_min": 5 * cellsize,
                    "pa": 0.0,
                }
            ],
            noise_sigma=0.01,
            cellsize=cellsize,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = imfit(xds, n_components=1, beam_var="NONEXISTENT")
            beam_warnings = [x for x in w if "NONEXISTENT" in str(x.message)]
            assert len(beam_warnings) >= 1
        assert "fwhm_major_deconv" not in ds.data_vars

    def test_beam_var_none_skips_deconv(self):
        """Setting beam_var=None should skip deconvolution entirely."""
        cellsize = 1e-5
        xds = _make_xradio_image(
            components=[
                {
                    "amp": 1.0,
                    "l0": 0,
                    "m0": 0,
                    "fwhm_maj": 5 * cellsize,
                    "fwhm_min": 5 * cellsize,
                    "pa": 0.0,
                }
            ],
            noise_sigma=0.01,
            cellsize=cellsize,
            beam_fwhm_maj=3 * cellsize,
            beam_fwhm_min=3 * cellsize,
            beam_pa=0.0,
        )
        ds = imfit(xds, n_components=1, beam_var=None)
        assert "fwhm_major_deconv" not in ds.data_vars

    def test_invalid_beam_units_raises(self):
        """Non-angular beam units should raise ValueError."""
        cellsize = 1e-5
        xds = _make_xradio_image(
            components=[
                {
                    "amp": 1.0,
                    "l0": 0,
                    "m0": 0,
                    "fwhm_maj": 5 * cellsize,
                    "fwhm_min": 5 * cellsize,
                    "pa": 0.0,
                }
            ],
            noise_sigma=0.01,
            cellsize=cellsize,
            beam_fwhm_maj=3 * cellsize,
            beam_fwhm_min=3 * cellsize,
            beam_pa=0.0,
        )
        xds["BEAM_FIT_PARAMS_SKY"].attrs["units"] = "meters"
        with pytest.raises(ValueError, match="angular unit"):
            imfit(xds, n_components=1)


class TestImfitCoverageGaps:
    """Tests for previously uncovered code paths."""

    def test_mask_found_and_used(self):
        """When a valid mask variable exists, _resolve_mask returns it (line 104)."""
        cellsize = 1e-5
        nl = 128
        fwhm = 10 * cellsize
        comp = {
            "amp": 1.0,
            "l0": 0.0,
            "m0": 0.0,
            "fwhm_maj": fwhm,
            "fwhm_min": fwhm,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl, nm=nl, components=[comp], noise_sigma=0.01, cellsize=cellsize
        )
        # Add a real mask variable: True everywhere (all pixels good)
        mask = xr.DataArray(
            np.ones_like(xds["SKY"].values, dtype=bool),
            dims=xds["SKY"].dims,
            coords=xds["SKY"].coords,
        )
        xds["FLAGS_SKY"] = mask

        # Should not warn about missing mask
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = imfit(xds, n_components=1, beam_var=None)
            mask_warnings = [x for x in w if "FLAGS_SKY" in str(x.message)]
            assert mask_warnings == [], f"Unexpected mask warning: {mask_warnings}"
        assert ds["success"].values

    def test_mask_var_accepts_boolean_array(self):
        """A boolean array mask should be accepted directly by imfit()."""
        cellsize = 1e-5
        target = {
            "amp": 1.0,
            "l0": -18 * cellsize,
            "m0": 14 * cellsize,
            "fwhm_maj": 7 * cellsize,
            "fwhm_min": 5 * cellsize,
            "pa": 0.0,
        }
        distractor = {
            "amp": 1.6,
            "l0": 20 * cellsize,
            "m0": -16 * cellsize,
            "fwhm_maj": 8 * cellsize,
            "fwhm_min": 6 * cellsize,
            "pa": 0.3,
        }
        xds = _make_xradio_image(
            nl=128,
            nm=128,
            components=[target, distractor],
            noise_sigma=0.0,
            cellsize=cellsize,
        )

        l = xds.coords["l"].values
        m = xds.coords["m"].values
        region_2d = (
            (l[:, None] >= -26 * cellsize) & (l[:, None] <= -10 * cellsize)
        ) & ((m[None, :] >= 8 * cellsize) & (m[None, :] <= 20 * cellsize))
        mask = np.broadcast_to(region_2d, xds["SKY"].shape)

        ds = imfit(xds, n_components=1, mask_var=mask, beam_var=None)

        assert ds["success"].values
        np.testing.assert_allclose(ds["x0_world"].values, target["l0"], atol=cellsize)
        np.testing.assert_allclose(ds["y0_world"].values, target["m0"], atol=cellsize)

    def test_mask_var_accepts_crtf_string(self):
        """A CRTF string mask should be forwarded to the selection layer."""
        cellsize = 1e-5
        target = {
            "amp": 1.0,
            "l0": -18 * cellsize,
            "m0": 14 * cellsize,
            "fwhm_maj": 7 * cellsize,
            "fwhm_min": 5 * cellsize,
            "pa": 0.0,
        }
        distractor = {
            "amp": 1.6,
            "l0": 20 * cellsize,
            "m0": -16 * cellsize,
            "fwhm_maj": 8 * cellsize,
            "fwhm_min": 6 * cellsize,
            "pa": 0.3,
        }
        xds = _make_xradio_image(
            nl=128,
            nm=128,
            components=[target, distractor],
            noise_sigma=0.0,
            cellsize=cellsize,
        )

        crtf = (
            "#CRTF\n"
            "global coordsys=lm\n"
            "circle[[-0.00018rad,0.00014rad],0.00008rad]"
        )
        ds = imfit(xds, n_components=1, mask_var=crtf, beam_var=None)

        assert ds["success"].values
        np.testing.assert_allclose(ds["x0_world"].values, target["l0"], atol=cellsize)
        np.testing.assert_allclose(ds["y0_world"].values, target["m0"], atol=cellsize)

    def test_mask_var_accepts_pixel_crtf_string(self):
        """A pixel-space CRTF string should honor the public (x, y) convention."""
        cellsize = 1e-5
        nl = 128
        target = {
            "amp": 1.0,
            "l0": -18 * cellsize,
            "m0": 14 * cellsize,
            "fwhm_maj": 7 * cellsize,
            "fwhm_min": 5 * cellsize,
            "pa": 0.0,
        }
        distractor = {
            "amp": 1.6,
            "l0": 20 * cellsize,
            "m0": -16 * cellsize,
            "fwhm_maj": 8 * cellsize,
            "fwhm_min": 6 * cellsize,
            "pa": 0.3,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[target, distractor],
            noise_sigma=0.0,
            cellsize=cellsize,
        )

        target_x = nl // 2 - int(round(target["l0"] / cellsize))
        target_y = nl // 2 + int(round(target["m0"] / cellsize))
        crtf = (
            "#CRTF\n"
            f"box[[{target_x - 6}pix,{target_y - 6}pix],"
            f"[{target_x + 6}pix,{target_y + 6}pix]]"
        )
        ds = imfit(xds, n_components=1, mask_var=crtf, beam_var=None)

        assert ds["success"].values
        np.testing.assert_allclose(ds["x0_world"].values, target["l0"], atol=cellsize)
        np.testing.assert_allclose(ds["y0_world"].values, target["m0"], atol=cellsize)

    def test_mask_var_accepts_crtf_file_name(self, tmp_path):
        """A CRTF filename should be read and applied as the mask definition."""
        cellsize = 1e-5
        nl = 128
        target = {
            "amp": 1.0,
            "l0": -18 * cellsize,
            "m0": 14 * cellsize,
            "fwhm_maj": 7 * cellsize,
            "fwhm_min": 5 * cellsize,
            "pa": 0.0,
        }
        distractor = {
            "amp": 1.6,
            "l0": 20 * cellsize,
            "m0": -16 * cellsize,
            "fwhm_maj": 8 * cellsize,
            "fwhm_min": 6 * cellsize,
            "pa": 0.3,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[target, distractor],
            noise_sigma=0.0,
            cellsize=cellsize,
        )

        target_x = nl // 2 - int(round(target["l0"] / cellsize))
        target_y = nl // 2 + int(round(target["m0"] / cellsize))
        crtf_path = tmp_path / "mask.crtf"
        crtf_path.write_text(
            "#CRTF\n"
            f"box[[{target_x - 6}pix,{target_y - 6}pix],"
            f"[{target_x + 6}pix,{target_y + 6}pix]]\n",
            encoding="utf-8",
        )

        ds = imfit(xds, n_components=1, mask_var=str(crtf_path), beam_var=None)

        assert ds["success"].values
        np.testing.assert_allclose(ds["x0_world"].values, target["l0"], atol=cellsize)
        np.testing.assert_allclose(ds["y0_world"].values, target["m0"], atol=cellsize)

    def test_no_sky_coords_without_reference_direction(self):
        """When no grids and no reference_direction exist, warn and skip (lines 368-373)."""
        cellsize = 1e-5
        nl = 128
        fwhm = 10 * cellsize
        comp = {
            "amp": 1.0,
            "l0": 0.0,
            "m0": 0.0,
            "fwhm_maj": fwhm,
            "fwhm_min": fwhm,
            "pa": 0.0,
        }
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            components=[comp],
            noise_sigma=0.01,
            cellsize=cellsize,
            add_sky_coord_grids=False,
        )
        # Remove coordinate_system_info so sky coords can't be computed
        xds.attrs["coordinate_system_info"] = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = imfit(xds, n_components=1, beam_var=None)
            sky_warnings = [
                x for x in w if "sky coordinates cannot be computed" in str(x.message)
            ]
            assert len(sky_warnings) >= 1
        # World l/m should still exist, but RA/Dec should not
        assert "x0_world" in ds.data_vars
        assert "Right Ascension" not in ds.data_vars
        assert "right_ascension" not in ds.data_vars

    def test_deconvolution_without_errors(self):
        """Exercise the no-error deconvolution path (lines 599-614).

        The generic fitter always produces error variables, so we call
        ``_deconvolve_and_attach`` directly on a Dataset that lacks them.
        """
        from astroviper.distributed_applications.image_analysis.imfit import (
            _deconvolve_and_attach,
        )

        # Build a minimal result Dataset that mimics fitter output
        # but without error variables.
        src_fwhm_maj = 1.5e-4  # larger than beam
        src_fwhm_min = 1.0e-4
        src_pa = 0.3
        beam_fwhm = 5e-5

        ds = xr.Dataset(
            {
                "fwhm_major_world": xr.DataArray(
                    np.array([[[[src_fwhm_maj]]]]),
                    dims=("time", "frequency", "polarization", "component"),
                ),
                "fwhm_minor_world": xr.DataArray(
                    np.array([[[[src_fwhm_min]]]]),
                    dims=("time", "frequency", "polarization", "component"),
                ),
                "pa": xr.DataArray(
                    np.array([[[[src_pa]]]]),
                    dims=("time", "frequency", "polarization", "component"),
                ),
                "sigma_major_world": xr.DataArray(
                    np.array([[[[src_fwhm_maj * FWHM2SIG]]]]),
                    dims=("time", "frequency", "polarization", "component"),
                ),
                "sigma_minor_world": xr.DataArray(
                    np.array([[[[src_fwhm_min * FWHM2SIG]]]]),
                    dims=("time", "frequency", "polarization", "component"),
                ),
                "sigma_major_pixel": xr.DataArray(
                    np.array([[[[src_fwhm_maj * FWHM2SIG / 1e-5]]]]),
                    dims=("time", "frequency", "polarization", "component"),
                ),
                "sigma_minor_pixel": xr.DataArray(
                    np.array([[[[src_fwhm_min * FWHM2SIG / 1e-5]]]]),
                    dims=("time", "frequency", "polarization", "component"),
                ),
            }
        )
        # No *_err variables — this forces the no-error branch

        bmaj = xr.DataArray(
            np.array([[[beam_fwhm]]]), dims=("time", "frequency", "polarization")
        )
        bmin = xr.DataArray(
            np.array([[[beam_fwhm]]]), dims=("time", "frequency", "polarization")
        )
        bpa = xr.DataArray(
            np.array([[[0.0]]]), dims=("time", "frequency", "polarization")
        )

        ds = _deconvolve_and_attach(ds, (bmaj, bmin, bpa))

        assert "fwhm_major_deconv" in ds.data_vars
        assert "is_unresolved" in ds.data_vars
        assert not ds["is_unresolved"].values.flat[0]
        # No error variables should be present
        assert "fwhm_major_deconv_err" not in ds.data_vars
        assert "is_marginally_resolved" not in ds.data_vars
        # Deconvolved sizes should be smaller than source sizes
        assert ds["fwhm_major_deconv"].values.flat[0] < src_fwhm_maj
        assert ds["fwhm_minor_deconv"].values.flat[0] < src_fwhm_min

    def test_deconvolution_with_errors_dask_flags_are_bool(self):
        """Dask deconvolution with errors should preserve boolean flag dtypes."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _deconvolve_and_attach,
        )

        dims = ("time", "frequency", "polarization", "component")

        def _data_array(value):
            data = da.from_array(
                np.array([[[[value]]]], dtype=float), chunks=(1, 1, 1, 1)
            )
            return xr.DataArray(data, dims=dims)

        ds = xr.Dataset(
            {
                "fwhm_major_world": _data_array(1.5e-4),
                "fwhm_minor_world": _data_array(1.0e-4),
                "pa": _data_array(0.3),
                "fwhm_major_world_err": _data_array(1.0e-6),
                "fwhm_minor_world_err": _data_array(1.0e-6),
                "pa_err": _data_array(1.0e-3),
            }
        )
        beam_dims = ("time", "frequency", "polarization")
        bmaj = xr.DataArray(
            da.from_array(np.array([[[5.0e-5]]], dtype=float), chunks=(1, 1, 1)),
            dims=beam_dims,
        )
        bmin = xr.DataArray(
            da.from_array(np.array([[[5.0e-5]]], dtype=float), chunks=(1, 1, 1)),
            dims=beam_dims,
        )
        bpa = xr.DataArray(
            da.from_array(np.array([[[0.0]]], dtype=float), chunks=(1, 1, 1)),
            dims=beam_dims,
        )

        out = _deconvolve_and_attach(ds, (bmaj, bmin, bpa))

        assert out["is_unresolved"].dtype == np.dtype(bool)
        assert out["is_marginally_resolved"].dtype == np.dtype(bool)


class TestImfitHelperCoverage:
    """Targeted helper coverage for supported alternate input shapes and edges."""

    def test_normalize_initial_guesses_preserves_array_form(self):
        """Array-form guesses should bypass imfit-specific dict conversion."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _normalize_imfit_initial_guesses,
        )

        xds = _make_xradio_image(components=[], noise_sigma=0.0)
        initial_guesses = np.array([[1.0, 2.0, 3.0]])

        normalized = _normalize_imfit_initial_guesses(xds, initial_guesses)

        assert normalized is initial_guesses

    def test_normalize_initial_guesses_converts_wrapped_component_mapping(self):
        """Wrapped mapping guesses should convert component dicts and preserve offset."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _normalize_imfit_initial_guesses,
        )

        cellsize = 1e-5
        nl = 128
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            cellsize=cellsize,
            components=[],
            noise_sigma=0.0,
        )
        initial_guesses = {
            "offset": 0.25,
            "components": [
                {
                    "amp": 0.9,
                    "l0": 2 * cellsize,
                    "m0": -1 * cellsize,
                    "fwhm_major": 6 * cellsize,
                    "fwhm_minor": 4 * cellsize,
                    "theta": 0.0,
                }
            ],
        }

        normalized = _normalize_imfit_initial_guesses(xds, initial_guesses)

        assert normalized["offset"] == 0.25
        comp = normalized["components"][0]
        np.testing.assert_allclose(comp["x0"], nl // 2 - 2, atol=1e-12)
        np.testing.assert_allclose(comp["y0"], nl // 2 - 1, atol=1e-12)
        np.testing.assert_allclose(comp["fwhm_major"], 6.0, atol=1e-12)
        np.testing.assert_allclose(comp["fwhm_minor"], 4.0, atol=1e-12)

    def test_normalize_initial_guesses_converts_string_xy_sky_centers(self):
        """String-valued x0/y0 should be treated as sky coordinates, not pixels."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _normalize_imfit_initial_guesses,
        )

        xds = _make_xradio_image(components=[], noise_sigma=0.0)
        initial_guesses = {
            "amp": 0.9,
            "x0": Angle(1.0, unit="rad").to_string(unit="hourangle", sep=":"),
            "y0": Angle(0.5, unit="rad").to_string(unit="deg", sep=":"),
        }

        normalized = _normalize_imfit_initial_guesses(xds, initial_guesses)

        np.testing.assert_allclose(normalized["x0"], 64.0, atol=1e-12)
        np.testing.assert_allclose(normalized["y0"], 64.0, atol=1e-12)

    @pytest.mark.parametrize(
        "initial_guess",
        [
            {"amp": 0.9, "ra": 1.0},
            {"amp": 0.9, "dec": 0.5},
            {"amp": 0.9, "right_ascension": 1.0},
            {"amp": 0.9, "latitude": 0.5},
        ],
    )
    def test_normalize_initial_guesses_rejects_incomplete_sky_key_pairs(
        self, initial_guess
    ):
        """Sky-key initial guesses should require paired longitude/latitude values."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _normalize_imfit_initial_guesses,
        )

        xds = _make_xradio_image(components=[], noise_sigma=0.0)

        with pytest.raises(
            ValueError, match="must include both longitude and latitude"
        ):
            _normalize_imfit_initial_guesses(xds, initial_guess)

    @pytest.mark.parametrize(
        "initial_guess",
        [
            {
                "amp": 0.9,
                "x0": Angle(1.0, unit="rad").to_string(unit="hourangle", sep=":"),
            },
            {
                "amp": 0.9,
                "y0": Angle(0.5, unit="rad").to_string(unit="deg", sep=":"),
            },
        ],
    )
    def test_normalize_initial_guesses_rejects_incomplete_xy_sky_pairs(
        self, initial_guess
    ):
        """String-valued x0/y0 sky guesses should require paired values."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _normalize_imfit_initial_guesses,
        )

        xds = _make_xradio_image(components=[], noise_sigma=0.0)

        with pytest.raises(ValueError, match="must provide both x0 and y0"):
            _normalize_imfit_initial_guesses(xds, initial_guess)

    def test_normalize_initial_guesses_rejects_sky_guesses_without_reference_direction(
        self,
    ):
        """Sky-coordinate guesses require a phase center in coordinate metadata."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _normalize_imfit_initial_guesses,
        )

        xds = _make_xradio_image(components=[], noise_sigma=0.0)
        xds.attrs["coordinate_system_info"] = {}

        with pytest.raises(ValueError, match="reference_direction"):
            _normalize_imfit_initial_guesses(
                xds,
                {
                    "amp": 0.9,
                    "x0": Angle(1.0, unit="rad").to_string(unit="hourangle", sep=":"),
                    "y0": Angle(0.5, unit="rad").to_string(unit="deg", sep=":"),
                },
            )

    def test_normalize_initial_guesses_accepts_world_centers_without_widths(self):
        """Center-only world guesses should still be converted into pixel centers."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _normalize_imfit_initial_guesses,
        )

        cellsize = 1e-5
        nl = 128
        xds = _make_xradio_image(
            nl=nl,
            nm=nl,
            cellsize=cellsize,
            components=[],
            noise_sigma=0.0,
        )

        normalized = _normalize_imfit_initial_guesses(
            xds,
            {"amp": 0.9, "l0": 3 * cellsize, "m0": -2 * cellsize},
        )

        np.testing.assert_allclose(normalized["x0"], nl // 2 - 3, atol=1e-12)
        np.testing.assert_allclose(normalized["y0"], nl // 2 - 2, atol=1e-12)
        assert "l0" not in normalized
        assert "m0" not in normalized

    @pytest.mark.parametrize(
        "initial_guess",
        [
            {"amp": 0.9, "l0": 0.0},
            {"amp": 0.9, "m0": 0.0},
        ],
    )
    def test_normalize_initial_guesses_rejects_incomplete_lm_centers(
        self, initial_guess
    ):
        """Native world-center guesses should require paired l0/m0 values."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _normalize_imfit_initial_guesses,
        )

        xds = _make_xradio_image(components=[], noise_sigma=0.0)

        with pytest.raises(ValueError, match="require both 'l0' and 'm0'"):
            _normalize_imfit_initial_guesses(xds, initial_guess)

    def test_resolve_mask_none_returns_none(self):
        """Explicitly disabling masks should short-circuit without warnings."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _resolve_mask,
        )

        xds = _make_xradio_image(components=[], noise_sigma=0.0)

        assert _resolve_mask(xds, None) is None

    def test_lm_to_radec_from_wcs_warns_for_non_sin_projection(self):
        """Unsupported projections should warn while falling back to the SIN inverse."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _lm_to_radec_from_wcs,
        )

        l_vals = xr.DataArray([0.0], dims=("component",))
        m_vals = xr.DataArray([0.0], dims=("component",))

        with pytest.warns(UserWarning, match="falling back to SIN projection"):
            ra, dec, attrs = _lm_to_radec_from_wcs(
                l_vals,
                m_vals,
                phase_center=(1.0, 0.5),
                frame="icrs",
                projection="TAN",
            )

        np.testing.assert_allclose(ra.values, [1.0], atol=1e-12)
        np.testing.assert_allclose(dec.values, [0.5], atol=1e-12)
        assert attrs["frame"] == "icrs"

    def test_attach_sky_coordinates_returns_input_when_world_centers_absent(self):
        """Sky-coordinate attachment should no-op when l/m centers are unavailable."""
        from astroviper.distributed_applications.image_analysis.imfit import (
            _attach_sky_coordinates,
        )

        xds = _make_xradio_image(components=[], noise_sigma=0.0)
        ds = xr.Dataset({"success": xr.DataArray(np.array(True))})

        out = _attach_sky_coordinates(ds, xds)

        assert out is ds


class TestImfitMetadata:
    """Tests for metadata propagation."""

    def test_coordinate_system_info_propagated(self):
        """coordinate_system_info should be copied to result."""
        cellsize = 1e-5
        xds = _make_xradio_image(
            components=[
                {
                    "amp": 1.0,
                    "l0": 0,
                    "m0": 0,
                    "fwhm_maj": 5 * cellsize,
                    "fwhm_min": 5 * cellsize,
                    "pa": 0.0,
                }
            ],
            noise_sigma=0.01,
            cellsize=cellsize,
        )
        ds = imfit(xds, n_components=1, beam_var=None)
        assert "coordinate_system_info" in ds.attrs
        assert ds.attrs["theta_convention"] == "pa"
        assert "pa_definition" in ds.attrs

    def test_return_model_and_residual(self):
        """Model and residual planes should be optionally included."""
        cellsize = 1e-5
        xds = _make_xradio_image(
            components=[
                {
                    "amp": 1.0,
                    "l0": 0,
                    "m0": 0,
                    "fwhm_maj": 5 * cellsize,
                    "fwhm_min": 5 * cellsize,
                    "pa": 0.0,
                }
            ],
            noise_sigma=0.01,
            cellsize=cellsize,
        )
        ds = imfit(
            xds,
            n_components=1,
            beam_var=None,
            return_model=True,
            return_residual=True,
        )
        assert "model" in ds.data_vars
        assert "residual" in ds.data_vars
