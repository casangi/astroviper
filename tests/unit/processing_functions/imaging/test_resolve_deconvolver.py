import numpy as np
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    model_update_mtmfs_single_field,
)


def _continuum_image(shape=(12, 12)):
    yy, xx = np.indices(shape)
    psf = np.exp(-((xx - shape[1] // 2) ** 2 + (yy - shape[0] // 2) ** 2) / 2)
    residual = np.zeros(shape)
    residual[shape[0] // 2, shape[1] // 2] = 1.0
    coords = {
        "time": [0.0],
        "taylor_term": [0],
        "psf_taylor_order": [0],
        "polarization": ["I"],
        "l": np.arange(shape[0], dtype=float),
        "m": np.arange(shape[1], dtype=float),
    }
    xds = xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                residual[None, None, None],
            ),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                psf[None, None, None],
            ),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "polarization"),
                np.asarray([[0.1]]),
            ),
            "VISIBILITY_NORMALIZATION": (
                ("time", "taylor_term", "polarization"),
                np.asarray([[[100.0]]]),
            ),
            "PRIMARY_BEAM": (
                ("time", "frequency", "polarization", "l", "m"),
                np.ones((1, 1, 1, *shape)),
            ),
        },
        coords=coords,
        attrs={
            "reference_frequency": 1.0e9,
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "visibility_normalization": "VISIBILITY_NORMALIZATION",
                    "primary_beam": "PRIMARY_BEAM",
                }
            },
        },
    )
    return xds


def test_resolve_continuum_backend_returns_increment_and_state():
    image = _continuum_image()
    stats, timing, state = model_update_mtmfs_single_field(
        image,
        "resolve",
        {
            "threshold": 0.0,
            "resolve": {
                "n_vi_iterations": 0,
                "prior_mean": 0.1,
                "seed": 7,
            },
        },
    )

    assert state.major_cycle == 1
    assert "T_resolve" in timing
    assert "SKY_MODEL" in image
    assert "SKY_POSTERIOR_STD" in image
    assert "SKY_POSTERIOR_MEAN" in image
    assert np.isfinite(image["SKY_MODEL"]).all()
    np.testing.assert_allclose(image["SKY_MODEL"], image["SKY_POSTERIOR_MEAN"])
    assert image.attrs["resolve_deconvolution"]["backend"] == "resolve"
    assert image.attrs["resolve_deconvolution"]["mask_policy"].startswith("freeze")
    assert image.attrs["resolve_deconvolution"]["noise_covariance_scale"] == 0.01
    assert image.attrs["resolve_deconvolution"]["noise_kernel"]["mode"] == "fitted"
    assert (
        image.attrs["resolve_deconvolution"]["noise_kernel"]["minimum_eigenvalue"] > 0
    )
    assert stats.sel(time=0, pol=0, chan=0)["iter_done"] == [0]


def test_resolve_continuum_backend_allows_clipped_noise_kernel():
    image = _continuum_image()

    model_update_mtmfs_single_field(
        image,
        "resolve",
        {
            "threshold": 0.0,
            "resolve": {
                "n_vi_iterations": 0,
                "prior_mean": 0.1,
                "noise_kernel_mode": "clipped",
            },
        },
    )

    assert image.attrs["resolve_deconvolution"]["noise_kernel"]["mode"] == "clipped"


def test_resolve_continuum_backend_constructs_primary_beam_mask():
    image = _continuum_image()
    image["PRIMARY_BEAM"].data[..., :3, :] = 0.1

    model_update_mtmfs_single_field(
        image,
        "resolve",
        {
            "threshold": 0.0,
            "primary_beam_limit": 0.2,
            "resolve": {"n_vi_iterations": 0, "prior_mean": 0.1, "seed": 7},
        },
    )

    mask_name = image.attrs["data_groups"]["residual"]["mask"]
    assert mask_name in image
    assert not np.any(image[mask_name].values[..., :3, :])
    assert not np.any(image["SKY_MODEL"].values[..., :3, :])


def test_resolve_continuum_backend_reuses_first_automatic_noise_scale():
    from resolve.re import ResolveDeconvolver

    ResolveDeconvolver.clear_convolution_cache()
    first_image = _continuum_image()
    _, _, first_state = model_update_mtmfs_single_field(
        first_image,
        "resolve",
        {
            "threshold": 0.0,
            "resolve": {"n_vi_iterations": 0, "prior_mean": 0.1},
        },
    )

    second_image = _continuum_image()
    second_image["VISIBILITY_NORMALIZATION"].data[...] = 50.0
    _, timing, second_state = model_update_mtmfs_single_field(
        second_image,
        "resolve",
        {
            "threshold": 0.0,
            "resolve": {"n_vi_iterations": 0, "prior_mean": 0.1},
        },
        previous_model_xds=first_image,
        deconvolver_state=first_state,
    )

    assert first_state.noise_covariance_scale == 0.01
    assert second_state.noise_covariance_scale == 0.01
    assert second_image.attrs["resolve_deconvolution"]["noise_covariance_scale"] == 0.01
    assert (
        second_image.attrs["resolve_deconvolution"]["diagnostics"]["operator_changed"]
        is False
    )
    assert bool(timing["resolve_operator_cache_hit"].iloc[0]) is True


def test_resolve_continuum_backend_uses_registered_mvc_weight_sum():
    image = _continuum_image().drop_vars("VISIBILITY_NORMALIZATION")
    image["MVC_RESIDUAL_WEIGHT_SUM"] = (
        ("time", "polarization"),
        np.asarray([[250.0]]),
    )
    image.attrs["data_groups"]["residual"]["visibility_normalization"] = (
        "MVC_RESIDUAL_WEIGHT_SUM"
    )

    _, _, state = model_update_mtmfs_single_field(
        image,
        "resolve",
        {
            "threshold": 0.0,
            "resolve": {"n_vi_iterations": 0, "prior_mean": 0.1},
        },
    )

    assert state.noise_covariance_scale == 1.0 / 250.0
    assert image.attrs["resolve_deconvolution"]["noise_covariance_scale"] == (
        1.0 / 250.0
    )


def test_resolve_continuum_backend_rejects_missing_automatic_noise_scale():
    image = _continuum_image().drop_vars("VISIBILITY_NORMALIZATION")

    with np.testing.assert_raises_regex(
        ValueError,
        "requires a positive image-noise normalization",
    ):
        model_update_mtmfs_single_field(
            image,
            "resolve",
            {
                "threshold": 0.0,
                "resolve": {"n_vi_iterations": 0, "prior_mean": 0.1},
            },
        )


def test_resolve_continuum_backend_accepts_explicit_noise_scale_without_weights():
    image = _continuum_image().drop_vars("VISIBILITY_NORMALIZATION")

    _, _, state = model_update_mtmfs_single_field(
        image,
        "resolve",
        {
            "threshold": 0.0,
            "resolve": {
                "n_vi_iterations": 0,
                "prior_mean": 0.1,
                "noise_covariance_scale": 0.25,
            },
        },
    )

    assert state.noise_covariance_scale == 0.25


def test_resolve_continuum_backend_rejects_multiple_taylor_terms():
    image = _continuum_image()
    residual = image["SKY_RESIDUAL"]
    image = image.drop_dims("taylor_term")
    image["SKY_RESIDUAL"] = xr.concat(
        [residual, residual], dim="taylor_term"
    ).assign_coords(taylor_term=[0, 1])

    with np.testing.assert_raises_regex(NotImplementedError, "requires nterms=1"):
        model_update_mtmfs_single_field(
            image,
            "resolve",
            {"resolve": {"n_vi_iterations": 0}},
        )
