"""
``exact_psf_subtraction``: the Hogbom model updates subtract every component
from the whole residual image with a PSF of twice the image size, made once
per channel from the imaging weights of the image.

Tested on the PSF maker (shape, centre, natural and Briggs weights, cell, one
uv grid for imaging weights equal in every correlation) and on the 48 x 48
imaging cycle loop of ``test_max_cycles_auto``: the default is
unchanged, the option gives to the bit what the image-sized kernel gives on a
2n image holding the residual, with a 2n PSF made independently, and the 2n
PSF is freed before the restore.
"""

import importlib
import pickle

import numpy as np
import pytest
from xradio.image import make_empty_sky_image

from astroviper.processing_functions.imaging.utils.imaging_dict import (
    CYCLE_TIMING_FIELDS,
)
from tests.unit.processing_functions.imaging.test_max_cycles_auto import (
    FREQUENCY,
    IMAGE_PARAMS,
    _processing_set,
)

try:
    from astroviper.processing_functions.imaging.deconvolvers import hogbom

    HOGBOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    HOGBOM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
)

CONTROLS = {
    "max_iter": 2000,
    "max_cycles": 4,
    "threshold": 1e-4,
    "primary_beam_limit": 0.2,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.0,
    "max_iter_per_cycle": -1,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}


def _empty_image(n):
    return make_empty_sky_image(
        phase_center=IMAGE_PARAMS["phase_direction"],
        image_size=[n, n],
        cell_size=IMAGE_PARAMS["cell_size"],
        frequency_coords=[FREQUENCY],
        pol_coords=["XX", "YY"],
        time_coords=[0],
        do_sky_coords=False,
    )


def _run(controls, single_precision_image=False, deconvolver="hogbom", **kwargs):
    from astroviper.processing_functions.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    return image_cube_single_field(
        _processing_set(),
        _empty_image(IMAGE_PARAMS["image_size"][0]),
        IMAGE_PARAMS,
        {"weighting": "natural"},
        {**CONTROLS, **controls},
        processing_set_data_group_name="base",
        deconvolver=deconvolver,
        single_precision_image=single_precision_image,
        fft_backend="scipy",
        image_data_variables_keep=["sky_residual", "sky_model"],
        **kwargs,
    )


def _untimed(record):
    """The record without the wall times of its cycles."""
    return {
        key: {f: v for f, v in fields.items() if f not in CYCLE_TIMING_FIELDS}
        for key, fields in record.data.items()
    }


def _same(a, b, extra=()):
    """Residual, model, record and timing columns (times excepted) to the bit;
    ``a`` may have the timing columns ``extra`` besides."""
    (img_a, timing_a, record_a), (img_b, timing_b, record_b) = a, b
    for name in ("SKY_RESIDUAL", "SKY_MODEL"):
        assert np.array_equal(img_a[name].values, img_b[name].values)
    assert pickle.dumps(_untimed(record_a)) == pickle.dumps(_untimed(record_b))
    assert [c for c in timing_a.columns if c not in extra] == list(timing_b.columns)
    counts = [c for c in timing_b.columns if not c.startswith("T_")]
    assert timing_a[counts].equals(timing_b[counts])


# ---------------------------------------------------------------------------
# The PSF of twice the image size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [48, 47])
@pytest.mark.parametrize("single_precision_image", [False, True])
@pytest.mark.parametrize(
    "imaging_weights_params",
    [{"weighting": "natural"}, {"weighting": "briggs", "robust": 0.0}],
    ids=["natural", "briggs"],
)
def test_the_psf_of_twice_the_image_size(
    monkeypatch, n, single_precision_image, imaging_weights_params
):
    import astroviper.processing_functions.imaging.make_point_spread_function as mpsf
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )
    from astroviper.processing_functions.imaging.residual_update import (
        imaging_setup_single_field,
    )

    image_params = {**IMAGE_PARAMS, "image_size": [n, n]}
    ps_xdt = _processing_set()
    img_xds, _ = imaging_setup_single_field(
        ps_xdt,
        _empty_image(n),
        image_params,
        imaging_weights_params,
        processing_set_data_group_name="base",
        single_precision_image=single_precision_image,
        fft_backend="scipy",
    )
    weights = ps_xdt["ms_0"]["WEIGHT_IMAGING"].values.copy()
    before = img_xds.copy(deep=True)
    grids = []
    original = mpsf.make_point_spread_function_single_field

    def spy(ps, psf_xds, params, **kwargs):
        grids.append((dict(psf_xds.sizes), psf_xds.xr_img.get_lm_cell_size(), params))
        return original(ps, psf_xds, params, **kwargs)

    monkeypatch.setattr(mpsf, "make_point_spread_function_single_field", spy)
    psf, timing = mpsf.make_subtraction_point_spread_function_single_field(
        ps_xdt,
        img_xds,
        image_params,
        ["XX", "YY"],
        ms_data_group_in_name="base",
        fft_backend="scipy",
        complex_dtype=np.complex64 if single_precision_image else np.complex128,
    )

    assert psf.shape == (1, 1, 2, 2 * n, 2 * n)
    assert psf.dtype == (np.float32 if single_precision_image else np.float64)
    assert psf.flags.c_contiguous
    ((sizes, cell, params),) = grids
    assert (sizes["l"], sizes["m"]) == (2 * n, 2 * n)
    # the imaging weights are equal in both correlations: one is gridded
    assert sizes["polarization"] == 1
    np.testing.assert_allclose(cell, img_xds.xr_img.get_lm_cell_size(), rtol=1e-12)
    assert params["fft_padding"] == image_params["fft_padding"]
    assert "T_uv_sampling_grid" in timing.columns
    # the PSF maker on a 2n image of the correlation basis gives the same bits
    reference = _empty_image(2 * n)
    reference.attrs["type"] = "image_dataset"
    reference = reference.xr_img.add_data_group(
        new_data_group_name="residual",
        new_data_group={"description": "test", "date": "2026"},
    )
    reference, _ = original(
        ps_xdt,
        reference,
        {**image_params, "image_size": [2 * n, 2 * n]},
        ms_data_group_in_name="base",
        fft_backend="scipy",
        complex_dtype=np.complex64 if single_precision_image else np.complex128,
    )
    assert np.array_equal(psf, reference["POINT_SPREAD_FUNCTION"].values)
    # The central crop is the image PSF up to the gridding errors of the two,
    # mostly the image PSF's (measured against a direct Fourier sum of the
    # weights): the lookup of the nearest kernel sample, which grows with the
    # distance from the centre, and toward the edge of the image the aliasing
    # of its padded grid, 6e-4 (3e-5 with fft_padding 2). Here 1.0e-4 within
    # 0.1 n and 1.07e-3 at the edge, 1.3e-4 and 1.16e-3 with Briggs weights.
    image_psf = img_xds["POINT_SPREAD_FUNCTION"].values
    start = n - n // 2
    yy, xx = np.mgrid[:n, :n] - n // 2
    main_lobe = np.maximum(np.abs(yy), np.abs(xx)) <= 0.1 * n
    for p in range(2):
        plane = psf[0, 0, p]
        assert np.unravel_index(np.argmax(plane), plane.shape) == (n, n)
        assert plane[n, n] == pytest.approx(1.0, abs=1e-5)
        diff = np.abs(plane[start : start + n, start : start + n] - image_psf[0, 0, p])
        assert diff[main_lobe].max() < 2e-4
        assert diff.max() < 2e-3
    # the weights are read, not recomputed, and the image is not touched
    assert np.array_equal(ps_xdt["ms_0"]["WEIGHT_IMAGING"].values, weights)
    assert img_xds.identical(before)
    if imaging_weights_params["weighting"] == "briggs":
        # which tells: Briggs weights made for the 2n image (half the uv cell)
        # differ from these
        recomputed = _processing_set()
        calculate_imaging_weights(
            recomputed,
            _empty_image(2 * n),
            imaging_weights_params,
            ms_data_group_in_name="base",
            ms_data_group_out_name="base",
        )
        assert not np.allclose(
            recomputed["ms_0"]["WEIGHT_IMAGING"].values, weights, rtol=0.1
        )


@pytest.mark.parametrize(
    ("weights_changed", "gridded"), [("flagged", 1), ("unequal", 2)]
)
def test_one_uv_grid_when_the_correlation_weights_are_equal(
    monkeypatch, weights_changed, gridded
):
    """Imaging weights equal in every correlation, flagged (``NaN``) samples
    included, give one uv grid, whose PSF is copied to the other correlation;
    unequal weights a grid per correlation. Both give the bits of the PSF
    maker on a 2n image of both correlations."""
    import astroviper.processing_functions.imaging.make_point_spread_function as mpsf
    from astroviper.processing_functions.imaging.residual_update import (
        imaging_setup_single_field,
    )

    n = IMAGE_PARAMS["image_size"][0]
    ps_xdt = _processing_set()
    img_xds, _ = imaging_setup_single_field(
        ps_xdt,
        _empty_image(n),
        IMAGE_PARAMS,
        {"weighting": "natural"},
        processing_set_data_group_name="base",
        fft_backend="scipy",
    )
    weights = ps_xdt["ms_0"]["WEIGHT_IMAGING"].values
    if weights_changed == "flagged":
        weights[::3] = np.nan
    else:
        weights[::2, :, :, 1] *= 0.5
    grids = []
    original = mpsf.add_uv_sampling_grid_single_field

    def spy(ms_xdt, cgk_1D, psf_xds, **kwargs):
        original(ms_xdt, cgk_1D, psf_xds, **kwargs)
        grids.append(psf_xds["UV_SAMPLING"].sizes["polarization"])

    monkeypatch.setattr(mpsf, "add_uv_sampling_grid_single_field", spy)
    # the measurement sets as a dict, as the node task hands them over
    psf, _ = mpsf.make_subtraction_point_spread_function_single_field(
        dict(ps_xdt.items()),
        img_xds,
        IMAGE_PARAMS,
        ["XX", "YY"],
        ms_data_group_in_name="base",
        fft_backend="scipy",
    )
    monkeypatch.undo()

    assert grids == [gridded]
    assert psf.shape == (1, 1, 2, 2 * n, 2 * n)
    assert psf.flags.c_contiguous
    reference = _empty_image(2 * n)
    reference.attrs["type"] = "image_dataset"
    reference = reference.xr_img.add_data_group(
        new_data_group_name="residual",
        new_data_group={"description": "test", "date": "2026"},
    )
    reference, _ = mpsf.make_point_spread_function_single_field(
        ps_xdt,
        reference,
        {**IMAGE_PARAMS, "image_size": [2 * n, 2 * n]},
        ms_data_group_in_name="base",
        fft_backend="scipy",
    )
    assert np.array_equal(psf, reference["POINT_SPREAD_FUNCTION"].values)
    assert np.array_equal(psf[0, 0, 0], psf[0, 0, 1]) == (gridded == 1)


# ---------------------------------------------------------------------------
# The imaging cycle loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("single_precision_image", [False, True])
@pytest.mark.parametrize("deconvolver", ["hogbom", "hogbom_many_threads"])
def test_the_default_is_unchanged(single_precision_image, deconvolver):
    _same(
        _run({}, single_precision_image, deconvolver),
        _run({"exact_psf_subtraction": False}, single_precision_image, deconvolver),
    )


@pytest.mark.parametrize(
    ("deconvolver", "kernel"),
    [("hogbom", "clean_cube"), ("hogbom_many_threads", "clean_cube_many_threads")],
)
def test_equals_the_image_sized_kernel_on_an_embedded_image(
    monkeypatch, deconvolver, kernel
):
    """The option against the window test's emulation: a 2n PSF made right
    after the image PSF, from the correlation-basis image the setup hands the
    PSF maker, and the image-sized kernel on a 2n image holding the residual
    at (n // 2, n // 2), searched only there."""
    exact = _run(
        {"exact_psf_subtraction": True, "record_iteration_history": True},
        deconvolver=deconvolver,
    )

    import astroviper.processing_functions.imaging.make_point_spread_function as mpsf

    made = []
    original_psf = mpsf.make_point_spread_function_single_field
    original_kernel = getattr(hogbom, kernel)

    def psf_maker(ps_xdt, img_xds, image_params, **kwargs):
        pol = img_xds.polarization.values.copy()
        out = original_psf(ps_xdt, img_xds, image_params, **kwargs)
        n = image_params["image_size"][0]
        size = [2 * n, 2 * n]
        psf_xds = make_empty_sky_image(
            phase_center=image_params["phase_direction"],
            image_size=size,
            cell_size=image_params["cell_size"],
            frequency_coords=img_xds.frequency.values,
            pol_coords=pol,
            time_coords=image_params["time_coords"],
            do_sky_coords=False,
        )
        psf_xds.attrs["type"] = "image_dataset"
        psf_xds = psf_xds.xr_img.add_data_group(
            new_data_group_name="residual",
            new_data_group={"description": "test", "date": "2026"},
        )
        psf_xds, _ = original_psf(
            ps_xdt, psf_xds, {**image_params, "image_size": size}, **kwargs
        )
        made.append(psf_xds["POINT_SPREAD_FUNCTION"].values)
        return out

    def embedded(residual_cube, psf_cube, model_cube, mask_cube, clean_box, **kw):
        (psf_2n,) = made
        ny, nx = residual_cube.shape[-2:]
        assert psf_cube.shape[-2:] == (ny, nx)
        oy, ox = ny // 2, nx // 2
        region = (..., slice(oy, oy + ny), slice(ox, ox + nx))
        shape = residual_cube.shape[:3] + (2 * ny, 2 * nx)
        residual_2n = np.zeros(shape, residual_cube.dtype)
        model_2n = np.zeros(shape, residual_cube.dtype)
        mask_2n = np.zeros(shape, bool)
        residual_2n[region] = residual_cube
        model_2n[region] = model_cube
        mask_2n[region] = mask_cube if mask_cube.size else True
        xbeg, xend, ybeg, yend = (
            v if v != -1 else d for v, d in zip(clean_box, (0, nx, 0, ny), strict=True)
        )
        out = original_kernel(
            residual_cube=residual_2n,
            psf_cube=psf_2n,
            model_cube=model_2n,
            mask_cube=mask_2n,
            clean_box=(xbeg + ox, xend + ox, ybeg + oy, yend + oy),
            **kw,
        )
        residual_cube[...] = residual_2n[region]
        model_cube[...] = model_2n[region]
        # the components at their pixels of the image
        for field, offset in (
            ("iteration_component_l", oy),
            ("iteration_component_m", ox),
        ):
            for per_frequency in out[field]:
                for per_polarization in per_frequency:
                    for history in per_polarization:
                        history -= offset
        return out

    monkeypatch.setattr(mpsf, "make_point_spread_function_single_field", psf_maker)
    monkeypatch.setattr(hogbom, kernel, embedded)
    emulated = _run({"record_iteration_history": True}, deconvolver=deconvolver)
    monkeypatch.undo()

    assert len(made) == 1
    _same(exact, emulated, extra=("T_prep_subtraction_psf",))
    assert exact[1]["n_cycles"].iloc[0] > 1
    default = _run({"record_iteration_history": True}, deconvolver=deconvolver)
    assert not np.array_equal(
        default[0]["SKY_RESIDUAL"].values, exact[0]["SKY_RESIDUAL"].values
    )


def test_asp_is_refused_before_any_imaging(monkeypatch):
    pf = importlib.import_module(
        "astroviper.processing_functions.imaging.image_cube_single_field"
    )

    def no_imaging(*args, **kwargs):
        raise AssertionError("imaging started")

    monkeypatch.setattr(pf, "imaging_preparation_single_field", no_imaging)
    with pytest.raises(ValueError, match="exact_psf_subtraction is supported"):
        _run({"exact_psf_subtraction": True}, deconvolver="asp")
    with pytest.raises(ValueError, match="exact_psf_subtraction must be a bool"):
        _run({"exact_psf_subtraction": "yes"})


@pytest.mark.parametrize(
    ("option", "max_iter", "made"), [(True, 2000, 1), (True, 0, 0), (False, 2000, 0)]
)
def test_the_psf_is_made_once_when_a_model_update_can_use_it(
    monkeypatch, option, max_iter, made
):
    import astroviper.processing_functions.imaging.make_point_spread_function as mpsf

    calls = []
    original = mpsf.make_subtraction_point_spread_function_single_field

    def spy(*args, **kwargs):
        calls.append(original(*args, **kwargs))
        return calls[-1]

    monkeypatch.setattr(
        mpsf, "make_subtraction_point_spread_function_single_field", spy
    )
    img_xds, timing, _ = _run({"exact_psf_subtraction": option, "max_iter": max_iter})
    assert len(calls) == made
    assert ("T_prep_subtraction_psf" in timing.columns) == bool(made)
    if made:
        assert timing["T_prep"].iloc[0] >= timing["T_prep_subtraction_psf"].iloc[0] > 0
    n = IMAGE_PARAMS["image_size"][0]
    assert (img_xds.sizes["l"], img_xds.sizes["m"]) == (n, n)


def test_the_psf_is_freed_before_the_restore(monkeypatch):
    import gc
    import weakref

    import astroviper.processing_functions.imaging.make_point_spread_function as mpsf
    import astroviper.processing_functions.imaging.restore as restore

    made, alive = [], []
    original_make = mpsf.make_subtraction_point_spread_function_single_field
    original_restore = restore.restore_image

    def make(*args, **kwargs):
        cube, timing = original_make(*args, **kwargs)
        made.append(weakref.ref(cube))
        return cube, timing

    def restore_image(*args, **kwargs):
        alive.append(made[0]() is not None)
        return original_restore(*args, **kwargs)

    monkeypatch.setattr(
        mpsf, "make_subtraction_point_spread_function_single_field", make
    )
    monkeypatch.setattr(restore, "restore_image", restore_image)
    # reference counting alone frees it
    gc.disable()
    try:
        _run({"exact_psf_subtraction": True}, restore=True)
    finally:
        gc.enable()
    assert len(made) == 1
    assert alive == [False]
