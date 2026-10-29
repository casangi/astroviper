"""Unit tests for the C++ direct Fourier transform extension.

Coverage:
  * cos(2 pi t), sin(2 pi t) of the kernels against NumPy, special angles exact
  * dft_image, dft_image_uv_sampling and dft_predict against a NumPy evaluation of
    their definitions (w term and 1/n included), every output dtype
  * the samples skipped (non-finite uvw, NaN or zero weight, zero weighted
    visibility) and the normalization (sum of the weights of the samples imaged)
  * accumulation over calls; the NaN-skip contract of the prediction
  * results independent of the thread count (bit for bit) and repeatable
  * binding input validation
"""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
    cos_sin_turns,
    dft_image,
    dft_image_uv_sampling,
    dft_predict,
)

C = 299792458.0


def _inputs(
    seed=0,
    n_time=3,
    n_baseline=20,
    n_vis_chan=2,
    n_pol=4,
    n_l=12,
    n_m=10,
    w_term="large",
):
    """Random samples: phases of up to ~100 turns and, with ``w_term="large"``,
    w (n - 1) of up to ~10 turns.

    The imaging evaluates the w term in three ways, which ``w_term`` selects:
    ``"zero"`` (w = 0, the separable phase), ``"small"`` (|2 pi w (n - 1)| below
    2^-7 rad, the small angle polynomials), ``"large"`` (the reduction per pixel)
    and ``"mixed"`` (a third of the samples each).
    """
    rng = np.random.default_rng(seed)
    shape = (n_time, n_baseline, n_vis_chan, n_pol)
    vis = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    weight = rng.uniform(0.1, 2.0, shape)
    uvw = rng.uniform(-200.0, 200.0, (n_time, n_baseline, 3))
    small_w = rng.uniform(-0.01, 0.01, (n_time, n_baseline))  # metres
    if w_term == "zero":
        uvw[..., 2] = 0.0
    elif w_term == "small":
        uvw[..., 2] = small_w
    elif w_term == "mixed":
        uvw[:, 0::3, 2] = 0.0
        uvw[:, 1::3, 2] = small_w[:, 1::3]
    frequency = np.array([1.0e9, 1.3e9])[:n_vis_chan]
    l = (np.arange(n_l) - n_l // 2) * -0.03  # noqa: E741
    m = (np.arange(n_m) - n_m // 2) * 0.004
    return dict(
        vis_data=vis,
        uvw=uvw,
        frequency_coord=frequency,
        frequency_map=np.arange(n_vis_chan, dtype=np.int64),
        time_map=np.zeros(n_time, dtype=np.int64),
        pol_map=np.arange(n_pol, dtype=np.int64),
        weight=weight,
        l=l,
        m=m,
    )


def _phase_terms(uvw_b, frequency, l, m):  # noqa: E741
    """exp(+2 pi i (u l + v m + w (n - 1))) on the image and n, for one sample."""
    big_l, big_m = np.meshgrid(l, m, indexing="ij")
    n = np.sqrt(1.0 - big_l**2 - big_m**2)
    u, v, w = uvw_b * frequency / C
    return np.exp(2j * np.pi * (u * big_l + v * big_m + w * (n - 1.0))), n


def _reference_image(inp, m_time, m_chan, m_pol, uv_sampling=False):
    """n sum W V exp(-i phase) (or n sum W cos(phase)) and sum W, sample by sample."""
    l, m = inp["l"], inp["m"]  # noqa: E741
    image = np.zeros((m_time, m_chan, m_pol, l.size, m.size), dtype=complex)
    normalization = np.zeros((m_time, m_chan, m_pol))
    n_time, n_baseline, n_vis_chan, n_pol = inp["weight"].shape
    for t in range(n_time):
        for b in range(n_baseline):
            uvw_b = inp["uvw"][t, b]
            if not np.all(np.isfinite(uvw_b)):
                continue
            for c in range(n_vis_chan):
                fringe, n = _phase_terms(uvw_b, inp["frequency_coord"][c], l, m)
                for p in range(n_pol):
                    weight = inp["weight"][t, b, c, p]
                    value = (
                        weight if uv_sampling else inp["vis_data"][t, b, c, p] * weight
                    )
                    if np.isnan(value) or value == 0:
                        continue
                    plane = (
                        inp["time_map"][t],
                        inp["frequency_map"][c],
                        inp["pol_map"][p],
                    )
                    term = n * value * np.conj(fringe)
                    image[plane] += term.real if uv_sampling else term
                    normalization[plane] += weight
    return image, normalization


def _reference_predict(model, inp):
    """sum M exp(+i phase) / n over all pixels, sample by sample."""
    l, m = inp["l"], inp["m"]  # noqa: E741
    n_time, n_baseline, n_vis_chan, n_pol = inp["vis_data"].shape
    vis = np.zeros((n_time, n_baseline, n_vis_chan, n_pol), dtype=complex)
    for t in range(n_time):
        for b in range(n_baseline):
            for c in range(n_vis_chan):
                fringe, n = _phase_terms(
                    inp["uvw"][t, b], inp["frequency_coord"][c], l, m
                )
                for p in range(n_pol):
                    plane = model[
                        inp["time_map"][t], inp["frequency_map"][c], inp["pol_map"][p]
                    ]
                    vis[t, b, c, p] = np.sum(plane * fringe / n)
    return vis


def _image_args(inp):
    return [
        inp[k]
        for k in (
            "vis_data",
            "uvw",
            "frequency_coord",
            "frequency_map",
            "time_map",
            "pol_map",
            "weight",
            "l",
            "m",
        )
    ]


def _uv_sampling_args(inp):
    return [
        inp[k]
        for k in (
            "uvw",
            "frequency_coord",
            "frequency_map",
            "time_map",
            "pol_map",
            "weight",
            "l",
            "m",
        )
    ]


def _predict_args(inp):
    return [
        inp[k]
        for k in (
            "uvw",
            "frequency_coord",
            "frequency_map",
            "time_map",
            "pol_map",
            "l",
            "m",
        )
    ]


def _empty(inp, m_chan=2, dtype=np.complex128):
    n_pol = inp["weight"].shape[3]
    shape = (1, m_chan, n_pol, inp["l"].size, inp["m"].size)
    return np.zeros(shape, dtype=dtype), np.zeros(shape[:3])


def test_cos_sin_turns_against_numpy():
    rng = np.random.default_rng(3)
    turns = np.concatenate(
        [rng.uniform(-1e4, 1e4, 200_000), rng.uniform(-1, 1, 100_000)]
    )
    cos_values, sin_values = cos_sin_turns(turns)
    # numpy on the exactly reduced angle: the reference error is a few 1e-16 too
    reduced = turns - np.rint(turns)
    assert np.max(np.abs(cos_values - np.cos(2 * np.pi * reduced))) < 6e-16
    assert np.max(np.abs(sin_values - np.sin(2 * np.pi * reduced))) < 6e-16
    assert np.max(np.abs(cos_values**2 + sin_values**2 - 1.0)) < 1e-15


def test_cos_sin_turns_special_angles_are_exact():
    turns = np.array([0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, 3.0, 1e6 + 0.25])
    cos_values, sin_values = cos_sin_turns(turns)
    np.testing.assert_array_equal(cos_values, [1, 0, -1, 0, 1, 0, -1, 1, 0])
    np.testing.assert_array_equal(sin_values, [0, 1, 0, -1, 0, -1, 0, 0, 1])


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64, np.float64, np.float32])
def test_dft_image_matches_the_definition(dtype):
    inp = _inputs()
    image, normalization = _empty(inp, dtype=dtype)
    dft_image(image, normalization, *_image_args(inp), processing_function_threads=2)
    expected, expected_norm = _reference_image(inp, 1, 2, 4)
    if np.dtype(dtype).kind == "f":
        expected = expected.real
    scale = np.sum(np.abs(inp["vis_data"] * inp["weight"]))
    rtol = 1e-13 if dtype in (np.complex128, np.float64) else 1e-6
    assert np.max(np.abs(image - expected)) < rtol * scale
    np.testing.assert_allclose(normalization, expected_norm, rtol=1e-14)


@pytest.mark.parametrize("w_term", ["zero", "small", "large", "mixed"])
def test_every_evaluation_of_the_w_term_matches_the_definition(w_term):
    """The separable phase (w = 0), the small angle w term and the per pixel
    reduction, over several image tiles (8 rows x 64 columns each)."""
    inp = _inputs(seed=13, n_l=20, n_m=70, w_term=w_term)
    image, normalization = _empty(inp)
    dft_image(image, normalization, *_image_args(inp))
    expected, _ = _reference_image(inp, 1, 2, 4)
    scale = np.sum(np.abs(inp["vis_data"] * inp["weight"]))
    assert np.max(np.abs(image - expected)) < 1e-13 * scale
    psf, psf_norm = _empty(inp, dtype=np.float64)
    dft_image_uv_sampling(psf, psf_norm, *_uv_sampling_args(inp))
    expected, _ = _reference_image(inp, 1, 2, 4, uv_sampling=True)
    assert np.max(np.abs(psf - expected.real)) < 1e-13 * np.sum(inp["weight"])


@pytest.mark.parametrize(
    "uvw_m",
    [(0.0, 0.0, 0.006), (0.0, 0.0, -0.006), (0.2, -0.15, 0.0), (0.2, -0.15, 0.006)],
    ids=["w", "minus_w", "separable", "both"],
)
def test_phase_factors_are_accurate_to_rounding(uvw_m):
    """One unit sample, phases below a turn (NumPy accurate): the w term up to
    0.9 of the small angle limit and the separable factors to a few 1e-16."""
    inp = _inputs(
        seed=14, n_time=1, n_baseline=1, n_vis_chan=1, n_pol=1, n_l=20, n_m=70
    )
    inp["vis_data"][:] = 1.0
    inp["weight"][:] = 1.0
    inp["uvw"][0, 0] = uvw_m
    image, normalization = _empty(inp, m_chan=1)
    dft_image(image, normalization, *_image_args(inp))
    fringe, n = _phase_terms(
        inp["uvw"][0, 0], inp["frequency_coord"][0], inp["l"], inp["m"]
    )
    u, v, w = inp["uvw"][0, 0] * inp["frequency_coord"][0] / C
    assert np.abs(2 * np.pi * w * (n - 1)).max() < 2**-7  # the small angle path
    assert np.max(np.abs(image[0, 0, 0] - n * np.conj(fringe))) < 5e-16


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_dft_image_uv_sampling_matches_the_definition(dtype):
    inp = _inputs(seed=1)
    image, normalization = _empty(inp, dtype=dtype)
    dft_image_uv_sampling(
        image, normalization, *_uv_sampling_args(inp), processing_function_threads=3
    )
    expected, expected_norm = _reference_image(inp, 1, 2, 4, uv_sampling=True)
    scale = np.sum(inp["weight"])
    rtol = 1e-13 if dtype == np.float64 else 1e-6
    assert np.max(np.abs(image - expected.real)) < rtol * scale
    np.testing.assert_allclose(normalization, expected_norm, rtol=1e-14)


def test_skipped_samples_and_normalization():
    """Non-finite uvw, NaN / zero weight and zero weighted visibility count nowhere."""
    inp = _inputs(seed=2)
    inp["uvw"][0, 3, 2] = np.nan  # a w of NaN drops the whole row
    inp["uvw"][2, 8, 0] = np.inf  # as does any non-finite u, v or w
    inp["weight"][1, 4, 0, 1] = np.nan  # flagged
    inp["weight"][1, 5, 1, :] = 0.0
    inp["vis_data"][2, 6, 0, 2] = 0.0  # weighted visibility zero: as the gridder
    inp["vis_data"][2, 7, 1, 3] = np.nan
    image, normalization = _empty(inp)
    dft_image(image, normalization, *_image_args(inp))
    expected, expected_norm = _reference_image(inp, 1, 2, 4)
    assert np.all(np.isfinite(image))
    scale = np.nansum(np.abs(inp["vis_data"] * inp["weight"]))
    assert np.max(np.abs(image - expected)) < 1e-13 * scale
    np.testing.assert_allclose(normalization, expected_norm, rtol=1e-14)
    valid = np.isfinite(inp["vis_data"] * inp["weight"]) & (
        inp["vis_data"] * inp["weight"] != 0
    )
    valid &= np.all(np.isfinite(inp["uvw"]), axis=-1)[..., None, None]
    np.testing.assert_allclose(
        normalization[0], np.where(valid, inp["weight"], 0).sum(axis=(0, 1)), rtol=1e-14
    )


def test_calls_accumulate():
    inp_a, inp_b = _inputs(seed=4), _inputs(seed=5)
    image, normalization = _empty(inp_a)
    dft_image(image, normalization, *_image_args(inp_a))
    dft_image(image, normalization, *_image_args(inp_b))
    a, norm_a = _reference_image(inp_a, 1, 2, 4)
    b, norm_b = _reference_image(inp_b, 1, 2, 4)
    scale = np.sum(np.abs(inp_a["vis_data"] * inp_a["weight"])) * 2
    assert np.max(np.abs(image - (a + b))) < 1e-13 * scale
    np.testing.assert_allclose(normalization, norm_a + norm_b, rtol=1e-14)


def test_maps_collapse_channels_and_polarizations():
    """Two visibility channels onto one image channel, a reversed polarization map."""
    inp = _inputs(seed=6)
    inp["frequency_map"] = np.zeros(2, dtype=np.int64)
    inp["pol_map"] = np.array([3, 2, 1, 0], dtype=np.int64)
    image, normalization = _empty(inp, m_chan=1)
    dft_image(image, normalization, *_image_args(inp))
    expected, expected_norm = _reference_image(inp, 1, 1, 4)
    scale = np.sum(np.abs(inp["vis_data"] * inp["weight"]))
    assert np.max(np.abs(image - expected)) < 1e-13 * scale
    np.testing.assert_allclose(normalization, expected_norm, rtol=1e-14)


@pytest.mark.parametrize(
    "model_dtype, vis_dtype",
    [
        (np.complex128, np.complex128),
        (np.float64, np.complex128),
        (np.float32, np.complex64),
        (np.complex64, np.complex128),
    ],
)
def test_dft_predict_matches_the_definition(model_dtype, vis_dtype):
    inp = _inputs(seed=7)
    rng = np.random.default_rng(8)
    model = np.zeros((1, 2, 4, inp["l"].size, inp["m"].size), dtype=model_dtype)
    for plane in np.ndindex(model.shape[:3]):
        pixels = rng.choice(model[plane].size, 7, replace=False)
        values = rng.standard_normal(7)
        if np.dtype(model_dtype).kind == "c":
            values = values + 1j * rng.standard_normal(7)
        model[plane].flat[pixels] = values
    model[0, 1, 2] = 0.0  # one plane without components (other planes have them there)
    vis = np.zeros(inp["vis_data"].shape, dtype=vis_dtype)
    dft_predict(model, vis, *_predict_args(inp), processing_function_threads=4)
    expected = _reference_predict(model.astype(np.complex128), inp)
    rtol = 1e-13 if vis_dtype == np.complex128 and model_dtype != np.complex64 else 1e-6
    assert np.max(np.abs(vis - expected)) < rtol * np.sum(np.abs(model))


def test_dft_predict_leaves_nan_samples_and_nan_uvw_alone():
    inp = _inputs(seed=9)
    model = np.zeros((1, 2, 4, inp["l"].size, inp["m"].size))
    model[0, 0, :, 3, 4] = 1.5  # channel 1 has no component: predicted 0
    inp["uvw"][1, 2] = np.nan
    inp["uvw"][2, 5, 2] = -np.inf
    vis = np.full(inp["vis_data"].shape, 7.0 + 7.0j)
    vis[0, 0, 0, 0] = np.nan
    dft_predict(model, vis, *_predict_args(inp))
    assert np.isnan(vis[0, 0, 0, 0])
    np.testing.assert_array_equal(vis[1, 2], 7.0 + 7.0j)
    np.testing.assert_array_equal(vis[2, 5], 7.0 + 7.0j)
    inp["uvw"][1, 2] = inp["uvw"][2, 5] = 0.0
    expected = _reference_predict(model, inp)
    keep = np.ones(vis.shape, dtype=bool)
    keep[0, 0, 0, 0] = keep[1, 2] = keep[2, 5] = False
    # phases of up to ~150 turns: the reference itself is rounded to ~1e-13
    np.testing.assert_allclose(vis[keep], expected[keep], rtol=0, atol=1e-12)
    assert np.all(vis[:, :, 1][keep[:, :, 1]] == 0)


def test_prediction_and_imaging_of_a_point_source_invert():
    """The image of a predicted point source is its flux at its pixel (w and 1/n exact)."""
    inp = _inputs(seed=10)
    model = np.zeros((1, 2, 4, inp["l"].size, inp["m"].size), dtype=np.complex128)
    model[0, :, :, 2, 8] = [0.7, 0.1 + 0.2j, 0.1 - 0.2j, 0.3]
    vis = np.zeros(inp["vis_data"].shape, dtype=np.complex128)
    dft_predict(model, vis, *_predict_args(inp))
    inp["vis_data"] = vis
    image, normalization = _empty(inp)
    dft_image(image, normalization, *_image_args(inp))
    image /= normalization[..., None, None]
    np.testing.assert_allclose(
        image[0, :, :, 2, 8], model[0, :, :, 2, 8], rtol=0, atol=1e-14
    )


def test_results_do_not_depend_on_the_thread_count():
    inp = _inputs(seed=11, n_time=4, n_baseline=40, n_l=33, n_m=17)
    model = np.zeros((1, 2, 4, 33, 17))
    model[0, :, :, 5:9, 3:16] = 0.25
    results = []
    for threads in (1, 1, 3, 7, 0):
        image, normalization = _empty(inp)
        dft_image(
            image, normalization, *_image_args(inp), processing_function_threads=threads
        )
        psf, psf_norm = _empty(inp, dtype=np.float64)
        dft_image_uv_sampling(
            psf, psf_norm, *_uv_sampling_args(inp), processing_function_threads=threads
        )
        vis = np.zeros(inp["vis_data"].shape, dtype=np.complex128)
        dft_predict(
            model, vis, *_predict_args(inp), processing_function_threads=threads
        )
        results.append((image, normalization, psf, psf_norm, vis))
    for other in results[1:]:
        for a, b in zip(results[0], other, strict=True):
            np.testing.assert_array_equal(a, b)


def test_binding_validation():
    inp = _inputs(seed=12)
    image, normalization = _empty(inp)
    args = _image_args(inp)
    with pytest.raises(RuntimeError, match="5-D"):
        dft_image(image[0], normalization, *args)
    with pytest.raises(RuntimeError, match="normalization must have shape"):
        dft_image(image, normalization[:, :1], *args)
    with pytest.raises(RuntimeError, match="normalization must be float64"):
        dft_image(image, normalization.astype(np.float32), *args)
    with pytest.raises(RuntimeError, match="image must be"):
        dft_image(image.real.astype(np.int64), normalization, *args)
    read_only = image.copy()
    read_only.setflags(write=False)
    with pytest.raises(RuntimeError, match="writable"):
        dft_image(read_only, normalization, *args)
    with pytest.raises(RuntimeError, match="C-contiguous"):
        dft_image(np.zeros(image.shape[::-1], dtype=complex).T, normalization, *args)
    bad = dict(inp, frequency_map=np.array([0, 2], dtype=np.int64))
    with pytest.raises(RuntimeError, match="outside the image axis"):
        dft_image(image, normalization, *_image_args(bad))
    bad = dict(inp, pol_map=np.arange(3, dtype=np.int64))
    with pytest.raises(RuntimeError, match="pol_map"):
        dft_image(image, normalization, *_image_args(bad))
    bad = dict(inp, weight=np.ascontiguousarray(inp["weight"][:, :, :1]))
    with pytest.raises(RuntimeError, match="weight must have the shape"):
        dft_image(image, normalization, *_image_args(bad))
    bad = dict(inp, l=inp["l"] * 40)
    with pytest.raises(RuntimeError, match="hemisphere"):
        dft_image(image, normalization, *_image_args(bad))
    bad = dict(inp, m=inp["m"][:-1])
    with pytest.raises(RuntimeError, match="image column"):
        dft_image(image, normalization, *_image_args(bad))
    # the large inputs are never converted: another dtype is refused
    bad = dict(inp, vis_data=inp["vis_data"].astype(np.complex64))
    with pytest.raises(TypeError):
        dft_image(image, normalization, *_image_args(bad))
    vis = np.zeros(inp["vis_data"].shape, dtype=np.complex128)
    with pytest.raises(RuntimeError, match="vis_data must be complex64 or complex128"):
        dft_predict(image, vis.real.copy(), *_predict_args(inp))
    with pytest.raises(RuntimeError, match="model must be"):
        dft_predict(image.real.astype(np.int32), vis, *_predict_args(inp))
    with pytest.raises(RuntimeError, match="image must be float32 or float64"):
        dft_image_uv_sampling(image, normalization, *_uv_sampling_args(inp))
