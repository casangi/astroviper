"""
The iteration history of a model update (``record_iteration_history``).

With the flag every deconvolver records, per plane and for every component
of the model update, the signed residual value at the peak before it, the RMS
of the residual over the searched pixels (the clean box inside the mask)
after it, and the component (its pixel, scale and flux); ``deconvolve`` adds
the model flux after every component. These tests check

- the kernels against a brute force NumPy Hogbom clean that recomputes the RMS
  from the whole residual after every iteration, and the recorded peaks
  against the progress callback of the single-plane kernel;
- that recording is off by default and changes no result (the residual, the
  model and the stop codes are the same to the bit with and without it);
- the per-plane arrays of the cube kernels (as long as the plane's own
  iterations, no padding) and the agreement of the two cube kernels;
- the Asp history: one record per component, whose fluxes add up to the
  model and whose RMS is that of the masked residual;
- the path of the history through ``deconvolve``, the per-cycle merge, the
  DataFrame (and feather), the pretty printer and the imaging cycle loop.
"""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.deconvolution import (
    _validate_deconvolve_params,
    asp_clean,
    deconvolve,
    validate_record_iteration_history,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    build_residual_imaging_dict,
    format_imaging_dict,
    merge_imaging_dicts,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    FIELD_ACCUM,
    ITERATION_HISTORY_FIELDS,
    ImagingDict,
    Key,
    imaging_dict_to_dataframe,
)

try:
    from astroviper.processing_functions.imaging.deconvolvers import hogbom

    HOGBOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    HOGBOM_AVAILABLE = False

requires_hogbom = pytest.mark.skipif(
    not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
)

NY, NX = 24, 32
GAIN = 0.2
CUBE_KERNELS = ["clean_cube", "clean_cube_many_threads"]
KERNELS = ["clean", *CUBE_KERNELS]
# The history the kernels record (deconvolve adds the model flux).
KERNEL_FIELDS = tuple(
    f for f in ITERATION_HISTORY_FIELDS if f != "iteration_model_flux"
)


def _gaussian(dy, dx, sigma=1.5):
    yy, xx = np.mgrid[0:NY, 0:NX]
    return np.exp(
        -((yy - NY // 2 - dy) ** 2 + (xx - NX // 2 - dx) ** 2) / (2 * sigma**2)
    )


def _psf(dtype=np.float64):
    """A unit main lobe with two negative sidelobes, so the RMS of the
    residual does not simply fall with the peak."""
    psf = _gaussian(0, 0) - 0.2 * _gaussian(0, 5) - 0.2 * _gaussian(0, -5)
    return (psf / psf[NY // 2, NX // 2]).astype(dtype)


def _residual(seed=0, dtype=np.float64):
    """A few PSF-shaped sources of either sign on low noise."""
    rng = np.random.default_rng(seed)
    psf = _psf(np.float64)
    residual = 0.01 * rng.standard_normal((NY, NX))
    for amplitude, dy, dx in ((1.0, -3, 4), (-0.6, 5, -6), (0.35, 2, 9)):
        residual += amplitude * np.roll(np.roll(psf, dy, 0), dx, 1)
    return residual.astype(dtype)


def _mask():
    """Clean mask without a border and a block, so masked pixels matter."""
    mask = np.zeros((NY, NX), dtype=bool)
    mask[2:-2, 3:-3] = True
    mask[4:9, 18:24] = False
    return mask


def _box_selection(clean_box):
    xbeg, xend, ybeg, yend = clean_box
    box = np.zeros((NY, NX), dtype=bool)
    box[
        0 if ybeg == -1 else ybeg : NY if yend == -1 else yend,
        0 if xbeg == -1 else xbeg : NX if xend == -1 else xend,
    ] = True
    return box


def _reference_history(residual, psf, searched, gain, max_iter, threshold=0.0):
    """Hogbom clean of one plane in NumPy, with the kernel's selection rule
    (first largest |residual| in row order) and its arithmetic. Returns the
    signed peak each iteration cleaned, the RMS of the searched pixels after
    its subtraction (recomputed from the whole residual), the residual and the
    (row, column) of every peak."""
    residual = residual.copy()
    rows, cols = np.arange(NY), np.arange(NX)
    peaks, rms, positions = [], [], []
    for _ in range(max_iter):
        magnitude = np.where(searched, np.abs(residual), -1.0)
        py, px = np.unravel_index(np.argmax(magnitude), residual.shape)
        if magnitude[py, px] <= threshold:
            break
        peak = residual[py, px]
        psf_rows, psf_cols = NY // 2 + rows - py, NX // 2 + cols - px
        keep_rows = (psf_rows >= 0) & (psf_rows < NY)
        keep_cols = (psf_cols >= 0) & (psf_cols < NX)
        residual[np.ix_(rows[keep_rows], cols[keep_cols])] -= (
            residual.dtype.type(gain) * peak
        ) * psf[np.ix_(psf_rows[keep_rows], psf_cols[keep_cols])]
        peaks.append(float(peak))
        positions.append((py, px))
        rms.append(float(np.sqrt(np.mean(residual[searched].astype(np.float64) ** 2))))
    return np.array(peaks), np.array(rms), residual, np.array(positions).reshape(-1, 2)


def _run(kernel, residual, psf, mask, max_iter, threshold=0.0, clean_box=None, **kw):
    """One plane through ``kernel``; returns the result dict, the residual and
    the model. ``clean`` is the single-plane kernel, the others take a cube."""
    residual = residual.copy()
    model = np.zeros_like(residual)
    clean_box = (-1, -1, -1, -1) if clean_box is None else clean_box
    if kernel == "clean":
        result = hogbom.clean(
            dirty_image=residual,
            psf=psf,
            model=model,
            mask=mask,
            clean_box=clean_box,
            max_iter_per_cycle_updated=max_iter,
            gain=GAIN,
            threshold_per_cycle=threshold,
            **kw,
        )
        return result, residual, model
    cube = np.ascontiguousarray(residual[None, None, None])
    model = np.zeros_like(cube)
    result = getattr(hogbom, kernel)(
        residual_cube=cube,
        psf_cube=np.ascontiguousarray(psf[None, None, None]),
        model_cube=model,
        mask_cube=np.ascontiguousarray(mask[None, None, None]),
        clean_box=clean_box,
        max_iter_per_cycle_updated=np.full((1, 1, 1), max_iter, np.int32),
        gain=GAIN,
        threshold_per_cycle=np.full((1, 1, 1), threshold, residual.dtype),
        processing_function_threads=3,
        **kw,
    )
    return result, cube[0, 0, 0], model[0, 0, 0]


def _plane_history(kernel, result):
    """The recorded peaks and RMS of the one plane of ``_run``."""
    if kernel == "clean":
        return result["iteration_peak_residual"], result["iteration_residual_rms"]
    return (
        result["iteration_peak_residual"][0][0][0],
        result["iteration_residual_rms"][0][0][0],
    )


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------


@requires_hogbom
class TestKernelHistory:
    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("record", [None, False])
    def test_off_by_default(self, kernel, record):
        kw = {} if record is None else {"record_iteration_history": record}
        result, _, _ = _run(kernel, _residual(), _psf(), _mask(), 10, **kw)
        for field in KERNEL_FIELDS:
            assert field not in result

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize(
        ("max_iter", "threshold", "clean_box"),
        [
            (60, 0.0, None),  # spends the budget: the last RMS needs a scan
            (500, 0.05, None),  # stops at the threshold
            (60, 0.0, (5, 27, 3, 20)),  # searches a clean box inside the mask
        ],
    )
    def test_history_matches_brute_force(
        self, kernel, dtype, max_iter, threshold, clean_box
    ):
        residual, psf, mask = _residual(dtype=dtype), _psf(dtype), _mask()
        searched = mask & _box_selection(clean_box or (-1, -1, -1, -1))
        peaks, rms, reference_residual, positions = _reference_history(
            residual, psf, searched, GAIN, max_iter, threshold
        )
        result, final_residual, _ = _run(
            kernel,
            residual,
            psf,
            mask,
            max_iter,
            threshold,
            clean_box,
            record_iteration_history=True,
        )
        recorded_peaks, recorded_rms = _plane_history(kernel, result)
        n_iter = int(np.asarray(result["iterations_performed"]).ravel()[0])
        assert n_iter == len(peaks) > 0
        # the case exercises the stop it is meant to
        assert (n_iter < max_iter) == (threshold > 0)
        assert recorded_peaks.dtype == recorded_rms.dtype == np.float64
        np.testing.assert_array_equal(recorded_peaks, peaks)
        np.testing.assert_allclose(recorded_rms, rms, rtol=1e-10, atol=0)
        # the reference follows the kernel's arithmetic to the bit
        np.testing.assert_array_equal(final_residual, reference_residual)
        # every component is a point of gain x peak at the peak's pixel
        component = {
            field: (result[field] if kernel == "clean" else result[field][0][0][0])
            for field in KERNEL_FIELDS
        }
        np.testing.assert_array_equal(
            component["iteration_component_l"], positions[:, 0]
        )
        np.testing.assert_array_equal(
            component["iteration_component_m"], positions[:, 1]
        )
        np.testing.assert_array_equal(component["iteration_component_scale"], 0.0)
        np.testing.assert_array_equal(
            component["iteration_component_flux"],
            (dtype(GAIN) * peaks.astype(dtype)).astype(np.float64),
        )

    def test_peaks_are_the_progress_callback_peaks(self):
        calls = []
        result, _, _ = _run(
            "clean",
            _residual(),
            _psf(),
            _mask(),
            50,
            record_iteration_history=True,
            progress_callback=lambda it, px, py, peak: calls.append((it, peak)),
        )
        peaks = result["iteration_peak_residual"]
        assert len(peaks) == result["iterations_performed"] == 50
        # the callbacks inside the loop report the peak of the iteration they
        # belong to; the closing one reports the iteration count
        in_loop = [(it, peak) for it, peak in calls if it < len(peaks)]
        assert len(in_loop) >= 5
        for it, peak in in_loop:
            assert peaks[it] == peak

    def test_start_iteration_records_the_iterations_of_the_call(self):
        result, _, _ = _run(
            "clean",
            _residual(),
            _psf(),
            _mask(),
            20,
            start_iter=5,
            record_iteration_history=True,
        )
        assert result["iterations_performed"] == 20
        for field in KERNEL_FIELDS:
            assert result[field].shape == (15,)
            assert np.isfinite(result[field]).all()

    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_recording_changes_no_result(self, kernel, dtype):
        residual, psf, mask = _residual(dtype=dtype), _psf(dtype), _mask()
        off, residual_off, model_off = _run(kernel, residual, psf, mask, 80)
        on, residual_on, model_on = _run(
            kernel, residual, psf, mask, 80, record_iteration_history=True
        )
        np.testing.assert_array_equal(residual_on, residual_off)
        np.testing.assert_array_equal(model_on, model_off)
        for key in off:
            np.testing.assert_array_equal(on[key], off[key])

    def test_planes_record_their_own_iterations(self):
        """One array per plane, as long as the plane's own iterations (no
        padding to the longest plane), identical in both cube kernels:
        planes stopped by the budget, by the threshold, by an all-zero
        residual and by a zero budget."""
        psf = _psf()
        residual = np.zeros((1, 2, 2, NY, NX))
        residual[0, 0, 0] = _residual(seed=1)
        residual[0, 0, 1] = _residual(seed=2)
        residual[0, 1, 1] = _residual(seed=3)
        psf_cube = np.ascontiguousarray(np.broadcast_to(psf, (1, 2, 1, NY, NX)))
        mask = np.ascontiguousarray(np.broadcast_to(_mask(), residual.shape))
        max_iter = np.array([[[40, 400], [40, 0]]], dtype=np.int32)
        threshold = np.array([[[0.0, 0.1], [0.0, 0.0]]])
        results = {}
        for kernel in CUBE_KERNELS:
            cube = residual.copy()
            results[kernel] = getattr(hogbom, kernel)(
                residual_cube=cube,
                psf_cube=psf_cube,
                model_cube=np.zeros_like(cube),
                mask_cube=mask,
                max_iter_per_cycle_updated=max_iter,
                gain=GAIN,
                threshold_per_cycle=threshold,
                processing_function_threads=4,
                record_iteration_history=True,
            )
        result = results["clean_cube"]
        iterations = result["iterations_performed"]
        assert iterations[0, 0, 0] == 40  # budget
        assert 0 < iterations[0, 0, 1] < 400  # threshold
        assert iterations[0, 1, 0] == 0  # all-zero residual
        assert iterations[0, 1, 1] == 0  # no budget
        for field in KERNEL_FIELDS:
            history = result[field]
            for tt, nn, pp in np.ndindex(iterations.shape):
                plane = history[tt][nn][pp]
                assert plane.dtype == np.float64
                assert plane.shape == (int(iterations[tt, nn, pp]),)
                assert np.isfinite(plane).all()
                np.testing.assert_array_equal(
                    results["clean_cube_many_threads"][field][tt][nn][pp], plane
                )

    @pytest.mark.parametrize("kernel", CUBE_KERNELS)
    def test_no_iteration_gives_an_empty_history(self, kernel):
        result, _, _ = _run(
            kernel, _residual(), _psf(), _mask(), 0, record_iteration_history=True
        )
        for field in KERNEL_FIELDS:
            assert result[field][0][0][0].shape == (0,)


# ---------------------------------------------------------------------------
# deconvolve and the parameters
# ---------------------------------------------------------------------------


def _image_dataset(npol=2, dtype=np.float64):
    """A one-channel image dataset with a residual group as ``deconvolve`` reads it."""
    dims = ("time", "frequency", "polarization", "l", "m")
    residual = np.stack([_residual(seed=pp, dtype=dtype) for pp in range(npol)])
    xds = xr.Dataset(
        {
            "SKY_RESIDUAL": (dims, residual[None, None]),
            "POINT_SPREAD_FUNCTION": (
                dims,
                np.broadcast_to(_psf(dtype), (1, 1, npol, NY, NX)).copy(),
            ),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "frequency", "polarization"),
                np.full((1, 1, npol), 0.2),
            ),
            "MASK": (dims, np.broadcast_to(_mask(), (1, 1, npol, NY, NX)).copy()),
        },
        coords={
            "time": [0.0],
            "frequency": [1.0e9],
            "polarization": ["I", "Q", "U", "V"][:npol],
        },
    )
    xds.attrs["data_groups"] = {
        "residual": {
            "sky": "SKY_RESIDUAL",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
            "max_sidelobe_point_spread_function": "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
            "mask": "MASK",
        }
    }
    return xds


class TestDeconvolve:
    @requires_hogbom
    @pytest.mark.parametrize("algorithm", ["hogbom", "hogbom_many_threads"])
    def test_every_plane_gets_its_history(self, algorithm):
        xds = _image_dataset()
        params = {"gain": GAIN, "max_iter": 30, "max_iter_divergence": -1}
        imaging_dict = deconvolve(
            img_xds=xds,
            algorithm=algorithm,
            deconvolve_params={**params, "record_iteration_history": True},
            processing_function_threads=2,
        )
        for key, entry in imaging_dict.data.items():
            (n_iter,) = entry["iter_done"]
            assert n_iter == 30
            for field in ITERATION_HISTORY_FIELDS:
                (history,) = entry[field]
                assert history.shape == (n_iter,)
                assert history.dtype == np.float64
            residual = _residual(seed=key.pol)
            peaks, rms, _, _ = _reference_history(
                residual, _psf(), _mask(), GAIN, n_iter
            )
            np.testing.assert_array_equal(entry["iteration_peak_residual"][0], peaks)
            np.testing.assert_allclose(
                entry["iteration_residual_rms"][0], rms, rtol=1e-10
            )
            # the model flux after every component ends at the model's flux
            np.testing.assert_allclose(
                entry["iteration_model_flux"][0],
                np.cumsum(GAIN * peaks),
                rtol=1e-12,
            )
            assert entry["iteration_model_flux"][0][-1] == pytest.approx(
                entry["model_flux"][0], rel=1e-12
            )

    @requires_hogbom
    def test_no_history_by_default(self):
        imaging_dict = deconvolve(
            img_xds=_image_dataset(),
            deconvolve_params={"gain": GAIN, "max_iter": 10},
        )
        for entry in imaging_dict.data.values():
            for field in ITERATION_HISTORY_FIELDS:
                assert field not in entry

    @pytest.mark.parametrize("algorithm", ["asp", "aspclean", "asp_clean"])
    def test_asp_records_every_component(self, algorithm):
        """One record per component (Asp also counts the iteration that ends
        the model update on its threshold test); the component fluxes add up
        to the model, and the RMS is that of the masked residual left."""
        xds = _image_dataset()
        imaging_dict = deconvolve(
            img_xds=xds,
            algorithm=algorithm,
            deconvolve_params={
                "gain": GAIN,
                "max_iter": 40,
                "max_iter_divergence": -1,
                "record_iteration_history": True,
            },
        )
        for key, entry in imaging_dict.data.items():
            history = {field: entry[field][0] for field in ITERATION_HISTORY_FIELDS}
            n = len(history["iteration_component_flux"])
            assert 0 < n <= entry["iter_done"][0]
            n_components = (
                entry["n_point_components"][0]
                + entry["n_scale_components"][0]
                + entry["n_residual_components"][0]
            )
            assert n == n_components
            for values in history.values():
                assert values.shape == (n,) and np.isfinite(values).all()
            model = xds["SKY_MODEL"].values[0, 0, key.pol]
            assert history["iteration_model_flux"][-1] == pytest.approx(
                model.sum(), rel=1e-9
            )
            residual = xds["SKY_RESIDUAL"].values[0, 0, key.pol]
            assert history["iteration_residual_rms"][-1] == pytest.approx(
                np.sqrt(np.mean(residual[_mask()] ** 2)), rel=1e-9
            )
            assert np.all(history["iteration_component_scale"] >= -1)

    def test_asp_clean_records_the_history(self):
        cube = _residual()[None, None, None]
        result = asp_clean(
            cube.copy(),
            _psf()[None, None, None],
            np.zeros_like(cube),
            deconvolve_params={"max_iter": 10, "record_iteration_history": True},
        )
        for field in KERNEL_FIELDS:
            assert len(result[field][0][0][0]) > 0

    def test_validate_record_iteration_history(self):
        for deconvolver in ("hogbom", "HOGBOM_MANY_THREADS", "asp", "mtmfs"):
            validate_record_iteration_history(deconvolver, True)
        validate_record_iteration_history("multiscale", False)
        with pytest.raises(ValueError, match="'multiscale'"):
            validate_record_iteration_history("multiscale", True)

    def test_parameter_defaults_to_false(self):
        assert _validate_deconvolve_params({})["record_iteration_history"] is False

    @pytest.mark.parametrize("value", [1, "yes", None])
    def test_parameter_must_be_a_bool(self, value):
        with pytest.raises(ValueError, match="record_iteration_history must be a bool"):
            _validate_deconvolve_params({"record_iteration_history": value})


# ---------------------------------------------------------------------------
# The record: merge, DataFrame, printing, the residual seed
# ---------------------------------------------------------------------------


def _two_cycle_record():
    """One plane, two cycles of 3 and 2 iterations, merged as the loop does."""
    combined = ImagingDict()
    for peaks, rms in (([1.0, -0.8, 0.7], [0.3, 0.25, 0.2]), ([0.5, 0.4], [0.15, 0.1])):
        cycle = ImagingDict()
        cycle.add(
            {
                "iter_done": len(peaks),
                "iteration_peak_residual": np.array(peaks),
                "iteration_residual_rms": np.array(rms),
            },
            time=0,
            pol=0,
            chan=0,
        )
        combined = merge_imaging_dicts([combined, cycle])
    return combined


class TestRecord:
    def test_the_history_fields_accumulate_per_cycle(self):
        assert set(ITERATION_HISTORY_FIELDS) <= FIELD_ACCUM

    def test_merge_keeps_one_array_per_cycle(self):
        entry = _two_cycle_record().data[Key(time=0, pol=0, chan=0)]
        assert entry["iter_done"] == [3, 2]
        peaks = entry["iteration_peak_residual"]
        assert [len(cycle) for cycle in peaks] == [3, 2]
        np.testing.assert_array_equal(np.concatenate(peaks), [1.0, -0.8, 0.7, 0.5, 0.4])

    def test_dataframe_and_feather_carry_the_history(self, tmp_path):
        import pandas as pd

        frame = imaging_dict_to_dataframe(_two_cycle_record())
        row = frame.iloc[0]
        assert row["iter_total"] == 5
        assert [list(cycle) for cycle in row["iteration_residual_rms"]] == [
            [0.3, 0.25, 0.2],
            [0.15, 0.1],
        ]
        path = tmp_path / "convergence.ft"
        frame.to_feather(path)
        back = pd.read_feather(path).iloc[0]
        assert [list(cycle) for cycle in back["iteration_peak_residual"]] == [
            [1.0, -0.8, 0.7],
            [0.5, 0.4],
        ]

    def test_pretty_print_shows_the_lengths(self):
        text = format_imaging_dict(_two_cycle_record())
        assert "<3 values>" in text and "<2 values>" in text

    @pytest.mark.parametrize("record", [False, True])
    def test_residual_record_holds_empty_histories(self, record):
        xds = _image_dataset(npol=1)
        imaging_dict = build_residual_imaging_dict(
            xds, "residual", {"gain": GAIN, "record_iteration_history": record}
        )
        (entry,) = imaging_dict.data.values()
        for field in ITERATION_HISTORY_FIELDS:
            if record:
                (history,) = entry[field]
                assert history.shape == (0,)
            else:
                assert field not in entry
