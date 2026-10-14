"""
``max_cycles="auto"``: the limit of the imaging cycles predicted from the
dirty image.

After the first residual update of a channel every plane predicts the cycles
that bring its peak residual ``P0`` down to a level ``L``,
``n = predicted_imaging_cycles(P0, L, f)`` with ``f`` its PSF fraction, and
takes ``n + EXTRA_CYCLES`` when ``L`` is its threshold or
``NOISY_CYCLE_SAFETY * n`` when ``L`` is its theoretical point source
sensitivity (threshold 0), worked out from the weights. The channel's
``max_cycles`` is the largest of these limits and then counts down like a
number. Tested on the controller with synthetic records, on an image dataset
and a processing set (the clean mask and the sensitivity) and on small
imaging runs of the processing function. The same runs test the stop test,
made after every residual update on the residual it has just made.
"""

import copy

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_CONTINUE,
    IMAGING_DIVERGENCE,
    IMAGING_ENTROPY,
    IMAGING_MAX_CYCLES,
    IMAGING_MAX_ITER,
    IMAGING_NO_PROGRESS,
    IMAGING_THRESHOLD,
    IMAGING_ZERO_MASK,
    MODEL_UPDATE_DIVERGENCE,
    MODEL_UPDATE_MAX_ITER_PER_CYCLE,
    MODEL_UPDATE_THRESHOLD_PER_CYCLE,
    CycleController,
    StopCode,
    format_imaging_dict,
    resolve_auto_max_cycles,
    validate_max_cycles,
)
from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
    EXTRA_CYCLES,
    NOISY_CYCLE_SAFETY,
    predicted_imaging_cycles,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    ITERATION_HISTORY_FIELDS,
    ImagingDict,
    Key,
    imaging_dict_to_dataframe,
)


def _dirty_record(planes):
    """Statistics of a dirty image: ``{(time, chan, pol): (peak, sidelobe,
    masksum)}``, as :func:`build_residual_imaging_dict` records them."""
    record = ImagingDict()
    for (tt, nn, pp), (peak, sidelobe, masksum) in planes.items():
        record.add(
            {
                "peakres": peak,
                "peakres_nomask": peak,
                "masksum": masksum,
                "iter_done": 0,
                "max_psf_sidelobe": sidelobe,
                "gain": 0.1,
            },
            time=tt,
            pol=pp,
            chan=nn,
        )
    return record


def _controller(threshold=0.0, **kw):
    return CycleController(
        max_iter=1000,
        max_cycles="auto",
        threshold=threshold,
        psf_sidelobe_factor=1.0,
        min_psf_fraction=0.05,
        max_psf_fraction=0.8,
        **kw,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "value", [-1, 0, 7, np.int64(3), 5.0, np.float64(5), -1.0, "auto"]
    )
    def test_accepted(self, value):
        assert validate_max_cycles(value) == value
        assert CycleController(max_cycles=value).max_cycles == value

    @pytest.mark.parametrize("value", [5.0, np.float64(5), np.float32(5), np.int64(5)])
    def test_a_number_becomes_an_int(self, value):
        for resolved in (
            validate_max_cycles(value),
            CycleController(max_cycles=value).max_cycles,
        ):
            assert resolved == 5
            assert type(resolved) is int

    @pytest.mark.parametrize(
        "value",
        [
            "AUTO",
            "5",
            1.5,
            2.7,
            np.float64(2.5),
            -2.0,
            float("nan"),
            float("inf"),
            -2,
            True,
            np.bool_(True),
            None,
            [3],
        ],
    )
    def test_refused(self, value):
        with pytest.raises(ValueError, match='integer >= -1 or "auto"'):
            validate_max_cycles(value)
        with pytest.raises(ValueError, match="max_cycles"):
            CycleController(max_cycles=value)

    def test_auto_needs_a_psf_fraction_below_one(self):
        with pytest.raises(ValueError, match="max_psf_fraction below 1"):
            validate_max_cycles("auto", 1.0)
        with pytest.raises(ValueError, match="max_psf_fraction below 1"):
            CycleController(max_cycles="auto", max_psf_fraction=1.0)
        assert validate_max_cycles("auto", 0.8) == "auto"
        # a number keeps accepting it
        assert validate_max_cycles(3, 1.0) == 3
        assert CycleController(max_cycles=3, max_psf_fraction=1.0).max_cycles == 3

    def test_update_parameters(self):
        controller = CycleController(max_cycles=4)
        assert controller.update_parameters(max_cycles=6.0) == (0, "")
        assert controller.max_cycles == 6
        assert type(controller.max_cycles) is int
        code, message = controller.update_parameters(max_cycles="auto")
        assert code == -1
        assert "given when the controller is created" in message
        for value in (2.7, True, -2, "5"):
            code, message = controller.update_parameters(max_cycles=value)
            assert code == -1
            assert message.startswith("max_cycles must be an integer >= -1")
        assert controller.max_cycles == 6


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------


class TestResolveMaxCycles:
    def test_threshold_reference(self):
        controller = _controller(threshold=0.01)
        record = _dirty_record({(0, 0, 0): (1.0, 0.2, 100)})
        expected = predicted_imaging_cycles(1.0, 0.01, 0.2) + EXTRA_CYCLES
        assert expected == 3 + EXTRA_CYCLES
        assert controller.resolve_max_cycles(record) == expected
        assert controller.max_cycles == expected
        assert controller.max_cycles_auto == {(0, 0, 0): expected}

    def test_noise_reference(self):
        controller = _controller(threshold=0.0)
        record = _dirty_record({(0, 0, 0): (-1.0, 0.2, 100)})
        noise = np.full((1, 1, 1), 1e-3)
        expected = NOISY_CYCLE_SAFETY * predicted_imaging_cycles(1.0, 1e-3, 0.2)
        assert expected == NOISY_CYCLE_SAFETY * 5
        assert controller.resolve_max_cycles(record, sensitivity=noise) == expected

    def test_psf_fraction_is_clamped(self):
        # sidelobe 0.9 -> f = max_psf_fraction = 0.8; sidelobe 0.01 -> 0.05
        for sidelobe, fraction in ((0.9, 0.8), (0.01, 0.05)):
            controller = _controller(threshold=0.01)
            controller.resolve_max_cycles(
                _dirty_record({(0, 0, 0): (1.0, sidelobe, 100)})
            )
            assert controller.max_cycles == (
                predicted_imaging_cycles(1.0, 0.01, fraction) + EXTRA_CYCLES
            )

    def test_zero_psf_fraction_is_one_cycle(self):
        controller = CycleController(
            max_cycles="auto", threshold=0.01, min_psf_fraction=0.0
        )
        controller.resolve_max_cycles(_dirty_record({(0, 0, 0): (1.0, 0.0, 100)}))
        assert controller.max_cycles == 1 + EXTRA_CYCLES

    def test_largest_limit_of_the_planes(self):
        """Per plane threshold: Stokes I against its threshold, Stokes Q
        against its noise; the channel takes the larger limit."""
        controller = _controller(threshold=[0.01, 0.0])
        record = _dirty_record(
            {(0, 0, 0): (1.0, 0.2, 100), (0, 0, 1): (0.05, 0.2, 100)}
        )
        noise = np.array([[[np.nan, 1e-5]]])
        limit_i = predicted_imaging_cycles(1.0, 0.01, 0.2) + EXTRA_CYCLES
        limit_q = NOISY_CYCLE_SAFETY * predicted_imaging_cycles(0.05, 1e-5, 0.2)
        assert limit_q > limit_i
        controller.resolve_max_cycles(record, sensitivity=noise)
        assert controller.max_cycles == limit_q
        assert controller.max_cycles_auto == {(0, 0, 0): limit_i, (0, 0, 1): limit_q}

    def test_planes_without_a_prediction_are_left_out(self):
        controller = _controller(threshold=0.0)
        record = _dirty_record(
            {
                (0, 0, 0): (1.0, 0.2, 100),  # the only plane with a limit
                (0, 0, 1): (50.0, 0.2, 100),  # mask without a pixel (below)
                (0, 0, 2): (0.0, 0.2, 100),  # all-zero residual
                (0, 0, 3): (1.0, 0.2, 100),  # no data: infinite sensitivity
            }
        )
        noise = np.array([[[1e-3, 1e-3, 1e-3, np.inf]]])
        pixels = np.array([[[100, 0, 100, 100]]])
        controller.resolve_max_cycles(record, sensitivity=noise, mask_pixels=pixels)
        expected = NOISY_CYCLE_SAFETY * predicted_imaging_cycles(1.0, 1e-3, 0.2)
        assert controller.max_cycles == expected
        assert controller.max_cycles_auto == {
            (0, 0, 0): expected,
            (0, 0, 1): None,
            (0, 0, 2): None,
            (0, 0, 3): None,
        }

    def test_masksum_of_the_record_without_mask_pixels(self):
        controller = _controller(threshold=0.01)
        controller.resolve_max_cycles(
            _dirty_record({(0, 0, 0): (1.0, 0.2, 0), (0, 0, 1): (0.5, 0.2, 10)})
        )
        assert controller.max_cycles_auto[(0, 0, 0)] is None
        assert controller.max_cycles == (
            predicted_imaging_cycles(0.5, 0.01, 0.2) + EXTRA_CYCLES
        )

    def test_no_plane_with_a_prediction_gives_one_cycle(self):
        controller = _controller(threshold=0.0)
        controller.resolve_max_cycles(_dirty_record({(0, 0, 0): (1.0, 0.2, 100)}))
        assert controller.max_cycles == 1
        assert controller.max_cycles_auto == {(0, 0, 0): None}

    def test_a_number_is_left_alone(self):
        controller = CycleController(max_cycles=4, threshold=0.01)
        assert (
            controller.resolve_max_cycles(_dirty_record({(0, 0, 0): (1, 0.2, 9)})) == 4
        )
        assert controller.max_cycles_auto is None

    def test_counts_down_like_a_number(self):
        controller = _controller(threshold=0.01)
        record = _dirty_record({(0, 0, 0): (1.0, 0.2, 100)})
        limit = controller.resolve_max_cycles(record)
        model_update = ImagingDict()
        model_update.add(
            {"peakres": 0.5, "masksum": 100, "iter_done": 10}, time=0, pol=0, chan=0
        )
        n_model_updates = 0
        while True:
            stopcode, _ = controller.check_convergence(record)
            if stopcode.imaging != IMAGING_CONTINUE:
                break
            n_model_updates += 1
            controller.update_counts(model_update)
            controller.record_model_update(model_update)
        assert stopcode.imaging == IMAGING_MAX_CYCLES
        assert n_model_updates == limit

    def test_checks_stamp_the_limit_of_the_plane(self):
        controller = _controller(threshold=0.01)
        record = _dirty_record({(0, 0, 0): (1.0, 0.2, 100), (0, 0, 1): (0.0, 0.2, 9)})
        controller.resolve_max_cycles(record)
        controller.check_convergence(record)
        assert record.data[Key(time=0, pol=0, chan=0)]["max_cycles_auto"] == (
            controller.max_cycles
        )
        assert record.data[Key(time=0, pol=1, chan=0)]["max_cycles_auto"] is None

    def test_a_number_stamps_nothing(self):
        record = _dirty_record({(0, 0, 0): (1.0, 0.2, 100)})
        CycleController(max_cycles=3, threshold=0.01).check_convergence(record)
        assert "max_cycles_auto" not in record.data[Key(time=0, pol=0, chan=0)]

    def test_unresolved_auto_is_refused(self):
        record = _dirty_record({(0, 0, 0): (1.0, 0.2, 100)})
        with pytest.raises(ValueError, match="not resolved"):
            _controller().check_convergence(record)
        with pytest.raises(ValueError, match="not resolved"):
            _controller().update_counts(record)


# ---------------------------------------------------------------------------
# The clean mask of an image dataset and the sensitivity of the weights
# ---------------------------------------------------------------------------

NY, NX = 16, 20


def _dirty_dataset(with_mask=False, seed=0):
    """Two Stokes planes of noise inside the primary beam and strong emission
    outside it, where the clean mask (the primary beam at or above 0.2 of its
    peak) ends. Plane 1 lies wholly below the cutoff."""
    rng = np.random.default_rng(seed)
    dims = ("time", "frequency", "polarization", "l", "m")
    inside = np.zeros((NY, NX), dtype=bool)
    inside[3:13, 4:16] = True
    primary_beam = np.where(inside, 1.0, 0.1)[None, None, None].repeat(2, axis=2)
    primary_beam[0, 0, 1] = 0.15
    residual = rng.standard_normal((1, 1, 2, NY, NX))
    residual[..., ~inside] = 100.0
    residual[0, 0, 0, 8, 10] = 25.0
    xds = xr.Dataset(
        {
            "SKY_RESIDUAL": (dims, residual),
            "PRIMARY_BEAM": (dims, primary_beam),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "frequency", "polarization"),
                np.full((1, 1, 2), 0.2),
            ),
        },
        coords={"frequency": [FREQUENCY], "polarization": ["I", "Q"]},
    )
    group = {
        "sky": "SKY_RESIDUAL",
        "primary_beam": "PRIMARY_BEAM",
        "max_sidelobe_point_spread_function": "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
    }
    if with_mask:
        mask = np.zeros((1, 1, 2, NY, NX), dtype=bool)
        mask[0, 0, :, 3:8, 4:16] = True
        xds["MASK"] = (dims, mask)
        group["mask"] = "MASK"
    xds.attrs["data_groups"] = {"residual": group}
    return xds, inside


NOISE = 1e-3


def _weighted_processing_set(xds, noise=NOISE):
    """:func:`_processing_set` (absolute weights) with the natural imaging
    weights the imaging setup adds to its data group."""
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )

    ps_xdt = _processing_set(noise)
    calculate_imaging_weights(
        ps_xdt,
        xds,
        imaging_weights_params={"weighting": "natural"},
        ms_data_group_in_name="base",
        ms_data_group_out_name="base",
    )
    return ps_xdt


def _sensitivity(noise=NOISE, n_antenna=8, n_time=16):
    """Thermal noise of a Stokes plane of the natural weighted image of
    :func:`_processing_set`: two correlations of ``n_time`` times every
    baseline, each of noise ``noise`` per real and imaginary part."""
    n_samples = n_time * n_antenna * (n_antenna - 1) // 2
    return noise / np.sqrt(2 * n_samples)


class TestResolveAutoMaxCycles:
    def _resolve(self, xds, ps_xdt, threshold=0.0):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            build_residual_imaging_dict,
        )

        controller = _controller(threshold=threshold)
        params = {"gain": 0.1, "primary_beam_limit": 0.2}
        record = build_residual_imaging_dict(xds, "residual", params)
        return controller, resolve_auto_max_cycles(
            controller,
            record,
            ps_xdt,
            xds,
            params,
            processing_set_data_group_name="base",
        )

    def test_level_is_the_sensitivity_of_the_weights(self, monkeypatch):
        """``L`` of a plane with threshold 0 is the ``SENSITIVITY`` of the
        weight statistics, the thermal noise of the plane for absolute
        weights, inside the mask the model update will make."""
        from astroviper.processing_functions.imaging.weight_statistics import (
            calculate_weight_statistics_single_field,
        )

        xds, _ = _dirty_dataset()
        ps_xdt = _weighted_processing_set(xds)
        data_groups = copy.deepcopy(xds.attrs["data_groups"])
        ms_xds = ps_xdt["ms_0"].to_dataset().copy(deep=True)
        levels = []
        resolve = CycleController.resolve_max_cycles

        def spy(controller, imaging_dict, sensitivity=None, mask_pixels=None):
            levels.append(sensitivity)
            return resolve(controller, imaging_dict, sensitivity, mask_pixels)

        monkeypatch.setattr(CycleController, "resolve_max_cycles", spy)
        controller, limit = self._resolve(xds, ps_xdt)
        (level,) = levels
        statistics = calculate_weight_statistics_single_field(ps_xdt, xds, "base")
        np.testing.assert_array_equal(level, statistics["SENSITIVITY"].values[None])
        np.testing.assert_allclose(level, _sensitivity(), rtol=1e-12)
        # P0 is the peak inside the clean mask, as for threshold_per_cycle
        expected = NOISY_CYCLE_SAFETY * predicted_imaging_cycles(
            25.0, _sensitivity(), 0.2
        )
        assert limit == controller.max_cycles == expected
        # plane 1 has no pixel at or above the cutoff: left out
        assert controller.max_cycles_auto == {(0, 0, 0): expected, (0, 0, 1): None}
        # neither the dataset nor the processing set is modified
        assert "MASK" not in xds
        assert xds.attrs["data_groups"] == data_groups
        xr.testing.assert_identical(ps_xdt["ms_0"].to_dataset(), ms_xds)

    def test_mask_of_the_residual_group(self):
        """The mask of the residual group has pixels in plane 1, below the
        primary beam cutoff: that plane gets a limit as well."""
        xds, _ = _dirty_dataset(with_mask=True)
        controller, _ = self._resolve(xds, _weighted_processing_set(xds))
        # P0 is the peak inside that mask: noise only
        peaks = np.abs(xds["SKY_RESIDUAL"].values[0, 0, :, 3:8, 4:16]).max(
            axis=(-2, -1)
        )
        assert controller.max_cycles_auto == {
            (0, 0, pol): NOISY_CYCLE_SAFETY
            * predicted_imaging_cycles(peaks[pol], _sensitivity(), 0.2)
            for pol in range(2)
        }

    def test_no_plane_needs_the_weights(self):
        """The weights are read only for a plane with threshold 0 that has a
        pixel in its clean mask: here a threshold, and a threshold 0 plane
        wholly below the primary beam cutoff (no processing set needed)."""
        xds, _ = _dirty_dataset()
        controller, _ = self._resolve(xds, None, threshold=0.5)
        assert controller.max_cycles == (
            predicted_imaging_cycles(25.0, 0.5, 0.2) + EXTRA_CYCLES
        )
        controller, _ = self._resolve(xds, None, threshold=[0.5, 0.0])
        assert controller.max_cycles_auto == {
            (0, 0, 0): predicted_imaging_cycles(25.0, 0.5, 0.2) + EXTRA_CYCLES,
            (0, 0, 1): None,
        }

    def test_a_number_is_left_alone(self):
        xds, _ = _dirty_dataset()
        controller = CycleController(max_cycles=2)
        assert resolve_auto_max_cycles(controller, ImagingDict(), None, xds, {}) == 2


# ---------------------------------------------------------------------------
# The imaging cycle loop of the processing function
# ---------------------------------------------------------------------------

FREQUENCY = 1.4e9
WAVELENGTH = 299792458.0 / FREQUENCY
CELL = 1.0e-4
IMAGE_SIZE = [48, 48]
IMAGE_PARAMS = {
    "image_size": IMAGE_SIZE,
    "cell_size": np.array([-CELL, CELL]),
    "phase_direction": np.array([1.0, 0.5]),
    "time_coords": [0],
    "polarization_coords": ["I", "Q"],
    "fft_padding": 1.2,
}
SOURCES = [(0.0, 0.0, 1.0), (8 * CELL, -5 * CELL, 0.4)]


def _processing_set(noise=0.0, n_antenna=8, n_time=16, seed=0):
    """Unpolarized point sources (XX = YY, so Stokes Q is empty) observed at
    random uv points out to 0.3 / cell, with optional thermal noise of
    ``noise`` per real and imaginary part and then the absolute weights
    ``1 / noise**2`` a simulation with noise writes (unit weights without)."""
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(n_antenna) for b in range(a + 1, n_antenna)]
    shape = (n_time, len(pairs))
    radius = 0.3 / CELL * np.sqrt(rng.uniform(0.02, 1.0, shape))
    angle = rng.uniform(0.0, 2.0 * np.pi, shape)
    u, v = radius * np.cos(angle), radius * np.sin(angle)
    vis = np.zeros(shape, dtype=complex)
    for l0, m0, flux in SOURCES:
        vis += flux * np.exp(2j * np.pi * (u * l0 + v * m0))
    vis = np.repeat(vis[:, :, None, None], 2, axis=-1)
    vis += noise * (
        rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape)
    )
    dims = ("time", "baseline_id", "frequency", "polarization")
    ms_xds = xr.Dataset(
        {
            "VISIBILITY": (dims, vis),
            "FLAG": (dims, np.zeros(vis.shape, dtype=bool)),
            "WEIGHT": (dims, np.full(vis.shape, 1.0 / noise**2 if noise else 1.0)),
            "UVW": (
                ("time", "baseline_id", "uvw_label"),
                np.stack([u * WAVELENGTH, v * WAVELENGTH, np.zeros(shape)], -1),
            ),
        },
        coords={
            "time": np.arange(n_time, dtype=float),
            "baseline_id": np.arange(len(pairs)),
            "baseline_antenna1_name": ("baseline_id", [f"a{a}" for a, _ in pairs]),
            "baseline_antenna2_name": ("baseline_id", [f"a{b}" for _, b in pairs]),
            "frequency": [FREQUENCY],
            "polarization": ["XX", "YY"],
            "uvw_label": ["u", "v", "w"],
        },
        attrs={
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                }
            }
        },
    )
    return xr.DataTree.from_dict({"ms_0": ms_xds})


def _image(
    cycle_control_params,
    noise=0.0,
    deconvolver="hogbom",
    restore=False,
    **visibilities,
):
    from xradio.image import make_empty_sky_image

    from astroviper.processing_functions.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    params = {
        "max_iter": 2000,
        "max_cycles": -1,
        "threshold": 0.0,
        "primary_beam_limit": 0.2,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.0,
        "max_iter_per_cycle": -1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        **cycle_control_params,
    }
    img_xds = make_empty_sky_image(
        phase_center=IMAGE_PARAMS["phase_direction"],
        image_size=IMAGE_SIZE,
        cell_size=IMAGE_PARAMS["cell_size"],
        frequency_coords=[FREQUENCY],
        pol_coords=["XX", "YY"],
        time_coords=[0],
        do_sky_coords=False,
    )
    return image_cube_single_field(
        _processing_set(noise, **visibilities),
        img_xds,
        IMAGE_PARAMS,
        {"weighting": "natural"},
        params,
        processing_set_data_group_name="base",
        deconvolver=deconvolver,
        single_precision_image=False,
        fft_backend="scipy",
        image_data_variables_keep=["sky_residual", "sky_model"],
        restore=restore,
    )


STOKES_I, STOKES_Q = Key(time=0, pol=0, chan=0), Key(time=0, pol=1, chan=0)


class TestImagingCycleLoop:
    def test_stops_on_the_threshold_within_the_limit(self):
        _, timing, record = _image({"max_cycles": "auto", "threshold": 0.01})
        stokes_i = record.data[STOKES_I]
        limit = stokes_i["max_cycles_auto"]
        p0 = abs(stokes_i["start_peakres"][0])
        fraction = stokes_i["max_psf_sidelobe"]  # inside the clamp here
        assert limit == predicted_imaging_cycles(p0, 0.01, fraction) + EXTRA_CYCLES
        assert stokes_i["stop_code"].imaging == IMAGING_THRESHOLD
        # one entry per model update and the closing stop test
        assert len(stokes_i["iter_done"]) == timing["n_cycles"].iloc[0] + 1
        assert timing["n_cycles"].iloc[0] <= limit
        # the empty Stokes Q plane has nothing to predict
        assert record.data[STOKES_Q]["max_cycles_auto"] is None

    def test_the_limit_binds(self):
        """A model update of two iterations cannot keep up with the
        prediction: the predicted limit ends the cycles."""
        _, timing, record = _image(
            {"max_cycles": "auto", "threshold": 0.001, "max_iter_per_cycle": 2}
        )
        stokes_i = record.data[STOKES_I]
        assert stokes_i["stop_code"].imaging == IMAGING_MAX_CYCLES
        assert len(stokes_i["iter_done"]) == timing["n_cycles"].iloc[0] + 1
        assert timing["n_cycles"].iloc[0] == stokes_i["max_cycles_auto"]

    def test_the_record_carries_the_model_update_stop_code(self):
        """Every model update of two iterations spends them: the record keeps
        the stop code of the deconvolver next to the imaging stop code."""
        _, timing, record = _image({"max_cycles": 3, "max_iter_per_cycle": 2})
        assert timing["n_cycles"].iloc[0] == 3
        assert record.data[STOKES_I]["stop_code"] == StopCode(
            IMAGING_MAX_CYCLES, MODEL_UPDATE_MAX_ITER_PER_CYCLE
        )

    def test_max_cycles_divergence_reaches_the_controller(self):
        with pytest.raises(ValueError, match="max_cycles_divergence"):
            _image({"max_cycles_divergence": 0})

    def test_sensitivity_reference(self):
        """Threshold 0: every plane predicts its limit from the thermal noise
        of its image, the sensitivity of the absolute weights."""
        noise = 0.05
        _, timing, record = _image({"max_cycles": "auto"}, noise=noise)
        limits = []
        for key in (STOKES_I, STOKES_Q):
            entry = record.data[key]
            limits.append(entry["max_cycles_auto"])
            assert limits[-1] == NOISY_CYCLE_SAFETY * predicted_imaging_cycles(
                abs(entry["start_peakres"][0]),
                _sensitivity(noise),
                entry["max_psf_sidelobe"],
            )
        assert timing["n_cycles"].iloc[0] <= max(limits)

    def test_a_noisy_clean_reaches_the_noise(self):
        """Few visibilities (5 antennas, 2 times): sidelobes fill the dirty
        image inside the mask, and its robust noise (1.4826 MAD, 0.17 against
        a thermal noise of 1.6e-5) gave a limit of 9 cycles, which stopped the
        clean about 400 times above the noise. The sensitivity of the weights
        lets the clean reach the noise."""
        noise, n_antenna, n_time = 1e-4, 5, 2
        _, _, record = _image(
            {"max_cycles": "auto", "threshold": 0.0, "max_iter": 100000},
            noise=noise,
            n_antenna=n_antenna,
            n_time=n_time,
        )
        stokes_i = record.data[STOKES_I]
        sensitivity = _sensitivity(noise, n_antenna, n_time)
        assert stokes_i["max_cycles_auto"] == (
            NOISY_CYCLE_SAFETY
            * predicted_imaging_cycles(
                abs(stokes_i["start_peakres"][0]),
                sensitivity,
                stokes_i["max_psf_sidelobe"],
            )
        )
        # the cycles bring the peak within 10 times the noise before the
        # limit can stop them
        assert np.abs(stokes_i["start_peakres"]).min() < 10 * sensitivity

    def test_a_number_adds_no_field(self):
        _, _, record = _image({"max_cycles": 2, "threshold": 0.01})
        for entry in record.data.values():
            assert "max_cycles_auto" not in entry
            for field in ITERATION_HISTORY_FIELDS:
                assert field not in entry

    def test_iteration_history_changes_no_image(self):
        params = {"max_cycles": 3, "threshold": 0.01}
        img_off, _, record_off = _image(params)
        img_on, _, record_on = _image({**params, "record_iteration_history": True})
        for name in ("SKY_RESIDUAL", "SKY_MODEL"):
            np.testing.assert_array_equal(img_on[name].values, img_off[name].values)
        for key, entry in record_on.data.items():
            assert entry["iter_done"] == record_off.data[key]["iter_done"]
            for field in ITERATION_HISTORY_FIELDS:
                assert [len(cycle) for cycle in entry[field]] == entry["iter_done"]
        # the peaks a model update cleaned start at the peak it started from;
        # the closing stop test ran no iteration
        stokes_i = record_on.data[STOKES_I]
        *histories, closing = stokes_i["iteration_peak_residual"]
        assert closing.shape == (0,)
        for start_peak, peaks in zip(stokes_i["start_peakres"], histories, strict=True):
            assert peaks[0] == pytest.approx(start_peak, rel=1e-12)

    def test_cycle_without_model_update_has_an_empty_history(self):
        _, _, record = _image({"max_cycles": 0, "record_iteration_history": True})
        for entry in record.data.values():
            assert entry["iter_done"] == [0]
            for field in ITERATION_HISTORY_FIELDS:
                (history,) = entry[field]
                assert history.shape == (0,)

    def test_asp_refuses_the_history_before_imaging(self):
        with pytest.raises(ValueError, match="record_iteration_history"):
            _image({"record_iteration_history": True}, deconvolver="asp")


# ---------------------------------------------------------------------------
# The stop test on the residual of every residual update
# ---------------------------------------------------------------------------


def _spy_on_the_loop(monkeypatch, on_record=None, on_model_update=None):
    """Spy on the functions the imaging cycle loop calls, through the module
    attributes it imports them from. ``events`` lists the calls; the residual
    of every residual update, the record of every stop test, the model after
    every model update and the residual the restore gets are kept.
    ``on_record(record, n)`` and ``on_model_update(record)`` may change a
    record before the loop sees it."""
    import importlib

    imaging = "astroviper.processing_functions.imaging"
    residual_update = importlib.import_module(f"{imaging}.residual_update")
    model_update = importlib.import_module(f"{imaging}.model_update")
    utils = importlib.import_module(f"{imaging}.utils")
    restore = importlib.import_module(f"{imaging}.restore")
    original_ru = residual_update.residual_update_cube_single_field
    original_record = utils.build_residual_imaging_dict
    original_mu = model_update.model_update_cube_single_field
    original_restore = restore.restore_image
    seen = {key: [] for key in ("events", "residuals", "records", "models", "restored")}

    def ru(*args, **kwargs):
        out = original_ru(*args, **kwargs)
        seen["events"].append("RU")
        seen["residuals"].append(out[0]["SKY_RESIDUAL"].values.copy())
        return out

    def record(img_xds, image_data_group_in_name, cycle_control_params):
        rd = original_record(img_xds, image_data_group_in_name, cycle_control_params)
        if on_record is not None:
            on_record(rd, len(seen["records"]))
        seen["events"].append("check")
        seen["records"].append(rd)
        return rd

    def mu(img_xds, *args, **kwargs):
        out = original_mu(img_xds, *args, **kwargs)
        if on_model_update is not None:
            on_model_update(out[0])
        seen["events"].append("MU")
        seen["models"].append(img_xds["SKY_MODEL"].values.copy())
        return out

    def restore_image(img_xds, *args, **kwargs):
        seen["events"].append("restore")
        seen["restored"].append(img_xds["SKY_RESIDUAL"].values.copy())
        return original_restore(img_xds, *args, **kwargs)

    monkeypatch.setattr(residual_update, "residual_update_cube_single_field", ru)
    monkeypatch.setattr(utils, "build_residual_imaging_dict", record)
    monkeypatch.setattr(model_update, "model_update_cube_single_field", mu)
    monkeypatch.setattr(restore, "restore_image", restore_image)
    return seen


def _masked_peak(img_xds, pol=0):
    """Largest absolute value of the written residual inside the clean mask."""
    residual = img_xds["SKY_RESIDUAL"].values[0, 0, pol]
    return np.abs(np.where(img_xds["MASK"].values[0, 0, pol], residual, 0.0)).max()


def _diverged(imaging_dict):
    for entry in imaging_dict.data.values():
        entry["stop_code"] = StopCode(IMAGING_CONTINUE, MODEL_UPDATE_DIVERGENCE)


def _entropy_falls(record, n):
    for entry in record.data.values():
        entry["entropy"] = [5.0 if n == 0 else 4.0]


def _blank_mask(record, n):
    for entry in record.data.values():
        entry["masksum"] = [0]


class TestTheStopTestIsOnTheFreshResidual:
    """The stop test is made after every residual update, on the residual it
    has just made; the residual a model update ends with decides nothing."""

    def test_the_model_updates_own_peak_does_not_stop_the_imaging(self, monkeypatch):
        def converged(imaging_dict):
            for entry in imaging_dict.data.values():
                entry["peakres"] = [0.0]
                entry["stop_code"] = StopCode(
                    IMAGING_CONTINUE, MODEL_UPDATE_THRESHOLD_PER_CYCLE
                )

        seen = _spy_on_the_loop(monkeypatch, on_model_update=converged)
        img, timing, record = _image(
            {"threshold": 0.01, "max_iter_per_cycle": 5, "max_cycles": 3}
        )
        stokes_i = record.data[STOKES_I]
        assert stokes_i["stop_code"] == StopCode(
            IMAGING_MAX_CYCLES, MODEL_UPDATE_THRESHOLD_PER_CYCLE
        )
        assert seen["events"].count("MU") == timing["n_cycles"].iloc[0] == 3
        assert seen["events"].count("RU") == 4
        assert stokes_i["peakres"][:3] == [0.0, 0.0, 0.0]
        assert stokes_i["peakres"][3] == _masked_peak(img) > 0.01

    @pytest.mark.parametrize(
        ("controls", "spies", "code", "n_model_updates"),
        [
            ({"max_iter": 0}, {}, IMAGING_MAX_ITER, 0),
            ({"max_cycles": 0}, {}, IMAGING_MAX_CYCLES, 0),
            (
                {"max_iter": 6, "max_iter_per_cycle": 2, "threshold": 1e-6},
                {},
                IMAGING_MAX_ITER,
                None,
            ),
            ({"max_cycles": 2, "max_iter_per_cycle": 2}, {}, IMAGING_MAX_CYCLES, 2),
            ({"threshold": 0.01}, {}, IMAGING_THRESHOLD, None),
            (
                {"max_iter_per_cycle": 0, "threshold": 1e-6},
                {},
                IMAGING_NO_PROGRESS,
                2,
            ),
            (
                {"max_cycles_divergence": 1, "threshold": 1e-6},
                {"on_model_update": _diverged},
                IMAGING_DIVERGENCE,
                1,
            ),
            (
                {"entropy_stop": True, "threshold": 1e-6},
                {"on_record": _entropy_falls},
                IMAGING_ENTROPY,
                1,
            ),
            ({}, {"on_record": _blank_mask}, IMAGING_ZERO_MASK, 0),
        ],
        ids=[
            "max_iter_0",
            "max_cycles_0",
            "max_iter",
            "max_cycles",
            "threshold",
            "no_progress",
            "divergence",
            "entropy",
            "blank_mask",
        ],
    )
    def test_one_residual_update_more_than_model_updates(
        self, monkeypatch, controls, spies, code, n_model_updates
    ):
        seen = _spy_on_the_loop(monkeypatch, **spies)
        _, timing, record = _image(controls)
        n_mu = seen["events"].count("MU")
        assert seen["events"].count("RU") == len(seen["records"]) == n_mu + 1
        assert timing["n_cycles"].iloc[0] == n_mu
        stokes_i = record.data[STOKES_I]
        assert len(stokes_i["iter_done"]) == n_mu + 1
        assert stokes_i["iter_done"][-1] == 0
        assert (imaging_dict_to_dataframe(record).n_cycles == n_mu).all()
        assert stokes_i["stop_code"].imaging == code
        # the stop test is timed on its own, outside the model update
        assert timing["T_stop_test"].iloc[0] > 0
        plural = "s" if n_mu != 1 else ""
        assert (
            f"Per-cycle history ({n_mu} cycle{plural} and the closing stop test)"
            in format_imaging_dict(record)
        )
        if n_model_updates is not None:
            assert n_mu == n_model_updates

    def test_the_images_come_from_the_last_residual_update(self, monkeypatch):
        seen = _spy_on_the_loop(monkeypatch)
        img, _, record = _image(
            {"max_cycles": 3, "max_iter_per_cycle": 5, "threshold": 1e-6},
            restore=True,
        )
        assert seen["events"] == ["RU", "check", "MU"] * 3 + ["RU", "check", "restore"]
        np.testing.assert_array_equal(img["SKY_RESIDUAL"].values, seen["residuals"][-1])
        np.testing.assert_array_equal(seen["restored"][0], seen["residuals"][-1])
        np.testing.assert_array_equal(img["SKY_MODEL"].values, seen["models"][-1])
        # every model update starts from the peak its stop test saw, and the
        # record closes with the peak of the residual written
        stokes_i = record.data[STOKES_I]
        tested = [rd.data[STOKES_I]["peakres"][0] for rd in seen["records"]]
        assert [abs(peak) for peak in stokes_i["start_peakres"]] == tested[:-1]
        assert stokes_i["peakres"][-1] == tested[-1] == _masked_peak(img)


class TestThePerCycleThresholdOfTheFreshResidual:
    """Every model update cleans down to its PSF fraction of the peak the stop
    test before it measured, not of the peak the previous model update ended
    with."""

    def _run(self, monkeypatch, on_model_update=None):
        used = []

        def spy(imaging_dict):
            entry = imaging_dict.data[STOKES_I]
            used.append((entry["threshold_per_cycle"], entry["peakres"][0]))
            if on_model_update is not None:
                on_model_update(imaging_dict)

        seen = _spy_on_the_loop(monkeypatch, on_model_update=spy)
        _, _, record = _image(
            {"max_cycles": 3, "max_iter_per_cycle": 5, "threshold": 1e-6}
        )
        stokes_i = record.data[STOKES_I]
        fraction = stokes_i["max_psf_sidelobe"]  # inside the clamp here
        tested = [rd.data[STOKES_I]["peakres"][0] for rd in seen["records"]]
        expected = [fraction * peak for peak in tested[:-1]]
        return used, tested, expected, stokes_i

    def test_the_base_is_the_peak_of_the_stop_test(self, monkeypatch):
        used, tested, expected, _ = self._run(monkeypatch)
        thresholds, own_peaks = (list(values) for values in zip(*used, strict=True))
        assert thresholds == pytest.approx(expected, rel=1e-12)
        # the residual update moves the peak a model update ended with (here
        # by 2e-8 to 5e-7 of it)
        for own, fresh in zip(own_peaks, tested[1:], strict=True):
            assert abs(own) != fresh

    def test_a_diverged_model_update_leaves_a_finite_threshold(self, monkeypatch):
        def diverged_to_infinity(imaging_dict):
            for entry in imaging_dict.data.values():
                entry["peakres"] = [np.inf]
                entry["stop_code"] = StopCode(IMAGING_CONTINUE, MODEL_UPDATE_DIVERGENCE)

        used, _, expected, stokes_i = self._run(monkeypatch, diverged_to_infinity)
        assert [threshold for threshold, _ in used] == pytest.approx(
            expected, rel=1e-12
        )
        assert stokes_i["iter_done"] == [5, 5, 5, 0]


class TestTheRecordOfTheStopTest:
    """The peak of the record is taken inside the clean mask, the mask the
    first model update makes or the mask of the residual group."""

    PARAMS = {"gain": 0.1, "primary_beam_limit": 0.2}

    def test_the_mask_the_first_model_update_makes(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            build_residual_imaging_dict,
        )

        xds, inside = _dirty_dataset()
        data_groups = copy.deepcopy(xds.attrs["data_groups"])
        record = build_residual_imaging_dict(xds, "residual", self.PARAMS)
        stokes_i = record.data[STOKES_I]
        assert stokes_i["peakres"] == [25.0]
        assert stokes_i["peakres_nomask"] == [100.0]
        assert stokes_i["masksum"] == [int(inside.sum())]
        # plane 1 lies wholly below the cutoff
        assert record.data[STOKES_Q]["peakres"] == [0.0]
        assert record.data[STOKES_Q]["masksum"] == [0]
        assert "MASK" not in xds
        assert xds.attrs["data_groups"] == data_groups

    @pytest.mark.parametrize("primary_beam_limit", [0.0, 0.2])
    def test_the_mask_is_the_one_make_mask_makes(self, primary_beam_limit):
        from astroviper.processing_functions.image_analysis.make_mask import (
            make_mask,
        )
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            _clean_mask,
        )

        xds, _ = _dirty_dataset()
        xds["PRIMARY_BEAM"].values[0, 0, 0, 0, :4] = (np.nan, 0.0, -0.5, 0.2)
        mask = _clean_mask(xds, "residual", primary_beam_limit)
        make_mask(
            xds,
            primary_beam_limit=primary_beam_limit,
            image_data_group_in_name="residual",
            image_data_group_out_name="residual",
            overwrite=True,
        )
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, xds["MASK"].values)

    def test_the_mask_of_the_residual_group(self):
        from astroviper.processing_functions.imaging.utils.cycle_control import (
            build_residual_imaging_dict,
        )

        xds, _ = _dirty_dataset(with_mask=True)
        record = build_residual_imaging_dict(xds, "residual", self.PARAMS)
        residual = xds["SKY_RESIDUAL"].values[0, 0]
        for pol, key in enumerate((STOKES_I, STOKES_Q)):
            assert record.data[key]["peakres"] == [
                np.abs(residual[pol, 3:8, 4:16]).max()
            ]
            assert record.data[key]["masksum"] == [5 * 12]
