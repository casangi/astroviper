"""Tests of ms_spectral_frame_conversion on the Antennae TOPO test data.

``Antennae_fld1_topo.ps.zarr``: 70 times over 15 days, 86 baselines, 166
channels of 186.7 kHz at 344.9 GHz, a line simulated at a fixed LSRK frequency.
``Antennae_fld1_casa_lsrk.ps.zarr``: the same data regridded to LSRK by CASA.
"""

import warnings

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import LSRK, SpectralCoord
from astropy.time import Time
from toolviper.utils.data import download

from astroviper.processing_functions.visibility_manipulation import (
    ms_spectral_frame_conversion as conversion,
)
from astroviper.processing_functions.visibility_manipulation.ms_spectral_frame_conversion import (
    doppler_factor,
    get_all_itrs_loc,
    get_phase_center,
    ms_spectral_frame_conversion,
    outframe_freq,
)

TOPO_PS = "Antennae_fld1_topo.ps.zarr"
CASA_PS = "Antennae_fld1_casa_lsrk.ps.zarr"
REGRIDDED = ("VISIBILITY", "WEIGHT", "FLAG")


def _first_ms(path):
    ps = xr.open_datatree(path)
    return ps[next(iter(ps.children))].load()


def _replace(ms, **arrays):
    """Shallow copy of ms with data variables replaced by arrays of the same shape."""
    out = ms.copy()
    dataset = out.to_dataset()
    for name, array in arrays.items():
        dataset[name] = (dataset[name].dims, array, dataset[name].attrs)
    out.dataset = dataset
    return out


def _position(ms, output_frequency):
    """Output channels on the TOPO axis of each time, in input channels (time, channel)."""
    frequency = ms.frequency.values
    shifted = output_frequency[np.newaxis, :] / doppler_factor(ms)[:, np.newaxis]
    return (shifted - frequency[0]) / (frequency[1] - frequency[0])


def _outside(ms, output_frequency, shape):
    position = _position(ms, output_frequency)
    outside = (position < 0) | (position > ms.sizes["frequency"] - 1)
    return np.broadcast_to(outside[:, np.newaxis, :, np.newaxis], shape)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("spectral_frame_conversion")
    for name in (TOPO_PS, CASA_PS):
        download(name, folder=str(folder))
    return folder


@pytest.fixture(scope="module")
def topo_ms(data_dir):
    return _first_ms(data_dir / TOPO_PS)


@pytest.fixture(scope="module")
def lsrk_ms(topo_ms):
    return ms_spectral_frame_conversion(topo_ms)


def test_doppler_factor_is_the_conversion_of_every_channel(topo_ms):
    frequency = topo_ms.frequency.values
    observers = get_all_itrs_loc(topo_ms)
    target = get_phase_center(topo_ms, topo_ms.field_name.values[0])
    doppler = doppler_factor(topo_ms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no distance or velocity on the target
        for t in (0, 35, 69):
            converted = (
                SpectralCoord(frequency * u.Hz, observer=observers[t], target=target)
                .with_observer_stationary_relative_to(LSRK)
                .to_value(u.Hz)
            )
            np.testing.assert_allclose(frequency * doppler[t], converted, rtol=1e-15)


def test_default_grid(topo_ms, lsrk_ms):
    frequency = topo_ms.frequency.values
    doppler = doppler_factor(topo_ms)
    grid = lsrk_ms.frequency.values
    np.testing.assert_array_equal(grid, outframe_freq(topo_ms))
    assert grid.size == frequency.size
    np.testing.assert_allclose(grid[0], frequency.min() * doppler.min(), rtol=1e-15)
    width = (frequency.max() * doppler.max() - grid[0]) / frequency.size
    np.testing.assert_allclose(np.diff(grid), width, rtol=1e-9)


def test_simulated_line_lands_in_channel_74(topo_ms, lsrk_ms):
    times = Time(topo_ms.time.values, format="unix", scale="utc")
    for t in (0, 20, 40, 50):
        spectrum = lsrk_ms.VISIBILITY.sel(
            time=times[t].unix, baseline_id=1, method="nearest"
        )
        np.testing.assert_array_equal(spectrum.argmax(dim="frequency"), 74)


def test_visibilities_are_linear_interpolations(topo_ms, lsrk_ms):
    frequency = topo_ms.frequency.values
    grid = lsrk_ms.frequency.values
    doppler = doppler_factor(topo_ms)
    vis_in = topo_ms.VISIBILITY.values.astype(np.complex128)
    vis_out, flag_out = lsrk_ms.VISIBILITY.values, lsrk_ms.FLAG.values
    tolerance = 1e-6 * np.nanmax(np.abs(vis_in))
    n_checked = 0
    for t in range(0, topo_ms.sizes["time"], 7):
        position = grid / doppler[t]
        for b in range(0, topo_ms.sizes["baseline_id"], 5):
            for p in range(topo_ms.sizes["polarization"]):
                spectrum = vis_in[t, b, :, p]
                expected = np.interp(
                    position, frequency, spectrum.real
                ) + 1j * np.interp(position, frequency, spectrum.imag)
                use = ~flag_out[t, b, :, p] & np.isfinite(expected)
                np.testing.assert_allclose(
                    vis_out[t, b, use, p], expected[use], rtol=0, atol=tolerance
                )
                n_checked += use.sum()
    assert n_checked > 10_000


def test_outside_input_frequencies_flagged_with_zero_data_and_weight(topo_ms, lsrk_ms):
    outside = _outside(topo_ms, lsrk_ms.frequency.values, lsrk_ms.FLAG.shape)
    assert outside.any()
    assert lsrk_ms.FLAG.values[outside].all()
    assert not lsrk_ms.VISIBILITY.values[outside].any()
    assert not lsrk_ms.WEIGHT.values[outside].any()
    assert not np.any((lsrk_ms.VISIBILITY.values == 0) & ~lsrk_ms.FLAG.values)


def test_flag_of_an_input_channel_reaches_the_outputs_interpolated_from_it(topo_ms):
    flag = np.zeros(topo_ms.FLAG.shape, dtype=bool)
    flag[:, :, 80, :] = True
    out = ms_spectral_frame_conversion(_replace(topo_ms, FLAG=flag))
    position = _position(topo_ms, out.frequency.values)
    uses_channel_80 = np.broadcast_to(
        (np.abs(position - 80) < 1)[:, np.newaxis, :, np.newaxis], flag.shape
    )
    outside = _outside(topo_ms, out.frequency.values, flag.shape)
    np.testing.assert_array_equal(out.FLAG.values, uses_channel_80 | outside)


def test_weights_are_interpolated(topo_ms):
    ramp = np.broadcast_to(
        np.arange(topo_ms.sizes["frequency"], dtype=np.float32)[
            np.newaxis, np.newaxis, :, np.newaxis
        ],
        topo_ms.WEIGHT.shape,
    ).copy()
    out = ms_spectral_frame_conversion(_replace(topo_ms, WEIGHT=ramp))
    position = np.broadcast_to(
        _position(topo_ms, out.frequency.values)[:, np.newaxis, :, np.newaxis],
        ramp.shape,
    )
    expected = np.where(
        _outside(topo_ms, out.frequency.values, ramp.shape), 0, position
    )
    np.testing.assert_allclose(out.WEIGHT.values, expected, rtol=1e-6, atol=1e-4)


def test_frequency_chunks_reproduce_the_full_band(topo_ms, lsrk_ms):
    frequency = topo_ms.frequency.values
    grid = lsrk_ms.frequency.values
    doppler = doppler_factor(topo_ms)
    edges = np.linspace(0, grid.size, 5).astype(int)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        # The input channels the block needs at any time, and one more on each side.
        first = max(0, np.searchsorted(frequency, grid[start] / doppler.max()) - 1)
        last = min(
            frequency.size,
            np.searchsorted(frequency, grid[stop - 1] / doppler.min()) + 1,
        )
        assert last - first < frequency.size
        block = ms_spectral_frame_conversion(
            topo_ms.isel(frequency=slice(first, last)),
            output_frequency=grid[start:stop],
        )
        for name in REGRIDDED:
            np.testing.assert_array_equal(
                block[name].values, lsrk_ms[name].values[:, :, start:stop]
            )


def test_time_blocks_reproduce_one_block(topo_ms, lsrk_ms):
    dataset = topo_ms.to_dataset()
    variables = {
        name: dataset[name].copy(data=dataset[name].values) for name in REGRIDDED
    }
    for variable in variables.values():
        variable.encoding = {}
    roles = {"VISIBILITY": "correlated_data", "WEIGHT": "weight", "FLAG": "flag"}
    blocks = conversion._regrid(
        variables,
        roles,
        topo_ms.frequency.values,
        doppler_factor(topo_ms),
        lsrk_ms.frequency.values,
        block_bytes=10**6,
    )
    for name in REGRIDDED:
        np.testing.assert_array_equal(blocks[name].values, lsrk_ms[name].values)


def test_decreasing_frequency_axis(topo_ms, lsrk_ms):
    out = ms_spectral_frame_conversion(topo_ms.isel(frequency=slice(None, None, -1)))
    np.testing.assert_array_equal(out.frequency.values, lsrk_ms.frequency.values)
    for name in REGRIDDED:
        np.testing.assert_array_equal(out[name].values, lsrk_ms[name].values)


def test_single_output_channel(topo_ms, lsrk_ms):
    out = ms_spectral_frame_conversion(
        topo_ms, output_frequency=lsrk_ms.frequency.values[[74]]
    )
    for name in REGRIDDED:
        np.testing.assert_array_equal(
            out[name].values, lsrk_ms[name].values[:, :, [74]]
        )
    np.testing.assert_allclose(
        out.frequency.attrs["channel_width"]["data"],
        topo_ms.frequency.attrs["channel_width"]["data"]
        * doppler_factor(topo_ms).mean(),
    )


def test_lazy_input(data_dir, lsrk_ms):
    ps = xr.open_datatree(data_dir / TOPO_PS)
    out = ms_spectral_frame_conversion(ps[next(iter(ps.children))])
    for name in REGRIDDED:
        np.testing.assert_array_equal(out[name].values, lsrk_ms[name].values)


def test_output_metadata(topo_ms, lsrk_ms):
    attrs = lsrk_ms.frequency.attrs
    assert attrs["observer"] == "lsrk"
    np.testing.assert_allclose(
        attrs["channel_width"]["data"], np.diff(lsrk_ms.frequency.values).mean()
    )
    assert (
        attrs["reference_frequency"] == topo_ms.frequency.attrs["reference_frequency"]
    )
    assert lsrk_ms.attrs["data_groups"] == topo_ms.attrs["data_groups"]
    assert set(lsrk_ms.children) == set(topo_ms.children)
    for name in REGRIDDED:
        assert lsrk_ms[name].dims == topo_ms[name].dims
        assert lsrk_ms[name].dtype == topo_ms[name].dtype
    assert np.shares_memory(lsrk_ms.UVW.values, topo_ms.UVW.values)
    # The input is left as it was.
    assert topo_ms.frequency.attrs["observer"] == "TOPO"
    assert topo_ms.frequency.attrs["channel_width"]["data"] == pytest.approx(
        186730.0244
    )


def test_casa_regrid_onto_the_same_grid(data_dir, topo_ms):
    """Against CASA's LSRK regrid, on CASA's grid, where neither flags the data.

    The Doppler factors of astropy and casacore differ by about 0.05 m/s
    (3e-4 channel), which moves the steep simulated line by that much.
    """
    casa = _first_ms(data_dir / CASA_PS)
    out = ms_spectral_frame_conversion(topo_ms, output_frequency=casa.frequency.values)
    vis, reference = out.VISIBILITY.values, casa.VISIBILITY.values
    both = (
        ~out.FLAG.values & ~casa.FLAG.values & np.isfinite(vis) & np.isfinite(reference)
    )
    assert both.mean() > 0.5
    difference = np.abs(vis[both] - reference[both]) / np.abs(reference[both]).max()
    assert difference.max() < 2e-3
    assert np.sqrt(np.mean(difference**2)) < 2e-4


def test_refusals(topo_ms):
    with pytest.raises(NotImplementedError, match="freqrange"):
        ms_spectral_frame_conversion(topo_ms, freqrange=[3.4e11, 3.5e11])
    with pytest.raises(ValueError, match="Unsupported spectral frame"):
        ms_spectral_frame_conversion(topo_ms, outframe="LSRK")
    with pytest.raises(ValueError, match="at least two channels"):
        ms_spectral_frame_conversion(topo_ms.isel(frequency=[0]))
    with pytest.raises(ValueError, match="strictly increasing or decreasing"):
        ms_spectral_frame_conversion(topo_ms.isel(frequency=[0, 2, 1]))
    with pytest.raises(ValueError, match="output_frequency"):
        ms_spectral_frame_conversion(topo_ms, output_frequency=np.ones((2, 2)))

    dataset = topo_ms.to_dataset()
    for modified, message in (
        (
            dataset.assign_coords(
                frequency=topo_ms.frequency.assign_attrs(observer="lsrk")
            ),
            "must be TOPO",
        ),
        (
            dataset.assign_coords(
                frequency=topo_ms.frequency.assign_attrs(units="GHz")
            ),
            "must be in Hz",
        ),
        (
            dataset.assign(EFFECTIVE_CHANNEL_WIDTH=topo_ms.WEIGHT.astype(np.float64)),
            "EFFECTIVE_CHANNEL_WIDTH",
        ),
        (
            dataset.assign(WEIGHT=topo_ms.WEIGHT.transpose("baseline_id", "time", ...)),
            "time as its first dimension",
        ),
        (
            dataset.assign_coords(
                channel=("frequency", np.arange(topo_ms.sizes["frequency"]))
            ),
            "Coordinate channel",
        ),
    ):
        ms = topo_ms.copy()
        ms.dataset = modified
        with pytest.raises(ValueError, match=message):
            ms_spectral_frame_conversion(ms)
