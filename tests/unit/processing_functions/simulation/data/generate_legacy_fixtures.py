"""Generate reference fixtures from the legacy SIRIUS simulator.

The fixtures in this directory (``legacy_*.npz``) were produced by the original
SIRIUS code (https://github.com/casangi/sirius, commit 42ecf56) and are the oracle
for the AstroVIPER ``simulation`` port: the ported processing functions must
reproduce SIRIUS' visibilities, uvw, parallactic angles and beam images.

SIRIUS predates numpy 2 / python 3.13, so it is *not* a test dependency.  To
regenerate the fixtures, build a throw-away environment::

    python -m venv --system-site-packages legacy_env   # python with numba, astropy, xarray
    legacy_env/bin/pip install --no-deps numba-scipy
    SIRIUS_SRC=/path/to/sirius legacy_env/bin/python generate_legacy_fixtures.py

``SIRIUS_SRC`` must point at a checkout of SIRIUS (``sirius/`` and ``sirius_data/``
packages).  The script shims the removed numpy aliases SIRIUS relies on.
"""

import os
import sys

import numpy as np

SIRIUS_SRC = os.environ.get("SIRIUS_SRC", "/Users/jsteeb/Dropbox/viper_dev/sirius")
OUT_DIR = os.path.dirname(__file__)

# --- shims for the 2022-era SIRIUS code -------------------------------------
np.int = int  # noqa: NPY001
np.float = float  # noqa: NPY001
np.complex = complex  # noqa: NPY001
sys.path.insert(0, SIRIUS_SRC)

import sirius.calc_vis as _legacy_calc_vis  # noqa: E402
import xarray as xr  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from sirius.calc_beam import evaluate_beam_models  # noqa: E402
from sirius.calc_noise import calc_noise_chunk  # noqa: E402
from sirius.calc_uvw import calc_uvw_chunk  # noqa: E402
from sirius.calc_vis import calc_vis_chunk  # noqa: E402
from sirius_data.beam_1d_func_models.airy_disk import aca, alma, vla  # noqa: E402

_orig_to_tuple = _legacy_calc_vis._beam_models_to_tuple


def _writeable_beam_models_to_tuple(beam_models):
    # numba requires homogeneous tuple member types when beam models are mixed;
    # xarray coordinate values are read-only arrays, so copy them.
    return tuple(
        tuple(np.array(x) if isinstance(x, np.ndarray) else x for x in bm)
        for bm in _orig_to_tuple(beam_models)
    )


_legacy_calc_vis._beam_models_to_tuple = _writeable_beam_models_to_tuple

DATA = os.path.join(SIRIUS_SRC, "sirius_data")
TEL_VLA_D = os.path.join(DATA, "telescope_layout/data/vla.d.tel.zarr")
TEL_ALMA = os.path.join(DATA, "telescope_layout/data/alma.all.tel.zarr")
APC_SBAND = os.path.join(
    DATA,
    "aperture_polynomial_coefficient_models/data/EVLA_avg_zcoeffs_SBand_lookup.apc.zarr",
)
BPC_EVLA = os.path.join(DATA, "beam_polynomial_coefficient_models/data/EVLA_.bpc.zarr")

DEFAULT_BEAM_PARMS = {
    "mueller_selection": np.array([0, 5, 10, 15]),
    "pa_radius": 0.2,
    "image_size": np.array([1000, 1000]),
    "fov_scaling": 4.0,
    "zernike_freq_interp": "nearest",
}


def ra_dec(ra, dec):
    c = SkyCoord(ra=ra, dec=dec, frame="fk5")
    return np.array([c.ra.rad, c.dec.rad])


def load_tel(path):
    return xr.open_zarr(path, consolidated=False).load()


def run_scenario(
    name,
    tel_xds,
    time_chunk,
    chan_chunk,
    pol,
    point_source_flux,
    point_source_ra_dec,
    phase_center_ra_dec,
    beam_models,
    beam_model_map,
    beam_parms,
    pointing_ra_dec=None,
    noise_parms=None,
    auto_corr=False,
    save_beam=False,
):
    pol = np.array(pol)
    uvw_parms = {"calc_method": "astropy", "auto_corr": auto_corr}
    if phase_center_ra_dec.shape[0] == 1:
        uvw, antenna1, antenna2 = calc_uvw_chunk(
            tel_xds, time_chunk, phase_center_ra_dec, uvw_parms, check_parms=False
        )
    else:
        # legacy astropy uvw only supports a single phase center per call
        uvw_list = []
        for i_time in range(len(time_chunk)):
            uvw_i, antenna1, antenna2 = calc_uvw_chunk(
                tel_xds,
                time_chunk[i_time : i_time + 1],
                phase_center_ra_dec[i_time : i_time + 1],
                uvw_parms,
                check_parms=False,
            )
            uvw_list.append(uvw_i)
        uvw = np.concatenate(uvw_list, axis=0)
    eval_beam_models, parallactic_angle = evaluate_beam_models(
        beam_models,
        time_chunk,
        chan_chunk,
        phase_center_ra_dec,
        tel_xds.site_pos[0],
        beam_parms,
        check_parms=False,
    )
    vis_shape = np.array([uvw.shape[0], uvw.shape[1], len(chan_chunk), len(pol)])
    vis = calc_vis_chunk(
        uvw,
        vis_shape,
        point_source_flux,
        point_source_ra_dec,
        pointing_ra_dec,
        phase_center_ra_dec,
        antenna1,
        antenna2,
        chan_chunk,
        beam_model_map,
        eval_beam_models,
        parallactic_angle,
        pol,
        beam_parms["mueller_selection"],
        check_parms=False,
    )
    out = {
        "time": time_chunk,
        "frequency": chan_chunk,
        "polarization": np.array(pol),
        "point_source_flux": point_source_flux,
        "point_source_ra_dec": point_source_ra_dec,
        "phase_center_ra_dec": phase_center_ra_dec,
        "pointing_ra_dec": (
            pointing_ra_dec if pointing_ra_dec is not None else np.zeros(0)
        ),
        "antenna_position": tel_xds.ANT_POS.values,
        "antenna_name": tel_xds.ant_name.values,
        "dish_diameter": tel_xds.DISH_DIAMETER.values,
        "site_position": np.array(
            [tel_xds.site_pos[0][k]["value"] for k in ("m0", "m1", "m2")]
        ),
        "beam_model_map": np.asarray(beam_model_map),
        "mueller_selection": beam_parms["mueller_selection"],
        "uvw": uvw,
        "antenna1": antenna1,
        "antenna2": antenna2,
        "parallactic_angle": parallactic_angle,
        "visibility": vis,
    }
    if noise_parms is not None:
        np.random.seed(42)
        noise, weight, sigma = calc_noise_chunk(
            vis.shape,
            uvw,
            beam_model_map,
            eval_beam_models,
            antenna1,
            antenna2,
            noise_parms,
            check_parms=False,
        )
        out["noise_weight"] = weight
        out["noise_sigma"] = sigma
        out["noise_std"] = np.array([noise.real.std(), noise.imag.std()])
    if save_beam:
        J_xds = eval_beam_models[0]
        out["beam_pa"] = J_xds.pa.values
        out["beam_frequency"] = J_xds.chan.values
        out["beam_polarization"] = J_xds.pol.values
        out["beam_l"] = J_xds.l.values
        out["beam_m"] = J_xds.m.values
        # subsample the (pa, chan, pol, l, m) Jones image to keep the fixture small
        out["beam_jones_subsampled"] = J_xds.J.values[:, :, :, ::25, ::25]
        out["beam_jones_abs_max"] = np.abs(J_xds.J.values).max(axis=(3, 4))
        out["beam_jones_center"] = J_xds.J.values[
            :, :, :, J_xds.sizes["l"] // 2, J_xds.sizes["m"] // 2
        ]
    path = os.path.join(OUT_DIR, f"legacy_{name}.npz")
    np.savez_compressed(path, **out)
    print(name, "->", path, "vis", vis.shape, "|vis| max", np.abs(vis).max())


def main():
    vla_d = load_tel(TEL_VLA_D)
    n_ant_vla = vla_d.sizes["ant_name"]
    times3 = np.array(
        [
            "2019-10-03T19:00:00.000",
            "2019-10-03T20:00:00.000",
            "2019-10-03T21:00:00.000",
        ]
    )
    chans2 = np.array([3.0e9, 3.4e9])
    pc = ra_dec("19h59m28.5s", "+40d44m01.5s")
    src = ra_dec("19h59m50.51793355s", "+40d48m11.3694551s")
    src2 = ra_dec("19h59m10.0s", "+40d40m00.0s")

    # 1. VLA-D, single Airy model, RR/LL, one offset source
    run_scenario(
        "vla_airy",
        vla_d,
        times3,
        chans2,
        [5, 8],
        np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        src[None, None, :],
        pc[None, :],
        [vla],
        np.zeros(n_ant_vla, int),
        DEFAULT_BEAM_PARMS,
    )

    # 2. VLA-D, Airy, two sources with time/chan dependent flux, full 4 pol,
    #    per-antenna pointing offsets
    rng = np.random.default_rng(1)
    flux = np.zeros((2, 3, 2, 4))
    flux[0] = np.array([1.0, 0.1, 0.1, 0.9]) * np.linspace(1, 1.2, 3)[:, None, None]
    flux[1] = np.array([0.5, 0.0, 0.0, 0.5]) * np.array([1.0, 0.8])[None, :, None]
    pointing = np.tile(pc, (1, n_ant_vla, 1)) + rng.normal(0, 0.5e-3, (1, n_ant_vla, 2))
    run_scenario(
        "vla_airy_pointing",
        vla_d,
        times3,
        chans2,
        [5, 6, 7, 8],
        flux,
        np.stack([src, src2])[None, :, :],
        pc[None, :],
        [vla],
        np.zeros(n_ant_vla, int),
        {**DEFAULT_BEAM_PARMS, "mueller_selection": np.arange(16)},
        pointing_ra_dec=pointing,
    )

    # 3. ALMA heterogeneous array (4 x 7 m + 8 x 12 m), XX/YY, 2 fields, noise
    alma_all = load_tel(TEL_ALMA)
    sel = np.concatenate(
        [
            np.where(alma_all.DISH_DIAMETER.values == 7)[0][:4],
            np.where(alma_all.DISH_DIAMETER.values == 12)[0][:8],
        ]
    )
    alma_het = alma_all.isel(ant_name=sel)
    dish = alma_het.DISH_DIAMETER.values
    bmm = np.where(dish == 7, 0, 1)
    times4 = np.array(
        [
            "2019-10-03T19:00:00.000",
            "2019-10-03T19:33:20.000",
            "2019-10-03T20:06:40.000",
            "2019-10-03T20:40:00.000",
        ]
    )
    chans3 = np.array([90.0e9, 90.5e9, 91.0e9])
    pc1 = ra_dec("19h59m28.5s", "-40d44m01.5s")
    pc2 = ra_dec("19h59m28.5s", "-40d44m51.5s")
    src_a = ra_dec("19h59m28.5s", "-40d44m21.5s")
    src_b = ra_dec("19h59m30.0s", "-40d44m31.5s")
    alma_params = dict(aca)
    run_scenario(
        "alma_het_mosaic_noise",
        alma_het,
        times4,
        chans3,
        [9, 12],
        np.array([[1.0, 0, 0, 1.0], [0.5, 0, 0, 0.5]])[:, None, None, :],
        np.stack([src_a, src_b])[None, :, :],
        np.array([pc1, pc1, pc2, pc2]),
        [alma_params, dict(alma)],
        bmm,
        DEFAULT_BEAM_PARMS,
        noise_parms={
            "mode": "tsys-manual",
            "t_atmos": 250.0,
            "tau": 0.1,
            "ant_efficiency": 0.8,
            "spill_efficiency": 0.85,
            "corr_efficiency": 0.88,
            "quantization_efficiency": 0.96,
            "t_receiver": 50.0,
            "t_cmb": 2.725,
            "auto_corr": False,
            "freq_resolution": 0.5e9,
            "time_delta": 2000.0,
        },
    )

    # 4. EVLA polynomial (BPC) beam model
    bpc_xds = xr.open_zarr(BPC_EVLA, consolidated=False).load()
    run_scenario(
        "evla_polynomial_beam",
        vla_d,
        times3,
        chans2,
        [5, 8],
        np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        src[None, None, :],
        pc[None, :],
        [bpc_xds],
        np.zeros(n_ant_vla, int),
        DEFAULT_BEAM_PARMS,
    )

    # 5. EVLA Zernike (APC) beam model, all 16 Mueller terms, 4 pol, small beam image
    zpc_xds = xr.open_zarr(APC_SBAND, consolidated=False).load()
    # numba needs homogeneous (writeable) tuple members when models are mixed
    zpc_xds = zpc_xds.copy(
        data={k: np.array(v.values) for k, v in zpc_xds.data_vars.items()}
    )
    zpc_parms = {
        **DEFAULT_BEAM_PARMS,
        "mueller_selection": np.arange(16),
        "image_size": np.array([500, 500]),
    }
    # wide time span so that several parallactic-angle groups are used
    times_pa = np.array(
        [
            "2019-10-03T16:00:00.000",
            "2019-10-03T18:00:00.000",
            "2019-10-03T20:00:00.000",
            "2019-10-03T22:00:00.000",
        ]
    )
    run_scenario(
        "evla_zernike_beam",
        vla_d,
        times_pa,
        np.array([3.0e9]),
        [5, 6, 7, 8],
        np.array([[1.0, 0, 0, 1.0], [0.5, 0, 0, 0.5]])[:, None, None, :],
        np.stack([src, src2])[None, :, :],
        pc[None, :],
        [zpc_xds],
        np.zeros(n_ant_vla, int),
        zpc_parms,
        save_beam=True,
    )

    # 6. mixed models: Zernike for antenna 0..9, Airy for the rest (Mueller path vs product path)
    bmm_mixed = np.zeros(n_ant_vla, int)
    bmm_mixed[10:] = 1
    run_scenario(
        "evla_mixed_beams",
        vla_d,
        times3,
        np.array([3.0e9]),
        [5, 8],
        np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        src[None, None, :],
        pc[None, :],
        [zpc_xds, vla],
        bmm_mixed,
        zpc_parms,
    )


if __name__ == "__main__":
    main()
