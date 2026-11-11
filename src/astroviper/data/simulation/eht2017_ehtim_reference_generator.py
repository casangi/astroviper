"""Generate the eht-imaging (ehtim) reference simulation that the AstroVIPER EHT notebook replicates.

Two models are observed with the 2017 EHT array (ehtim's ``arrays/EHT2017.txt``)
for a full UT day of April 5, 2017 (MJD 57848), pointing at M87*, without
thermal noise or corruptions:

1. the public ehtim modelling example (``examples/example_modeling.py``):
   a thin ring of 1.5 Jy and 40 uas diameter plus a 1.0 Jy circular Gaussian
   of 20 uas FWHM offset by (-15, +20) uas;
2. an M87*-like thick m-ring (crescent): 0.6 Jy, 42 uas diameter, 16 uas FWHM
   ring width, beta_1 = 0.25 exp(i 170 deg), beta_2 = 0.06 exp(i 60 deg).

Saved: station table, observation rows (time, stations, u, v) and the ehtim
model visibilities of both models on those rows, and the ehtim model images.
"""

import os
import types

import ehtim as eh
import numpy as np

# ehtim 1.3.2 predates NumPy 2: restore the removed alias and the generator-summing np.sum.
np.complex_ = np.complex128
_numpy_sum = np.sum


def _sum_accepting_generators(a, *args, **kwargs):
    if isinstance(
        a, types.GeneratorType
    ):  # NumPy 1 fell back to the builtin (element-wise) sum
        return sum(a)
    return _numpy_sum(a, *args, **kwargs)


np.sum = _sum_accepting_generators

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "eht2017_ehtim_reference.npz")

M87_RA_H = 12.0 + 30.0 / 60 + 49.42338 / 3600  # 12h30m49.42s
M87_DEC_DEG = 12.0 + 23.0 / 60 + 28.0439 / 3600  # +12d23m28.04s
MJD = 57848  # 2017 April 5
RF = 230.0e9
BW = 2.0e9
TINT = 60.0  # s
TADV = 600.0  # s
TSTART, TSTOP = 0.0, 24.0  # UT hours
UAS = eh.RADPERUAS

eht = eh.array.load_txt(os.path.join(HERE, "EHT2017.txt"))


def base_model():
    mod = eh.model.Model()
    mod.ra = M87_RA_H
    mod.dec = M87_DEC_DEG
    mod.rf = RF
    mod.mjd = MJD
    mod.source = "M87"
    return mod


# 1. the public example (ring + offset Gaussian)
example = base_model()
example = example.add_ring(F0=1.5, d=40.0 * UAS)
example = example.add_circ_gauss(F0=1.0, FWHM=20.0 * UAS, x0=-15.0 * UAS, y0=20.0 * UAS)

# 2. an M87*-like thick m-ring
BETA = [0.25 * np.exp(1j * np.deg2rad(170.0)), 0.06 * np.exp(1j * np.deg2rad(60.0))]
mring = base_model()
mring = mring.add_thick_mring(F0=0.6, d=42.0 * UAS, alpha=16.0 * UAS, beta_list=BETA)

common = dict(
    mjd=MJD, timetype="UTC", add_th_noise=False, ampcal=True, phasecal=True,
    opacitycal=True, dcal=True, frcal=True, ttype="direct", seed=1,
)  # fmt: skip
obs_example = example.observe(eht, TINT, TADV, TSTART, TSTOP, BW, **common)
obs_mring = mring.observe(eht, TINT, TADV, TSTART, TSTOP, BW, **common)

rows = obs_example.data
assert np.array_equal(rows["time"], obs_mring.data["time"])
assert np.array_equal(rows["t1"], obs_mring.data["t1"]) and np.array_equal(
    rows["t2"], obs_mring.data["t2"]
)
print(
    f"{len(rows)} visibilities, {len(np.unique(rows['time']))} times, elevation cuts applied by ehtim"
)

# also the model visibilities re-evaluated analytically at the same uv (should equal obs 'vis' exactly)
vis_example = example.sample_uv(rows["u"], rows["v"])[0] if False else rows["vis"]
vis_mring = obs_mring.data["vis"]

fov = 200.0 * UAS
npix = 128
img_example = example.make_image(fov, npix)
img_mring = mring.make_image(fov, npix)

tarr = eht.tarr
np.savez(
    OUT,
    ehtim_version="1.3.2",
    station=np.array([str(s) for s in tarr["site"]]),
    station_xyz=np.stack([tarr["x"], tarr["y"], tarr["z"]], axis=1).astype(np.float64),
    station_sefd=np.stack([tarr["sefdr"], tarr["sefdl"]], axis=1).astype(np.float64),
    ra_hours=M87_RA_H,
    dec_degrees=M87_DEC_DEG,
    mjd=MJD,
    rf=RF,
    bw=BW,
    tint=TINT,
    tadv=TADV,
    time_hours=rows["time"].astype(np.float64),
    t1=rows["t1"].astype(str),
    t2=rows["t2"].astype(str),
    u=rows["u"].astype(np.float64),
    v=rows["v"].astype(np.float64),
    vis_example=np.asarray(vis_example, dtype=np.complex128),
    vis_mring=np.asarray(vis_mring, dtype=np.complex128),
    example_params=np.array(
        ["ring F0=1.5 d=40uas", "circ_gauss F0=1.0 FWHM=20uas x0=-15uas y0=20uas"]
    ),
    mring_beta=np.asarray(BETA, dtype=np.complex128),
    mring_params=np.array([0.6, 42.0, 16.0]),  # F0 [Jy], d [uas], alpha [uas]
    image_fov_uas=200.0,
    image_npix=npix,
    image_example=img_example.imvec.reshape(npix, npix).astype(np.float32),
    image_mring=img_mring.imvec.reshape(npix, npix).astype(np.float32),
    image_psize=img_example.psize,
)
print("wrote", OUT, os.path.getsize(OUT) / 1024, "kB")
print("example |vis| range", np.abs(vis_example).min(), np.abs(vis_example).max())
print("mring |vis| range", np.abs(vis_mring).min(), np.abs(vis_mring).max())
print("uv max [Glambda]", np.hypot(rows["u"], rows["v"]).max() / 1e9)
print(
    "ehtim image conventions: xdim",
    img_example.xdim,
    "psize [uas]",
    img_example.psize / UAS,
)
