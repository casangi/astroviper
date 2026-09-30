"""Generate the CASA ``immoments`` reference fixture for ``test_moments_casa.py``.

Run MANUALLY in an environment that has ``casatools`` (the test suite itself
must not depend on CASA):

    python tests/component/data/generate_casa_moments_reference.py

Writes ``casa_moments_reference.npz`` next to this script: the synthetic line
cube, the CASA default ``fromarray`` frequency axis, the rest frequency, and
every CASA moment plane (+ mask) the test compares against.  Regenerate only
when the comparison scenario changes.
"""

import os

import numpy as np

INCLUDE_RANGE = (0.1, 1e30)
MOMENT_CODES = {
    "mean": -1,
    "integrated": 0,
    "weighted_coord": 1,
    "median": 3,
    "standard_deviation": 5,
    "rms": 6,
    "abs_mean_dev": 7,
    "maximum": 8,
    "maximum_coord": 9,
    "minimum": 10,
    "minimum_coord": 11,
}
INCLUDED_CODES = {"weighted_coord_included": 1, "weighted_dispersion_coord_included": 2}


def make_sky():
    """The synthetic line cube of the comparison (single source of truth)."""
    n_chan, n_l, n_m = 24, 48, 40
    rng = np.random.default_rng(11)
    l_idx, m_idx = np.meshgrid(np.arange(n_l), np.arange(n_m), indexing="ij")
    spatial = np.exp(
        -(((l_idx - n_l / 2) ** 2 + (m_idx - n_m / 2) ** 2) / (2 * 6.0**2))
    )
    chan = np.arange(n_chan)
    center = n_chan / 2 + 4 * (l_idx - n_l / 2) / n_l
    profile = np.exp(-((chan[:, None, None] - center[None]) ** 2) / (2 * 2.5**2))
    return (spatial[None] * profile + rng.normal(0, 0.01, profile.shape)).astype(
        np.float64
    )  # (chan, l, m); no NaNs so CASA and AstroVIPER see identical pixels


def main():
    import shutil
    import tempfile

    import casatools

    sky = make_sky()
    n_chan = sky.shape[0]
    work = tempfile.mkdtemp(prefix="casa_moments_ref_")
    casa_image = os.path.join(work, "casa_input.im")

    ia = casatools.image()
    ia.fromarray(
        outfile=casa_image,
        pixels=np.transpose(sky, (1, 2, 0))[:, :, np.newaxis, :],
        overwrite=True,
    )
    frequency = np.array(
        [ia.toworld([0, 0, 0, c])["numeric"][3] for c in range(n_chan)]
    )
    # Rest frequency at band centre (keeps the float32 moment velocities small;
    # see the test module docstring).
    rest_frequency = float(frequency.mean())
    cs = ia.coordsys()
    cs.setrestfrequency(value=f"{rest_frequency}Hz")
    ia.setcoordsys(cs.torecord())
    cs.done()
    ia.close()

    out = {"sky": sky, "frequency": frequency, "rest_frequency": rest_frequency}

    def run(name, code, includepix=None):
        ia = casatools.image()
        ia.open(casa_image)
        kwargs = {} if includepix is None else {"includepix": includepix}
        moment_image = ia.moments(
            moments=[code],
            axis=3,
            outfile=os.path.join(work, f"casa_{name}.im"),
            drop=False,
            **kwargs,
        )
        ia.close()
        out[f"{name}_plane"] = np.squeeze(moment_image.getchunk())
        out[f"{name}_mask"] = np.squeeze(moment_image.getchunk(getmask=True))
        moment_image.close()

    for name, code in MOMENT_CODES.items():
        run(name, code)
    for name, code in INCLUDED_CODES.items():
        run(name, code, includepix=list(INCLUDE_RANGE))

    target = os.path.join(os.path.dirname(__file__),
                          "casa_moments_reference.npz")  # fmt: skip
    np.savez_compressed(target, **out)
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", target, os.path.getsize(target) // 1024, "KiB")


if __name__ == "__main__":
    main()
