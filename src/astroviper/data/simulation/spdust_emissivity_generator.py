"""Generate the SpDust emissivity table that the AstroVIPER ALMA SED notebook reads.

Yoon et al. 2026, ApJ 1004, 157, Section 3.5: SpDust (Ali-Haimoud et al. 2009) for a
CNM, WNM and PDR (the paper cites Draine 2011 for these conditions; the values below
are Draine & Lazarian 1998b Table 1, as in SpDust's environment files), with the
radiation field raised by the CMB (Eq. 10): chi' = chi + U_CMB(0) / u_ISRF (1 + z)^4,
U_CMB(0) = 4.2e-13 and u_ISRF = 1.05e-12 erg cm^-3, at z = 0.01.

SpyDust pins numpy<2.4, so this runs in its own environment:

    python3 -m venv venv-spydust
    venv-spydust/bin/pip install "numpy<2.4" scipy matplotlib pandas
    venv-spydust/bin/pip install --no-deps spydust==1.3.2
    venv-spydust/bin/python spdust_emissivity_generator.py
    venv-spydust/bin/python spdust_emissivity_generator.py --check PDR_emiss.out

Writes spdust_emissivity.csv next to this script: rest frequency (GHz) and
j_nu / n_H (Jy sr^-1 cm^2 per H atom) per environment. ``--check`` compares the
PDR at chi = 3000 with SpDust's own IDL output (SpyDust repository,
notebooks/IDL_outputs/PDR_emiss.out) instead; they agree to 0.02 %.
"""

import argparse
import os
import sys
import types

import numpy as np

try:
    import mpi4py  # noqa: F401
except ImportError:  # SpyDust imports mpi4py even for a serial run: one rank

    class _World:
        """Single-rank stand-in for mpi4py's COMM_WORLD."""

        rank, size = 0, 1

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

    _mpi = types.ModuleType("mpi4py.MPI")
    _mpi.COMM_WORLD = _World()
    sys.modules["mpi4py"] = types.ModuleType("mpi4py")
    sys.modules["mpi4py"].MPI = _mpi
    sys.modules["mpi4py.MPI"] = _mpi

from SpyDust.SPDUST_as_is.spdust import SPDUST  # noqa: E402

REDSHIFT = 0.01
CHI_CMB = 4.2e-13 / 1.05e-12 * (1 + REDSHIFT) ** 4
# Draine & Lazarian (1998b) Table 1, as in SpDust's environment files.
ENVIRONMENTS = {
    "CNM": {"nh": 30.0, "T": 100.0, "Chi": 1.0, "xh": 1.2e-3, "xC": 3e-4, "y": 0.0},
    "WNM": {"nh": 0.4, "T": 6000.0, "Chi": 1.0, "xh": 0.1, "xC": 3e-4, "y": 0.0},
    "PDR": {"nh": 1e5, "T": 300.0, "Chi": 3000.0, "xh": 1e-4, "xC": 2e-4, "y": 0.5},
}
GRAINS = {"gamma": 0.0, "dipole": 9.3, "line": 7}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spdust_emissivity.csv")


def emissivity(name, chi_extra, min_freq, max_freq, n_freq):
    """SpDust emissivity per H atom of one environment.

    Parameters
    ----------
    name : str
        Key of ``ENVIRONMENTS``: ``"CNM"``, ``"WNM"`` or ``"PDR"``.
    chi_extra : float
        Added to the environment's chi (radiation field in units of u_ISRF).
    min_freq, max_freq : float
        Edges (GHz) of the logarithmic grid; SpDust returns its bin centres.
    n_freq : int
        Number of frequencies.

    Returns
    -------
    nu_ghz, j_per_h : np.ndarray
        Rest frequency (GHz) and j_nu / n_H (Jy sr^-1 cm^2 per H atom).
    """
    env = dict(ENVIRONMENTS[name], **GRAINS)
    env["Chi"] += chi_extra
    nu_ghz, j_per_h = SPDUST(env, min_freq=min_freq, max_freq=max_freq, n_freq=n_freq)
    return nu_ghz, j_per_h


parser = argparse.ArgumentParser()
parser.add_argument("--check", help="SpDust IDL output of the PDR at chi = 3000")
args = parser.parse_args()

if args.check:
    idl = np.loadtxt(args.check, comments=";")
    half_step = np.sqrt(idl[1, 0] / idl[0, 0])  # the IDL file holds bin centres
    nu, j = emissivity(
        "PDR", 0.0, idl[0, 0] / half_step, idl[-1, 0] * half_step, len(idl)
    )
    assert np.allclose(nu, idl[:, 0], rtol=1e-6)
    ratio = np.log(j / idl[:, 1])
    above = idl[:, 1] > 1e-3 * idl[:, 1].max()
    print(
        f"PDR SpyDust / IDL: max |log ratio| {np.abs(ratio[above]).max():.2e} "
        f"where j > 1e-3 of the peak"
    )
else:
    columns = {}
    for name in ENVIRONMENTS:
        nu, columns[name] = emissivity(name, CHI_CMB, 10.0, 1000.0, 300)
    header = (
        f"SpDust (SpyDust 1.3.2, SPDUST_as_is) j_nu/n_H [Jy sr^-1 cm^2 H^-1], "
        f"z = {REDSHIFT}, chi' = chi + {CHI_CMB:.4f}\nnu_rest_GHz," + ",".join(columns)
    )
    np.savetxt(
        OUT,
        np.column_stack([nu, *columns.values()]),
        delimiter=",",
        header=header,
        fmt="%.6e",
    )
    print("wrote", OUT)
