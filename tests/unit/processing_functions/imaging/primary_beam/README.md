# CASA primary-beam reference values

The ALMA and VLA tables in this directory contain samples from CASA 6.7.7.6
`tclean` primary-beam products; they were not generated from AstroVIPER's beam
formula. To regenerate them, run `tclean` with `gridder="standard"`,
`specmode="mfs"`, `nterms=1`, and the reference frequency used by the matching
test geometry, then sample the resulting `.pb` image at the listed pixel or
world-coordinate positions before applying a primary-beam cutoff.

The ALMA samples came from the six continuum-demonstrator datasets
2017.1.00661.S, 2017.1.00999.S, 2018.1.01079.S, 2018.1.01284.S,
2019.1.00678.S, and 2023.1.01342.S, using matched reference frequencies. The
VLA MFS values came from the single-field, 1.5 GHz off-axis fixture imaged with
200 by 200 pixels of 10 arcsec; the MVC values sample its 20-channel `.pb`
cube. Preserve values below CASA's 0.1 PB cutoff in this table so the test
validates the numerical response rather than merely its clipping behavior.
