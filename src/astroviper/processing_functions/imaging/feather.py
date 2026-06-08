"""Single-dish / interferometer image combination (the *feather* algorithm).

This is the science (processing-function) layer: a pure, stateless function that
operates on in-memory ``xarray`` / NumPy objects and performs **no** I/O and
**no** Dask/graph work (see ``AGENTS.md``).  Disk reads and the parallel chunk
writes are handled by the node-task layer
(:func:`astroviper.node_tasks.imaging.feather`) and the graph driver
(:func:`astroviper.distributed_graphs.imaging.feather`), mirroring the
``image_cube_single_field`` layering.
"""

import numpy as np
import xarray as xr
from astropy import units as u
from xradio.image import make_empty_aperture_image
import toolviper.utils.logger as logger

from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
    fft_lm_to_uv,
    ifft_uv_to_lm,
)

_sky = "SKY"
_beam = "BEAM_FIT_PARAMS_SKY"


def _compute_u_v(xds):
    """Build the (u, v) spatial-frequency grids matching an image's l/m grid.

    Parameters
    ----------
    xds : xarray.Dataset
        Image dataset providing the ``l`` and ``m`` coordinates.

    Returns
    -------
    tuple of numpy.ndarray
        ``(u, v)`` arrays, each of shape ``(n_l, n_m)``.
    """
    shape = [xds.sizes["l"], xds.sizes["m"]]
    sky_image_cell_size = np.abs(2 * [xds["l"][1] - xds["l"][0]])
    w_xds = make_empty_aperture_image(
        phase_center=[0, 0],
        image_size=shape,
        sky_image_cell_size=sky_image_cell_size,
        frequency_coords=[1],
        pol_coords=["I"],
        time_coords=[0],
    )
    u_grid = np.zeros(shape)
    v_grid = np.zeros(shape)
    for i, ux in enumerate(w_xds.coords["u"]):
        u_grid[i, :] = ux
    for i, vx in enumerate(w_xds.coords["v"]):
        v_grid[:, i] = vx
    return (u_grid, v_grid)


def _compute_w_multiple_beams(sd_xds, uv):
    """Compute the single-dish weighting function ``w`` in the uv plane.

    ``w`` is the (per-channel) Fourier transform of the single-dish beam,
    evaluated on the (u, v) grid.  It downweights the interferometer data and
    upweights the single-dish data on the spatial-frequency scales the single
    dish measures well.

    Parameters
    ----------
    sd_xds : xarray.Dataset
        Single-dish image dataset; its ``BEAM_FIT_PARAMS_SKY`` variable provides
        the per-plane major/minor/pa beam parameters.
    uv : tuple of numpy.ndarray
        ``(u, v)`` grids from :func:`_compute_u_v`.

    Returns
    -------
    numpy.ndarray
        ``w`` with the same shape as the single-dish ``SKY`` cube
        ``(time, frequency, polarization, l, m)``.
    """
    beams = sd_xds[_beam]
    bunit = beams.attrs["units"]

    # add the l and m axes so the per-plane beam params broadcast against (u, v)
    bmaj = np.expand_dims(beams.sel(beam_params_label="major"), (-1, -2))
    bmin = np.expand_dims(beams.sel(beam_params_label="minor"), (-1, -2))
    bpa = np.expand_dims(beams.sel(beam_params_label="pa"), (-1, -2))

    alpha = (bmaj * u.Unit(bunit)).to(u.rad).value
    beta = (bmin * u.Unit(bunit)).to(u.rad).value
    phi = (bpa * u.Unit(bunit)).to(u.rad).value

    alpha2 = alpha * alpha
    beta2 = beta * beta

    # uu, vv (not u, v) because u is already bound to astropy.units above.
    uu, vv = uv
    uu = uu[np.newaxis, np.newaxis, np.newaxis, :, :]
    vv = vv[np.newaxis, np.newaxis, np.newaxis, :, :]
    aterm2 = (uu * np.sin(phi) - vv * np.cos(phi)) ** 2
    bterm2 = (uu * np.cos(phi) + vv * np.sin(phi)) ** 2
    return np.exp(-np.pi * np.pi / 4.0 / np.log(2) * (alpha2 * aterm2 + beta2 * bterm2))


def feather_core(
    sd_xds: xr.Dataset,
    int_xds: xr.Dataset,
    sdfactor: float = 1.0,
    axes: tuple = ("l", "m"),
    fft_backend: str = "scipy",
    num_threads: int = 1,
) -> xr.Dataset:
    """Combine a single-dish and an interferometer image with the feather algorithm.

    Both input images are Fourier transformed to the uv plane, combined with the
    standard feather weighting (the single dish supplies the short spacings the
    interferometer is missing), and the result is transformed back to the image
    plane.  This is a pure science function: the inputs are already-loaded,
    NumPy-backed ``xarray`` datasets and the output is a new ``xarray`` dataset;
    it performs no disk I/O.

    Parameters
    ----------
    sd_xds : xarray.Dataset
        Single-dish (low-resolution) image with a ``SKY`` data variable and a
        ``BEAM_FIT_PARAMS_SKY`` beam variable.
    int_xds : xarray.Dataset
        Interferometer (high-resolution) image with the same variables and the
        same shape as ``sd_xds``.  The output adopts the interferometer beam and
        coordinates.
    sdfactor : float, default 1.0
        Single-dish scaling factor ``s`` applied in the feather combination.
    axes : tuple of str, default ("l", "m")
        Names of the two spatial axes the 2-D FFT is applied over.
    fft_backend : {"scipy", "pyfftw"}, default "scipy"
        FFT backend forwarded to :func:`fft_lm_to_uv` / :func:`ifft_uv_to_lm`.
        ``scipy`` is always available and faster on typical feather cube sizes;
        ``"pyfftw"`` selects the FFTW-backed backend.
    num_threads : int, default 1
        Number of FFT worker threads.

    Returns
    -------
    xarray.Dataset
        A dataset carrying the feathered ``SKY`` cube with dims
        ``(time, frequency, polarization, l, m)`` and the interferometer
        coordinates.

    Raises
    ------
    RuntimeError
        If either input is missing its ``BEAM_FIT_PARAMS_SKY`` variable.
    """
    if _beam not in int_xds.data_vars:
        raise RuntimeError("Unable to find BEAM data variable in interferometer image.")
    if _beam not in sd_xds.data_vars:
        raise RuntimeError("Unable to find BEAM data variable in single dish image.")

    fft_plane = (
        sd_xds[_sky].dims.index(axes[0]),
        sd_xds[_sky].dims.index(axes[1]),
    )

    # Transform both images to the uv (aperture) plane. fft_lm_to_uv keeps the
    # full-precision complex aperture; the FFT is done in complex128 regardless
    # of the input dtype.
    int_ap = fft_lm_to_uv(
        int_xds[_sky].values,
        fft_plane,
        num_threads=num_threads,
        fft_backend=fft_backend,
    )
    sd_ap = fft_lm_to_uv(
        sd_xds[_sky].values, fft_plane, num_threads=num_threads, fft_backend=fft_backend
    )

    # Preserve the lower-precision dtype of the two inputs for the output image.
    if sd_xds[_sky].dtype.itemsize <= int_xds[_sky].dtype.itemsize:
        out_dtype = sd_xds[_sky].dtype
    else:
        out_dtype = int_xds[_sky].dtype

    w = _compute_w_multiple_beams(sd_xds, _compute_u_v(sd_xds))
    one_minus_w = 1 - w
    s = sdfactor

    int_ba = int_xds[_beam].sel(beam_params_label="major") * int_xds[_beam].sel(
        beam_params_label="minor"
    )
    sd_ba = sd_xds[_beam].sel(beam_params_label="major") * sd_xds[_beam].sel(
        beam_params_label="minor"
    )
    # Use .values: the obs times of the two images generally differ, so keeping
    # the xarray coords would align to an empty time intersection.
    beam_ratio = int_ba.values / sd_ba.values
    beam_ratio = np.expand_dims(beam_ratio, (-1, -2))

    term = (one_minus_w * int_ap + s * beam_ratio * sd_ap) / (one_minus_w + s * w)
    # The inverse transform is complex; the sky image is its real part (the
    # imaginary part is negligible for real emission).
    feather_npary = np.real(
        ifft_uv_to_lm(term, fft_plane, num_threads=num_threads, fft_backend=fft_backend)
    ).astype(out_dtype)

    featherd_img_xds = xr.Dataset(coords=int_xds.coords)
    featherd_img_xds[_sky] = xr.DataArray(
        feather_npary, dims=["time", "frequency", "polarization", "l", "m"]
    )
    return featherd_img_xds
