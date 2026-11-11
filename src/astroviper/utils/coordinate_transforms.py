"""Celestial-sphere / image-plane coordinate transforms used by the simulator.

All angles are in radians and all functions are vectorised NumPy (ported from the
numba kernels in SIRIUS ``_coord_transforms.py``).  The coordinate convention is
the one of the Measurement Set v2/v3 definition (casacore note 264, p. 12): the
``Z`` axis points to the north celestial pole, ``X`` to the east (``ra = pi/2,
dec = 0``) and ``-Y`` to the vernal equinox; ``uvw`` is aligned with ``XYZ`` when
the source is at ``ra = 0, dec = pi/2``.
"""

from __future__ import annotations

import numpy as np


def rotation_matrix_x(theta: float) -> np.ndarray:
    """Right-handed rotation matrix about the ``x`` axis by ``theta`` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_matrix_y(psi: float) -> np.ndarray:
    """Right-handed rotation matrix about the ``y`` axis by ``psi`` radians."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_matrix_z(phi: float) -> np.ndarray:
    """Right-handed rotation matrix about the ``z`` axis by ``phi`` radians."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def directional_cosines(ra_dec: np.ndarray) -> np.ndarray:
    """Direction cosines ``(l, m, n)`` of sky positions.

    Parameters
    ----------
    ra_dec : np.ndarray, [..., 2], radians
        Right ascension and declination.

    Returns
    -------
    np.ndarray, [..., 3]
        ``l = sin(ra) cos(dec)``, ``m = -cos(ra) cos(dec)``, ``n = sin(dec)``.
    """
    ra_dec = np.asarray(ra_dec, dtype=np.float64)
    ra, dec = ra_dec[..., 0], ra_dec[..., 1]
    cos_dec = np.cos(dec)
    return np.stack([np.sin(ra) * cos_dec, -np.cos(ra) * cos_dec, np.sin(dec)], axis=-1)


def sin_project(ra_dec_center: np.ndarray, ra_dec: np.ndarray) -> np.ndarray:
    """Orthographic (SIN) projection of sky positions onto the tangent plane.

    See equation 10 of AIPS memo 27 (http://tdc-www.harvard.edu/wcstools/aips27.pdf).

    Parameters
    ----------
    ra_dec_center : np.ndarray, [2], radians
        Right ascension and declination of the projection (image) centre.
    ra_dec : np.ndarray, [..., 2], radians
        Positions on the celestial sphere.

    Returns
    -------
    np.ndarray, [..., 2]
        Direction cosines ``(l, m)`` on the image plane.
    """
    ra_dec = np.asarray(ra_dec, dtype=np.float64)
    ra_o, dec_o = float(ra_dec_center[0]), float(ra_dec_center[1])
    ra, dec = ra_dec[..., 0], ra_dec[..., 1]
    d_ra = ra - ra_o
    l = np.cos(dec) * np.sin(d_ra)  # noqa: E741
    m = np.sin(dec) * np.cos(dec_o) - np.cos(dec) * np.sin(dec_o) * np.cos(d_ra)
    return np.stack([l, m], axis=-1)


def inverse_sin_project(ra_dec_center: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Inverse orthographic (SIN) projection from the tangent plane to the sphere.

    See section 3.2.2 of AIPS memo 27.

    Parameters
    ----------
    ra_dec_center : np.ndarray, [2], radians
        Right ascension and declination of the projection (image) centre.
    lm : np.ndarray, [..., 2]
        Direction cosines ``(l, m)`` on the image plane.

    Returns
    -------
    np.ndarray, [..., 2], radians
        Right ascension and declination.
    """
    lm = np.asarray(lm, dtype=np.float64)
    ra_o, dec_o = float(ra_dec_center[0]), float(ra_dec_center[1])
    l, m = lm[..., 0], lm[..., 1]  # noqa: E741
    n = np.sqrt(1.0 - l**2 - m**2)
    ra = ra_o + np.arctan2(l, n * np.cos(dec_o) - m * np.sin(dec_o))
    dec = np.arcsin(m * np.cos(dec_o) + n * np.sin(dec_o))
    return np.stack([ra, dec], axis=-1)


def tan_project(ra_dec_center: np.ndarray, ra_dec: np.ndarray) -> np.ndarray:
    """Gnomonic (TAN) projection of sky positions onto the tangent plane.

    See equation 9 of AIPS memo 27.

    Parameters
    ----------
    ra_dec_center : np.ndarray, [2], radians
        Right ascension and declination of the projection centre.
    ra_dec : np.ndarray, [..., 2], radians
        Positions on the celestial sphere.

    Returns
    -------
    np.ndarray, [..., 2]
        Tangent-plane coordinates ``(l, m)``.
    """
    ra_dec = np.asarray(ra_dec, dtype=np.float64)
    ra_o, dec_o = float(ra_dec_center[0]), float(ra_dec_center[1])
    ra, dec = ra_dec[..., 0], ra_dec[..., 1]
    d_ra = ra - ra_o
    div = np.sin(dec) * np.sin(dec_o) + np.cos(dec) * np.cos(dec_o) * np.cos(d_ra)
    l = np.cos(dec) * np.sin(d_ra) / div  # noqa: E741
    m = (np.sin(dec) * np.cos(dec_o) - np.cos(dec) * np.sin(dec_o) * np.cos(d_ra)) / div
    return np.stack([l, m], axis=-1)


def sin_pixel_to_celestial_coord(
    ra_dec_center: np.ndarray,
    image_size: np.ndarray,
    cell_size: np.ndarray,
    pixel: np.ndarray,
) -> np.ndarray:
    """Convert (fractional) pixel coordinates of a SIN-projected image to sky positions.

    The reference pixel is ``image_size // 2``.

    Parameters
    ----------
    ra_dec_center : np.ndarray, [2], radians
        Right ascension and declination of the image centre.
    image_size : np.ndarray, [2]
        Number of pixels along ``l`` and ``m``.
    cell_size : np.ndarray, [2], radians
        Pixel size along ``l`` and ``m`` (``cell_size[0]`` is normally negative
        because right ascension increases to the left).
    pixel : np.ndarray, [..., 2]
        Pixel coordinates.

    Returns
    -------
    np.ndarray, [..., 2], radians
        Right ascension and declination.
    """
    image_center = np.asarray(image_size) // 2
    lm = (np.asarray(pixel, dtype=np.float64) - image_center) * np.asarray(cell_size)
    return inverse_sin_project(ra_dec_center, lm)


def celestial_coord_to_sin_pixel(
    ra_dec_center: np.ndarray,
    image_size: np.ndarray,
    cell_size: np.ndarray,
    ra_dec: np.ndarray,
) -> np.ndarray:
    """Convert sky positions to (fractional) pixel coordinates of a SIN-projected image.

    Inverse of :func:`sin_pixel_to_celestial_coord`.
    """
    image_center = np.asarray(image_size) // 2
    lm = sin_project(ra_dec_center, ra_dec)
    return lm / np.asarray(cell_size) + image_center


def calculate_uvw_rotation(
    ra_dec_center: np.ndarray, ra_dec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation matrix and rotated direction vector for a source offset from the phase centre.

    Used to evaluate the interferometer phase of a point source at ``ra_dec``
    for baselines whose ``uvw`` are defined relative to ``ra_dec_center``::

        phase = 2 pi * lmn_rot . (uvw @ uvw_rotation_matrix) * frequency / c

    Parameters
    ----------
    ra_dec_center : np.ndarray, [2], radians
        Phase centre.
    ra_dec : np.ndarray, [2], radians
        Source position.

    Returns
    -------
    uvw_rotation_matrix : np.ndarray, [3, 3]
    lmn_rot : np.ndarray, [3]
        ``(l, m, 1 - n)`` of the source in the rotated frame (``lmn_rot[2]`` is
        ``1 - n``, so the ``1/n`` factor is ``1 / (1 - lmn_rot[2])``).
    """
    ra_dec_center = np.asarray(ra_dec_center, dtype=np.float64)
    ra_dec = np.asarray(ra_dec, dtype=np.float64)
    uvw_rotation_matrix = (
        rotation_matrix_x(-(np.pi / 2 - ra_dec_center[1]))
        @ rotation_matrix_z(ra_dec[0] - ra_dec_center[0])
        @ rotation_matrix_x(np.pi / 2 - ra_dec[1])
    )
    out_rotation_matrix = rotation_matrix_x(
        -(np.pi / 2 - ra_dec[1])
    ) @ rotation_matrix_z(-ra_dec[0])
    lmn_out = directional_cosines(ra_dec)
    lmn_in = directional_cosines(ra_dec_center)
    lmn_rot = out_rotation_matrix @ (lmn_out - lmn_in)
    return uvw_rotation_matrix, lmn_rot


def rotate_coordinates(
    x: np.ndarray, y: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate plane coordinates clockwise by ``angle`` radians.

    ``x' = cos(a) x + sin(a) y``, ``y' = -sin(a) x + cos(a) y``.
    """
    c, s = np.cos(angle), np.sin(angle)
    return c * x + s * y, -s * x + c * y


def make_rotated_grid(
    image_size: np.ndarray, cell_size: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """Centred 2-D coordinate grids rotated clockwise by ``angle`` radians.

    Parameters
    ----------
    image_size : np.ndarray, [2]
        Grid size along ``x`` and ``y``.
    cell_size : np.ndarray, [2]
        Cell size along ``x`` and ``y``.
    angle : float, radians
        Rotation angle (``0`` returns the unrotated grid).

    Returns
    -------
    x_grid, y_grid : np.ndarray, [image_size[0], image_size[1]]
        ``meshgrid(..., indexing="ij")`` grids, i.e. axis 0 is ``x`` and axis 1 is ``y``.
    """
    image_size = np.asarray(image_size)
    image_center = image_size // 2
    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell_size[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell_size[1]
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    if angle != 0:
        x_grid, y_grid = rotate_coordinates(x_grid, y_grid, angle)
    return x_grid, y_grid


def wrapped_angle_difference(angle_1: np.ndarray, angle_2: np.ndarray) -> np.ndarray:
    """Absolute difference between angles wrapped into ``[0, pi]``."""
    return np.abs(
        ((np.asarray(angle_1) - np.asarray(angle_2) + np.pi) % (2 * np.pi)) - np.pi
    )
