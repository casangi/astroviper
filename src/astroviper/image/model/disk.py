import numpy as np

def generate_disk(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    x0: float,
    y0: float,
    a_d: float,
    b_d: float,
    theta_d_deg: float,
    A_d: float = 1.0,
) -> np.ndarray:
    """
    Create a uniform elliptical disk on a given (x,y) grid.

    Parameters
    ----------
    xgrid, ygrid : np.ndarray
        2D arrays (same shape) of x and y coordinates.
    x0, y0 : float
        Disk center in the same units as xgrid/ygrid.
    a_d, b_d : float
        Semi-major and semi-minor axes of the disk (same units as x/y).
    theta_d_deg : float
        Disk major-axis angle in degrees, counterclockwise from +x.
    A_d : float, optional
        Constant value inside the disk (default 1.0).

    Returns
    -------
    disk : np.ndarray
        2D array with value A_d inside the ellipse and 0 outside.
    """
    theta = np.deg2rad(theta_d_deg)
    Xc = xgrid - x0
    Yc = ygrid - y0

    # Rotate coords into the ellipse principal frame
    Xr =  Xc * np.cos(theta) + Yc * np.sin(theta)
    Yr = -Xc * np.sin(theta) + Yc * np.cos(theta)

    mask = (Xr / a_d) ** 2 + (Yr / b_d) ** 2 <= 1.0
    disk = np.zeros_like(xgrid, dtype=float)
    disk[mask] = A_d
    return disk


