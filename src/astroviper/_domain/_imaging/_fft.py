import numpy as np
from typing import List

def _fft_lm_to_uv(ary: np.ndarray, axes: List[int]) -> np.ndarray:
    """
    Do a 2-D FFT. This returns a complex-valued array
    :ary : np.ndarray
        Numpy array to FT
    :axes : List[int]
        List of two axes to transform
    Returns
    -------
    np.ndarray
    """
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(ary, axes=axes), axes=axes
        ), axes=axes
    )
