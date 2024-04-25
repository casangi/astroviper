import numpy as np
from typing import List

def _ifft_uv_to_lm(ary: np.ndarray, axes: List[int]) -> np.ndarray:
    """
    Do a 2-D iFFT. A real valued array will be returned
    :ary : np.ndarray
        Numpy array to iFT
    :axes : List[int]
        List of two axes to transform
    Returns
    -------
    np.ndarray
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(ary, axes=axes),
            axes=axes),
        axes=axes
    ).real
