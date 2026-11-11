"""Choose FFT-friendly (5-smooth) grid sizes.

The cost of an FFT depends strongly on the prime factorization of its length.
Sizes whose only prime factors are 2, 3 and 5 -- "5-smooth" or *regular*
numbers -- use the fast mixed-radix path in pocketfft (scipy / numpy) and FFTW,
whereas a large prime factor forces the slow Bluestein / Rader fallback.

Rounding a gridded image's padded size up to the next 5-smooth number therefore
makes the grid<->image FFTs faster for "unlucky" image sizes.  For example an
8192-pixel image padded by ``fft_padding=1.2`` gives ``9830 = 2 * 5 * 983`` with
983 prime; :func:`next_fft_friendly_size` rounds that up to ``10000 = 2**4 * 5**4``.
"""


def next_fft_friendly_size(target, even=False):
    """Return the smallest 5-smooth integer ``>= target``.

    A 5-smooth (regular) number factors as ``2**a * 3**b * 5**c`` -- it has no
    prime factor larger than 5.  These are the lengths for which pocketfft and
    FFTW use their fast mixed-radix transforms.

    Parameters
    ----------
    target : int
        Lower bound, e.g. ``ceil(image_size * fft_padding)``.
    even : bool, optional
        If ``True`` return the smallest *even* 5-smooth integer ``>= target``.
        Imaging grids are normally even so the phase centre falls on a pixel.
        Default ``False``.

    Returns
    -------
    int
        The smallest 5-smooth integer ``>= target`` (and even when requested).

    Examples
    --------
    >>> next_fft_friendly_size(9830)        # 2 * 5 * 983  ->  next regular number
    10000
    >>> next_fft_friendly_size(13500)       # already 2**2 * 3**3 * 5**3
    13500
    >>> next_fft_friendly_size(1001, even=True)
    1008
    """
    target = int(target)
    if target <= 0:
        return 1

    # Every even 5-smooth number is 2 * (a 5-smooth number), so the smallest
    # even one >= target is 2 * (smallest 5-smooth >= ceil(target / 2)).
    if even:
        return 2 * next_fft_friendly_size(-(-target // 2), even=False)

    if target <= 6:
        return target
    # Already a power of two (a 5-smooth number)?
    if not (target & (target - 1)):
        return target

    # Smallest 2**a * 3**b * 5**c >= target. For each (3**b * 5**c) factor take
    # the smallest power of two that reaches target; keep the running minimum.
    # (The classic "next regular number" search, as in scipy's _next_regular.)
    match = None
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            quotient = -(-target // p35)  # ceil(target / p35)
            p2 = 1 << (quotient - 1).bit_length()  # next power of two >= quotient
            n = p2 * p35
            if n == target:
                return n
            if match is None or n < match:
                match = n
            p35 *= 3
            if p35 == target:
                return p35
        if match is None or p35 < match:  # p35 now >= target (a pure 3*5 number)
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if match is None or p5 < match:  # p5 now >= target (a pure power of 5)
        match = p5
    return match


def padded_grid_size(image_lm, fft_padding):
    """Padded uv-grid size for an ``(l, m)`` image, rounded to FFT-friendly sizes.

    Returns ``[n_u, n_v]`` where each axis is the smallest *even* 5-smooth size
    ``>= ceil(fft_padding * image_size)``, so the grid<->image FFTs in the
    gridder use a fast mixed-radix length instead of an "unlucky" size with a
    large prime factor.  The chosen size is logged at debug level.

    Parameters
    ----------
    image_lm : sequence of int
        ``[n_l, n_m]`` unpadded image dimensions.
    fft_padding : float
        Gridding/FFT padding factor (e.g. ``1.2``).

    Returns
    -------
    numpy.ndarray
        ``[n_u, n_v]`` int padded grid dimensions.
    """
    import numpy as np
    import toolviper.utils.logger as logger

    n_l, n_m = int(image_lm[0]), int(image_lm[1])
    raw_u = fft_padding * n_l
    raw_v = fft_padding * n_m
    n_u = next_fft_friendly_size(int(np.ceil(raw_u)), even=True)
    n_v = next_fft_friendly_size(int(np.ceil(raw_v)), even=True)
    logger.debug(
        f"Padded uv-grid size: image [{n_l}, {n_m}] * fft_padding {fft_padding} "
        f"= [{raw_u:.1f}, {raw_v:.1f}] -> FFT-friendly [{n_u}, {n_v}]"
    )
    return np.array([n_u, n_v], dtype=int)
