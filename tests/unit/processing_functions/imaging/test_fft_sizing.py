"""Unit tests for :func:`next_fft_friendly_size`."""

import pytest

from astroviper.processing_functions.imaging.utils.fft_sizing import (
    next_fft_friendly_size,
)


def _is_five_smooth(n):
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def _brute_next_five_smooth(target, even=False):
    n = target
    while True:
        if _is_five_smooth(n) and (not even or n % 2 == 0):
            return n
        n += 1


class TestNextFftFriendlySize:
    def test_known_values(self):
        assert next_fft_friendly_size(9830) == 10000  # 2*5*983 -> 2**4 * 5**4
        assert next_fft_friendly_size(13500) == 13500  # 2**2 * 3**3 * 5**3
        assert next_fft_friendly_size(9600) == 9600  # 2**7 * 3 * 5**2
        assert next_fft_friendly_size(8192) == 8192  # 2**13
        assert next_fft_friendly_size(1) == 1
        assert next_fft_friendly_size(7) == 8

    def test_result_is_five_smooth_and_ge_target(self):
        for t in [1, 2, 7, 100, 251, 999, 1000, 1024, 4096, 9830, 11251, 13501]:
            r = next_fft_friendly_size(t)
            assert r >= t
            assert _is_five_smooth(r), f"{r} (for {t}) is not 5-smooth"

    def test_minimality_matches_brute_force(self):
        for t in range(1, 2000):
            assert next_fft_friendly_size(t) == _brute_next_five_smooth(t), t

    def test_even_option(self):
        for t in [1, 7, 9, 25, 81, 1001, 9831, 12345]:
            r = next_fft_friendly_size(t, even=True)
            assert r >= t
            assert r % 2 == 0
            assert _is_five_smooth(r)
            assert r == _brute_next_five_smooth(t, even=True), t

    def test_already_friendly_is_unchanged(self):
        for n in (2, 3, 5, 6, 10, 16, 1000, 1024, 12000, 13500):
            assert next_fft_friendly_size(n) == n
