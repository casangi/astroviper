#pragma once

// Self-contained complex/real FFT used by the Asp (Adaptive Scale Pixel)
// deconvolver. This replaces casacore's FFTServer with a dependency-free
// implementation that uses only the C++ standard library.
//
// The transforms work for *arbitrary* image sizes: powers of two use an
// iterative radix-2 Cooley-Tukey FFT, and all other lengths use Bluestein's
// chirp-z algorithm (which internally calls the radix-2 routine on a padded
// power-of-two length). All arithmetic is done in double precision regardless
// of the storage type of the caller's images, which keeps the FFT-based
// convolutions accurate enough to match the original (FFTW-backed) results.
//
// Conventions (matching what the Asp algorithm needs):
//   forward 1-D : A[k]   = sum_n a[n] exp(-2*pi*i*k*n/N)        (unnormalized)
//   inverse 1-D : a[n]   = (1/N) sum_k A[k] exp(+2*pi*i*k*n/N)
// The 2-D forward transform is therefore unnormalized and the 2-D inverse is
// normalized by 1/(nx*ny); a forward followed by an inverse is the identity.
// With this pair, ifft2(fft2(a) * fft2(b)) is the *standard* circular
// convolution sum_m a[m] b[(n-m) mod N], independent of any internal
// normalization split -- exactly what the deconvolver assumes.

#include <complex>
#include <cstddef>
#include <vector>

namespace aspfft {

using cd = std::complex<double>;

namespace detail {

inline bool is_pow2(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

inline std::size_t next_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// In-place iterative radix-2 Cooley-Tukey FFT. `invert` performs the inverse
// transform and divides the result by the length. The length of `a` must be a
// power of two.
inline void fft_pow2(std::vector<cd>& a, bool invert) {
    const std::size_t n = a.size();
    if (n <= 1) return;

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    const double PI = 3.14159265358979323846264338327950288;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * PI / static_cast<double>(len) * (invert ? 1.0 : -1.0);
        const cd wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const cd u = a[i + k];
                const cd v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        const double inv = 1.0 / static_cast<double>(n);
        for (auto& x : a) x *= inv;
    }
}

// Bluestein's algorithm: arbitrary-length DFT expressed as a convolution that
// is evaluated with power-of-two FFTs. `invert` selects the sign of the
// exponent; this routine is *unnormalized* (the 1/N for the inverse is applied
// by the caller in fft1d).
inline void bluestein(std::vector<cd>& a, bool invert) {
    const std::size_t n = a.size();
    if (n <= 1) return;

    const double PI = 3.14159265358979323846264338327950288;
    const double sign = invert ? 1.0 : -1.0;

    // Chirp w[k] = exp(sign * i * pi * k^2 / n). Use k^2 mod 2n to keep the
    // argument small and accurate for large n.
    std::vector<cd> w(n);
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t kk = (k * k) % (2 * n);
        const double ang = sign * PI * static_cast<double>(kk) / static_cast<double>(n);
        w[k] = cd(std::cos(ang), std::sin(ang));
    }

    const std::size_t m = next_pow2(2 * n - 1);
    std::vector<cd> A(m, cd(0.0, 0.0));
    std::vector<cd> B(m, cd(0.0, 0.0));

    for (std::size_t k = 0; k < n; ++k) A[k] = a[k] * w[k];
    B[0] = w[0];
    for (std::size_t k = 1; k < n; ++k) {
        B[k] = std::conj(w[k]);
        B[m - k] = std::conj(w[k]);
    }

    fft_pow2(A, false);
    fft_pow2(B, false);
    for (std::size_t i = 0; i < m; ++i) A[i] *= B[i];
    fft_pow2(A, true);  // inverse (normalized by m): yields the convolution

    for (std::size_t k = 0; k < n; ++k) a[k] = A[k] * w[k];
}

}  // namespace detail

// 1-D DFT. `invert == false` is the unnormalized forward transform;
// `invert == true` is the normalized inverse transform (divided by N).
inline void fft1d(std::vector<cd>& a, bool invert) {
    const std::size_t n = a.size();
    if (n <= 1) return;
    if (detail::is_pow2(n)) {
        detail::fft_pow2(a, invert);
        return;
    }
    detail::bluestein(a, invert);
    if (invert) {
        const double inv = 1.0 / static_cast<double>(n);
        for (auto& x : a) x *= inv;
    }
}

// 2-D forward FFT of a real image stored row-major as data[i + j*nx], i in
// [0,nx) (fast/"x" axis), j in [0,ny). The complex result uses the same
// layout and is unnormalized.
inline std::vector<cd> rfft2_forward(const double* data, int nx, int ny) {
    std::vector<cd> spec(static_cast<std::size_t>(nx) * ny);
    for (std::size_t i = 0; i < spec.size(); ++i) spec[i] = cd(data[i], 0.0);

    // Transform each row (length nx, contiguous).
    std::vector<cd> row(nx);
    for (int j = 0; j < ny; ++j) {
        cd* base = spec.data() + static_cast<std::size_t>(j) * nx;
        for (int i = 0; i < nx; ++i) row[i] = base[i];
        fft1d(row, false);
        for (int i = 0; i < nx; ++i) base[i] = row[i];
    }
    // Transform each column (length ny, stride nx).
    std::vector<cd> col(ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) col[j] = spec[static_cast<std::size_t>(j) * nx + i];
        fft1d(col, false);
        for (int j = 0; j < ny; ++j) spec[static_cast<std::size_t>(j) * nx + i] = col[j];
    }
    return spec;
}

// 2-D forward FFT of a complex image (same layout), unnormalized.
inline std::vector<cd> cfft2_forward(std::vector<cd> spec, int nx, int ny) {
    std::vector<cd> row(nx);
    for (int j = 0; j < ny; ++j) {
        cd* base = spec.data() + static_cast<std::size_t>(j) * nx;
        for (int i = 0; i < nx; ++i) row[i] = base[i];
        fft1d(row, false);
        for (int i = 0; i < nx; ++i) base[i] = row[i];
    }
    std::vector<cd> col(ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) col[j] = spec[static_cast<std::size_t>(j) * nx + i];
        fft1d(col, false);
        for (int j = 0; j < ny; ++j) spec[static_cast<std::size_t>(j) * nx + i] = col[j];
    }
    return spec;
}

// 2-D inverse FFT, returning the real part into `out` (normalized by nx*ny).
inline void irfft2_inverse(std::vector<cd> spec, int nx, int ny, double* out) {
    std::vector<cd> row(nx);
    for (int j = 0; j < ny; ++j) {
        cd* base = spec.data() + static_cast<std::size_t>(j) * nx;
        for (int i = 0; i < nx; ++i) row[i] = base[i];
        fft1d(row, true);
        for (int i = 0; i < nx; ++i) base[i] = row[i];
    }
    std::vector<cd> col(ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) col[j] = spec[static_cast<std::size_t>(j) * nx + i];
        fft1d(col, true);
        for (int j = 0; j < ny; ++j) spec[static_cast<std::size_t>(j) * nx + i] = col[j];
    }
    for (std::size_t k = 0; k < spec.size(); ++k) out[k] = spec[k].real();
}

}  // namespace aspfft
