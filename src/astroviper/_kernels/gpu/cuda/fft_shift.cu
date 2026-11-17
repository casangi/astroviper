// The shifts and the normalisation of the uv <-> lm transforms on the GPU, for astroviper._kernels.gpu.fft: CuPy raw
// kernels, shipped as package data and compiled by NVRTC at the first call. cuFFT transforms the planes in between.
//
// The reference (fft_normalize_anti_alias_gridder.py: ifft_uv_to_lm, fft_lm_to_uv) computes, per plane of n0 x n1,
// fftshift(F(ifftshift(x))) for the transform F (the inverse one divided by n0 n1) when an axis is odd. When both axes
// are even it computes cb * F(cb * x) with cb = (-1)^(i + j) instead, its fold of each shift into a sign flip of the
// other domain, which is fftshift(F(ifftshift(x))) only up to the sign (-1)^(n0/2 + n1/2) (each roll by n/2 of an
// axis of n is the flip of the other domain times (-1)^(n/2)). The GPU keeps the reference's results, so its signs:
// checkerboard() makes the fold here, and roll() the two shifts when an axis is odd. Both act on a stack of planes,
// given as `rows` rows of n1 complex values (planes of n0 rows, one after another), and take the 1/(n0 n1) of an
// inverse transform as `scale` (1 elsewhere). The sign flips and the shifts are exact; the scaling rounds to nearest,
// once for a double and twice for a float (to double, then to float: see scaled()); none flushes a subnormal to zero:
// CuPy compiles with -ftz=true, so the sign flips move bits and the float conversions are PTX without .ftz (as in
// hogbom_exact.cu).

typedef long long i64;
typedef unsigned int u32;
typedef unsigned long long u64;

// The complex64 and complex128 values of NumPy and cuFFT: (real, imaginary) pairs, loaded and stored as one vector.
struct __align__(8) c64 {
    float re, im;
};
struct __align__(16) c128 {
    double re, im;
};

__device__ __forceinline__ u32 bits_of(float a) { u32 u; asm("mov.b32 %0, %1;" : "=r"(u) : "f"(a)); return u; }
__device__ __forceinline__ float of_bits(u32 u) { float a; asm("mov.b32 %0, %1;" : "=f"(a) : "r"(u)); return a; }
__device__ __forceinline__ u64 bits_of(double a) { u64 u; asm("mov.b64 %0, %1;" : "=l"(u) : "d"(a)); return u; }
__device__ __forceinline__ double of_bits(u64 u) { double a; asm("mov.b64 %0, %1;" : "=d"(a) : "l"(u)); return a; }
// -a, for every a: the sign bit flipped (the reference's a * -1.0, NaN signs aside).
__device__ __forceinline__ float negated(float a) { return of_bits(bits_of(a) ^ 0x80000000u); }
__device__ __forceinline__ double negated(double a) { return of_bits(bits_of(a) ^ 0x8000000000000000ull); }
// a * s, rounded to nearest. For a float, the product of its exact conversion to double is rounded to double, then
// to float: two roundings, which can differ from one rounding of the exact product by a unit in the last place.
__device__ __forceinline__ double scaled(double a, double s) { return __dmul_rn(a, s); }
__device__ __forceinline__ float scaled(float a, double s) {
    double d;
    asm("cvt.f64.f32 %0, %1;" : "=d"(d) : "f"(a));
    float c;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(c) : "d"(__dmul_rn(d, s)));
    return c;
}

// x[r, j] *= (-1)^(i + j) * scale, in place, with i = r mod n0 the row of the plane: the checkerboard, before a
// transform (scale 1: the odd pixels' signs flipped, the others untouched) or after it.
template <typename C>
__global__ void checkerboard(C* x, const i64 rows, const int n0, const int n1, const double scale) {
    for (i64 r = blockIdx.y; r < rows; r += gridDim.y) {
        const int i = (int)(r % n0);
        C* row = x + r * n1;
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n1; j += gridDim.x * blockDim.x) {
            const bool odd = (i ^ j) & 1;
            if (scale == 1.0) {
                if (odd) {
                    C v = row[j];
                    v.re = negated(v.re);
                    v.im = negated(v.im);
                    row[j] = v;
                }
            } else {
                const double s = odd ? negated(scale) : scale;
                C v = row[j];
                v.re = scaled(v.re, s);
                v.im = scaled(v.im, s);
                row[j] = v;
            }
        }
    }
}

// y[r, j] = x[(i + s0) mod n0, (j + s1) mod n1] * scale within each plane, with 0 <= s0 < n0 and 0 <= s1 < n1: the
// ifftshift before a transform (s = n / 2, rounded down) or the fftshift after it (s = n - n / 2). x and y differ.
template <typename C>
__global__ void roll(const C* __restrict__ x, C* __restrict__ y, const i64 rows, const int n0, const int n1,
                     const int s0, const int s1, const double scale) {
    for (i64 r = blockIdx.y; r < rows; r += gridDim.y) {
        const i64 plane = r / n0;
        const i64 i = r - plane * n0;
        const i64 source_row = plane * n0 + (i + s0 < n0 ? i + s0 : i + s0 - n0);
        const C* source = x + source_row * n1;
        C* row = y + r * n1;
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n1; j += gridDim.x * blockDim.x) {
            const i64 k = (i64)j + s1;
            C v = source[k < n1 ? k : k - n1];
            if (scale != 1.0) {
                v.re = scaled(v.re, scale);
                v.im = scaled(v.im, scale);
            }
            row[j] = v;
        }
    }
}
