#pragma once

// Pruned, fused and threaded 2-D real transforms of the fast Asp kernel,
// bit-identical to aspfft::rfft2_half / irfft2_half (asp_fft.hpp of the
// reference).
//
// The reference transforms are pocketfft's multi-axis calls: forward, r2c
// along x (the rows) then c2c along y (the columns); inverse, c2c along y then
// c2r along x with the factor 1 / (nx ny). pocketfft computes every 1-D line
// on its own, and its SIMD path (VL lines in the lanes of a GCC vector) does
// per lane exactly the IEEE operations of its scalar path (no contraction:
// -ffp-contract=off). So each line here is the same plan of the same length,
// run on the same values with the same factor, in VL lanes; only which lines
// are computed, in which order and by which thread differs:
//  * a row known to be zero is not transformed: its half spectrum (signed
//    zeros only) is that of one zero row run through the same plan;
//  * rows of an inverse transform the caller never reads are not computed;
//  * column block by column block, the forward column transforms are followed
//    by the caller's product and the inverse column transforms, so a spectrum
//    is read once and never written back;
//  * the rows of an inverse transform go to the caller as they are made
//    (fused residual updates and peak searches): no image is stored.
// Elementwise products keep the operand order of the reference expressions.
//
// The lanes are those of the baseline instruction set (VL = 2 doubles with
// SSE2): the module has no AVX2 build, whose lanes give some NaNs another
// sign bit than the reference's.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "asp_fft.hpp"

// Transparent huge pages for the large buffers are off: madvise(MADV_HUGEPAGE)
// is compiled in only with -DASTROVIPER_ASP_FAST_HUGEPAGES=1 on a platform
// that has it (the runs of this work keep NumPy's NUMPY_MADVISE_HUGEPAGE=0).
#if defined(ASTROVIPER_ASP_FAST_HUGEPAGES) && ASTROVIPER_ASP_FAST_HUGEPAGES
#include <sys/mman.h>
#endif

#ifdef POCKETFFT_NO_VECTORS
#error "the fast Asp transforms need pocketfft's SIMD lanes (GCC or Clang 5 or later)"
#endif

namespace aspfft2 {

namespace pd = pocketfft::detail;
using cd = std::complex<double>;
using cx = pd::cmplx<double>;
constexpr std::size_t VL = pd::VLEN<double>::val;  // 2 with SSE2
using vd = pd::vtype_t<double>;
using cv = pd::cmplx<vd>;
static_assert(sizeof(cd) == sizeof(cx) && alignof(cd) == alignof(cx),
              "std::complex<double> and pocketfft's cmplx<double> share a layout");

// ---------------------------------------------------------------------------
// Memory: uninitialised, 64-byte aligned buffers (2 MB aligned and advised
// for huge pages from 4 MB on, when built with ASTROVIPER_ASP_FAST_HUGEPAGES).
// ---------------------------------------------------------------------------
inline void* alloc_bytes(std::size_t bytes) {
    std::size_t align = 64;
#if defined(ASTROVIPER_ASP_FAST_HUGEPAGES) && ASTROVIPER_ASP_FAST_HUGEPAGES && defined(MADV_HUGEPAGE)
    constexpr std::size_t kHuge = std::size_t(2) << 20;
    const bool big = bytes >= 2 * kHuge;
    if (big) align = kHuge;
#endif
    const std::size_t rounded = (std::max<std::size_t>(bytes, 1) + align - 1) / align * align;
    void* p = std::aligned_alloc(align, rounded);
    if (p == nullptr) throw std::bad_alloc();
#if defined(ASTROVIPER_ASP_FAST_HUGEPAGES) && ASTROVIPER_ASP_FAST_HUGEPAGES && defined(MADV_HUGEPAGE)
    if (big) madvise(p, rounded, MADV_HUGEPAGE);
#endif
    return p;
}

template <typename T>
class Buf {  // T trivial; resize() neither keeps nor initialises the contents
public:
    Buf() = default;
    explicit Buf(std::size_t n) { resize(n); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p_(o.p_), n_(o.n_), cap_(o.cap_) { o.p_ = nullptr; o.n_ = o.cap_ = 0; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) {
            std::free(p_);
            p_ = o.p_; n_ = o.n_; cap_ = o.cap_;
            o.p_ = nullptr; o.n_ = o.cap_ = 0;
        }
        return *this;
    }
    ~Buf() { std::free(p_); }
    void resize(std::size_t n) {
        if (n > cap_) {
            std::free(p_);
            p_ = nullptr;
            n_ = cap_ = 0;  // still empty if the allocation throws
            p_ = static_cast<T*>(alloc_bytes(n * sizeof(T)));
            cap_ = n;
        }
        n_ = n;
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    std::size_t size() const { return n_; }
    T& operator[](std::size_t k) { return p_[k]; }
    const T& operator[](std::size_t k) const { return p_[k]; }

private:
    T* p_ = nullptr;
    std::size_t n_ = 0, cap_ = 0;
};

// ---------------------------------------------------------------------------
// Threads: contiguous parts of [0, n) on pocketfft's pool.
// ---------------------------------------------------------------------------
template <typename F>
void parallel_for(std::size_t nthreads, std::size_t n, F&& f) {
    if (n == 0) return;
    nthreads = std::max<std::size_t>(1, std::min(nthreads, n));
    if (nthreads == 1) {
        f(std::size_t(0), n);
        return;
    }
    pd::threading::thread_map(nthreads, [&] {
        const std::size_t t = pd::threading::thread_id(), m = pd::threading::num_threads();
        const std::size_t b = n * t / m, e = n * (t + 1) / m;
        if (b < e) f(b, e);
    });
}

// Threads worth using for `work` units of about `grain` units per thread.
inline std::size_t threads_for(std::size_t nthreads, double work, double grain) {
    const double t = work / grain;
    return std::max<std::size_t>(1, std::min<std::size_t>(nthreads, t < 1.0 ? 1 : static_cast<std::size_t>(t)));
}

// Per-thread scratch kept across calls (pocketfft's pool threads live as long
// as the process), so no transform faults in fresh pages.
template <typename T, int Slot>
inline Buf<T>& tl_buf(std::size_t n) {
    thread_local Buf<T> b;
    b.resize(n);
    return b;
}
template <typename T>
inline std::vector<Buf<T>>& tl_bufs(std::size_t nbuf, std::size_t len) {
    thread_local std::vector<Buf<T>> bufs;
    if (bufs.size() < nbuf) bufs.resize(nbuf);
    for (std::size_t q = 0; q < nbuf; ++q) bufs[q].resize(len);
    return bufs;
}

// ---------------------------------------------------------------------------
// Complex products in VL lanes, with the operand order of aspfft::mul and
// std::complex's componentwise real scaling.
// ---------------------------------------------------------------------------
inline cv vmul(const cv& a, const cv& b) { return cv(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); }
inline cv vscale(const cv& a, double s) { return cv(a.r * s, a.i * s); }
inline cv vadd(const cv& a, const cv& b) { return cv(a.r + b.r, a.i + b.i); }
inline cv vsub(const cv& a, const cv& b) { return cv(a.r - b.r, a.i - b.i); }
inline cv vbroadcast(const cd& z) {
    cv v;
    for (std::size_t l = 0; l < VL; ++l) { v.r[l] = z.real(); v.i[l] = z.imag(); }
    return v;
}
// Lane l = z[k0 + l] for l < nl, 0 beyond.
inline cv vlanes(const cd* z, std::size_t k0, std::size_t nl) {
    cv v;
    for (std::size_t l = 0; l < VL; ++l) {
        const cd w = l < nl ? z[k0 + l] : cd(0.0, 0.0);
        v.r[l] = w.real();
        v.i[l] = w.imag();
    }
    return v;
}

// ---------------------------------------------------------------------------
// Packing of a real line's r2c output (pocketfft general_r2c, forward) and
// unpacking of a half spectrum for c2r (general_c2r, backward).
// ---------------------------------------------------------------------------
template <typename Get>
inline void pack_r2c(std::size_t n, cx* o, Get&& t) {
    o[0].Set(t(0));
    std::size_t i = 1, ii = 1;
    for (; i < n - 1; i += 2, ++ii) o[ii].Set(t(i), t(i + 1));
    if (i < n) o[ii].Set(t(i));
}

// The half spectrum of an all-zero real row of length n as pocketfft's r2c
// makes it (signed zeros), cached per length.
inline const std::vector<cx>& zero_row(std::size_t n) {
    static std::mutex mut;
    static std::list<std::pair<std::size_t, std::vector<cx>>> cache;
    std::lock_guard<std::mutex> lock(mut);
    for (auto& e : cache)
        if (e.first == n) return e.second;
    auto plan = pd::get_plan<pd::pocketfft_r<double>>(n);
    std::vector<double> t(n, 0.0);
    plan->exec(t.data(), 1.0, true);
    std::vector<cx> out(n / 2 + 1);
    pack_r2c(n, out.data(), [&](std::size_t i) { return t[i]; });
    cache.emplace_back(n, std::move(out));
    return cache.back().second;
}

inline bool all_positive_zero(const double* a, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        std::uint64_t bits;
        std::memcpy(&bits, a + k, sizeof(bits));
        if (bits != 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Row passes.
// ---------------------------------------------------------------------------

// Forward r2c of real rows of length n: load(r, dst) writes row r (n values),
// its half spectrum (n / 2 + 1 values) goes to dst_row(r).
template <typename Load, typename Dst>
void r2c_rows(std::size_t n, std::size_t nrows, std::size_t nthreads, Load&& load, Dst&& dst_row) {
    if (nrows == 0) return;
    auto plan = pd::get_plan<pd::pocketfft_r<double>>(n);
    const std::size_t ngroups = (nrows + VL - 1) / VL;
    parallel_for(threads_for(nthreads, double(nrows) * n, 32768.0), ngroups, [&](std::size_t g0, std::size_t g1) {
        Buf<vd>& tv = tl_buf<vd, 0>(n);
        Buf<double>& rows = tl_buf<double, 1>(VL * n);
        for (std::size_t g = g0; g < g1; ++g) {
            const std::size_t r0 = g * VL, nl = std::min(VL, nrows - r0);
            for (std::size_t l = 0; l < VL; ++l) {
                double* row = rows.data() + l * n;
                if (l < nl) load(r0 + l, row);
                else std::fill_n(row, n, 0.0);
            }
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t l = 0; l < VL; ++l) tv[i][l] = rows[l * n + i];
            plan->exec(tv.data(), 1.0, true);
            for (std::size_t l = 0; l < nl; ++l)
                pack_r2c(n, dst_row(r0 + l), [&](std::size_t i) { return tv[i][l]; });
        }
    });
}

// Inverse c2r of half-spectrum rows (n / 2 + 1 values at src_row(r)) of length
// n with factor fct; use(r, row) gets the n real values of row r.
template <typename Src, typename Use>
void c2r_rows(std::size_t n, std::size_t nrows, double fct, std::size_t nthreads, Src&& src_row, Use&& use) {
    if (nrows == 0) return;
    auto plan = pd::get_plan<pd::pocketfft_r<double>>(n);
    const std::size_t ngroups = (nrows + VL - 1) / VL;
    parallel_for(threads_for(nthreads, double(nrows) * n, 32768.0), ngroups, [&](std::size_t g0, std::size_t g1) {
        Buf<vd>& tv = tl_buf<vd, 0>(n);
        Buf<double>& row = tl_buf<double, 1>(n);
        for (std::size_t g = g0; g < g1; ++g) {
            const std::size_t r0 = g * VL, nl = std::min(VL, nrows - r0);
            for (std::size_t l = 0; l < VL; ++l) {
                if (l >= nl) {
                    for (std::size_t i = 0; i < n; ++i) tv[i][l] = 0.0;
                    continue;
                }
                const cx* in = src_row(r0 + l);
                tv[0][l] = in[0].r;
                std::size_t i = 1, ii = 1;
                for (; i < n - 1; i += 2, ++ii) {
                    tv[i][l] = in[ii].r;
                    tv[i + 1][l] = in[ii].i;
                }
                if (i < n) tv[i][l] = in[ii].r;
            }
            plan->exec(tv.data(), fct, false);
            for (std::size_t l = 0; l < nl; ++l) {
                for (std::size_t i = 0; i < n; ++i) row[i] = tv[i][l];
                use(r0 + l, static_cast<const double*>(row.data()));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Column passes: body(k0, nl, bufs) for the column blocks [k0, k0 + nl) of a
// half spectrum hx columns wide, nl <= VL, with `nbuf` per-thread vector
// buffers of `len` values each.
// ---------------------------------------------------------------------------
template <typename Body>
void column_blocks(std::size_t hx, std::size_t len, std::size_t nbuf, std::size_t nthreads, Body&& body) {
    const std::size_t nblocks = (hx + VL - 1) / VL;
    parallel_for(threads_for(nthreads, double(hx) * len * nbuf, 16384.0), nblocks,
                 [&](std::size_t b0, std::size_t b1) {
                     std::vector<Buf<cv>>& bufs = tl_bufs<cv>(nbuf, len);
                     for (std::size_t b = b0; b < b1; ++b) body(b * VL, std::min(VL, hx - b * VL), bufs);
                 });
}

// buf[r] (r in [r0, r1)) = lanes of columns [k0, k0 + nl) of rows r - r0 + s0 of S (row stride ld).
inline void gather_from(const cx* S, std::size_t ld, std::size_t s0, std::size_t k0, std::size_t nl,
                        std::size_t r0, std::size_t r1, cv* buf) {
    for (std::size_t r = r0; r < r1; ++r) {
        const cx* s = S + (r - r0 + s0) * ld + k0;
        cv v;
        for (std::size_t l = 0; l < VL; ++l) {
            v.r[l] = l < nl ? s[l].r : 0.0;
            v.i[l] = l < nl ? s[l].i : 0.0;
        }
        buf[r] = v;
    }
}

// buf[r] (r in [r0, r1)) = lanes of columns [k0, k0 + nl) of rows r of S (row stride ld).
inline void gather(const cx* S, std::size_t ld, std::size_t k0, std::size_t nl, std::size_t r0, std::size_t r1,
                   cv* buf) {
    gather_from(S, ld, r0, k0, nl, r0, r1, buf);
}

// buf[r] (r in [r0, r1)) = the zero-row spectrum at columns [k0, k0 + nl).
inline void gather_zero(const std::vector<cx>& z, std::size_t k0, std::size_t nl, std::size_t r0, std::size_t r1,
                        cv* buf) {
    cv v;
    for (std::size_t l = 0; l < VL; ++l) {
        v.r[l] = l < nl ? z[k0 + l].r : 0.0;
        v.i[l] = l < nl ? z[k0 + l].i : 0.0;
    }
    for (std::size_t r = r0; r < r1; ++r) buf[r] = v;
}

// Rows [r0, r1) of buf to U (row stride ld, row r at U + (r - r0) * ld), columns [k0, k0 + nl).
inline void scatter(const cv* buf, std::size_t r0, std::size_t r1, cx* U, std::size_t ld, std::size_t k0,
                    std::size_t nl) {
    for (std::size_t r = r0; r < r1; ++r) {
        cx* u = U + (r - r0) * ld + k0;
        for (std::size_t l = 0; l < nl; ++l) u[l].Set(buf[r].r[l], buf[r].i[l]);
    }
}

}  // namespace aspfft2
