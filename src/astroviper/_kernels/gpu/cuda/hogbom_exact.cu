// Exact Hogbom minor cycle on the GPU, for astroviper._kernels.gpu.hogbom: CuPy raw kernels, shipped as package data
// and compiled by NVRTC at the first call (no CUDA toolkit at build time).
//
// The results are the reference's bit for bit (deconvolvers/hogbom/src/hclean.cpp: clean<T>, which clean_cube runs on
// each plane, and clean_cube_many_threads; their stop tests are deconvolvers/include/plane_stop_test.hpp):
//  * each row of the search box is scanned in column order by one lane, as scan_row does: the sum of the squares of
//    its pixels inside the mask (a double sum in column order) and its first strict maximum of |v|;
//  * the row sums are added in row order by one lane; the peak is the largest row maximum, the lowest row first;
//  * every operation rounds as the reference build does (SSE2, -ffp-contract=off): a rounded multiply, then a rounded
//    add or subtract, never fused (--fmad=false and explicit __d*_rn / PTX .rn), and no float operation flushes a
//    subnormal to zero (CuPy compiles with -ftz=true, so float arithmetic and comparisons are PTX without .ftz, and
//    the orders of |v| are compared as integers: the bits of a number >= 0 sort as its value, a NaN gets key 0 and so
//    never wins, as v > m is false for it on the CPU).
//
// hogbom_exact_chunk runs the iterations [it_begin, it_end) of one plane in one cooperative launch (all its blocks are
// resident). Each iteration has two phases. In phase 1 every block, one strip of 32 rows at a time, subtracts the
// previous component from its footprint and scans the search box; in phase 2 block 0 reduces the rows, runs the stop
// tests and adds the component to the model. A grid barrier ends each phase. Its waits count their spins: a wait past
// spin_limit sets the abort flag, every block then leaves the kernel, and the host raises (the GPU may drive a display
// and have no watchdog). The plane's state (st_i, st_d) and the row partials carry over from one launch to the next.
//
// plane_summary and gather_abs give the binding's summary of a cleaned plane: the extrema of its residual inside the
// mask (maximg) and, in plane order, the |model| values that are not zero, which the host adds up in order.

typedef unsigned char u8;
typedef long long i64;
typedef unsigned long long u64;
#define FULL_MASK 0xffffffffu

enum { STOP_NONE = 0, STOP_MAX_ITER = 1, STOP_THRESHOLD = 2, STOP_DIVERGED = 4 };
// The entry whose conventions the kernel follows: the pixel of a zero peak is the last peak's (clean_cube, (0, 0) at
// first) or the first of the clean box (clean_cube_many_threads).
enum { ENTRY_CLEAN_CUBE = 0, ENTRY_MANY_THREADS = 1 };
// st_i: the pending subtraction, its pixel (row, column), stopped, components, stop reason, and the divergence test's
//       state (started, iterations above the RMS limit); st_d: the pending component's flux, the test's start peak
//       and its lowest sum of squares. bar: the grid barrier's count, generation and abort flag.
enum { S_SUB = 0, S_PY, S_PX, S_STOP, S_COMPONENTS, S_REASON, S_STARTED, S_ABOVE };
enum { D_PV = 0, D_START_PEAK, D_MIN_SS };

// ------------------------------------------------------------------------------------- IEEE operations, never flushed
__device__ __forceinline__ double mul_rn(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double add_rn(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double sub_rn(double a, double b) { return __dsub_rn(a, b); }
__device__ __forceinline__ float mul_rn(float a, float b) {
    float c; asm("mul.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b)); return c; }
__device__ __forceinline__ float add_rn(float a, float b) {
    float c; asm("add.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b)); return c; }
__device__ __forceinline__ float sub_rn(float a, float b) {
    float c; asm("sub.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b)); return c; }
__device__ __forceinline__ double to_f64(double a) { return a; }
__device__ __forceinline__ double to_f64(float a) { double c; asm("cvt.f64.f32 %0, %1;" : "=d"(c) : "f"(a)); return c; }
template <typename T> __device__ __forceinline__ T from_f64(double a);  // exact for values that were T
template <> __device__ __forceinline__ double from_f64<double>(double a) { return a; }
template <> __device__ __forceinline__ float from_f64<float>(double a) {
    float c; asm("cvt.rn.f32.f64 %0, %1;" : "=f"(c) : "d"(a)); return c; }
__device__ __forceinline__ double abs_t(double a) {
    return __longlong_as_double(__double_as_longlong(a) & 0x7fffffffffffffffll); }
__device__ __forceinline__ float abs_t(float a) { return __uint_as_float(__float_as_uint(a) & 0x7fffffffu); }
__device__ __forceinline__ bool finite_t(double a) {
    return (__double_as_longlong(a) & 0x7ff0000000000000ll) != 0x7ff0000000000000ll; }
__device__ __forceinline__ bool finite_t(float a) { return (__float_as_uint(a) & 0x7f800000u) != 0x7f800000u; }
// Not +-0: nonzero, or NaN (|NaN| is NaN, and adding it is not a no-op).
__device__ __forceinline__ bool nonzero_t(double a) { return (__double_as_longlong(a) & 0x7fffffffffffffffll) != 0; }
__device__ __forceinline__ bool nonzero_t(float a) { return (__float_as_uint(a) & 0x7fffffffu) != 0; }
__device__ __forceinline__ bool gt(double a, double b) { return a > b; }
__device__ __forceinline__ bool lt(double a, double b) { return a < b; }
__device__ __forceinline__ bool le(double a, double b) { return a <= b; }
__device__ __forceinline__ bool gt(float a, float b) {
    unsigned int p; asm("{ .reg .pred q; setp.gt.f32 q, %1, %2; selp.u32 %0, 1, 0, q; }" : "=r"(p) : "f"(a), "f"(b));
    return p != 0; }
__device__ __forceinline__ bool lt(float a, float b) {
    unsigned int p; asm("{ .reg .pred q; setp.lt.f32 q, %1, %2; selp.u32 %0, 1, 0, q; }" : "=r"(p) : "f"(a), "f"(b));
    return p != 0; }
__device__ __forceinline__ bool le(float a, float b) {
    unsigned int p; asm("{ .reg .pred q; setp.le.f32 q, %1, %2; selp.u32 %0, 1, 0, q; }" : "=r"(p) : "f"(a), "f"(b));
    return p != 0; }
// The order key of a = |v|: its bits; a NaN gets 0.
__device__ __forceinline__ u64 key_t(double a) {
    const u64 k = (u64)__double_as_longlong(a); return k > 0x7ff0000000000000ull ? 0ull : k; }
__device__ __forceinline__ u64 key_t(float a) {
    const unsigned int k = __float_as_uint(a); return k > 0x7f800000u ? 0ull : (u64)k; }
template <typename T> __device__ __forceinline__ T from_key(u64 k);
template <> __device__ __forceinline__ double from_key<double>(u64 k) { return __longlong_as_double((long long)k); }
template <> __device__ __forceinline__ float from_key<float>(u64 k) { return __uint_as_float((unsigned int)k); }

// ------------------------------------------------------------------------------------------------- grid barrier
// For a kernel whose blocks are all resident (a cooperative launch). False when the kernel is aborting: a block that
// waited more than spin_limit spins set the abort flag, and every block leaves at its next wait.
__device__ __forceinline__ bool grid_sync(unsigned int* bar, unsigned int nblocks, unsigned int spin_limit) {
    __shared__ int s_abort;
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* gen = bar + 1;
        volatile unsigned int* abort_flag = bar + 2;
        const unsigned int g0 = *gen;
        int ab = 0;
        __threadfence();
        if (atomicAdd(bar, 1u) == nblocks - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            unsigned int spins = 0;
            while (*gen == g0) {
                if (*abort_flag) { ab = 1; break; }
                if (++spins > spin_limit) { atomicExch(bar + 2, 1u); ab = 1; break; }
            }
        }
        __threadfence();
        s_abort = ab;
    }
    __syncthreads();
    return s_abort == 0;
}

// ------------------------------------------------------------------------------------------------- one plane
struct Plane {
    int nx, ny, pnx, pny;  // image and PSF sizes
    int i0, i1, j0, j1;    // the search box: the clean box shrunk to the mask's bounding box, inclusive
    int has_mask;
    int xbeg, ybeg;        // the clean box's first pixel (clean_cube_many_threads' zero peak)
};

template <typename T, int TW> struct StripSmem { T v[2][32][TW + 1]; u8 m[2][32][TW + 4]; };

// A strip of 32 rows from row r0: warps 1.. subtract the pending component from the strip's part of its footprint
// and stage the tiles of TW columns in shared memory (double-buffered); warp 0, one lane per row of the search box,
// scans the previous tile in column order as scan_row does.
template <typename T, int TW>
__device__ void strip_exact(StripSmem<T, TW>& sm, T* __restrict__ r, const T* __restrict__ psf,
                            const u8* __restrict__ mask, const Plane& p, int r0, int do_sub, int py, int px, T pv,
                            u64* __restrict__ row_key, int* __restrict__ row_ix, double* __restrict__ row_ss) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nmov = blockDim.x - 32;
    const int rlast = min(r0 + 31, p.ny - 1);
    // the footprint: offsets -pn / 2 .. (pn - 1) / 2 around the component, inside the image
    const int y1 = max(0, py - p.pny / 2), y2 = min(p.ny - 1, py + (p.pny - 1) / 2);
    const int x1 = max(0, px - p.pnx / 2), x2 = min(p.nx - 1, px + (p.pnx - 1) / 2);
    const bool sub_rows = do_sub && rlast >= y1 && r0 <= y2;
    const bool box_rows = rlast >= p.j0 && r0 <= p.j1;
    int lo = 0x7fffffff, hi = -1;
    if (sub_rows) { lo = min(lo, x1); hi = max(hi, x2); }
    if (box_rows) { lo = min(lo, p.i0); hi = max(hi, p.i1); }
    const int my = r0 + lane;
    const bool mine = warp == 0 && box_rows && my >= p.j0 && my <= p.j1;
    u64 km = 0ull;
    int mi = p.i0;
    double ss = 0.0;
    if (hi >= lo) {
        const int ntiles = (hi - lo + TW) / TW;
        for (int k = -1; k < ntiles; ++k) {
            if (warp != 0) {
                if (k + 1 < ntiles) {
                    const int c0 = lo + (k + 1) * TW, buf = (k + 1) & 1;
                    for (int e = tid - 32; e < 32 * TW; e += nmov) {
                        const int row = e / TW, col = e - row * TW, y = r0 + row, x = c0 + col;
                        T v = T(0);
                        u8 mk = 0;
                        if (y <= rlast && x <= hi) {
                            const bool in_box = y >= p.j0 && y <= p.j1 && x >= p.i0 && x <= p.i1;
                            const bool in_fp = sub_rows && y >= y1 && y <= y2 && x >= x1 && x <= x2;
                            if (in_box || in_fp) {
                                const i64 idx = (i64)y * p.nx + x;
                                v = r[idx];
                                if (in_fp) {
                                    const T w = __ldg(psf + (i64)(p.pny / 2 + y - py) * p.pnx + (p.pnx / 2 + x - px));
                                    v = sub_rn(v, mul_rn(pv, w));
                                    r[idx] = v;
                                }
                                mk = (in_box && (!p.has_mask || __ldg(mask + idx))) ? 1 : 0;
                            }
                        }
                        sm.v[buf][row][col] = v;
                        sm.m[buf][row][col] = mk;
                    }
                }
            } else if (k >= 0 && mine) {
                const int c0 = lo + k * TW, buf = k & 1;
#pragma unroll 8
                for (int col = 0; col < TW; ++col) {
                    if (sm.m[buf][lane][col]) {
                        const T v = sm.v[buf][lane][col];
                        const double d = to_f64(v);
                        ss = add_rn(ss, mul_rn(d, d));
                        const u64 kv = key_t(abs_t(v));
                        if (kv > km) { km = kv; mi = c0 + col; }
                    }
                }
            }
            __syncthreads();
        }
    }
    if (mine) { row_key[my] = km; row_ix[my] = mi; row_ss[my] = ss; }
}

// The decision of iteration `it` (thread 0 of block 0): PlaneStopTest::check and the model update of clean<T>.
// `s` is the sum of squares the search found (the residual the previous iteration left), `found` whether a peak
// above zero was found, at (fpy, fpx) with key bk. History entry k = it - it_begin of this launch: the component of
// iteration it, and the sum of squares after the previous one.
template <typename T, int ENTRY>
__device__ void decide(int it, int it_begin, double s, int found, u64 bk, int fpy, int fpx, const T* r, T* model,
                       const Plane& p, int max_iter, T gain, T thres, const T* __restrict__ thres_factor,
                       int use_factor, int max_iter_div, double rms_factor, T peak_factor, int* st_i, double* st_d,
                       int record, double* h_peak, double* h_ss, int* h_l, int* h_m, double* h_flux) {
    const int k = it - it_begin;
    if (record && it > 0) h_ss[k] = s;
    int py, px;
    T absval = T(0);
    if (found) {
        py = fpy; px = fpx; absval = from_key<T>(bk);
    } else if (ENTRY == ENTRY_MANY_THREADS) {
        py = p.ybeg; px = p.xbeg;
    } else {
        py = st_i[S_PY]; px = st_i[S_PX];
    }
    int reason = STOP_NONE, stop = 0;
    if (it >= max_iter) {  // the search after the last component (the host launches planes with a budget only)
        stop = 1; reason = STOP_MAX_ITER;
    } else {
        const T maxval = __ldcg(r + (i64)py * p.nx + px);
        const T cthres = use_factor ? mul_rn(thres, __ldg(thres_factor + k)) : thres;  // thres x 2^(it / speedup)
        if (!finite_t(absval)) {
            reason = STOP_DIVERGED;
        } else if (le(absval, cthres)) {
            reason = STOP_THRESHOLD;
        } else {
            int started = st_i[S_STARTED], n_above = st_i[S_ABOVE];
            T start_peak = from_f64<T>(st_d[D_START_PEAK]);
            double min_ss = st_d[D_MIN_SS];
            if (!started) { started = 1; start_peak = absval; min_ss = s; }
            if (max_iter_div >= 0) {
                if (gt(absval, mul_rn(peak_factor, start_peak))) reason = STOP_DIVERGED;
                else if (s > mul_rn(mul_rn(rms_factor, rms_factor), min_ss)) {
                    if (++n_above >= max_iter_div) reason = STOP_DIVERGED;
                } else {
                    n_above = 0;
                }
            }
            if (reason == STOP_NONE && s < min_ss) min_ss = s;
            st_i[S_STARTED] = started; st_i[S_ABOVE] = n_above;
            st_d[D_START_PEAK] = to_f64(start_peak); st_d[D_MIN_SS] = min_ss;
        }
        if (reason != STOP_NONE) {
            stop = 1;
        } else {
            const T pv = mul_rn(gain, maxval);
            const i64 idx = (i64)py * p.nx + px;
            model[idx] = add_rn(model[idx], pv);
            if (record) { h_peak[k] = to_f64(maxval); h_l[k] = py; h_m[k] = px; h_flux[k] = to_f64(pv); }
            st_d[D_PV] = to_f64(pv); st_i[S_SUB] = 1; st_i[S_COMPONENTS] = it + 1;
        }
    }
    st_i[S_PY] = py; st_i[S_PX] = px;
    if (stop) { st_i[S_STOP] = 1; st_i[S_REASON] = reason; st_i[S_SUB] = 0; }
}

// Block 0: the largest row maximum (the lowest row first), the row sums added in row order, then decide().
template <typename T, int ENTRY>
__device__ void phase2_exact(int it, int it_begin, const T* r, T* model, const Plane& p, int max_iter, T gain, T thres,
                             const T* thres_factor, int use_factor, int max_iter_div, double rms_factor,
                             T peak_factor, const u64* row_key, const int* row_ix, const double* row_ss, int* st_i,
                             double* st_d, int record, double* h_peak, double* h_ss, int* h_l, int* h_m,
                             double* h_flux) {
    __shared__ u64 s_key[32];
    __shared__ int s_row[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
    u64 bk = 0ull;
    int br = -1;
    for (int j = p.j0 + tid; j <= p.j1; j += blockDim.x) {
        const u64 k = __ldcg(row_key + j);
        if (k > bk) { bk = k; br = j; }
    }
    for (int off = 16; off; off >>= 1) {
        const u64 ok = __shfl_down_sync(FULL_MASK, bk, off);
        const int orow = __shfl_down_sync(FULL_MASK, br, off);
        if (ok > bk || (ok == bk && orow < br)) { bk = ok; br = orow; }
    }
    if (lane == 0) { s_key[warp] = bk; s_row[warp] = br; }
    __syncthreads();
    if (warp == 0) {
        bk = lane < nwarps ? s_key[lane] : 0ull;
        br = lane < nwarps ? s_row[lane] : -1;
        for (int off = 16; off; off >>= 1) {
            const u64 ok = __shfl_down_sync(FULL_MASK, bk, off);
            const int orow = __shfl_down_sync(FULL_MASK, br, off);
            if (ok > bk || (ok == bk && orow < br)) { bk = ok; br = orow; }
        }
        double s = 0.0;
        for (int base = p.j0; base <= p.j1; base += 32) {
            const int j = base + lane, cnt = min(32, p.j1 - base + 1);
            const double v = (j <= p.j1) ? __ldcg(row_ss + j) : 0.0;
            double x[32];
#pragma unroll
            for (int k = 0; k < 32; ++k) x[k] = __shfl_sync(FULL_MASK, v, k);
#pragma unroll
            for (int k = 0; k < 32; ++k) if (k < cnt) s = add_rn(s, x[k]);
        }
        if (lane == 0)
            decide<T, ENTRY>(it, it_begin, s, br >= 0, bk, br, br >= 0 ? __ldcg(row_ix + br) : 0, r, model, p, max_iter,
                             gain, thres, thres_factor, use_factor, max_iter_div, rms_factor, peak_factor, st_i, st_d,
                             record, h_peak, h_ss, h_l, h_m, h_flux);
    }
    __syncthreads();
}

// The iterations [it_begin, it_end) of one plane; see the top of the file.
template <typename T, int TW, int ENTRY>
__global__ void __launch_bounds__(256) hogbom_exact_chunk(
        T* r, const T* psf, const u8* mask, T* model, int nx, int ny, int pnx, int pny, int i0, int i1, int j0, int j1,
        int has_mask, int xbeg, int ybeg, int max_iter, T gain, T thres, const T* thres_factor, int use_factor,
        int max_iter_div, double rms_factor, T peak_factor, u64* row_key, int* row_ix, double* row_ss, int* st_i,
        double* st_d, int record, double* h_peak, double* h_ss, int* h_l, int* h_m, double* h_flux, unsigned int* bar,
        unsigned int spin_limit, int it_begin, int it_end) {
    __shared__ StripSmem<T, TW> sm;
    const Plane p{nx, ny, pnx, pny, i0, i1, j0, j1, has_mask, xbeg, ybeg};
    const int nstrips = (ny + 31) >> 5;
    for (int it = it_begin; it < it_end; ++it) {
        if (__ldcg(st_i + S_STOP)) break;  // written by block 0 between the barriers of the previous iteration
        const int do_sub = __ldcg(st_i + S_SUB), py = __ldcg(st_i + S_PY), px = __ldcg(st_i + S_PX);
        const T pv = from_f64<T>(__ldcg(st_d + D_PV));
        for (int s = blockIdx.x; s < nstrips; s += gridDim.x)
            strip_exact<T, TW>(sm, r, psf, mask, p, s << 5, do_sub, py, px, pv, row_key, row_ix, row_ss);
        if (!grid_sync(bar, gridDim.x, spin_limit)) return;
        if (blockIdx.x == 0)
            phase2_exact<T, ENTRY>(it, it_begin, r, model, p, max_iter, gain, thres, thres_factor, use_factor,
                                   max_iter_div, rms_factor, peak_factor, row_key, row_ix, row_ss, st_i, st_d, record,
                                   h_peak, h_ss, h_l, h_m, h_flux);
        if (!grid_sync(bar, gridDim.x, spin_limit)) return;
    }
}

// ------------------------------------------------------------------------------------------------- the summary
#define SUM_THREADS 256
#define SUM_PER_THREAD 16
#define SUM_TILE (SUM_THREADS * SUM_PER_THREAD)

// Per tile of SUM_TILE pixels (one block): the extrema of the residual inside the mask, from min_init and max_init (the
// 1e20 and -1e20 of maximg; a NaN never wins), and the number of model values that are not +-0.
template <typename T>
__global__ void __launch_bounds__(SUM_THREADS) plane_summary(const T* __restrict__ r, const u8* __restrict__ mask,
                                                             int has_mask, const T* __restrict__ model, i64 n,
                                                             T min_init, T max_init, T* part_min, T* part_max,
                                                             int* part_nz) {
    __shared__ T s_min[SUM_THREADS / 32], s_max[SUM_THREADS / 32];
    __shared__ int s_nz[SUM_THREADS / 32];
    const i64 base = (i64)blockIdx.x * SUM_TILE + (i64)threadIdx.x * SUM_PER_THREAD;
    T mn = min_init, mx = max_init;
    int nz = 0;
    for (int e = 0; e < SUM_PER_THREAD; ++e) {
        const i64 i = base + e;
        if (i < n) {
            if (!has_mask || mask[i]) {
                const T v = r[i];
                if (gt(v, mx)) mx = v;
                if (lt(v, mn)) mn = v;
            }
            nz += nonzero_t(model[i]) ? 1 : 0;
        }
    }
    for (int off = 16; off; off >>= 1) {
        const T omn = __shfl_down_sync(FULL_MASK, mn, off), omx = __shfl_down_sync(FULL_MASK, mx, off);
        nz += __shfl_down_sync(FULL_MASK, nz, off);
        if (lt(omn, mn)) mn = omn;
        if (gt(omx, mx)) mx = omx;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { s_min[warp] = mn; s_max[warp] = mx; s_nz[warp] = nz; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < SUM_THREADS / 32; ++w) {
            if (lt(s_min[w], mn)) mn = s_min[w];
            if (gt(s_max[w], mx)) mx = s_max[w];
            nz += s_nz[w];
        }
        part_min[blockIdx.x] = mn; part_max[blockIdx.x] = mx; part_nz[blockIdx.x] = nz;
    }
}

// The |model| values that are not +-0, in plane order: block b takes tile tile0 + b and writes its values from
// out[offsets[b]] on (the host computed the offsets from plane_summary's counts).
template <typename T>
__global__ void __launch_bounds__(SUM_THREADS) gather_abs(const T* __restrict__ model, i64 n, int tile0,
                                                          const int* __restrict__ part_nz,
                                                          const i64* __restrict__ offsets, T* __restrict__ out) {
    __shared__ int s_warp[SUM_THREADS / 32];
    if (part_nz[tile0 + blockIdx.x] == 0) return;
    const i64 base = (i64)(tile0 + blockIdx.x) * SUM_TILE + (i64)threadIdx.x * SUM_PER_THREAD;
    int c = 0;
    for (int e = 0; e < SUM_PER_THREAD; ++e) c += (base + e < n && nonzero_t(model[base + e])) ? 1 : 0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int incl = c;  // inclusive scan of the counts over the warp, then over the warps
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(FULL_MASK, incl, off);
        if (lane >= off) incl += y;
    }
    if (lane == 31) s_warp[warp] = incl;
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += s_warp[w];
    i64 pos = offsets[blockIdx.x] + before + incl - c;
    for (int e = 0; e < SUM_PER_THREAD; ++e) {
        const i64 i = base + e;
        if (i < n) {
            const T v = model[i];
            if (nonzero_t(v)) out[pos++] = abs_t(v);
        }
    }
}
