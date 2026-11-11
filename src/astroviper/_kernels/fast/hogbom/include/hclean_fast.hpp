#pragma once

// The fast Hogbom minor cycle: the results of the reference's clean_cube and
// clean_cube_many_threads (deconvolvers/hogbom, hclean.cpp and the cube
// summary of its binding), bit for bit, from fewer and faster passes. Bit for
// bit means every value that is not a NaN: a result that is a NaN is a NaN in
// both, but its sign and payload are unspecified. Which payload an operation
// on two NaNs keeps depends on the operand order that the compiler picks for
// a commutative operation, in the reference as here (the reference's own
// choice differs between its scan, its last sum and its two entries).
//
// Each plane is split into
//   S  the searched pixels: the clean box (shrunk to the mask's bounding box,
//      as the reference does) inside the mask. They are the only pixels that
//      the peak search and the stop tests read, so they are kept current: one
//      pass per iteration subtracts the previous component from them and
//      searches them for the next one (the fused pass). The pass also keeps
//      current the other pixels of the clean box that lie in short runs (of
//      up to 32 pixels) between searched ones, which cost less updated at
//      once than deferred;
//   E  every other pixel. Nothing reads them during the model update, so
//      their subtractions are deferred to its end and then applied component
//      after component in the original order (or earlier, when a zero peak
//      that a negative threshold keeps running points at one of them).
// Small planes (of at most 8,192 pixels, or of at most 64 columns and 65,536
// pixels) are not split: on them the plans, the per-row results and the
// deferred subtractions over short runs would cost more than they save. Each
// iteration's pass there (the direct pass) subtracts the previous component
// from its whole footprint and then scans the box, as the reference does in
// two passes. A box of at most 64 columns, of at least 8 rows or on a plane of
// at most 64 columns and as many rows, is held column by column after the
// plane's first 9 passes (when its budget exceeds 16), so that the lanes take
// a row each instead of a few pixels of one. Every pixel therefore undergoes
// the reference's sequence of rounded operations, and the row sums and peaks
// are formed exactly as the reference's scan_row forms them
// (hclean_fast_kernels.inc), so the residual, model, iterations, stop codes
// and iteration history are the reference's.
//
// Threads: a call uses up to processing_function_threads of them (< 1 counts
// as 1, as in both reference entries), and no more than the CPUs it may run
// on and its work justify. Small planes, and planes too small for a crew of
// two, get no more threads than planes with a budget, and run as the
// reference clean_cube runs them: each thread cleans whole planes from its
// start and waits for none (small planes on up to about sqrt(work / the cost
// of a thread) threads, which start one after another while planes are left
// that none has taken; when their boxes are small, a plane that stops at its
// first search counts for none). A small clean box on a plane of at least
// 65,536 pixels, whose final passes subtract the deferred components from the
// rest of the plane, has them shared by a team. Other planes run in crews:
// whole planes per thread when at least twice as many planes as threads
// iterate, else one crew for large planes (or a lone iterating one), which
// runs them one after another, or several smaller crews side by side. A plane
// that stops at its first search (Stokes V beside I) counts for no crew. A
// crew synchronises at every iteration; its threads wait for the next step by
// polling for a bounded time that adapts to how their recent waits ended, and
// then sleep. Every thread of a call has ended when it returns. Unlike the
// reference clean_cube, which runs one thread per plane, the fast one puts
// several threads on one plane when that is faster: the same results, from
// more cores at once. The results never depend on the number of threads or on
// the instruction set.
//
// Memory: besides its inputs and the iteration history, a call keeps the
// plans of its split planes with a budget (a few bytes per row and per
// segment of each) and at most 16,384 deferred subtractions per plane in
// flight (32 bytes each), which are applied, in order, whenever that many are
// pending; and for a small plane held column by column, a copy of the plane,
// its mask and its PSF per thread.

#include <cstdint>

#include "hclean.hpp"
#include "hclean_fast_kernels.hpp"

namespace hclean_fast {

using hclean::DivergenceControl;
using hclean::IterationHistory;

// The reference entry whose results a call reproduces. The two differ only
// in corner cases: the pixel of a zero peak that a negative threshold keeps
// running (clean_cube keeps the previous peak's pixel, (0, 0) before the
// first; clean_cube_many_threads takes the first pixel of the clean box), and
// the iterations reported for a negative budget (clean_cube reports the
// budget, clean_cube_many_threads 0).
enum class Entry { clean_cube, clean_cube_many_threads };

// A call, with the arguments as the reference binding validates and converts
// them, and its per-plane results. Planes are in (time, frequency,
// polarization) C order.
template <typename T>
struct Cube {
    Entry entry;
    T* residual;  // (nt, nf, np_img, ny, nx), updated in place
    T* model;     // likewise
    const T* psf;  // (nt, nf, np_psf, psf_ny, psf_nx)
    const std::uint8_t* mask;  // (nt, nf, np_img, ny, nx) bool bytes (not 0: searched), or null
    int nt, nf, np_img, np_psf, ny, nx, psf_ny, psf_nx;
    int xbeg, xend, ybeg, yend;  // the clean box as the binding clamps it
    const int* max_iter;         // per plane
    const T* threshold;          // per plane
    T gain, speedup;
    DivergenceControl<T> divergence;
    int threads;  // processing_function_threads
    // The results, per plane. The history is null when it is not recorded.
    int* iterations;
    int* stop_code;
    IterationHistory* history;
    T* final_peak;
    T* total_flux;
    bool* converged;
};

// Runs a call. Returns false, having written nothing, if it cannot allocate
// what it needs before the first write (the caller then runs the reference).
// An exception after that (std::bad_alloc of a growing component list or
// iteration history) propagates, as the reference's would.
template <typename T>
bool clean_cube(const Cube<T>& cube);

// The routines of the instruction set chosen for this process
// (astroviper::fast::select_isa of fast/common/isa.hpp), read once, at the
// first call (the module's import makes it).
const Kernels& kernels();

extern template bool clean_cube<float>(const Cube<float>&);
extern template bool clean_cube<double>(const Cube<double>&);

}  // namespace hclean_fast
