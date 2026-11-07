#pragma once

// The fast Hogbom minor cycle: the results of the reference's clean_cube and
// clean_cube_many_threads (deconvolvers/hogbom, hclean.cpp and the cube
// summary of its binding), bit for bit, from fewer and faster passes.
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
// Every pixel therefore undergoes the reference's sequence of rounded
// operations, and the row sums and peaks are formed exactly as the
// reference's scan_row forms them (hclean_fast_kernels.inc), so the residual,
// model, iterations, stop codes and iteration history are the reference's.
//
// Threads: a call uses up to processing_function_threads of them (< 1 counts
// as 1, as in both reference entries), and no more than the CPUs it may run
// on and its work justify. Planes run in crews: one crew of every thread for
// large planes, one after another, or several smaller crews side by side for
// small planes (a thread alone on a plane when they are small enough). A crew
// synchronises at every iteration; its threads wait for the next step by
// polling for a bounded time that adapts to how their recent waits ended, and
// then sleep. Planes too small for a crew of two, with no more threads than
// planes, run as the reference clean_cube runs them: each thread cleans whole
// planes from its start and waits for none. Every thread of a call has ended
// when it returns. Unlike
// the reference clean_cube, which runs one thread per plane, the fast one
// puts several threads on one plane when that is faster: the same results,
// from more cores at once. The results never depend on the number of threads
// or on the instruction set.

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
