// Compile probe of the fast Asp module (cmake/AstroviperKernels.cmake): the
// language and library features its sources use, POSIX mmap (its large
// buffers), and pocketfft's SIMD lanes, which its transforms need (pocketfft
// turns them off for compilers it does not know). The reference's include
// directory is on the probe's path. The probe is compiled and linked, never
// run: the value main returns only states what the calls are for.
#include <sys/mman.h>

#include <atomic>
#include <cstdlib>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include "pocketfft_hdronly.h"

#ifdef POCKETFFT_NO_VECTORS
#error "pocketfft has no SIMD lanes with this compiler"
#endif

int main() {
    namespace pd = pocketfft::detail;
    thread_local std::vector<pd::vtype_t<double>> lanes;
    std::atomic<int> calls{0};
    std::thread worker([&] {
        lanes.resize(4);
        calls.fetch_add(1);
    });
    worker.join();
    pd::threading::thread_map(2, [&] { calls.fetch_add(1); });
    auto plan = pd::get_plan<pd::pocketfft_r<double>>(8);
    std::vector<double> line(8, 1.0);
    plan->exec(line.data(), 1.0, true);
    void* buffer = std::aligned_alloc(64, 128);
    std::free(buffer);
    void* mapped = mmap(nullptr, 1 << 22, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    const bool unmapped = mapped != MAP_FAILED && munmap(mapped, 1 << 22) == 0;
    std::deque<std::list<int>> cache(1);
    return buffer != nullptr && unmapped && calls.load() == 3 && line[0] == 8.0 && cache.size() == 1 ? 0 : 1;
}
