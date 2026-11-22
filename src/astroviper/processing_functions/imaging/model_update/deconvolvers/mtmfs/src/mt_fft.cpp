// FFT helpers of the MTMFS deconvolver. See mt_fft.hpp.

#include "../include/mt_fft.hpp"

namespace mtmfs {

template <typename T>
void recenter_convolution(T* real, int nx, int ny, T* scratch, int centered_kernel_count) {
    // Cancel each kernel's center offset; handle parity independently per axis.
    const int hx = (nx - (centered_kernel_count * (nx / 2)) % nx) % nx;
    const int hy = (ny - (centered_kernel_count * (ny / 2)) % ny) % ny;
    if (hx == 0 && hy == 0) return;
    for (int iy = 0; iy < ny; ++iy) {
        const int sy = (iy + hy) % ny;
        for (int ix = 0; ix < nx; ++ix) {
            const int sx = (ix + hx) % nx;
            scratch[static_cast<std::size_t>(sy) * nx + sx] = real[static_cast<std::size_t>(iy) * nx + ix];
        }
    }
    const std::size_t n = static_cast<std::size_t>(nx) * ny;
    for (std::size_t i = 0; i < n; ++i) real[i] = scratch[i];
}

template void recenter_convolution<float>(float*, int, int, float*, int);
template void recenter_convolution<double>(double*, int, int, double*, int);

}  // namespace mtmfs
