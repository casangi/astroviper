// Every transform of fft.hpp, in float and double: within a tolerance of a
// direct summation, and identical to the bit for every thread count (a count
// below 1 runs one thread). fft.hpp must hand pocketfft max(1, count) threads,
// and the sizes are large enough that pocketfft runs both passes of a 2-D
// transform on several threads.

#include "fft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreadCounts[] = {-1, 0, 2, 3, 8};  // each compared with one thread

template <typename T>
std::vector<T> random_values(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::vector<T> values(n);
    for (auto& value : values)
        value = static_cast<T>(static_cast<double>(generator() >> 11) * 0x1.0p-52 - 1.0);
    return values;
}

template <typename T>
std::vector<std::complex<T>> random_complex(std::size_t n, std::uint64_t seed) {
    const std::vector<T> parts = random_values<T>(2 * n, seed);
    std::vector<std::complex<T>> values(n);
    for (std::size_t i = 0; i < n; ++i) values[i] = std::complex<T>(parts[2 * i], parts[2 * i + 1]);
    return values;
}

std::vector<std::complex<double>> direct_dft1(const std::vector<std::complex<double>>& in, bool forward) {
    const std::int64_t n = static_cast<std::int64_t>(in.size());
    std::vector<std::complex<double>> twiddle(in.size()), out(in.size());
    for (std::int64_t q = 0; q < n; ++q) {
        const double angle = (forward ? -2.0 : 2.0) * std::numbers::pi * static_cast<double>(q) / static_cast<double>(n);
        twiddle[static_cast<std::size_t>(q)] = std::complex<double>(std::cos(angle), std::sin(angle));
    }
    for (std::int64_t k = 0; k < n; ++k) {
        std::complex<double> sum = 0.0;
        for (std::int64_t j = 0; j < n; ++j)
            sum += in[static_cast<std::size_t>(j)] * twiddle[static_cast<std::size_t>(k * j % n)];
        out[static_cast<std::size_t>(k)] = forward ? sum : sum / static_cast<double>(n);
    }
    return out;
}

// Direct summation of the 2-D DFT of an [ny][nx] image, row by row and then
// column by column; the inverse is normalized by nx*ny.
std::vector<std::complex<double>> direct_dft2(const std::vector<std::complex<double>>& in, int nx, int ny,
                                              bool forward) {
    std::vector<std::complex<double>> out(in.size()), line;
    for (int j = 0; j < ny; ++j) {
        line.assign(in.begin() + static_cast<std::ptrdiff_t>(j) * nx, in.begin() + static_cast<std::ptrdiff_t>(j + 1) * nx);
        line = direct_dft1(line, forward);
        std::copy(line.begin(), line.end(), out.begin() + static_cast<std::ptrdiff_t>(j) * nx);
    }
    for (int i = 0; i < nx; ++i) {
        line.resize(static_cast<std::size_t>(ny));
        for (int j = 0; j < ny; ++j) line[static_cast<std::size_t>(j)] = out[static_cast<std::size_t>(j) * nx + i];
        line = direct_dft1(line, forward);
        for (int j = 0; j < ny; ++j) out[static_cast<std::size_t>(j) * nx + i] = line[static_cast<std::size_t>(j)];
    }
    return out;
}

template <typename T>
std::vector<std::complex<double>> widen(const std::vector<T>& values) {
    return std::vector<std::complex<double>>(values.begin(), values.end());
}

// The first nx/2 + 1 columns of a full [ny][nx] spectrum.
std::vector<std::complex<double>> half_of(const std::vector<std::complex<double>>& full, int nx, int ny) {
    const int hx = nx / 2 + 1;
    std::vector<std::complex<double>> half(static_cast<std::size_t>(hx) * ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < hx; ++i)
            half[static_cast<std::size_t>(j) * hx + i] = full[static_cast<std::size_t>(j) * nx + i];
    return half;
}

template <typename T>
std::vector<std::complex<T>> narrow(const std::vector<std::complex<double>>& values) {
    std::vector<std::complex<T>> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        out[i] = std::complex<T>(static_cast<T>(values[i].real()), static_cast<T>(values[i].imag()));
    return out;
}

void check_close(const std::vector<std::complex<double>>& got, const std::vector<std::complex<double>>& reference,
                 double tolerance, const std::string& what) {
    if (got.size() != reference.size()) throw std::runtime_error(what + ": wrong size");
    double scale = 0.0, error = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        scale = std::max(scale, std::abs(reference[i]));
        error = std::max(error, std::abs(got[i] - reference[i]));
    }
    if (!(error <= tolerance * scale))
        throw std::runtime_error(what + ": differs from the direct summation by " + std::to_string(error) +
                                 " of " + std::to_string(scale));
}

template <typename V>
bool same_bits(const V& a, const V& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
}

// pocketfft's own rule: the sizes must give several threads to the rows and
// the columns of the half spectrum, or the identity checks prove nothing.
template <typename T>
void check_sizes_use_threads(int nx, int ny) {
    const std::size_t vlen = pocketfft::detail::VLEN<T>::val;
    const pocketfft::shape_t image{static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
    const pocketfft::shape_t half{static_cast<std::size_t>(ny), static_cast<std::size_t>(nx / 2 + 1)};
    if (pocketfft::detail::util::thread_count(8, image, 1, vlen) < 2 ||
        pocketfft::detail::util::thread_count(8, half, 0, vlen) < 2)
        throw std::runtime_error("the image is too small for pocketfft to use several threads");
}

// transform(threads) for one thread and for every count of kThreadCounts,
// which must give the same bits; returns the result of one thread.
template <typename Transform>
auto for_each_thread_count(Transform transform, const std::string& what) {
    const auto one_thread = transform(1);
    for (int threads : kThreadCounts)
        if (!same_bits(one_thread, transform(threads)))
            throw std::runtime_error(what + ": " + std::to_string(threads) + " threads differ from one thread");
    return one_thread;
}

template <typename T>
void check_transforms(int nx, int ny, double tolerance) {
    check_sizes_use_threads<T>(nx, ny);
    const std::size_t n = static_cast<std::size_t>(nx) * ny;
    const std::vector<T> image = random_values<T>(n, 1);
    const std::vector<std::complex<double>> spectrum = direct_dft2(widen(image), nx, ny, true);
    const std::vector<std::complex<double>> half = half_of(spectrum, nx, ny);

    const auto half_aligned = for_each_thread_count(
        [&](int threads) {
            std::vector<std::complex<T>> out;
            astroviper::fft::rfft2_half(image.data(), nx, ny, out, threads);
            return out;
        },
        "rfft2_half");
    check_close(widen(half_aligned), half, tolerance, "rfft2_half");
    if (!same_bits(half_aligned, astroviper::fft::rfft2_half(image.data(), nx, ny, 1)))
        throw std::runtime_error("rfft2_half returning the spectrum differs from rfft2_half into a vector");

    // The same image one byte off its natural alignment: the row-by-row path.
    std::vector<unsigned char> bytes(n * sizeof(T) + sizeof(T));
    std::memcpy(bytes.data() + 1, image.data(), n * sizeof(T));
    const T* unaligned = reinterpret_cast<const T*>(bytes.data() + 1);
    const auto half_unaligned = for_each_thread_count(
        [&](int threads) { return astroviper::fft::rfft2_half(unaligned, nx, ny, threads); },
        "rfft2_half, unaligned input");
    check_close(widen(half_unaligned), half, tolerance, "rfft2_half, unaligned input");

    const auto full = for_each_thread_count(
        [&](int threads) { return astroviper::fft::rfft2_forward(image.data(), nx, ny, threads); }, "rfft2_forward");
    check_close(widen(full), spectrum, tolerance, "rfft2_forward");

    const std::vector<std::complex<T>> complex_image = random_complex<T>(n, 2);
    const auto forward = for_each_thread_count(
        [&](int threads) { return astroviper::fft::cfft2_forward(complex_image, nx, ny, threads); }, "cfft2_forward");
    check_close(widen(forward), direct_dft2(widen(complex_image), nx, ny, true), tolerance, "cfft2_forward");
    const auto inverse = for_each_thread_count(
        [&](int threads) { return astroviper::fft::cfft2_inverse(complex_image, nx, ny, threads); }, "cfft2_inverse");
    check_close(widen(inverse), direct_dft2(widen(complex_image), nx, ny, false), tolerance, "cfft2_inverse");

    // The inverse transforms of the image's spectrum give the image back.
    const std::vector<std::complex<T>> half_spectrum = narrow<T>(half);
    const auto from_half = for_each_thread_count(
        [&](int threads) {
            std::vector<std::complex<T>> scratch = half_spectrum;
            std::vector<T> out(n);
            astroviper::fft::irfft2_half(scratch.data(), nx, ny, out.data(), threads);
            return out;
        },
        "irfft2_half");
    check_close(widen(from_half), widen(image), tolerance, "irfft2_half");
    const std::vector<std::complex<T>> full_spectrum = narrow<T>(spectrum);
    const auto from_full = for_each_thread_count(
        [&](int threads) {
            std::vector<T> out(n);
            astroviper::fft::irfft2_inverse(full_spectrum, nx, ny, out.data(), threads);
            return out;
        },
        "irfft2_inverse");
    check_close(widen(from_full), widen(image), tolerance, "irfft2_inverse");

    for (int length : {nx, ny}) {
        const std::vector<T> line = random_values<T>(static_cast<std::size_t>(length), 3 + length);
        const std::vector<std::complex<double>> line_spectrum = direct_dft1(widen(line), true);
        const auto spectrum_1d = for_each_thread_count(
            [&](int threads) { return astroviper::fft::fft1d_forward(line, threads); }, "fft1d_forward");
        check_close(widen(spectrum_1d), line_spectrum, tolerance, "fft1d_forward");
        const auto half_1d = for_each_thread_count(
            [&](int threads) { return astroviper::fft::rfft1d_half(line, threads); }, "rfft1d_half");
        check_close(widen(half_1d), half_of(line_spectrum, length, 1), tolerance, "rfft1d_half");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected one scenario name");
        using astroviper::fft::detail::pocketfft_threads;
        if (pocketfft_threads(-1) != 1 || pocketfft_threads(0) != 1 || pocketfft_threads(3) != 3 ||
            pocketfft_threads(8) != 8)
            throw std::runtime_error("fft.hpp must hand pocketfft max(1, count) threads");
        const std::string scenario = argv[1];
        if (scenario == "double_64x64") {
            check_transforms<double>(64, 64, 1e-13);
        } else if (scenario == "double_101x97") {
            check_transforms<double>(101, 97, 1e-13);
        } else if (scenario == "float_128x128") {
            check_transforms<float>(128, 128, 1e-5);
        } else if (scenario == "float_101x97") {
            check_transforms<float>(101, 97, 1e-5);
        } else {
            throw std::runtime_error("Unknown scenario");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
