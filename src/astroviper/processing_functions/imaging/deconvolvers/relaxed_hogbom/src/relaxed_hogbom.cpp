#include "../include/relaxed_hogbom.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include "asp_fft.hpp"
#include "asp_lbfgs.hpp"

namespace relaxed_hogbom {
namespace {

using Complex = std::complex<double>;

struct Component {
    double flux;
    double x;
    double y;
};

double squared_norm(const std::vector<double>& image) {
    double value = 0.0;
    for (double pixel : image) value += pixel * pixel;
    return value;
}

template<typename T>
double squared_norm(const T* image, std::size_t size) {
    double value = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double pixel = static_cast<double>(image[i]);
        value += pixel * pixel;
    }
    return value;
}

int wrap(int value, int size) {
    value %= size;
    return value < 0 ? value + size : value;
}

std::vector<double> correlate(const std::vector<double>& left,
                              const std::vector<double>& right,
                              int nx, int ny) {
    auto lhs = aspfft::rfft2_forward(left.data(), nx, ny);
    const auto rhs = aspfft::rfft2_forward(right.data(), nx, ny);
    for (std::size_t i = 0; i < lhs.size(); ++i) lhs[i] *= std::conj(rhs[i]);
    std::vector<double> result(lhs.size());
    aspfft::irfft2_inverse(std::move(lhs), nx, ny, result.data());
    return result;
}

void render(const std::vector<Component>& components,
            std::vector<double>& image, int nx, int ny) {
    std::fill(image.begin(), image.end(), 0.0);
    for (const Component& component : components) {
        const double px = std::clamp(component.x, 0.0, double(nx - 1));
        const double py = std::clamp(component.y, 0.0, double(ny - 1));
        const int x0 = static_cast<int>(std::floor(px));
        const int y0 = static_cast<int>(std::floor(py));
        const int x1 = std::min(nx - 1, x0 + 1);
        const int y1 = std::min(ny - 1, y0 + 1);
        const double dx = px - x0;
        const double dy = py - y0;
        image[static_cast<std::size_t>(y0) * nx + x0] +=
            component.flux * (1.0 - dx) * (1.0 - dy);
        image[static_cast<std::size_t>(y0) * nx + x1] +=
            component.flux * dx * (1.0 - dy);
        image[static_cast<std::size_t>(y1) * nx + x0] +=
            component.flux * (1.0 - dx) * dy;
        image[static_cast<std::size_t>(y1) * nx + x1] +=
            component.flux * dx * dy;
    }
}

std::size_t occupied_pixel_count(const std::vector<double>& model) {
    return static_cast<std::size_t>(std::count_if(
        model.begin(), model.end(), [](double value) { return value != 0.0; }));
}

void convolve_sparse(const std::vector<double>& model,
                     const std::vector<double>& psf,
                     std::vector<double>& output, int nx, int ny) {
    std::fill(output.begin(), output.end(), 0.0);
    const int cx = nx / 2;
    const int cy = ny / 2;
    for (int sy = 0; sy < ny; ++sy) {
        for (int sx = 0; sx < nx; ++sx) {
            const double amplitude = model[static_cast<std::size_t>(sy) * nx + sx];
            if (amplitude == 0.0) continue;
            const int shift_x = sx - cx;
            const int shift_y = sy - cy;
            for (int py = 0; py < ny; ++py) {
                const int y = wrap(py + shift_y, ny);
                for (int px = 0; px < nx; ++px) {
                    const int x = wrap(px + shift_x, nx);
                    output[static_cast<std::size_t>(y) * nx + x] +=
                        amplitude * psf[static_cast<std::size_t>(py) * nx + px];
                }
            }
        }
    }
}

void recenter_convolution(std::vector<double>& image, int nx, int ny) {
    const int cx = nx / 2;
    const int cy = ny / 2;
    std::vector<double> centered(image.size());
    for (int y = 0; y < ny; ++y) {
        const int source_y = (y + cy) % ny;
        for (int x = 0; x < nx; ++x) {
            const int source_x = (x + cx) % nx;
            centered[static_cast<std::size_t>(y) * nx + x] =
                image[static_cast<std::size_t>(source_y) * nx + source_x];
        }
    }
    image.swap(centered);
}

class AdaptiveConvolver {
public:
    AdaptiveConvolver(const std::vector<double>& psf, int nx, int ny)
        : psf_(psf), nx_(nx), ny_(ny) {}

    void convolve(const std::vector<double>& model,
                  std::vector<double>& output) {
        const std::size_t occupied_pixels = occupied_pixel_count(model);
        if (prefer_sparse_convolution(model.size(), occupied_pixels)) {
            convolve_sparse(model, psf_, output, nx_, ny_);
            return;
        }
        if (psf_spectrum_.empty())
            psf_spectrum_ = aspfft::rfft2_forward(psf_.data(), nx_, ny_);
        auto model_spectrum = aspfft::rfft2_forward(model.data(), nx_, ny_);
        for (std::size_t i = 0; i < model_spectrum.size(); ++i)
            model_spectrum[i] *= psf_spectrum_[i];
        output.resize(model.size());
        aspfft::irfft2_inverse(
            std::move(model_spectrum), nx_, ny_, output.data());
        recenter_convolution(output, nx_, ny_);
    }

private:
    const std::vector<double>& psf_;
    int nx_;
    int ny_;
    std::vector<Complex> psf_spectrum_;
};

double psf_width(const std::vector<double>& psf, int nx, int ny) {
    const int cx = nx / 2;
    const int cy = ny / 2;
    const double peak = std::abs(psf[static_cast<std::size_t>(cy) * nx + cx]);
    if (!(peak > 0.0)) return 1.0;
    const double half = 0.5 * peak;
    int left = cx;
    int right = cx;
    int bottom = cy;
    int top = cy;
    while (left > 0 && std::abs(psf[static_cast<std::size_t>(cy) * nx + left]) >= half) --left;
    while (right + 1 < nx && std::abs(psf[static_cast<std::size_t>(cy) * nx + right]) >= half) ++right;
    while (bottom > 0 && std::abs(psf[static_cast<std::size_t>(bottom) * nx + cx]) >= half) --bottom;
    while (top + 1 < ny && std::abs(psf[static_cast<std::size_t>(top) * nx + cx]) >= half) ++top;
    const double wx = std::max(1, right - left - 1);
    const double wy = std::max(1, top - bottom - 1);
    return 0.5 * (wx + wy);
}

struct Parameterization {
    std::vector<Component> seed;
    std::vector<double> lower_x, upper_x, lower_y, upper_y;

    static double logistic(double value) {
        if (value >= 0.0) {
            const double e = std::exp(-value);
            return 1.0 / (1.0 + e);
        }
        const double e = std::exp(value);
        return e / (1.0 + e);
    }

    std::vector<Component> decode(const std::vector<double>& parameters,
                                  std::vector<double>* derivatives = nullptr) const {
        std::vector<Component> output(seed.size());
        if (derivatives != nullptr) derivatives->assign(parameters.size(), 0.0);
        for (std::size_t i = 0; i < seed.size(); ++i) {
            const std::size_t j = 3 * i;
            const double flux = std::copysign(std::exp(parameters[j]), seed[i].flux);
            const double sx = logistic(parameters[j + 1]);
            const double sy = logistic(parameters[j + 2]);
            output[i] = {flux,
                         lower_x[i] + (upper_x[i] - lower_x[i]) * sx,
                         lower_y[i] + (upper_y[i] - lower_y[i]) * sy};
            if (derivatives != nullptr) {
                (*derivatives)[j] = flux;
                (*derivatives)[j + 1] = (upper_x[i] - lower_x[i]) * sx * (1.0 - sx);
                (*derivatives)[j + 2] = (upper_y[i] - lower_y[i]) * sy * (1.0 - sy);
            }
        }
        return output;
    }
};

double logit_for(double value, double lower, double upper) {
    if (!(upper > lower)) return 0.0;
    const double fraction = std::clamp((value - lower) / (upper - lower), 1e-9, 1.0 - 1e-9);
    return std::log(fraction / (1.0 - fraction));
}

struct Objective {
    const std::vector<double>& residual_psf_correlation;
    const std::vector<double>& psf_autocorrelation;
    const Parameterization& parameterization;
    double base_norm;
    int nx;
    int ny;

    double operator()(const std::vector<double>& parameters,
                      std::vector<double>& gradient) const {
        std::vector<double> parameter_derivatives;
        const auto components = parameterization.decode(parameters, &parameter_derivatives);
        struct Tap {
            std::size_t component;
            int x, y;
            double amplitude, weight, dx, dy;
        };
        std::vector<Tap> taps;
        taps.reserve(4 * components.size());
        for (std::size_t i = 0; i < components.size(); ++i) {
            const auto& c = components[i];
            const int x0 = static_cast<int>(std::floor(c.x));
            const int y0 = static_cast<int>(std::floor(c.y));
            const int x1 = std::min(nx - 1, x0 + 1);
            const int y1 = std::min(ny - 1, y0 + 1);
            const double fx = c.x - x0;
            const double fy = c.y - y0;
            const double w[4] = {(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy};
            const double dwx[4] = {-(1-fy), 1-fy, -fy, fy};
            const double dwy[4] = {-(1-fx), -fx, 1-fx, fx};
            const int tx[4] = {x0, x1, x0, x1};
            const int ty[4] = {y0, y0, y1, y1};
            for (int k = 0; k < 4; ++k)
                taps.push_back({i, tx[k], ty[k], c.flux*w[k], w[k], dwx[k], dwy[k]});
        }
        const int cx = nx / 2;
        const int cy = ny / 2;
        double value = base_norm;
        std::vector<double> tap_gradient(taps.size(), 0.0);
        for (std::size_t a = 0; a < taps.size(); ++a) {
            const int lx = wrap(taps[a].x - cx, nx);
            const int ly = wrap(taps[a].y - cy, ny);
            const double cross = residual_psf_correlation[static_cast<std::size_t>(ly)*nx + lx];
            value -= 2.0 * taps[a].amplitude * cross;
            tap_gradient[a] = -2.0 * cross;
        }
        for (std::size_t a = 0; a < taps.size(); ++a) {
            for (std::size_t b = 0; b < taps.size(); ++b) {
                const int lx = wrap(taps[b].x - taps[a].x, nx);
                const int ly = wrap(taps[b].y - taps[a].y, ny);
                const double gram = psf_autocorrelation[static_cast<std::size_t>(ly)*nx + lx];
                value += taps[a].amplitude * taps[b].amplitude * gram;
                tap_gradient[a] += 2.0 * taps[b].amplitude * gram;
            }
        }
        gradient.assign(parameters.size(), 0.0);
        for (std::size_t k = 0; k < taps.size(); ++k) {
            const std::size_t j = 3 * taps[k].component;
            const double flux = components[taps[k].component].flux;
            gradient[j] += taps[k].weight * tap_gradient[k] * parameter_derivatives[j];
            gradient[j+1] += flux * taps[k].dx * tap_gradient[k] * parameter_derivatives[j+1];
            gradient[j+2] += flux * taps[k].dy * tap_gradient[k] * parameter_derivatives[j+2];
        }
        return value;
    }
};

template<typename T>
bool relax_plane(T* residual, T* model, const T* model_start, const T* psf_data,
                 int nx, int ny, const Controls& controls, int& count,
                 double& initial_norm, double& final_norm) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny;
    std::vector<Component> components;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * nx + x;
            const double added = static_cast<double>(model[i]) - static_cast<double>(model_start[i]);
            if (added > 0.0) components.push_back({added, double(x), double(y)});
        }
    }
    std::sort(components.begin(), components.end(), [](const Component& a, const Component& b) {
        return std::abs(a.flux) > std::abs(b.flux);
    });
    if (static_cast<int>(components.size()) > controls.max_active_components)
        components.resize(static_cast<std::size_t>(controls.max_active_components));
    count = static_cast<int>(components.size());
    initial_norm = squared_norm(residual, size);
    final_norm = initial_norm;
    if (components.empty()) return false;

    std::vector<double> psf(size), current_residual(size), selected(size), selected_response(size), base(size);
    for (std::size_t i = 0; i < size; ++i) {
        psf[i] = static_cast<double>(psf_data[i]);
        current_residual[i] = static_cast<double>(residual[i]);
    }
    AdaptiveConvolver convolver(psf, nx, ny);
    render(components, selected, nx, ny);
    convolver.convolve(selected, selected_response);
    for (std::size_t i = 0; i < size; ++i) base[i] = current_residual[i] + selected_response[i];

    const double trust = controls.constraint_radius_beams * psf_width(psf, nx, ny);
    Parameterization parameterization;
    parameterization.seed = components;
    std::vector<double> parameters(3 * components.size());
    std::vector<double> scale(parameters.size(), 1.0);
    for (std::size_t i = 0; i < components.size(); ++i) {
        const std::size_t j = 3 * i;
        parameterization.lower_x.push_back(std::max(0.0, components[i].x - trust));
        parameterization.upper_x.push_back(std::min(double(nx - 1), components[i].x + trust));
        parameterization.lower_y.push_back(std::max(0.0, components[i].y - trust));
        parameterization.upper_y.push_back(std::min(double(ny - 1), components[i].y + trust));
        parameters[j] = std::log(std::max(std::abs(components[i].flux), 1e-30));
        parameters[j+1] = logit_for(components[i].x, parameterization.lower_x[i], parameterization.upper_x[i]);
        parameters[j+2] = logit_for(components[i].y, parameterization.lower_y[i], parameterization.upper_y[i]);
    }
    Objective objective{correlate(base, psf, nx, ny), correlate(psf, psf, nx, ny),
                        parameterization, squared_norm(base), nx, ny};
    std::vector<double> initial_gradient;
    const double initial_objective = objective(parameters, initial_gradient);
    asplbfgs::Options options;
    options.m = 5;
    options.max_iters = std::max(1, controls.optimizer_max_iterations);
    options.epsg = 1e-6;
    options.epsf = 1e-8;
    options.epsx = 1e-5;
    asplbfgs::minimize(parameters, scale,
        [&objective](const std::vector<double>& x, std::vector<double>& gradient) {
            return objective(x, gradient);
        }, options);
    std::vector<double> final_gradient;
    const double final_objective = objective(parameters, final_gradient);
    if (!std::isfinite(final_objective) || !(final_objective < initial_objective)) return false;

    const auto fitted_components = parameterization.decode(parameters);
    std::vector<double> fitted(size), fitted_response(size), trial_residual(size);
    render(fitted_components, fitted, nx, ny);
    convolver.convolve(fitted, fitted_response);
    for (std::size_t i = 0; i < size; ++i) trial_residual[i] = base[i] - fitted_response[i];
    final_norm = squared_norm(trial_residual);
    const double improvement = (initial_norm - final_norm) /
        std::max(initial_norm, std::numeric_limits<double>::epsilon());
    if (!(improvement >= controls.minimum_improvement)) {
        final_norm = initial_norm;
        return false;
    }
    for (std::size_t i = 0; i < size; ++i) {
        model[i] = static_cast<T>(static_cast<double>(model[i]) - selected[i] + fitted[i]);
        residual[i] = static_cast<T>(trial_residual[i]);
    }
    return true;
}

template<typename T>
void clean_one_plane(T* residual, T* model, const T* psf, const bool* mask,
                     int nx, int ny, int xbeg, int xend, int ybeg, int yend,
                     int max_iterations, T gain, T threshold,
                     const hclean::DivergenceControl<T>& divergence,
                     const Controls& controls, int& iterations, int& stop_reason,
                     bool& attempted, bool& accepted, int& component_count,
                     double& initial_norm, double& final_norm, double& rms_ratio) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny;
    std::vector<T> model_start(model, model + size);
    const double pre_norm = squared_norm(residual, size);
    std::function<void(int, int, int, T)> no_progress =
        [](int, int, int, T) {};
    std::function<void(int&)> no_stop = [](int& value) { value = 0; };
    hclean::clean(model, residual, psf, mask == nullptr ? 0 : 1, mask,
                  nx, ny, xbeg, xend, ybeg, yend, max_iterations, 0,
                  iterations, gain, threshold, T(0), no_progress, no_stop,
                  divergence, stop_reason);
    const double post_norm = squared_norm(residual, size);
    bool activate = true;
    rms_ratio = std::sqrt(post_norm / std::max(pre_norm, std::numeric_limits<double>::epsilon()));
    if (controls.adaptive_start && iterations > 0) {
        const double normalized = std::pow(rms_ratio,
            double(controls.rms_normalization_iterations) / double(iterations));
        activate = normalized <= controls.rapid_progress_rms_ratio ||
                   normalized >= controls.stalled_progress_rms_ratio;
    }
    attempted = activate;
    initial_norm = post_norm;
    final_norm = post_norm;
    accepted = activate && relax_plane(residual, model, model_start.data(), psf,
                                       nx, ny, controls, component_count,
                                       initial_norm, final_norm);
}

}  // namespace

bool prefer_sparse_convolution(std::size_t image_pixels,
                               std::size_t occupied_pixels) {
    if (occupied_pixels == 0) return true;
    // CASA FFTServer benchmarks put the crossover at about 20 occupied taps
    // through 256^2 and about 10 taps for larger images.  Stay conservative
    // near a tie, and count rendered pixels rather than logical components.
    const std::size_t maximum_sparse_pixels =
        image_pixels <= 256u * 256u ? 20u : 10u;
    return occupied_pixels <= maximum_sparse_pixels;
}

void convolve_components(const double* model, const double* psf, double* output,
                         int nx, int ny) {
    const std::size_t size = static_cast<std::size_t>(nx) * ny;
    std::vector<double> model_vector(model, model + size);
    std::vector<double> psf_vector(psf, psf + size);
    std::vector<double> result(size);
    AdaptiveConvolver convolver(psf_vector, nx, ny);
    convolver.convolve(model_vector, result);
    std::copy(result.begin(), result.end(), output);
}

template<typename T>
void clean_cube(T* residual_cube, T* model_cube, const T* psf_cube,
                int domask, const bool* mask_cube,
                int nt, int nf, int np_img, int np_psf, int ny, int nx,
                int xbeg, int xend, int ybeg, int yend,
                const int* max_iter_remaining, T gain, const T* threshold,
                int processing_function_threads,
                const hclean::DivergenceControl<T>& divergence,
                const Controls& controls, const bool* relaxation_active,
                int* iter_out, int* stop_out,
                bool* attempted_out, bool* accepted_out,
                int* component_count_out, double* initial_norm_out,
                double* final_norm_out, double* rms_ratio_out) {
    const int nplanes = nt * nf * np_img;
    const std::size_t plane_size = static_cast<std::size_t>(ny) * nx;
    std::atomic<int> next{0};
    const int workers = std::clamp(processing_function_threads, 1, std::max(1, nplanes));
    auto work = [&]() {
        while (true) {
            const int plane = next.fetch_add(1);
            if (plane >= nplanes) break;
            const int pp = plane % np_img;
            const int tf = plane / np_img;
            const int psf_plane = tf * np_psf + (np_psf == 1 ? 0 : pp);
            T* residual = residual_cube + static_cast<std::size_t>(plane) * plane_size;
            T* model = model_cube + static_cast<std::size_t>(plane) * plane_size;
            const T* psf = psf_cube + static_cast<std::size_t>(psf_plane) * plane_size;
            const bool* mask = domask ? mask_cube + static_cast<std::size_t>(plane) * plane_size : nullptr;
            Controls plane_controls = controls;
            if (relaxation_active != nullptr && relaxation_active[plane])
                plane_controls.adaptive_start = false;
            clean_one_plane(residual, model, psf, mask, nx, ny,
                            xbeg, xend, ybeg, yend, max_iter_remaining[plane],
                            gain, threshold[plane], divergence, plane_controls,
                            iter_out[plane], stop_out[plane], attempted_out[plane],
                            accepted_out[plane], component_count_out[plane],
                            initial_norm_out[plane], final_norm_out[plane],
                            rms_ratio_out[plane]);
        }
    };
    if (workers == 1) {
        work();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (int i = 0; i < workers; ++i) threads.emplace_back(work);
        for (auto& thread : threads) thread.join();
    }
}

template void clean_cube<float>(float*, float*, const float*, int, const bool*,
    int, int, int, int, int, int, int, int, int, int, const int*, float,
    const float*, int, const hclean::DivergenceControl<float>&, const Controls&,
    const bool*,
    int*, int*, bool*, bool*, int*, double*, double*, double*);
template void clean_cube<double>(double*, double*, const double*, int, const bool*,
    int, int, int, int, int, int, int, int, int, int, const int*, double,
    const double*, int, const hclean::DivergenceControl<double>&, const Controls&,
    const bool*,
    int*, int*, bool*, bool*, int*, double*, double*, double*);

}  // namespace relaxed_hogbom
