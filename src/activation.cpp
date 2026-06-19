#include "activation.hpp"
#include <cmath>
#include <algorithm>


Matrix Activations::activate(const Matrix& x, ActivationType type) {
    switch(type) {
        case ActivationType::LINEAR: return x;
        case ActivationType::RELU: return x.apply([](double v) { return std::max(0.0, v); });
        case ActivationType::LEAKY_RELU: {
            constexpr double alpha = 0.01;
            return x.apply([](double v) { return v > 0.0 ? v : alpha * v; });
        }

        case ActivationType::SOFTMAX: {
            size_t rows = x.rows();
            size_t cols = x.cols();
            Matrix result(rows, cols);
            const double* __restrict src = x.data();
            double* __restrict dst = result.data();

            for (size_t j = 0; j < cols; ++j) {
                double max_val = src[j];
                for (size_t i = 1; i < rows; ++i) if (src[i * cols + j] > max_val) max_val = src[i * cols + j];

                double exp_sum = 0.0;
                for (size_t i = 0; i < rows; ++i) {
                    dst[i * cols + j] = std::exp(src[i * cols + j] - max_val);
                    exp_sum += dst[i * cols + j];
                }

                const double inv_sum = 1.0 / exp_sum;
                for (size_t i = 0; i < rows; ++i) dst[i * cols + j] = std::max(dst[i * cols + j] * inv_sum, 1e-12);
            }

            return result;
        }

        default: throw std::runtime_error("Activation type unspecified");
    }
}

Matrix Activations::deriv_activate(const Matrix& x, ActivationType type) {
    switch(type) {
        case ActivationType::RELU: return x.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; });
        case ActivationType::LEAKY_RELU: {
            constexpr double alpha = 0.01;
            return x.apply([](double v) { return v > 0.0 ? 1.0 : alpha; });
        }

        default: throw std::runtime_error("Activation type unspecified");
    }
}

std::string Activations::to_string(ActivationType type) {
    switch(type) {
        case ActivationType::RELU: return "RELU";
        case ActivationType::LEAKY_RELU: return "LEAKY_RELU";
        case ActivationType::SOFTMAX: return "SOFTMAX";
        default: throw std::runtime_error("Activation type unspecified");
    }
}


Matrix ReLU::activate(const Matrix& X) {
    return X.apply([](double v) { return std::max(0.0, v); });
}

Matrix LeakyReLU::activate(const Matrix& X) {
    constexpr double alpha = 0.01;
    return X.apply([](double v) { return v > 0.0 ? v : alpha * v; });
}





Matrix Softmax::activate(const Matrix& X) {
    return X;
}