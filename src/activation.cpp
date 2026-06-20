#include "activation.hpp"
#include <cmath>
#include <algorithm>


// =========================
// ReLU Class Implementation
// =========================
Matrix ReLU::forward(const Matrix& X) {
    return X.apply([](double v) { return std::max(0.0, v); });
}

Matrix ReLU::deriv_activate(const Matrix& X) {
    return X.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; });
}


// ==============================
// LeakyReLU Class Implementation
// ==============================
Matrix LeakyReLU::forward(const Matrix& X) {
    return X.apply([&](double v) { return v > 0.0 ? v : alpha * v; });
}

Matrix LeakyReLU::deriv_activate(const Matrix& X) {
    return X.apply([&](double v) { return v > 0.0 ? 1.0 : alpha; });
}


// ============================
// Softmax Class Implementation
// ============================
Matrix Softmax::forward(const Matrix& X) {
    const size_t rows = X.rows();
    const size_t cols = X.cols();
    Matrix result(rows, cols);

    const double* __restrict src = X.data();
    double* __restrict dst = result.data();

    for (size_t j = 0; j < cols; ++j) {
        double max_val = src[j];
        for (size_t i = 1; i < rows; ++i) {
            double val = src[i * cols + j];
            if (val > max_val) max_val = val;
        }

        double sum_exp = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            size_t idx = i * cols + j;
            dst[idx] = std::exp(src[idx] - max_val);
            sum_exp += dst[idx];
        }

        const double inv_sum = 1.0 / sum_exp;
        for (size_t i = 0; i < rows; ++i) {
            size_t idx = i * cols + j;
            dst[idx] = std::max(dst[idx] * inv_sum, 1e-12);
        }
    }

    return result;
}