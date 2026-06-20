#include "loss.hpp"
#include "utils.hpp"


// =======================================
// Mean Squared Error Class Implementation
// =======================================
double MeanSquaredError::compute_loss(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    const size_t n = A.size();
    const double scalar = (0.5 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double base = y[i] - a[i];
        sum += scalar * base * base;
    }

    return sum;
}

Matrix MeanSquaredError::compute_dA(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    Matrix result(A.rows(), A.cols());
    const size_t n = A.size();
    const double scalar = -(1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double* __restrict r = result.data();
    for (size_t i = 0; i < n; ++i) r[i] = scalar * (y[i] - a[i]);
    return result;
}


// ========================================
// Mean Absolute Error Class Implementation
// ========================================
double MeanAbsoluteError::compute_loss(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    const size_t n = A.size();
    const double scalar = (1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += scalar * std::abs(y[i] - a[i]);
    return sum;
}

Matrix MeanAbsoluteError::compute_dA(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    Matrix result(A.rows(), A.cols());
    const size_t n = A.size();
    const double scalar = (1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double* __restrict r = result.data();

    for (size_t i = 0; i < n; ++i) {
        double val = (a[i] - y[i]);
        r[i] = scalar * ((val > 0.0) - (val < 0.0));
    }

    return result;
}


// ==========================
// Huber Class Implementation
// ==========================
double Huber::compute_loss(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    const size_t n = A.size();
    const double scalar = (1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double diff = std::abs(y[i] - a[i]);
        sum += (diff <= delta_) ? 0.5 * scalar * diff * diff : delta_ * scalar * (diff - 0.5 * delta_);
    }

    return sum;
}

Matrix Huber::compute_dA(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    Matrix result(A.rows(), A.cols());
    const size_t n = A.size();
    const double scalar = (1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double* __restrict r = result.data();

    for (size_t i = 0; i < n; ++i) {
        double diff = y[i] - a[i];
        r[i] = (std::abs(diff) <= delta_) ? -scalar * diff : -delta_ * scalar * ((diff > 0.0) - (diff < 0.0));
    }

    return result;
}


// =========================================
// Binary Cross Entropy Class Implementation
// =========================================
double BinaryCrossEntropy::compute_loss(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    ASSERT(A.rows() == 1, "Binary Cross Entropy Loss expecting matrices of size=(1, batch_size)");
    const size_t batch_size = A.cols();
    const double scalar = (1.0 / batch_size);
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double sum = 0.0;
    for (size_t i = 0; i < batch_size; ++i) sum -= scalar * ((y[i] * std::log(a[i])) + ((1.0 - y[i]) * std::log(1.0 - a[i])));
    return sum;
}

Matrix BinaryCrossEntropy::compute_dA(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    ASSERT(A.rows() == 1, "Binary Cross Entropy Loss expecting matrices of size=(1, batch_size)");
    Matrix result(1, A.cols());
    const size_t batch_size = A.cols();
    const double scalar = -(1.0 / batch_size);
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double* __restrict r = result.data();
    for (size_t i = 0; i < batch_size; ++i) r[i] = scalar * ((y[i] / a[i]) - ((1.0 - y[i]) / (1.0 - a[i])));
    return result;
}


// ==============================================
// Categorical Cross Entropy Class Implementation
// ==============================================
double CategoricalCrossEntropy::compute_loss(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    const size_t n = A.size();
    const double scalar = (1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum -= scalar * y[i] * std::log(a[i]);
    return sum;
}

Matrix CategoricalCrossEntropy::compute_dA(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    Matrix result(A.rows(), A.cols());
    const size_t n = A.size();
    const double scalar = -(1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double* __restrict r = result.data();
    for (size_t i = 0; i < n; ++i) r[i] = scalar * (y[i] / a[i]);
    return result;
}


// ====================================================
// Multilabel Binary Cross Entropy Class Implementation
// ====================================================
double MultilabelBinaryCrossEntropy::compute_loss(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    const size_t n = A.size();
    const double scalar = (1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum -= scalar * ((y[i] * std::log(a[i])) + ((1.0 - y[i]) * std::log(1.0 - a[i])));
    return sum;
}

Matrix MultilabelBinaryCrossEntropy::compute_dA(const Matrix& y_true, const Matrix& A) {
    ASSERT(Matrix::matchDim(y_true, A), "Output and labels must match dimensions");
    Matrix result(A.rows(), A.cols());
    const size_t n = A.size();
    const double scalar = -(1.0 / A.cols());
    const double* __restrict y = y_true.data();
    const double* __restrict a = A.data();
    double* __restrict r = result.data();
    for (size_t i = 0; i < n; ++i) r[i] = scalar * ((y[i] / a[i]) - ((1.0 - y[i]) / (1.0 - a[i])));
    return result;
}