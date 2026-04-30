#include "matrix.h"
#include "utils.h"
#include <stdexcept>
#include <random>


// Matrix Class Constructors
Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

Matrix::Matrix(size_t rows, size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.rows_ = other.cols_ = 0;
    other.data_.clear();
}


// Assignment Operators
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = std::move(other.data_);
        other.rows_ = other.cols_ = 0;
        other.data_.clear();
    }

    return *this;
}


// Element Access
double& Matrix::operator()(size_t row, size_t col) {
    ASSERT(row < rows_ && col < cols_, "Index out of range");
    return data_[row * cols_ + col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    ASSERT(row < rows_ && col < cols_, "Index out of range");
    return data_[row * cols_ + col];
}


// Matrix Operations
Matrix Matrix::operator+(const Matrix& other) const {
    const double* __restrict a = data_.data();
    const double* __restrict b = other.data_.data();
    const size_t n = data_.size();

    if (matchDim(*this, other)) {
        Matrix result(rows_, cols_);
        double* __restrict r = result.data_.data();
        for (size_t i = 0; i < n; ++i) r[i] = a[i] + b[i];
        return result;

    } else if (rows_ == other.rows_ && other.cols_ == 1) {
        Matrix result(rows_, cols_);
        double* __restrict r = result.data_.data();
        for (size_t row = 0; row < rows_; ++row) {
            const double bias = b[row];
            const size_t base = row * cols_;
            for (size_t col = 0; col < cols_; ++col) r[base + col] = a[base + col] + bias;
        }

        return result;
    }

    throw std::invalid_argument(__PRETTY_FUNCTION__ + std::string(": Matrix dimensions must be valid for addition"));
}
// CHANGE TO INCLUDE BROADCASTING^^^^^^^^^^

Matrix Matrix::operator-(const Matrix& other) const {
    ASSERT(matchDim(*this, other), "Matrix dimensions must match");
    Matrix result(rows_, cols_);
    const size_t n = data_.size();
    const double* __restrict a = data_.data();
    const double* __restrict b = other.data_.data();
    double* __restrict r = result.data_.data();
    for (size_t i = 0; i < n; ++i) r[i] = a[i] - b[i];
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    ASSERT(cols_ == other.rows_, "Matrix dimensions must be valid for multiplication");
    Matrix result(rows_, other.cols_);
    const double* __restrict a = data_.data();
    const double* __restrict b = other.data_.data();
    double* __restrict r = result.data_.data();
    
    for (size_t i = 0; i < rows_; ++i) {
        const double* a_row = a + i * cols_;
        double* r_row = r + i * other.cols_;

        for (size_t k = 0; k < cols_; ++k) {
            const double a_ik = a_row[k];
            const double* b_row = b + k * other.cols_;
            for (size_t j = 0; j < other.cols_; ++j) r_row[j] += a_ik * b_row[j];
        }
    }

    return result;
}


// Scalar Operations
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    const size_t n = data_.size();
    const double* __restrict a = data_.data();
    double* __restrict r = result.data_.data();
    for (size_t i = 0; i < n; ++i) r[i] = a[i] * scalar;
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    ASSERT(std::abs(scalar) > 1e-12, "Divide by zero error");
    return *this * (1.0 / scalar);
}


// In-place Matrix Operations
Matrix& Matrix::operator+=(const Matrix& other) {
    ASSERT(matchDim(*this, other), "Matrix dimensions must match");
    const size_t n = data_.size();
    double* __restrict a = data_.data();
    const double* __restrict b = other.data_.data();
    for (size_t i = 0; i < n; ++i) a[i] += b[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    ASSERT(matchDim(*this, other), "Matrix dimensions must match");
    const size_t n = data_.size();
    double* __restrict a = data_.data();
    const double* __restrict b = other.data_.data();
    for (size_t i = 0; i < n; ++i) a[i] -= b[i];
    return *this;
}

void Matrix::updateScaled(const Matrix& other, double scalar) {
    ASSERT(matchDim(*this, other), "Matrix dimensions must match");
    const size_t n = data_.size();
    double* __restrict a = data_.data();
    const double* __restrict b = other.data_.data();
    for (size_t i = 0; i < n; ++i) a[i] += b[i] * scalar;
}


// In-place Scalar Operations
Matrix& Matrix::operator*=(double scalar) {
    for (double &v : data_) v *= scalar;
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    ASSERT(std::abs(scalar) > 1e-12, "Divide by zero error");
    const double inv = 1.0 / scalar;
    for (double &v : data_) v *= inv;
    return *this;
}


// Boolean Operations
bool Matrix::operator==(const Matrix& other) const {
    if (!matchDim(*this, other)) return false;
    const size_t n = data_.size();
    const double* __restrict a = data_.data();
    const double* __restrict b = other.data_.data();
    for (size_t i = 0; i < n; ++i) if (std::abs(a[i] - b[i]) > 1e-9) return false;
    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}


// Specialized Operations
Matrix Matrix::hadamard(const Matrix& other) const {
    ASSERT(matchDim(*this, other), "Matrix dimensions must match");
    Matrix result(rows_, cols_);
    const size_t n = data_.size();
    const double* __restrict a = data_.data();
    const double* __restrict b = other.data_.data();
    double* __restrict r = result.data_.data();
    for (size_t i = 0; i < n; ++i) r[i] = a[i] * b[i];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(rows_, cols_);
    const double* __restrict a = data_.data();
    double* __restrict r = result.data_.data();

    constexpr size_t BLOCK = 32;
    for (size_t i = 0; i < rows_; i += BLOCK) {
        for (size_t j = 0; j < cols_; j += BLOCK) {
            for (size_t ii = i; ii < std::min(i + BLOCK, rows_); ++ii) {
                for (size_t jj = j; j < std::min(j + BLOCK, cols_); ++jj) {
                    r[jj * rows_ + ii] = a[ii * cols_ + jj];
                }
            }
        }
    }

    return result;
}

Matrix Matrix::sumCols() const {
    Matrix result(rows_, 1);
    const double* __restrict a = data_.data();
    double* __restrict r = result.data_.data();
    for (size_t row = 0; row < rows_; ++row) {
        const double* row_ptr = a + row * cols_;
        for (size_t col = 0; col < cols_; ++col) r[row] += row_ptr[col];
    }

    return result;
}


// Initialization Methods
void Matrix::randomize(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dis(min, max);
    for (double& v : data_) v = dis(rng);
}

void Matrix::xavierInit() {
    double limit = std::sqrt(6.0 / (rows_ + cols_));
    randomize(-limit, limit);
}

void Matrix::heInit() {
    static std::mt19937 rng(std::random_device{}());
    double stddev = std::sqrt(2.0 / cols_);
    std::normal_distribution<double> dis(0.0, stddev);
    for (double& v : data_) v = dis(rng);
}

void Matrix::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}


// Utility Functions
bool Matrix::matchDim(const Matrix& a, const Matrix& b) {
    return (a.rows_ == b.rows_ && a.cols_ == b.cols_);
}