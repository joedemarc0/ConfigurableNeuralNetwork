#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <functional>


class Matrix {
    private:
        size_t rows_, cols_;
        std::vector<double> data_;
    
    public:
        // Constructors
        Matrix();
        Matrix(size_t rows, size_t cols);
        Matrix(size_t rows, size_t cols, double value);
        Matrix(Matrix&& other) noexcept;
        Matrix(const Matrix& other) = default;

        // Getters
        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        double* data() { return data_.data(); }
        const double* data() const { return data_.data(); }

        // Assignment Operators
        Matrix& operator=(Matrix&& other) noexcept;
        Matrix& operator=(const Matrix& other) = default;

        // Element Access
        double& operator()(size_t row, size_t col);
        const double& operator()(size_t row, size_t col) const;
        double& at(size_t row, size_t col) { return data_[row * cols_ + col]; }
        const double& at(size_t row, size_t col) const { return data_[row * cols_ + col]; }

        // Function applying
        template <typename Fn>
        Matrix apply(Fn&& func) const {
            Matrix result(rows_, cols_);
            const double* __restrict a = data_.data();
            double* __restrict r = result.data_.data();

            const size_t n = data_.size();
            for (size_t i = 0; i < n; ++i) r[i] = func(a[i]);
            return result;
        }

        // Matrix Operations
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;

        // Scalar Operations
        Matrix operator*(double scalar) const;
        Matrix operator/(double scalar) const;

        // In-place Matrix Operations
        Matrix& operator+=(const Matrix& other);
        Matrix& operator-=(const Matrix& other);

        // In-place Scalar Operations
        Matrix& operator*=(double scalar);
        Matrix& operator/=(double scalar);

        // Boolean Operations
        bool operator==(const Matrix& other) const;
        bool operator!=(const Matrix& other) const;

        // Specialized Operations
        Matrix hadamard(const Matrix& other) const;
        Matrix transpose() const;
        Matrix sumCols() const;

        // Initialization methods
        void randomize(double min=0.0, double max=1.0);
        void xavierInit();
        void heInit();
        void fill(double value);

        // Utility Functions
        static bool matchDim(const Matrix& a, const Matrix& b);
};

// Scalar multiplication is commutative
inline Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

// Stream output operator
inline std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "Matrix(" << matrix.rows() << ", " << matrix.cols() << ")";
    return os;
}


#endif // MATRIX_H