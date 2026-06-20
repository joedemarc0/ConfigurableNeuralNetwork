#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp"

enum class LossFunction {
    MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
    HUBER,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY,
    MULTILABEL_BINARY_CROSS_ENTROPY,
    SPARSE_CATEGORICAL_CROSS_ENTROPY,
    KL_DIVERGENCE,
    HINGE,
    SQUARED_HINGE
};

class Loss {
    public:
        Loss();

        virtual double compute_loss(const Matrix& y_true, const Matrix& A) = 0;
        virtual Matrix compute_dA(const Matrix& y_true, const Matrix& A) = 0;

        virtual LossFunction type() const = 0;
}; // Loss Class


class MeanSquaredError : public Loss {
    public:
        double compute_loss(const Matrix& y_true, const Matrix& A) override;
        Matrix compute_dA(const Matrix& y_true, const Matrix& A) override;

        LossFunction type() const override { return LossFunction::MEAN_SQUARED_ERROR; }
}; // MeanSquaredError Class


class MeanAbsoluteError : public Loss {
    public:
        double compute_loss(const Matrix& y_true, const Matrix& A) override;
        Matrix compute_dA(const Matrix& y_true, const Matrix& A) override;

        LossFunction type() const override { return LossFunction::MEAN_ABSOLUTE_ERROR; }
}; // MeanAbsoluteError Class


class Huber : public Loss {
    protected:
        double delta_;

    public:
        Huber(double delta);

        double compute_loss(const Matrix& y_true, const Matrix& A) override;
        Matrix compute_dA(const Matrix& y_true, const Matrix& A) override;

        LossFunction type() const override { return LossFunction::HUBER; }
}; // Huber Class


class BinaryCrossEntropy : public Loss {
    public:
        double compute_loss(const Matrix& y_true, const Matrix& A) override;
        Matrix compute_dA(const Matrix& y_true, const Matrix& A) override;

        LossFunction type() const override { return LossFunction::BINARY_CROSS_ENTROPY; }
}; // BinaryCrossEntropy Class


class CategoricalCrossEntropy : public Loss {
    public:
        double compute_loss(const Matrix& y_true, const Matrix& A) override;
        Matrix compute_dA(const Matrix& y_true, const Matrix& A) override;

        LossFunction type() const override { return LossFunction::CATEGORICAL_CROSS_ENTROPY; }
}; // CategoricalCrossEntropy Class


class MultilabelBinaryCrossEntropy : public Loss {
    public:
        double compute_loss(const Matrix& y_true, const Matrix& A) override;
        Matrix compute_dA(const Matrix& y_true, const Matrix& A) override;

        LossFunction type() const override { return LossFunction::MULTILABEL_BINARY_CROSS_ENTROPY; }
}; // MultilabelBinaryCrossEntropy Class


#endif // LOSS_HPP