/**
 * File to contain all of the activation functions and their derivatives
 * 
 * Option to use SIGMOID, RELU, LEAKY RELU, and SOFTMAX Activation Functions - Option to use NONE as well
 * Layer and Network Classes will allow for use of any activation function when creating those objects
 * Keep in mind that not all of these functions will be used however the option to use another function will be
 * I will probably implement ReLU functions on the two hidden layers and softmax on the output since I achieved 97% accuracy 
 * on test sets with this method (784 -> 128 -> 64 -> 10)
 */

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "matrix.hpp"

enum class ActivationType { LINEAR, RELU, LEAKY_RELU, SOFTMAX };
enum class JacobianType { DIAGONAL, NONDIAGONAL };

class Activation {
    protected:
    public:
        virtual Matrix forward(const Matrix& X) = 0;
        virtual ActivationType type() const = 0;
        virtual JacobianType jacobianType() const = 0;
}; // Activation Class


class DiagonalJacobian {
    public:
        virtual Matrix deriv_activate(const Matrix& X) = 0;
}; // DiagonalJacobian Class


class NonDiagonalJacobian {
    public:
        virtual Matrix jacobian_transpose(const Matrix& X) = 0;
}; // NonDiagonalJacobian Class


class Linear : public Activation, public DiagonalJacobian {
    public:
        Linear();
        Matrix forward(const Matrix& X) override { return X; }
        Matrix deriv_activate(const Matrix& X) override;

        ActivationType type() const override { return ActivationType::LINEAR; }
        virtual JacobianType jacobianType() const override { return JacobianType::DIAGONAL; }
}; // Linear Activation Class


class ReLU : public Activation, public DiagonalJacobian {
    public:
        ReLU();
        Matrix forward(const Matrix& X) override;
        Matrix deriv_activate(const Matrix& X) override;

        ActivationType type() const override { return ActivationType::RELU; }
        virtual JacobianType jacobianType() const override { return JacobianType::DIAGONAL; }
}; // ReLU Activation Class


class LeakyReLU : public Activation, public DiagonalJacobian {
    protected:
        const double alpha;

    public:
        LeakyReLU();
        LeakyReLU(double alpha);
        
        Matrix forward(const Matrix& X) override;
        Matrix deriv_activate(const Matrix& X) override;

        ActivationType type() const override { return ActivationType::LEAKY_RELU; }
        virtual JacobianType jacobianType() const override { return JacobianType::DIAGONAL; }
}; // LeakyReLU Activation Class


class Softmax : public Activation, public NonDiagonalJacobian {
    public:
        Softmax();
        Matrix forward(const Matrix& X) override;

        ActivationType type() const override { return ActivationType::SOFTMAX; }
        virtual JacobianType jacobianType() const override { return JacobianType::NONDIAGONAL; }
}; // Softmax Activation Class



#endif // ACTIVATION_HPP