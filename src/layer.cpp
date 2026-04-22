#include "layer.hpp"
#include "utils.h"


// ===========================
// Input Layer Implementations
// ===========================
Input::Input(size_t input_size) : inputSize(input_size) {}
Matrix Input::forward(const Matrix& X) { return X; }
Matrix Input::backward(const Matrix& dA) { return dA; }


// ========================
// Dense Layer Constructors
// ========================
Dense::Dense(
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type
) : inputSize(0),
    outputSize(output_size),
    actType(act_type),
    initType(init_type)
{}

Dense::Dense(
    size_t input_size,
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type
) : inputSize(input_size),
    outputSize(output_size),
    actType(act_type),
    initType(init_type)
{
    ASSERT(input_size != 0, "Input size cannot be 0");
}

// =============================
// Dense Layer Private Functions
// =============================
void Dense::build(size_t input_size) {}


// ============================
// Dense Layer Public Functions
// ============================
Matrix Dense::forward(const Matrix& X) {
    ASSERT(X.rows() == inputSize, "Forwarding matrix of incorrect size");

    input = X;
    Matrix Z = (weights * X) + biases;
    preActivation = Z;
    Matrix A = Activations::activate(Z, actType);
    return A;
}

Matrix Dense::backward(const Matrix& dA, size_t batch_size, double learning_rate) {
    Matrix dZ;
    if (actType == Activations::ActivationType::SOFTMAX) {
        dZ = dA;
    } else {
        Matrix sigma_prime = Activations::deriv_activate(preActivation, actType);
        dZ = dA.hadamard(sigma_prime);
    }

    Matrix dWeights = dZ * input.transpose();
    Matrix dbiases = dZ.sumCols();
    dWeights /= batch_size;
    dbiases /= batch_size;

    Matrix dA_out = weights.transpose() * dZ;
    updateParams(dWeights, dbiases, learning_rate);

    return dA_out;
}