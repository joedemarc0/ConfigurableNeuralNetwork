#include "layer.hpp"
#include "utils.h"


// ===========================
// Input Layer Implementations
// ===========================
Input::Input(size_t input_size) : inputSize(input_size) {}
Matrix Input::forward(const Matrix& X) { return X; }
Matrix Input::backward(const Matrix& dA) { return dA; }
void Input::build(size_t input_size) { 
    inputSize = outputSize = input_size;
    built = true;
}


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
    build();
}

// =============================
// Dense Layer Private Functions
// =============================
void Dense::initialize() {
    switch(initType) {
        case InitType::RANDOM: { weights.randomize(); break; }
        case InitType::XAVIER: { weights.xavierInit(); break; }
        case InitType::HE: { weights.heInit(); break; }
        case InitType::NONE: { break; }
        default: throw std::runtime_error("Layer init type unspecified");
    }
}

void Dense::updateParams(double learning_rate) {
    weights -= dWeights * learning_rate;
    biases -= dbiases * learning_rate;
}

void Dense::build() {
    weights = Matrix(outputSize, inputSize);
    biases = Matrix(outputSize, 1);

    initialize();
    built = true;
}


// ============================
// Dense Layer Public Functions
// ============================
void Dense::build(size_t input_size) {
    if (built) return;

    inputSize = input_size;
    weights = Matrix(outputSize, inputSize);
    biases = Matrix(outputSize, 1);

    initialize();
    built = true;
}

Matrix Dense::forward(const Matrix& X) {
    ASSERT(X.rows() == inputSize, "Forwarding matrix of incorrect size");

    input = X;
    Matrix Z = (weights * X) + biases;
    preActivation = Z;
    Matrix A = Activations::activate(Z, actType);
    return A;
}

Matrix Dense::backward(const Matrix& dA) {
    size_t batch_size = dA.cols();
    Matrix dZ;

    if (actType == Activations::ActivationType::SOFTMAX) {
        dZ = dA;
    } else {
        Matrix sigma_prime = Activations::deriv_activate(preActivation, actType);
        dZ = dA.hadamard(sigma_prime);
    }

    dWeights = dZ * input.transpose();
    dWeights /= batch_size;

    dbiases = dZ.sumCols();
    dbiases /= batch_size;

    return weights.transpose() * dZ;
}