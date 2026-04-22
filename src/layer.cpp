#include "layer.hpp"


// ===========================
// Input Layer Implementations
// ===========================
Input::Input(size_t input_size) : inputSize(input_size) {}

Matrix Input::forward(const Matrix& X) {
    return X;
}

Matrix Input::backward(const Matrix& dZ) {
    return dZ;
}


// ========================
// Dense Layer Constructors
// ========================
Dense::Dense(
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type
) : outputSize(output_size),
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
{}

// ============================
// Dense Layer Public Functions
// ============================
Matrix Dense::forward(const Matrix& X) {
    return X;
}