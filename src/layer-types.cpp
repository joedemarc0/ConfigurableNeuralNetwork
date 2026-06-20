#include "layer-types.hpp"


// ================================
// Layers and their Implementations
// Input Layer Implementation
// ==========================

Input::Input(size_t input_size)
    : Layer()
{
    ASSERT(input_size != 0, "Cannot create Input with size = 0");
    inputSize_ = input_size;
}

Matrix Input::forward(const Matrix& X) { return X; }
Matrix Input::backward(const Matrix& dA) { return dA; }
void Input::build() {
    ASSERT(hasInputSize(), "Input built without input size");
    outputSize_ = inputSize_;
    built = true;
}


// ===========================
// Dense Layer Implementations
// ===========================

// Constructors
Dense::Dense(size_t output_size)
    : Layer(), activation(std::make_unique<Linear>()), initType(InitType::NONE)
{
    ASSERT(output_size != 0, "Cannot create Dense with output size = 0");
    outputSize_ = output_size;
}

Dense::Dense(
    size_t output_size,
    std::unique_ptr<Activation> act,
    InitType init_type
) : Layer(),
    activation(std::move(act)),
    initType(init_type)
{
    ASSERT(output_size != 0, "Cannot create Dense with output size = 0");
    outputSize_ = output_size;
}

Dense::Dense(
    size_t input_size,
    size_t output_size,
    std::unique_ptr<Activation> act,
    InitType init_type
) : Layer(input_size, output_size),
    activation(std::move(act)),
    initType(init_type)
{
    ASSERT(input_size != 0, "Cannot create Dense with size = 0");
    ASSERT(output_size != 0, "Cannot create Dense with output size = 0");
}


// Dense Layer Private Functions
void Dense::initialize() {
    switch(initType) {
        case InitType::RANDOM: { weights.randomize(); break; }
        case InitType::XAVIER: { weights.xavierInit(); break; }
        case InitType::HE: { weights.heInit(); break; }
        case InitType::NONE: { weights.fill(1.0); break; }
        default: throw std::runtime_error("Layer init type unspecified");
    }
}

void Dense::updateParams(double learning_rate) {
    weights.updateScaled(dWeights, -learning_rate);
    biases.updateScaled(dbiases, -learning_rate);
}


// Dense Layer Public Functions
Matrix Dense::forward(const Matrix& X) {
    ASSERT(X.rows() == *inputSize_, "Forwarding matrix has incorrect size");

    input = X;
    Matrix Z = (weights * X) + biases;
    preActivation = Z;
    output = activation->forward(Z);
    return output;
}

Matrix Dense::backward(const Matrix& dA) {
    Matrix dZ;

    if (auto* t = dynamic_cast<DiagonalJacobian*>(activation.get())) {
        dZ = dA.hadamard(t->deriv_activate(preActivation));
    } else if (auto* t = dynamic_cast<NonDiagonalJacobian*>(activation.get())) {
        dZ = t->jacobian_transpose(output) * dA;
    }

    dWeights = dZ * input.transpose();
    dbiases = dZ.sumCols();

    return weights.transpose() * dZ;
}

void Dense::build() {
    ASSERT(hasInputSize(), "Dense built without input size");
    weights = Matrix(outputSize_, inputSize_);
    biases = Matrix(outputSize_, 1);
    initialize();
    built = true;
}


// ============================
// Dropout Layer Implementation
// ============================

// Constructor
Dropout::Dropout(double dropout_rate) : Layer(), rate(dropout_rate) {}

Matrix Dropout::forward(const Matrix& X) {
    //if (!training) return X;
    // Need to find a way to implement this part^^^^


    mask = Matrix::mask(X.rows(), X.cols(), rate);
    return X.hadamard(mask) / (1.0 - rate);
}

Matrix Dropout::backward(const Matrix& dA) {
    return dA.hadamard(mask) / (1.0 - rate);
}

void Dropout::build() {
    ASSERT(hasInputSize(), "Dropout built without input size");
    outputSize_ = inputSize_;
    built = true;
}