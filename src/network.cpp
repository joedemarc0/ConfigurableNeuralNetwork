#include "network.h"
#include "utils.h"


// ====================
// Network Constructors
// ====================
Network::Network(
    size_t input_size,
    size_t num_classes
) : networkInputSize(input_size),
    numClasses(num_classes)
{}



// ===============================
// Nested Layer Class Constructors
// ===============================
Network::Layer::Layer(
    size_t input_size,
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type
) : inputSize(input_size),
    outputSize(output_size),
    actType(act_type),
    initType(init_type)
{}



// ====================================
// Nested Layer Class Private Functions
// ====================================
void Network::Layer::initialize() {
    switch(initType) {
        case InitType::RANDOM: { weights.randomize(); break; }
        case InitType::XAVIER: { weights.xavierInit(); break; }
        case InitType::HE: { weights.heInit(); break; }
        case InitType::NONE: { weights.fill(0.0); break; }
    }
}
// NEED TO IMPLEMENT INITIALIZATION METHODS^^^^^^^^^

void Network::Layer::updateParams(const Matrix& dWeights, const Matrix& dbiases, double learning_rate) {
    weights -= dWeights * learning_rate;
    biases -= dbiases * learning_rate;
}


// ===================================
// Nested Layer Class Public Functions
// ===================================
Matrix Network::Layer::forward(const Matrix& X) {
    ASSERT(X.rows() == inputSize, "Forwarding matrix has invalid dimensions");

    input = X;
    Matrix Z = (weights * X) + biases;
    preActivation = Z;
    Matrix A = Activations::activate(Z, actType);
    return A;
}

Matrix Network::Layer::backward(const Matrix& dA, size_t batch_size, double learning_rate) {
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

// ===============================
// Network Class Private Functions
// ===============================
void Network::addOutputLayer() {
    ASSERT(!layers.empty(), "Network must have at least one hidden layer");
    size_t input_dim = layers.back().getOutputSize();
    size_t output_dim = numClasses;
    layers.emplace_back(input_dim, output_dim, Activations::ActivationType::SOFTMAX, InitType::XAVIER);
}

Matrix Network::forward(const Matrix& X) {
    ASSERT(isCompiled, "Network is not compiled");
    ASSERT(X.rows() == networkInputSize, "Forwarding matrix has invalid dimensions");

    Matrix A = X;
    for (auto& layer : layers) A = layer.forward(A);
    lastOutput = A;
    return lastOutput;
}

void Network::backward(const Matrix& y_true, double learning_rate) {
    size_t batch_size = y_true.cols();
    Matrix dA = lastOutput - y_true;
    for (size_t i = layers.size(); i-- > 0; ) {
        dA = layers[i].backward(dA, batch_size, learning_rate);
    }
}


// ==============================
// Network Class Public Functions
// ==============================
void Network::addLayer(size_t neurons, Activations::ActivationType act_type, InitType init_type) {
    ASSERT(neurons != 0, "Cannot add layer with 0 neurons");
    ASSERT(!isCompiled, "Network is already compiled");

    size_t input_dim = layers.empty() ? networkInputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, act_type, init_type);
}

void Network::compile() {
    ASSERT(!isCompiled, "Network is already compiled");
    ASSERT(!layers.empty(), "Network cannot be compiled with zero hidden layers");

    size_t layer_idx = 0;
    size_t expected_input_size = networkInputSize;
    for (auto& layer : layers) {
        size_t input_size = layer.getInputSize();
        auto act = layer.getActivationType();

        if (act == Activations::ActivationType::SOFTMAX) {
            throw std::runtime_error(
                "Invalid activation function type at layer " + std::to_string(layer_idx) +
                ", activation function: " + Activations::to_string(act)
            );
        }

        if (input_size != expected_input_size) {
            throw std::runtime_error(
                "Dimension mismatch at layer " + std::to_string(layer_idx) +
                ", expected: " + std::to_string(expected_input_size) + 
                ", got: " + std::to_string(input_size)
            );
        }

        expected_input_size = layer.getOutputSize();
        layer_idx++;
    }

    addOutputLayer();
    isCompiled = true;
}