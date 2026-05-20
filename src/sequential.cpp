#include "sequential.hpp"
#include "utils.h"


// =================================
// Sequential Class Public Functions
// =================================

void Sequential::compile() {
    if (state != CompiledState::NONE) return;
    ASSERT(!layerConfig.empty(), "Cannot compile Sequential with zero layers");

    Layer* prev_layer = nullptr;
    size_t expected_input_size = 0;
    layerConfig.forEachLayer([&](LayerConfig::Iterator it) {
        Layer* curr = it.operator->();
        ASSERT(it->type() != LayerType::Input, "Input layer found after head node");

        if (it == layerConfig.begin()) {
            prev_layer = curr;
            return;
        }

        expected_input_size = prev_layer->getOutputSize();
        curr->build(expected_input_size);
        prev_layer = curr;
    });

    state = CompiledState::SEMICOMPILED;
}


// ==================================
// Sequential Class Private Functions
// ==================================

Matrix Sequential::forward(const Matrix& X) {
    switch(state) {
        case CompiledState::NONE: {
            compile();
            forward(X);
            break;
        }

        case CompiledState::SEMICOMPILED: {
            layerConfig.input()->build(X.rows());
            layerConfig.begin()->build(layerConfig.input()->getOutputSize());
            state = CompiledState::COMPILED;
            break;
        }

        case CompiledState::COMPILED: { break; }
    }

    Matrix A = X;
    layerConfig.forEachFromInput([&](LayerConfig::Iterator it) { A = it->forward(A); });
    lastOutput = A;
    return lastOutput;
}
