#include "sequential.hpp"
#include "utils.h"


// =================================
// Sequential Class Public Functions
// =================================

void Sequential::compile() {
    if (semiCompiled || fullyCompiled) return;
    ASSERT(!layerConfig.empty(), "Cannot compile Sequential with zero layers");

    Layer* prev_layer = nullptr;
    size_t expected_input_size = 0;
    for (auto it = layerConfig.begin(); it != layerConfig.end(); ++it) {
        Layer* curr = it.operator->();
        ASSERT(curr->type() != LayerType::Input, "Input layer found after head node");

        if (it == layerConfig.begin()) {
            prev_layer = curr;
            continue;
        }

        expected_input_size = prev_layer->getOutputSize();
        curr->build(expected_input_size);
        prev_layer = curr;
    }

    semiCompiled = true;
}


// ==================================
// Sequential Class Private Functions
// ==================================

Matrix Sequential::forward(const Matrix& X) {
    if (!fullyCompiled) {
        if (!semiCompiled) compile();
        layerConfig.input()->build(X.rows());
        layerConfig.begin()->build(layerConfig.input()->getOutputSize());
        fullyCompiled = true;
    }

    // Forwarding math and stuff
}