#include "sequential.hpp"
#include "utils.h"


// ============================
// Sequential Class Constructor
// ============================
Sequential::Sequential() : isCompiled(false) {}


// ==================================
// Sequential Class Private Functions
// ==================================
Matrix Sequential::forward(const Matrix& X) {
    if (waitToCompile) {
        layerConfig.insert(layerConfig.begin(), Input(X.rows()));
        compile();
    }

    ASSERT(isCompiled, "Network must be compiled");
}


// =================================
// Sequential Class Public Functions
// =================================
/**
void Sequential::addLayer(std::unique_ptr<Layer> layer) {
    ASSERT(!isCompiled, "Cannot add layer after compiling network");
    ASSERT(layer, "Null layer passed");
    if (!layerConfig.empty()) ASSERT(layer->type() != LayerType::Input, "Only the first layer can be an Input layer");

    layerConfig.push_back(std::move(layer));
}
*/

void Sequential::compile() {
    if (isCompiled) return;
    ASSERT(!layerConfig.empty(), "Cannot compile network with 0 layers");

    Layer* prev_layer = nullptr;
    for (auto it = layerConfig.begin(); it != layerConfig.end(); ++it) {
        Layer* curr = it.operator->();

        if (it == layerConfig.begin()) {
            if (curr->type() == LayerType::Input) {
                curr->build(247);
                prev_layer = curr;
                continue;
            } else if (curr->getInputSize() == 0) {
                waitToCompile = true;
                continue;
            } else {
                layerConfig.push_front(Input(curr->getInputSize()));
                compile();
            }
        }

        ASSERT(curr->type() != LayerType::Input, "Input layer must be the first layer");
        size_t expected_input_size = prev_layer->getOutputSize();
        curr->build(expected_input_size);
        prev_layer = curr;
    }

    isCompiled = true;
}