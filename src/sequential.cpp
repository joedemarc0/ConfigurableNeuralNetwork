#include "sequential.hpp"
#include "utils.h"


// ============================
// Sequential Class Constructor
// ============================
Sequential::Sequential() : isCompiled(false) {}


// =================================
// Sequential Class Public Functions
// =================================


// =================================
// Sequential Class Public Functions
// =================================
void Sequential::addLayer(std::unique_ptr<Layer> layer) {
    ASSERT(!isCompiled, "Cannot add layer after compiling network");
    ASSERT(layer, "Null layer passed");
    if (!layerConfig.empty()) ASSERT(layer->type() != LayerType::Input, "Only the first layer can be an Input layer");

    layerConfig.push_back(std::move(layer));
}


void Sequential::compile() {
    if (isCompiled) return;
    ASSERT(!layerConfig.empty(), "Cannot compile network with 0 layers");

    Layer* prev_layer = nullptr;
    size_t layer_num = 0;

    for (auto it = layerConfig.begin(); it != layerConfig.end(); ++it) {
        Layer* curr = it.operator->();

        if (layer_num == 0) {
            ASSERT(curr->type() == LayerType::Input, "First layer must be Input layer for now");
            prev_layer = curr;
            ++layer_num;
            continue;
        }

        size_t expected_input_size = prev_layer->getOutputSize();
        if (curr->isBuilt()) {
            ASSERT(curr->getInputSize() == expected_input_size, "Dimension mismatch between layers");
        } else {
            curr->build(expected_input_size);
        }

        prev_layer = curr;
        ++layer_num;
    }

    isCompiled = true;
}