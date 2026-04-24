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
    if (!layers.empty()) ASSERT(layer->type() != LayerType::Input, "Only the first layer can be an Input layer");

    layers.push_back(std::move(layer));
}

void Sequential::compile() {
    if (isCompiled) return;
    ASSERT(!layers.empty(), "Cannot compile network with 0 layers");

    size_t layer_num = 0;
    for (auto& layer : layers) {
        if (layer_num == 0) {
            ASSERT(layer->type() != LayerType::Input, "Cannot compile network through compile() with no Input layer");
            layer_num++;
            continue;
        }

        size_t input_size = layer->getInputSize();
        size_t expected_input_size = layers[layer_num - 1]->getOutputSize();
        if (layer->isBuilt()) { ASSERT(input_size == expected_input_size, "Dimensions mismatch between layers"); }
        else { layer->build(expected_input_size); }
    }

    isCompiled = true;
}