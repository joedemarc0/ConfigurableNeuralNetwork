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
        if (layer_num == 0 && layer->type() != LayerType::Input) {
            throw std::runtime_error("Error: network cannot be compiled before forward passing if there is no Input Layer");
        }

        size_t input_size = layer->getInputSize();
        size_t expected_input_size = layers[layer_num - 1]->getOutputSize();
        if (layer->isBuilt()) { ASSERT(input_size == expected_input_size, "Dimension mismatch between layers"); }
        else { layer->build(expected_input_size); }

        layer_num++;
    }

    isCompiled = true;
}