#include "sequential.hpp"
#include "utils.h"


// =================================
// Sequential Class Public Functions
// =================================

void Sequential::compile() {
    if (state != CompiledState::NONE) return;
    ASSERT(!layerConfig.empty(), "Cannot compile Sequential with zero layers");
    layerConfig.compile();
    state = CompiledState::SEMICOMPILED;
}


// ==================================
// Sequential Class Private Functions
// ==================================

Matrix Sequential::forward(const Matrix& X) {
    switch(state) {
        case CompiledState::NONE: {
            compile();
            return forward(X);
        }

        case CompiledState::SEMICOMPILED: {
            layerConfig.input()->setInputSize(X.rows());
            layerConfig.forEach(layerConfig.input(), ++layerConfig.begin(), [&](LayerConfig::Iterator it) {
                layerConfig.buildLayer(it);
            });

            state = CompiledState::COMPILED;
            break;
        }

        case CompiledState::COMPILED: { break; }
    }

    lastOutput = layerConfig.forward(X);
    return lastOutput;
}

void Sequential::backward(const Matrix& y_true, double learning_rate) {
    // size_t batch_size = y_true.cols();
    Matrix dA = lastOutput - y_true;
    layerConfig.forEachBackwards(layerConfig.end(), layerConfig.input(), [&](LayerConfig::Iterator it) {
        dA = it->backward(dA);
        if (auto* t = dynamic_cast<Trainable*>(it.operator->())) t->update(learning_rate);
    });
}