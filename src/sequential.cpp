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

    Matrix A = X;
    layerConfig.forEachFromInput([&](LayerConfig::Iterator it) { A = it->forward(A); });
    lastOutput = A;
    return lastOutput;
}
