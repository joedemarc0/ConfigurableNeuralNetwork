#include "sequential.hpp"
#include "utils.hpp"


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

void Sequential::backward(const Matrix& y_true, double learning_rate) {
    // I know this computes dZ and that it's wrong because I'm supposed to pass in dA, but in my previous design I was supposed to pass in dZ, so it DID work
    Matrix dA = lastOutput - y_true;

    // Some line like dA = Loss::compute_dA(lastOutput, y_true) probably
    // Haven't implemented Loss though yet
    layerConfig.forEachBackwards(layerConfig.end(), layerConfig.input(), [&](LayerConfig::Iterator it) {
        dA = it->backward(dA);
        if (auto* t = dynamic_cast<Trainable*>(it.operator->())) t->update(learning_rate);
    });
}