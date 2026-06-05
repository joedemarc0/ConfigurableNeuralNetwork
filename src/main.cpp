#include <iostream>
#include <random>

#include "layer-config.hpp"


int main() {
    Dense dense1(128, Activations::ActivationType::RELU, InitType::HE);
    Dense dense2(128, Activations::ActivationType::RELU, InitType::HE);
    Dense dense3(128, Activations::ActivationType::RELU, InitType::HE);
    Dropout dropout(0.7);
    dense1.setInputSize(482);
    dense3.setInputSize(784);
    dense3.build();
    std::cout << dense1 << std::endl;
    std::cout << dense2 << std::endl;
    std::cout << dense3 << std::endl;
    std::cout << dropout << std::endl;
    return 0;
}