#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "layer-config.hpp"

enum class CompiledState { NONE, SEMICOMPILED, COMPILED };

class Sequential {
    public:
        Sequential();

        template <
            typename First,
            typename... Rest,
            typename = std::enable_if_t<std::is_base_of_v<Layer, std::decay_t<First>>>
        >
        Sequential(First&& first, Rest&&... rest);

        void compile();

        CompiledState getState() const { return state; }
        const LayerConfig& getLayerConfig() const { return layerConfig; }
    
    private:
        friend class SequentialAccessor;
        CompiledState state = CompiledState::NONE;
        LayerConfig layerConfig;
        Matrix lastOutput;

        template <
            typename First,
            typename... Rest,
            typename = std::enable_if_t<std::is_base_of_v<Layer, std::decay_t<First>>>
        >
        void addLayer(First&& first, Rest&&... rest);

        Matrix forward(const Matrix& X);
        void backward(const Matrix& y_true, double learning_rate);
};


template <typename First, typename... Rest, typename>
Sequential::Sequential(First&& first, Rest&&... rest) {
    static_assert(std::is_base_of_v<Layer, std::decay_t<First>>, "All arguments must derive from Layer");

    if constexpr (std::is_same_v<std::decay_t<First>, Input>) {
        layerConfig = LayerConfig(std::forward<First>(first));
    } else {
        layerConfig.push_back(std::forward<First>(first));
    }

    layerConfig.push_layers(std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest, typename>
void Sequential::addLayer(First&& first, Rest&&... rest) {
    static_assert(std::is_base_of_v<Layer, std::decay_t<First>>, "All arguments must derive from Layer");

    // Basically assuming that Sequential is built already, but that more layers need to be added in a similar fashion
    // So what we need to do:
    //  We need to check that the compile state is NONE
    //   IF Sequential is empty, basically have free reign like the constructor as long as it is in correct order
    //   IF though there are already layers, we must assure that an Input layer is not added
    ASSERT(state == CompiledState::NONE, "Sequential compiled, cannot add more layers");
    if constexpr (std::is_same_v<std::decay_t<First>, Input>) {
        ASSERT(layerConfig.empty(), "Cannot add Input layer");
        layerConfig = LayerConfig(std::forward<First>(first));
    } else {
        layerConfig.push_back(std::forward<First>(first));
    }

    layerConfig.push_layers(std::forward<Rest>(rest)...);
}



// IMPLEMENT UGH
inline std::ostream& operator<<(std::ostream& os, const Sequential& model) {
    return os;
}


#endif // SEQUENTIAL_HPP