#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "layer-config.hpp"

enum class CompiledState { NONE, SEMICOMPILED, COMPILED };

class Sequential {
    public:
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

        template <typename T, typename... Rest>
        void pushLayers(T&& layer, Rest&&... rest);
        void pushLayers() {}

        Matrix forward(const Matrix& X);
};


template <
    typename First,
    typename... Rest,
    typename
>
Sequential::Sequential(First&& first, Rest&&... rest) {
    static_assert(std::is_base_of_v<Layer, std::decay_t<First>>, "All arguments must derive from Layer");

    if constexpr (std::is_same_v<std::decay_t<First>, Input>) {
        layerConfig = LayerConfig(std::forward<First>(first));
    } else {
        layerConfig.push_back(std::forward<First>(first));
    }

    pushLayers(std::forward<Rest>(rest)...);
}

template <typename T, typename... Rest>
void Sequential::pushLayers(T&& layer, Rest&&... rest) {
    static_assert(std::is_base_of_v<Layer, std::decay_t<T>>, "All arguments must derive from Layer");
    static_assert(!std::is_same_v<std::decay_t<T>, Input>, "No layer but the first can be an Input type");

    layerConfig.push_back(std::forward<T>(layer));
    pushLayers(std::forward<Rest>(rest)...);
}





// IMPLEMENT UGH
inline std::ostream& operator<<(std::ostream& os, const Sequential& model) {
    return os;
}


#endif // SEQUENTIAL_HPP