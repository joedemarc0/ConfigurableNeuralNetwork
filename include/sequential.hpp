#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "layer-config.hpp"


class Sequential {
    public:
        template <typename First, typename... Rest>
        Sequential(First&& first, Rest&&... rest);

        void compile();
    
    private:
        LayerConfig layerConfig;
        bool semiCompiled = false;
        bool fullyCompiled = false;

        template <typename T, typename... Rest>
        void pushLayers(T&& layer, Rest&&... rest);
        void pushLayers() {}

        Matrix forward(const Matrix& X);
};


template <typename First, typename... Rest>
Sequential::Sequential(First&& first, Rest&&... rest) {
    static_assert(std::is_base_of_v<Layer, std::decay_t<First>>, "All arguments must derive from Layer");

    if constexpr (std::is_same_v<std::decay_t<First>, Input>) {
        layerConfig.setInput(std::forward<First>(first));
    } else {
        layerConfig.push_back(std::forward<First>(first));
    }

    pushLayers(std::forward<Rest>(rest)...);
}

template <typename T, typename... Rest>
void Sequential::pushLayers(T&& layer, Rest&&... rest) {
    static_assert(std::is_base_of_v<Layer, std::decay_t<T>>, "All arguments must derive from Layer");
    static_assert(!std::is_same_v<std::decay_t<T>, Input>, "No layer but the first can be an Input type");

    layers.push_back(std::forward<T>(layer));
    pushLayers(std::forward<Rest>(rest)...);
}

inline std::ostream& operator<<(std::ostream& os, const Sequential& model) {}


#endif // SEQUENTIAL_HPP