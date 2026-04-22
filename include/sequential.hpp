#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "layer.hpp"


class Sequential {
    public:
        Sequential();

        template <
            typename... Args,
            typename = std::enable_if_t<(std::is_base_of_v<Dense, Args>&& ...)>
        >
        Sequential(Args... args);
    
    private:
        bool isCompiled = false;
        std::vector<std::unique_ptr<Layer>> layers;
    
    public:
        void compile();
};

template <
    typename... Args,
    typename = std::enable_if_t<(std::is_base_of_v<Layer, Args>&& ...)>
>
Sequential::Sequential(Args... args)
    : isCompiled(false)
{
    size_t layer_num = 0;
    auto add_one = [&](auto&& layer) {
        using T = std::decay_t<decltype(layer)>;
        if (layer_num > 0) ASSERT(layer.type() != LayerType::Input, "Only the first layer can be an input layer");
        layers.push_back(std::make_unique<T>(std::forward<decltype(layer)>(layer)));
        ++layer_num;
    }

    (add_one(std::forward<Args>(args)), ...);
}


inline std::ostream& operator<<(std::ostream& os, const Sequential& model) {}


#endif // SEQUENTIAL_H