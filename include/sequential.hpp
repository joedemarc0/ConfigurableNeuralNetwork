#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "layer.hpp"


class Sequential {
    public:
        Sequential();
        
        template <typename... Args, typename = std::enable_if_t<(std::is_same_v<Dense, Args>&& ...)>>
        Sequential(Args... args);
    
    private:
        bool isCompiled = false;
        std::vector<std::unique_ptr<Layer>> layers;
    
    public:
};

template <typename... Args, typename = std::enable_if_t<(std::is_same_v<Layer, Args>&& ...)>>
Sequential::Sequential(Args... args)
    : isCompiled(false)
{}


inline std::ostream& operator<<(std::ostream& os, const Sequential& model) {}


#endif // SEQUENTIAL_H