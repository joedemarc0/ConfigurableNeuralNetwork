#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include "init.h"

enum class LayerType { Dense, Input };

class Layer {
    private:
        size_t inputSize;
        size_t outputSize;

    public:
        virtual Matrix forward(const Matrix& X) = 0;
        virtual Matrix backward(const Matrix& dA) = 0;
        virtual void build(size_t input_size) = 0;

        virtual LayerType type() const = 0;
        virtual bool isBuilt() const = 0;
        virtual size_t getInputSize() const = 0;
        virtual size_t getOutputSize() const = 0;

        virtual ~Layer() = default;
};


class Input : public Layer {
    private:
        size_t inputSize;
        size_t outputSize;
        bool built = false;
    
    public:
        Input(size_t input_size);

        Matrix forward(const Matrix& X) override;
        Matrix backward(const Matrix& dA) override;
        void build(size_t input_size) override;

        LayerType type() const override { return LayerType::Input; }
        bool isBuilt() const override { return built; }
        size_t getInputSize() const override { return inputSize; }
        size_t getOutputSize() const override { return outputSize; }
};


class Dense : public Layer {
    private:
        size_t inputSize;
        size_t outputSize;

        Matrix weights;
        Matrix biases;
        Matrix input;
        Matrix preActivation;

        Matrix dWeights;
        Matrix dbiases;

        bool built = false;
        Activations::ActivationType actType;
        InitType initType;

        void initialize();
        void updateParams(double learning_rate);
        void build();
    
    public:
        Dense(
            size_t output_size,
            Activations::ActivationType act_type,
            InitType init_type
        );

        Dense(
            size_t input_size,
            size_t output_size,
            Activations::ActivationType act_type,
            InitType init_type
        );

        Matrix forward(const Matrix& X) override;
        Matrix backward(const Matrix& dA) override;
        void build(size_t input_size) override;

        LayerType type() const override { return LayerType::Dense; }
        bool isBuilt() const override { return built; }
        size_t getInputSize() const override { return inputSize; }
        size_t getOutputSize() const override { return outputSize; }

        // Getters
        const Matrix& getWeights() const { return weights; }
        const Matrix& getBiases() const { return biases; }
        const Matrix& getZ() const { return preActivation; }
        Activations::ActivationType getActivationType() const { return actType; }
        InitType getInitType() const { return initType; }
};


inline std::string to_string(LayerType type) {
    switch(type) {
        case LayerType::Input: { return "Input"; }
        case LayerType::Dense: { return "Dense"; }
    }
}

inline std::ostream& operator<<(std::ostream& os, const Layer& layer) {
    os << to_string(layer.type()) << "(" << layer.getInputSize() << ", " << layer.getOutputSize() << ")";
}


#endif // LAYER_H