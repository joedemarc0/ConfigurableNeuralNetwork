#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include "init.h"

enum class LayerType { Dense, Input };

class Layer {
    private:
        virtual void build() = 0;
        virtual void build(size_t input_size) = 0;
        virtual bool isBuilt() const = 0;
        virtual void updateParams(const Matrix& dWeights, const Matrix& dbiases, double learning_rate) = 0;

    public:
        virtual Matrix forward(const Matrix& X) = 0;
        virtual Matrix backward(const Matrix& dA) = 0;
        virtual Matrix backward(const Matrix& dA, size_t batch_size, double learning_rate) = 0;
        virtual LayerType type() const = 0;
        virtual ~Layer() = default;
};


class Input : public Layer {
    private:
        size_t inputSize;
    
    public:
        Input(size_t input_size);

        Matrix forward(const Matrix& X) override;
        Matrix backward(const Matrix& dA) override;
        LayerType type() const override { return LayerType::Input; }
};


class Dense : public Layer {
    private:
        size_t inputSize;
        size_t outputSize;

        Matrix weights;
        Matrix biases;
        Matrix input;
        Matrix preActivation;

        bool built = false;
        Activations::ActivationType actType;
        InitType initType;

        void initialize();
        void updateParams(const Matrix& dWeights, const Matrix& dbiases, double learning_rate) override;

        void build(size_t input_size) override;
        bool isBuilt() const override { return built; }
    
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
        Matrix backward(const Matrix& dA, size_t batch_size, double learning_rate) override;
        LayerType type() const override { return LayerType::Dense; }

        // Getters
        size_t getInputSize() const { return inputSize; }
        size_t getOutputSize() const { return outputSize; }
        const Matrix& getWeights() const { return weights; }
        const Matrix& getBiases() const { return biases; }
        const Matrix& getZ() const { return preActivation; }
        Activations::ActivationType getActivationType() const { return actType; }
        InitType getInitType() const { return initType; }
};


#endif // LAYER_H