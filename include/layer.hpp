#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include "init.h"


class Layer {
    public:
        virtual Matrix forward(const Matrix& X) = 0;
        virtual Matrix backward(const Matrix& dZ) = 0;
        virtual void updateParams(double learning_rate) = 0;

        virtual void build() = 0;
        virtual bool isBuilt() const = 0;

        virtual ~Layer() = default;
};


class Dense : public Layer {
    private:
        size_t inputSize;
        size_t outputSize;

        Matrix weights;
        Matrix biases;

        bool built = false;
        Activations::ActivationType actType;
        InitType initType;
    
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

        void build() override;
        bool isBuilt() const override;
};


#endif // LAYER_H