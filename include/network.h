#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "activation.h"
#include "init.h"


class Network {
    private:
        class Layer {
            private:
                const size_t inputSize;
                const size_t outputSize;

                Matrix weights;
                Matrix biases;
                Matrix input;
                Matrix preActivation;

                const Activations::ActivationType actType;
                const InitType initType;

                void initialize();
                void updateParams(const Matrix& dWeights, const Matrix& dbiases, double learning_rate);
            
            public:
                Layer(
                    size_t input_size,
                    size_t output_size,
                    Activations::ActivationType act_type,
                    InitType init_type
                );

                Matrix forward(const Matrix& X);
                Matrix backward(const Matrix& dA, size_t batch_size, double learning_rate);

                // Getters
                size_t getInputSize() const { return inputSize; }
                size_t getOutputSize() const { return outputSize; }
                const Matrix& getWeights() const { return weights; }
                const Matrix& getBiases() const { return biases; }
                const Matrix& getZ() const { return preActivation; }
                Activations::ActivationType getActivationType() const { return actType; }
                InitType getInitType() const { return initType; }

                // Setters
                void setWeights(const Matrix& W) { weights = W; }
                void setBiases(const Matrix& b) { biases = b; }
        };

        bool isCompiled = false;
        std::vector<Layer> layers;
        const size_t networkInputSize;
        const size_t numClasses;
        Matrix lastOutput;

        void addOutputLayer();
        Matrix forward(const Matrix& X);
        void backward(const Matrix& y_true, double learning_rate);
    
    public:
        Network(
            size_t input_size,
            size_t num_classes
        );

        void addLayer(size_t neurons, Activations::ActivationType act_type, InitType init_type);
        void compile();
};


#endif // NETWORK_H