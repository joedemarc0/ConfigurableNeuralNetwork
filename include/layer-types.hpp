#ifndef LAYER_TYPES_HPP
#define LAYER_TYPES_HPP

#include "matrix.hpp"
#include "activation.hpp"
#include "utils.hpp"
#include "init.hpp"


enum class LayerType { Input, Dense, Dropout };

class Layer {
    protected:
        Layer() = default;
        Layer(size_t input_size, size_t output_size)
            : inputSize_(input_size), outputSize_(output_size)
        {}

        std::optional<size_t> inputSize_ = std::nullopt;
        std::optional<size_t> outputSize_ = std::nullopt;
        bool built = false;
    
    public:
        virtual Matrix forward(const Matrix& X) = 0;
        virtual Matrix backward(const Matrix& dA) = 0;
        virtual void build() = 0;
        virtual LayerType type() const = 0;

        void setInputSize(size_t input_size) { inputSize_ = input_size; }
        bool hasInputSize() const { return inputSize_.has_value(); }
        bool hasOutputSize() const { return outputSize_.has_value(); }
        bool isBuilt() const { return built; }

        size_t inputSize() const {
            ASSERT(inputSize_.has_value(), "Input size accessed before it was set");
            return *inputSize_;
        }

        size_t outputSize() const {
            ASSERT(outputSize_.has_value(), "Output size accessed before it was set");
            return *outputSize_;
        }

        virtual ~Layer() = default;
}; // Layer Class Template

class Trainable {
    public:
        virtual void update(double learning_rate) = 0;
        virtual ~Trainable() = default;
};

class Input : public Layer {
    public:
        Input(size_t input_size);

        Matrix forward(const Matrix& X) override;
        Matrix backward(const Matrix& dA) override;
        void build() override;
        LayerType type() const override { return LayerType::Input; }
}; // Input Class

class Dense : public Layer, public Trainable {
    private:
        Matrix weights;
        Matrix biases;
        Matrix input;
        Matrix preActivation;

        Matrix dWeights;
        Matrix dbiases;

        std::unique_ptr<Activation> activation;
        InitType initType;

        void initialize();
        void updateParams(double learning_rate);

    public:
        Dense(size_t output_size);

        Dense(
            size_t output_size,
            std::unique_ptr<Activation> act,
            InitType init_type
        );

        Dense(
            size_t input_size,
            size_t output_size,
            std::unique_ptr<Activation> act,
            InitType init_type
        );

        template <typename T>
        Dense(
            size_t output_size,
            T&& act,
            InitType init_type
        );

        // Need a SetActivation Function FOR SURE.  aldknalkndlknslknsldknslaknlakndlkaalskdnlakdslakndlkansldknalkdnalksnd
        Matrix forward(const Matrix& X) override;
        Matrix backward(const Matrix& dA) override;
        void build() override;
        LayerType type() const override { return LayerType::Dense; }
        void update(double learning_rate) override { updateParams(learning_rate); }

        const Matrix& getWeights() const { return weights; }
        const Matrix& getBiases() const { return biases; }
        const Matrix& getZ() const { return preActivation; }
        ActivationType getActivationType() const { return activation->type(); }
        InitType getInitType() const { return initType; }
}; // Dense Class

class Dropout : public Layer {
    private:
        double rate;
        Matrix mask;

    public:
        Dropout(double dropout_rate);

        Matrix forward(const Matrix& X) override;
        Matrix backward(const Matrix& dA) override;
        void build() override;
        LayerType type() const override { return LayerType::Dropout; }

        const Matrix& getMask() const { return mask; }
}; // Dropout Class


#endif // LAYER_TYPES_HPP