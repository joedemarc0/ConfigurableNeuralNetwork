#ifndef LAYER_CONFIG_HPP
#define LAYER_CONFIG_HPP

#include "matrix.h"
#include "activation.h"
#include "init.h"
#include <cassert>


enum class LayerType { Input, Dense };
class Layer {
    protected:
        size_t inputSize;
        size_t outputSize;
        bool built = false;

        Layer(
            size_t input_size = 0,
            size_t output_size = 0
        ) : inputSize(input_size),
            outputSize(output_size)
        {}
    
    public:
        virtual Matrix forward(const Matrix& X) = 0;
        virtual Matrix backward(const Matrix& dA) = 0;
        virtual void build(size_t input_size) = 0;

        virtual LayerType type() const = 0;
        virtual bool isBuilt() const { return built; }
        virtual size_t getInputSize() const { return inputSize; }
        virtual size_t getOutputSize() const { return outputSize; }

        virtual ~Layer() = default;
}; // Layer Class Template

class Input : public Layer {
    public:
        Input(size_t input_size);

        Matrix forward(const Matrix& X) override;
        Matrix backward(const Matrix& dA) override;
        void build(size_t input_size) override;
        LayerType type() const override { return LayerType::Input; }
}; // Input Class

class Dense : public Layer {
    private:
        Matrix weights;
        Matrix biases;
        Matrix input;
        Matrix preActivation;

        Matrix dWeights;
        Matrix dbiases;

        Activations::ActivationType actType;
        InitType initType;

        void initialize();
        void updateParams(double learning_rate);

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

        const Matrix& getWeights() const { return weights; }
        const Matrix& getBiases() const { return biases; }
        const Matrix& getZ() const { return preActivation; }
        Activations::ActivationType getActivationType() const { return actType; }
        InitType getInitType() const { return initType; }
}; // Dense Class


class LayerConfig {
    public:
        LayerConfig();
        LayerConfig(const LayerConfig& other) = delete;
        LayerConfig& operator=(const LayerConfig& other) = delete;
        ~LayerConfig();

        bool empty() const { return head.next == &sTail; }
        size_t size() const { return size_; }

        Input* getInput() const;
        void setInput(Input input);

        std::unique_ptr<Layer>& front() const;
        std::unique_ptr<Layer>& back() const;
        void pop_front();
        void pop_back();
        void clear();
        void push_front(std::unique_ptr<Layer> layer);
        void push_back(std::unique_ptr<Layer> layer);

        template <typename T> void push_front(T&& layer);
        template <typename T> void push_back(T&& layer);
    
    private:
        struct Node {
            Node* next = nullptr;
            Node* prev = nullptr;
            std::unique_ptr<Layer> layer;

            Node() = default;
            Node(std::unique_ptr<Layer> l) : layer(std::move(l)) {}
        }; // Node Struct

        mutable Node head;
        mutable Node sTail;
        size_t size_;
    
    public:
        class Iterator {
            public:
                Iterator(const Iterator& i);
                Iterator& operator=(const Iterator& i);
                ~Iterator();

                // Increment/Decrement
                Iterator& operator--();
                Iterator& operator++();

                // Dereference
                std::unique_ptr<Layer>& operator*() const;
                Layer* operator->() const;

                // Comparison
                bool operator==(const Iterator& other) const;
                bool operator!=(const Iterator& other) const;
            
            private:
                friend class LayerConfig;

                Node* node_ptr;
                Iterator() { node_ptr = nullptr; }
                Iterator(Node* p) { node_ptr = p; }
        }; // Iterator Class

        Iterator input() const { return Iterator(&head); }
        Iterator begin() const { return Iterator(head.next); }
        Iterator end() const { return Iterator(&sTail); }
        void erase(Iterator i);
        void insert(Iterator i, std::unique_ptr<Layer> layer);
        void replace(Iterator i, std::unique_ptr<Layer> layer);

        template <typename T> void insert(Iterator i, T&& layer);
        template <typename T> void replace(Iterator i, T&& layer);
}; // LayerConfig Class


template <typename T>
void LayerConfig::push_front(T&& layer) {
    using LayerT = std::decay_t<T>;
    static_assert(std::is_base_of_v<Layer, LayerT>, "Must be Layer class derivative");
    Node* node = new Node(std::make_unique<LayerT>(std::forward<T>(layer)));

    node->next = head.next;
    node->prev = &head;

    head.next->prev = node;
    head.next = node;

    ++size_;
}

template <typename T>
void LayerConfig::push_back(T&& layer) {
    using LayerT = std::decay_t<T>;
    static_assert(std::is_base_of_v<Layer, LayerT>, "Must be Layer class derivative");
    Node* node = new Node(std::make_unique<LayerT>(std::forward<T>(layer)));

    node->next = &sTail;
    node->prev = sTail.prev;

    sTail.prev->next = node;
    sTail.prev = node;

    ++size_;
}

template <typename T>
void LayerConfig::insert(Iterator i, T&& layer) {
    using LayerT = std::decay_t<T>;
    static_assert(std::is_base_of_v<Layer, LayerT>, "Must be Layer class derivative");

    if (i == end()) {
        push_back(std::make_unique<LayerT>(std::forward<T>(layer)));
        return;
    } else if (i == begin() || i == input()) {
        push_front(std::make_unique<LayerT>(std::forward<T>(layer)));
        return;
    }

    Node* node = new Node(std::make_unique<LayerT>(std::forward<T>(layer)));
    node->next = i.node_ptr;
    node->prev = i.node_ptr->prev;

    i.node_ptr->prev->next = node;
    i.node_ptr->prev = node;
    ++size_;
}

template <typename T>
void LayerConfig::replace(Iterator i, T&& layer) {
    using LayerT = std::decay_t<T>;
    static_assert(std::is_base_of_v<Layer, LayerT>, "Must be Layer class derivative");

    assert(i != input() && i != end());
    assert(i.node_ptr);

    Node* node = new Node(std::make_unique<LayerT>(std::forward<T>(layer)));
    Node* target = i.node_ptr;

    node->next = target->next;
    node->prev = target->prev;
    delete target;
}


#endif // LAYER_CONFIG_HPP