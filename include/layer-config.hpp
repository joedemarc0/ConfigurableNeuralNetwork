#ifndef LAYER_CONFIG_HPP
#define LAYER_CONFIG_HPP

#include "matrix.h"
#include "activation.h"
#include "init.h"
#include "utils.h"
#include <cassert>

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

        Activations::ActivationType actType;
        InitType initType;

        void initialize();
        void updateParams(double learning_rate);

    public:
        Dense(size_t output_size);

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
        void build() override;
        LayerType type() const override { return LayerType::Dense; }
        void update(double learning_rate) override { updateParams(learning_rate); }

        const Matrix& getWeights() const { return weights; }
        const Matrix& getBiases() const { return biases; }
        const Matrix& getZ() const { return preActivation; }
        Activations::ActivationType getActivationType() const { return actType; }
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


class LayerConfig {
    public:
        LayerConfig();
        explicit LayerConfig(Input input);
        ~LayerConfig();

        LayerConfig(LayerConfig&& other) noexcept;
        LayerConfig& operator=(LayerConfig&& other) noexcept;

        LayerConfig(const LayerConfig& other) = delete;
        LayerConfig& operator=(const LayerConfig& other) = delete;

        bool empty() const { return head.next == &sTail; }
        size_t size() const { return size_; }
        std::unique_ptr<Layer>& front() const;
        std::unique_ptr<Layer>& back() const;
        void pop_front();
        void pop_back();
        void clear();
        void push_front(std::unique_ptr<Layer> layer);
        void push_back(std::unique_ptr<Layer> layer);

        template <typename T> void push_front(T&& layer);
        template <typename T> void push_back(T&& layer);
        template <typename T, typename... Rest> void push_layers(T&& layer, Rest&&... rest);
        void push_layers() {}
    
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
        class ConstIterator;
        class Iterator {
            public:
                Iterator(const Iterator& i);
                Iterator& operator=(const Iterator& other);
                ~Iterator();

                // Increment/Decrement
                Iterator& operator++();
                Iterator& operator--();

                // Dereference
                std::unique_ptr<Layer>& operator*() const;
                Layer* operator->() const;

                // Comparison
                bool operator==(const Iterator& other) const;
                bool operator!=(const Iterator& other) const;

                bool operator==(const ConstIterator& other) const;
                bool operator!=(const ConstIterator& other) const;
            
            private:
                friend class LayerConfig;

                Node* node_ptr;
                Iterator() { node_ptr = nullptr; }
                Iterator(Node* p) { node_ptr = p; }
        }; // Iterator Class

        class ConstIterator {
            public:
                ConstIterator(const ConstIterator& i);
                ConstIterator(const Iterator& i);
                ConstIterator& operator=(const ConstIterator& other);
                ConstIterator& operator=(const Iterator& other);
                ~ConstIterator();

                // Increment/Decrement
                ConstIterator& operator++();
                ConstIterator& operator--();

                // Dereference
                const Layer& operator*() const;
                const Layer* operator->() const;

                // Comparison
                bool operator==(const ConstIterator& other) const;
                bool operator!=(const ConstIterator& other) const;

                bool operator==(const Iterator& other) const;
                bool operator!=(const Iterator& other) const;
            
            private:
                friend class LayerConfig;

                const Node* node_ptr;
                ConstIterator() { node_ptr = nullptr; }
                ConstIterator(const Node* p) { node_ptr = p; }
        }; // ConstIterator Class

        Iterator input() { return Iterator(&head); }
        Iterator begin() { return Iterator(head.next); }
        Iterator end() { return Iterator(&sTail); }
        ConstIterator input() const { return ConstIterator(&head); }
        ConstIterator begin() const { return ConstIterator(head.next); }
        ConstIterator end() const { return ConstIterator(&sTail); }

        void erase(Iterator i);
        void insert(Iterator i, std::unique_ptr<Layer> layer);
        void replace(Iterator i, std::unique_ptr<Layer> layer);

        template <typename T> void insert(Iterator i, T&& layer);
        template <typename T> void replace(Iterator i, T&& layer);

        // Overloading for loops over LayerConfig Layers since they are ugly and long
        // Overloads for loops from the first non-Input and from the Input
        template <typename Fn> void forEach(Iterator start, Iterator end, Fn&& func);
        template <typename Fn> void forEach(ConstIterator start, ConstIterator end, Fn&& func) const;

        template <typename Fn> void forEach(Iterator start, Fn&& func) { forEach(start, end(), std::forward<Fn>(func)); }
        template <typename Fn> void forEach(ConstIterator start, Fn&& func) const { forEach(start, end(), std::forward<Fn>(func)); }

        template <typename Fn> void forEachLayer(Fn&& func) { forEach(begin(), std::forward<Fn>(func)); }
        template <typename Fn> void forEachLayer(Fn&& func) const { forEach(begin(), std::forward<Fn>(func)); }

        template <typename Fn> void forEachFromInput(Fn&& func) { forEach(input(), std::forward<Fn>(func)); }
        template <typename Fn> void forEachFromInput(Fn&& func) const { forEach(input(), std::forward<Fn>(func)); }

        template <typename Fn> void forEachBackwards(Iterator end, Iterator start, Fn&& func);
        template <typename Fn> void forEachBackwards(ConstIterator end, ConstIterator start, Fn&& func) const;

        void buildLayer(Iterator it);
        void compile();
        Matrix forward(const Matrix& X);
        void backward(const Matrix& y_true, double learning_rate);
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

template <typename T, typename... Rest>
void LayerConfig::push_layers(T&& layer, Rest&&... rest) {
    static_assert(std::is_base_of_v<Layer, std::decay_t<T>>, "All arguments must derive from Layer");
    static_assert(!std::is_same_v<std::decay_t<T>, Input>, "Layer after First must not be Input");

    push_back(std::forward<T>(layer));
    push_layers(std::forward<Rest>(rest)...);
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

    node->prev->next = node;
    node->next->prev = node;
    delete target;
}

template <typename Fn>
void LayerConfig::forEach(Iterator start, Iterator end, Fn&& func) {
    for (auto it = start; it != end; ++it) func(it);
}

template <typename Fn>
void LayerConfig::forEach(ConstIterator start, ConstIterator end, Fn&& func) const {
    for (auto it = start; it != end; ++it) func(it);
}

template <typename Fn>
void LayerConfig::forEachBackwards(Iterator end, Iterator start, Fn&& func) {
    for (auto it = end; it != start; --it) func(it);
}

template <typename Fn>
void LayerConfig::forEachBackwards(ConstIterator end, ConstIterator start, Fn&& func) const {
    for (auto it = end; it != start; --it) func(it);
}


// Inline Functions
inline std::string to_string(LayerType type) {
    switch (type) {
        case LayerType::Input: return "Input";
        case LayerType::Dense: return "Dense";
        case LayerType::Dropout: return "Dropout";
        default: return "Unknown";
    }
}

// IMPLEMENT THESE TOO
inline std::ostream& operator<<(std::ostream& os, const Layer& layer) {
    os << to_string(layer.type()) << "([" << (layer.isBuilt() ? "built" : "unbuilt") << "], ";
    layer.hasInputSize() ? os << "shape=(" << layer.inputSize() << ", " : os << "shape=( - , ";
    layer.hasOutputSize() ? os << layer.outputSize() << "), " : os << "- ), ";
    // os << getActType()
    // os << getInitType()
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const LayerConfig& config) {
    config.forEachFromInput([&](LayerConfig::ConstIterator it) {
        os << *it << " -> ";
    });

    return os;
}


#endif // LAYER_CONFIG_HPP