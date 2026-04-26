#include "layer-config2.hpp"
#include "utils.h"


// ================================
// Layers and their Implementations
// Input Layer Implementation
// ==========================
Input::Input(size_t input_size) : inputSize(input_size) {}
Matrix Input::forward(const Matrix& X) { return X; }
Matrix Input::backward(const Matrix& dA) { return dA; }
void Input::build(size_t input_size) {
    inputSize = outputSize = input_size;
    built = true;
}

// ===========================
// Dense Layer Implementations
// ===========================

// Constructors
Dense::Dense(
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type
) : inputSize(0),
    outputSize(output_size),
    actType(act_type),
    initType(init_type)
{}

Dense::Dense(
    size_t input_size,
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type
) : inputSize(input_size),
    outputSize(output_size),
    actType(act_type),
    initType(init_type)
{
    ASSERT(input_size != 0, "Input size cannot be 0");
    build();
}

// Dense Layer Private Functions
void Dense::initialize() {}

void Dense::updateParams(double learning_rate) {}

void Dense::build() {}

// Dense Layer Public Functions
Matrix Dense::forward(const Matrix& X) {}

Matrix Dense::backward(const Matrix& dA) {}

void Dense::build(size_t input_size) {}



// =========================================
// LayerConfig Class Constructor/Destructors
// =========================================
LayerConfig::LayerConfig() : size_(0) {
    sHead.next = &sTail;
    sTail.prev = &sHead;
}

LayerConfig::~LayerConfig() {
    clear();
    sHead.next = sTail.prev = nullptr;
}


// ==================================
// LayerConfig Class Public Functions
// ==================================
std::unique_ptr<Layer>& LayerConfig::front() const {
    assert(!empty());
    return sHead.next->layer;
}

std::unique_ptr<Layer>& LayerConfig::back() const {
    assert(!empty());
    return sTail.prev->layer;
}

void LayerConfig::pop_front() {
    assert(!empty());

    Node* target = sHead.next;

    target->next->prev = &sHead;
    sHead.next = target->next;

    delete target;
    --size_;
}

void LayerConfig::pop_back() {
    assert(!empty());

    Node* target = sTail.prev;

    target->prev->next = &sTail;
    sTail.prev = target->prev;

    delete target;
    --size_;
}

void LayerConfig::clear() {
    while (!empty()) {
        pop_front();
    }
}

void LayerConfig::push_front(std::unique_ptr<Layer> layer) {
    Node* node = new Node(std::move(layer));

    node->next = sHead.next;
    node->prev = &sHead;

    sHead.next->prev = node;
    sHead.next = node;

    ++size_;
}

void LayerConfig::push_back(std::unique_ptr<Layer> layer) {
    Node* node = new Node(std::move(layer));

    node->next = &sTail;
    node->prev = sTail.prev;

    sTail.prev->next = node;
    sTail.prev = node;

    ++size_;
}


// =======================================================
// Nested Iterator Class Constructor/ASSignment/Destructor
// =======================================================
LayerConfig::Iterator::Iterator(const Iterator& other) : node_ptr(other.node_ptr) {}

LayerConfig::Iterator& LayerConfig::Iterator::operator=(const Iterator& other) {
    if (this != &other) {
        node_ptr = other.node_ptr;
    }

    return *this;
}

LayerConfig::Iterator::~Iterator() { node_ptr = nullptr; }


// ===============================
// Nested Iterator Class Operators
// ===============================
LayerConfig::Iterator& LayerConfig::Iterator::operator--() {
    assert(node_ptr);
    node_ptr = node_ptr->prev;
    return *this;
}

LayerConfig::Iterator& LayerConfig::Iterator::operator++() {
    assert(node_ptr && node_ptr->next);
    node_ptr = node_ptr->next;
    return *this;
}

std::unique_ptr<Layer>& LayerConfig::Iterator::operator*() const {
    assert(node_ptr);
    return node_ptr->layer;
}

Layer* LayerConfig::Iterator::operator->() const {
    assert(node_ptr);
    return node_ptr->layer.get();
}

bool LayerConfig::Iterator::operator==(const Iterator& other) const {
    return node_ptr == other.node_ptr;
}

bool LayerConfig::Iterator::operator!=(const Iterator& other) const {
    return node_ptr != other.node_ptr;
}


// ======================================
// LayerConfig Public Iterating Functions
// ======================================
void LayerConfig::erase(Iterator i) {
    assert(i != end());
    assert(i.node_ptr);
    Node* target = i.node_ptr;

    target->prev->next = target->next;
    target->next->prev = target->prev;

    delete target;
    --size_;
}

void LayerConfig::insert(Iterator i, std::unique_ptr<Layer> layer) {
    if (i == end()) {
        push_back(std::move(layer));
        return;
    } else if (i == begin()) {
        push_front(std::move(layer));
        return;
    }

    Node* node = new Node(std::move(layer));
    node->next = i.node_ptr;
    node->prev = i.node_ptr->prev;

    i.node_ptr->prev->next = node;
    i.node_ptr->prev = node;
    ++size_;
}
