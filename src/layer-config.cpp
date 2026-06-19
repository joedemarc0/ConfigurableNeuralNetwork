#include "layer-config.hpp"


// ================================
// LayerConfig Class Implementation
// ================================

LayerConfig::LayerConfig()
    : size_(0)
{
    head.layer = std::make_unique<Input>(0);
    head.next = &sTail;
    sTail.prev = &head;
}

LayerConfig::LayerConfig(Input input)
    : size_(0)
{
    head.layer = std::make_unique<Input>(std::move(input));
    head.next = &sTail;
    sTail.prev = &head;
}

LayerConfig::~LayerConfig() {
    Node* current = head.next;
    while (current != &sTail) {
        Node* next = current->next;
        delete current;
        current = next;
    }
}

LayerConfig::LayerConfig(LayerConfig&& other) noexcept
    : size_(other.size_)
{
    head.layer = std::move(other.head.layer);

    if (other.empty()) {
        head.next = &sTail;
        sTail.prev = &head;
    } else {
        head.next = other.head.next;
        sTail.prev = other.sTail.prev;
        head.next->prev = &head;
        sTail.prev->next = &sTail;
    }

    other.head.next = &other.sTail;
    other.sTail.prev = &other.head;
    other.size_ = 0;
}

LayerConfig& LayerConfig::operator=(LayerConfig&& other) noexcept {
    if (this == &other) return *this;

    head.layer = std::move(other.head.layer);
    if (other.empty()) {
        head.next = &sTail;
        sTail.prev = &head;
    } else {
        head.next = other.head.next;
        sTail.prev = other.sTail.prev;
        head.next->prev = &head;
        sTail.prev->next = &sTail;
    }

    other.head.next = &other.sTail;
    other.sTail.prev = &other.head;
    other.size_ = 0;
    return *this;
}

// ==================================
// LayerConfig Class Public Functions
// ==================================

std::unique_ptr<Layer>& LayerConfig::front() const {
    assert(!empty());
    return head.next->layer;
}

std::unique_ptr<Layer>& LayerConfig::back() const {
    assert(!empty());
    return sTail.prev->layer;
}

void LayerConfig::pop_front() {
    assert(!empty());

    Node* target = head.next;
    target->next->prev = &head;
    head.next = target->next;

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

    node->next = head.next;
    node->prev = &head;

    head.next->prev = node;
    head.next = node;

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

void LayerConfig::erase(Iterator i) {
    assert(i != end());
    assert(i.node_ptr);

    Node* target = i.node_ptr;
    target->next->prev = target->prev;
    target->prev->next = target->next;

    delete target;
    --size_;
}

void LayerConfig::insert(Iterator i, std::unique_ptr<Layer> layer) {
    if (i == end()) {
        push_back(std::move(layer));
        return;
    } else if (i == begin() || i == input()) {
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

void LayerConfig::replace(Iterator i, std::unique_ptr<Layer> layer) {
    assert(i != input() && i != end());
    assert(i.node_ptr);

    Node* node = new Node(std::move(layer));
    Node* target = i.node_ptr;

    node->next = target->next;
    node->prev = target->prev;

    node->prev->next = node;
    node->next->prev = node;
    delete target;
}

void LayerConfig::buildLayer(Iterator it) {
    Node* node = it.node_ptr;
    bool hasNext = (node->next != &sTail && node->next->layer);

    if (it == input()) {
        ASSERT(it->hasInputSize(), "Input being built without input size");
        it->build();
        return;
    }

    if (it->isBuilt()) {
        ASSERT(it->inputSize() == node->prev->layer->outputSize(), "Dimension mismatch between built layers");
        if (hasNext) ASSERT(it->outputSize() == node->next->layer->inputSize(), "Dimension mismatch between built layers");
        return;
    }

    if (!it->hasInputSize()) {
        it->setInputSize(node->prev->layer->outputSize());
    } else {
        ASSERT(it->inputSize() == node->prev->layer->outputSize(), "Dimension mismatch between unbuilt however initialized layers");
    }

    it->build();
}

void LayerConfig::compile() {
    forEachLayer([&](Iterator it) {
        ASSERT(it->type() != LayerType::Input, "Input layer found after head node");
        if (it == begin()) return;
        buildLayer(it);
    });
}


// =======================================================
// Nested Iterator Class Constructor/ASSignment/Destructor
// =======================================================

LayerConfig::Iterator::Iterator(const Iterator& i) : node_ptr(i.node_ptr) {}

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

// Increment/Decrement
LayerConfig::Iterator& LayerConfig::Iterator::operator++() {
    assert(node_ptr);
    node_ptr = node_ptr->next;
    return *this;
}

LayerConfig::Iterator& LayerConfig::Iterator::operator--() {
    assert(node_ptr);
    node_ptr = node_ptr->prev;
    return *this;
}

// Dereference
std::unique_ptr<Layer>& LayerConfig::Iterator::operator*() const {
    assert(node_ptr);
    return node_ptr->layer;
}

Layer* LayerConfig::Iterator::operator->() const {
    assert(node_ptr);
    return node_ptr->layer.get();
}

// Comparison
bool LayerConfig::Iterator::operator==(const Iterator& other) const {
    return node_ptr == other.node_ptr;
}

bool LayerConfig::Iterator::operator!=(const Iterator& other) const {
    return node_ptr != other.node_ptr;
}

bool LayerConfig::Iterator::operator==(const ConstIterator& other) const {
    return node_ptr == other.node_ptr;
}

bool LayerConfig::Iterator::operator!=(const ConstIterator& other) const {
    return node_ptr != other.node_ptr;
}


// ============================================================
// Nested ConstIterator Class Constructor/ASSignment/Destructor
// ============================================================

LayerConfig::ConstIterator::ConstIterator(const ConstIterator& i) : node_ptr(i.node_ptr) {}

LayerConfig::ConstIterator::ConstIterator(const Iterator& i) : node_ptr(i.node_ptr) {}

LayerConfig::ConstIterator& LayerConfig::ConstIterator::operator=(const ConstIterator& other) {
    if (this != &other) {
        node_ptr = other.node_ptr;
    }

    return *this;
}

LayerConfig::ConstIterator& LayerConfig::ConstIterator::operator=(const Iterator& other) {
    node_ptr = other.node_ptr;
    return *this;
}

LayerConfig::ConstIterator::~ConstIterator() { node_ptr = nullptr; }


// ====================================
// Nested ConstIterator Class Operators
// ====================================

// Increment/Decrement
LayerConfig::ConstIterator& LayerConfig::ConstIterator::operator++() {
    assert(node_ptr);
    node_ptr = node_ptr->next;
    return *this;
}

LayerConfig::ConstIterator& LayerConfig::ConstIterator::operator--() {
    assert(node_ptr);
    node_ptr = node_ptr->prev;
    return *this;
}

// Dereference
const Layer& LayerConfig::ConstIterator::operator*() const {
    assert(node_ptr);
    return *node_ptr->layer;
}

const Layer* LayerConfig::ConstIterator::operator->() const {
    assert(node_ptr);
    return node_ptr->layer.get();
}

// Comparison
bool LayerConfig::ConstIterator::operator==(const ConstIterator& other) const {
    return node_ptr == other.node_ptr;
}

bool LayerConfig::ConstIterator::operator!=(const ConstIterator& other) const {
    return node_ptr != other.node_ptr;
}

bool LayerConfig::ConstIterator::operator==(const Iterator& other) const {
    return node_ptr == other.node_ptr;
}

bool LayerConfig::ConstIterator::operator!=(const Iterator& other) const {
    return node_ptr != other.node_ptr;
}