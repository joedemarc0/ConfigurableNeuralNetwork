#include "layer-config.hpp"


// =====================================================
// LayerConfig Class Constructors/Assignment/Destructors
// =====================================================
LayerConfig::LayerConfig() : head(nullptr), tail(nullptr), size_(0) {}

LayerConfig::~LayerConfig() {
    clear();
}


// ==================================
// LayerConfig Class Public Functions
// ==================================
std::unique_ptr<Layer>& LayerConfig::front() {
    assert(!empty());
    return head->layer;
}

std::unique_ptr<Layer>& LayerConfig::back() {
    assert(!empty());
    return tail->layer;
}

void LayerConfig::pop_front() {
    assert(!empty());

    Node* target = head;
    if (size_ > 1) {
        head = head->next;
        head->prev = nullptr;
    } else {
        head = tail = nullptr;
    }

    delete target;
    --size_;
}

void LayerConfig::pop_back() {
    assert(!empty());

    Node* target = tail;
    if (size_ > 1) {
        tail = tail->prev;
        tail->next = nullptr;
    } else {
        head = tail = nullptr;
    }

    delete target;
    --size_;
}

void LayerConfig::clear() {
    while (size_ > 0) {
        pop_front();
    }
}


// ========================================================
// Nested Iterator Class Constructors/Assignment/Destructor
// ========================================================
LayerConfig::Iterator::Iterator() : node_ptr(nullptr) {}

LayerConfig::Iterator::Iterator(const Iterator& other) : node_ptr(other.node_ptr) {}

LayerConfig::Iterator& LayerConfig::Iterator::operator=(const Iterator& other) {
    if (this != std::addressof(other)) {
        node_ptr = other.node_ptr;
    }

    return *this;
}

LayerConfig::Iterator::~Iterator() { node_ptr = nullptr; }


// ===============================
// Nested Iterator Class Operators
// ===============================
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
    assert(i.node_ptr);
    if (i.node_ptr == head) {
        pop_front();
        return;
    } else if (i.node_ptr == tail) {
        pop_back();
        return;
    }

    Node* target = i.node_ptr;
    target->prev->next = target->next;
    target->next->prev = target->prev;
    delete target;
    --size_;
}
