#include "layer-config.hpp"


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
