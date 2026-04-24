#include "layer-config.hpp"


// =====================================================
// LayerConfig Class Constructors/Assignment/Destructors
// =====================================================
LayerConfig::LayerConfig() {}

LayerConfig::LayerConfig(const LayerConfig& other) {}

LayerConfig& LayerConfig::operator=(const LayerConfig& other) {}

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