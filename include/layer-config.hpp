#ifndef LAYER_CONFIG_H
#define LAYER_CONFIG_H

#include "layer.hpp"
#include <cassert>

class LayerConfig {
    public:
        LayerConfig();
        LayerConfig(const LayerConfig& other);
        LayerConfig& operator=(const LayerConfig& other);
        ~LayerConfig();

        bool empty() const { return head == nullptr; }
        size_t size() const { return size_;}
        std::unique_ptr<Layer>& front();
        std::unique_ptr<Layer>& back();
        void pop_front();
        void pop_back();
        void clear();

        template <typename T>
        void push_front(const T& layer);

        template <typename T>
        void push_back(const T& layer);
    
    private:
        struct Node {
            Node* next;
            Node* prev;
            std::unique_ptr<Layer> layer;

            Node(std::unique_ptr<Layer> l) : next(nullptr), prev(nullptr), layer(std::move(l)) {}
        };

        Node* head;
        Node* tail;
        size_t size_;
};


template <typename T>
void LayerConfig::push_front(const T& layer) {
    static_assert(std::is_base_of_v<Layer, T>, "Must be Layer class derivative");
    Node* target = new Node(std::make_unique<T>(std::move(layer)));
    target->next = head;
    target->prev = nullptr;

    if (!empty()) {
        head->prev = target;
        head = target;
    } else {
        head = tail = target;
    }

    ++size_;
}

template <typename T>
void LayerConfig::push_back(const T& layer) {
    static_assert(std::is_base_of_v<Layer, T>, "Must be Layer class derivative");
    Node* target = new Node(std::make_unique<T>(std::move(layer)));
    target->next = nullptr;
    target->prev = tail;

    if (!empty()) {
        tail->next = target;
        tail = target;
    } else {
        head = tail = target;
    }

    ++size_;
}






#endif // LAYER_CONFIG_H