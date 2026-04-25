#ifndef LAYER_CONFIG_H
#define LAYER_CONFIG_H

#include "layer.hpp"
#include <cassert>

class LayerConfig {
    public:
        LayerConfig();
        LayerConfig(const LayerConfig& other) = delete;
        LayerConfig& operator=(const LayerConfig& other) = delete;
        ~LayerConfig();

        bool empty() const { return head == nullptr; }
        size_t size() const { return size_;}
        std::unique_ptr<Layer>& front();
        std::unique_ptr<Layer>& back();
        void pop_front();
        void pop_back();
        void clear();

        template <typename T>
        void push_front(T&& layer);

        template <typename T>
        void push_back(T&& layer);
    
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
    
    public:
        class Iterator {
            public:
                Iterator();
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
                Iterator(Node* p) { node_ptr = p; }
        };

        Iterator begin() const { return Iterator(head); }
        Iterator end() const { return Iterator(); }
        void erase(Iterator i);

        template <typename T>
        void insert(Iterator i, T&& layer);
};


template <typename T>
void LayerConfig::push_front(T&& layer) {
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
void LayerConfig::push_back(T&& layer) {
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

template <typename T>
void LayerConfig::insert(Iterator i, T&& layer) {
    static_assert(std::is_base_of_v<Layer, T>, "Must be Layer class derivative");
    if (i == end()) {
        push_back(std::move(layer));
        return;
    } else if (i.node_ptr == head) {
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


#endif // LAYER_CONFIG_H