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

        bool empty() const { return sHead.next == &sTail; }
        size_t size() const { return size_; }
        std::unique_ptr<Layer>& front() const;
        std::unique_ptr<Layer>& back() const;
        void pop_front();
        void pop_back();
        void clear();
        void push_front(std::unique_ptr<Layer> layer);
        void push_back(std::unique_ptr<Layer> layer);

        template <typename T>
        void push_front(T&& layer);

        template <typename T>
        void push_back(T&& layer);
    
    private:
        struct Node {
            Node* next = nullptr;
            Node* prev = nullptr;
            std::unique_ptr<Layer> layer;

            Node() = default;   // for sentinels
            Node(std::unique_ptr<Layer> l) : layer(std::move(l)) {}
        };

        mutable Node sHead;
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

        Iterator begin() const { return Iterator(sHead.next); }
        Iterator end() const { return Iterator(&sTail); }
        void erase(Iterator i);
        void insert(Iterator i, std::unique_ptr<Layer> layer);

        template <typename T>
        void insert(Iterator i, T&& layer);
}; // Class LayerConfig


template <typename T>
void LayerConfig::push_front(T&& layer) {
    static_assert(std::is_base_of_v<Layer, T>, "Must be Layer class derivative");
    Node* node = new Node(std::make_unique<T>(std::forward<T>(layer)));

    node->next = sHead.next;
    node->prev = &sHead;

    sHead.next->prev = node;
    sHead.next = node;

    ++size_;
}

template <typename T>
void LayerConfig::push_back(T&& layer) {
    static_assert(std::is_base_of_v<Layer, T>, "Must be Layer class derivative");
    Node* node = new Node(std::make_unique<T>(std::forward<T>(layer)));

    node->next = &sTail;
    node->prev = sTail.prev;

    sTail.prev->next = node;
    sTail.prev = node;

    ++size_;
}

template <typename T>
void LayerConfig::insert(Iterator i, T&& layer) {
    static_assert(std::is_base_of_v<Layer, T>, "Must be Layer class derivative");

    if (i == end()) {
        push_back(std::make_unique<T>(std::forward<T>(layer)));
        return;
    } else if (i == begin()) {
        push_front(std::make_unique<T>(std::forward<T>(layer)));
        return;
    }

    Node* node = new Node(std::make_unique<T>(std::forward<T>(layer)));
    node->next = i.node_ptr;
    node->prev = i.node_ptr->prev;

    i.node_ptr->prev->next = node;
    i.node_ptr->prev = node;
    ++size_;
}


#endif // LAYER_CONFIG_H