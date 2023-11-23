#pragma once

template <typename T>
class TensorRTHolder {
    T* holder;
public:
    explicit TensorRTHolder(T* holder_) : holder(holder_) {}
    ~TensorRTHolder() {
        if (holder)
            holder->destroy();
    }
    TensorRTHolder(const TensorRTHolder&) = delete;
    TensorRTHolder& operator=(const TensorRTHolder&) = delete;
    TensorRTHolder(TensorRTHolder && rhs) noexcept{
        holder = rhs.holder;
        rhs.holder = nullptr;
    }
    TensorRTHolder& operator=(TensorRTHolder&& rhs) noexcept {
        if (this == &rhs) {
            return *this;
        }
        if (holder) holder->destroy();
        holder = rhs.holder;
        rhs.holder = nullptr;
        return *this;
    }
    T* operator->() {
        return holder;
    }
    T* get() { return holder; }
    explicit operator bool() { return holder != nullptr; }
    T& operator*() noexcept { return *holder; }
};

template <typename T>
TensorRTHolder<T> make_holder(T* holder) {
    return TensorRTHolder<T>(holder);
}

template <typename T>
using TensorRTNonHolder = T*;