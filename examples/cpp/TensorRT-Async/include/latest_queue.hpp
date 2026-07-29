// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace yolo {

template <typename T>
class LatestQueue {
public:
    explicit LatestQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) throw std::invalid_argument("queue capacity must be greater than zero");
    }

    LatestQueue(const LatestQueue&) = delete;
    LatestQueue& operator=(const LatestQueue&) = delete;

    bool push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return false;
            while (queue_.size() >= capacity_) {
                queue_.pop_front();
                ++dropped_;
            }
            queue_.push_back(std::move(value));
        }
        ready_.notify_one();
        return true;
    }

    std::optional<T> popLatest() {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return std::nullopt;

        dropped_ += queue_.size() - 1;
        T value = std::move(queue_.back());
        queue_.clear();
        return value;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }

    std::size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<T> queue_;
    std::size_t dropped_ = 0;
    bool closed_ = false;
};

}  // namespace yolo
