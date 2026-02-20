#ifndef MSCKF_SYSTEM_LOCKFREE_QUEUE_HPP
#define MSCKF_SYSTEM_LOCKFREE_QUEUE_HPP

#include "../math/types.hpp"
#include <cstring>
#include <atomic>

namespace msckf {

inline void memoryBarrierRelease() {
    std::atomic_thread_fence(std::memory_order_release);
}

inline void memoryBarrierAcquire() {
    std::atomic_thread_fence(std::memory_order_acquire);
}

inline void memoryBarrierSeqCst() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

template<typename T, uint32 Capacity>
class SPSCQueue {
private:
    alignas(64) T buffer_[Capacity];
    alignas(64) volatile uint32 head_;
    alignas(64) volatile uint32 tail_;
    
public:
    SPSCQueue() : head_(0), tail_(0) {}
    
    bool push(const T& item) {
        uint32 nextHead = (head_ + 1) % Capacity;
        
        if (nextHead == tail_) {
            return false;
        }
        
        buffer_[head_] = item;
        
        memoryBarrierRelease();
        
        head_ = nextHead;
        
        return true;
    }
    
    bool pop(T& item) {
        if (tail_ == head_) {
            return false;
        }
        
        memoryBarrierAcquire();
        
        item = buffer_[tail_];
        
        tail_ = (tail_ + 1) % Capacity;
        
        return true;
    }
    
    bool peek(T& item) const {
        if (tail_ == head_) {
            return false;
        }
        
        item = buffer_[tail_];
        return true;
    }
    
    bool empty() const {
        return head_ == tail_;
    }
    
    bool full() const {
        return ((head_ + 1) % Capacity) == tail_;
    }
    
    uint32 size() const {
        if (head_ >= tail_) {
            return head_ - tail_;
        }
        return Capacity - tail_ + head_;
    }
    
    uint32 capacity() const {
        return Capacity - 1;
    }
    
    void clear() {
        head_ = 0;
        tail_ = 0;
    }
};

template<typename T, uint32 Capacity>
class MPSCQueue {
private:
    alignas(64) T buffer_[Capacity];
    alignas(64) std::atomic<uint32> head_;
    alignas(64) volatile uint32 tail_;
    
    bool tryPush(const T& item, uint32 expectedHead) {
        uint32 nextHead = (expectedHead + 1) % Capacity;
        
        if (nextHead == tail_) {
            return false;
        }
        
        buffer_[expectedHead] = item;
        
        memoryBarrierRelease();
        
        return head_.compare_exchange_strong(expectedHead, nextHead, 
                                             std::memory_order_relaxed, 
                                             std::memory_order_relaxed);
    }
    
public:
    MPSCQueue() : head_(0), tail_(0) {}
    
    bool push(const T& item) {
        uint32 expectedHead;
        
        do {
            expectedHead = head_.load(std::memory_order_relaxed);
            uint32 nextHead = (expectedHead + 1) % Capacity;
            
            if (nextHead == tail_) {
                return false;
            }
        } while (!tryPush(item, expectedHead));
        
        return true;
    }
    
    bool pop(T& item) {
        if (tail_ == head_.load(std::memory_order_relaxed)) {
            return false;
        }
        
        memoryBarrierAcquire();
        
        item = buffer_[tail_];
        
        tail_ = (tail_ + 1) % Capacity;
        
        return true;
    }
    
    bool empty() const {
        return head_.load(std::memory_order_relaxed) == tail_;
    }
    
    uint32 size() const {
        uint32 h = head_.load(std::memory_order_relaxed);
        if (h >= tail_) {
            return h - tail_;
        }
        return Capacity - tail_ + h;
    }
};

template<typename T, uint32 Capacity>
class SPMCQueue {
private:
    alignas(64) T buffer_[Capacity];
    alignas(64) volatile uint32 head_;
    alignas(64) std::atomic<uint32> tail_;
    
public:
    SPMCQueue() : head_(0), tail_(0) {}
    
    bool push(const T& item) {
        uint32 nextHead = (head_ + 1) % Capacity;
        
        if (nextHead == tail_.load(std::memory_order_relaxed)) {
            return false;
        }
        
        buffer_[head_] = item;
        
        memoryBarrierRelease();
        
        head_ = nextHead;
        
        return true;
    }
    
    bool pop(T& item) {
        uint32 currentTail;
        uint32 nextTail;
        
        do {
            currentTail = tail_.load(std::memory_order_relaxed);
            
            if (currentTail == head_) {
                return false;
            }
            
            nextTail = (currentTail + 1) % Capacity;
            
            memoryBarrierAcquire();
            item = buffer_[currentTail];
            
        } while (!tail_.compare_exchange_strong(currentTail, nextTail,
                                                 std::memory_order_relaxed,
                                                 std::memory_order_relaxed));
        
        return true;
    }
    
    bool empty() const {
        return head_ == tail_.load(std::memory_order_relaxed);
    }
    
    uint32 size() const {
        uint32 t = tail_.load(std::memory_order_relaxed);
        if (head_ >= t) {
            return head_ - t;
        }
        return Capacity - t + head_;
    }
};

template<typename T, uint32 Capacity>
class MPMCQueue {
private:
    struct Cell {
        T data;
        std::atomic<uint32> sequence;
    };
    
    alignas(64) Cell buffer_[Capacity];
    alignas(64) std::atomic<uint32> enqueuePos_;
    alignas(64) std::atomic<uint32> dequeuePos_;
    
    static constexpr uint32 BUFFER_MASK = Capacity - 1;
    
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

public:
    MPMCQueue() : enqueuePos_(0), dequeuePos_(0) {
        for (uint32 i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }
    
    bool push(const T& item) {
        Cell* cell;
        uint32 pos = enqueuePos_.load(std::memory_order_relaxed);
        
        for (;;) {
            cell = &buffer_[pos & BUFFER_MASK];
            uint32 seq = cell->sequence.load(std::memory_order_acquire);
            int32 diff = static_cast<int32>(seq) - static_cast<int32>(pos);
            
            if (diff == 0) {
                if (enqueuePos_.compare_exchange_weak(pos, pos + 1,
                                                      std::memory_order_relaxed,
                                                      std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = enqueuePos_.load(std::memory_order_relaxed);
            }
        }
        
        cell->data = item;
        cell->sequence.store(pos + 1, std::memory_order_release);
        
        return true;
    }
    
    bool pop(T& item) {
        Cell* cell;
        uint32 pos = dequeuePos_.load(std::memory_order_relaxed);
        
        for (;;) {
            cell = &buffer_[pos & BUFFER_MASK];
            uint32 seq = cell->sequence.load(std::memory_order_acquire);
            int32 diff = static_cast<int32>(seq) - static_cast<int32>(pos + 1);
            
            if (diff == 0) {
                if (dequeuePos_.compare_exchange_weak(pos, pos + 1,
                                                      std::memory_order_relaxed,
                                                      std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeuePos_.load(std::memory_order_relaxed);
            }
        }
        
        item = cell->data;
        cell->sequence.store(pos + Capacity, std::memory_order_release);
        
        return true;
    }
    
    bool empty() const {
        uint32 pos = dequeuePos_.load(std::memory_order_relaxed);
        Cell* cell = &buffer_[pos & BUFFER_MASK];
        uint32 seq = cell->sequence.load(std::memory_order_acquire);
        int32 diff = static_cast<int32>(seq) - static_cast<int32>(pos + 1);
        return diff < 0;
    }
};

template<typename T>
class AtomicPtr {
private:
    std::atomic<T*> ptr_;
    
public:
    AtomicPtr() : ptr_(nullptr) {}
    explicit AtomicPtr(T* p) : ptr_(p) {}
    
    T* load() const {
        return ptr_.load(std::memory_order_acquire);
    }
    
    void store(T* p) {
        ptr_.store(p, std::memory_order_release);
    }
    
    bool compareExchangeStrong(T*& expected, T* desired) {
        return ptr_.compare_exchange_strong(expected, desired,
                                            std::memory_order_seq_cst,
                                            std::memory_order_seq_cst);
    }
    
    T* exchange(T* desired) {
        return ptr_.exchange(desired, std::memory_order_acq_rel);
    }
};

class SpinLock {
private:
    std::atomic_flag lock_;
    
public:
    SpinLock() : lock_(ATOMIC_FLAG_INIT) {}
    
    void lock() {
        while (lock_.test_and_set(std::memory_order_acquire)) {
            while (lock_.test(std::memory_order_relaxed)) {
                #if defined(__ARM_ARCH) || defined(__arm__) || defined(__aarch64__)
                asm volatile("yield" ::: "memory");
                #elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                asm volatile("pause" ::: "memory");
                #endif
            }
        }
    }
    
    void unlock() {
        lock_.clear(std::memory_order_release);
    }
    
    bool tryLock() {
        return !lock_.test_and_set(std::memory_order_acquire);
    }
};

template<typename T, uint32 Capacity>
class RingBuffer {
private:
    alignas(16) T buffer_[Capacity];
    volatile uint32 head_;
    volatile uint32 tail_;
    SpinLock lock_;
    
public:
    RingBuffer() : head_(0), tail_(0) {}
    
    bool push(const T& item) {
        lock_.lock();
        
        uint32 nextHead = (head_ + 1) % Capacity;
        if (nextHead == tail_) {
            lock_.unlock();
            return false;
        }
        
        buffer_[head_] = item;
        head_ = nextHead;
        
        lock_.unlock();
        return true;
    }
    
    bool pop(T& item) {
        lock_.lock();
        
        if (tail_ == head_) {
            lock_.unlock();
            return false;
        }
        
        item = buffer_[tail_];
        tail_ = (tail_ + 1) % Capacity;
        
        lock_.unlock();
        return true;
    }
    
    bool empty() const {
        return head_ == tail_;
    }
    
    uint32 size() const {
        if (head_ >= tail_) {
            return head_ - tail_;
        }
        return Capacity - tail_ + head_;
    }
};

}

#endif
