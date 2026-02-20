#ifndef MSCKF_TESTS_TEST_LOCKFREE_HPP
#define MSCKF_TESTS_TEST_LOCKFREE_HPP

#include "test_framework.hpp"
#include "../include/system/lockfree_queue.hpp"
#include "../include/math/types.hpp"
#include <thread>
#include <vector>
#include <atomic>

namespace msckf {
namespace test {

bool test_spsc_queue_empty() {
    TEST_BEGIN("SPSC queue empty");
    
    SPSCQueue<uint32, 10> queue;
    
    TEST_ASSERT_TRUE(queue.empty());
    TEST_ASSERT_FALSE(queue.full());
    TEST_ASSERT_EQ(0u, queue.size());
    
    TEST_END();
}

bool test_spsc_queue_push_pop() {
    TEST_BEGIN("SPSC queue push pop");
    
    SPSCQueue<uint32, 10> queue;
    
    uint32 value = 42;
    bool pushed = queue.push(value);
    
    TEST_ASSERT_TRUE(pushed);
    TEST_ASSERT_FALSE(queue.empty());
    TEST_ASSERT_EQ(1u, queue.size());
    
    uint32 popped_value = 0;
    bool popped = queue.pop(popped_value);
    
    TEST_ASSERT_TRUE(popped);
    TEST_ASSERT_EQ(42u, popped_value);
    TEST_ASSERT_TRUE(queue.empty());
    
    TEST_END();
}

bool test_spsc_queue_full() {
    TEST_BEGIN("SPSC queue full");
    
    SPSCQueue<uint32, 5> queue;
    
    for (uint32 i = 0; i < 4; ++i) {
        bool pushed = queue.push(i);
        TEST_ASSERT_TRUE(pushed);
    }
    
    TEST_ASSERT_TRUE(queue.full());
    
    bool pushed = queue.push(100);
    TEST_ASSERT_FALSE(pushed);
    
    TEST_END();
}

bool test_spsc_queue_fifo_order() {
    TEST_BEGIN("SPSC queue FIFO order");
    
    SPSCQueue<uint32, 100> queue;
    
    for (uint32 i = 0; i < 50; ++i) {
        queue.push(i);
    }
    
    for (uint32 i = 0; i < 50; ++i) {
        uint32 value = 0;
        queue.pop(value);
        TEST_ASSERT_EQ(i, value);
    }
    
    TEST_END();
}

bool test_spsc_queue_peek() {
    TEST_BEGIN("SPSC queue peek");
    
    SPSCQueue<uint32, 10> queue;
    
    queue.push(123);
    
    uint32 value = 0;
    bool peeked = queue.peek(value);
    
    TEST_ASSERT_TRUE(peeked);
    TEST_ASSERT_EQ(123u, value);
    TEST_ASSERT_FALSE(queue.empty());
    
    TEST_END();
}

bool test_spsc_queue_clear() {
    TEST_BEGIN("SPSC queue clear");
    
    SPSCQueue<uint32, 10> queue;
    
    for (uint32 i = 0; i < 5; ++i) {
        queue.push(i);
    }
    
    queue.clear();
    
    TEST_ASSERT_TRUE(queue.empty());
    TEST_ASSERT_EQ(0u, queue.size());
    
    TEST_END();
}

bool test_spsc_queue_capacity() {
    TEST_BEGIN("SPSC queue capacity");
    
    SPSCQueue<uint32, 10> queue;
    
    TEST_ASSERT_EQ(9u, queue.capacity());
    
    TEST_END();
}

bool test_mpmc_queue_basic() {
    TEST_BEGIN("MPMC queue basic");
    
    MPMCQueue<uint32, 16> queue;
    
    TEST_ASSERT_TRUE(queue.empty());
    
    bool pushed = queue.push(42);
    TEST_ASSERT_TRUE(pushed);
    
    uint32 value = 0;
    bool popped = queue.pop(value);
    TEST_ASSERT_TRUE(popped);
    TEST_ASSERT_EQ(42u, value);
    
    TEST_END();
}

bool test_mpmc_queue_multiple_push_pop() {
    TEST_BEGIN("MPMC queue multiple push pop");
    
    MPMCQueue<uint32, 64> queue;
    
    for (uint32 i = 0; i < 32; ++i) {
        queue.push(i);
    }
    
    for (uint32 i = 0; i < 32; ++i) {
        uint32 value = 0;
        bool popped = queue.pop(value);
        TEST_ASSERT_TRUE(popped);
    }
    
    TEST_ASSERT_TRUE(queue.empty());
    
    TEST_END();
}

bool test_spin_lock_basic() {
    TEST_BEGIN("Spin lock basic");
    
    SpinLock lock;
    
    lock.lock();
    lock.unlock();
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_spin_lock_try_lock() {
    TEST_BEGIN("Spin lock try lock");
    
    SpinLock lock;
    
    bool acquired = lock.tryLock();
    TEST_ASSERT_TRUE(acquired);
    
    lock.unlock();
    
    TEST_END();
}

bool test_spin_lock_contention() {
    TEST_BEGIN("Spin lock contention");
    
    SpinLock lock;
    uint32 counter = 0;
    const uint32 iterations = 1000;
    
    std::vector<std::thread> threads;
    for (uint32 t = 0; t < 4; ++t) {
        threads.emplace_back([&lock, &counter, iterations]() {
            for (uint32 i = 0; i < iterations; ++i) {
                lock.lock();
                counter++;
                lock.unlock();
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    TEST_ASSERT_EQ(4u * iterations, counter);
    
    TEST_END();
}

bool test_ring_buffer_basic() {
    TEST_BEGIN("Ring buffer basic");
    
    RingBuffer<uint32, 10> buffer;
    
    TEST_ASSERT_TRUE(buffer.empty());
    
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    
    TEST_ASSERT_EQ(3u, buffer.size());
    
    uint32 value = 0;
    buffer.pop(value);
    TEST_ASSERT_EQ(1u, value);
    
    TEST_END();
}

bool test_ring_buffer_overflow() {
    TEST_BEGIN("Ring buffer overflow");
    
    RingBuffer<uint32, 5> buffer;
    
    for (uint32 i = 0; i < 10; ++i) {
        buffer.push(i);
    }
    
    TEST_ASSERT_TRUE(buffer.size() <= 4);
    
    TEST_END();
}

bool test_atomic_ptr_basic() {
    TEST_BEGIN("Atomic ptr basic");
    
    AtomicPtr<uint32> ptr;
    
    TEST_ASSERT_TRUE(ptr.isNull());
    
    uint32 value = 42;
    ptr.store(&value);
    
    TEST_ASSERT_FALSE(ptr.isNull());
    TEST_ASSERT_EQ(42u, *ptr.load());
    
    TEST_END();
}

bool test_atomic_ptr_exchange() {
    TEST_BEGIN("Atomic ptr exchange");
    
    AtomicPtr<uint32> ptr;
    
    uint32 value1 = 42;
    uint32 value2 = 100;
    
    ptr.store(&value1);
    
    uint32* old = ptr.exchange(&value2);
    
    TEST_ASSERT_EQ(&value1, old);
    TEST_ASSERT_EQ(&value2, ptr.load());
    
    TEST_END();
}

bool test_spsc_queue_multithread() {
    TEST_BEGIN("SPSC queue multithread");
    
    SPSCQueue<uint32, 1000> queue;
    const uint32 numItems = 10000;
    std::atomic<uint32> consumedSum{0};
    std::atomic<uint32> producedSum{0};
    
    std::thread producer([&queue, numItems, &producedSum]() {
        for (uint32 i = 1; i <= numItems; ++i) {
            while (!queue.push(i)) {}
            producedSum += i;
        }
    });
    
    std::thread consumer([&queue, numItems, &consumedSum]() {
        uint32 count = 0;
        while (count < numItems) {
            uint32 value = 0;
            if (queue.pop(value)) {
                consumedSum += value;
                count++;
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    TEST_ASSERT_EQ(producedSum.load(), consumedSum.load());
    
    TEST_END();
}

REGISTER_TEST("SPSC queue empty", test_spsc_queue_empty, "Lockfree");
REGISTER_TEST("SPSC queue push pop", test_spsc_queue_push_pop, "Lockfree");
REGISTER_TEST("SPSC queue full", test_spsc_queue_full, "Lockfree");
REGISTER_TEST("SPSC queue FIFO order", test_spsc_queue_fifo_order, "Lockfree");
REGISTER_TEST("SPSC queue peek", test_spsc_queue_peek, "Lockfree");
REGISTER_TEST("SPSC queue clear", test_spsc_queue_clear, "Lockfree");
REGISTER_TEST("SPSC queue capacity", test_spsc_queue_capacity, "Lockfree");
REGISTER_TEST("MPMC queue basic", test_mpmc_queue_basic, "Lockfree");
REGISTER_TEST("MPMC queue multiple push pop", test_mpmc_queue_multiple_push_pop, "Lockfree");
REGISTER_TEST("Spin lock basic", test_spin_lock_basic, "Lockfree");
REGISTER_TEST("Spin lock try lock", test_spin_lock_try_lock, "Lockfree");
REGISTER_TEST("Spin lock contention", test_spin_lock_contention, "Lockfree");
REGISTER_TEST("Ring buffer basic", test_ring_buffer_basic, "Lockfree");
REGISTER_TEST("Ring buffer overflow", test_ring_buffer_overflow, "Lockfree");
REGISTER_TEST("Atomic ptr basic", test_atomic_ptr_basic, "Lockfree");
REGISTER_TEST("Atomic ptr exchange", test_atomic_ptr_exchange, "Lockfree");
REGISTER_TEST("SPSC queue multithread", test_spsc_queue_multithread, "Lockfree");

}
}

#endif
