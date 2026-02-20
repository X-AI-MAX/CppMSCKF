#ifndef MSCKF_TESTS_TEST_MEMORY_POOL_HPP
#define MSCKF_TESTS_TEST_MEMORY_POOL_HPP

#include "test_framework.hpp"
#include "../include/system/memory_pool.hpp"
#include "../include/system/memory_safety.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_memory_pool_default_constructor() {
    TEST_BEGIN("Memory pool default constructor");
    
    MemoryPool pool;
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_memory_pool_init() {
    TEST_BEGIN("Memory pool init");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 100;
    
    pool.init(config);
    
    TEST_ASSERT_TRUE(pool.isInitialized());
    
    TEST_END();
}

bool test_memory_pool_allocate_single() {
    TEST_BEGIN("Memory pool allocate single");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 10;
    pool.init(config);
    
    void* ptr = pool.allocate();
    
    TEST_ASSERT_TRUE(ptr != nullptr);
    
    TEST_END();
}

bool test_memory_pool_allocate_multiple() {
    TEST_BEGIN("Memory pool allocate multiple");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 10;
    pool.init(config);
    
    void* ptrs[10];
    for (uint32 i = 0; i < 10; ++i) {
        ptrs[i] = pool.allocate();
        TEST_ASSERT_TRUE(ptrs[i] != nullptr);
    }
    
    TEST_END();
}

bool test_memory_pool_allocate_exhaust() {
    TEST_BEGIN("Memory pool allocate exhaust");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 5;
    pool.init(config);
    
    void* ptrs[6];
    for (uint32 i = 0; i < 5; ++i) {
        ptrs[i] = pool.allocate();
    }
    
    ptrs[5] = pool.allocate();
    TEST_ASSERT_TRUE(ptrs[5] == nullptr);
    
    TEST_END();
}

bool test_memory_pool_deallocate() {
    TEST_BEGIN("Memory pool deallocate");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 5;
    pool.init(config);
    
    void* ptr = pool.allocate();
    pool.deallocate(ptr);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_memory_pool_reuse() {
    TEST_BEGIN("Memory pool reuse");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 5;
    pool.init(config);
    
    void* ptr1 = pool.allocate();
    pool.deallocate(ptr1);
    void* ptr2 = pool.allocate();
    
    TEST_ASSERT_TRUE(ptr1 == ptr2);
    
    TEST_END();
}

bool test_memory_pool_available_blocks() {
    TEST_BEGIN("Memory pool available blocks");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 10;
    pool.init(config);
    
    TEST_ASSERT_EQ(10u, pool.availableBlocks());
    
    pool.allocate();
    TEST_ASSERT_EQ(9u, pool.availableBlocks());
    
    TEST_END();
}

bool test_memory_pool_used_blocks() {
    TEST_BEGIN("Memory pool used blocks");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 10;
    pool.init(config);
    
    TEST_ASSERT_EQ(0u, pool.usedBlocks());
    
    pool.allocate();
    TEST_ASSERT_EQ(1u, pool.usedBlocks());
    
    TEST_END();
}

bool test_memory_pool_reset() {
    TEST_BEGIN("Memory pool reset");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 10;
    pool.init(config);
    
    for (uint32 i = 0; i < 5; ++i) {
        pool.allocate();
    }
    
    pool.reset();
    
    TEST_ASSERT_EQ(10u, pool.availableBlocks());
    TEST_ASSERT_EQ(0u, pool.usedBlocks());
    
    TEST_END();
}

bool test_memory_pool_block_size() {
    TEST_BEGIN("Memory pool block size");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 128;
    config.numBlocks = 10;
    pool.init(config);
    
    TEST_ASSERT_EQ(128u, pool.blockSize());
    
    TEST_END();
}

bool test_memory_pool_total_capacity() {
    TEST_BEGIN("Memory pool total capacity");
    
    MemoryPool pool;
    MemoryPool::Config config;
    config.blockSize = 64;
    config.numBlocks = 100;
    pool.init(config);
    
    TEST_ASSERT_EQ(100u, pool.totalCapacity());
    
    TEST_END();
}

bool test_safe_array_basic() {
    TEST_BEGIN("Safe array basic");
    
    SafeArray<uint32, 10> arr;
    
    for (uint32 i = 0; i < 10; ++i) {
        arr[i] = i;
    }
    
    for (uint32 i = 0; i < 10; ++i) {
        TEST_ASSERT_EQ(i, arr[i]);
    }
    
    TEST_END();
}

bool test_safe_array_bounds_check() {
    TEST_BEGIN("Safe array bounds check");
    
    SafeArray<uint32, 10> arr;
    
    arr[100] = 42;
    
    TEST_ASSERT_TRUE(arr[100] != 42 || arr[100] == 0);
    
    TEST_END();
}

bool test_safe_array_at() {
    TEST_BEGIN("Safe array at");
    
    SafeArray<uint32, 10> arr;
    
    arr.at(5) = 42;
    TEST_ASSERT_EQ(42u, arr.at(5));
    
    TEST_END();
}

bool test_safe_array_size() {
    TEST_BEGIN("Safe array size");
    
    SafeArray<uint32, 100> arr;
    
    TEST_ASSERT_EQ(100u, arr.size());
    
    TEST_END();
}

bool test_safe_array_canaries() {
    TEST_BEGIN("Safe array canaries");
    
    SafeArray<uint32, 10> arr;
    
    TEST_ASSERT_TRUE(arr.checkCanaries());
    
    TEST_END();
}

bool test_safe_array_is_valid() {
    TEST_BEGIN("Safe array is valid");
    
    SafeArray<uint32, 10> arr;
    
    TEST_ASSERT_TRUE(arr.isValid());
    
    TEST_END();
}

bool test_safe_pointer_basic() {
    TEST_BEGIN("Safe pointer basic");
    
    uint32 value = 42;
    SafePointer<uint32> ptr(&value, 1);
    
    TEST_ASSERT_FALSE(ptr.isNull());
    TEST_ASSERT_EQ(42u, *ptr);
    
    TEST_END();
}

bool test_safe_pointer_null() {
    TEST_BEGIN("Safe pointer null");
    
    SafePointer<uint32> ptr;
    
    TEST_ASSERT_TRUE(ptr.isNull());
    TEST_ASSERT_FALSE(ptr.isValid());
    
    TEST_END();
}

bool test_safe_pointer_bounds_check() {
    TEST_BEGIN("Safe pointer bounds check");
    
    uint32 values[5] = {1, 2, 3, 4, 5};
    SafePointer<uint32> ptr(values, 5);
    
    TEST_ASSERT_TRUE(ptr.checkBounds(4));
    TEST_ASSERT_FALSE(ptr.checkBounds(5));
    
    TEST_END();
}

bool test_safe_pointer_array_access() {
    TEST_BEGIN("Safe pointer array access");
    
    uint32 values[5] = {1, 2, 3, 4, 5};
    SafePointer<uint32> ptr(values, 5);
    
    TEST_ASSERT_EQ(1u, ptr[0]);
    TEST_ASSERT_EQ(3u, ptr[2]);
    TEST_ASSERT_EQ(5u, ptr[4]);
    
    TEST_END();
}

bool test_loop_counter_basic() {
    TEST_BEGIN("Loop counter basic");
    
    LoopCounter counter(100);
    
    for (uint32 i = 0; i < 50; ++i) {
        TEST_ASSERT_TRUE(counter.increment());
    }
    
    TEST_ASSERT_EQ(50u, counter.getCount());
    
    TEST_END();
}

bool test_loop_counter_overflow() {
    TEST_BEGIN("Loop counter overflow");
    
    LoopCounter counter(10);
    
    for (uint32 i = 0; i < 15; ++i) {
        counter.increment();
    }
    
    TEST_ASSERT_TRUE(counter.getOverflowCount() > 0);
    
    TEST_END();
}

bool test_loop_counter_reset() {
    TEST_BEGIN("Loop counter reset");
    
    LoopCounter counter(100);
    
    for (uint32 i = 0; i < 50; ++i) {
        counter.increment();
    }
    
    counter.reset();
    
    TEST_ASSERT_EQ(0u, counter.getCount());
    
    TEST_END();
}

bool test_stack_canary_basic() {
    TEST_BEGIN("Stack canary basic");
    
    StackCanary canary;
    
    TEST_ASSERT_TRUE(canary.check());
    
    TEST_END();
}

bool test_stack_canary_reset() {
    TEST_BEGIN("Stack canary reset");
    
    StackCanary canary;
    canary.reset();
    
    TEST_ASSERT_TRUE(canary.check());
    
    TEST_END();
}

bool test_stack_monitor_basic() {
    TEST_BEGIN("Stack monitor basic");
    
    StackMonitor monitor;
    StackMonitor::Config config;
    config.maxStackDepth = 4096;
    monitor.init(config);
    
    TEST_ASSERT_EQ(0u, monitor.getCurrentDepth());
    
    TEST_END();
}

bool test_stack_monitor_enter_exit() {
    TEST_BEGIN("Stack monitor enter exit");
    
    StackMonitor monitor;
    StackMonitor::Config config;
    config.maxStackDepth = 4096;
    monitor.init(config);
    
    monitor.enterFunction("test", 100);
    TEST_ASSERT_EQ(100u, monitor.getCurrentDepth());
    
    monitor.exitFunction("test", 100);
    TEST_ASSERT_EQ(0u, monitor.getCurrentDepth());
    
    TEST_END();
}

bool test_stack_monitor_overflow() {
    TEST_BEGIN("Stack monitor overflow");
    
    StackMonitor monitor;
    StackMonitor::Config config;
    config.maxStackDepth = 100;
    monitor.init(config);
    
    monitor.enterFunction("test", 200);
    
    TEST_ASSERT_TRUE(monitor.hasOverflow());
    
    TEST_END();
}

REGISTER_TEST("Memory pool default constructor", test_memory_pool_default_constructor, "MemoryPool");
REGISTER_TEST("Memory pool init", test_memory_pool_init, "MemoryPool");
REGISTER_TEST("Memory pool allocate single", test_memory_pool_allocate_single, "MemoryPool");
REGISTER_TEST("Memory pool allocate multiple", test_memory_pool_allocate_multiple, "MemoryPool");
REGISTER_TEST("Memory pool allocate exhaust", test_memory_pool_allocate_exhaust, "MemoryPool");
REGISTER_TEST("Memory pool deallocate", test_memory_pool_deallocate, "MemoryPool");
REGISTER_TEST("Memory pool reuse", test_memory_pool_reuse, "MemoryPool");
REGISTER_TEST("Memory pool available blocks", test_memory_pool_available_blocks, "MemoryPool");
REGISTER_TEST("Memory pool used blocks", test_memory_pool_used_blocks, "MemoryPool");
REGISTER_TEST("Memory pool reset", test_memory_pool_reset, "MemoryPool");
REGISTER_TEST("Memory pool block size", test_memory_pool_block_size, "MemoryPool");
REGISTER_TEST("Memory pool total capacity", test_memory_pool_total_capacity, "MemoryPool");
REGISTER_TEST("Safe array basic", test_safe_array_basic, "MemoryPool");
REGISTER_TEST("Safe array bounds check", test_safe_array_bounds_check, "MemoryPool");
REGISTER_TEST("Safe array at", test_safe_array_at, "MemoryPool");
REGISTER_TEST("Safe array size", test_safe_array_size, "MemoryPool");
REGISTER_TEST("Safe array canaries", test_safe_array_canaries, "MemoryPool");
REGISTER_TEST("Safe array is valid", test_safe_array_is_valid, "MemoryPool");
REGISTER_TEST("Safe pointer basic", test_safe_pointer_basic, "MemoryPool");
REGISTER_TEST("Safe pointer null", test_safe_pointer_null, "MemoryPool");
REGISTER_TEST("Safe pointer bounds check", test_safe_pointer_bounds_check, "MemoryPool");
REGISTER_TEST("Safe pointer array access", test_safe_pointer_array_access, "MemoryPool");
REGISTER_TEST("Loop counter basic", test_loop_counter_basic, "MemoryPool");
REGISTER_TEST("Loop counter overflow", test_loop_counter_overflow, "MemoryPool");
REGISTER_TEST("Loop counter reset", test_loop_counter_reset, "MemoryPool");
REGISTER_TEST("Stack canary basic", test_stack_canary_basic, "MemoryPool");
REGISTER_TEST("Stack canary reset", test_stack_canary_reset, "MemoryPool");
REGISTER_TEST("Stack monitor basic", test_stack_monitor_basic, "MemoryPool");
REGISTER_TEST("Stack monitor enter exit", test_stack_monitor_enter_exit, "MemoryPool");
REGISTER_TEST("Stack monitor overflow", test_stack_monitor_overflow, "MemoryPool");

}
}

#endif
