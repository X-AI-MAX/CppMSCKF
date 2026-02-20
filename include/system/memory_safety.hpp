#ifndef MSCKF_SYSTEM_MEMORY_SAFETY_HPP
#define MSCKF_SYSTEM_MEMORY_SAFETY_HPP

#include "../math/types.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 STACK_CANARY_VALUE = 0xDEADBEEF;
constexpr uint32 HEAP_GUARD_VALUE = 0xCAFEBABE;
constexpr uint32 MAX_ALLOCATION_SIZE = 1024 * 1024;
constexpr uint32 MAX_STACK_USAGE = 32768;

struct StackCanary {
    uint32 canaryValue;
    
    StackCanary() : canaryValue(STACK_CANARY_VALUE) {}
    
    bool check() const {
        return canaryValue == STACK_CANARY_VALUE;
    }
    
    void reset() {
        canaryValue = STACK_CANARY_VALUE;
    }
};

class StackMonitor {
public:
    struct Config {
        uint32 maxStackDepth;
        bool enableCanaryCheck;
        bool enableUsageTracking;
        
        Config() 
            : maxStackDepth(4096)
            , enableCanaryCheck(true)
            , enableUsageTracking(true) {}
    };

private:
    Config config_;
    
    uint32 currentDepth_;
    uint32 maxObservedDepth_;
    uint32 overflowCount_;
    
    StackCanary canaries_[64];
    uint32 canaryCount_;

public:
    StackMonitor() 
        : currentDepth_(0)
        , maxObservedDepth_(0)
        , overflowCount_(0)
        , canaryCount_(0) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    void enterFunction(const char* funcName, uint32 estimatedUsage) {
        currentDepth_ += estimatedUsage;
        
        if (currentDepth_ > maxObservedDepth_) {
            maxObservedDepth_ = currentDepth_;
        }
        
        if (currentDepth_ > config_.maxStackDepth) {
            overflowCount_++;
        }
        
        if (config_.enableCanaryCheck && canaryCount_ < 64) {
            canaries_[canaryCount_++].reset();
        }
    }
    
    void exitFunction(const char* funcName, uint32 estimatedUsage) {
        if (config_.enableCanaryCheck && canaryCount_ > 0) {
            canaryCount_--;
            if (!canaries_[canaryCount_].check()) {
                overflowCount_++;
            }
        }
        
        if (currentDepth_ >= estimatedUsage) {
            currentDepth_ -= estimatedUsage;
        } else {
            currentDepth_ = 0;
        }
    }
    
    bool checkAllCanaries() const {
        for (uint32 i = 0; i < canaryCount_; ++i) {
            if (!canaries_[i].check()) {
                return false;
            }
        }
        return true;
    }
    
    uint32 getCurrentDepth() const { return currentDepth_; }
    uint32 getMaxObservedDepth() const { return maxObservedDepth_; }
    uint32 getOverflowCount() const { return overflowCount_; }
    bool hasOverflow() const { return overflowCount_ > 0; }
    
    void reset() {
        currentDepth_ = 0;
        maxObservedDepth_ = 0;
        overflowCount_ = 0;
        canaryCount_ = 0;
    }
};

class SafeAllocator {
public:
    struct Stats {
        uint32 totalAllocations;
        uint32 totalDeallocations;
        uint32 currentAllocations;
        uint64 totalBytesAllocated;
        uint64 currentBytesAllocated;
        uint32 failedAllocations;
        uint32 boundaryViolations;
        
        Stats() 
            : totalAllocations(0)
            , totalDeallocations(0)
            , currentAllocations(0)
            , totalBytesAllocated(0)
            , currentBytesAllocated(0)
            , failedAllocations(0)
            , boundaryViolations(0) {}
    };

private:
    Stats stats_;
    uint32 maxAllocationSize_;
    bool enableGuardBands_;
    
    static constexpr uint32 MAX_POOL_SIZE = 65536;
    uint8 memoryPool_[MAX_POOL_SIZE];
    uint32 poolUsed_;
    bool poolAllocated_[MAX_POOL_SIZE / 16];
    uint32 allocationSizes_[MAX_POOL_SIZE / 16];
    uint32 numPoolAllocations_;

    struct AllocationHeader {
        uint32 guard;
        uint32 size;
        uint32 id;
        uint32 checksum;
    };

public:
    SafeAllocator() 
        : maxAllocationSize_(MAX_ALLOCATION_SIZE)
        , enableGuardBands_(true)
        , poolUsed_(0)
        , numPoolAllocations_(0) {
        for (uint32 i = 0; i < MAX_POOL_SIZE; ++i) {
            memoryPool_[i] = 0;
        }
        for (uint32 i = 0; i < MAX_POOL_SIZE / 16; ++i) {
            poolAllocated_[i] = false;
            allocationSizes_[i] = 0;
        }
    }
    
    void init(uint32 maxAllocSize = MAX_ALLOCATION_SIZE, bool guardBands = true) {
        maxAllocationSize_ = min(maxAllocSize, MAX_POOL_SIZE);
        enableGuardBands_ = guardBands;
        poolUsed_ = 0;
        numPoolAllocations_ = 0;
        
        for (uint32 i = 0; i < MAX_POOL_SIZE / 16; ++i) {
            poolAllocated_[i] = false;
            allocationSizes_[i] = 0;
        }
    }
    
    void* allocate(uint32 size) {
        if (size == 0 || size > maxAllocationSize_) {
            stats_.failedAllocations++;
            return nullptr;
        }
        
        uint32 totalSize = size;
        if (enableGuardBands_) {
            totalSize += sizeof(AllocationHeader) + 16;
        }
        
        totalSize = (totalSize + 15) & ~15;
        
        if (poolUsed_ + totalSize > MAX_POOL_SIZE) {
            stats_.failedAllocations++;
            return nullptr;
        }
        
        uint32 blockIdx = poolUsed_ / 16;
        uint8* ptr = &memoryPool_[poolUsed_];
        poolUsed_ += totalSize;
        
        if (blockIdx < MAX_POOL_SIZE / 16) {
            poolAllocated_[blockIdx] = true;
            allocationSizes_[blockIdx] = totalSize;
        }
        
        if (enableGuardBands_) {
            AllocationHeader* header = reinterpret_cast<AllocationHeader*>(ptr);
            header->guard = HEAP_GUARD_VALUE;
            header->size = size;
            header->id = stats_.totalAllocations;
            header->checksum = computeChecksum(size, header->id);
            
            for (uint32 i = 0; i < 16; ++i) {
                ptr[sizeof(AllocationHeader) + size + i] = 0xAA;
            }
            
            stats_.totalAllocations++;
            stats_.currentAllocations++;
            stats_.totalBytesAllocated += size;
            stats_.currentBytesAllocated += size;
            
            return ptr + sizeof(AllocationHeader);
        } else {
            stats_.totalAllocations++;
            stats_.currentAllocations++;
            stats_.totalBytesAllocated += size;
            stats_.currentBytesAllocated += size;
            
            return ptr;
        }
    }
    
    void deallocate(void* ptr) {
        if (ptr == nullptr) return;
        
        if (enableGuardBands_) {
            uint8* rawPtr = static_cast<uint8*>(ptr) - sizeof(AllocationHeader);
            AllocationHeader* header = reinterpret_cast<AllocationHeader*>(rawPtr);
            
            if (header->guard != HEAP_GUARD_VALUE) {
                stats_.boundaryViolations++;
                return;
            }
            
            uint32 expectedChecksum = computeChecksum(header->size, header->id);
            if (header->checksum != expectedChecksum) {
                stats_.boundaryViolations++;
                return;
            }
            
            for (uint32 i = 0; i < 16; ++i) {
                uint8 guard = rawPtr[sizeof(AllocationHeader) + header->size + i];
                if (guard != 0xAA) {
                    stats_.boundaryViolations++;
                    break;
                }
            }
            
            stats_.currentAllocations--;
            stats_.currentBytesAllocated -= header->size;
            stats_.totalDeallocations++;
        } else {
            stats_.currentAllocations--;
            stats_.totalDeallocations++;
        }
    }
    
    bool checkBounds(void* ptr, uint32 offset, uint32 accessSize) {
        if (ptr == nullptr) return false;
        
        if (enableGuardBands_) {
            uint8* rawPtr = static_cast<uint8*>(ptr) - sizeof(AllocationHeader);
            AllocationHeader* header = reinterpret_cast<AllocationHeader*>(rawPtr);
            
            if (header->guard != HEAP_GUARD_VALUE) {
                return false;
            }
            
            if (offset + accessSize > header->size) {
                stats_.boundaryViolations++;
                return false;
            }
        }
        
        return true;
    }
    
    const Stats& getStats() const { return stats_; }
    
    void resetStats() {
        stats_ = Stats();
        poolUsed_ = 0;
        numPoolAllocations_ = 0;
        for (uint32 i = 0; i < MAX_POOL_SIZE / 16; ++i) {
            poolAllocated_[i] = false;
            allocationSizes_[i] = 0;
        }
    }
    
    void resetPool() {
        poolUsed_ = 0;
        numPoolAllocations_ = 0;
        for (uint32 i = 0; i < MAX_POOL_SIZE / 16; ++i) {
            poolAllocated_[i] = false;
            allocationSizes_[i] = 0;
        }
    }

private:
    uint32 computeChecksum(uint32 size, uint32 id) {
        return size ^ id ^ HEAP_GUARD_VALUE;
    }
};

template<typename T, uint32 N>
class SafeArray {
private:
    T data_[N];
    uint32 canaryStart_;
    uint32 canaryEnd_;
    bool initialized_;

public:
    SafeArray() : canaryStart_(STACK_CANARY_VALUE), canaryEnd_(STACK_CANARY_VALUE), initialized_(false) {
        for (uint32 i = 0; i < N; ++i) {
            data_[i] = T();
        }
        initialized_ = true;
    }
    
    T& operator[](uint32 index) {
        if (index >= N) {
            static T dummy;
            return dummy;
        }
        return data_[index];
    }
    
    const T& operator[](uint32 index) const {
        if (index >= N) {
            static T dummy;
            return dummy;
        }
        return data_[index];
    }
    
    T& at(uint32 index) {
        if (index >= N) {
            static T dummy;
            return dummy;
        }
        return data_[index];
    }
    
    const T& at(uint32 index) const {
        if (index >= N) {
            static T dummy;
            return dummy;
        }
        return data_[index];
    }
    
    bool checkCanaries() const {
        return canaryStart_ == STACK_CANARY_VALUE && 
               canaryEnd_ == STACK_CANARY_VALUE;
    }
    
    uint32 size() const { return N; }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    bool isValid() const { return initialized_ && checkCanaries(); }
};

template<typename T>
class SafePointer {
private:
    T* ptr_;
    uint32 size_;
    uint32 guard_;

public:
    SafePointer() : ptr_(nullptr), size_(0), guard_(HEAP_GUARD_VALUE) {}
    
    SafePointer(T* ptr, uint32 size) 
        : ptr_(ptr), size_(size), guard_(HEAP_GUARD_VALUE) {}
    
    void set(T* ptr, uint32 size) {
        ptr_ = ptr;
        size_ = size;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    
    T& operator*() { 
        if (ptr_ == nullptr) {
            static T dummy;
            return dummy;
        }
        return *ptr_; 
    }
    
    const T& operator*() const { 
        if (ptr_ == nullptr) {
            static T dummy;
            return dummy;
        }
        return *ptr_; 
    }
    
    T* operator->() { return ptr_; }
    const T* operator->() const { return ptr_; }
    
    T& operator[](uint32 index) {
        if (ptr_ == nullptr || index >= size_) {
            static T dummy;
            return dummy;
        }
        return ptr_[index];
    }
    
    const T& operator[](uint32 index) const {
        if (ptr_ == nullptr || index >= size_) {
            static T dummy;
            return dummy;
        }
        return ptr_[index];
    }
    
    bool isNull() const { return ptr_ == nullptr; }
    bool isValid() const { return ptr_ != nullptr && guard_ == HEAP_GUARD_VALUE; }
    uint32 size() const { return size_; }
    
    bool checkBounds(uint32 index) const {
        return ptr_ != nullptr && index < size_;
    }
};

class LoopCounter {
private:
    uint32 count_;
    uint32 maxIterations_;
    uint32 overflowCount_;

public:
    explicit LoopCounter(uint32 maxIter = MAX_ITERATION_COUNT) 
        : count_(0), maxIterations_(maxIter), overflowCount_(0) {}
    
    void reset() { count_ = 0; }
    
    bool increment() {
        count_++;
        if (count_ > maxIterations_) {
            overflowCount_++;
            return false;
        }
        return true;
    }
    
    bool check() const {
        return count_ <= maxIterations_;
    }
    
    uint32 getCount() const { return count_; }
    uint32 getOverflowCount() const { return overflowCount_; }
};

#define LOOP_LIMIT_CHECK(counter, maxIter) \
    LoopCounter counter(maxIter); \
    for (; counter.check(); counter.increment())

#define SAFE_ARRAY_ACCESS(arr, idx, size) \
    ((idx) < (size) ? (arr)[(idx)] : (arr)[0])

#define SAFE_MATRIX_ACCESS(mat, row, col, rows, cols) \
    (((row) < (rows) && (col) < (cols)) ? (mat)((row), (col)) : 0)

#define POINTER_CHECK(ptr) \
    if ((ptr) == nullptr) { return; }

#define POINTER_CHECK_RET(ptr, retval) \
    if ((ptr) == nullptr) { return (retval); }

#define DIVISION_CHECK(divisor) \
    if (abs(divisor) < 1e-12) { return 0; }

#define DIVISION_CHECK_RET(divisor, retval) \
    if (abs(divisor) < 1e-12) { return (retval); }

#define SQRT_CHECK(value) \
    if ((value) < 0) { return 0; }

#define SQRT_CHECK_RET(value, retval) \
    if ((value) < 0) { return (retval); }

class MemorySafetyMonitor {
public:
    struct Config {
        bool enableStackMonitoring;
        bool enableHeapMonitoring;
        bool enableLoopMonitoring;
        uint32 maxStackDepth;
        uint32 maxAllocationSize;
        uint32 maxLoopIterations;
        
        Config() 
            : enableStackMonitoring(true)
            , enableHeapMonitoring(true)
            , enableLoopMonitoring(true)
            , maxStackDepth(4096)
            , maxAllocationSize(MAX_ALLOCATION_SIZE)
            , maxLoopIterations(MAX_ITERATION_COUNT) {}
    };

private:
    Config config_;
    StackMonitor stackMonitor_;
    SafeAllocator allocator_;
    
    uint32 totalViolations_;
    uint32 stackViolations_;
    uint32 heapViolations_;
    uint32 loopViolations_;

public:
    MemorySafetyMonitor() 
        : totalViolations_(0)
        , stackViolations_(0)
        , heapViolations_(0)
        , loopViolations_(0) {}
    
    void init(const Config& config) {
        config_ = config;
        
        StackMonitor::Config stackConfig;
        stackConfig.maxStackDepth = config.maxStackDepth;
        stackMonitor_.init(stackConfig);
        
        allocator_.init(config.maxAllocationSize);
    }
    
    void* safeAllocate(uint32 size) {
        void* ptr = allocator_.allocate(size);
        if (ptr == nullptr) {
            heapViolations_++;
            totalViolations_++;
        }
        return ptr;
    }
    
    void safeDeallocate(void* ptr) {
        allocator_.deallocate(ptr);
    }
    
    bool checkArrayBounds(uint32 index, uint32 size) {
        if (index >= size) {
            heapViolations_++;
            totalViolations_++;
            return false;
        }
        return true;
    }
    
    bool checkLoopIteration(uint32 current, uint32 max) {
        if (current > max) {
            loopViolations_++;
            totalViolations_++;
            return false;
        }
        return true;
    }
    
    void reportStackViolation() {
        stackViolations_++;
        totalViolations_++;
    }
    
    uint32 getTotalViolations() const { return totalViolations_; }
    uint32 getStackViolations() const { return stackViolations_; }
    uint32 getHeapViolations() const { return heapViolations_; }
    uint32 getLoopViolations() const { return loopViolations_; }
    
    const SafeAllocator::Stats& getAllocatorStats() const {
        return allocator_.getStats();
    }
    
    void reset() {
        totalViolations_ = 0;
        stackViolations_ = 0;
        heapViolations_ = 0;
        loopViolations_ = 0;
        stackMonitor_.reset();
        allocator_.resetStats();
    }
};

}

#endif
