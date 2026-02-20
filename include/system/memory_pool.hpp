#ifndef MSCKF_SYSTEM_MEMORY_POOL_HPP
#define MSCKF_SYSTEM_MEMORY_POOL_HPP

#include "../math/types.hpp"
#include <cstring>

namespace msckf {

template<typename T, uint32 Capacity>
class FixedMemoryPool {
public:
    struct Block {
        alignas(16) T data;
        Block* next;
        bool inUse;
    };

private:
    alignas(16) uint8 buffer_[sizeof(Block) * Capacity];
    Block* freeList_;
    uint32 allocatedCount_;
    uint32 peakCount_;

public:
    FixedMemoryPool() 
        : freeList_(nullptr)
        , allocatedCount_(0)
        , peakCount_(0) {
        Block* blocks = reinterpret_cast<Block*>(buffer_);
        for (uint32 i = 0; i < Capacity; ++i) {
            blocks[i].inUse = false;
            blocks[i].next = (i < Capacity - 1) ? &blocks[i + 1] : nullptr;
        }
        freeList_ = &blocks[0];
    }
    
    T* allocate() {
        if (!freeList_) {
            return nullptr;
        }
        
        Block* block = freeList_;
        freeList_ = block->next;
        block->inUse = true;
        block->next = nullptr;
        
        allocatedCount_++;
        if (allocatedCount_ > peakCount_) {
            peakCount_ = allocatedCount_;
        }
        
        new (&block->data) T();
        
        return &block->data;
    }
    
    void deallocate(T* ptr) {
        if (!ptr) return;
        
        Block* block = reinterpret_cast<Block*>(
            reinterpret_cast<uint8*>(ptr) - offsetof(Block, data)
        );
        
        if (!block->inUse) return;
        
        block->data.~T();
        block->inUse = false;
        block->next = freeList_;
        freeList_ = block;
        
        allocatedCount_--;
    }
    
    uint32 capacity() const { return Capacity; }
    uint32 allocated() const { return allocatedCount_; }
    uint32 available() const { return Capacity - allocatedCount_; }
    uint32 peak() const { return peakCount_; }
    bool isFull() const { return freeList_ == nullptr; }
    bool isEmpty() const { return allocatedCount_ == 0; }
};

template<uint32 BlockSize, uint32 NumBlocks>
class BlockMemoryPool {
private:
    alignas(16) uint8 buffer_[BlockSize * NumBlocks];
    uint32 freeList_[NumBlocks];
    uint32 freeListHead_;
    uint32 allocatedCount_;
    
public:
    BlockMemoryPool() 
        : freeListHead_(0)
        , allocatedCount_(0) {
        for (uint32 i = 0; i < NumBlocks; ++i) {
            freeList_[i] = i;
        }
    }
    
    void* allocate() {
        if (freeListHead_ >= NumBlocks) {
            return nullptr;
        }
        
        uint32 blockIdx = freeList_[freeListHead_++];
        allocatedCount_++;
        
        return buffer_ + blockIdx * BlockSize;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        uint8* bytePtr = static_cast<uint8*>(ptr);
        if (bytePtr < buffer_ || bytePtr >= buffer_ + BlockSize * NumBlocks) {
            return;
        }
        
        uint32 blockIdx = (bytePtr - buffer_) / BlockSize;
        
        if (freeListHead_ > 0) {
            freeList_[--freeListHead_] = blockIdx;
            allocatedCount_--;
        }
    }
    
    uint32 capacity() const { return NumBlocks; }
    uint32 allocated() const { return allocatedCount_; }
    uint32 available() const { return NumBlocks - allocatedCount_; }
    bool isFull() const { return freeListHead_ >= NumBlocks; }
};

template<uint32 Size>
class StackAllocator {
private:
    alignas(16) uint8 buffer_[Size];
    uint32 offset_;

public:
    StackAllocator() : offset_(0) {}
    
    void* allocate(uint32 size, uint32 alignment = 16) {
        uint32 alignedOffset = (offset_ + alignment - 1) & ~(alignment - 1);
        
        if (alignedOffset + size > Size) {
            return nullptr;
        }
        
        void* ptr = buffer_ + alignedOffset;
        offset_ = alignedOffset + size;
        
        return ptr;
    }
    
    void reset() {
        offset_ = 0;
    }
    
    uint32 used() const { return offset_; }
    uint32 available() const { return Size - offset_; }
    uint32 capacity() const { return Size; }
    
    template<typename T>
    T* create() {
        void* ptr = allocate(sizeof(T), alignof(T));
        if (ptr) {
            return new (ptr) T();
        }
        return nullptr;
    }
    
    template<typename T>
    void destroy(T* ptr) {
        if (ptr) {
            ptr->~T();
        }
    }
};

template<typename T, uint32 Capacity>
class ObjectPool {
private:
    alignas(16) uint8 storage_[sizeof(T) * Capacity];
    bool inUse_[Capacity];
    uint32 allocatedCount_;

public:
    ObjectPool() 
        : allocatedCount_(0) {
        for (uint32 i = 0; i < Capacity; ++i) {
            inUse_[i] = false;
        }
    }
    
    template<typename... Args>
    T* create(Args&&... args) {
        for (uint32 i = 0; i < Capacity; ++i) {
            if (!inUse_[i]) {
                T* ptr = reinterpret_cast<T*>(storage_ + i * sizeof(T));
                new (ptr) T(static_cast<Args&&>(args)...);
                inUse_[i] = true;
                allocatedCount_++;
                return ptr;
            }
        }
        return nullptr;
    }
    
    void destroy(T* ptr) {
        if (!ptr) return;
        
        uint8* bytePtr = reinterpret_cast<uint8*>(ptr);
        uint32 idx = (bytePtr - storage_) / sizeof(T);
        
        if (idx < Capacity && inUse_[idx]) {
            ptr->~T();
            inUse_[idx] = false;
            allocatedCount_--;
        }
    }
    
    uint32 capacity() const { return Capacity; }
    uint32 allocated() const { return allocatedCount_; }
    uint32 available() const { return Capacity - allocatedCount_; }
};

}

#endif
