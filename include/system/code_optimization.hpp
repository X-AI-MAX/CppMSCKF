#ifndef MSCKF_SYSTEM_CODE_OPTIMIZATION_HPP
#define MSCKF_SYSTEM_CODE_OPTIMIZATION_HPP

#include "../math/types.hpp"

namespace msckf {

#if defined(__GNUC__) || defined(__clang__)
    #define MSCKF_INLINE __attribute__((always_inline)) inline
    #define MSCKF_NOINLINE __attribute__((noinline))
    #define MSCKF_HOT __attribute__((hot))
    #define MSCKF_COLD __attribute__((cold))
    #define MSCKF_FLATTEN __attribute__((flatten))
    #define MSCKF_SECTION(x) __attribute__((section(x)))
    #define MSCKF_UNUSED __attribute__((unused))
    #define MSCKF_USED __attribute__((used))
    #define MSCKF_PACKED __attribute__((packed))
    #define MSCKF_ALIGNED(x) __attribute__((aligned(x)))
    #define MSCKF_LIKELY(x) __builtin_expect(!!(x), 1)
    #define MSCKF_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define MSCKF_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define MSCKF_INLINE __forceinline
    #define MSCKF_NOINLINE __declspec(noinline)
    #define MSCKF_HOT
    #define MSCKF_COLD
    #define MSCKF_FLATTEN
    #define MSCKF_SECTION(x) __declspec(allocate(x))
    #define MSCKF_UNUSED
    #define MSCKF_USED
    #define MSCKF_PACKED
    #define MSCKF_ALIGNED(x) __declspec(align(x))
    #define MSCKF_LIKELY(x) (x)
    #define MSCKF_UNLIKELY(x) (x)
    #define MSCKF_RESTRICT __restrict
#else
    #define MSCKF_INLINE inline
    #define MSCKF_NOINLINE
    #define MSCKF_HOT
    #define MSCKF_COLD
    #define MSCKF_FLATTEN
    #define MSCKF_SECTION(x)
    #define MSCKF_UNUSED
    #define MSCKF_USED
    #define MSCKF_PACKED
    #define MSCKF_ALIGNED(x)
    #define MSCKF_LIKELY(x) (x)
    #define MSCKF_UNLIKELY(x) (x)
    #define MSCKF_RESTRICT
#endif

#define MSCKF_CRITICAL_PATH MSCKF_INLINE MSCKF_HOT
#define MSCKF_ERROR_PATH MSCKF_COLD

struct LinkerScriptConfig {
    uint32 flashOrigin;
    uint32 flashLength;
    uint32 ramOrigin;
    uint32 ramLength;
    uint32 ccmRamOrigin;
    uint32 ccmRamLength;
    
    uint32 isrVectorSize;
    uint32 textSectionAlign;
    uint32 dataSectionAlign;
    uint32 bssSectionAlign;
    uint32 stackSize;
    uint32 heapSize;
    
    bool enableMemoryRegions;
    bool enableSectionGarbage;
    
    LinkerScriptConfig() 
        : flashOrigin(0x08000000)
        , flashLength(0x00040000)
        , ramOrigin(0x20000000)
        , ramLength(0x00010000)
        , ccmRamOrigin(0x10000000)
        , ccmRamLength(0x00010000)
        , isrVectorSize(0x200)
        , textSectionAlign(4)
        , dataSectionAlign(4)
        , bssSectionAlign(4)
        , stackSize(0x2000)
        , heapSize(0x4000)
        , enableMemoryRegions(true)
        , enableSectionGarbage(true) {}
};

class LinkerScriptGenerator {
public:
    static void generateScript(const LinkerScriptConfig& config, char* buffer, uint32 bufferSize) {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "/* MSCKF Linker Script - Auto Generated */\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "/* Target: Flash=%uKB, RAM=%uKB */\n\n",
            config.flashLength / 1024, config.ramLength / 1024);
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "MEMORY\n{\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    FLASH (rx)  : ORIGIN = 0x%08X, LENGTH = %uK\n",
            config.flashOrigin, config.flashLength / 1024);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    RAM (rwx)   : ORIGIN = 0x%08X, LENGTH = %uK\n",
            config.ramOrigin, config.ramLength / 1024);
        if (config.enableMemoryRegions) {
            offset += snprintf(buffer + offset, bufferSize - offset,
                "    CCMRAM (rw) : ORIGIN = 0x%08X, LENGTH = %uK\n",
                config.ccmRamOrigin, config.ccmRamLength / 1024);
        }
        offset += snprintf(buffer + offset, bufferSize - offset, "}\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "_estack = ORIGIN(RAM) + LENGTH(RAM);\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "SECTIONS\n{\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    .isr_vector :\n    {\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(%u);\n", config.isrVectorSize);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        KEEP(*(.isr_vector))\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    } >FLASH\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    .text :\n    {\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(%u);\n", config.textSectionAlign);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.text)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.text*)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.msckf_hot)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.rodata)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.rodata*)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(4);\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        _etext = .;\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    } >FLASH\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    .data :\n    {\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(%u);\n", config.dataSectionAlign);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        _sdata = .;\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.data)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.data*)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(4);\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        _edata = .;\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    } >RAM AT> FLASH\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    .bss :\n    {\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(%u);\n", config.bssSectionAlign);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        _sbss = .;\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.bss)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.bss*)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(COMMON)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(4);\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        _ebss = .;\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    } >RAM\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    .msckf_fast (NOLOAD) :\n    {\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(8);\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        *(.msckf_fast)\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    } >CCMRAM\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    ._user_heap_stack :\n    {\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(8);\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        PROVIDE(end = .);\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        PROVIDE(_end = .);\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = . + %u;\n", config.heapSize);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = . + %u;\n", config.stackSize);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "        . = ALIGN(8);\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    } >RAM\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "}\n");
    }
};

struct CodeSizeStats {
    uint32 textSectionSize;
    uint32 dataSectionSize;
    uint32 bssSectionSize;
    uint32 totalFlashUsage;
    uint32 totalRamUsage;
    uint32 stackUsage;
    uint32 heapUsage;
    
    uint32 functionCount;
    uint32 inlinedFunctionCount;
    uint32 removedDeadCode;
    
    CodeSizeStats() 
        : textSectionSize(0)
        , dataSectionSize(0)
        , bssSectionSize(0)
        , totalFlashUsage(0)
        , totalRamUsage(0)
        , stackUsage(0)
        , heapUsage(0)
        , functionCount(0)
        , inlinedFunctionCount(0)
        , removedDeadCode(0) {}
    
    bool isWithinLimits(uint32 maxFlash, uint32 maxRam) const {
        return totalFlashUsage <= maxFlash && totalRamUsage <= maxRam;
    }
    
    float32 getFlashUtilization(uint32 maxFlash) const {
        if (maxFlash == 0) return 0;
        return static_cast<float32>(totalFlashUsage) / maxFlash * 100.0f;
    }
    
    float32 getRamUtilization(uint32 maxRam) const {
        if (maxRam == 0) return 0;
        return static_cast<float32>(totalRamUsage) / maxRam * 100.0f;
    }
};

class CodeSizeAnalyzer {
private:
    CodeSizeStats stats_;
    
    struct FunctionInfo {
        const char* name;
        uint32 size;
        bool isInlined;
        bool isHot;
        uint32 callCount;
    };
    
    static constexpr uint32 MAX_FUNCTIONS = 500;
    FunctionInfo functions_[MAX_FUNCTIONS];
    uint32 functionCount_;

public:
    CodeSizeAnalyzer() : functionCount_(0) {}
    
    void registerFunction(const char* name, uint32 size, bool isInlined, bool isHot) {
        if (functionCount_ < MAX_FUNCTIONS) {
            functions_[functionCount_].name = name;
            functions_[functionCount_].size = size;
            functions_[functionCount_].isInlined = isInlined;
            functions_[functionCount_].isHot = isHot;
            functions_[functionCount_].callCount = 0;
            functionCount_++;
            
            stats_.functionCount++;
            if (isInlined) stats_.inlinedFunctionCount++;
        }
    }
    
    void recordCall(const char* name) {
        for (uint32 i = 0; i < functionCount_; ++i) {
            if (functions_[i].name && strcmp(functions_[i].name, name) == 0) {
                functions_[i].callCount++;
                break;
            }
        }
    }
    
    void updateStats(uint32 textSize, uint32 dataSize, uint32 bssSize) {
        stats_.textSectionSize = textSize;
        stats_.dataSectionSize = dataSize;
        stats_.bssSectionSize = bssSize;
        stats_.totalFlashUsage = textSize + dataSize;
        stats_.totalRamUsage = dataSize + bssSize;
    }
    
    void analyzeDeadCode() {
        for (uint32 i = 0; i < functionCount_; ++i) {
            if (functions_[i].callCount == 0 && !functions_[i].isHot) {
                stats_.removedDeadCode++;
            }
        }
    }
    
    const CodeSizeStats& getStats() const { return stats_; }
    uint32 getFunctionCount() const { return functionCount_; }
    const FunctionInfo& getFunction(uint32 idx) const { return functions_[idx]; }
    
    void generateReport(char* buffer, uint32 bufferSize, 
                       uint32 maxFlash = 256 * 1024, uint32 maxRam = 64 * 1024) const {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "Code Size Analysis Report\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "========================\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "Memory Usage:\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "  Text Section:  %u bytes (%.1f%% of flash)\n",
            stats_.textSectionSize, stats_.getFlashUtilization(maxFlash));
        offset += snprintf(buffer + offset, bufferSize - offset,
            "  Data Section:  %u bytes\n", stats_.dataSectionSize);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "  BSS Section:   %u bytes\n", stats_.bssSectionSize);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "  Total Flash:   %u bytes (%.1f%% utilized)\n",
            stats_.totalFlashUsage, stats_.getFlashUtilization(maxFlash));
        offset += snprintf(buffer + offset, bufferSize - offset,
            "  Total RAM:     %u bytes (%.1f%% utilized)\n\n",
            stats_.totalRamUsage, stats_.getRamUtilization(maxRam));
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "Optimization Results:\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "  Total Functions:     %u\n", stats_.functionCount);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "  Inlined Functions:   %u\n", stats_.inlinedFunctionCount);
        offset += snprintf(buffer + offset, bufferSize - offset,
            "  Dead Code Removed:   %u\n\n", stats_.removedDeadCode);
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "Status: %s\n",
            stats_.isWithinLimits(maxFlash, maxRam) ? "WITHIN LIMITS" : "EXCEEDS LIMITS");
    }
};

class OptimizationConfig {
public:
    struct CompilerFlags {
        bool optimizeForSize;
        bool optimizeForSpeed;
        bool linkTimeOptimization;
        bool deadCodeElimination;
        bool functionSections;
        bool dataSections;
        bool gcSections;
        int32_t optimizationLevel;
        
        CompilerFlags() 
            : optimizeForSize(true)
            , optimizeForSpeed(false)
            , linkTimeOptimization(true)
            , deadCodeElimination(true)
            , functionSections(true)
            , dataSections(true)
            , gcSections(true)
            , optimizationLevel(2) {}
    };
    
    struct InliningConfig {
        bool enableAutoInlining;
        uint32 maxInlineSize;
        uint32 maxInlineDepth;
        bool inlineHotFunctions;
        bool noinlineColdFunctions;
        
        InliningConfig() 
            : enableAutoInlining(true)
            , maxInlineSize(200)
            , maxInlineDepth(10)
            , inlineHotFunctions(true)
            , noinlineColdFunctions(true) {}
    };

private:
    CompilerFlags compilerFlags_;
    InliningConfig inliningConfig_;
    
    uint32 maxFlashSize_;
    uint32 maxRamSize_;

public:
    OptimizationConfig() 
        : maxFlashSize_(256 * 1024)
        , maxRamSize_(64 * 1024) {}
    
    void setMaxFlashSize(uint32 size) { maxFlashSize_ = size; }
    void setMaxRamSize(uint32 size) { maxRamSize_ = size; }
    
    uint32 getMaxFlashSize() const { return maxFlashSize_; }
    uint32 getMaxRamSize() const { return maxRamSize_; }
    
    CompilerFlags& compilerFlags() { return compilerFlags_; }
    const CompilerFlags& compilerFlags() const { return compilerFlags_; }
    
    InliningConfig& inliningConfig() { return inliningConfig_; }
    const InliningConfig& inliningConfig() const { return inliningConfig_; }
    
    void generateCompilerOptions(char* buffer, uint32 bufferSize) const {
        uint32 offset = 0;
        
        if (compilerFlags_.optimizeForSize) {
            offset += snprintf(buffer + offset, bufferSize - offset, "-Os ");
        } else if (compilerFlags_.optimizeForSpeed) {
            offset += snprintf(buffer + offset, bufferSize - offset, "-O%d ",
                             compilerFlags_.optimizationLevel);
        }
        
        if (compilerFlags_.linkTimeOptimization) {
            offset += snprintf(buffer + offset, bufferSize - offset, "-flto ");
        }
        
        if (compilerFlags_.functionSections) {
            offset += snprintf(buffer + offset, bufferSize - offset, "-ffunction-sections ");
        }
        
        if (compilerFlags_.dataSections) {
            offset += snprintf(buffer + offset, bufferSize - offset, "-fdata-sections ");
        }
        
        if (compilerFlags_.gcSections) {
            offset += snprintf(buffer + offset, bufferSize - offset, "-Wl,--gc-sections ");
        }
        
        if (inliningConfig_.enableAutoInlining) {
            offset += snprintf(buffer + offset, bufferSize - offset,
                             "-finline-functions-called-once ");
            offset += snprintf(buffer + offset, bufferSize - offset,
                             "--param max-inline-insns-size=%u ",
                             inliningConfig_.maxInlineSize);
        }
    }
};

namespace CriticalPath {

MSCKF_CRITICAL_PATH float64 fastSqrt(float64 x) {
    if (x <= 0) return 0;
    float64 result = x * 0.5;
    float64 halfX = x * 0.5;
    for (int32_t i = 0; i < 4; ++i) {
        result = (result + halfX / result) * 0.5;
    }
    return result;
}

MSCKF_CRITICAL_PATH float64 fastInvSqrt(float64 x) {
    if (x <= 0) return 0;
    float64 xhalf = 0.5 * x;
    int64_t i = *(int64_t*)&x;
    i = 0x5fe6eb50c7b537a9LL - (i >> 1);
    x = *(float64*)&i;
    x = x * (1.5 - xhalf * x * x);
    x = x * (1.5 - xhalf * x * x);
    return x;
}

MSCKF_CRITICAL_PATH float32 fastSqrtF(float32 x) {
    if (x <= 0) return 0;
    float32 result = x * 0.5f;
    float32 halfX = x * 0.5f;
    for (int32_t i = 0; i < 3; ++i) {
        result = (result + halfX / result) * 0.5f;
    }
    return result;
}

MSCKF_CRITICAL_PATH float32 fastInvSqrtF(float32 x) {
    if (x <= 0) return 0;
    float32 xhalf = 0.5f * x;
    int32_t i = *(int32_t*)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float32*)&i;
    x = x * (1.5f - xhalf * x * x);
    return x;
}

MSCKF_CRITICAL_PATH int32_t fastAbs(int32_t x) {
    int32_t mask = x >> 31;
    return (x + mask) ^ mask;
}

MSCKF_CRITICAL_PATH float64 fastAbsF(float64 x) {
    return x >= 0 ? x : -x;
}

MSCKF_CRITICAL_PATH float64 fastAtan2(float64 y, float64 x) {
    float64 absY = fastAbsF(y) + 1e-10;
    float64 r, angle;
    
    if (x >= 0) {
        r = (x - absY) / (x + absY);
        angle = 0.1963 * r * r * r - 0.9817 * r + 0.785398;
    } else {
        r = (x + absY) / (absY - x);
        angle = 0.1963 * r * r * r - 0.9817 * r + 2.356194;
    }
    
    return y < 0 ? -angle : angle;
}

}

}

#endif
