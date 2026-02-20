#ifndef MSCKF_SYSTEM_PERFORMANCE_MONITOR_HPP
#define MSCKF_SYSTEM_PERFORMANCE_MONITOR_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"

namespace msckf {

struct WCETStats {
    float64 minTime;
    float64 maxTime;
    float64 avgTime;
    float64 p99Time;
    float64 p999Time;
    uint32 sampleCount;
    
    WCETStats() 
        : minTime(1e10)
        , maxTime(0)
        , avgTime(0)
        , p99Time(0)
        , p999Time(0)
        , sampleCount(0) {}
};

class ExecutionTimer {
private:
    uint64 startTime_;
    uint64 endTime_;
    bool running_;

public:
    ExecutionTimer() : startTime_(0), endTime_(0), running_(false) {}
    
    void start() {
        startTime_ = getCycleCount();
        running_ = true;
    }
    
    void stop() {
        endTime_ = getCycleCount();
        running_ = false;
    }
    
    float64 getElapsedUs() const {
        uint64 elapsed = running_ ? (getCycleCount() - startTime_) : (endTime_ - startTime_);
        return static_cast<float64>(elapsed) / getClockFrequency() * 1e6;
    }
    
    float64 getElapsedMs() const {
        return getElapsedUs() / 1000.0;
    }
    
    static uint64 getCycleCount() {
        return 0;
    }
    
    static float64 getClockFrequency() {
        return 1e9;
    }
};

class PerformanceCounter {
public:
    static constexpr uint32 MAX_SAMPLES = 10000;

private:
    float64 samples_[MAX_SAMPLES];
    uint32 sampleIndex_;
    uint32 sampleCount_;
    
    float64 totalTime_;
    float64 minTime_;
    float64 maxTime_;
    
    const char* name_;
    float64 wcetLimit_;

public:
    PerformanceCounter() 
        : sampleIndex_(0)
        , sampleCount_(0)
        , totalTime_(0)
        , minTime_(1e10)
        , maxTime_(0)
        , name_(nullptr)
        , wcetLimit_(1000.0) {}
    
    void init(const char* name, float64 wcetLimitUs) {
        name_ = name;
        wcetLimit_ = wcetLimitUs;
        reset();
    }
    
    void record(float64 timeUs) {
        if (sampleCount_ < MAX_SAMPLES) {
            samples_[sampleIndex_] = timeUs;
            sampleIndex_ = (sampleIndex_ + 1) % MAX_SAMPLES;
            sampleCount_++;
        }
        
        totalTime_ += timeUs;
        if (timeUs < minTime_) minTime_ = timeUs;
        if (timeUs > maxTime_) maxTime_ = timeUs;
    }
    
    WCETStats computeStats() const {
        WCETStats stats;
        stats.minTime = minTime_;
        stats.maxTime = maxTime_;
        stats.sampleCount = sampleCount_;
        
        if (sampleCount_ > 0) {
            stats.avgTime = totalTime_ / sampleCount_;
        }
        
        if (sampleCount_ > 10) {
            float64 sorted[MAX_SAMPLES];
            uint32 n = min(sampleCount_, MAX_SAMPLES);
            
            for (uint32 i = 0; i < n; ++i) {
                sorted[i] = samples_[i];
            }
            
            for (uint32 i = 0; i < n - 1; ++i) {
                for (uint32 j = i + 1; j < n; ++j) {
                    if (sorted[j] < sorted[i]) {
                        float64 tmp = sorted[i];
                        sorted[i] = sorted[j];
                        sorted[j] = tmp;
                    }
                }
            }
            
            uint32 p99Idx = static_cast<uint32>(n * 0.99);
            uint32 p999Idx = static_cast<uint32>(n * 0.999);
            
            stats.p99Time = sorted[p99Idx];
            stats.p999Time = sorted[p999Idx];
        }
        
        return stats;
    }
    
    bool isWCETExceeded() const {
        return maxTime_ > wcetLimit_;
    }
    
    float64 getWCETUtilization() const {
        if (wcetLimit_ <= 0) return 0;
        return maxTime_ / wcetLimit_ * 100.0;
    }
    
    const char* getName() const { return name_; }
    float64 getWCETLimit() const { return wcetLimit_; }
    float64 getMaxTime() const { return maxTime_; }
    float64 getAvgTime() const { return sampleCount_ > 0 ? totalTime_ / sampleCount_ : 0; }
    uint32 getSampleCount() const { return sampleCount_; }
    
    void reset() {
        sampleIndex_ = 0;
        sampleCount_ = 0;
        totalTime_ = 0;
        minTime_ = 1e10;
        maxTime_ = 0;
    }
};

class PerformanceMonitor {
public:
    enum class CounterId {
        IMU_PROPAGATION = 0,
        VISUAL_UPDATE = 1,
        FEATURE_TRACKING = 2,
        FEATURE_DETECTION = 3,
        MARGINALIZATION = 4,
        TRIANGULATION = 5,
        COVARIANCE_UPDATE = 6,
        STATE_INJECTION = 7,
        IMAGE_UNDISTORTION = 8,
        TOTAL_FRAME = 9,
        NUM_COUNTERS = 10
    };

private:
    PerformanceCounter counters_[static_cast<uint32>(CounterId::NUM_COUNTERS)];
    ExecutionTimer timer_;
    
    uint64 frameCount_;
    uint64 totalExecutionTime_;
    uint64 wcetViolations_;
    
    float64 targetFrameTime_;
    float64 maxFrameTime_;

public:
    PerformanceMonitor() 
        : frameCount_(0)
        , totalExecutionTime_(0)
        , wcetViolations_(0)
        , targetFrameTime_(33333.0)
        , maxFrameTime_(50000.0) {
        
        counters_[static_cast<uint32>(CounterId::IMU_PROPAGATION)].init("IMU_Propagation", 1000.0);
        counters_[static_cast<uint32>(CounterId::VISUAL_UPDATE)].init("Visual_Update", 20000.0);
        counters_[static_cast<uint32>(CounterId::FEATURE_TRACKING)].init("Feature_Tracking", 5000.0);
        counters_[static_cast<uint32>(CounterId::FEATURE_DETECTION)].init("Feature_Detection", 3000.0);
        counters_[static_cast<uint32>(CounterId::MARGINALIZATION)].init("Marginalization", 2000.0);
        counters_[static_cast<uint32>(CounterId::TRIANGULATION)].init("Triangulation", 1000.0);
        counters_[static_cast<uint32>(CounterId::COVARIANCE_UPDATE)].init("Covariance_Update", 5000.0);
        counters_[static_cast<uint32>(CounterId::STATE_INJECTION)].init("State_Injection", 100.0);
        counters_[static_cast<uint32>(CounterId::IMAGE_UNDISTORTION)].init("Image_Undistortion", 2000.0);
        counters_[static_cast<uint32>(CounterId::TOTAL_FRAME)].init("Total_Frame", 33333.0);
    }
    
    void startTimer() {
        timer_.start();
    }
    
    float64 stopTimer(CounterId id) {
        timer_.stop();
        float64 elapsed = timer_.getElapsedUs();
        counters_[static_cast<uint32>(id)].record(elapsed);
        return elapsed;
    }
    
    void recordTime(CounterId id, float64 timeUs) {
        counters_[static_cast<uint32>(id)].record(timeUs);
        
        if (timeUs > counters_[static_cast<uint32>(id)].getWCETLimit()) {
            wcetViolations_++;
        }
    }
    
    void beginFrame() {
        startTimer();
    }
    
    void endFrame() {
        float64 frameTime = stopTimer(CounterId::TOTAL_FRAME);
        frameCount_++;
        totalExecutionTime_ += static_cast<uint64>(frameTime);
        
        if (frameTime > maxFrameTime_) {
            wcetViolations_++;
        }
    }
    
    WCETStats getStats(CounterId id) const {
        return counters_[static_cast<uint32>(id)].computeStats();
    }
    
    const PerformanceCounter& getCounter(CounterId id) const {
        return counters_[static_cast<uint32>(id)];
    }
    
    float64 getFrameRate() const {
        if (totalExecutionTime_ == 0) return 0;
        return 1e6 * frameCount_ / totalExecutionTime_;
    }
    
    float64 getAvgFrameTime() const {
        if (frameCount_ == 0) return 0;
        return static_cast<float64>(totalExecutionTime_) / frameCount_;
    }
    
    uint64 getWCETViolations() const { return wcetViolations_; }
    uint64 getFrameCount() const { return frameCount_; }
    
    bool allWCETMet() const {
        for (uint32 i = 0; i < static_cast<uint32>(CounterId::NUM_COUNTERS); ++i) {
            if (counters_[i].isWCETExceeded()) {
                return false;
            }
        }
        return true;
    }
    
    void reset() {
        for (uint32 i = 0; i < static_cast<uint32>(CounterId::NUM_COUNTERS); ++i) {
            counters_[i].reset();
        }
        frameCount_ = 0;
        totalExecutionTime_ = 0;
        wcetViolations_ = 0;
    }
};

class ResourceMonitor {
public:
    struct MemoryStats {
        uint64 totalAllocated;
        uint64 currentUsed;
        uint64 peakUsed;
        uint32 allocationCount;
        uint32 deallocationCount;
        
        MemoryStats() 
            : totalAllocated(0)
            , currentUsed(0)
            , peakUsed(0)
            , allocationCount(0)
            , deallocationCount(0) {}
    };
    
    struct CpuStats {
        float64 utilization;
        uint64 totalCycles;
        uint64 idleCycles;
        uint32 contextSwitches;
        
        CpuStats() 
            : utilization(0)
            , totalCycles(0)
            , idleCycles(0)
            , contextSwitches(0) {}
    };

private:
    MemoryStats memoryStats_;
    CpuStats cpuStats_;
    
    uint64 maxMemoryLimit_;
    float64 maxCpuUtilization_;
    
    uint32 memoryWarnings_;
    uint32 cpuWarnings_;

public:
    ResourceMonitor() 
        : maxMemoryLimit_(64 * 1024)
        , maxCpuUtilization_(80.0)
        , memoryWarnings_(0)
        , cpuWarnings_(0) {}
    
    void setLimits(uint64 maxMemoryKB, float64 maxCpuPercent) {
        maxMemoryLimit_ = maxMemoryKB;
        maxCpuUtilization_ = maxCpuPercent;
    }
    
    void recordAllocation(uint64 size) {
        memoryStats_.totalAllocated += size;
        memoryStats_.currentUsed += size;
        memoryStats_.allocationCount++;
        
        if (memoryStats_.currentUsed > memoryStats_.peakUsed) {
            memoryStats_.peakUsed = memoryStats_.currentUsed;
        }
        
        if (memoryStats_.currentUsed > maxMemoryLimit_ * 1024) {
            memoryWarnings_++;
        }
    }
    
    void recordDeallocation(uint64 size) {
        if (memoryStats_.currentUsed >= size) {
            memoryStats_.currentUsed -= size;
        } else {
            memoryStats_.currentUsed = 0;
        }
        memoryStats_.deallocationCount++;
    }
    
    void updateCpuStats(float64 utilization) {
        cpuStats_.utilization = utilization;
        cpuStats_.totalCycles++;
        
        if (utilization < 5.0) {
            cpuStats_.idleCycles++;
        }
        
        if (utilization > maxCpuUtilization_) {
            cpuWarnings_++;
        }
    }
    
    const MemoryStats& getMemoryStats() const { return memoryStats_; }
    const CpuStats& getCpuStats() const { return cpuStats_; }
    
    uint64 getMaxMemoryLimit() const { return maxMemoryLimit_; }
    float64 getMaxCpuUtilization() const { return maxCpuUtilization_; }
    
    uint32 getMemoryWarnings() const { return memoryWarnings_; }
    uint32 getCpuWarnings() const { return cpuWarnings_; }
    
    float64 getMemoryUtilization() const {
        if (maxMemoryLimit_ == 0) return 0;
        return static_cast<float64>(memoryStats_.currentUsed) / (maxMemoryLimit_ * 1024) * 100.0;
    }
    
    bool isMemoryWithinLimit() const {
        return memoryStats_.currentUsed <= maxMemoryLimit_ * 1024;
    }
    
    bool isCpuWithinLimit() const {
        return cpuStats_.utilization <= maxCpuUtilization_;
    }
    
    void reset() {
        memoryStats_ = MemoryStats();
        cpuStats_ = CpuStats();
        memoryWarnings_ = 0;
        cpuWarnings_ = 0;
    }
};

class AdaptiveScheduler {
public:
    struct Config {
        float64 targetFrameTime;
        float64 minFrameTime;
        float64 maxFrameTime;
        float64 adjustmentRate;
        uint32 historySize;
        
        Config() 
            : targetFrameTime(33333.0)
            , minFrameTime(20000.0)
            , maxFrameTime(50000.0)
            , adjustmentRate(0.1)
            , historySize(30) {}
    };

private:
    Config config_;
    
    float64 frameTimes_[30];
    uint32 frameIndex_;
    uint32 frameCount_;
    
    float64 currentTargetTime_;
    uint32 skipFrameCounter_;
    uint32 skipFrameInterval_;
    
    bool highLoadMode_;
    uint32 consecutiveHighLoad_;

public:
    AdaptiveScheduler() 
        : frameIndex_(0)
        , frameCount_(0)
        , currentTargetTime_(33333.0)
        , skipFrameCounter_(0)
        , skipFrameInterval_(1)
        , highLoadMode_(false)
        , consecutiveHighLoad_(0) {}
    
    void init(const Config& config) {
        config_ = config;
        currentTargetTime_ = config.targetFrameTime;
        
        for (uint32 i = 0; i < config.historySize; ++i) {
            frameTimes_[i] = config.targetFrameTime;
        }
    }
    
    void recordFrameTime(float64 timeUs) {
        frameTimes_[frameIndex_] = timeUs;
        frameIndex_ = (frameIndex_ + 1) % config_.historySize;
        frameCount_++;
        
        float64 avgTime = computeAverageFrameTime();
        
        if (avgTime > config_.targetFrameTime * 1.2) {
            consecutiveHighLoad_++;
            
            if (consecutiveHighLoad_ > 10 && !highLoadMode_) {
                enterHighLoadMode();
            }
        } else {
            consecutiveHighLoad_ = 0;
            
            if (highLoadMode_ && avgTime < config_.targetFrameTime * 0.8) {
                exitHighLoadMode();
            }
        }
    }
    
    float64 computeAverageFrameTime() const {
        float64 sum = 0;
        uint32 n = min(frameCount_, config_.historySize);
        
        for (uint32 i = 0; i < n; ++i) {
            sum += frameTimes_[i];
        }
        
        return n > 0 ? sum / n : config_.targetFrameTime;
    }
    
    void enterHighLoadMode() {
        highLoadMode_ = true;
        skipFrameInterval_ = 2;
    }
    
    void exitHighLoadMode() {
        highLoadMode_ = false;
        skipFrameInterval_ = 1;
    }
    
    bool shouldProcessFrame() {
        skipFrameCounter_++;
        
        if (skipFrameCounter_ >= skipFrameInterval_) {
            skipFrameCounter_ = 0;
            return true;
        }
        return false;
    }
    
    uint32 getVisualUpdateRate() const {
        if (highLoadMode_) {
            return 15;
        }
        return 30;
    }
    
    uint32 getFeatureDetectionRate() const {
        if (highLoadMode_) {
            return 10;
        }
        return 30;
    }
    
    bool isInHighLoadMode() const { return highLoadMode_; }
    float64 getCurrentTargetTime() const { return currentTargetTime_; }
    uint32 getSkipInterval() const { return skipFrameInterval_; }
    
    void reset() {
        frameIndex_ = 0;
        frameCount_ = 0;
        currentTargetTime_ = config_.targetFrameTime;
        skipFrameCounter_ = 0;
        skipFrameInterval_ = 1;
        highLoadMode_ = false;
        consecutiveHighLoad_ = 0;
        
        for (uint32 i = 0; i < config_.historySize; ++i) {
            frameTimes_[i] = config_.targetFrameTime;
        }
    }
};

}

#endif
