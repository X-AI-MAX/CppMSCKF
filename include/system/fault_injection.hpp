#ifndef MSCKF_SYSTEM_FAULT_INJECTION_HPP
#define MSCKF_SYSTEM_FAULT_INJECTION_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../core/state.hpp"
#include <cstring>
#include <cstdlib>

namespace msckf {

enum class FaultType {
    NONE = 0,
    SINGLE_BIT_FLIP = 1,
    MULTI_BIT_FLIP = 2,
    STUCK_AT_ZERO = 3,
    STUCK_AT_ONE = 4,
    SENSOR_SATURATION = 5,
    SENSOR_DISCONNECT = 6,
    TIMESTAMP_ERROR = 7,
    DATA_CORRUPTION = 8,
    MEMORY_OVERFLOW = 9,
    DIVISION_BY_ZERO = 10,
    NAN_INJECTION = 11,
    INF_INJECTION = 12
};

enum class FaultLocation {
    IMU_ACCEL = 0,
    IMU_GYRO = 1,
    IMU_TIMESTAMP = 2,
    CAMERA_IMAGE = 3,
    CAMERA_TIMESTAMP = 4,
    FEATURE_COORD = 5,
    STATE_POSITION = 6,
    STATE_VELOCITY = 7,
    STATE_ORIENTATION = 8,
    STATE_BIAS = 9,
    COVARIANCE = 10,
    EXTRINSIC = 11
};

struct FaultConfig {
    FaultType type;
    FaultLocation location;
    uint32 startIteration;
    uint32 endIteration;
    float64 probability;
    uint32 seed;
    bool enabled;
    
    FaultConfig() 
        : type(FaultType::NONE)
        , location(FaultLocation::IMU_ACCEL)
        , startIteration(0)
        , endIteration(1000)
        , probability(0.01)
        , seed(12345)
        , enabled(false) {}
    
    FaultConfig(FaultType t, FaultLocation loc, uint32 start, uint32 end, 
                float64 prob = 0.01, uint32 s = 12345)
        : type(t), location(loc), startIteration(start), endIteration(end)
        , probability(prob), seed(s), enabled(true) {}
};

struct FaultResult {
    FaultType type;
    FaultLocation location;
    uint32 iteration;
    uint64 timestamp;
    bool detected;
    bool recovered;
    float64 impact;
    
    FaultResult() 
        : type(FaultType::NONE)
        , location(FaultLocation::IMU_ACCEL)
        , iteration(0)
        , timestamp(0)
        , detected(false)
        , recovered(false)
        , impact(0) {}
};

class FaultInjector {
public:
    struct Config {
        bool enableInjection;
        bool enableDetection;
        bool enableLogging;
        uint32 maxFaults;
        
        Config() 
            : enableInjection(false)
            , enableDetection(true)
            , enableLogging(true)
            , maxFaults(100) {}
    };

private:
    Config config_;
    FaultConfig faultConfigs_[20];
    uint32 numConfigs_;
    
    uint32 currentIteration_;
    uint32 faultCount_;
    
    FaultResult faultHistory_[100];
    uint32 historyCount_;
    
    uint32 randomSeed_;
    
    bool faultActive_;
    FaultType activeFaultType_;
    FaultLocation activeFaultLocation_;

public:
    FaultInjector() 
        : numConfigs_(0)
        , currentIteration_(0)
        , faultCount_(0)
        , historyCount_(0)
        , randomSeed_(12345)
        , faultActive_(false)
        , activeFaultType_(FaultType::NONE)
        , activeFaultLocation_(FaultLocation::IMU_ACCEL) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    void addFaultConfig(const FaultConfig& faultConfig) {
        if (numConfigs_ < 20) {
            faultConfigs_[numConfigs_++] = faultConfig;
        }
    }
    
    void clearFaultConfigs() {
        numConfigs_ = 0;
    }
    
    void setIteration(uint32 iter) {
        currentIteration_ = iter;
    }
    
    void advanceIteration() {
        currentIteration_++;
    }
    
    uint32 getRandom() {
        randomSeed_ = randomSeed_ * 1103515245 + 12345;
        return (randomSeed_ >> 16) & 0x7FFF;
    }
    
    float64 getRandomFloat() {
        return static_cast<float64>(getRandom()) / 32768.0;
    }
    
    bool shouldInjectFault(const FaultConfig& cfg) {
        if (!cfg.enabled) return false;
        if (!config_.enableInjection) return false;
        if (currentIteration_ < cfg.startIteration) return false;
        if (currentIteration_ > cfg.endIteration) return false;
        
        return getRandomFloat() < cfg.probability;
    }
    
    void injectImuFault(Vector3d& accel, Vector3d& gyro, uint64& timestamp) {
        for (uint32 i = 0; i < numConfigs_; ++i) {
            const FaultConfig& cfg = faultConfigs_[i];
            
            if (!shouldInjectFault(cfg)) continue;
            
            faultActive_ = true;
            activeFaultType_ = cfg.type;
            faultCount_++;
            
            switch (cfg.location) {
                case FaultLocation::IMU_ACCEL:
                    injectVectorFault(accel, cfg.type);
                    break;
                case FaultLocation::IMU_GYRO:
                    injectVectorFault(gyro, cfg.type);
                    break;
                case FaultLocation::IMU_TIMESTAMP:
                    injectTimestampFault(timestamp, cfg.type);
                    break;
                default:
                    break;
            }
            
            recordFault(cfg.type, cfg.location, currentIteration_, timestamp);
        }
    }
    
    void injectVectorFault(Vector3d& vec, FaultType type) {
        uint32 targetIdx = getRandom() % 3;
        
        switch (type) {
            case FaultType::SINGLE_BIT_FLIP: {
                uint32 bitIdx = getRandom() % 64;
                uint64* ptr = reinterpret_cast<uint64*>(&vec[targetIdx]);
                *ptr ^= (1ULL << bitIdx);
                break;
            }
            case FaultType::MULTI_BIT_FLIP: {
                for (uint32 i = 0; i < 4; ++i) {
                    uint32 bitIdx = getRandom() % 64;
                    uint64* ptr = reinterpret_cast<uint64*>(&vec[targetIdx]);
                    *ptr ^= (1ULL << bitIdx);
                }
                break;
            }
            case FaultType::STUCK_AT_ZERO:
                vec[targetIdx] = 0;
                break;
            case FaultType::STUCK_AT_ONE:
                vec[targetIdx] = 1e10;
                break;
            case FaultType::SENSOR_SATURATION:
                vec[targetIdx] = (getRandom() % 2 == 0) ? 1e6 : -1e6;
                break;
            case FaultType::NAN_INJECTION:
                vec[targetIdx] = NAN;
                break;
            case FaultType::INF_INJECTION:
                vec[targetIdx] = (getRandom() % 2 == 0) ? INFINITY : -INFINITY;
                break;
            default:
                break;
        }
    }
    
    void injectTimestampFault(uint64& timestamp, FaultType type) {
        switch (type) {
            case FaultType::TIMESTAMP_ERROR:
                if (getRandom() % 2 == 0) {
                    timestamp = 0;
                } else {
                    timestamp += 100000000;
                }
                break;
            case FaultType::SINGLE_BIT_FLIP: {
                uint32 bitIdx = getRandom() % 64;
                timestamp ^= (1ULL << bitIdx);
                break;
            }
            default:
                break;
        }
    }
    
    void injectFeatureFault(float64& u, float64& v, FaultType type) {
        switch (type) {
            case FaultType::DATA_CORRUPTION:
                u += (getRandomFloat() - 0.5) * 100;
                v += (getRandomFloat() - 0.5) * 100;
                break;
            case FaultType::SINGLE_BIT_FLIP: {
                uint32 bitIdx = getRandom() % 64;
                uint64* ptr = reinterpret_cast<uint64*>(&u);
                *ptr ^= (1ULL << bitIdx);
                break;
            }
            default:
                break;
        }
    }
    
    void injectStateFault(MSCKFState& state, FaultLocation location, FaultType type) {
        switch (location) {
            case FaultLocation::STATE_POSITION:
                injectVectorFault(state.imuState.position, type);
                break;
            case FaultLocation::STATE_VELOCITY:
                injectVectorFault(state.imuState.velocity, type);
                break;
            case FaultLocation::STATE_ORIENTATION: {
                Vector3d euler = {state.imuState.orientation.x,
                                 state.imuState.orientation.y,
                                 state.imuState.orientation.z};
                injectVectorFault(euler, type);
                break;
            }
            case FaultLocation::STATE_BIAS:
                injectVectorFault(state.imuState.accelBias, type);
                injectVectorFault(state.imuState.gyroBias, type);
                break;
            case FaultLocation::COVARIANCE: {
                uint32 row = getRandom() % state.covarianceDim;
                uint32 col = getRandom() % state.covarianceDim;
                if (type == FaultType::NAN_INJECTION) {
                    state.covariance(row, col) = NAN;
                } else if (type == FaultType::SINGLE_BIT_FLIP) {
                    uint64* ptr = reinterpret_cast<uint64*>(&state.covariance(row, col));
                    *ptr ^= (1ULL << (getRandom() % 64));
                }
                break;
            }
            default:
                break;
        }
    }
    
    void recordFault(FaultType type, FaultLocation location, 
                    uint32 iteration, uint64 timestamp) {
        if (historyCount_ < 100) {
            faultHistory_[historyCount_].type = type;
            faultHistory_[historyCount_].location = location;
            faultHistory_[historyCount_].iteration = iteration;
            faultHistory_[historyCount_].timestamp = timestamp;
            faultHistory_[historyCount_].detected = false;
            faultHistory_[historyCount_].recovered = false;
            historyCount_++;
        }
    }
    
    bool detectFault(const Vector3d& accel, const Vector3d& gyro) {
        if (!config_.enableDetection) return false;
        
        for (uint32 i = 0; i < 3; ++i) {
            if (accel[i] != accel[i]) return true;
            if (gyro[i] != gyro[i]) return true;
            if (accel[i] > 1e10 || accel[i] < -1e10) return true;
            if (gyro[i] > 1e10 || gyro[i] < -1e10) return true;
        }
        
        return false;
    }
    
    bool detectStateFault(const MSCKFState& state) {
        if (!config_.enableDetection) return false;
        
        for (uint32 i = 0; i < 3; ++i) {
            if (state.imuState.position[i] != state.imuState.position[i]) return true;
            if (state.imuState.velocity[i] != state.imuState.velocity[i]) return true;
        }
        
        for (uint32 i = 0; i < state.covarianceDim; ++i) {
            if (state.covariance(i, i) != state.covariance(i, i)) return true;
            if (state.covariance(i, i) <= 0) return true;
        }
        
        return false;
    }
    
    bool isFaultActive() const { return faultActive_; }
    FaultType getActiveFaultType() const { return activeFaultType_; }
    uint32 getFaultCount() const { return faultCount_; }
    uint32 getHistoryCount() const { return historyCount_; }
    const FaultResult& getFaultHistory(uint32 idx) const { return faultHistory_[idx]; }
    
    void reset() {
        currentIteration_ = 0;
        faultCount_ = 0;
        historyCount_ = 0;
        faultActive_ = false;
        activeFaultType_ = FaultType::NONE;
    }
};

class FaultRecoveryHandler {
public:
    struct Config {
        float64 positionRecoveryThreshold;
        float64 velocityRecoveryThreshold;
        float64 orientationRecoveryThreshold;
        float64 covarianceRecoveryScale;
        uint32 recoveryIterations;
        
        Config() 
            : positionRecoveryThreshold(10.0)
            , velocityRecoveryThreshold(5.0)
            , orientationRecoveryThreshold(0.5)
            , covarianceRecoveryScale(100.0)
            , recoveryIterations(100) {}
    };

private:
    Config config_;
    
    bool inRecovery_;
    uint32 recoveryCounter_;
    uint32 totalRecoveries_;
    
    Vector3d preFaultPosition_;
    Vector3d preFaultVelocity_;
    Quaterniond preFaultOrientation_;

public:
    FaultRecoveryHandler() 
        : inRecovery_(false)
        , recoveryCounter_(0)
        , totalRecoveries_(0) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    void savePreFaultState(const MSCKFState& state) {
        preFaultPosition_ = state.imuState.position;
        preFaultVelocity_ = state.imuState.velocity;
        preFaultOrientation_ = state.imuState.orientation;
    }
    
    void startRecovery() {
        inRecovery_ = true;
        recoveryCounter_ = 0;
        totalRecoveries_++;
    }
    
    void applyRecovery(MSCKFState& state) {
        if (!inRecovery_) return;
        
        for (uint32 i = 0; i < state.covarianceDim; ++i) {
            state.covariance(i, i) *= config_.covarianceRecoveryScale;
        }
        state.covariance.symmetrize();
        
        for (uint32 i = 0; i < 3; ++i) {
            state.imuState.accelBias[i] *= 0.5;
            state.imuState.gyroBias[i] *= 0.5;
        }
        
        recoveryCounter_++;
        
        if (recoveryCounter_ >= config_.recoveryIterations) {
            inRecovery_ = false;
        }
    }
    
    bool checkRecovery(const MSCKFState& state) {
        if (!inRecovery_) return true;
        
        Vector3d posDiff = state.imuState.position - preFaultPosition_;
        Vector3d velDiff = state.imuState.velocity - preFaultVelocity_;
        
        if (posDiff.norm() > config_.positionRecoveryThreshold) return false;
        if (velDiff.norm() > config_.velocityRecoveryThreshold) return false;
        
        return true;
    }
    
    bool isInRecovery() const { return inRecovery_; }
    uint32 getTotalRecoveries() const { return totalRecoveries_; }
    
    void reset() {
        inRecovery_ = false;
        recoveryCounter_ = 0;
    }
};

class FaultInjectionTestSuite {
public:
    struct TestResult {
        const char* testName;
        bool passed;
        uint32 faultCount;
        uint32 detectedCount;
        uint32 recoveredCount;
        float64 positionError;
        float64 velocityError;
        float64 orientationError;
        uint64 executionTime;
        
        TestResult() 
            : testName(nullptr)
            , passed(false)
            , faultCount(0)
            , detectedCount(0)
            , recoveredCount(0)
            , positionError(0)
            , velocityError(0)
            , orientationError(0)
            , executionTime(0) {}
    };

private:
    FaultInjector injector_;
    FaultRecoveryHandler recovery_;
    
    TestResult results_[50];
    uint32 numResults_;
    
    uint64 testStartTime_;

public:
    FaultInjectionTestSuite() : numResults_(0), testStartTime_(0) {}
    
    void init() {
        FaultInjector::Config injConfig;
        injConfig.enableInjection = true;
        injConfig.enableDetection = true;
        injector_.init(injConfig);
        
        recovery_.init(FaultRecoveryHandler::Config());
    }
    
    void runSingleBitFlipTest() {
        TestResult& result = results_[numResults_];
        result.testName = "Single Bit Flip Test";
        
        injector_.clearFaultConfigs();
        injector_.addFaultConfig(FaultConfig(
            FaultType::SINGLE_BIT_FLIP,
            FaultLocation::IMU_ACCEL,
            100, 200, 0.1
        ));
        injector_.reset();
        
        uint32 detectedCount = 0;
        for (uint32 i = 0; i < 300; ++i) {
            injector_.setIteration(i);
            
            Vector3d accel({0.1, 0.2, 9.8});
            Vector3d gyro({0.01, 0.02, 0.03});
            uint64 timestamp = i * 1000;
            
            injector_.injectImuFault(accel, gyro, timestamp);
            
            if (injector_.detectFault(accel, gyro)) {
                detectedCount++;
            }
        }
        
        result.faultCount = injector_.getFaultCount();
        result.detectedCount = detectedCount;
        result.passed = (detectedCount >= result.faultCount * 0.9);
        
        numResults_++;
    }
    
    void runSensorDisconnectTest() {
        TestResult& result = results_[numResults_];
        result.testName = "Sensor Disconnect Test";
        
        injector_.clearFaultConfigs();
        injector_.addFaultConfig(FaultConfig(
            FaultType::SENSOR_DISCONNECT,
            FaultLocation::IMU_ACCEL,
            50, 100, 0.5
        ));
        injector_.reset();
        
        uint32 detectedCount = 0;
        for (uint32 i = 0; i < 150; ++i) {
            injector_.setIteration(i);
            
            Vector3d accel({0.1, 0.2, 9.8});
            Vector3d gyro({0.01, 0.02, 0.03});
            uint64 timestamp = i * 1000;
            
            injector_.injectImuFault(accel, gyro, timestamp);
            
            if (injector_.detectFault(accel, gyro)) {
                detectedCount++;
            }
        }
        
        result.faultCount = injector_.getFaultCount();
        result.detectedCount = detectedCount;
        result.passed = (detectedCount >= result.faultCount * 0.95);
        
        numResults_++;
    }
    
    void runNanInjectionTest() {
        TestResult& result = results_[numResults_];
        result.testName = "NaN Injection Test";
        
        injector_.clearFaultConfigs();
        injector_.addFaultConfig(FaultConfig(
            FaultType::NAN_INJECTION,
            FaultLocation::IMU_ACCEL,
            100, 150, 0.2
        ));
        injector_.reset();
        
        uint32 detectedCount = 0;
        for (uint32 i = 0; i < 200; ++i) {
            injector_.setIteration(i);
            
            Vector3d accel({0.1, 0.2, 9.8});
            Vector3d gyro({0.01, 0.02, 0.03});
            uint64 timestamp = i * 1000;
            
            injector_.injectImuFault(accel, gyro, timestamp);
            
            if (injector_.detectFault(accel, gyro)) {
                detectedCount++;
            }
        }
        
        result.faultCount = injector_.getFaultCount();
        result.detectedCount = detectedCount;
        result.passed = (detectedCount >= result.faultCount);
        
        numResults_++;
    }
    
    void runTimestampJumpTest() {
        TestResult& result = results_[numResults_];
        result.testName = "Timestamp Jump Test";
        
        injector_.clearFaultConfigs();
        injector_.addFaultConfig(FaultConfig(
            FaultType::TIMESTAMP_ERROR,
            FaultLocation::IMU_TIMESTAMP,
            100, 200, 0.1
        ));
        injector_.reset();
        
        uint32 detectedCount = 0;
        uint64 prevTimestamp = 0;
        
        for (uint32 i = 0; i < 300; ++i) {
            injector_.setIteration(i);
            
            Vector3d accel({0.1, 0.2, 9.8});
            Vector3d gyro({0.01, 0.02, 0.03});
            uint64 timestamp = i * 1000;
            
            injector_.injectImuFault(accel, gyro, timestamp);
            
            if (timestamp <= prevTimestamp || timestamp > prevTimestamp + 100000) {
                detectedCount++;
            }
            prevTimestamp = timestamp;
        }
        
        result.faultCount = injector_.getFaultCount();
        result.detectedCount = detectedCount;
        result.passed = (detectedCount >= result.faultCount * 0.8);
        
        numResults_++;
    }
    
    void runCovarianceCorruptionTest() {
        TestResult& result = results_[numResults_];
        result.testName = "Covariance Corruption Test";
        
        injector_.clearFaultConfigs();
        injector_.addFaultConfig(FaultConfig(
            FaultType::NAN_INJECTION,
            FaultLocation::COVARIANCE,
            50, 100, 0.1
        ));
        injector_.reset();
        
        uint32 detectedCount = 0;
        
        for (uint32 i = 0; i < 150; ++i) {
            injector_.setIteration(i);
            
            MSCKFState state;
            state.covarianceDim = 21;
            
            if (injector_.shouldInjectFault(FaultConfig(
                FaultType::NAN_INJECTION, FaultLocation::COVARIANCE, 50, 100, 0.1))) {
                injector_.injectStateFault(state, FaultLocation::COVARIANCE, FaultType::NAN_INJECTION);
            }
            
            if (injector_.detectStateFault(state)) {
                detectedCount++;
            }
        }
        
        result.faultCount = injector_.getFaultCount();
        result.detectedCount = detectedCount;
        result.passed = (detectedCount >= result.faultCount * 0.9);
        
        numResults_++;
    }
    
    void runAllTests() {
        runSingleBitFlipTest();
        runSensorDisconnectTest();
        runNanInjectionTest();
        runTimestampJumpTest();
        runCovarianceCorruptionTest();
    }
    
    uint32 getNumResults() const { return numResults_; }
    const TestResult& getResult(uint32 idx) const { return results_[idx]; }
    
    uint32 getPassedCount() const {
        uint32 count = 0;
        for (uint32 i = 0; i < numResults_; ++i) {
            if (results_[i].passed) count++;
        }
        return count;
    }
    
    uint32 getFailedCount() const {
        return numResults_ - getPassedCount();
    }
    
    float64 getPassRate() const {
        if (numResults_ == 0) return 0;
        return static_cast<float64>(getPassedCount()) / numResults_;
    }
    
    void reset() {
        numResults_ = 0;
        injector_.reset();
        recovery_.reset();
    }
};

}

#endif
