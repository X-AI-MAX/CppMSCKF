#ifndef MSCKF_CORE_SAFETY_GUARD_HPP
#define MSCKF_CORE_SAFETY_GUARD_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "state.hpp"

namespace msckf {

constexpr float64 SAFE_DIVISION_EPSILON = 1e-12;
constexpr float64 SAFE_SQRT_MIN = 1e-15;
constexpr float64 INVERSE_DEPTH_MIN = 1e-6;
constexpr float64 INVERSE_DEPTH_MAX = 1e6;
constexpr float64 MIN_FEATURE_DEPTH = 0.1;
constexpr float64 MAX_FEATURE_DEPTH = 1000.0;
constexpr float64 MAX_REPROJECTION_ERROR = 50.0;
constexpr float64 MIN_TRANSLATION_NORM = 0.01;
constexpr float64 MAX_ACCEL_NORM = 100.0;
constexpr float64 MAX_GYRO_NORM = 50.0;
constexpr float64 SHOCK_ACCEL_THRESHOLD = 100.0;
constexpr float64 SHOCK_GYRO_THRESHOLD = 30.0;
constexpr uint32 MAX_ITERATION_COUNT = 1000;
constexpr uint32 MAX_LOOP_ITERATIONS = 10000;
constexpr float64 MAX_TIMESTAMP_JUMP = 100000000;
constexpr float64 MIN_TIMESTAMP_INTERVAL = 100;

enum class SafetyStatus {
    OK = 0,
    WARNING = 1,
    ERROR = 2,
    CRITICAL = 3
};

struct SafetyReport {
    SafetyStatus status;
    uint32 errorCode;
    const char* message;
    uint64 timestamp;
    
    SafetyReport() 
        : status(SafetyStatus::OK)
        , errorCode(0)
        , message(nullptr)
        , timestamp(0) {}
    
    SafetyReport(SafetyStatus s, uint32 code, const char* msg, uint64 ts)
        : status(s), errorCode(code), message(msg), timestamp(ts) {}
};

class SafetyGuard {
public:
    struct Config {
        bool enableBoundaryCheck;
        bool enableNumericalCheck;
        bool enableTimeCheck;
        bool enableShockDetection;
        bool enablePureRotationDetection;
        bool enableDynamicObjectFilter;
        
        float64 maxAccelNorm;
        float64 maxGyroNorm;
        float64 shockAccelThreshold;
        float64 shockGyroThreshold;
        float64 pureRotationThreshold;
        float64 maxTimestampJumpUs;
        float64 minTimestampIntervalUs;
        
        Config() 
            : enableBoundaryCheck(true)
            , enableNumericalCheck(true)
            , enableTimeCheck(true)
            , enableShockDetection(true)
            , enablePureRotationDetection(true)
            , enableDynamicObjectFilter(true)
            , maxAccelNorm(MAX_ACCEL_NORM)
            , maxGyroNorm(MAX_GYRO_NORM)
            , shockAccelThreshold(SHOCK_ACCEL_THRESHOLD)
            , shockGyroThreshold(SHOCK_GYRO_THRESHOLD)
            , pureRotationThreshold(MIN_TRANSLATION_NORM)
            , maxTimestampJumpUs(100000.0)
            , minTimestampIntervalUs(100.0) {}
    };

private:
    Config config_;
    
    uint64 lastImuTimestamp_;
    uint64 lastCameraTimestamp_;
    uint64 lastValidVisualTime_;
    
    uint32 consecutiveShockCount_;
    uint64 shockStartTime_;
    bool inShockRecovery_;
    
    uint32 consecutiveTimeJumpCount_;
    uint32 consecutiveZeroFeatureCount_;
    
    bool pureRotationMode_;
    float64 accumulatedTranslation_;
    uint64 pureRotationStartTime_;
    
    uint32 totalSafetyViolations_;
    uint32 shockEvents_;
    uint32 timeJumpEvents_;
    uint32 numericalErrors_;

public:
    SafetyGuard() 
        : lastImuTimestamp_(0)
        , lastCameraTimestamp_(0)
        , lastValidVisualTime_(0)
        , consecutiveShockCount_(0)
        , shockStartTime_(0)
        , inShockRecovery_(false)
        , consecutiveTimeJumpCount_(0)
        , consecutiveZeroFeatureCount_(0)
        , pureRotationMode_(false)
        , accumulatedTranslation_(0)
        , pureRotationStartTime_(0)
        , totalSafetyViolations_(0)
        , shockEvents_(0)
        , timeJumpEvents_(0)
        , numericalErrors_(0) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    void reset() {
        lastImuTimestamp_ = 0;
        lastCameraTimestamp_ = 0;
        lastValidVisualTime_ = 0;
        consecutiveShockCount_ = 0;
        shockStartTime_ = 0;
        inShockRecovery_ = false;
        consecutiveTimeJumpCount_ = 0;
        consecutiveZeroFeatureCount_ = 0;
        pureRotationMode_ = false;
        accumulatedTranslation_ = 0;
        pureRotationStartTime_ = 0;
    }
    
    SafetyReport checkImuTimestamp(uint64 timestamp) {
        if (!config_.enableTimeCheck) {
            return SafetyReport(SafetyStatus::OK, 0, nullptr, timestamp);
        }
        
        if (lastImuTimestamp_ == 0) {
            lastImuTimestamp_ = timestamp;
            return SafetyReport(SafetyStatus::OK, 0, nullptr, timestamp);
        }
        
        if (timestamp < lastImuTimestamp_) {
            timeJumpEvents_++;
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1001, 
                              "IMU timestamp out of order", timestamp);
        }
        
        if (timestamp == lastImuTimestamp_) {
            return SafetyReport(SafetyStatus::WARNING, 1002,
                              "IMU timestamp duplicate", timestamp);
        }
        
        float64 dt = static_cast<float64>(timestamp - lastImuTimestamp_);
        
        if (dt < config_.minTimestampIntervalUs) {
            return SafetyReport(SafetyStatus::WARNING, 1003,
                              "IMU timestamp interval too small", timestamp);
        }
        
        if (dt > config_.maxTimestampJumpUs) {
            consecutiveTimeJumpCount_++;
            timeJumpEvents_++;
            totalSafetyViolations_++;
            
            if (consecutiveTimeJumpCount_ > 5) {
                return SafetyReport(SafetyStatus::CRITICAL, 1004,
                                  "IMU timestamp large jump consecutive", timestamp);
            }
            return SafetyReport(SafetyStatus::WARNING, 1004,
                              "IMU timestamp large jump", timestamp);
        }
        
        consecutiveTimeJumpCount_ = 0;
        lastImuTimestamp_ = timestamp;
        return SafetyReport(SafetyStatus::OK, 0, nullptr, timestamp);
    }
    
    SafetyReport checkCameraTimestamp(uint64 timestamp) {
        if (!config_.enableTimeCheck) {
            return SafetyReport(SafetyStatus::OK, 0, nullptr, timestamp);
        }
        
        if (lastCameraTimestamp_ == 0) {
            lastCameraTimestamp_ = timestamp;
            return SafetyReport(SafetyStatus::OK, 0, nullptr, timestamp);
        }
        
        if (timestamp <= lastCameraTimestamp_) {
            timeJumpEvents_++;
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1011,
                              "Camera timestamp not increasing", timestamp);
        }
        
        float64 dt = static_cast<float64>(timestamp - lastCameraTimestamp_);
        
        if (dt > config_.maxTimestampJumpUs * 10) {
            timeJumpEvents_++;
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1012,
                              "Camera frame drop detected", timestamp);
        }
        
        lastCameraTimestamp_ = timestamp;
        return SafetyReport(SafetyStatus::OK, 0, nullptr, timestamp);
    }
    
    SafetyReport checkImuData(const Vector3d& accel, const Vector3d& gyro, uint64 timestamp) {
        if (!config_.enableShockDetection) {
            return SafetyReport(SafetyStatus::OK, 0, nullptr, timestamp);
        }
        
        float64 accelNorm = accel.norm();
        float64 gyroNorm = gyro.norm();
        
        if (accelNorm > config_.maxAccelNorm || gyroNorm > config_.maxGyroNorm) {
            numericalErrors_++;
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1021,
                              "IMU data out of valid range", timestamp);
        }
        
        bool isShock = (accelNorm > config_.shockAccelThreshold || 
                       gyroNorm > config_.shockGyroThreshold);
        
        if (isShock) {
            consecutiveShockCount_++;
            
            if (consecutiveShockCount_ == 1) {
                shockStartTime_ = timestamp;
            }
            
            if (consecutiveShockCount_ >= 3) {
                shockEvents_++;
                totalSafetyViolations_++;
                inShockRecovery_ = true;
                return SafetyReport(SafetyStatus::CRITICAL, 1022,
                                  "Shock/impact detected", timestamp);
            }
            
            return SafetyReport(SafetyStatus::WARNING, 1023,
                              "Abnormal IMU reading", timestamp);
        }
        
        if (consecutiveShockCount_ > 0) {
            consecutiveShockCount_ = 0;
        }
        
        return SafetyReport(SafetyStatus::OK, 0, nullptr, timestamp);
    }
    
    SafetyReport checkStateBounds(const MSCKFState& state) {
        if (!config_.enableBoundaryCheck) {
            return SafetyReport(SafetyStatus::OK, 0, nullptr, state.timestamp);
        }
        
        if (state.numCameraStates > MAX_CAMERA_FRAMES) {
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1031,
                              "Camera state overflow", state.timestamp);
        }
        
        if (state.covarianceDim > MAX_STATE_DIM) {
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1032,
                              "Covariance dimension overflow", state.timestamp);
        }
        
        uint32 expectedDim = IMU_STATE_DIM + EXTRINSIC_STATE_DIM + 
                            state.numCameraStates * CAMERA_STATE_DIM;
        if (state.covarianceDim != expectedDim) {
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1033,
                              "Covariance dimension mismatch", state.timestamp);
        }
        
        return SafetyReport(SafetyStatus::OK, 0, nullptr, state.timestamp);
    }
    
    SafetyReport checkFeatureValidity(const Feature& feature, uint32 imageWidth, uint32 imageHeight) {
        if (!config_.enableBoundaryCheck) {
            return SafetyReport(SafetyStatus::OK, 0, nullptr, 0);
        }
        
        if (feature.numObservations == 0) {
            return SafetyReport(SafetyStatus::WARNING, 1041,
                              "Feature has no observations", 0);
        }
        
        if (feature.numObservations > Feature::MAX_OBSERVATIONS) {
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1042,
                              "Feature observation overflow", 0);
        }
        
        for (uint32 i = 0; i < feature.numObservations; ++i) {
            float64 u = feature.observations[i].u;
            float64 v = feature.observations[i].v;
            
            if (u < 0 || u >= imageWidth || v < 0 || v >= imageHeight) {
                totalSafetyViolations_++;
                return SafetyReport(SafetyStatus::ERROR, 1043,
                                  "Feature observation out of image bounds", 0);
            }
        }
        
        if (feature.isTriangulated) {
            if (feature.position[2] < MIN_FEATURE_DEPTH) {
                return SafetyReport(SafetyStatus::WARNING, 1044,
                                  "Feature depth too small", 0);
            }
            
            if (feature.position[2] > MAX_FEATURE_DEPTH) {
                return SafetyReport(SafetyStatus::WARNING, 1045,
                                  "Feature depth too large", 0);
            }
        }
        
        return SafetyReport(SafetyStatus::OK, 0, nullptr, 0);
    }
    
    SafetyReport checkInverseDepth(float64 invDepth) {
        if (!config_.enableNumericalCheck) {
            return SafetyReport(SafetyStatus::OK, 0, nullptr, 0);
        }
        
        if (invDepth < INVERSE_DEPTH_MIN) {
            numericalErrors_++;
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1051,
                              "Inverse depth too small (point at infinity)", 0);
        }
        
        if (invDepth > INVERSE_DEPTH_MAX) {
            numericalErrors_++;
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1052,
                              "Inverse depth too large (negative depth)", 0);
        }
        
        if (invDepth < 0) {
            numericalErrors_++;
            totalSafetyViolations_++;
            return SafetyReport(SafetyStatus::ERROR, 1053,
                              "Negative inverse depth", 0);
        }
        
        return SafetyReport(SafetyStatus::OK, 0, nullptr, 0);
    }
    
    bool detectPureRotation(const Vector3d& translation, uint64 timestamp) {
        if (!config_.enablePureRotationDetection) {
            return false;
        }
        
        accumulatedTranslation_ += translation.norm();
        
        if (accumulatedTranslation_ < config_.pureRotationThreshold) {
            if (!pureRotationMode_) {
                pureRotationMode_ = true;
                pureRotationStartTime_ = timestamp;
            }
            return true;
        } else {
            accumulatedTranslation_ = 0;
            pureRotationMode_ = false;
            return false;
        }
    }
    
    bool isInPureRotationMode() const {
        return pureRotationMode_;
    }
    
    void exitPureRotationMode() {
        pureRotationMode_ = false;
        accumulatedTranslation_ = 0;
    }
    
    bool isInShockRecovery() const {
        return inShockRecovery_;
    }
    
    void exitShockRecovery() {
        inShockRecovery_ = false;
        consecutiveShockCount_ = 0;
    }
    
    void reportValidVisual(uint64 timestamp) {
        lastValidVisualTime_ = timestamp;
        consecutiveZeroFeatureCount_ = 0;
    }
    
    void reportZeroFeatures(uint64 timestamp) {
        consecutiveZeroFeatureCount_++;
    }
    
    bool shouldEnterImuOnlyMode(uint64 timestamp) const {
        if (consecutiveZeroFeatureCount_ > 10) {
            return true;
        }
        
        if (lastValidVisualTime_ > 0) {
            float64 dt = static_cast<float64>(timestamp - lastValidVisualTime_);
            if (dt > 5000000.0) {
                return true;
            }
        }
        
        return false;
    }
    
    uint32 getTotalViolations() const { return totalSafetyViolations_; }
    uint32 getShockEvents() const { return shockEvents_; }
    uint32 getTimeJumpEvents() const { return timeJumpEvents_; }
    uint32 getNumericalErrors() const { return numericalErrors_; }
};

inline float64 safeDivide(float64 numerator, float64 denominator, float64 defaultValue = 0.0) {
    if (abs(denominator) < SAFE_DIVISION_EPSILON) {
        return defaultValue;
    }
    return numerator / denominator;
}

inline float64 safeSqrt(float64 x, float64 defaultValue = 0.0) {
    if (x < 0) {
        return defaultValue;
    }
    if (x < SAFE_SQRT_MIN) {
        return 0.0;
    }
    return sqrt(x);
}

inline float64 safeAcos(float64 x) {
    return acos(clamp(x, -1.0, 1.0));
}

inline float64 safeAsin(float64 x) {
    return asin(clamp(x, -1.0, 1.0));
}

inline float64 safeAtan2(float64 y, float64 x) {
    if (abs(y) < SAFE_DIVISION_EPSILON && abs(x) < SAFE_DIVISION_EPSILON) {
        return 0.0;
    }
    return atan2(y, x);
}

inline float64 safeLog(float64 x, float64 defaultValue = -1e10) {
    if (x <= 0) {
        return defaultValue;
    }
    return log(x);
}

inline float64 safeExp(float64 x, float64 maxValue = 1e10) {
    if (x > 700.0) {
        return maxValue;
    }
    if (x < -700.0) {
        return 0.0;
    }
    return exp(x);
}

inline bool checkArrayBounds(uint32 index, uint32 size) {
    return index < size;
}

inline bool checkMatrixIndex(uint32 row, uint32 col, uint32 rows, uint32 cols) {
    return row < rows && col < cols;
}

inline bool isValidFeatureDepth(float64 depth) {
    return depth >= MIN_FEATURE_DEPTH && depth <= MAX_FEATURE_DEPTH;
}

inline bool isValidInverseDepth(float64 invDepth) {
    return invDepth >= INVERSE_DEPTH_MIN && invDepth <= INVERSE_DEPTH_MAX;
}

inline bool isValidPixelCoord(float64 u, float64 v, uint32 width, uint32 height) {
    return u >= 0 && u < width && v >= 0 && v < height;
}

inline bool isValidQuaternion(const Quaterniond& q) {
    float64 norm = q.norm();
    return abs(norm - 1.0) < 0.01;
}

inline void normalizeQuaternionSafe(Quaterniond& q) {
    float64 norm = q.norm();
    if (norm > SAFE_DIVISION_EPSILON) {
        float64 invNorm = 1.0 / norm;
        q.w *= invNorm;
        q.x *= invNorm;
        q.y *= invNorm;
        q.z *= invNorm;
    } else {
        q = Quaterniond::identity();
    }
}

inline bool isCovarianceValid(const Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>& P, uint32 dim) {
    for (uint32 i = 0; i < dim; ++i) {
        if (P(i, i) <= 0) {
            return false;
        }
        
        for (uint32 j = i + 1; j < dim; ++j) {
            float64 diag_i = P(i, i);
            float64 diag_j = P(j, j);
            float64 offDiag = P(i, j);
            
            if (offDiag * offDiag >= diag_i * diag_j) {
                return false;
            }
        }
    }
    return true;
}

inline void enforceCovariancePositiveDefinite(Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>& P, 
                                              uint32 dim, float64 minEigenvalue = 1e-6) {
    P.symmetrize();
    
    float64 minDiag = P(0, 0);
    for (uint32 i = 1; i < dim; ++i) {
        if (P(i, i) < minDiag) {
            minDiag = P(i, i);
        }
    }
    
    if (minDiag < minEigenvalue) {
        float64 correction = minEigenvalue - minDiag;
        for (uint32 i = 0; i < dim; ++i) {
            P(i, i) += correction;
        }
    }
    
    for (uint32 i = 0; i < dim; ++i) {
        for (uint32 j = i + 1; j < dim; ++j) {
            float64 maxOffDiag = sqrt(P(i, i) * P(j, j)) * 0.9999;
            if (abs(P(i, j)) > maxOffDiag) {
                P(i, j) = sign(P(i, j)) * maxOffDiag;
                P(j, i) = P(i, j);
            }
        }
    }
}

}

#endif
