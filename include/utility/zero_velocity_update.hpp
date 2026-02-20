#ifndef MSCKF_UTILITY_ZERO_VELOCITY_UPDATE_HPP
#define MSCKF_UTILITY_ZERO_VELOCITY_UPDATE_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../core/state.hpp"
#include "../hal/imu_types.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 ZUPT_WINDOW_SIZE = 50;
constexpr uint32 ZUPT_HISTORY_SIZE = 100;

enum class ZUPTState : uint8 {
    MOVING = 0,
    STATIONARY_DETECTED = 1,
    ZUPT_ACTIVE = 2,
    ZUPT_APPLIED = 3
};

struct ZUPTDetectionResult {
    ZUPTState state;
    float64 confidence;
    float64 accelVariance;
    float64 gyroVariance;
    float64 velocityMagnitude;
    uint32 stationaryCount;
    uint64 detectionTime;
    
    ZUPTDetectionResult() 
        : state(ZUPTState::MOVING)
        , confidence(0)
        , accelVariance(0)
        , gyroVariance(0)
        , velocityMagnitude(0)
        , stationaryCount(0)
        , detectionTime(0) {}
};

struct ZUPTMeasurement {
    Vector3d velocity;
    Matrix<3, 3, float64> velocityCovariance;
    uint64 timestamp;
    bool isValid;
    
    ZUPTMeasurement() 
        : velocity({0, 0, 0})
        , isValid(false) {
        velocityCovariance = Matrix<3, 3, float64>::identity() * 0.01;
    }
};

class ZeroVelocityDetector {
public:
    struct Config {
        float64 accelVarianceThreshold;
        float64 gyroVarianceThreshold;
        float64 velocityMagnitudeThreshold;
        uint32 minStationaryCount;
        uint32 maxStationaryCount;
        float64 gravityMagnitude;
        float64 gravityTolerance;
        float64 detectionConfidenceThreshold;
        bool useAdaptiveThreshold;
        float64 adaptationRate;
        
        Config() 
            : accelVarianceThreshold(0.05)
            , gyroVarianceThreshold(0.001)
            , velocityMagnitudeThreshold(0.1)
            , minStationaryCount(10)
            , maxStationaryCount(100)
            , gravityMagnitude(GRAVITY)
            , gravityTolerance(0.5)
            , detectionConfidenceThreshold(0.8)
            , useAdaptiveThreshold(true)
            , adaptationRate(0.01) {}
    };

private:
    Config config_;
    
    Vector3d accelBuffer_[ZUPT_WINDOW_SIZE];
    Vector3d gyroBuffer_[ZUPT_WINDOW_SIZE];
    uint64 timestampBuffer_[ZUPT_WINDOW_SIZE];
    uint32 bufferHead_;
    uint32 bufferSize_;
    
    float64 accelVarianceHistory_[ZUPT_HISTORY_SIZE];
    float64 gyroVarianceHistory_[ZUPT_HISTORY_SIZE];
    uint32 varianceHistoryHead_;
    uint32 varianceHistorySize_;
    
    ZUPTState currentState_;
    uint32 stationaryCount_;
    float64 currentConfidence_;
    
    float64 adaptiveAccelThreshold_;
    float64 adaptiveGyroThreshold_;
    
    uint64 lastDetectionTime_;
    uint64 lastZUPTTime_;
    
    uint32 totalDetections_;
    uint32 successfulZUPTs_;

public:
    ZeroVelocityDetector() 
        : bufferHead_(0)
        , bufferSize_(0)
        , varianceHistoryHead_(0)
        , varianceHistorySize_(0)
        , currentState_(ZUPTState::MOVING)
        , stationaryCount_(0)
        , currentConfidence_(0)
        , adaptiveAccelThreshold_(0.05)
        , adaptiveGyroThreshold_(0.001)
        , lastDetectionTime_(0)
        , lastZUPTTime_(0)
        , totalDetections_(0)
        , successfulZUPTs_(0) {
        memset(accelBuffer_, 0, sizeof(accelBuffer_));
        memset(gyroBuffer_, 0, sizeof(gyroBuffer_));
        memset(timestampBuffer_, 0, sizeof(timestampBuffer_));
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        adaptiveAccelThreshold_ = config.accelVarianceThreshold;
        adaptiveGyroThreshold_ = config.gyroVarianceThreshold;
        
        bufferHead_ = 0;
        bufferSize_ = 0;
        varianceHistoryHead_ = 0;
        varianceHistorySize_ = 0;
        stationaryCount_ = 0;
        currentState_ = ZUPTState::MOVING;
    }
    
    void addIMUData(const IMUData& imuData) {
        accelBuffer_[bufferHead_] = imuData.accel;
        gyroBuffer_[bufferHead_] = imuData.gyro;
        timestampBuffer_[bufferHead_] = imuData.timestamp;
        
        bufferHead_ = (bufferHead_ + 1) % ZUPT_WINDOW_SIZE;
        if (bufferSize_ < ZUPT_WINDOW_SIZE) {
            bufferSize_++;
        }
        
        lastDetectionTime_ = imuData.timestamp;
    }
    
    ZUPTDetectionResult detect() {
        ZUPTDetectionResult result;
        
        if (bufferSize_ < config_.minStationaryCount) {
            result.state = ZUPTState::MOVING;
            return result;
        }
        
        float64 accelVar = computeAccelVariance();
        float64 gyroVar = computeGyroVariance();
        
        accelVarianceHistory_[varianceHistoryHead_] = accelVar;
        gyroVarianceHistory_[varianceHistoryHead_] = gyroVar;
        varianceHistoryHead_ = (varianceHistoryHead_ + 1) % ZUPT_HISTORY_SIZE;
        if (varianceHistorySize_ < ZUPT_HISTORY_SIZE) {
            varianceHistorySize_++;
        }
        
        result.accelVariance = accelVar;
        result.gyroVariance = gyroVar;
        
        bool isStationary = checkStationary(accelVar, gyroVar);
        
        if (isStationary) {
            stationaryCount_++;
            if (stationaryCount_ > config_.maxStationaryCount) {
                stationaryCount_ = config_.maxStationaryCount;
            }
        } else {
            stationaryCount_ = stationaryCount_ > 0 ? stationaryCount_ - 2 : 0;
        }
        
        result.stationaryCount = stationaryCount_;
        
        currentConfidence_ = computeConfidence(stationaryCount_, accelVar, gyroVar);
        result.confidence = currentConfidence_;
        
        if (stationaryCount_ >= config_.minStationaryCount && 
            currentConfidence_ >= config_.detectionConfidenceThreshold) {
            if (currentState_ == ZUPTState::MOVING) {
                currentState_ = ZUPTState::STATIONARY_DETECTED;
                totalDetections_++;
            } else if (currentState_ == ZUPTState::STATIONARY_DETECTED) {
                currentState_ = ZUPTState::ZUPT_ACTIVE;
            }
        } else {
            currentState_ = ZUPTState::MOVING;
        }
        
        result.state = currentState_;
        result.detectionTime = lastDetectionTime_;
        
        if (config_.useAdaptiveThreshold) {
            adaptThresholds(accelVar, gyroVar, isStationary);
        }
        
        return result;
    }
    
    float64 computeAccelVariance() {
        if (bufferSize_ < 2) return 1e10;
        
        Vector3d mean({0, 0, 0});
        for (uint32 i = 0; i < bufferSize_; ++i) {
            mean += accelBuffer_[i];
        }
        mean = mean / bufferSize_;
        
        float64 variance = 0;
        for (uint32 i = 0; i < bufferSize_; ++i) {
            Vector3d diff = accelBuffer_[i] - mean;
            variance += diff.dot(diff);
        }
        variance /= bufferSize_;
        
        return variance;
    }
    
    float64 computeGyroVariance() {
        if (bufferSize_ < 2) return 1e10;
        
        Vector3d mean({0, 0, 0});
        for (uint32 i = 0; i < bufferSize_; ++i) {
            mean += gyroBuffer_[i];
        }
        mean = mean / bufferSize_;
        
        float64 variance = 0;
        for (uint32 i = 0; i < bufferSize_; ++i) {
            Vector3d diff = gyroBuffer_[i] - mean;
            variance += diff.dot(diff);
        }
        variance /= bufferSize_;
        
        return variance;
    }
    
    bool checkStationary(float64 accelVar, float64 gyroVar) {
        if (accelVar > adaptiveAccelThreshold_) return false;
        if (gyroVar > adaptiveGyroThreshold_) return false;
        
        Vector3d meanAccel({0, 0, 0});
        for (uint32 i = 0; i < bufferSize_; ++i) {
            meanAccel += accelBuffer_[i];
        }
        meanAccel = meanAccel / bufferSize_;
        
        float64 accelMag = meanAccel.norm();
        if (abs(accelMag - config_.gravityMagnitude) > config_.gravityTolerance) {
            return false;
        }
        
        return true;
    }
    
    float64 computeConfidence(uint32 count, float64 accelVar, float64 gyroVar) {
        float64 countFactor = static_cast<float64>(count) / config_.maxStationaryCount;
        
        float64 accelFactor = exp(-accelVar / adaptiveAccelThreshold_);
        float64 gyroFactor = exp(-gyroVar / adaptiveGyroThreshold_);
        
        float64 confidence = countFactor * 0.4 + 
                            accelFactor * 0.3 + 
                            gyroFactor * 0.3;
        
        return clamp(confidence, 0.0, 1.0);
    }
    
    void adaptThresholds(float64 accelVar, float64 gyroVar, bool isStationary) {
        if (varianceHistorySize_ < 10) return;
        
        float64 avgAccelVar = 0, avgGyroVar = 0;
        uint32 count = min(varianceHistorySize_, 20u);
        uint32 startIdx = (varianceHistoryHead_ + ZUPT_HISTORY_SIZE - count) % ZUPT_HISTORY_SIZE;
        
        for (uint32 i = 0; i < count; ++i) {
            uint32 idx = (startIdx + i) % ZUPT_HISTORY_SIZE;
            avgAccelVar += accelVarianceHistory_[idx];
            avgGyroVar += gyroVarianceHistory_[idx];
        }
        avgAccelVar /= count;
        avgGyroVar /= count;
        
        if (isStationary) {
            adaptiveAccelThreshold_ = adaptiveAccelThreshold_ * (1 - config_.adaptationRate) +
                                     max(accelVar * 2, config_.accelVarianceThreshold * 0.5) * config_.adaptationRate;
            adaptiveGyroThreshold_ = adaptiveGyroThreshold_ * (1 - config_.adaptationRate) +
                                    max(gyroVar * 2, config_.gyroVarianceThreshold * 0.5) * config_.adaptationRate;
        } else {
            adaptiveAccelThreshold_ = adaptiveAccelThreshold_ * (1 - config_.adaptationRate) +
                                     config_.accelVarianceThreshold * config_.adaptationRate;
            adaptiveGyroThreshold_ = adaptiveGyroThreshold_ * (1 - config_.adaptationRate) +
                                    config_.gyroVarianceThreshold * config_.adaptationRate;
        }
    }
    
    ZUPTState getCurrentState() const { return currentState_; }
    float64 getCurrentConfidence() const { return currentConfidence_; }
    uint32 getStationaryCount() const { return stationaryCount_; }
    float64 getAdaptiveAccelThreshold() const { return adaptiveAccelThreshold_; }
    float64 getAdaptiveGyroThreshold() const { return adaptiveGyroThreshold_; }
    uint32 getTotalDetections() const { return totalDetections_; }
    uint32 getSuccessfulZUPTs() const { return successfulZUPTs_; }
    
    void incrementSuccessfulZUPT() { successfulZUPTs_++; }
    
    void reset() {
        bufferHead_ = 0;
        bufferSize_ = 0;
        varianceHistoryHead_ = 0;
        varianceHistorySize_ = 0;
        stationaryCount_ = 0;
        currentState_ = ZUPTState::MOVING;
        currentConfidence_ = 0;
        adaptiveAccelThreshold_ = config_.accelVarianceThreshold;
        adaptiveGyroThreshold_ = config_.gyroVarianceThreshold;
    }
};

class ZeroVelocityUpdater {
public:
    struct Config {
        float64 velocityMeasurementNoise;
        float64 biasCorrectionFactor;
        float64 covarianceReductionFactor;
        bool correctGyroBias;
        bool correctAccelBias;
        float64 maxVelocityCorrection;
        float64 maxBiasCorrection;
        uint32 minUpdatesBetweenZUPT;
        
        Config() 
            : velocityMeasurementNoise(0.01)
            , biasCorrectionFactor(0.1)
            , covarianceReductionFactor(0.1)
            , correctGyroBias(true)
            , correctAccelBias(false)
            , maxVelocityCorrection(1.0)
            , maxBiasCorrection(0.01)
            , minUpdatesBetweenZUPT(10) {}
    };

private:
    Config config_;
    ZeroVelocityDetector detector_;
    
    Matrix<3, 15, float64> H_;
    Matrix<3, 3, float64> R_;
    Vector3d measurement_;
    
    uint64 lastZUPTTime_;
    uint32 updateCount_;
    
    float64 totalVelocityCorrection_;
    float64 totalBiasCorrection_;
    uint32 appliedZUPTs_;

public:
    ZeroVelocityUpdater() 
        : lastZUPTTime_(0)
        , updateCount_(0)
        , totalVelocityCorrection_(0)
        , totalBiasCorrection_(0)
        , appliedZUPTs_(0) {
        
        H_ = Matrix<3, 15, float64>();
        H_(0, 3) = 1.0;
        H_(1, 4) = 1.0;
        H_(2, 5) = 1.0;
        
        R_ = Matrix<3, 3, float64>::identity() * config_.velocityMeasurementNoise;
    }
    
    void init(const ZeroVelocityDetector::Config& detectorConfig = ZeroVelocityDetector::Config(),
              const Config& config = Config()) {
        config_ = config;
        detector_.init(detectorConfig);
        
        R_ = Matrix<3, 3, float64>::identity() * config_.velocityMeasurementNoise;
        
        lastZUPTTime_ = 0;
        updateCount_ = 0;
        appliedZUPTs_ = 0;
    }
    
    void addIMUData(const IMUData& imuData) {
        detector_.addIMUData(imuData);
    }
    
    bool shouldApplyZUPT() {
        ZUPTDetectionResult result = detector_.detect();
        
        if (result.state == ZUPTState::ZUPT_ACTIVE) {
            if (updateCount_ >= config_.minUpdatesBetweenZUPT) {
                return true;
            }
        }
        
        return false;
    }
    
    bool applyZUPT(MSCKFState& state) {
        if (!shouldApplyZUPT()) return false;
        
        measurement_ = Vector3d({0, 0, 0}) - state.imuState.velocity;
        
        float64 correctionMag = measurement_.norm();
        if (correctionMag > config_.maxVelocityCorrection) {
            measurement_ = measurement_.normalized() * config_.maxVelocityCorrection;
        }
        
        Matrix<3, 3, float64> S = H_ * state.covariance.block<3, 15>(0, 0) * H_.transpose() + R_;
        Matrix<15, 3, float64> K = state.covariance.block<15, 3>(0, 0) * H_.transpose() * S.inverse();
        
        Vector15d deltaX = K * measurement_;
        
        state.imuState.velocity += Vector3d({deltaX[3], deltaX[4], deltaX[5]});
        
        if (config_.correctGyroBias) {
            Vector3d gyroCorrection(deltaX[9], deltaX[10], deltaX[11]);
            if (gyroCorrection.norm() < config_.maxBiasCorrection) {
                state.imuState.gyroBias += gyroCorrection * config_.biasCorrectionFactor;
            }
        }
        
        if (config_.correctAccelBias) {
            Vector3d accelCorrection(deltaX[12], deltaX[13], deltaX[14]);
            if (accelCorrection.norm() < config_.maxBiasCorrection) {
                state.imuState.accelBias += accelCorrection * config_.biasCorrectionFactor;
            }
        }
        
        Matrix<15, 15, float64> I_KH = Matrix<15, 15, float64>::identity() - K * H_;
        Matrix<15, 15, float64> P_new = I_KH * state.covariance.block<15, 15>(0, 0);
        
        for (uint32 i = 0; i < 15; ++i) {
            for (uint32 j = 0; j < 15; ++j) {
                state.covariance(i, j) = P_new(i, j);
            }
        }
        
        for (uint32 i = 3; i < 6; ++i) {
            state.covariance(i, i) *= config_.covarianceReductionFactor;
        }
        
        state.covariance.symmetrize();
        
        totalVelocityCorrection_ += correctionMag;
        appliedZUPTs_++;
        updateCount_ = 0;
        lastZUPTTime_ = state.timestamp;
        
        detector_.incrementSuccessfulZUPT();
        
        return true;
    }
    
    void update() {
        updateCount_++;
    }
    
    ZUPTDetectionResult detect() {
        return detector_.detect();
    }
    
    ZeroVelocityDetector& getDetector() { return detector_; }
    
    float64 getTotalVelocityCorrection() const { return totalVelocityCorrection_; }
    float64 getAverageVelocityCorrection() const {
        return appliedZUPTs_ > 0 ? totalVelocityCorrection_ / appliedZUPTs_ : 0;
    }
    uint32 getAppliedZUPTs() const { return appliedZUPTs_; }
    ZUPTState getCurrentState() const { return detector_.getCurrentState(); }
    float64 getCurrentConfidence() const { return detector_.getCurrentConfidence(); }
    
    void reset() {
        detector_.reset();
        lastZUPTTime_ = 0;
        updateCount_ = 0;
        totalVelocityCorrection_ = 0;
        appliedZUPTs_ = 0;
    }
};

class NonHolonomicConstraint {
public:
    struct Config {
        bool enablePlanarConstraint;
        bool enableZeroLateralVelocity;
        bool enableZeroVerticalVelocity;
        float64 lateralVelocityNoise;
        float64 verticalVelocityNoise;
        float64 planarPositionNoise;
        float64 maxLateralVelocity;
        float64 maxVerticalVelocity;
        
        Config() 
            : enablePlanarConstraint(true)
            , enableZeroLateralVelocity(true)
            , enableZeroVerticalVelocity(false)
            , lateralVelocityNoise(0.01)
            , verticalVelocityNoise(0.01)
            , planarPositionNoise(0.1)
            , maxLateralVelocity(0.5)
            , maxVerticalVelocity(0.5) {}
    };

private:
    Config config_;
    
    Matrix<2, 15, float64> H_lateral_;
    Matrix<1, 15, float64> H_vertical_;
    
    uint32 constraintCount_;
    float64 totalLateralCorrection_;
    float64 totalVerticalCorrection_;

public:
    NonHolonomicConstraint() 
        : constraintCount_(0)
        , totalLateralCorrection_(0)
        , totalVerticalCorrection_(0) {
        
        H_lateral_ = Matrix<2, 15, float64>();
        H_vertical_ = Matrix<1, 15, float64>();
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        constraintCount_ = 0;
        totalLateralCorrection_ = 0;
        totalVerticalCorrection_ = 0;
    }
    
    bool applyPlanarConstraint(MSCKFState& state) {
        if (!config_.enablePlanarConstraint) return false;
        
        Matrix3d R = state.imuState.orientation.toRotationMatrix();
        
        Vector3d lateralVelocity = R.col(0) * state.imuState.velocity[0] +
                                   R.col(1) * state.imuState.velocity[1];
        
        float64 lateralSpeed = sqrt(lateralVelocity[0] * lateralVelocity[0] +
                                   lateralVelocity[1] * lateralVelocity[1]);
        
        if (lateralSpeed > config_.maxLateralVelocity) {
            return false;
        }
        
        H_lateral_ = Matrix<2, 15, float64>();
        H_lateral_(0, 3) = R(1, 0);
        H_lateral_(0, 4) = R(1, 1);
        H_lateral_(1, 3) = R(2, 0);
        H_lateral_(1, 4) = R(2, 1);
        
        Vector2d r;
        r[0] = -lateralVelocity[1];
        r[1] = -lateralVelocity[2];
        
        Matrix<2, 2, float64> R_mat = Matrix<2, 2, float64>::identity() * config_.lateralVelocityNoise;
        
        Matrix<2, 2, float64> S = H_lateral_ * state.covariance.block<2, 15>(0, 0) * H_lateral_.transpose() + R_mat;
        Matrix<15, 2, float64> K = state.covariance.block<15, 2>(0, 0) * H_lateral_.transpose() * S.inverse();
        
        Vector15d deltaX = K * r;
        
        state.imuState.velocity += Vector3d({deltaX[3], deltaX[4], deltaX[5]});
        
        Matrix<15, 15, float64> I_KH = Matrix<15, 15, float64>::identity() - K * H_lateral_;
        Matrix<15, 15, float64> P_new = I_KH * state.covariance.block<15, 15>(0, 0);
        
        for (uint32 i = 0; i < 15; ++i) {
            for (uint32 j = 0; j < 15; ++j) {
                state.covariance(i, j) = P_new(i, j);
            }
        }
        state.covariance.symmetrize();
        
        totalLateralCorrection_ += r.norm();
        constraintCount_++;
        
        return true;
    }
    
    bool applyZeroLateralVelocityConstraint(MSCKFState& state) {
        if (!config_.enableZeroLateralVelocity) return false;
        
        Matrix3d R = state.imuState.orientation.toRotationMatrix();
        
        float64 lateralVel = R.col(1).dot(state.imuState.velocity);
        
        if (abs(lateralVel) > config_.maxLateralVelocity) {
            return false;
        }
        
        H_lateral_ = Matrix<2, 15, float64>();
        H_lateral_(0, 3) = R(0, 1);
        H_lateral_(0, 4) = R(1, 1);
        H_lateral_(0, 5) = R(2, 1);
        
        Vector2d r;
        r[0] = -lateralVel;
        
        Matrix<2, 2, float64> R_mat = Matrix<2, 2, float64>::identity() * config_.lateralVelocityNoise;
        
        Matrix<1, 1, float64> S;
        S(0, 0) = 0;
        for (uint32 i = 0; i < 15; ++i) {
            for (uint32 j = 0; j < 15; ++j) {
                S(0, 0) += H_lateral_(0, i) * state.covariance(i, j) * H_lateral_(0, j);
            }
        }
        S(0, 0) += config_.lateralVelocityNoise;
        
        float64 K[15];
        for (uint32 i = 0; i < 15; ++i) {
            K[i] = 0;
            for (uint32 j = 0; j < 15; ++j) {
                K[i] += state.covariance(i, j) * H_lateral_(0, j);
            }
            K[i] /= S(0, 0);
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            state.imuState.velocity[i] += K[3 + i] * r[0];
        }
        
        for (uint32 i = 0; i < 15; ++i) {
            for (uint32 j = 0; j < 15; ++j) {
                state.covariance(i, j) -= K[i] * H_lateral_(0, j) * state.covariance(j, i);
            }
        }
        state.covariance.symmetrize();
        
        totalLateralCorrection_ += abs(lateralVel);
        constraintCount_++;
        
        return true;
    }
    
    uint32 getConstraintCount() const { return constraintCount_; }
    float64 getTotalLateralCorrection() const { return totalLateralCorrection_; }
    float64 getTotalVerticalCorrection() const { return totalVerticalCorrection_; }
    
    void reset() {
        constraintCount_ = 0;
        totalLateralCorrection_ = 0;
        totalVerticalCorrection_ = 0;
    }
};

}

#endif
