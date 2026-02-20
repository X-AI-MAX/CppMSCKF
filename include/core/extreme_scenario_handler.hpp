
#ifndef MSCKF_CORE_EXTREME_SCENARIO_HANDLER_HPP
#define MSCKF_CORE_EXTREME_SCENARIO_HANDLER_HPP

#include "state.hpp"
#include "safety_guard.hpp"
#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"

namespace msckf {

class PureRotationHandler {
public:
    struct Config {
        float64 translationThreshold;
        float64 minObservationTime;
        float64 fixedInverseDepth;
        uint32 minObservationsForSwitch;
        bool enableAutoSwitch;
        
        Config() 
            : translationThreshold(0.05)
            , minObservationTime(0.5)
            , fixedInverseDepth(0.5)
            , minObservationsForSwitch(10)
            , enableAutoSwitch(true) {}
    };

private:
    Config config_;
    
    bool inPureRotationMode_;
    uint64 modeStartTime_;
    
    Vector3d accumulatedTranslation_;
    uint32 observationCount_;
    
    float64 rotationOnlyCovarianceScale_;
    
    Quaterniond prevOrientation_;
    bool hasPrevOrientation_;

public:
    PureRotationHandler() 
        : inPureRotationMode_(false)
        , modeStartTime_(0)
        , accumulatedTranslation_({0, 0, 0})
        , observationCount_(0)
        , rotationOnlyCovarianceScale_(100.0)
        , hasPrevOrientation_(false) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    bool checkAndUpdate(const MSCKFState& state, const Vector3d& deltaTranslation) {
        accumulatedTranslation_ += deltaTranslation;
        observationCount_++;
        
        float64 translationNorm = accumulatedTranslation_.norm();
        
        if (translationNorm < config_.translationThreshold && 
            observationCount_ >= config_.minObservationsForSwitch) {
            
            if (!inPureRotationMode_ && config_.enableAutoSwitch) {
                enterPureRotationMode(state.timestamp);
            }
            return true;
        } else {
            if (inPureRotationMode_) {
                exitPureRotationMode();
            }
            return false;
        }
    }
    
    void enterPureRotationMode(uint64 timestamp) {
        inPureRotationMode_ = true;
        modeStartTime_ = timestamp;
        accumulatedTranslation_ = Vector3d({0, 0, 0});
        observationCount_ = 0;
    }
    
    void exitPureRotationMode() {
        inPureRotationMode_ = false;
        modeStartTime_ = 0;
        accumulatedTranslation_ = Vector3d({0, 0, 0});
        observationCount_ = 0;
    }
    
    void modifyJacobianForPureRotation(Matrix<2, 3, float64>& H_f, 
                                       Matrix<2, 15, float64>& H_imu,
                                       const Vector3d& p_C) {
        H_f = Matrix<2, 3, float64>();
        
        for (uint32 i = 0; i < 2; ++i) {
            for (uint32 j = 0; j < 15; ++j) {
                if (j >= 6 && j < 9) {
                } else {
                    H_imu(i, j) *= 0.1;
                }
            }
        }
    }
    
    float64 getFixedInverseDepth() const {
        return config_.fixedInverseDepth;
    }
    
    void inflateCovarianceForUnobservable(MSCKFState& state) {
        if (!inPureRotationMode_) return;
        
        for (uint32 i = 0; i < 3; ++i) {
            state.covariance(i, i) *= rotationOnlyCovarianceScale_;
        }
        
        state.covariance.symmetrize();
    }
    
    bool isInPureRotationMode() const { return inPureRotationMode_; }
    uint64 getModeStartTime() const { return modeStartTime_; }
};

class DynamicObjectFilter {
public:
    struct Config {
        float64 ransacInlierRatioThreshold;
        float64 multiFrameConsistencyThreshold;
        uint32 minFramesForConsistency;
        float64 maxVelocityThreshold;
        float64 epipolarThreshold;
        
        Config() 
            : ransacInlierRatioThreshold(0.7)
            , multiFrameConsistencyThreshold(2.0)
            , minFramesForConsistency(3)
            , maxVelocityThreshold(5.0)
            , epipolarThreshold(2.0) {}
    };

private:
    Config config_;
    
    struct FeatureMotionHistory {
        uint32 featureId;
        Vector3d positions[10];
        uint64 timestamps[10];
        uint32 numSamples;
        bool isStatic;
        
        FeatureMotionHistory() : featureId(0), numSamples(0), isStatic(true) {}
    };
    
    FeatureMotionHistory motionHistory_[MAX_FEATURES];
    uint32 numHistories_;
    
    float64 dynamicInlierRatio_;
    uint32 staticFeatureCount_;
    uint32 dynamicFeatureCount_;

public:
    DynamicObjectFilter() 
        : numHistories_(0)
        , dynamicInlierRatio_(1.0)
        , staticFeatureCount_(0)
        , dynamicFeatureCount_(0) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    void reset() {
        numHistories_ = 0;
        dynamicInlierRatio_ = 1.0;
        staticFeatureCount_ = 0;
        dynamicFeatureCount_ = 0;
    }
    
    float64 computeDynamicInlierRatio(const Feature* features, uint32 numFeatures,
                                      const MSCKFState& state) {
        uint32 totalFeatures = 0;
        uint32 staticFeatures = 0;
        
        for (uint32 i = 0; i < numFeatures; ++i) {
            const Feature& feat = features[i];
            
            if (feat.numObservations < config_.minFramesForConsistency) {
                continue;
            }
            
            totalFeatures++;
            
            if (checkFeatureStaticConsistency(feat, state)) {
                staticFeatures++;
            }
        }
        
        if (totalFeatures > 0) {
            dynamicInlierRatio_ = static_cast<float64>(staticFeatures) / totalFeatures;
        }
        
        staticFeatureCount_ = staticFeatures;
        dynamicFeatureCount_ = totalFeatures - staticFeatures;
        
        return dynamicInlierRatio_;
    }
    
    bool checkFeatureStaticConsistency(const Feature& feature, const MSCKFState& state) {
        if (feature.numObservations < config_.minFramesForConsistency) {
            return true;
        }
        
        if (!feature.isTriangulated) {
            return true;
        }
        
        Vector3d p_f = feature.position;
        
        float64 maxReprojError = 0;
        
        for (uint32 i = 0; i < feature.numObservations; ++i) {
            uint32 frameId = feature.observations[i].frameId;
            
            if (frameId >= state.numCameraStates) continue;
            
            const CameraState& camState = state.cameraStates[frameId];
            
            Matrix3d R_GC = camState.orientation.toRotationMatrix();
            Vector3d p_GC = camState.position;
            
            Matrix3d R_CG = R_GC.transpose();
            Vector3d p_C = R_CG * (p_f - p_GC);
            
            if (p_C[2] < MIN_FEATURE_DEPTH) continue;
            
            float64 u_proj = p_C[0] / p_C[2];
            float64 v_proj = p_C[1] / p_C[2];
            
            float64 du = feature.observations[i].u - u_proj;
            float64 dv = feature.observations[i].v - v_proj;
            float64 reprojError = sqrt(du * du + dv * dv);
            
            if (reprojError > maxReprojError) {
                maxReprojError = reprojError;
            }
        }
        
        return maxReprojError < config_.multiFrameConsistencyThreshold;
    }
    
    float64 getAdaptiveRansacThreshold() const {
        if (dynamicInlierRatio_ < config_.ransacInlierRatioThreshold) {
            return config_.epipolarThreshold * 0.5;
        }
        return config_.epipolarThreshold;
    }
    
    void filterDynamicFeatures(Feature* features, uint32& numFeatures, const MSCKFState& state) {
        uint32 writeIdx = 0;
        
        for (uint32 i = 0; i < numFeatures; ++i) {
            if (checkFeatureStaticConsistency(features[i], state)) {
                if (writeIdx != i) {
                    features[writeIdx] = features[i];
                }
                writeIdx++;
            }
        }
        
        numFeatures = writeIdx;
    }
    
    float64 getDynamicInlierRatio() const { return dynamicInlierRatio_; }
    uint32 getStaticFeatureCount() const { return staticFeatureCount_; }
    uint32 getDynamicFeatureCount() const { return dynamicFeatureCount_; }
};

class ShockRecoveryHandler {
public:
    struct Config {
        float64 shockAccelThreshold;
        float64 shockGyroThreshold;
        uint32 shockCountThreshold;
        float64 covarianceRecoveryScale;
        float64 biasRecoveryScale;
        uint64 recoveryTimeThreshold;
        
        Config() 
            : shockAccelThreshold(100.0)
            , shockGyroThreshold(30.0)
            , shockCountThreshold(3)
            , covarianceRecoveryScale(10.0)
            , biasRecoveryScale(5.0)
            , recoveryTimeThreshold(1000000) {}
    };

private:
    Config config_;
    
    bool inShock_;
    uint64 shockStartTime_;
    uint32 consecutiveShockCount_;
    
    bool inRecovery_;
    uint64 recoveryStartTime_;
    
    Vector3d preShockAccelBias_;
    Vector3d preShockGyroBias_;
    
    uint32 totalShockEvents_;

public:
    ShockRecoveryHandler() 
        : inShock_(false)
        , shockStartTime_(0)
        , consecutiveShockCount_(0)
        , inRecovery_(false)
        , recoveryStartTime_(0)
        , preShockAccelBias_({0, 0, 0})
        , preShockGyroBias_({0, 0, 0})
        , totalShockEvents_(0) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    bool detectShock(const Vector3d& accel, const Vector3d& gyro, uint64 timestamp) {
        float64 accelNorm = accel.norm();
        float64 gyroNorm = gyro.norm();
        
        bool isShock = (accelNorm > config_.shockAccelThreshold || 
                       gyroNorm > config_.shockGyroThreshold);
        
        if (isShock) {
            consecutiveShockCount_++;
            
            if (consecutiveShockCount_ == 1) {
                shockStartTime_ = timestamp;
            }
            
            if (consecutiveShockCount_ >= config_.shockCountThreshold && !inShock_) {
                inShock_ = true;
                totalShockEvents_++;
                return true;
            }
        } else {
            if (consecutiveShockCount_ > 0) {
                consecutiveShockCount_--;
            }
            
            if (inShock_ && consecutiveShockCount_ == 0) {
                inShock_ = false;
                inRecovery_ = true;
                recoveryStartTime_ = timestamp;
            }
        }
        
        return false;
    }
    
    void savePreShockState(const MSCKFState& state) {
        preShockAccelBias_ = state.imuState.accelBias;
        preShockGyroBias_ = state.imuState.gyroBias;
    }
    
    void applyShockRecovery(MSCKFState& state) {
        for (uint32 i = 0; i < IMU_STATE_DIM; ++i) {
            state.covariance(i, i) *= config_.covarianceRecoveryScale;
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            state.covariance(9 + i, 9 + i) *= config_.biasRecoveryScale;
            state.covariance(12 + i, 12 + i) *= config_.biasRecoveryScale;
        }
        
        state.covariance.symmetrize();
        
        enforceCovariancePositiveDefinite(state.covariance, state.covarianceDim);
    }
    
    void updateRecovery(MSCKFState& state, uint64 timestamp) {
        if (!inRecovery_) return;
        
        float64 elapsed = static_cast<float64>(timestamp - recoveryStartTime_);
        
        if (elapsed > config_.recoveryTimeThreshold) {
            inRecovery_ = false;
        }
    }
    
    bool isInShock() const { return inShock_; }
    bool isInRecovery() const { return inRecovery_; }
    uint32 getTotalShockEvents() const { return totalShockEvents_; }
    
    void reset() {
        inShock_ = false;
        inRecovery_ = false;
        consecutiveShockCount_ = 0;
        shockStartTime_ = 0;
        recoveryStartTime_ = 0;
    }
};

class LightingAdaptationHandler {
public:
    struct Config {
        float64 exposureTarget;
        float64 exposureMin;
        float64 exposureMax;
        float64 adaptationRate;
        uint32 histogramBins;
        float64 claheClipLimit;
        uint32 claheGridSize;
        
        Config() 
            : exposureTarget(128.0)
            , exposureMin(50.0)
            , exposureMax(200.0)
            , adaptationRate(0.1)
            , histogramBins(256)
            , claheClipLimit(2.0)
            , claheGridSize(8) {}
    };

private:
    Config config_;
    
    float64 currentExposure_;
    float64 avgBrightness_;
    float64 brightnessVariance_;
    
    uint32 histogram_[256];
    float64 cumulativeHist_[256];
    
    bool lightingChanged_;
    uint64 lastLightingChangeTime_;

public:
    LightingAdaptationHandler() 
        : currentExposure_(1.0)
        , avgBrightness_(128.0)
        , brightnessVariance_(0)
        , lightingChanged_(false)
        , lastLightingChangeTime_(0) {
        for (uint32 i = 0; i < 256; ++i) {
            histogram_[i] = 0;
            cumulativeHist_[i] = 0;
        }
    }
    
    void init(const Config& config) {
        config_ = config;
    }
    
    void analyzeImage(const uint8* image, uint32 width, uint32 height) {
        computeHistogram(image, width, height);
        
        computeStatistics();
        
        detectLightingChange();
    }
    
    void computeHistogram(const uint8* image, uint32 width, uint32 height) {
        uint32 size = width * height;
        
        for (uint32 i = 0; i < 256; ++i) {
            histogram_[i] = 0;
        }
        
        for (uint32 i = 0; i < size; ++i) {
            histogram_[image[i]]++;
        }
        
        cumulativeHist_[0] = static_cast<float64>(histogram_[0]) / size;
        for (uint32 i = 1; i < 256; ++i) {
            cumulativeHist_[i] = cumulativeHist_[i-1] + 
                                 static_cast<float64>(histogram_[i]) / size;
        }
    }
    
    void computeStatistics() {
        float64 sum = 0;
        float64 sumSq = 0;
        uint32 totalPixels = 0;
        
        for (uint32 i = 0; i < 256; ++i) {
            float64 count = static_cast<float64>(histogram_[i]);
            sum += i * count;
            sumSq += i * i * count;
            totalPixels += histogram_[i];
        }
        
        if (totalPixels > 0) {
            avgBrightness_ = sum / totalPixels;
            float64 avgSq = sumSq / totalPixels;
            brightnessVariance_ = avgSq - avgBrightness_ * avgBrightness_;
        }
    }
    
    void detectLightingChange() {
        float64 prevBrightness = avgBrightness_;
        
        float64 brightnessChange = abs(avgBrightness_ - prevBrightness);
        
        if (brightnessChange > 30.0 || brightnessVariance_ > 5000.0) {
            lightingChanged_ = true;
        } else {
            lightingChanged_ = false;
        }
    }
    
    void applyClahe(uint8* image, uint32 width, uint32 height) {
        uint32 gridSize = config_.claheGridSize;
        uint32 cellWidth = width / gridSize;
        uint32 cellHeight = height / gridSize;
        
        for (uint32 gy = 0; gy < gridSize; ++gy) {
            for (uint32 gx = 0; gx < gridSize; ++gx) {
                uint32 startX = gx * cellWidth;
                uint32 startY = gy * cellHeight;
                uint32 endX = (gx == gridSize - 1) ? width : (startX + cellWidth);
                uint32 endY = (gy == gridSize - 1) ? height : (startY + cellHeight);
                
                applyClaheToRegion(image, width, height, 
                                  startX, startY, endX, endY);
            }
        }
    }
    
    void applyClaheToRegion(uint8* image, uint32 width, uint32 height,
                            uint32 startX, uint32 startY, uint32 endX, uint32 endY) {
        uint32 cellHist[256] = {0};
        uint32 pixelCount = (endX - startX) * (endY - startY);
        
        for (uint32 y = startY; y < endY; ++y) {
            for (uint32 x = startX; x < endX; ++x) {
                cellHist[image[y * width + x]]++;
            }
        }
        
        float64 clipLimit = config_.claheClipLimit * pixelCount / 256.0;
        uint32 excess = 0;
        for (uint32 i = 0; i < 256; ++i) {
            if (cellHist[i] > clipLimit) {
                excess += cellHist[i] - static_cast<uint32>(clipLimit);
                cellHist[i] = static_cast<uint32>(clipLimit);
            }
        }
        
        uint32 redistribute = excess / 256;
        for (uint32 i = 0; i < 256; ++i) {
            cellHist[i] += redistribute;
        }
        
        float64 cumHist[256];
        cumHist[0] = static_cast<float64>(cellHist[0]) / pixelCount;
        for (uint32 i = 1; i < 256; ++i) {
            cumHist[i] = cumHist[i-1] + static_cast<float64>(cellHist[i]) / pixelCount;
        }
        
        for (uint32 y = startY; y < endY; ++y) {
            for (uint32 x = startX; x < endX; ++x) {
                uint8 pixel = image[y * width + x];
                float64 newValue = cumHist[pixel] * 255.0;
                image[y * width + x] = static_cast<uint8>(clamp(newValue, 0.0, 255.0));
            }
        }
    }
    
    float64 computeExposureCorrection() const {
        float64 error = config_.exposureTarget - avgBrightness_;
        return 1.0 + config_.adaptationRate * error / config_.exposureTarget;
    }
    
    bool hasLightingChanged() const { return lightingChanged_; }
    float64 getAvgBrightness() const { return avgBrightness_; }
    float64 getBrightnessVariance() const { return brightnessVariance_; }
};

class ExtremeScenarioManager {
public:
    struct Config {
        PureRotationHandler::Config pureRotationConfig;
        DynamicObjectFilter::Config dynamicObjectConfig;
        ShockRecoveryHandler::Config shockRecoveryConfig;
        LightingAdaptationHandler::Config lightingConfig;
        
        bool enablePureRotationHandling;
        bool enableDynamicObjectFiltering;
        bool enableShockRecovery;
        bool enableLightingAdaptation;
        
        Config() 
            : enablePureRotationHandling(true)
            , enableDynamicObjectFiltering(true)
            , enableShockRecovery(true)
            , enableLightingAdaptation(true) {}
    };

private:
    Config config_;
    
    PureRotationHandler pureRotationHandler_;
    DynamicObjectFilter dynamicObjectFilter_;
    ShockRecoveryHandler shockRecoveryHandler_;
    LightingAdaptationHandler lightingAdaptationHandler_;
    
    SafetyStatus overallStatus_;
    uint32 activeScenarios_;

public:
    ExtremeScenarioManager() 
        : overallStatus_(SafetyStatus::OK)
        , activeScenarios_(0) {}
    
    void init(const Config& config) {
        config_ = config;
        
        pureRotationHandler_.init(config.pureRotationConfig);
        dynamicObjectFilter_.init(config.dynamicObjectConfig);
        shockRecoveryHandler_.init(config.shockRecoveryConfig);
        lightingAdaptationHandler_.init(config.lightingConfig);
    }
    
    void processImuData(const Vector3d& accel, const Vector3d& gyro, 
                       const Vector3d& deltaTranslation, uint64 timestamp) {
        if (config_.enableShockRecovery) {
            shockRecoveryHandler_.detectShock(accel, gyro, timestamp);
        }
        
        if (config_.enablePureRotationHandling) {
            pureRotationHandler_.checkAndUpdate(deltaTranslation, deltaTranslation);
        }
    }
    
    void processImageData(const uint8* image, uint32 width, uint32 height) {
        if (config_.enableLightingAdaptation) {
            lightingAdaptationHandler_.analyzeImage(image, width, height);
        }
    }
    
    void processFeatures(Feature* features, uint32& numFeatures, MSCKFState& state) {
        if (config_.enableDynamicObjectFiltering) {
            dynamicObjectFilter_.computeDynamicInlierRatio(features, numFeatures, state);
            
            if (dynamicObjectFilter_.getDynamicInlierRatio() < 0.5) {
                dynamicObjectFilter_.filterDynamicFeatures(features, numFeatures, state);
            }
        }
    }
    
    void applyScenarioHandling(MSCKFState& state) {
        activeScenarios_ = 0;
        overallStatus_ = SafetyStatus::OK;
        
        if (shockRecoveryHandler_.isInShock()) {
            shockRecoveryHandler_.applyShockRecovery(state);
            activeScenarios_++;
            overallStatus_ = SafetyStatus::CRITICAL;
        }
        
        if (shockRecoveryHandler_.isInRecovery()) {
            shockRecoveryHandler_.updateRecovery(state, state.timestamp);
            activeScenarios_++;
            if (overallStatus_ < SafetyStatus::WARNING) {
                overallStatus_ = SafetyStatus::WARNING;
            }
        }
        
        if (pureRotationHandler_.isInPureRotationMode()) {
            pureRotationHandler_.inflateCovarianceForUnobservable(state);
            activeScenarios_++;
            if (overallStatus_ < SafetyStatus::WARNING) {
                overallStatus_ = SafetyStatus::WARNING;
            }
        }
    }
    
    bool isInPureRotationMode() const {
        return pureRotationHandler_.isInPureRotationMode();
    }
    
    bool isInShock() const {
        return shockRecoveryHandler_.isInShock();
    }
    
    bool isInRecovery() const {
        return shockRecoveryHandler_.isInRecovery();
    }
    
    bool hasLightingChanged() const {
        return lightingAdaptationHandler_.hasLightingChanged();
    }
    
    float64 getDynamicInlierRatio() const {
        return dynamicObjectFilter_.getDynamicInlierRatio();
    }
    
    SafetyStatus getOverallStatus() const { return overallStatus_; }
    uint32 getActiveScenarioCount() const { return activeScenarios_; }
    
    void reset() {
        pureRotationHandler_.exitPureRotationMode();
        shockRecoveryHandler_.reset();
        dynamicObjectFilter_.reset();
        overallStatus_ = SafetyStatus::OK;
        activeScenarios_ = 0;
    }
};

}

#endif
