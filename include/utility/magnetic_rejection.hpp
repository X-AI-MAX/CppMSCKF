#ifndef MSCKF_UTILITY_MAGNETIC_REJECTION_HPP
#define MSCKF_UTILITY_MAGNETIC_REJECTION_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../core/state.hpp"
#include "../hal/imu_types.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 MAG_HISTORY_SIZE = 100;
constexpr uint32 ESSENTIAL_MATRIX_SAMPLES = 8;

struct MagneticFieldMeasurement {
    Vector3d field;
    uint64 timestamp;
    float64 norm;
    float64 variance;
    
    MagneticFieldMeasurement() 
        : field({0, 0, 0})
        , timestamp(0)
        , norm(0)
        , variance(0) {}
};

struct MagneticCalibration {
    Vector3d bias;
    Vector3d scale;
    Matrix3d softIron;
    bool isValid;
    float64 residualError;
    
    MagneticCalibration() 
        : bias({0, 0, 0})
        , scale({1, 1, 1})
        , isValid(false)
        , residualError(0) {
        softIron = Matrix3d::identity();
    }
};

enum class MagneticState : uint8 {
    NORMAL = 0,
    DISTURBANCE_DETECTED = 1,
    HIGH_DISTURBANCE = 2,
    MAGNETIC_REJECTED = 3
};

class MagneticRejection {
public:
    struct Config {
        float64 nominalFieldStrength;
        float64 fieldStrengthTolerance;
        float64 varianceThreshold;
        float64 angularDeviationThreshold;
        float64 detectionWindow;
        float64 recoveryTime;
        float64 transitionSmoothingFactor;
        uint32 minSamplesForDetection;
        bool enableAutoCalibration;
        bool enableVisualHeadingFallback;
        
        Config() 
            : nominalFieldStrength(0.5)
            , fieldStrengthTolerance(0.15)
            , varianceThreshold(0.05)
            , angularDeviationThreshold(0.3)
            , detectionWindow(500000)
            , recoveryTime(2000000)
            , transitionSmoothingFactor(0.1)
            , minSamplesForDetection(5)
            , enableAutoCalibration(true)
            , enableVisualHeadingFallback(true) {}
    };

private:
    Config config_;
    
    MagneticFieldMeasurement magHistory_[MAG_HISTORY_SIZE];
    uint32 historyHead_;
    uint32 historySize_;
    
    MagneticCalibration calibration_;
    MagneticState currentState_;
    
    Vector3d referenceField_;
    Vector3d expectedField_;
    bool referenceSet_;
    
    float64 currentVariance_;
    float64 currentDeviation_;
    float64 fieldStrengthError_;
    
    uint64 disturbanceStartTime_;
    uint64 lastNormalTime_;
    uint32 consecutiveDisturbanceCount_;
    
    float64 magneticWeight_;
    float64 visualWeight_;
    float64 headingBlendFactor_;
    
    Vector3d visualHeading_;
    bool visualHeadingValid_;
    
    Quaterniond lastMagneticOrientation_;
    Quaterniond lastVisualOrientation_;
    Quaterniond blendedOrientation_;
    
    uint32 totalSamples_;
    uint32 disturbanceSamples_;
    uint32 rejectedSamples_;

public:
    MagneticRejection() 
        : historyHead_(0)
        , historySize_(0)
        , currentState_(MagneticState::NORMAL)
        , referenceSet_(false)
        , currentVariance_(0)
        , currentDeviation_(0)
        , fieldStrengthError_(0)
        , disturbanceStartTime_(0)
        , lastNormalTime_(0)
        , consecutiveDisturbanceCount_(0)
        , magneticWeight_(1.0)
        , visualWeight_(0.0)
        , headingBlendFactor_(1.0)
        , visualHeadingValid_(false)
        , totalSamples_(0)
        , disturbanceSamples_(0)
        , rejectedSamples_(0) {
        referenceField_ = Vector3d({0, 0, 0});
        expectedField_ = Vector3d({0, 0, config_.nominalFieldStrength});
        visualHeading_ = Vector3d({1, 0, 0});
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        historyHead_ = 0;
        historySize_ = 0;
        currentState_ = MagneticState::NORMAL;
        referenceSet_ = false;
        magneticWeight_ = 1.0;
        visualWeight_ = 0.0;
        consecutiveDisturbanceCount_ = 0;
        totalSamples_ = 0;
        disturbanceSamples_ = 0;
        rejectedSamples_ = 0;
    }
    
    void setReferenceField(const Vector3d& field) {
        referenceField_ = field;
        referenceSet_ = true;
        
        expectedField_ = field.normalized() * config_.nominalFieldStrength;
    }
    
    void processMagneticField(const Vector3d& rawField, uint64 timestamp) {
        Vector3d field = applyCalibration(rawField);
        
        addToHistory(field, timestamp);
        
        computeStatistics();
        
        detectDisturbance(timestamp);
        
        updateWeights(timestamp);
        
        totalSamples_++;
    }
    
    Vector3d applyCalibration(const Vector3d& rawField) {
        if (!calibration_.isValid) {
            return rawField;
        }
        
        Vector3d corrected = rawField - calibration_.bias;
        
        corrected[0] *= calibration_.scale[0];
        corrected[1] *= calibration_.scale[1];
        corrected[2] *= calibration_.scale[2];
        
        return calibration_.softIron * corrected;
    }
    
    void addToHistory(const Vector3d& field, uint64 timestamp) {
        MagneticFieldMeasurement& meas = magHistory_[historyHead_];
        meas.field = field;
        meas.timestamp = timestamp;
        meas.norm = field.norm();
        
        historyHead_ = (historyHead_ + 1) % MAG_HISTORY_SIZE;
        if (historySize_ < MAG_HISTORY_SIZE) {
            historySize_++;
        }
    }
    
    void computeStatistics() {
        if (historySize_ < config_.minSamplesForDetection) return;
        
        Vector3d mean({0, 0, 0});
        float64 meanNorm = 0;
        
        uint32 count = min(historySize_, config_.minSamplesForDetection);
        uint32 startIdx = (historyHead_ + MAG_HISTORY_SIZE - count) % MAG_HISTORY_SIZE;
        
        for (uint32 i = 0; i < count; ++i) {
            uint32 idx = (startIdx + i) % MAG_HISTORY_SIZE;
            mean += magHistory_[idx].field;
            meanNorm += magHistory_[idx].norm;
        }
        mean = mean / count;
        meanNorm = meanNorm / count;
        
        currentVariance_ = 0;
        for (uint32 i = 0; i < count; ++i) {
            uint32 idx = (startIdx + i) % MAG_HISTORY_SIZE;
            float64 diff = magHistory_[idx].norm - meanNorm;
            currentVariance_ += diff * diff;
        }
        currentVariance_ = sqrt(currentVariance_ / count);
        
        if (referenceSet_) {
            Vector3d normalizedField = mean.normalized();
            Vector3d normalizedRef = referenceField_.normalized();
            
            float64 dot = normalizedField.dot(normalizedRef);
            dot = clamp(dot, -1.0, 1.0);
            currentDeviation_ = acos(dot);
        }
        
        fieldStrengthError_ = abs(meanNorm - config_.nominalFieldStrength) / config_.nominalFieldStrength;
    }
    
    void detectDisturbance(uint64 timestamp) {
        bool isDisturbed = false;
        
        if (currentVariance_ > config_.varianceThreshold) {
            isDisturbed = true;
        }
        
        if (fieldStrengthError_ > config_.fieldStrengthTolerance) {
            isDisturbed = true;
        }
        
        if (currentDeviation_ > config_.angularDeviationThreshold) {
            isDisturbed = true;
        }
        
        if (isDisturbed) {
            consecutiveDisturbanceCount_++;
            
            if (consecutiveDisturbanceCount_ == 1) {
                disturbanceStartTime_ = timestamp;
            }
            
            disturbanceSamples_++;
            
            if (currentState_ == MagneticState::NORMAL) {
                currentState_ = MagneticState::DISTURBANCE_DETECTED;
            } else if (consecutiveDisturbanceCount_ > 10) {
                currentState_ = MagneticState::HIGH_DISTURBANCE;
            } else if (consecutiveDisturbanceCount_ > 20) {
                currentState_ = MagneticState::MAGNETIC_REJECTED;
                rejectedSamples_++;
            }
        } else {
            consecutiveDisturbanceCount_ = 0;
            lastNormalTime_ = timestamp;
            
            if (currentState_ != MagneticState::NORMAL) {
                uint64 timeSinceNormal = timestamp - lastNormalTime_;
                if (timeSinceNormal > config_.recoveryTime) {
                    currentState_ = MagneticState::NORMAL;
                }
            }
        }
    }
    
    void updateWeights(uint64 timestamp) {
        float64 targetMagneticWeight = 1.0;
        float64 targetVisualWeight = 0.0;
        
        switch (currentState_) {
            case MagneticState::NORMAL:
                targetMagneticWeight = 1.0;
                targetVisualWeight = 0.0;
                break;
                
            case MagneticState::DISTURBANCE_DETECTED:
                targetMagneticWeight = 0.7;
                targetVisualWeight = 0.3;
                break;
                
            case MagneticState::HIGH_DISTURBANCE:
                targetMagneticWeight = 0.3;
                targetVisualWeight = 0.7;
                break;
                
            case MagneticState::MAGNETIC_REJECTED:
                targetMagneticWeight = 0.0;
                targetVisualWeight = 1.0;
                break;
        }
        
        magneticWeight_ = magneticWeight_ * (1 - config_.transitionSmoothingFactor) +
                         targetMagneticWeight * config_.transitionSmoothingFactor;
        visualWeight_ = visualWeight_ * (1 - config_.transitionSmoothingFactor) +
                       targetVisualWeight * config_.transitionSmoothingFactor;
        
        float64 sum = magneticWeight_ + visualWeight_;
        if (sum > 0) {
            magneticWeight_ /= sum;
            visualWeight_ /= sum;
        }
    }
    
    void updateVisualHeading(const Vector2d* prevFeatures, const Vector2d* currFeatures,
                             uint32 numFeatures, const Matrix3d& K) {
        if (numFeatures < ESSENTIAL_MATRIX_SAMPLES) {
            visualHeadingValid_ = false;
            return;
        }
        
        Matrix3d E = computeEssentialMatrix(prevFeatures, currFeatures, numFeatures, K);
        
        Vector3d translation = extractTranslationFromEssential(E);
        
        if (translation.norm() > 0.01) {
            visualHeading_ = translation.normalized();
            visualHeading_[2] = 0;
            visualHeading_ = visualHeading_.normalized();
            visualHeadingValid_ = true;
        } else {
            visualHeadingValid_ = false;
        }
    }
    
    Matrix3d computeEssentialMatrix(const Vector2d* prev, const Vector2d* curr,
                                     uint32 numPoints, const Matrix3d& K) {
        Matrix3d K_inv = K.inverse();
        
        float64 A_data[8 * 9];
        
        uint32 usedPoints = min(numPoints, 8u);
        
        for (uint32 i = 0; i < usedPoints; ++i) {
            Vector3d p1({prev[i][0], prev[i][1], 1.0});
            Vector3d p2({curr[i][0], curr[i][1], 1.0});
            
            p1 = K_inv * p1;
            p2 = K_inv * p2;
            
            float64* row = A_data + i * 9;
            row[0] = p2[0] * p1[0];
            row[1] = p2[0] * p1[1];
            row[2] = p2[0];
            row[3] = p2[1] * p1[0];
            row[4] = p2[1] * p1[1];
            row[5] = p2[1];
            row[6] = p1[0];
            row[7] = p1[1];
            row[8] = 1.0;
        }
        
        float64 e[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        
        Matrix3d E;
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                E(i, j) = e[i * 3 + j];
            }
        }
        
        return E;
    }
    
    Vector3d extractTranslationFromEssential(const Matrix3d& E) {
        Matrix3d W;
        W(0, 0) = 0; W(0, 1) = -1; W(0, 2) = 0;
        W(1, 0) = 1; W(1, 1) = 0; W(1, 2) = 0;
        W(2, 0) = 0; W(2, 1) = 0; W(2, 2) = 1;
        
        Matrix3d U, Vt;
        float64 S[3];
        svdDecompose(E, U, S, Vt);
        
        Vector3d t;
        t[0] = U(0, 2);
        t[1] = U(1, 2);
        t[2] = U(2, 2);
        
        return t;
    }
    
    void svdDecompose(const Matrix3d& A, Matrix3d& U, float64* S, Matrix3d& Vt) {
        Matrix3d AtA = A.transpose() * A;
        
        float64 eigenvalues[3];
        Matrix3d eigenvectors;
        eigenDecompose3x3(AtA, eigenvalues, eigenvectors);
        
        for (uint32 i = 0; i < 3; ++i) {
            S[i] = sqrt(max(eigenvalues[i], 0.0));
        }
        
        Vt = eigenvectors.transpose();
        
        for (uint32 i = 0; i < 3; ++i) {
            if (S[i] > 1e-10) {
                Vector3d u = A * Vector3d({Vt(0, i), Vt(1, i), Vt(2, i)}) / S[i];
                for (uint32 j = 0; j < 3; ++j) {
                    U(j, i) = u[j];
                }
            }
        }
    }
    
    void eigenDecompose3x3(const Matrix3d& A, float64* eigenvalues, Matrix3d& eigenvectors) {
        eigenvalues[0] = A(0, 0);
        eigenvalues[1] = A(1, 1);
        eigenvalues[2] = A(2, 2);
        
        eigenvectors = Matrix3d::identity();
    }
    
    Quaterniond computeBlendedOrientation(const Quaterniond& magneticOrientation,
                                          const Quaterniond& visualOrientation) {
        lastMagneticOrientation_ = magneticOrientation;
        lastVisualOrientation_ = visualOrientation;
        
        if (!visualHeadingValid_ || visualWeight_ < 0.01) {
            blendedOrientation_ = magneticOrientation;
            return blendedOrientation_;
        }
        
        if (magneticWeight_ < 0.01) {
            blendedOrientation_ = visualOrientation;
            return blendedOrientation_;
        }
        
        float64 yawMag = magneticOrientation.toEulerAngles()[2];
        float64 yawVis = visualOrientation.toEulerAngles()[2];
        
        float64 diff = yawVis - yawMag;
        while (diff > PI) diff -= 2 * PI;
        while (diff < -PI) diff += 2 * PI;
        
        float64 blendedYaw = yawMag + diff * visualWeight_;
        
        float64 pitch = magneticOrientation.toEulerAngles()[1];
        float64 roll = magneticOrientation.toEulerAngles()[0];
        
        blendedOrientation_ = Quaterniond::fromEulerAngles(roll, pitch, blendedYaw);
        
        return blendedOrientation_;
    }
    
    float64 getHeadingWeight() const {
        return magneticWeight_;
    }
    
    bool shouldUseMagneticHeading() const {
        return magneticWeight_ > 0.5;
    }
    
    bool shouldUseVisualHeading() const {
        return visualWeight_ > 0.5;
    }
    
    bool isDisturbanceDetected() const {
        return currentState_ >= MagneticState::DISTURBANCE_DETECTED;
    }
    
    bool isMagneticRejected() const {
        return currentState_ == MagneticState::MAGNETIC_REJECTED;
    }
    
    void calibrateMagnetometer(const Vector3d* samples, uint32 numSamples) {
        if (numSamples < 10) return;
        
        Vector3d minVal({1e10, 1e10, 1e10});
        Vector3d maxVal({-1e10, -1e10, -1e10});
        
        for (uint32 i = 0; i < numSamples; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                if (samples[i][j] < minVal[j]) minVal[j] = samples[i][j];
                if (samples[i][j] > maxVal[j]) maxVal[j] = samples[i][j];
            }
        }
        
        calibration_.bias = (minVal + maxVal) * 0.5;
        
        Vector3d range = maxVal - minVal;
        float64 avgRange = (range[0] + range[1] + range[2]) / 3.0;
        
        for (uint32 j = 0; j < 3; ++j) {
            calibration_.scale[j] = avgRange / range[j];
        }
        
        calibration_.isValid = true;
        
        float64 residual = 0;
        for (uint32 i = 0; i < numSamples; ++i) {
            Vector3d corrected = applyCalibration(samples[i]);
            float64 r = abs(corrected.norm() - config_.nominalFieldStrength);
            residual += r;
        }
        calibration_.residualError = residual / numSamples;
    }
    
    MagneticState getCurrentState() const { return currentState_; }
    float64 getCurrentVariance() const { return currentVariance_; }
    float64 getCurrentDeviation() const { return currentDeviation_; }
    float64 getFieldStrengthError() const { return fieldStrengthError_; }
    float64 getMagneticWeight() const { return magneticWeight_; }
    float64 getVisualWeight() const { return visualWeight_; }
    const MagneticCalibration& getCalibration() const { return calibration_; }
    const Quaterniond& getBlendedOrientation() const { return blendedOrientation_; }
    
    uint32 getTotalSamples() const { return totalSamples_; }
    uint32 getDisturbanceSamples() const { return disturbanceSamples_; }
    uint32 getRejectedSamples() const { return rejectedSamples_; }
    float64 getDisturbanceRatio() const {
        return totalSamples_ > 0 ? static_cast<float64>(disturbanceSamples_) / totalSamples_ : 0;
    }
    
    void reset() {
        historyHead_ = 0;
        historySize_ = 0;
        currentState_ = MagneticState::NORMAL;
        magneticWeight_ = 1.0;
        visualWeight_ = 0.0;
        consecutiveDisturbanceCount_ = 0;
        totalSamples_ = 0;
        disturbanceSamples_ = 0;
        rejectedSamples_ = 0;
        referenceSet_ = false;
        visualHeadingValid_ = false;
    }
};

}

#endif
