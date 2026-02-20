#ifndef MSCKF_UTILITY_AUTO_TUNING_HPP
#define MSCKF_UTILITY_AUTO_TUNING_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../core/state.hpp"
#include "../hal/imu_types.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 ALLAN_SAMPLES_MAX = 10000;
constexpr uint32 TEXTURE_GRID_SIZE = 8;
constexpr uint32 PARAM_GRID_SIZE = 10;

struct AllanVarianceResult {
    float64 gyroNoiseDensity;
    float64 gyroBiasRandomWalk;
    float64 accelNoiseDensity;
    float64 accelBiasRandomWalk;
    float64 gyroBiasInstability;
    float64 accelBiasInstability;
    float64 correlationTime;
    bool isValid;
    
    AllanVarianceResult() 
        : gyroNoiseDensity(1e-4)
        , gyroBiasRandomWalk(1e-6)
        , accelNoiseDensity(1e-3)
        , accelBiasRandomWalk(1e-5)
        , gyroBiasInstability(1e-5)
        , accelBiasInstability(1e-4)
        , correlationTime(100)
        , isValid(false) {}
};

struct TextureAnalysisResult {
    float64 averageScore;
    float64 variance;
    float64 minScore;
    float64 maxScore;
    float64 uniformity;
    uint32 numFeatures;
    uint32 gridCells;
    TextureLevel level;
    
    TextureAnalysisResult() 
        : averageScore(0)
        , variance(0)
        , minScore(0)
        , maxScore(0)
        , uniformity(0)
        , numFeatures(0)
        , gridCells(TEXTURE_GRID_SIZE * TEXTURE_GRID_SIZE)
        , level(TextureLevel::MEDIUM) {}
};

enum class TextureLevel : uint8 {
    VERY_LOW = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    VERY_HIGH = 4
};

struct TuningParameters {
    IMUNoiseParams imuNoiseParams;
    float64 observationNoise;
    float64 maxReprojectionError;
    uint32 maxFeatures;
    float64 minTrackLength;
    float64 featureQualityThreshold;
    
    TuningParameters() 
        : observationNoise(1.0)
        , maxReprojectionError(3.0)
        , maxFeatures(300)
        , minTrackLength(3)
        , featureQualityThreshold(0.01) {}
};

class AllanVarianceEstimator {
public:
    struct Config {
        uint32 maxSamples;
        uint32 minSamples;
        float64 samplePeriod;
        uint32 maxClusterSize;
        
        Config() 
            : maxSamples(ALLAN_SAMPLES_MAX)
            , minSamples(1000)
            , samplePeriod(0.002)
            , maxClusterSize(100) {}
    };

private:
    Config config_;
    
    Vector3d gyroSamples_[ALLAN_SAMPLES_MAX];
    Vector3d accelSamples_[ALLAN_SAMPLES_MAX];
    uint64 timestamps_[ALLAN_SAMPLES_MAX];
    uint32 sampleCount_;
    
    float64 gyroAllanVar_[100];
    float64 accelAllanVar_[100];
    float64 tau_[100];
    uint32 numTau_;
    
    bool isStationary_;
    float64 stationaryThreshold_;
    uint32 stationaryCount_;
    
    AllanVarianceResult result_;

public:
    AllanVarianceEstimator() 
        : sampleCount_(0)
        , numTau_(0)
        , isStationary_(false)
        , stationaryThreshold_(0.01)
        , stationaryCount_(0) {
        memset(gyroAllanVar_, 0, sizeof(gyroAllanVar_));
        memset(accelAllanVar_, 0, sizeof(accelAllanVar_));
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        sampleCount_ = 0;
        numTau_ = 0;
        isStationary_ = false;
        stationaryCount_ = 0;
    }
    
    void addSample(const IMUData& imuData) {
        if (sampleCount_ >= config_.maxSamples) return;
        
        gyroSamples_[sampleCount_] = imuData.gyro;
        accelSamples_[sampleCount_] = imuData.accel;
        timestamps_[sampleCount_] = imuData.timestamp;
        sampleCount_++;
        
        checkStationary();
    }
    
    void checkStationary() {
        if (sampleCount_ < 100) {
            isStationary_ = false;
            return;
        }
        
        uint32 startIdx = sampleCount_ - 100;
        
        Vector3d gyroMean({0, 0, 0});
        Vector3d accelMean({0, 0, 0});
        
        for (uint32 i = startIdx; i < sampleCount_; ++i) {
            gyroMean += gyroSamples_[i];
            accelMean += accelSamples_[i];
        }
        gyroMean = gyroMean / 100;
        accelMean = accelMean / 100;
        
        float64 gyroVar = 0, accelVar = 0;
        for (uint32 i = startIdx; i < sampleCount_; ++i) {
            gyroVar += (gyroSamples_[i] - gyroMean).norm();
            accelVar += (accelSamples_[i] - accelMean).norm();
        }
        gyroVar /= 100;
        accelVar /= 100;
        
        isStationary_ = (gyroVar < stationaryThreshold_ && accelVar < stationaryThreshold_ * 10);
        
        if (isStationary_) {
            stationaryCount_++;
        } else {
            stationaryCount_ = 0;
        }
    }
    
    void computeAllanVariance() {
        if (sampleCount_ < config_.minSamples) return;
        
        numTau_ = 0;
        uint32 tau = 1;
        
        while (tau < sampleCount_ / 3 && numTau_ < 100) {
            float64 gyroVar = computeAllanVarianceForTau(gyroSamples_, sampleCount_, tau);
            float64 accelVar = computeAllanVarianceForTau(accelSamples_, sampleCount_, tau);
            
            gyroAllanVar_[numTau_] = gyroVar;
            accelAllanVar_[numTau_] = accelVar;
            tau_[numTau_] = tau * config_.samplePeriod;
            
            numTau_++;
            tau = tau * 2;
        }
        
        fitAllanParameters();
    }
    
    float64 computeAllanVarianceForTau(const Vector3d* samples, uint32 n, uint32 tau) {
        uint32 m = n / tau;
        if (m < 2) return 0;
        
        float64 sum = 0;
        uint32 count = 0;
        
        for (uint32 k = 0; k < 3; ++k) {
            float64* clusterMeans = new float64[m];
            
            for (uint32 i = 0; i < m; ++i) {
                float64 sum = 0;
                for (uint32 j = 0; j < tau; ++j) {
                    sum += samples[i * tau + j][k];
                }
                clusterMeans[i] = sum / tau;
            }
            
            for (uint32 i = 0; i < m - 1; ++i) {
                float64 diff = clusterMeans[i + 1] - clusterMeans[i];
                sum += diff * diff;
                count++;
            }
            
            delete[] clusterMeans;
        }
        
        return sum / (2 * count);
    }
    
    void fitAllanParameters() {
        if (numTau_ < 5) return;
        
        float64 gyroMinVar = gyroAllanVar_[0];
        uint32 gyroMinIdx = 0;
        float64 accelMinVar = accelAllanVar_[0];
        uint32 accelMinIdx = 0;
        
        for (uint32 i = 1; i < numTau_; ++i) {
            if (gyroAllanVar_[i] < gyroMinVar) {
                gyroMinVar = gyroAllanVar_[i];
                gyroMinIdx = i;
            }
            if (accelAllanVar_[i] < accelMinVar) {
                accelMinVar = accelAllanVar_[i];
                accelMinIdx = i;
            }
        }
        
        result_.gyroBiasInstability = sqrt(gyroMinVar);
        result_.accelBiasInstability = sqrt(accelMinVar);
        result_.correlationTime = tau_[gyroMinIdx];
        
        if (numTau_ >= 2) {
            float64 slope = (log(gyroAllanVar_[1]) - log(gyroAllanVar_[0])) /
                           (log(tau_[1]) - log(tau_[0]));
            
            if (abs(slope + 0.5) < 0.3) {
                result_.gyroNoiseDensity = sqrt(gyroAllanVar_[0] * tau_[0]);
            }
            
            slope = (log(accelAllanVar_[1]) - log(accelAllanVar_[0])) /
                   (log(tau_[1]) - log(tau_[0]));
            
            if (abs(slope + 0.5) < 0.3) {
                result_.accelNoiseDensity = sqrt(accelAllanVar_[0] * tau_[0]);
            }
        }
        
        if (numTau_ >= 3) {
            float64 slope = (log(gyroAllanVar_[2]) - log(gyroAllanVar_[1])) /
                           (log(tau_[2]) - log(tau_[1]));
            
            if (abs(slope - 0.5) < 0.3) {
                result_.gyroBiasRandomWalk = sqrt(gyroAllanVar_[2] / tau_[2]);
            }
            
            slope = (log(accelAllanVar_[2]) - log(accelAllanVar_[1])) /
                   (log(tau_[2]) - log(tau_[1]));
            
            if (abs(slope - 0.5) < 0.3) {
                result_.accelBiasRandomWalk = sqrt(accelAllanVar_[2] / tau_[2]);
            }
        }
        
        result_.isValid = true;
    }
    
    const AllanVarianceResult& getResult() const { return result_; }
    bool isStationary() const { return isStationary_ && stationaryCount_ > 100; }
    uint32 getSampleCount() const { return sampleCount_; }
    bool hasEnoughSamples() const { return sampleCount_ >= config_.minSamples; }
    
    void reset() {
        sampleCount_ = 0;
        numTau_ = 0;
        stationaryCount_ = 0;
        result_.isValid = false;
    }
};

class TextureAnalyzer {
public:
    struct Config {
        uint32 gridSize;
        float64 lowThreshold;
        float64 highThreshold;
        uint32 minFeaturesPerCell;
        
        Config() 
            : gridSize(TEXTURE_GRID_SIZE)
            , lowThreshold(0.005)
            , highThreshold(0.02)
            , minFeaturesPerCell(3) {}
    };

private:
    Config config_;
    
    float64 gridScores_[TEXTURE_GRID_SIZE * TEXTURE_GRID_SIZE];
    uint32 gridFeatureCounts_[TEXTURE_GRID_SIZE * TEXTURE_GRID_SIZE];
    
    TextureAnalysisResult result_;

public:
    TextureAnalyzer() {
        memset(gridScores_, 0, sizeof(gridScores_));
        memset(gridFeatureCounts_, 0, sizeof(gridFeatureCounts_));
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        memset(gridScores_, 0, sizeof(gridScores_));
        memset(gridFeatureCounts_, 0, sizeof(gridFeatureCounts_));
    }
    
    void analyze(const uint8* image, uint32 width, uint32 height,
                 const Feature2D* features, uint32 numFeatures) {
        uint32 cellWidth = width / config_.gridSize;
        uint32 cellHeight = height / config_.gridSize;
        
        memset(gridScores_, 0, sizeof(gridScores_));
        memset(gridFeatureCounts_, 0, sizeof(gridFeatureCounts_));
        
        for (uint32 i = 0; i < numFeatures; ++i) {
            uint32 gx = min(static_cast<uint32>(features[i].x) / cellWidth, config_.gridSize - 1);
            uint32 gy = min(static_cast<uint32>(features[i].y) / cellHeight, config_.gridSize - 1);
            uint32 idx = gy * config_.gridSize + gx;
            
            gridScores_[idx] += features[i].score;
            gridFeatureCounts_[idx]++;
        }
        
        for (uint32 i = 0; i < config_.gridSize * config_.gridSize; ++i) {
            if (gridFeatureCounts_[i] > 0) {
                gridScores_[i] /= gridFeatureCounts_[i];
            }
        }
        
        computeStatistics(numFeatures);
    }
    
    void computeStatistics(uint32 numFeatures) {
        result_.numFeatures = numFeatures;
        result_.averageScore = 0;
        result_.minScore = 1e10;
        result_.maxScore = 0;
        result_.gridCells = config_.gridSize * config_.gridSize;
        
        uint32 nonEmptyCells = 0;
        
        for (uint32 i = 0; i < result_.gridCells; ++i) {
            if (gridFeatureCounts_[i] > 0) {
                result_.averageScore += gridScores_[i];
                result_.minScore = min(result_.minScore, gridScores_[i]);
                result_.maxScore = max(result_.maxScore, gridScores_[i]);
                nonEmptyCells++;
            }
        }
        
        if (nonEmptyCells > 0) {
            result_.averageScore /= nonEmptyCells;
        }
        
        result_.uniformity = static_cast<float64>(nonEmptyCells) / result_.gridCells;
        
        result_.variance = 0;
        for (uint32 i = 0; i < result_.gridCells; ++i) {
            if (gridFeatureCounts_[i] > 0) {
                float64 diff = gridScores_[i] - result_.averageScore;
                result_.variance += diff * diff;
            }
        }
        if (nonEmptyCells > 1) {
            result_.variance /= (nonEmptyCells - 1);
        }
        
        classifyTexture();
    }
    
    void classifyTexture() {
        float64 score = result_.averageScore * result_.uniformity;
        
        if (score < config_.lowThreshold) {
            result_.level = TextureLevel::VERY_LOW;
        } else if (score < config_.lowThreshold * 2) {
            result_.level = TextureLevel::LOW;
        } else if (score < config_.highThreshold) {
            result_.level = TextureLevel::MEDIUM;
        } else if (score < config_.highThreshold * 2) {
            result_.level = TextureLevel::HIGH;
        } else {
            result_.level = TextureLevel::VERY_HIGH;
        }
    }
    
    const TextureAnalysisResult& getResult() const { return result_; }
    TextureLevel getTextureLevel() const { return result_.level; }
    
    void reset() {
        memset(gridScores_, 0, sizeof(gridScores_));
        memset(gridFeatureCounts_, 0, sizeof(gridFeatureCounts_));
        result_ = TextureAnalysisResult();
    }
};

class AutoTuning {
public:
    struct Config {
        bool enableIMUTuning;
        bool enableVisualTuning;
        bool enableOnlineTuning;
        float64 tuningInterval;
        uint32 minSamplesForTuning;
        float64 adaptationRate;
        
        Config() 
            : enableIMUTuning(true)
            , enableVisualTuning(true)
            , enableOnlineTuning(true)
            , tuningInterval(10000000)
            , minSamplesForTuning(100)
            , adaptationRate(0.1) {}
    };

private:
    Config config_;
    
    AllanVarianceEstimator allanEstimator_;
    TextureAnalyzer textureAnalyzer_;
    
    TuningParameters currentParams_;
    TuningParameters targetParams_;
    
    uint64 lastTuningTime_;
    uint32 tuningCount_;
    
    float64 performanceHistory_[100];
    uint32 perfHistoryHead_;
    uint32 perfHistorySize_;
    
    float64 bestScore_;
    TuningParameters bestParams_;
    
    bool isCalibratingIMU_;
    bool needsRetuning_;

public:
    AutoTuning() 
        : lastTuningTime_(0)
        , tuningCount_(0)
        , perfHistoryHead_(0)
        , perfHistorySize_(0)
        , bestScore_(0)
        , isCalibratingIMU_(false)
        , needsRetuning_(false) {
        memset(performanceHistory_, 0, sizeof(performanceHistory_));
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        allanEstimator_.init();
        textureAnalyzer_.init();
        lastTuningTime_ = 0;
        tuningCount_ = 0;
        isCalibratingIMU_ = false;
        needsRetuning_ = false;
    }
    
    void addIMUSample(const IMUData& imuData) {
        if (!config_.enableIMUTuning) return;
        
        allanEstimator_.addSample(imuData);
        
        if (isCalibratingIMU_ && allanEstimator_.hasEnoughSamples()) {
            allanEstimator_.computeAllanVariance();
            
            const AllanVarianceResult& result = allanEstimator_.getResult();
            if (result.isValid) {
                targetParams_.imuNoiseParams.gyroNoiseDensity = result.gyroNoiseDensity;
                targetParams_.imuNoiseParams.gyroBiasRandomWalk = result.gyroBiasRandomWalk;
                targetParams_.imuNoiseParams.accelNoiseDensity = result.accelNoiseDensity;
                targetParams_.imuNoiseParams.accelBiasRandomWalk = result.accelBiasRandomWalk;
                
                isCalibratingIMU_ = false;
                needsRetuning_ = true;
            }
        }
    }
    
    void analyzeTexture(const uint8* image, uint32 width, uint32 height,
                        const Feature2D* features, uint32 numFeatures) {
        if (!config_.enableVisualTuning) return;
        
        textureAnalyzer_.analyze(image, width, height, features, numFeatures);
        
        adjustVisualParameters();
    }
    
    void adjustVisualParameters() {
        const TextureAnalysisResult& result = textureAnalyzer_.getResult();
        
        switch (result.level) {
            case TextureLevel::VERY_LOW:
                targetParams_.maxFeatures = 500;
                targetParams_.featureQualityThreshold = 0.005;
                targetParams_.observationNoise = 2.0;
                targetParams_.maxReprojectionError = 5.0;
                break;
                
            case TextureLevel::LOW:
                targetParams_.maxFeatures = 400;
                targetParams_.featureQualityThreshold = 0.008;
                targetParams_.observationNoise = 1.5;
                targetParams_.maxReprojectionError = 4.0;
                break;
                
            case TextureLevel::MEDIUM:
                targetParams_.maxFeatures = 300;
                targetParams_.featureQualityThreshold = 0.01;
                targetParams_.observationNoise = 1.0;
                targetParams_.maxReprojectionError = 3.0;
                break;
                
            case TextureLevel::HIGH:
                targetParams_.maxFeatures = 250;
                targetParams_.featureQualityThreshold = 0.015;
                targetParams_.observationNoise = 0.8;
                targetParams_.maxReprojectionError = 2.5;
                break;
                
            case TextureLevel::VERY_HIGH:
                targetParams_.maxFeatures = 200;
                targetParams_.featureQualityThreshold = 0.02;
                targetParams_.observationNoise = 0.5;
                targetParams_.maxReprojectionError = 2.0;
                break;
        }
        
        needsRetuning_ = true;
    }
    
    void update(uint64 currentTime) {
        if (!config_.enableOnlineTuning) return;
        
        if (currentTime - lastTuningTime_ > config_.tuningInterval && needsRetuning_) {
            applyTuning();
            lastTuningTime_ = currentTime;
            tuningCount_++;
        }
    }
    
    void applyTuning() {
        currentParams_.imuNoiseParams.gyroNoiseDensity = 
            adapt(currentParams_.imuNoiseParams.gyroNoiseDensity, 
                  targetParams_.imuNoiseParams.gyroNoiseDensity);
        currentParams_.imuNoiseParams.gyroBiasRandomWalk = 
            adapt(currentParams_.imuNoiseParams.gyroBiasRandomWalk, 
                  targetParams_.imuNoiseParams.gyroBiasRandomWalk);
        currentParams_.imuNoiseParams.accelNoiseDensity = 
            adapt(currentParams_.imuNoiseParams.accelNoiseDensity, 
                  targetParams_.imuNoiseParams.accelNoiseDensity);
        currentParams_.imuNoiseParams.accelBiasRandomWalk = 
            adapt(currentParams_.imuNoiseParams.accelBiasRandomWalk, 
                  targetParams_.imuNoiseParams.accelBiasRandomWalk);
        
        currentParams_.maxFeatures = static_cast<uint32>(
            adapt(currentParams_.maxFeatures, targetParams_.maxFeatures));
        currentParams_.featureQualityThreshold = 
            adapt(currentParams_.featureQualityThreshold, targetParams_.featureQualityThreshold);
        currentParams_.observationNoise = 
            adapt(currentParams_.observationNoise, targetParams_.observationNoise);
        currentParams_.maxReprojectionError = 
            adapt(currentParams_.maxReprojectionError, targetParams_.maxReprojectionError);
        
        needsRetuning_ = false;
    }
    
    float64 adapt(float64 current, float64 target) {
        return current * (1 - config_.adaptationRate) + target * config_.adaptationRate;
    }
    
    void recordPerformance(float64 score) {
        performanceHistory_[perfHistoryHead_] = score;
        perfHistoryHead_ = (perfHistoryHead_ + 1) % 100;
        
        if (perfHistorySize_ < 100) {
            perfHistorySize_++;
        }
        
        if (score > bestScore_) {
            bestScore_ = score;
            bestParams_ = currentParams_;
        }
    }
    
    void startIMUCalibration() {
        isCalibratingIMU_ = true;
        allanEstimator_.reset();
    }
    
    void stopIMUCalibration() {
        isCalibratingIMU_ = false;
    }
    
    void gridSearchOptimize(const MSCKFState& state, Feature* features, uint32 numFeatures) {
        float64 param1Range[PARAM_GRID_SIZE];
        float64 param2Range[PARAM_GRID_SIZE];
        
        for (uint32 i = 0; i < PARAM_GRID_SIZE; ++i) {
            param1Range[i] = 0.5 + i * 0.1;
            param2Range[i] = 0.5 + i * 0.1;
        }
        
        float64 bestGridScore = -1e10;
        float64 bestP1 = currentParams_.observationNoise;
        float64 bestP2 = currentParams_.maxReprojectionError;
        
        for (uint32 i = 0; i < PARAM_GRID_SIZE; ++i) {
            for (uint32 j = 0; j < PARAM_GRID_SIZE; ++j) {
                float64 score = evaluateParameters(state, features, numFeatures,
                                                   param1Range[i], param2Range[j]);
                
                if (score > bestGridScore) {
                    bestGridScore = score;
                    bestP1 = param1Range[i];
                    bestP2 = param2Range[j];
                }
            }
        }
        
        targetParams_.observationNoise = bestP1;
        targetParams_.maxReprojectionError = bestP2;
        needsRetuning_ = true;
    }
    
    float64 evaluateParameters(const MSCKFState& state, Feature* features, uint32 numFeatures,
                               float64 obsNoise, float64 maxReproj) {
        float64 score = 0;
        
        uint32 validFeatures = 0;
        for (uint32 i = 0; i < numFeatures; ++i) {
            if (features[i].numObservations >= 3) {
                validFeatures++;
            }
        }
        
        score += validFeatures * 0.1;
        
        float64 traceCov = 0;
        for (uint32 i = 0; i < 3; ++i) {
            traceCov += state.covariance(i, i);
        }
        score -= traceCov * 0.01;
        
        score -= obsNoise * 0.5;
        score -= maxReproj * 0.1;
        
        return score;
    }
    
    const TuningParameters& getCurrentParameters() const { return currentParams_; }
    const TuningParameters& getTargetParameters() const { return targetParams_; }
    const TuningParameters& getBestParameters() const { return bestParams_; }
    const AllanVarianceResult& getAllanResult() const { return allanEstimator_.getResult(); }
    const TextureAnalysisResult& getTextureResult() const { return textureAnalyzer_.getResult(); }
    
    bool isCalibrating() const { return isCalibratingIMU_; }
    bool needsRetuning() const { return needsRetuning_; }
    uint32 getTuningCount() const { return tuningCount_; }
    float64 getBestScore() const { return bestScore_; }
    
    void reset() {
        allanEstimator_.reset();
        textureAnalyzer_.reset();
        tuningCount_ = 0;
        isCalibratingIMU_ = false;
        needsRetuning_ = false;
        perfHistorySize_ = 0;
        bestScore_ = 0;
    }
};

}

#endif
