#ifndef MSCKF_CORE_ADAPTIVE_KERNEL_HPP
#define MSCKF_CORE_ADAPTIVE_KERNEL_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include <cstring>

namespace msckf {

enum class KernelType : uint8 {
    HUBER = 0,
    CAUCHY = 1,
    TUKEY = 2,
    MCCLURE = 3,
    GEMAN_MCCLURE = 4
};

class AdaptiveRobustKernel {
public:
    struct Config {
        float64 initialThreshold;
        float64 minThreshold;
        float64 maxThreshold;
        float64 adaptationRate;
        float64 outlierRatioThreshold;
        uint32 residualHistorySize;
        bool useAdaptiveThreshold;
        bool useKernelSwitching;
        KernelType defaultKernel;
        
        Config() 
            : initialThreshold(1.345)
            , minThreshold(0.5)
            , maxThreshold(5.0)
            , adaptationRate(0.1)
            , outlierRatioThreshold(0.3)
            , residualHistorySize(100)
            , useAdaptiveThreshold(true)
            , useKernelSwitching(true)
            , defaultKernel(KernelType::HUBER) {}
    };

private:
    Config config_;
    
    float64 currentThreshold_;
    float64 outlierRatio_;
    KernelType currentKernel_;
    
    float64 residualHistory_[100];
    uint32 historyHead_;
    uint32 historySize_;
    
    float64 mad_;
    float64 median_;
    float64 sigma_;
    
    uint32 totalResiduals_;
    uint32 outlierCount_;
    
    float64 kernelWeights_[5];
    float64 convergenceSpeed_;
    float64 lastCost_;

public:
    AdaptiveRobustKernel() 
        : currentThreshold_(1.345)
        , outlierRatio_(0)
        , currentKernel_(KernelType::HUBER)
        , historyHead_(0)
        , historySize_(0)
        , mad_(0)
        , median_(0)
        , sigma_(1)
        , totalResiduals_(0)
        , outlierCount_(0)
        , convergenceSpeed_(0)
        , lastCost_(0) {
        memset(residualHistory_, 0, sizeof(residualHistory_));
        for (uint32 i = 0; i < 5; ++i) {
            kernelWeights_[i] = 1.0;
        }
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        currentThreshold_ = config.initialThreshold;
        currentKernel_ = config.defaultKernel;
        historyHead_ = 0;
        historySize_ = 0;
        outlierRatio_ = 0;
        totalResiduals_ = 0;
        outlierCount_ = 0;
        lastCost_ = 0;
    }
    
    float64 computeWeight(float64 residual) {
        float64 absR = abs(residual);
        float64 normalizedR = absR / currentThreshold_;
        
        float64 weight = 1.0;
        
        switch (currentKernel_) {
            case KernelType::HUBER:
                weight = huberWeight(normalizedR);
                break;
            case KernelType::CAUCHY:
                weight = cauchyWeight(normalizedR);
                break;
            case KernelType::TUKEY:
                weight = tukeyWeight(normalizedR);
                break;
            case KernelType::MCCLURE:
                weight = mcclureWeight(normalizedR);
                break;
            case KernelType::GEMAN_MCCLURE:
                weight = gemanMcclureWeight(normalizedR);
                break;
        }
        
        return weight;
    }
    
    float64 computeCost(float64 residual) {
        float64 absR = abs(residual);
        float64 normalizedR = absR / currentThreshold_;
        
        float64 cost = 0;
        
        switch (currentKernel_) {
            case KernelType::HUBER:
                cost = huberCost(normalizedR);
                break;
            case KernelType::CAUCHY:
                cost = cauchyCost(normalizedR);
                break;
            case KernelType::TUKEY:
                cost = tukeyCost(normalizedR);
                break;
            case KernelType::MCCLURE:
                cost = mcclureCost(normalizedR);
                break;
            case KernelType::GEMAN_MCCLURE:
                cost = gemanMcclureCost(normalizedR);
                break;
        }
        
        return cost * currentThreshold_ * currentThreshold_;
    }
    
    float64 huberWeight(float64 r) {
        if (r <= 1.0) {
            return 1.0;
        } else {
            return 1.0 / r;
        }
    }
    
    float64 huberCost(float64 r) {
        if (r <= 1.0) {
            return 0.5 * r * r;
        } else {
            return r - 0.5;
        }
    }
    
    float64 cauchyWeight(float64 r) {
        return 1.0 / (1.0 + r * r);
    }
    
    float64 cauchyCost(float64 r) {
        return 0.5 * log(1.0 + r * r);
    }
    
    float64 tukeyWeight(float64 r) {
        if (r <= 1.0) {
            float64 t = 1.0 - r * r;
            return t * t;
        } else {
            return 0;
        }
    }
    
    float64 tukeyCost(float64 r) {
        if (r <= 1.0) {
            float64 t = 1.0 - r * r;
            return (1.0/6.0) * (1.0 - t * t * t);
        } else {
            return 1.0/6.0;
        }
    }
    
    float64 mcclureWeight(float64 r) {
        return 1.0 / ((1.0 + r * r) * (1.0 + r * r));
    }
    
    float64 mcclureCost(float64 r) {
        return r * r / (2.0 * (1.0 + r * r));
    }
    
    float64 gemanMcclureWeight(float64 r) {
        float64 r2 = r * r;
        return (1.0 + r2) / ((1.0 + r2) * (1.0 + r2));
    }
    
    float64 gemanMcclureCost(float64 r) {
        float64 r2 = r * r;
        return r2 / (1.0 + r2);
    }
    
    void addResidual(float64 residual) {
        residualHistory_[historyHead_] = residual;
        historyHead_ = (historyHead_ + 1) % config_.residualHistorySize;
        
        if (historySize_ < config_.residualHistorySize) {
            historySize_++;
        }
        
        totalResiduals_++;
        
        if (abs(residual) > currentThreshold_ * 2.5) {
            outlierCount_++;
        }
    }
    
    void addResiduals(const float64* residuals, uint32 count) {
        for (uint32 i = 0; i < count; ++i) {
            addResidual(residuals[i]);
        }
    }
    
    void updateThreshold() {
        if (!config_.useAdaptiveThreshold || historySize_ < 10) return;
        
        computeStatistics();
        
        float64 kernelConstant;
        switch (currentKernel_) {
            case KernelType::HUBER:
                kernelConstant = 1.345;
                break;
            case KernelType::CAUCHY:
                kernelConstant = 2.385;
                break;
            case KernelType::TUKEY:
                kernelConstant = 4.685;
                break;
            case KernelType::MCCLURE:
            case KernelType::GEMAN_MCCLURE:
                kernelConstant = 1.345;
                break;
            default:
                kernelConstant = 1.345;
        }
        
        float64 newThreshold = kernelConstant * mad_;
        
        newThreshold = clamp(newThreshold, config_.minThreshold, config_.maxThreshold);
        
        currentThreshold_ = currentThreshold_ * (1.0 - config_.adaptationRate) + 
                           newThreshold * config_.adaptationRate;
        
        if (config_.useKernelSwitching) {
            updateKernelType();
        }
    }
    
    void computeStatistics() {
        if (historySize_ == 0) return;
        
        float64 sortedResiduals[100];
        for (uint32 i = 0; i < historySize_; ++i) {
            sortedResiduals[i] = abs(residualHistory_[i]);
        }
        
        for (uint32 i = 0; i < historySize_ - 1; ++i) {
            for (uint32 j = i + 1; j < historySize_; ++j) {
                if (sortedResiduals[j] < sortedResiduals[i]) {
                    float64 temp = sortedResiduals[i];
                    sortedResiduals[i] = sortedResiduals[j];
                    sortedResiduals[j] = temp;
                }
            }
        }
        
        if (historySize_ % 2 == 0) {
            median_ = (sortedResiduals[historySize_ / 2 - 1] + 
                      sortedResiduals[historySize_ / 2]) / 2.0;
        } else {
            median_ = sortedResiduals[historySize_ / 2];
        }
        
        float64 deviations[100];
        for (uint32 i = 0; i < historySize_; ++i) {
            deviations[i] = abs(sortedResiduals[i] - median_);
        }
        
        for (uint32 i = 0; i < historySize_ - 1; ++i) {
            for (uint32 j = i + 1; j < historySize_; ++j) {
                if (deviations[j] < deviations[i]) {
                    float64 temp = deviations[i];
                    deviations[i] = deviations[j];
                    deviations[j] = temp;
                }
            }
        }
        
        if (historySize_ % 2 == 0) {
            mad_ = (deviations[historySize_ / 2 - 1] + 
                   deviations[historySize_ / 2]) / 2.0;
        } else {
            mad_ = deviations[historySize_ / 2];
        }
        
        mad_ = mad_ * 1.4826;
        
        sigma_ = 0;
        for (uint32 i = 0; i < historySize_; ++i) {
            sigma_ += residualHistory_[i] * residualHistory_[i];
        }
        sigma_ = sqrt(sigma_ / historySize_);
    }
    
    void updateKernelType() {
        if (historySize_ < 20) return;
        
        outlierRatio_ = static_cast<float64>(outlierCount_) / totalResiduals_;
        
        float64 kurtosis = computeKurtosis();
        
        KernelType bestKernel = currentKernel_;
        float64 bestScore = evaluateKernel(currentKernel_);
        
        for (uint32 i = 0; i < 5; ++i) {
            KernelType testKernel = static_cast<KernelType>(i);
            float64 score = evaluateKernel(testKernel);
            
            if (score > bestScore) {
                bestScore = score;
                bestKernel = testKernel;
            }
        }
        
        currentKernel_ = bestKernel;
    }
    
    float64 computeKurtosis() {
        if (historySize_ < 4) return 0;
        
        float64 mean = 0;
        for (uint32 i = 0; i < historySize_; ++i) {
            mean += residualHistory_[i];
        }
        mean /= historySize_;
        
        float64 m2 = 0, m4 = 0;
        for (uint32 i = 0; i < historySize_; ++i) {
            float64 diff = residualHistory_[i] - mean;
            m2 += diff * diff;
            m4 += diff * diff * diff * diff;
        }
        m2 /= historySize_;
        m4 /= historySize_;
        
        if (m2 < 1e-10) return 0;
        
        return m4 / (m2 * m2) - 3.0;
    }
    
    float64 evaluateKernel(KernelType kernel) {
        float64 score = 0;
        
        float64 testThreshold = currentThreshold_;
        
        for (uint32 i = 0; i < historySize_; ++i) {
            float64 r = abs(residualHistory_[i]) / testThreshold;
            
            float64 weight;
            switch (kernel) {
                case KernelType::HUBER:
                    weight = huberWeight(r);
                    break;
                case KernelType::CAUCHY:
                    weight = cauchyWeight(r);
                    break;
                case KernelType::TUKEY:
                    weight = tukeyWeight(r);
                    break;
                case KernelType::MCCLURE:
                    weight = mcclureWeight(r);
                    break;
                case KernelType::GEMAN_MCCLURE:
                    weight = gemanMcclureWeight(r);
                    break;
                default:
                    weight = 1.0;
            }
            
            score += weight * residualHistory_[i] * residualHistory_[i];
        }
        
        float64 inlierRatio = 1.0 - outlierRatio_;
        score *= (1.0 + inlierRatio);
        
        if (kernel == KernelType::CAUCHY && outlierRatio_ > 0.2) {
            score *= 1.2;
        }
        
        if (kernel == KernelType::TUKEY && outlierRatio_ < 0.1) {
            score *= 1.1;
        }
        
        return score;
    }
    
    void updateConvergenceSpeed(float64 currentCost) {
        if (lastCost_ > 0) {
            float64 relativeChange = abs(currentCost - lastCost_) / (lastCost_ + 1e-10);
            convergenceSpeed_ = convergenceSpeed_ * 0.9 + relativeChange * 0.1;
        }
        lastCost_ = currentCost;
        
        if (convergenceSpeed_ < 0.01) {
            float64 newThreshold = currentThreshold_ * 0.95;
            currentThreshold_ = max(newThreshold, config_.minThreshold);
        } else if (convergenceSpeed_ > 0.1) {
            float64 newThreshold = currentThreshold_ * 1.05;
            currentThreshold_ = min(newThreshold, config_.maxThreshold);
        }
    }
    
    void applyWeightsToResiduals(float64* residuals, float64* weights, uint32 count) {
        for (uint32 i = 0; i < count; ++i) {
            weights[i] = computeWeight(residuals[i]);
            residuals[i] *= weights[i];
        }
    }
    
    void applyWeightsToJacobian(float64* H, const float64* weights, 
                                uint32 rows, uint32 cols) {
        for (uint32 i = 0; i < rows; ++i) {
            float64 w = sqrt(weights[i]);
            for (uint32 j = 0; j < cols; ++j) {
                H[i * cols + j] *= w;
            }
        }
    }
    
    void reset() {
        historyHead_ = 0;
        historySize_ = 0;
        outlierRatio_ = 0;
        totalResiduals_ = 0;
        outlierCount_ = 0;
        currentThreshold_ = config_.initialThreshold;
        currentKernel_ = config_.defaultKernel;
        lastCost_ = 0;
        convergenceSpeed_ = 0;
    }
    
    float64 getThreshold() const { return currentThreshold_; }
    float64 getOutlierRatio() const { return outlierRatio_; }
    KernelType getCurrentKernel() const { return currentKernel_; }
    float64 getMAD() const { return mad_; }
    float64 getMedian() const { return median_; }
    float64 getSigma() const { return sigma_; }
    float64 getConvergenceSpeed() const { return convergenceSpeed_; }
    uint32 getHistorySize() const { return historySize_; }
    
    const char* getKernelName() const {
        switch (currentKernel_) {
            case KernelType::HUBER: return "Huber";
            case KernelType::CAUCHY: return "Cauchy";
            case KernelType::TUKEY: return "Tukey";
            case KernelType::MCCLURE: return "McClure";
            case KernelType::GEMAN_MCCLURE: return "Geman-McClure";
            default: return "Unknown";
        }
    }
};

class MultiScaleRobustKernel {
public:
    struct ScaleLevel {
        float64 threshold;
        float64 weight;
        uint32 inlierCount;
        uint32 outlierCount;
        
        ScaleLevel() : threshold(1.0), weight(1.0), inlierCount(0), outlierCount(0) {}
    };

private:
    AdaptiveRobustKernel baseKernel_;
    ScaleLevel scales_[5];
    uint32 numScales_;
    float64 scaleFactors_[5];
    
public:
    MultiScaleRobustKernel() : numScales_(3) {
        scaleFactors_[0] = 0.5;
        scaleFactors_[1] = 1.0;
        scaleFactors_[2] = 2.0;
        scaleFactors_[3] = 4.0;
        scaleFactors_[4] = 8.0;
    }
    
    void init(uint32 numScales = 3) {
        numScales_ = min(numScales, 5u);
        baseKernel_.init();
        
        for (uint32 i = 0; i < numScales_; ++i) {
            scales_[i].threshold = baseKernel_.getThreshold() * scaleFactors_[i];
            scales_[i].weight = 1.0;
            scales_[i].inlierCount = 0;
            scales_[i].outlierCount = 0;
        }
    }
    
    float64 computeMultiScaleWeight(float64 residual) {
        float64 totalWeight = 0;
        float64 totalInfluence = 0;
        
        for (uint32 i = 0; i < numScales_; ++i) {
            float64 normalizedR = abs(residual) / scales_[i].threshold;
            float64 w = baseKernel_.huberWeight(normalizedR);
            
            totalWeight += w * scales_[i].weight;
            totalInfluence += scales_[i].weight;
            
            if (normalizedR <= 1.0) {
                scales_[i].inlierCount++;
            } else {
                scales_[i].outlierCount++;
            }
        }
        
        return totalWeight / totalInfluence;
    }
    
    void updateScaleWeights() {
        for (uint32 i = 0; i < numScales_; ++i) {
            uint32 total = scales_[i].inlierCount + scales_[i].outlierCount;
            if (total > 0) {
                float64 inlierRatio = static_cast<float64>(scales_[i].inlierCount) / total;
                scales_[i].weight = inlierRatio;
            }
        }
    }
    
    void reset() {
        baseKernel_.reset();
        for (uint32 i = 0; i < numScales_; ++i) {
            scales_[i].inlierCount = 0;
            scales_[i].outlierCount = 0;
        }
    }
};

}

#endif
