#ifndef MSCKF_UTILITY_BATTERY_MANAGER_HPP
#define MSCKF_UTILITY_BATTERY_MANAGER_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../core/state.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 BATTERY_HISTORY_SIZE = 100;
constexpr uint32 POWER_SAMPLES_SIZE = 50;

enum class BatteryAlertLevel : uint8 {
    NORMAL = 0,
    LOW_POWER = 1,
    CRITICAL = 2,
    EMERGENCY = 3
};

struct BatteryState {
    float64 voltage;
    float64 current;
    float64 chargeRemaining;
    float64 chargeCapacity;
    float64 percentage;
    float64 temperature;
    float64 health;
    uint64 timestamp;
    
    BatteryState() 
        : voltage(0)
        , current(0)
        , chargeRemaining(0)
        , chargeCapacity(5000)
        , percentage(100)
        , temperature(25)
        , health(100)
        , timestamp(0) {}
};

struct PowerModel {
    float64 hoverPower;
    float64 cruisePower;
    float64 climbPower;
    float64 descentPower;
    float64 basePower;
    
    float64 motorEfficiency;
    float64 batteryEfficiency;
    float64 windFactor;
    
    PowerModel() 
        : hoverPower(150)
        , cruisePower(120)
        , climbPower(200)
        , descentPower(80)
        , basePower(10)
        , motorEfficiency(0.85)
        , batteryEfficiency(0.95)
        , windFactor(1.0) {}
};

struct FlightPlan {
    Vector3d waypoints[20];
    uint32 numWaypoints;
    float64 totalDistance;
    float64 estimatedTime;
    float64 estimatedPower;
    
    FlightPlan() 
        : numWaypoints(0)
        , totalDistance(0)
        , estimatedTime(0)
        , estimatedPower(0) {}
};

class BatteryManager {
public:
    struct Config {
        float64 nominalVoltage;
        float64 minVoltage;
        float64 maxVoltage;
        float64 nominalCapacity;
        float64 lowPowerThreshold;
        float64 criticalThreshold;
        float64 emergencyThreshold;
        float64 voltageToPercentSlope;
        float64 voltageToPercentOffset;
        float64 currentIntegrationWeight;
        float64 voltageMeasurementWeight;
        uint32 smoothingWindowSize;
        bool enableAdaptiveModel;
        bool enablePredictiveAlert;
        
        Config() 
            : nominalVoltage(11.1)
            , minVoltage(9.0)
            , maxVoltage(12.6)
            , nominalCapacity(5000)
            , lowPowerThreshold(30.0)
            , criticalThreshold(20.0)
            , emergencyThreshold(10.0)
            , voltageToPercentSlope(33.33)
            , voltageToPercentOffset(-300)
            , currentIntegrationWeight(0.7)
            , voltageMeasurementWeight(0.3)
            , smoothingWindowSize(5)
            , enableAdaptiveModel(true)
            , enablePredictiveAlert(true) {}
    };

private:
    Config config_;
    BatteryState batteryState_;
    PowerModel powerModel_;
    
    float64 voltageHistory_[BATTERY_HISTORY_SIZE];
    float64 currentHistory_[BATTERY_HISTORY_SIZE];
    uint64 timestampHistory_[BATTERY_HISTORY_SIZE];
    uint32 historyHead_;
    uint32 historySize_;
    
    float64 powerSamples_[POWER_SAMPLES_SIZE];
    uint32 powerSampleHead_;
    uint32 powerSampleSize_;
    
    float64 avgPowerConsumption_;
    float64 instantPower_;
    float64 powerConsumptionRate_;
    
    float64 estimatedFlightTime_;
    float64 estimatedFlightDistance_;
    
    BatteryAlertLevel alertLevel_;
    Vector3d nearestLandingPoint_;
    float64 distanceToNearestLanding_;
    
    float64 totalFlightTime_;
    float64 totalEnergyConsumed_;
    
    float64 updateFrequencyScale_;
    bool lowPowerMode_;

public:
    BatteryManager() 
        : historyHead_(0)
        , historySize_(0)
        , powerSampleHead_(0)
        , powerSampleSize_(0)
        , avgPowerConsumption_(0)
        , instantPower_(0)
        , powerConsumptionRate_(0)
        , estimatedFlightTime_(0)
        , estimatedFlightDistance_(0)
        , alertLevel_(BatteryAlertLevel::NORMAL)
        , distanceToNearestLanding_(0)
        , totalFlightTime_(0)
        , totalEnergyConsumed_(0)
        , updateFrequencyScale_(1.0)
        , lowPowerMode_(false) {
        memset(voltageHistory_, 0, sizeof(voltageHistory_));
        memset(currentHistory_, 0, sizeof(currentHistory_));
        memset(timestampHistory_, 0, sizeof(timestampHistory_));
        memset(powerSamples_, 0, sizeof(powerSamples_));
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        batteryState_.chargeCapacity = config.nominalCapacity;
        batteryState_.chargeRemaining = config.nominalCapacity;
        batteryState_.percentage = 100;
        
        historyHead_ = 0;
        historySize_ = 0;
        powerSampleHead_ = 0;
        powerSampleSize_ = 0;
        
        alertLevel_ = BatteryAlertLevel::NORMAL;
        updateFrequencyScale_ = 1.0;
        lowPowerMode_ = false;
    }
    
    void update(const BatteryState& measurement) {
        addToHistory(measurement.voltage, measurement.current, measurement.timestamp);
        
        batteryState_.voltage = smoothVoltage(measurement.voltage);
        batteryState_.current = smoothCurrent(measurement.current);
        batteryState_.temperature = measurement.temperature;
        batteryState_.timestamp = measurement.timestamp;
        
        instantPower_ = batteryState_.voltage * batteryState_.current;
        addPowerSample(instantPower_);
        avgPowerConsumption_ = computeAveragePower();
        
        updateChargeEstimate();
        updateAlertLevel();
        updateEstimates();
        
        if (config_.enableAdaptiveModel) {
            adaptPowerModel();
        }
    }
    
    void addToHistory(float64 voltage, float64 current, uint64 timestamp) {
        voltageHistory_[historyHead_] = voltage;
        currentHistory_[historyHead_] = current;
        timestampHistory_[historyHead_] = timestamp;
        
        historyHead_ = (historyHead_ + 1) % BATTERY_HISTORY_SIZE;
        if (historySize_ < BATTERY_HISTORY_SIZE) {
            historySize_++;
        }
    }
    
    void addPowerSample(float64 power) {
        powerSamples_[powerSampleHead_] = power;
        powerSampleHead_ = (powerSampleHead_ + 1) % POWER_SAMPLES_SIZE;
        if (powerSampleSize_ < POWER_SAMPLES_SIZE) {
            powerSampleSize_++;
        }
    }
    
    float64 smoothVoltage(float64 rawVoltage) {
        if (historySize_ < config_.smoothingWindowSize) {
            return rawVoltage;
        }
        
        float64 sum = 0;
        uint32 count = 0;
        uint32 startIdx = (historyHead_ + BATTERY_HISTORY_SIZE - config_.smoothingWindowSize) 
                         % BATTERY_HISTORY_SIZE;
        
        for (uint32 i = 0; i < config_.smoothingWindowSize; ++i) {
            uint32 idx = (startIdx + i) % BATTERY_HISTORY_SIZE;
            sum += voltageHistory_[idx];
            count++;
        }
        
        return sum / count;
    }
    
    float64 smoothCurrent(float64 rawCurrent) {
        if (historySize_ < config_.smoothingWindowSize) {
            return rawCurrent;
        }
        
        float64 sum = 0;
        uint32 count = 0;
        uint32 startIdx = (historyHead_ + BATTERY_HISTORY_SIZE - config_.smoothingWindowSize) 
                         % BATTERY_HISTORY_SIZE;
        
        for (uint32 i = 0; i < config_.smoothingWindowSize; ++i) {
            uint32 idx = (startIdx + i) % BATTERY_HISTORY_SIZE;
            sum += currentHistory_[idx];
            count++;
        }
        
        return sum / count;
    }
    
    float64 computeAveragePower() {
        if (powerSampleSize_ == 0) return 0;
        
        float64 sum = 0;
        for (uint32 i = 0; i < powerSampleSize_; ++i) {
            sum += powerSamples_[i];
        }
        
        return sum / powerSampleSize_;
    }
    
    void updateChargeEstimate() {
        float64 voltagePercent = estimatePercentageFromVoltage(batteryState_.voltage);
        
        if (historySize_ >= 2) {
            uint32 prevIdx = (historyHead_ + BATTERY_HISTORY_SIZE - 2) % BATTERY_HISTORY_SIZE;
            uint64 dt = batteryState_.timestamp - timestampHistory_[prevIdx];
            float64 dtHours = dt / 3600000000.0;
            
            float64 coulombs = batteryState_.current * dtHours * 3600;
            batteryState_.chargeRemaining -= coulombs;
            batteryState_.chargeRemaining = max(batteryState_.chargeRemaining, 0.0);
            
            float64 currentPercent = batteryState_.chargeRemaining / batteryState_.chargeCapacity * 100;
            
            batteryState_.percentage = config_.currentIntegrationWeight * currentPercent +
                                       config_.voltageMeasurementWeight * voltagePercent;
        } else {
            batteryState_.percentage = voltagePercent;
        }
        
        batteryState_.percentage = clamp(batteryState_.percentage, 0.0, 100.0);
    }
    
    float64 estimatePercentageFromVoltage(float64 voltage) {
        float64 percent = (voltage - config_.minVoltage) / 
                         (config_.maxVoltage - config_.minVoltage) * 100;
        return clamp(percent, 0.0, 100.0);
    }
    
    void updateAlertLevel() {
        BatteryAlertLevel newLevel = BatteryAlertLevel::NORMAL;
        
        if (batteryState_.percentage <= config_.emergencyThreshold) {
            newLevel = BatteryAlertLevel::EMERGENCY;
        } else if (batteryState_.percentage <= config_.criticalThreshold) {
            newLevel = BatteryAlertLevel::CRITICAL;
        } else if (batteryState_.percentage <= config_.lowPowerThreshold) {
            newLevel = BatteryAlertLevel::LOW_POWER;
        }
        
        if (newLevel != alertLevel_) {
            handleAlertTransition(alertLevel_, newLevel);
            alertLevel_ = newLevel;
        }
    }
    
    void handleAlertTransition(BatteryAlertLevel oldLevel, BatteryAlertLevel newLevel) {
        if (newLevel == BatteryAlertLevel::LOW_POWER) {
            updateFrequencyScale_ = 0.8;
        } else if (newLevel == BatteryAlertLevel::CRITICAL) {
            lowPowerMode_ = true;
            updateFrequencyScale_ = 0.5;
        } else if (newLevel == BatteryAlertLevel::EMERGENCY) {
            lowPowerMode_ = true;
            updateFrequencyScale_ = 0.3;
        } else {
            lowPowerMode_ = false;
            updateFrequencyScale_ = 1.0;
        }
    }
    
    void updateEstimates() {
        if (avgPowerConsumption_ > 0) {
            float64 remainingEnergy = batteryState_.chargeRemaining * batteryState_.voltage / 1000;
            estimatedFlightTime_ = remainingEnergy / avgPowerConsumption_ * 60;
            
            float64 avgSpeed = 5.0;
            estimatedFlightDistance_ = estimatedFlightTime_ / 60 * avgSpeed;
        }
    }
    
    void adaptPowerModel() {
        if (powerSampleSize_ < 10) return;
        
        float64 recentAvg = 0;
        uint32 recentCount = min(powerSampleSize_, 10u);
        uint32 startIdx = (powerSampleHead_ + POWER_SAMPLES_SIZE - recentCount) % POWER_SAMPLES_SIZE;
        
        for (uint32 i = 0; i < recentCount; ++i) {
            recentAvg += powerSamples_[(startIdx + i) % POWER_SAMPLES_SIZE];
        }
        recentAvg /= recentCount;
        
        powerModel_.hoverPower = powerModel_.hoverPower * 0.95 + recentAvg * 0.05;
    }
    
    float64 estimatePowerForDistance(float64 distance, const Vector3d& velocity) {
        float64 speed = velocity.norm();
        
        float64 power = powerModel_.basePower;
        
        if (speed > 0.1) {
            power += powerModel_.cruisePower * (speed / 10.0);
        } else {
            power += powerModel_.hoverPower;
        }
        
        if (velocity[2] > 0.1) {
            power += powerModel_.climbPower * (velocity[2] / 5.0);
        } else if (velocity[2] < -0.1) {
            power += powerModel_.descentPower * (-velocity[2] / 5.0);
        }
        
        power *= powerModel_.windFactor;
        power /= powerModel_.motorEfficiency * powerModel_.batteryEfficiency;
        
        return power;
    }
    
    float64 estimatePowerForPath(const Vector3d* waypoints, uint32 numWaypoints,
                                  const Vector3d& currentPosition) {
        if (numWaypoints == 0) return 0;
        
        float64 totalPower = 0;
        Vector3d prevPos = currentPosition;
        
        for (uint32 i = 0; i < numWaypoints; ++i) {
            Vector3d diff = waypoints[i] - prevPos;
            float64 dist = diff.norm();
            
            Vector3d velocity = diff.normalized() * 5.0;
            float64 segmentPower = estimatePowerForDistance(dist, velocity);
            
            float64 time = dist / 5.0;
            totalPower += segmentPower * time;
            
            prevPos = waypoints[i];
        }
        
        return totalPower;
    }
    
    bool canCompleteFlight(const FlightPlan& plan) {
        float64 availableEnergy = batteryState_.chargeRemaining * batteryState_.voltage / 1000;
        float64 requiredEnergy = plan.estimatedPower;
        
        float64 safetyMargin = 1.2;
        
        return availableEnergy >= requiredEnergy * safetyMargin;
    }
    
    bool canReturnToHome(const Vector3d& currentPosition, const Vector3d& homePosition) {
        float64 distance = (homePosition - currentPosition).norm();
        
        Vector3d direction = (homePosition - currentPosition).normalized();
        Vector3d velocity = direction * 5.0;
        velocity[2] = 0;
        
        float64 power = estimatePowerForDistance(distance, velocity);
        float64 time = distance / 5.0;
        float64 energy = power * time;
        
        float64 availableEnergy = batteryState_.chargeRemaining * batteryState_.voltage / 1000;
        
        float64 safetyMargin = 1.3;
        
        return availableEnergy >= energy * safetyMargin;
    }
    
    Vector3d findNearestLandingPoint(const Vector3d& currentPosition,
                                      const Vector3d* safePoints, uint32 numPoints) {
        if (numPoints == 0) {
            return currentPosition;
        }
        
        Vector3d nearest = safePoints[0];
        float64 minDist = (safePoints[0] - currentPosition).norm();
        
        for (uint32 i = 1; i < numPoints; ++i) {
            float64 dist = (safePoints[i] - currentPosition).norm();
            if (dist < minDist) {
                minDist = dist;
                nearest = safePoints[i];
            }
        }
        
        nearestLandingPoint_ = nearest;
        distanceToNearestLanding_ = minDist;
        
        return nearest;
    }
    
    void setCurrentCapacity(float64 capacityMah) {
        batteryState_.chargeCapacity = capacityMah;
        batteryState_.chargeRemaining = capacityMah * batteryState_.percentage / 100;
    }
    
    void setPowerModel(const PowerModel& model) {
        powerModel_ = model;
    }
    
    const BatteryState& getBatteryState() const { return batteryState_; }
    const PowerModel& getPowerModel() const { return powerModel_; }
    BatteryAlertLevel getAlertLevel() const { return alertLevel_; }
    float64 getEstimatedFlightTime() const { return estimatedFlightTime_; }
    float64 getEstimatedFlightDistance() const { return estimatedFlightDistance_; }
    float64 getAveragePower() const { return avgPowerConsumption_; }
    float64 getInstantPower() const { return instantPower_; }
    float64 getUpdateFrequencyScale() const { return updateFrequencyScale_; }
    bool isLowPowerMode() const { return lowPowerMode_; }
    Vector3d getNearestLandingPoint() const { return nearestLandingPoint_; }
    float64 getDistanceToNearestLanding() const { return distanceToNearestLanding_; }
    
    float64 getPercentage() const { return batteryState_.percentage; }
    float64 getVoltage() const { return batteryState_.voltage; }
    float64 getCurrent() const { return batteryState_.current; }
    
    bool isLowPower() const { return alertLevel_ >= BatteryAlertLevel::LOW_POWER; }
    bool isCritical() const { return alertLevel_ >= BatteryAlertLevel::CRITICAL; }
    bool isEmergency() const { return alertLevel_ >= BatteryAlertLevel::EMERGENCY; }
    
    void reset() {
        historyHead_ = 0;
        historySize_ = 0;
        powerSampleHead_ = 0;
        powerSampleSize_ = 0;
        avgPowerConsumption_ = 0;
        instantPower_ = 0;
        alertLevel_ = BatteryAlertLevel::NORMAL;
        lowPowerMode_ = false;
        updateFrequencyScale_ = 1.0;
    }
};

}
#endif
