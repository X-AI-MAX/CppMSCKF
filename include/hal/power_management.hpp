#ifndef MSCKF_HAL_POWER_MANAGEMENT_HPP
#define MSCKF_HAL_POWER_MANAGEMENT_HPP

#include "../math/types.hpp"

namespace msckf {

enum class PowerState : uint8 {
    ACTIVE = 0,
    IDLE = 1,
    LOW_POWER = 2,
    SLEEP = 3,
    DEEP_SLEEP = 4
};

enum class PeripheralId : uint8 {
    IMU = 0,
    CAMERA = 1,
    SPI = 2,
    I2C = 3,
    UART = 4,
    DMA = 5,
    TIMER = 6,
    GPIO = 7,
    ADC = 8,
    FLASH = 9,
    NUM_PERIPHERALS = 10
};

struct PeripheralConfig {
    PeripheralId id;
    const char* name;
    uint32 baseAddress;
    uint32 clockEnableBit;
    uint32 clockDisableBit;
    uint32 resetBit;
    float32 typicalCurrentMa;
    float32 sleepCurrentUa;
    bool canBeDisabled;
    bool critical;
    
    PeripheralConfig() 
        : id(PeripheralId::IMU)
        , name(nullptr)
        , baseAddress(0)
        , clockEnableBit(0)
        , clockDisableBit(0)
        , resetBit(0)
        , typicalCurrentMa(10.0f)
        , sleepCurrentUa(100.0f)
        , canBeDisabled(true)
        , critical(false) {}
};

class PeripheralClockManager {
public:
    struct Config {
        bool enableDynamicClockGating;
        bool enableAutoSleep;
        uint32 idleThresholdMs;
        uint32 sleepThresholdMs;
        
        Config() 
            : enableDynamicClockGating(true)
            , enableAutoSleep(true)
            , idleThresholdMs(100)
            , sleepThresholdMs(1000) {}
    };

private:
    Config config_;
    
    PeripheralConfig peripherals_[static_cast<uint32>(PeripheralId::NUM_PERIPHERALS)];
    bool peripheralEnabled_[static_cast<uint32>(PeripheralId::NUM_PERIPHERALS)];
    uint64 lastActivityTime_[static_cast<uint32>(PeripheralId::NUM_PERIPHERALS)];
    uint32 usageCount_[static_cast<uint32>(PeripheralId::NUM_PERIPHERALS)];
    
    PowerState systemPowerState_;
    uint64 lastSystemActivity_;
    
    float32 totalCurrentMa_;
    uint32 activePeripheralCount_;
    
    uint32 clockGatingEvents_;
    uint32 wakeUpEvents_;

public:
    PeripheralClockManager() 
        : systemPowerState_(PowerState::ACTIVE)
        , lastSystemActivity_(0)
        , totalCurrentMa_(0)
        , activePeripheralCount_(0)
        , clockGatingEvents_(0)
        , wakeUpEvents_(0) {
        
        for (uint32 i = 0; i < static_cast<uint32>(PeripheralId::NUM_PERIPHERALS); ++i) {
            peripheralEnabled_[i] = false;
            lastActivityTime_[i] = 0;
            usageCount_[i] = 0;
        }
        
        initializePeripheralConfigs();
    }
    
    void init(const Config& config) {
        config_ = config;
    }
    
    void initializePeripheralConfigs() {
        peripherals_[static_cast<uint32>(PeripheralId::IMU)] = {
            PeripheralId::IMU, "IMU", 0x40000000, 0x01, 0x00, 0x01,
            5.0f, 50.0f, false, true
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::CAMERA)] = {
            PeripheralId::CAMERA, "CAMERA", 0x40010000, 0x02, 0x00, 0x02,
            50.0f, 500.0f, true, false
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::SPI)] = {
            PeripheralId::SPI, "SPI", 0x40020000, 0x04, 0x00, 0x04,
            2.0f, 10.0f, true, false
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::I2C)] = {
            PeripheralId::I2C, "I2C", 0x40030000, 0x08, 0x00, 0x08,
            1.0f, 5.0f, true, false
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::UART)] = {
            PeripheralId::UART, "UART", 0x40040000, 0x10, 0x00, 0x10,
            3.0f, 20.0f, true, false
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::DMA)] = {
            PeripheralId::DMA, "DMA", 0x40050000, 0x20, 0x00, 0x20,
            5.0f, 10.0f, false, true
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::TIMER)] = {
            PeripheralId::TIMER, "TIMER", 0x40060000, 0x40, 0x00, 0x40,
            0.5f, 1.0f, false, true
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::GPIO)] = {
            PeripheralId::GPIO, "GPIO", 0x40070000, 0x80, 0x00, 0x80,
            0.1f, 0.5f, false, true
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::ADC)] = {
            PeripheralId::ADC, "ADC", 0x40080000, 0x100, 0x00, 0x100,
            1.0f, 5.0f, true, false
        };
        
        peripherals_[static_cast<uint32>(PeripheralId::FLASH)] = {
            PeripheralId::FLASH, "FLASH", 0x40090000, 0x200, 0x00, 0x200,
            5.0f, 1.0f, false, true
        };
    }
    
    void enablePeripheral(PeripheralId id) {
        uint32 idx = static_cast<uint32>(id);
        if (idx >= static_cast<uint32>(PeripheralId::NUM_PERIPHERALS)) return;
        
        if (!peripheralEnabled_[idx]) {
            peripheralEnabled_[idx] = true;
            activePeripheralCount_++;
            totalCurrentMa_ += peripherals_[idx].typicalCurrentMa;
            
            clockGatingEvents_++;
        }
        
        lastActivityTime_[idx] = getCurrentTime();
        usageCount_[idx]++;
    }
    
    void disablePeripheral(PeripheralId id) {
        uint32 idx = static_cast<uint32>(id);
        if (idx >= static_cast<uint32>(PeripheralId::NUM_PERIPHERALS)) return;
        
        if (peripherals_[idx].critical) return;
        
        if (peripheralEnabled_[idx]) {
            peripheralEnabled_[idx] = false;
            activePeripheralCount_--;
            totalCurrentMa_ -= peripherals_[idx].typicalCurrentMa;
            totalCurrentMa_ += peripherals_[idx].sleepCurrentUa / 1000.0f;
            
            clockGatingEvents_++;
        }
    }
    
    void reportActivity(PeripheralId id) {
        uint32 idx = static_cast<uint32>(id);
        if (idx >= static_cast<uint32>(PeripheralId::NUM_PERIPHERALS)) return;
        
        lastActivityTime_[idx] = getCurrentTime();
        lastSystemActivity_ = getCurrentTime();
        
        if (systemPowerState_ != PowerState::ACTIVE) {
            wakeUpEvents_++;
            systemPowerState_ = PowerState::ACTIVE;
        }
    }
    
    void update(uint64 currentTime) {
        if (!config_.enableDynamicClockGating) return;
        
        for (uint32 i = 0; i < static_cast<uint32>(PeripheralId::NUM_PERIPHERALS); ++i) {
            if (!peripherals_[i].canBeDisabled) continue;
            if (!peripheralEnabled_[i]) continue;
            
            uint64 idleTime = currentTime - lastActivityTime_[i];
            
            if (idleTime > config_.sleepThresholdMs * 1000) {
                disablePeripheral(static_cast<PeripheralId>(i));
            }
        }
        
        updateSystemPowerState(currentTime);
    }
    
    void updateSystemPowerState(uint64 currentTime) {
        uint64 idleTime = currentTime - lastSystemActivity_;
        
        if (idleTime > config_.sleepThresholdMs * 1000) {
            systemPowerState_ = PowerState::SLEEP;
        } else if (idleTime > config_.idleThresholdMs * 1000) {
            systemPowerState_ = PowerState::IDLE;
        } else {
            systemPowerState_ = PowerState::ACTIVE;
        }
    }
    
    void enterLowPowerMode() {
        for (uint32 i = 0; i < static_cast<uint32>(PeripheralId::NUM_PERIPHERALS); ++i) {
            if (peripherals_[i].canBeDisabled && !peripherals_[i].critical) {
                disablePeripheral(static_cast<PeripheralId>(i));
            }
        }
        
        systemPowerState_ = PowerState::LOW_POWER;
    }
    
    void enterSleepMode() {
        for (uint32 i = 0; i < static_cast<uint32>(PeripheralId::NUM_PERIPHERALS); ++i) {
            if (peripherals_[i].canBeDisabled) {
                disablePeripheral(static_cast<PeripheralId>(i));
            }
        }
        
        systemPowerState_ = PowerState::SLEEP;
    }
    
    void wakeUp() {
        enablePeripheral(PeripheralId::IMU);
        enablePeripheral(PeripheralId::TIMER);
        enablePeripheral(PeripheralId::DMA);
        
        systemPowerState_ = PowerState::ACTIVE;
        lastSystemActivity_ = getCurrentTime();
        
        wakeUpEvents_++;
    }
    
    bool isPeripheralEnabled(PeripheralId id) const {
        uint32 idx = static_cast<uint32>(id);
        if (idx >= static_cast<uint32>(PeripheralId::NUM_PERIPHERALS)) return false;
        return peripheralEnabled_[idx];
    }
    
    PowerState getSystemPowerState() const { return systemPowerState_; }
    float32 getTotalCurrentMa() const { return totalCurrentMa_; }
    uint32 getActivePeripheralCount() const { return activePeripheralCount_; }
    uint32 getClockGatingEvents() const { return clockGatingEvents_; }
    uint32 getWakeUpEvents() const { return wakeUpEvents_; }
    
    float32 estimateBatteryLife(float32 batteryCapacityMah) const {
        if (totalCurrentMa_ <= 0) return 0;
        return batteryCapacityMah / totalCurrentMa_;
    }

private:
    uint64 getCurrentTime() const {
        return 0;
    }
};

class DVFSController {
public:
    enum class FrequencyLevel : uint8 {
        LOW = 0,
        MEDIUM = 1,
        HIGH = 2,
        TURBO = 3
    };
    
    struct FrequencyConfig {
        uint32 cpuFrequencyMhz;
        uint32 busFrequencyMhz;
        float32 voltage;
        float32 powerMa;
    };

private:
    FrequencyConfig frequencyConfigs_[4];
    FrequencyLevel currentLevel_;
    
    uint64 lastAdjustTime_;
    uint32 adjustInterval_;
    
    float32 cpuLoadThresholdLow_;
    float32 cpuLoadThresholdHigh_;
    
    uint32 levelUpCount_;
    uint32 levelDownCount_;

public:
    DVFSController() 
        : currentLevel_(FrequencyLevel::MEDIUM)
        , lastAdjustTime_(0)
        , adjustInterval_(100000)
        , cpuLoadThresholdLow_(30.0f)
        , cpuLoadThresholdHigh_(70.0f)
        , levelUpCount_(0)
        , levelDownCount_(0) {
        
        frequencyConfigs_[static_cast<uint32>(FrequencyLevel::LOW)] = {
            48, 24, 0.9f, 15.0f
        };
        frequencyConfigs_[static_cast<uint32>(FrequencyLevel::MEDIUM)] = {
            96, 48, 1.0f, 30.0f
        };
        frequencyConfigs_[static_cast<uint32>(FrequencyLevel::HIGH)] = {
            192, 96, 1.1f, 60.0f
        };
        frequencyConfigs_[static_cast<uint32>(FrequencyLevel::TURBO)] = {
            400, 200, 1.2f, 120.0f
        };
    }
    
    void adjustFrequency(float64 cpuLoad, uint64 currentTime) {
        if (currentTime - lastAdjustTime_ < adjustInterval_) return;
        
        lastAdjustTime_ = currentTime;
        
        float32 load = static_cast<float32>(cpuLoad);
        
        if (load > cpuLoadThresholdHigh_ && currentLevel_ < FrequencyLevel::TURBO) {
            currentLevel_ = static_cast<FrequencyLevel>(
                static_cast<uint32>(currentLevel_) + 1);
            levelUpCount_++;
            applyFrequencyConfig();
        } else if (load < cpuLoadThresholdLow_ && currentLevel_ > FrequencyLevel::LOW) {
            currentLevel_ = static_cast<FrequencyLevel>(
                static_cast<uint32>(currentLevel_) - 1);
            levelDownCount_++;
            applyFrequencyConfig();
        }
    }
    
    void setLevel(FrequencyLevel level) {
        currentLevel_ = level;
        applyFrequencyConfig();
    }
    
    void applyFrequencyConfig() {
        const FrequencyConfig& config = frequencyConfigs_[static_cast<uint32>(currentLevel_)];
        (void)config;
    }
    
    FrequencyLevel getCurrentLevel() const { return currentLevel_; }
    const FrequencyConfig& getCurrentConfig() const {
        return frequencyConfigs_[static_cast<uint32>(currentLevel_)];
    }
    
    uint32 getCpuFrequencyMhz() const {
        return getCurrentConfig().cpuFrequencyMhz;
    }
    
    float32 getCurrentPowerMa() const {
        return getCurrentConfig().powerMa;
    }
    
    uint32 getLevelUpCount() const { return levelUpCount_; }
    uint32 getLevelDownCount() const { return levelDownCount_; }
    
    void setThresholds(float32 low, float32 high) {
        cpuLoadThresholdLow_ = low;
        cpuLoadThresholdHigh_ = high;
    }
};

class PowerManager {
public:
    struct Config {
        PeripheralClockManager::Config peripheralConfig;
        bool enableDVFS;
        bool enableAutoPowerDown;
        uint32 targetFrameTimeUs;
        
        Config() 
            : enableDVFS(true)
            , enableAutoPowerDown(true)
            , targetFrameTimeUs(33333) {}
    };

private:
    Config config_;
    PeripheralClockManager peripheralManager_;
    DVFSController dvfsController_;
    
    float64 avgCpuLoad_;
    float64 avgFrameTime_;
    uint64 totalActiveTime_;
    uint64 totalIdleTime_;
    
    uint64 lastFrameStart_;
    uint32 frameCount_;
    
    bool lowPowerModeEnabled_;

public:
    PowerManager() 
        : avgCpuLoad_(0)
        , avgFrameTime_(0)
        , totalActiveTime_(0)
        , totalIdleTime_(0)
        , lastFrameStart_(0)
        , frameCount_(0)
        , lowPowerModeEnabled_(false) {}
    
    void init(const Config& config) {
        config_ = config;
        peripheralManager_.init(config.peripheralConfig);
    }
    
    void beginFrame(uint64 timestamp) {
        lastFrameStart_ = timestamp;
        
        peripheralManager_.reportActivity(PeripheralId::CAMERA);
        peripheralManager_.reportActivity(PeripheralId::IMU);
    }
    
    void endFrame(uint64 timestamp, float64 cpuLoad) {
        uint64 frameTime = timestamp - lastFrameStart_;
        avgFrameTime_ = avgFrameTime_ * 0.9 + frameTime * 0.1;
        avgCpuLoad_ = avgCpuLoad_ * 0.9 + cpuLoad * 0.1;
        
        frameCount_++;
        totalActiveTime_ += frameTime;
        
        if (config_.enableDVFS) {
            dvfsController_.adjustFrequency(cpuLoad, timestamp);
        }
        
        peripheralManager_.update(timestamp);
    }
    
    void reportIdle(uint64 idleTime) {
        totalIdleTime_ += idleTime;
        
        if (config_.enableAutoPowerDown && idleTime > 10000) {
            peripheralManager_.enterLowPowerMode();
        }
    }
    
    void enableLowPowerMode() {
        lowPowerModeEnabled_ = true;
        peripheralManager_.enterLowPowerMode();
        dvfsController_.setLevel(DVFSController::FrequencyLevel::LOW);
    }
    
    void disableLowPowerMode() {
        lowPowerModeEnabled_ = false;
        peripheralManager_.wakeUp();
        dvfsController_.setLevel(DVFSController::FrequencyLevel::MEDIUM);
    }
    
    float64 getCpuLoad() const { return avgCpuLoad_; }
    float64 getFrameTime() const { return avgFrameTime_; }
    bool isInLowPowerMode() const { return lowPowerModeEnabled_; }
    
    float32 getTotalPowerMa() const {
        return peripheralManager_.getTotalCurrentMa() + 
               dvfsController_.getCurrentPowerMa();
    }
    
    float32 estimateBatteryLife(float32 batteryMah) const {
        float32 totalMa = getTotalPowerMa();
        if (totalMa <= 0) return 0;
        return batteryMah / totalMa;
    }
    
    const PeripheralClockManager& getPeripheralManager() const {
        return peripheralManager_;
    }
    
    const DVFSController& getDVFSController() const {
        return dvfsController_;
    }
    
    uint32 getFrameCount() const { return frameCount_; }
    float64 getDutyCycle() const {
        uint64 totalTime = totalActiveTime_ + totalIdleTime_;
        if (totalTime == 0) return 0;
        return static_cast<float64>(totalActiveTime_) / totalTime * 100.0;
    }
};

}

#endif
