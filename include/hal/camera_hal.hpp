#ifndef MSCKF_HAL_CAMERA_HAL_HPP
#define MSCKF_HAL_CAMERA_HAL_HPP

#include "camera_types.hpp"
#include "imu_types.hpp"
#include "../math/types.hpp"
#include <cstring>

namespace msckf {

class Camera_HAL {
public:
    enum class SensorType {
        OV9281,
        AR0144,
        UNKNOWN
    };

private:
    SensorType sensorType_;
    CameraConfig config_;
    CameraIntrinsics intrinsics_;
    CameraExtrinsics extrinsics_;
    RollingShutterInfo rsInfo_;
    CameraDiagnostics diagnostics_;
    bool initialized_;
    
    static constexpr uint32 MAX_IMAGE_WIDTH = 1280;
    static constexpr uint32 MAX_IMAGE_HEIGHT = 800;
    static constexpr uint32 MAX_BUFFER_SIZE = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * 2;
    uint8 dmaBuffer0_[MAX_BUFFER_SIZE];
    uint8 dmaBuffer1_[MAX_BUFFER_SIZE];
    uint32 activeBuffer_;
    volatile bool frameReady_;
    volatile bool dmaOverflow_;
    
    uint64 lastFrameTimestamp_;
    uint64 lastTriggerTimestamp_;
    int64 imuSyncOffset_;

public:
    Camera_HAL()
        : sensorType_(SensorType::UNKNOWN)
        , initialized_(false)
        , activeBuffer_(0)
        , frameReady_(false)
        , dmaOverflow_(false)
        , lastFrameTimestamp_(0)
        , lastTriggerTimestamp_(0)
        , imuSyncOffset_(0) {
        for (uint32 i = 0; i < MAX_BUFFER_SIZE; ++i) {
            dmaBuffer0_[i] = 0;
            dmaBuffer1_[i] = 0;
        }
    }
    
    CameraStatus init(const CameraConfig& config, SensorType type = SensorType::OV9281) {
        if (initialized_) {
            return CameraStatus::OK;
        }
        
        config_ = config;
        sensorType_ = type;
        
        if (config.useDmaDoubleBuffer) {
            uint32 bufferSize = config.width * config.height * 2;
            if (bufferSize > MAX_BUFFER_SIZE) {
                return CameraStatus::INVALID_CONFIG;
            }
            for (uint32 i = 0; i < bufferSize; ++i) {
                dmaBuffer0_[i] = 0;
                dmaBuffer1_[i] = 0;
            }
        }
        
        CameraStatus status = resetSensor();
        if (status != CameraStatus::OK) {
            return status;
        }
        
        status = configureSensor();
        if (status != CameraStatus::OK) {
            return status;
        }
        
        status = configureInterface();
        if (status != CameraStatus::OK) {
            return status;
        }
        
        status = selfTest();
        if (status != CameraStatus::OK) {
            return status;
        }
        
        initialized_ = true;
        return CameraStatus::OK;
    }
    
    void setIntrinsics(const CameraIntrinsics& intr) {
        intrinsics_ = intr;
    }
    
    void setExtrinsics(const CameraExtrinsics& extr) {
        extrinsics_ = extr;
    }
    
    void setRollingShutterInfo(const RollingShutterInfo& rs) {
        rsInfo_ = rs;
    }
    
    const CameraIntrinsics& getIntrinsics() const {
        return intrinsics_;
    }
    
    const CameraExtrinsics& getExtrinsics() const {
        return extrinsics_;
    }
    
    const RollingShutterInfo& getRollingShutterInfo() const {
        return rsInfo_;
    }
    
    CameraStatus captureFrame(ImageData& image) {
        if (!initialized_) {
            return CameraStatus::NOT_INITIALIZED;
        }
        
        if (!frameReady_) {
            return CameraStatus::TIMEOUT;
        }
        
        uint8* srcBuffer = (activeBuffer_ == 0) ? dmaBuffer0_ : dmaBuffer1_;
        
        image.timestamp = lastFrameTimestamp_;
        image.exposureStartTimestamp = lastFrameTimestamp_;
        image.exposureEndTimestamp = lastFrameTimestamp_ + 
            static_cast<uint64>(rsInfo_.totalRows * rsInfo_.rowReadoutTime);
        image.data = srcBuffer;
        image.width = config_.width;
        image.height = config_.height;
        image.stride = config_.width;
        image.channels = 1;
        image.bitsPerPixel = 8;
        image.exposureTime = config_.exposureTime;
        image.gain = config_.gain;
        image.isBayer = true;
        image.bayerPattern = getBayerPattern();
        
        frameReady_ = false;
        diagnostics_.totalFrames++;
        
        return CameraStatus::OK;
    }
    
    CameraStatus waitForFrame(uint32 timeoutMs = 100) {
        if (!initialized_) {
            return CameraStatus::NOT_INITIALIZED;
        }
        
        uint32 elapsed = 0;
        while (!frameReady_ && elapsed < timeoutMs) {
            delayMs(1);
            elapsed++;
        }
        
        return frameReady_ ? CameraStatus::OK : CameraStatus::TIMEOUT;
    }
    
    void handleFrameInterrupt() {
        uint32 nextBuffer = 1 - activeBuffer_;
        
        if (dmaOverflow_) {
            diagnostics_.droppedFrames++;
            return;
        }
        
        lastFrameTimestamp_ = getHardwareTimestamp();
        activeBuffer_ = nextBuffer;
        frameReady_ = true;
        
        uint8* buffer = (activeBuffer_ == 0) ? dmaBuffer0_ : dmaBuffer1_;
        startDmaTransfer(buffer);
    }
    
    void handleTriggerInterrupt() {
        lastTriggerTimestamp_ = getHardwareTimestamp();
        
        if (config_.useHardwareTrigger) {
            triggerExposure();
        }
    }
    
    CameraStatus setExposure(float64 exposureUs) {
        config_.exposureTime = exposureUs;
        
        uint32 expReg = static_cast<uint32>(exposureUs / getExposureStep());
        
        if (sensorType_ == SensorType::OV9281) {
            writeRegister16(0x3500, expReg >> 12);
            writeRegister16(0x3501, (expReg >> 4) & 0xFF);
            writeRegister16(0x3502, expReg & 0x0F);
        } else if (sensorType_ == SensorType::AR0144) {
            writeRegister16(0x3012, expReg);
        }
        
        return CameraStatus::OK;
    }
    
    CameraStatus setGain(float64 gain) {
        config_.gain = gain;
        
        uint16 gainReg = 0;
        if (sensorType_ == SensorType::OV9281) {
            if (gain < 2.0) {
                gainReg = static_cast<uint16>(gain * 16);
            } else if (gain < 4.0) {
                gainReg = static_cast<uint16>((gain - 2.0) * 32 + 32);
            } else {
                gainReg = static_cast<uint16>((gain - 4.0) * 64 + 96);
            }
            writeRegister16(0x350A, gainReg >> 8);
            writeRegister16(0x350B, gainReg & 0xFF);
        } else if (sensorType_ == SensorType::AR0144) {
            gainReg = static_cast<uint16>(gain * 64);
            writeRegister16(0x305E, gainReg);
        }
        
        return CameraStatus::OK;
    }
    
    CameraStatus setFrameRate(uint32 fps) {
        config_.frameRate = fps;
        
        float64 frameTime = 1000000.0 / fps;
        float64 vts = frameTime / getLineTime();
        
        if (sensorType_ == SensorType::OV9281) {
            uint16 vtsReg = static_cast<uint16>(vts);
            writeRegister16(0x380E, vtsReg >> 8);
            writeRegister16(0x380F, vtsReg & 0xFF);
        } else if (sensorType_ == SensorType::AR0144) {
            uint16 vtsReg = static_cast<uint16>(vts);
            writeRegister16(0x300A, vtsReg);
        }
        
        return CameraStatus::OK;
    }
    
    CameraStatus enableHardwareTrigger(bool enable) {
        config_.useHardwareTrigger = enable;
        
        if (enable) {
            writeRegister16(0x3020, 0x0001);
            configureTriggerPin();
        } else {
            writeRegister16(0x3020, 0x0000);
        }
        
        return CameraStatus::OK;
    }
    
    void synchronizeWithIMU(const IMUData& imuData) {
        int64 offset = static_cast<int64>(lastFrameTimestamp_) - 
                       static_cast<int64>(imuData.timestamp);
        
        imuSyncOffset_ = (imuSyncOffset_ * 9 + offset) / 10;
    }
    
    uint64 getSyncedTimestamp(uint64 cameraTimestamp) const {
        return cameraTimestamp - imuSyncOffset_;
    }
    
    float64 getRowTimestamp(uint32 row, uint64 frameTimestamp) const {
        return rsInfo_.getTimestampForRow(row, frameTimestamp);
    }
    
    float64 getFeatureTimestamp(float64 v, uint64 frameTimestamp) const {
        uint32 row = static_cast<uint32>(v);
        return getRowTimestamp(row, frameTimestamp);
    }
    
    bool isFrameReady() const {
        return frameReady_;
    }
    
    bool isInitialized() const {
        return initialized_;
    }
    
    SensorType getSensorType() const {
        return sensorType_;
    }
    
    const CameraDiagnostics& getDiagnostics() const {
        return diagnostics_;
    }
    
    uint64 getHardwareTimestamp() {
        return 0;
    }

private:
    CameraStatus resetSensor() {
        if (sensorType_ == SensorType::OV9281) {
            writeRegister16(0x0103, 0x01);
            delayMs(10);
            
            uint16 pid = readRegister16(0x300A);
            if (pid != 0x9281) {
                diagnostics_.lastError = CameraStatus::COMM_ERROR;
                return CameraStatus::COMM_ERROR;
            }
        } else if (sensorType_ == SensorType::AR0144) {
            writeRegister16(0x301A, 0x0001);
            delayMs(10);
            
            uint16 pid = readRegister16(0x3000);
            if (pid != 0x0144) {
                diagnostics_.lastError = CameraStatus::COMM_ERROR;
                return CameraStatus::COMM_ERROR;
            }
        }
        
        return CameraStatus::OK;
    }
    
    CameraStatus configureSensor() {
        if (sensorType_ == SensorType::OV9281) {
            writeRegister16(0x0100, 0x00);
            
            writeRegister16(0x3800, 0x00);
            writeRegister16(0x3801, 0x00);
            writeRegister16(0x3802, 0x00);
            writeRegister16(0x3803, 0x00);
            writeRegister16(0x3804, (config_.width >> 8) & 0xFF);
            writeRegister16(0x3805, config_.width & 0xFF);
            writeRegister16(0x3806, (config_.height >> 8) & 0xFF);
            writeRegister16(0x3807, config_.height & 0xFF);
            
            writeRegister16(0x3808, (config_.width >> 8) & 0xFF);
            writeRegister16(0x3809, config_.width & 0xFF);
            writeRegister16(0x380A, (config_.height >> 8) & 0xFF);
            writeRegister16(0x380B, config_.height & 0xFF);
            
            writeRegister16(0x370C, 0x03);
            writeRegister16(0x5000, 0x06);
            
            rsInfo_.totalRows = config_.height;
            rsInfo_.rowReadoutTime = getLineTime();
            rsInfo_.isRollingShutter = true;
            
        } else if (sensorType_ == SensorType::AR0144) {
            writeRegister16(0x0100, 0x00);
            
            writeRegister16(0x3002, config_.width);
            writeRegister16(0x3004, config_.height);
            
            writeRegister16(0x3020, config_.useHardwareTrigger ? 0x0001 : 0x0000);
            
            rsInfo_.totalRows = config_.height;
            rsInfo_.rowReadoutTime = getLineTime();
            rsInfo_.isRollingShutter = true;
        }
        
        setExposure(config_.exposureTime);
        setGain(config_.gain);
        
        writeRegister16(0x0100, 0x01);
        
        return CameraStatus::OK;
    }
    
    CameraStatus configureInterface() {
        if (config_.useMipi) {
            configureMipiCsi();
        } else {
            configureDvp();
        }
        
        if (config_.useDmaDoubleBuffer) {
            configureDmaDoubleBuffer();
        }
        
        return CameraStatus::OK;
    }
    
    void configureMipiCsi() {
    }
    
    void configureDvp() {
    }
    
    void configureDmaDoubleBuffer() {
    }
    
    void configureTriggerPin() {
    }
    
    void startDmaTransfer(uint8* buffer) {
    }
    
    void triggerExposure() {
    }
    
    CameraStatus selfTest() {
        return CameraStatus::OK;
    }
    
    uint8 getBayerPattern() const {
        if (sensorType_ == SensorType::OV9281) {
            return 0;
        } else if (sensorType_ == SensorType::AR0144) {
            return 1;
        }
        return 0;
    }
    
    float64 getLineTime() const {
        return 22.222;
    }
    
    float64 getExposureStep() const {
        return 1.0;
    }
    
    void writeRegister16(uint16 reg, uint16 value) {
    }
    
    uint16 readRegister16(uint16 reg) {
        return 0;
    }
    
    void delayMs(uint32 ms) {
    }
};

}

#endif
