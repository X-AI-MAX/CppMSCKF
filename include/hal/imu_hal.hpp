#ifndef MSCKF_HAL_IMU_HAL_HPP
#define MSCKF_HAL_IMU_HAL_HPP

#include "imu_types.hpp"
#include "../math/types.hpp"
#include <cstring>

namespace msckf {

class IMU_HAL {
public:
    enum class DeviceType {
        MPU9250,
        ICM20689,
        UNKNOWN
    };

private:
    DeviceType deviceType_;
    IMUConfig config_;
    IMUCalibration calibration_;
    IMUDiagnostics diagnostics_;
    bool initialized_;
    
    static constexpr uint32 MAX_DMA_BUFFER_SIZE = 1024;
    uint8 dmaBuffer_[MAX_DMA_BUFFER_SIZE * 14];
    uint32 dmaWriteIndex_;
    uint32 dmaReadIndex_;
    volatile bool dmaOverflow_;
    
    static constexpr uint32 IMU_DATA_SIZE = 14;
    
    struct RawData {
        int16 accel[3];
        int16 temp;
        int16 gyro[3];
    };

public:
    IMU_HAL() 
        : deviceType_(DeviceType::UNKNOWN)
        , initialized_(false)
        , dmaWriteIndex_(0)
        , dmaReadIndex_(0)
        , dmaOverflow_(false) {
        for (uint32 i = 0; i < MAX_DMA_BUFFER_SIZE * IMU_DATA_SIZE; ++i) {
            dmaBuffer_[i] = 0;
        }
    }
    
    IMUStatus init(const IMUConfig& config, DeviceType type = DeviceType::ICM20689) {
        if (initialized_) {
            return IMUStatus::OK;
        }
        
        config_ = config;
        deviceType_ = type;
        
        if (config.useDma) {
            if (config.dmaBufferSize > MAX_DMA_BUFFER_SIZE) {
                config_.dmaBufferSize = MAX_DMA_BUFFER_SIZE;
            }
            for (uint32 i = 0; i < config_.dmaBufferSize * IMU_DATA_SIZE; ++i) {
                dmaBuffer_[i] = 0;
            }
        }
        
        IMUStatus status = resetDevice();
        if (status != IMUStatus::OK) {
            return status;
        }
        
        status = configureDevice();
        if (status != IMUStatus::OK) {
            return status;
        }
        
        status = selfTest();
        if (status != IMUStatus::OK) {
            return status;
        }
        
        initialized_ = true;
        return IMUStatus::OK;
    }
    
    void setCalibration(const IMUCalibration& cal) {
        calibration_ = cal;
    }
    
    const IMUCalibration& getCalibration() const {
        return calibration_;
    }
    
    IMUStatus readData(IMUData& data) {
        if (!initialized_) {
            return IMUStatus::NOT_INITIALIZED;
        }
        
        RawData rawData;
        IMUStatus status = readRawData(rawData);
        if (status != IMUStatus::OK) {
            return status;
        }
        
        data.timestamp = getHardwareTimestamp();
        
        float32 accelScale = config_.accelRange / 32768.0f;
        float32 gyroScale = config_.gyroRange / 32768.0f * (PI / 180.0f);
        
        data.accel[0] = rawData.accel[0] * accelScale;
        data.accel[1] = rawData.accel[1] * accelScale;
        data.accel[2] = rawData.accel[2] * accelScale;
        
        data.gyro[0] = rawData.gyro[0] * gyroScale;
        data.gyro[1] = rawData.gyro[1] * gyroScale;
        data.gyro[2] = rawData.gyro[2] * gyroScale;
        
        data.temperature = rawData.temp / 326.8f + 25.0f;
        
        data.accel = calibration_.correctAccel(data.accel);
        data.gyro = calibration_.correctGyro(data.gyro);
        data.accel = calibration_.transformToBody(data.accel);
        data.gyro = calibration_.transformToBody(data.gyro);
        
        diagnostics_.totalSamples++;
        
        return IMUStatus::OK;
    }
    
    IMUStatus readFromDma(IMUData& data) {
        if (!config_.useDma || !dmaBuffer_) {
            return IMUStatus::INVALID_CONFIG;
        }
        
        if (dmaReadIndex_ == dmaWriteIndex_) {
            return IMUStatus::DATA_OVERFLOW;
        }
        
        RawData rawData;
        uint8* ptr = dmaBuffer_ + dmaReadIndex_ * IMU_DATA_SIZE;
        memcpy(&rawData, ptr, IMU_DATA_SIZE);
        
        dmaReadIndex_ = (dmaReadIndex_ + 1) % config_.dmaBufferSize;
        
        data.timestamp = getHardwareTimestamp();
        
        float32 accelScale = config_.accelRange / 32768.0f;
        float32 gyroScale = config_.gyroRange / 32768.0f * (PI / 180.0f);
        
        data.accel[0] = rawData.accel[0] * accelScale;
        data.accel[1] = rawData.accel[1] * accelScale;
        data.accel[2] = rawData.accel[2] * accelScale;
        
        data.gyro[0] = rawData.gyro[0] * gyroScale;
        data.gyro[1] = rawData.gyro[1] * gyroScale;
        data.gyro[2] = rawData.gyro[2] * gyroScale;
        
        data.temperature = rawData.temp / 326.8f + 25.0f;
        
        data.accel = calibration_.correctAccel(data.accel);
        data.gyro = calibration_.correctGyro(data.gyro);
        data.accel = calibration_.transformToBody(data.accel);
        data.gyro = calibration_.transformToBody(data.gyro);
        
        diagnostics_.totalSamples++;
        
        return IMUStatus::OK;
    }
    
    void handleDmaTransfer() {
        if (!dmaBuffer_) return;
        
        RawData rawData;
        if (readRawDataDirect(rawData) == IMUStatus::OK) {
            uint8* ptr = dmaBuffer_ + dmaWriteIndex_ * IMU_DATA_SIZE;
            memcpy(ptr, &rawData, IMU_DATA_SIZE);
            
            uint32 nextWrite = (dmaWriteIndex_ + 1) % config_.dmaBufferSize;
            if (nextWrite == dmaReadIndex_) {
                dmaOverflow_ = true;
                diagnostics_.overflowCount++;
            } else {
                dmaWriteIndex_ = nextWrite;
            }
        }
    }
    
    bool hasDmaData() const {
        return dmaReadIndex_ != dmaWriteIndex_;
    }
    
    bool isDmaOverflow() const {
        return dmaOverflow_;
    }
    
    void clearDmaOverflow() {
        dmaOverflow_ = false;
    }
    
    uint32 getDmaDataCount() const {
        if (dmaWriteIndex_ >= dmaReadIndex_) {
            return dmaWriteIndex_ - dmaReadIndex_;
        }
        return config_.dmaBufferSize - dmaReadIndex_ + dmaWriteIndex_;
    }
    
    bool isDataReady() {
        uint8 intStatus = readRegister(0x3A);
        return (intStatus & 0x01) != 0;
    }
    
    const IMUDiagnostics& getDiagnostics() const {
        return diagnostics_;
    }
    
    bool isInitialized() const {
        return initialized_;
    }
    
    DeviceType getDeviceType() const {
        return deviceType_;
    }
    
    uint64 getHardwareTimestamp() {
        return 0;
    }
    
    bool checkMotionValid(const IMUData& data) {
        float64 accelNorm = data.accel.norm();
        float64 gyroNorm = data.gyro.norm();
        
        if (accelNorm > 4.0 * GRAVITY || gyroNorm > 10.0) {
            diagnostics_.errorCount++;
            diagnostics_.lastError = IMUStatus::DATA_OVERFLOW;
            diagnostics_.lastErrorTimestamp = data.timestamp;
            return false;
        }
        
        return true;
    }

private:
    IMUStatus resetDevice() {
        writeRegister(0x6B, 0x80);
        
        uint32 timeout = 100000;
        while (timeout--) {
            uint8 whoami = readRegister(0x75);
            if (whoami == 0x71 || whoami == 0x98 || whoami == 0xAF) {
                break;
            }
        }
        
        if (timeout == 0) {
            return IMUStatus::COMM_ERROR;
        }
        
        writeRegister(0x6B, 0x01);
        
        return IMUStatus::OK;
    }
    
    IMUStatus configureDevice() {
        uint8 dlpfCfg = 0;
        if (config_.bandwidth <= 5) dlpfCfg = 6;
        else if (config_.bandwidth <= 10) dlpfCfg = 5;
        else if (config_.bandwidth <= 20) dlpfCfg = 4;
        else if (config_.bandwidth <= 41) dlpfCfg = 3;
        else if (config_.bandwidth <= 92) dlpfCfg = 2;
        else if (config_.bandwidth <= 188) dlpfCfg = 1;
        else dlpfCfg = 0;
        
        writeRegister(0x1A, dlpfCfg);
        writeRegister(0x1D, dlpfCfg);
        
        uint8 accelRange = 0;
        if (config_.accelRange <= 2) { accelRange = 0; config_.accelRange = 2; }
        else if (config_.accelRange <= 4) { accelRange = 1; config_.accelRange = 4; }
        else if (config_.accelRange <= 8) { accelRange = 2; config_.accelRange = 8; }
        else { accelRange = 3; config_.accelRange = 16; }
        writeRegister(0x1C, accelRange << 3);
        
        uint8 gyroRange = 0;
        if (config_.gyroRange <= 250) { gyroRange = 0; config_.gyroRange = 250; }
        else if (config_.gyroRange <= 500) { gyroRange = 1; config_.gyroRange = 500; }
        else if (config_.gyroRange <= 1000) { gyroRange = 2; config_.gyroRange = 1000; }
        else { gyroRange = 3; config_.gyroRange = 2000; }
        writeRegister(0x1B, gyroRange << 3);
        
        uint8 smplrtDiv = 1000 / config_.sampleRate - 1;
        writeRegister(0x19, smplrtDiv);
        
        if (config_.useInterrupt) {
            writeRegister(0x37, 0x10);
            writeRegister(0x38, 0x01);
        }
        
        return IMUStatus::OK;
    }
    
    IMUStatus selfTest() {
        writeRegister(0x1C, 0xE0);
        writeRegister(0x1B, 0xE0);
        
        delayMs(10);
        
        writeRegister(0x1C, 0x00);
        writeRegister(0x1B, 0x00);
        
        delayMs(10);
        
        return IMUStatus::OK;
    }
    
    IMUStatus readRawData(RawData& data) {
        uint8 buffer[14];
        
        if (config_.useSpi) {
            if (!spiReadBurst(0x3B, buffer, 14)) {
                diagnostics_.errorCount++;
                diagnostics_.lastError = IMUStatus::COMM_ERROR;
                return IMUStatus::COMM_ERROR;
            }
        } else {
            if (!i2cReadBurst(0x3B, buffer, 14)) {
                diagnostics_.errorCount++;
                diagnostics_.lastError = IMUStatus::COMM_ERROR;
                return IMUStatus::COMM_ERROR;
            }
        }
        
        data.accel[0] = (int16)((buffer[0] << 8) | buffer[1]);
        data.accel[1] = (int16)((buffer[2] << 8) | buffer[3]);
        data.accel[2] = (int16)((buffer[4] << 8) | buffer[5]);
        data.temp = (int16)((buffer[6] << 8) | buffer[7]);
        data.gyro[0] = (int16)((buffer[8] << 8) | buffer[9]);
        data.gyro[1] = (int16)((buffer[10] << 8) | buffer[11]);
        data.gyro[2] = (int16)((buffer[12] << 8) | buffer[13]);
        
        return IMUStatus::OK;
    }
    
    IMUStatus readRawDataDirect(RawData& data) {
        return readRawData(data);
    }
    
    uint8 readRegister(uint8 reg) {
        uint8 value = 0;
        if (config_.useSpi) {
            spiRead(reg | 0x80, value);
        } else {
            i2cRead(reg, value);
        }
        return value;
    }
    
    void writeRegister(uint8 reg, uint8 value) {
        if (config_.useSpi) {
            spiWrite(reg, value);
        } else {
            i2cWrite(reg, value);
        }
    }
    
    bool spiRead(uint8 reg, uint8& value) {
        return true;
    }
    
    bool spiWrite(uint8 reg, uint8 value) {
        return true;
    }
    
    bool spiReadBurst(uint8 reg, uint8* buffer, uint32 length) {
        return true;
    }
    
    bool i2cRead(uint8 reg, uint8& value) {
        return true;
    }
    
    bool i2cWrite(uint8 reg, uint8 value) {
        return true;
    }
    
    bool i2cReadBurst(uint8 reg, uint8* buffer, uint32 length) {
        return true;
    }
    
    void delayMs(uint32 ms) {
    }
};

}

#endif
