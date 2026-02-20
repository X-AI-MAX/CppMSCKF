#ifndef MSCKF_SYSTEM_COMMUNICATION_HPP
#define MSCKF_SYSTEM_COMMUNICATION_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../core/state.hpp"

namespace msckf {

struct OutputPose {
    uint64 timestamp;
    float64 position[3];
    float64 velocity[3];
    float64 orientation[4];
    float64 gyroBias[3];
    float64 accelBias[3];
    float64 positionCov[9];
    float64 orientationCov[9];
    uint32 quality;
    uint32 numFeatures;
    uint32 numCameraFrames;
    
    void fromState(const MSCKFState& state) {
        timestamp = state.timestamp;
        
        position[0] = state.imuState.position[0];
        position[1] = state.imuState.position[1];
        position[2] = state.imuState.position[2];
        
        velocity[0] = state.imuState.velocity[0];
        velocity[1] = state.imuState.velocity[1];
        velocity[2] = state.imuState.velocity[2];
        
        orientation[0] = state.imuState.orientation.w;
        orientation[1] = state.imuState.orientation.x;
        orientation[2] = state.imuState.orientation.y;
        orientation[3] = state.imuState.orientation.z;
        
        gyroBias[0] = state.imuState.gyroBias[0];
        gyroBias[1] = state.imuState.gyroBias[1];
        gyroBias[2] = state.imuState.gyroBias[2];
        
        accelBias[0] = state.imuState.accelBias[0];
        accelBias[1] = state.imuState.accelBias[1];
        accelBias[2] = state.imuState.accelBias[2];
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                positionCov[i * 3 + j] = state.covariance(i, j);
                orientationCov[i * 3 + j] = state.covariance(i + 6, j + 6);
            }
        }
        
        numCameraFrames = state.numCameraStates;
    }
};

class BinaryProtocol {
public:
    static constexpr uint8 FRAME_HEADER_0 = 0xAA;
    static constexpr uint8 FRAME_HEADER_1 = 0x55;
    static constexpr uint16 MAX_PAYLOAD_SIZE = 512;
    
    enum class MessageType : uint8 {
        POSE = 0x01,
        IMU_RAW = 0x02,
        IMAGE_FEATURES = 0x03,
        STATUS = 0x04,
        CALIBRATION = 0x05,
        DIAGNOSTIC = 0x06
    };
    
    struct FrameHeader {
        uint8 header[2];
        uint8 type;
        uint16 length;
        uint8 sequence;
    };
    
    struct Frame {
        FrameHeader header;
        uint8 payload[MAX_PAYLOAD_SIZE];
        uint16 crc;
    };

private:
    uint8 txBuffer_[MAX_PAYLOAD_SIZE + 10];
    uint8 rxBuffer_[MAX_PAYLOAD_SIZE + 10];
    uint8 sequence_;
    uint32 txCount_;
    uint32 rxCount_;
    uint32 crcErrorCount_;
    
public:
    BinaryProtocol() 
        : sequence_(0)
        , txCount_(0)
        , rxCount_(0)
        , crcErrorCount_(0) {}
    
    uint32 encodePose(const OutputPose& pose, uint8* buffer) {
        uint32 idx = 0;
        
        buffer[idx++] = FRAME_HEADER_0;
        buffer[idx++] = FRAME_HEADER_1;
        buffer[idx++] = static_cast<uint8>(MessageType::POSE);
        
        uint16 lengthIdx = idx;
        idx += 2;
        
        buffer[idx++] = sequence_++;
        
        memcpy(buffer + idx, &pose.timestamp, sizeof(uint64));
        idx += sizeof(uint64);
        
        memcpy(buffer + idx, pose.position, sizeof(float64) * 3);
        idx += sizeof(float64) * 3;
        
        memcpy(buffer + idx, pose.velocity, sizeof(float64) * 3);
        idx += sizeof(float64) * 3;
        
        memcpy(buffer + idx, pose.orientation, sizeof(float64) * 4);
        idx += sizeof(float64) * 4;
        
        memcpy(buffer + idx, pose.gyroBias, sizeof(float64) * 3);
        idx += sizeof(float64) * 3;
        
        memcpy(buffer + idx, pose.accelBias, sizeof(float64) * 3);
        idx += sizeof(float64) * 3;
        
        memcpy(buffer + idx, pose.positionCov, sizeof(float64) * 9);
        idx += sizeof(float64) * 9;
        
        memcpy(buffer + idx, pose.orientationCov, sizeof(float64) * 9);
        idx += sizeof(float64) * 9;
        
        memcpy(buffer + idx, &pose.quality, sizeof(uint32));
        idx += sizeof(uint32);
        
        memcpy(buffer + idx, &pose.numFeatures, sizeof(uint32));
        idx += sizeof(uint32);
        
        memcpy(buffer + idx, &pose.numCameraFrames, sizeof(uint32));
        idx += sizeof(uint32);
        
        uint16 length = idx - 6;
        buffer[lengthIdx] = length & 0xFF;
        buffer[lengthIdx + 1] = (length >> 8) & 0xFF;
        
        uint16 crc = computeCRC16(buffer + 6, length);
        buffer[idx++] = crc & 0xFF;
        buffer[idx++] = (crc >> 8) & 0xFF;
        
        txCount_++;
        
        return idx;
    }
    
    bool decodeFrame(const uint8* buffer, uint32 length, Frame& frame) {
        if (length < 7) return false;
        
        if (buffer[0] != FRAME_HEADER_0 || buffer[1] != FRAME_HEADER_1) {
            return false;
        }
        
        frame.header.header[0] = buffer[0];
        frame.header.header[1] = buffer[1];
        frame.header.type = buffer[2];
        frame.header.length = buffer[3] | (buffer[4] << 8);
        frame.header.sequence = buffer[5];
        
        if (frame.header.length > MAX_PAYLOAD_SIZE) {
            return false;
        }
        
        if (length < static_cast<uint32>(6 + frame.header.length + 2)) {
            return false;
        }
        
        memcpy(frame.payload, buffer + 6, frame.header.length);
        
        frame.crc = buffer[6 + frame.header.length] | 
                   (buffer[6 + frame.header.length + 1] << 8);
        
        uint16 computedCrc = computeCRC16(buffer + 6, frame.header.length);
        if (computedCrc != frame.crc) {
            crcErrorCount_++;
            return false;
        }
        
        rxCount_++;
        
        return true;
    }
    
    uint16 computeCRC16(const uint8* data, uint16 length) {
        uint16 crc = 0xFFFF;
        
        for (uint16 i = 0; i < length; ++i) {
            crc ^= data[i];
            for (uint8 j = 0; j < 8; ++j) {
                if (crc & 0x0001) {
                    crc = (crc >> 1) ^ 0xA001;
                } else {
                    crc >>= 1;
                }
            }
        }
        
        return crc;
    }
    
    uint32 getTxCount() const { return txCount_; }
    uint32 getRxCount() const { return rxCount_; }
    uint32 getCrcErrorCount() const { return crcErrorCount_; }
};

class UARTDriver {
public:
    struct Config {
        uint32 baudRate;
        uint32 txBufferSize;
        uint32 rxBufferSize;
        uint8 txPin;
        uint8 rxPin;
        bool useDma;
        bool useInterrupt;
        
        Config() {
            baudRate = 921600;
            txBufferSize = 2048;
            rxBufferSize = 2048;
            txPin = 0;
            rxPin = 0;
            useDma = true;
            useInterrupt = true;
        }
    };

private:
    Config config_;
    uint8* txBuffer_;
    uint8* rxBuffer_;
    volatile uint32 txHead_;
    volatile uint32 txTail_;
    volatile uint32 rxHead_;
    volatile uint32 rxTail_;
    bool initialized_;

public:
    UARTDriver() 
        : txBuffer_(nullptr)
        , rxBuffer_(nullptr)
        , txHead_(0), txTail_(0)
        , rxHead_(0), rxTail_(0)
        , initialized_(false) {}
    
    ~UARTDriver() {
        if (txBuffer_) delete[] txBuffer_;
        if (rxBuffer_) delete[] rxBuffer_;
    }
    
    void init(const Config& config) {
        config_ = config;
        
        txBuffer_ = new uint8[config.txBufferSize];
        rxBuffer_ = new uint8[config.rxBufferSize];
        
        txHead_ = 0;
        txTail_ = 0;
        rxHead_ = 0;
        rxTail_ = 0;
        
        configureHardware();
        
        initialized_ = true;
    }
    
    uint32 write(const uint8* data, uint32 length) {
        if (!initialized_) return 0;
        
        uint32 written = 0;
        for (uint32 i = 0; i < length; ++i) {
            uint32 nextHead = (txHead_ + 1) % config_.txBufferSize;
            if (nextHead == txTail_) break;
            
            txBuffer_[txHead_] = data[i];
            txHead_ = nextHead;
            written++;
        }
        
        startTransmission();
        
        return written;
    }
    
    uint32 read(uint8* data, uint32 maxLength) {
        if (!initialized_) return 0;
        
        uint32 read = 0;
        while (read < maxLength && rxHead_ != rxTail_) {
            data[read++] = rxBuffer_[rxTail_];
            rxTail_ = (rxTail_ + 1) % config_.rxBufferSize;
        }
        
        return read;
    }
    
    uint32 available() const {
        if (rxHead_ >= rxTail_) {
            return rxHead_ - rxTail_;
        }
        return config_.rxBufferSize - rxTail_ + rxHead_;
    }
    
    void handleRxInterrupt() {
        uint8 data = readDataRegister();
        
        uint32 nextHead = (rxHead_ + 1) % config_.rxBufferSize;
        if (nextHead != rxTail_) {
            rxBuffer_[rxHead_] = data;
            rxHead_ = nextHead;
        }
    }
    
    void handleTxInterrupt() {
        if (txHead_ != txTail_) {
            writeDataRegister(txBuffer_[txTail_]);
            txTail_ = (txTail_ + 1) % config_.txBufferSize;
        } else {
            disableTxInterrupt();
        }
    }

private:
    void configureHardware() {
    }
    
    void startTransmission() {
        enableTxInterrupt();
    }
    
    uint8 readDataRegister() {
        return 0;
    }
    
    void writeDataRegister(uint8 data) {
    }
    
    void enableTxInterrupt() {
    }
    
    void disableTxInterrupt() {
    }
};

}

#endif
