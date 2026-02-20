#ifndef MSCKF_UTILITY_BLACK_BOX_HPP
#define MSCKF_UTILITY_BLACK_BOX_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../core/state.hpp"
#include "../hal/imu_types.hpp"
#include "../hal/camera_types.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 BLACKBOX_BUFFER_SIZE = 60000;
constexpr uint32 BLACKBOX_IMU_SIZE = 10000;
constexpr uint32 BLACKBOX_IMAGE_SIZE = 100;
constexpr uint32 BLACKBOX_STATE_SIZE = 1000;
constexpr uint32 BLACKBOX_INPUT_SIZE = 500;

enum class BlackBoxTrigger : uint8 {
    NONE = 0,
    ATTITUDE_ANOMALY = 1,
    CONTROL_SATURATION = 2,
    MANUAL_MARK = 3,
    SENSOR_FAILURE = 4,
    ESTIMATOR_FAILURE = 5,
    LOW_BATTERY = 6,
    COMMUNICATION_LOSS = 7
};

struct BlackBoxEvent {
    BlackBoxTrigger trigger;
    uint64 timestamp;
    uint32 severity;
    char description[64];
    
    BlackBoxEvent() 
        : trigger(BlackBoxTrigger::NONE)
        , timestamp(0)
        , severity(0) {
        memset(description, 0, sizeof(description));
    }
};

struct BlackBoxIMURecord {
    Vector3d accel;
    Vector3d gyro;
    uint64 timestamp;
    uint8 flags;
    
    BlackBoxIMURecord() 
        : accel({0, 0, 0})
        , gyro({0, 0, 0})
        , timestamp(0)
        , flags(0) {}
};

struct BlackBoxStateRecord {
    Vector3d position;
    Vector3d velocity;
    Quaterniond orientation;
    Vector3d gyroBias;
    Vector3d accelBias;
    uint64 timestamp;
    uint32 numFeatures;
    uint32 numCameraFrames;
    float64 positionCov[9];
    float64 orientationCov[9];
    
    BlackBoxStateRecord() 
        : position({0, 0, 0})
        , velocity({0, 0, 0})
        , orientation()
        , gyroBias({0, 0, 0})
        , accelBias({0, 0, 0})
        , timestamp(0)
        , numFeatures(0)
        , numCameraFrames(0) {
        memset(positionCov, 0, sizeof(positionCov));
        memset(orientationCov, 0, sizeof(orientationCov));
    }
};

struct BlackBoxInputRecord {
    float64 throttle[4];
    Vector3d targetPosition;
    Vector3d targetVelocity;
    uint8 flightMode;
    uint64 timestamp;
    
    BlackBoxInputRecord() 
        : flightMode(0)
        , timestamp(0) {
        memset(throttle, 0, sizeof(throttle));
        targetPosition = Vector3d({0, 0, 0});
        targetVelocity = Vector3d({0, 0, 0});
    }
};

class BlackBox {
public:
    struct Config {
        uint32 maxRecordingTime;
        uint32 preEventTime;
        uint32 postEventTime;
        bool enableContinuousRecording;
        bool enableImageRecording;
        bool enableCompression;
        float64 attitudeAnomalyThreshold;
        float64 controlSaturationThreshold;
        uint32 maxEvents;
        
        Config() 
            : maxRecordingTime(600)
            , preEventTime(30)
            , postEventTime(30)
            , enableContinuousRecording(true)
            , enableImageRecording(false)
            , enableCompression(true)
            , attitudeAnomalyThreshold(0.5)
            , controlSaturationThreshold(0.95)
            , maxEvents(10) {}
    };

private:
    Config config_;
    
    BlackBoxIMURecord imuBuffer_[BLACKBOX_IMU_SIZE];
    uint32 imuHead_;
    uint32 imuTail_;
    uint32 imuCount_;
    
    BlackBoxStateRecord stateBuffer_[BLACKBOX_STATE_SIZE];
    uint32 stateHead_;
    uint32 stateTail_;
    uint32 stateCount_;
    
    BlackBoxInputRecord inputBuffer_[BLACKBOX_INPUT_SIZE];
    uint32 inputHead_;
    uint32 inputTail_;
    uint32 inputCount_;
    
    BlackBoxEvent events_[10];
    uint32 numEvents_;
    
    uint8 compressedBuffer_[BLACKBOX_BUFFER_SIZE];
    uint32 compressedSize_;
    
    bool isRecording_;
    bool isEventActive_;
    uint64 eventStartTime_;
    uint64 recordingStartTime_;
    
    uint64 lastIMUTime_;
    uint64 lastStateTime_;
    uint64 lastInputTime_;
    
    uint32 totalBytesWritten_;
    uint32 totalRecords_;

public:
    BlackBox() 
        : imuHead_(0), imuTail_(0), imuCount_(0)
        , stateHead_(0), stateTail_(0), stateCount_(0)
        , inputHead_(0), inputTail_(0), inputCount_(0)
        , numEvents_(0)
        , compressedSize_(0)
        , isRecording_(false)
        , isEventActive_(false)
        , eventStartTime_(0)
        , recordingStartTime_(0)
        , lastIMUTime_(0)
        , lastStateTime_(0)
        , lastInputTime_(0)
        , totalBytesWritten_(0)
        , totalRecords_(0) {
        for (uint32 i = 0; i < BLACKBOX_BUFFER_SIZE; ++i) {
            compressedBuffer_[i] = 0;
        }
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        
        compressedSize_ = 0;
        for (uint32 i = 0; i < BLACKBOX_BUFFER_SIZE; ++i) {
            compressedBuffer_[i] = 0;
        }
        
        isRecording_ = config_.enableContinuousRecording;
        isEventActive_ = false;
        numEvents_ = 0;
        totalBytesWritten_ = 0;
        totalRecords_ = 0;
    }
    
    void recordIMU(const IMUData& imuData) {
        if (!isRecording_) return;
        
        BlackBoxIMURecord record;
        record.accel = imuData.accel;
        record.gyro = imuData.gyro;
        record.timestamp = imuData.timestamp;
        record.flags = 0;
        
        imuBuffer_[imuHead_] = record;
        imuHead_ = (imuHead_ + 1) % BLACKBOX_IMU_SIZE;
        
        if (imuCount_ < BLACKBOX_IMU_SIZE) {
            imuCount_++;
        } else {
            imuTail_ = (imuTail_ + 1) % BLACKBOX_IMU_SIZE;
        }
        
        lastIMUTime_ = imuData.timestamp;
        totalRecords_++;
    }
    
    void recordState(const MSCKFState& state) {
        if (!isRecording_) return;
        
        BlackBoxStateRecord record;
        record.position = state.imuState.position;
        record.velocity = state.imuState.velocity;
        record.orientation = state.imuState.orientation;
        record.gyroBias = state.imuState.gyroBias;
        record.accelBias = state.imuState.accelBias;
        record.timestamp = state.timestamp;
        record.numFeatures = 0;
        record.numCameraFrames = state.numCameraStates;
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                record.positionCov[i * 3 + j] = state.covariance(i, j);
                record.orientationCov[i * 3 + j] = state.covariance(i + 6, j + 6);
            }
        }
        
        stateBuffer_[stateHead_] = record;
        stateHead_ = (stateHead_ + 1) % BLACKBOX_STATE_SIZE;
        
        if (stateCount_ < BLACKBOX_STATE_SIZE) {
            stateCount_++;
        } else {
            stateTail_ = (stateTail_ + 1) % BLACKBOX_STATE_SIZE;
        }
        
        lastStateTime_ = state.timestamp;
    }
    
    void recordInput(const BlackBoxInputRecord& input) {
        if (!isRecording_) return;
        
        inputBuffer_[inputHead_] = input;
        inputHead_ = (inputHead_ + 1) % BLACKBOX_INPUT_SIZE;
        
        if (inputCount_ < BLACKBOX_INPUT_SIZE) {
            inputCount_++;
        } else {
            inputTail_ = (inputTail_ + 1) % BLACKBOX_INPUT_SIZE;
        }
        
        lastInputTime_ = input.timestamp;
    }
    
    void triggerEvent(BlackBoxTrigger trigger, uint32 severity, const char* description) {
        if (numEvents_ >= config_.maxEvents) return;
        
        BlackBoxEvent& event = events_[numEvents_++];
        event.trigger = trigger;
        event.timestamp = lastIMUTime_;
        event.severity = severity;
        
        uint32 descLen = min(strlen(description), sizeof(event.description) - 1);
        memcpy(event.description, description, descLen);
        event.description[descLen] = '\0';
        
        isEventActive_ = true;
        eventStartTime_ = lastIMUTime_;
    }
    
    void checkForAnomalies(const MSCKFState& state, const float64* controls, uint32 numControls) {
        Vector3d euler = state.imuState.orientation.toEulerAngles();
        
        if (abs(euler[0]) > config_.attitudeAnomalyThreshold ||
            abs(euler[1]) > config_.attitudeAnomalyThreshold) {
            triggerEvent(BlackBoxTrigger::ATTITUDE_ANOMALY, 2, "Attitude anomaly detected");
        }
        
        if (controls && numControls >= 4) {
            bool saturated = true;
            for (uint32 i = 0; i < 4; ++i) {
                if (controls[i] < config_.controlSaturationThreshold) {
                    saturated = false;
                    break;
                }
            }
            
            if (saturated) {
                triggerEvent(BlackBoxTrigger::CONTROL_SATURATION, 1, "Control saturation detected");
            }
        }
    }
    
    void markManual() {
        triggerEvent(BlackBoxTrigger::MANUAL_MARK, 0, "Manual marker");
    }
    
    void update(uint64 currentTime) {
        if (isEventActive_) {
            uint64 eventDuration = (currentTime - eventStartTime_) / 1000000;
            
            if (eventDuration > config_.postEventTime) {
                isEventActive_ = false;
                saveEventData();
            }
        }
        
        if (config_.enableContinuousRecording) {
            uint64 recordingDuration = (currentTime - recordingStartTime_) / 1000000;
            
            if (recordingDuration > config_.maxRecordingTime) {
                compressAndSave();
            }
        }
    }
    
    void saveEventData() {
        uint32 preEventRecords = config_.preEventTime * 200;
        
        uint32 startIdx = 0;
        if (imuCount_ > preEventRecords) {
            startIdx = (imuHead_ + BLACKBOX_IMU_SIZE - preEventRecords) % BLACKBOX_IMU_SIZE;
        }
        
        uint32 recordCount = 0;
        uint32 idx = startIdx;
        
        while (idx != imuHead_) {
            recordCount++;
            idx = (idx + 1) % BLACKBOX_IMU_SIZE;
        }
        
        compressData(startIdx, recordCount);
    }
    
    void compressData(uint32 startIdx, uint32 count) {
        if (!config_.enableCompression) {
            return;
        }
        
        compressedSize_ = 0;
        
        uint8 header[16];
        header[0] = 0xBB;
        header[1] = 0x01;
        memcpy(header + 2, &count, sizeof(uint32));
        memcpy(header + 6, &lastIMUTime_, sizeof(uint64));
        
        memcpy(compressedBuffer_ + compressedSize_, header, 16);
        compressedSize_ += 16;
        
        uint32 idx = startIdx;
        for (uint32 i = 0; i < count && compressedSize_ < BLACKBOX_BUFFER_SIZE - 100; ++i) {
            BlackBoxIMURecord& record = imuBuffer_[idx];
            
            int16_t accel[3], gyro[3];
            for (uint32 j = 0; j < 3; ++j) {
                accel[j] = static_cast<int16_t>(record.accel[j] * 1000);
                gyro[j] = static_cast<int16_t>(record.gyro[j] * 1000);
            }
            
            memcpy(compressedBuffer_ + compressedSize_, accel, 6);
            compressedSize_ += 6;
            memcpy(compressedBuffer_ + compressedSize_, gyro, 6);
            compressedSize_ += 6;
            memcpy(compressedBuffer_ + compressedSize_, &record.timestamp, 8);
            compressedSize_ += 8;
            
            idx = (idx + 1) % BLACKBOX_IMU_SIZE;
        }
        
        totalBytesWritten_ += compressedSize_;
    }
    
    void compressAndSave() {
        compressData(imuTail_, imuCount_);
        
        imuHead_ = 0;
        imuTail_ = 0;
        imuCount_ = 0;
        
        stateHead_ = 0;
        stateTail_ = 0;
        stateCount_ = 0;
        
        recordingStartTime_ = lastIMUTime_;
    }
    
    uint32 serialize(uint8* buffer, uint32 maxSize, uint64 startTime, uint64 endTime) {
        uint32 idx = 0;
        
        buffer[idx++] = 0xBB;
        buffer[idx++] = 0x02;
        
        uint32 imuStart = 0, imuEnd = 0;
        findTimeRange(imuBuffer_, BLACKBOX_IMU_SIZE, imuHead_, startTime, endTime, imuStart, imuEnd);
        
        uint32 stateStart = 0, stateEnd = 0;
        findTimeRange(stateBuffer_, BLACKBOX_STATE_SIZE, stateHead_, startTime, endTime, stateStart, stateEnd);
        
        uint32 imuRecords = (imuEnd - imuStart + BLACKBOX_IMU_SIZE) % BLACKBOX_IMU_SIZE;
        uint32 stateRecords = (stateEnd - stateStart + BLACKBOX_STATE_SIZE) % BLACKBOX_STATE_SIZE;
        
        memcpy(buffer + idx, &imuRecords, sizeof(uint32)); idx += 4;
        memcpy(buffer + idx, &stateRecords, sizeof(uint32)); idx += 4;
        memcpy(buffer + idx, &numEvents_, sizeof(uint32)); idx += 4;
        
        uint32 imuIdx = imuStart;
        for (uint32 i = 0; i < imuRecords && idx + 32 < maxSize; ++i) {
            BlackBoxIMURecord& r = imuBuffer_[imuIdx];
            
            memcpy(buffer + idx, r.accel.data(), 24); idx += 24;
            memcpy(buffer + idx, r.gyro.data(), 24); idx += 24;
            memcpy(buffer + idx, &r.timestamp, 8); idx += 8;
            
            imuIdx = (imuIdx + 1) % BLACKBOX_IMU_SIZE;
        }
        
        uint32 stateIdx = stateStart;
        for (uint32 i = 0; i < stateRecords && idx + 200 < maxSize; ++i) {
            BlackBoxStateRecord& r = stateBuffer_[stateIdx];
            
            memcpy(buffer + idx, r.position.data(), 24); idx += 24;
            memcpy(buffer + idx, r.velocity.data(), 24); idx += 24;
            
            float64 quat[4] = {r.orientation.w, r.orientation.x, r.orientation.y, r.orientation.z};
            memcpy(buffer + idx, quat, 32); idx += 32;
            
            memcpy(buffer + idx, r.gyroBias.data(), 24); idx += 24;
            memcpy(buffer + idx, r.accelBias.data(), 24); idx += 24;
            memcpy(buffer + idx, &r.timestamp, 8); idx += 8;
            
            stateIdx = (stateIdx + 1) % BLACKBOX_STATE_SIZE;
        }
        
        for (uint32 i = 0; i < numEvents_ && idx + 80 < maxSize; ++i) {
            BlackBoxEvent& e = events_[i];
            
            buffer[idx++] = static_cast<uint8>(e.trigger);
            memcpy(buffer + idx, &e.timestamp, 8); idx += 8;
            memcpy(buffer + idx, &e.severity, 4); idx += 4;
            memcpy(buffer + idx, e.description, 64); idx += 64;
        }
        
        return idx;
    }
    
    template<typename T>
    void findTimeRange(T* buffer, uint32 size, uint32 head, 
                        uint64 startTime, uint64 endTime,
                        uint32& startIdx, uint32& endIdx) {
        startIdx = 0;
        endIdx = head;
        
        for (uint32 i = 0; i < size; ++i) {
            uint32 idx = (head + size - 1 - i) % size;
            if (buffer[idx].timestamp >= startTime && buffer[idx].timestamp <= endTime) {
                if (endIdx == head) endIdx = idx;
                startIdx = idx;
            }
        }
    }
    
    float64 getCompressionRatio() const {
        if (compressedSize_ == 0) return 1.0;
        
        uint32 originalSize = totalRecords_ * sizeof(BlackBoxIMURecord);
        return static_cast<float64>(compressedSize_) / originalSize;
    }
    
    void clear() {
        imuHead_ = 0;
        imuTail_ = 0;
        imuCount_ = 0;
        
        stateHead_ = 0;
        stateTail_ = 0;
        stateCount_ = 0;
        
        inputHead_ = 0;
        inputTail_ = 0;
        inputCount_ = 0;
        
        numEvents_ = 0;
        compressedSize_ = 0;
        totalRecords_ = 0;
        totalBytesWritten_ = 0;
        isEventActive_ = false;
    }
    
    bool isRecording() const { return isRecording_; }
    bool isEventActive() const { return isEventActive_; }
    uint32 getNumEvents() const { return numEvents_; }
    uint32 getIMUCount() const { return imuCount_; }
    uint32 getStateCount() const { return stateCount_; }
    uint32 getInputCount() const { return inputCount_; }
    uint32 getCompressedSize() const { return compressedSize_; }
    uint32 getTotalRecords() const { return totalRecords_; }
    uint32 getTotalBytesWritten() const { return totalBytesWritten_; }
    const BlackBoxEvent* getEvents() const { return events_; }
};

}

#endif
