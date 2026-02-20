#ifndef MSCKF_HAL_IMU_TYPES_HPP
#define MSCKF_HAL_IMU_TYPES_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"

namespace msckf {

struct IMUData {
    uint64 timestamp;
    Vector3d accel;
    Vector3d gyro;
    float32 temperature;
    
    IMUData() : timestamp(0), temperature(0) {}
    
    IMUData(uint64 ts, const Vector3d& a, const Vector3d& g, float32 temp = 0)
        : timestamp(ts), accel(a), gyro(g), temperature(temp) {}
};

struct IMUCalibration {
    Vector3d accelBias;
    Vector3d gyroBias;
    Vector3d accelScale;
    Vector3d gyroScale;
    Matrix3d accelMisalignment;
    Matrix3d gyroMisalignment;
    Matrix3d R_imu_to_body;
    Vector3d t_imu_to_body;
    
    IMUCalibration() {
        accelBias = Vector3d({0, 0, 0});
        gyroBias = Vector3d({0, 0, 0});
        accelScale = Vector3d({1, 1, 1});
        gyroScale = Vector3d({1, 1, 1});
        accelMisalignment = Matrix3d::identity();
        gyroMisalignment = Matrix3d::identity();
        R_imu_to_body = Matrix3d::identity();
        t_imu_to_body = Vector3d({0, 0, 0});
    }
    
    Vector3d correctAccel(const Vector3d& raw) const {
        Vector3d scaled;
        scaled[0] = (raw[0] - accelBias[0]) * accelScale[0];
        scaled[1] = (raw[1] - accelBias[1]) * accelScale[1];
        scaled[2] = (raw[2] - accelBias[2]) * accelScale[2];
        return accelMisalignment * scaled;
    }
    
    Vector3d correctGyro(const Vector3d& raw) const {
        Vector3d scaled;
        scaled[0] = (raw[0] - gyroBias[0]) * gyroScale[0];
        scaled[1] = (raw[1] - gyroBias[1]) * gyroScale[1];
        scaled[2] = (raw[2] - gyroBias[2]) * gyroScale[2];
        return gyroMisalignment * scaled;
    }
    
    Vector3d transformToBody(const Vector3d& imuVec) const {
        return R_imu_to_body * imuVec;
    }
};

struct IMUConfig {
    uint32 sampleRate;
    uint32 bandwidth;
    float32 accelRange;
    float32 gyroRange;
    bool useDma;
    bool useInterrupt;
    uint32 dmaBufferSize;
    uint32 interruptPin;
    uint32 spiCsPin;
    uint32 i2cAddress;
    bool useSpi;
    
    IMUConfig() {
        sampleRate = 1000;
        bandwidth = 200;
        accelRange = 16.0f;
        gyroRange = 2000.0f;
        useDma = true;
        useInterrupt = true;
        dmaBufferSize = 1024;
        interruptPin = 0;
        spiCsPin = 0;
        i2cAddress = 0x68;
        useSpi = true;
    }
};

enum class IMUStatus {
    OK = 0,
    COMM_ERROR = 1,
    DATA_OVERFLOW = 2,
    SELF_TEST_FAILED = 3,
    NOT_INITIALIZED = 4,
    INVALID_CONFIG = 5
};

struct IMUDiagnostics {
    uint32 totalSamples;
    uint32 errorCount;
    uint32 overflowCount;
    uint64 lastErrorTimestamp;
    IMUStatus lastError;
    float32 avgSampleInterval;
    float32 jitterNs;
    
    IMUDiagnostics() {
        totalSamples = 0;
        errorCount = 0;
        overflowCount = 0;
        lastErrorTimestamp = 0;
        lastError = IMUStatus::OK;
        avgSampleInterval = 0;
        jitterNs = 0;
    }
};

}

#endif
