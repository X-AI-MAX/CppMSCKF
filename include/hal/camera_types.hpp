#ifndef MSCKF_HAL_CAMERA_TYPES_HPP
#define MSCKF_HAL_CAMERA_TYPES_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"

namespace msckf {

struct CameraIntrinsics {
    float64 fx;
    float64 fy;
    float64 cx;
    float64 cy;
    float64 k1;
    float64 k2;
    float64 k3;
    float64 p1;
    float64 p2;
    uint32 width;
    uint32 height;
    
    CameraIntrinsics() 
        : fx(0), fy(0), cx(0), cy(0)
        , k1(0), k2(0), k3(0)
        , p1(0), p2(0)
        , width(0), height(0) {}
    
    CameraIntrinsics(float64 fx_, float64 fy_, float64 cx_, float64 cy_,
                     uint32 w, uint32 h)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_)
        , k1(0), k2(0), k3(0)
        , p1(0), p2(0)
        , width(w), height(h) {}
    
    Matrix<3, 3, float64> K() const {
        Matrix<3, 3, float64> Kmat = Matrix<3, 3, float64>::identity();
        Kmat(0, 0) = fx;
        Kmat(1, 1) = fy;
        Kmat(0, 2) = cx;
        Kmat(1, 2) = cy;
        return Kmat;
    }
    
    Matrix<3, 3, float64> Kinv() const {
        Matrix<3, 3, float64> KinvMat = Matrix<3, 3, float64>::identity();
        KinvMat(0, 0) = 1.0 / fx;
        KinvMat(1, 1) = 1.0 / fy;
        KinvMat(0, 2) = -cx / fx;
        KinvMat(1, 2) = -cy / fy;
        return KinvMat;
    }
};

struct CameraExtrinsics {
    Quaterniond q_cam_to_body;
    Vector3d t_cam_to_body;
    
    CameraExtrinsics() : q_cam_to_body(), t_cam_to_body({0, 0, 0}) {}
    
    CameraExtrinsics(const Quaterniond& q, const Vector3d& t)
        : q_cam_to_body(q), t_cam_to_body(t) {}
    
    SE3d camToBody() const {
        return SE3d(q_cam_to_body, t_cam_to_body);
    }
    
    SE3d bodyToCam() const {
        return camToBody().inverse();
    }
};

struct ImageData {
    uint64 timestamp;
    uint64 exposureStartTimestamp;
    uint64 exposureEndTimestamp;
    uint8* data;
    uint32 width;
    uint32 height;
    uint32 stride;
    uint8 channels;
    uint8 bitsPerPixel;
    float64 exposureTime;
    float64 gain;
    bool isBayer;
    uint8 bayerPattern;
    
    ImageData()
        : timestamp(0)
        , exposureStartTimestamp(0)
        , exposureEndTimestamp(0)
        , data(nullptr)
        , width(0), height(0), stride(0)
        , channels(1), bitsPerPixel(8)
        , exposureTime(0), gain(1.0)
        , isBayer(false), bayerPattern(0) {}
};

struct CameraConfig {
    uint32 width;
    uint32 height;
    uint32 frameRate;
    float64 exposureTime;
    float64 gain;
    bool autoExposure;
    bool autoGain;
    bool useHardwareTrigger;
    bool useMipi;
    uint32 mipiLanes;
    uint32 dvpBusWidth;
    bool useDmaDoubleBuffer;
    uint32 dmaBufferSize;
    uint32 triggerPin;
    uint32 flashPin;
    
    CameraConfig() {
        width = 640;
        height = 480;
        frameRate = 30;
        exposureTime = 10000;
        gain = 1.0;
        autoExposure = false;
        autoGain = false;
        useHardwareTrigger = true;
        useMipi = true;
        mipiLanes = 2;
        dvpBusWidth = 8;
        useDmaDoubleBuffer = true;
        dmaBufferSize = 1024 * 1024;
        triggerPin = 0;
        flashPin = 0;
    }
};

enum class CameraStatus {
    OK = 0,
    COMM_ERROR = 1,
    TIMEOUT = 2,
    BUFFER_OVERFLOW = 3,
    SELF_TEST_FAILED = 4,
    NOT_INITIALIZED = 5,
    INVALID_CONFIG = 6,
    TRIGGER_ERROR = 7
};

struct CameraDiagnostics {
    uint32 totalFrames;
    uint32 droppedFrames;
    uint32 errorCount;
    uint64 lastErrorTimestamp;
    CameraStatus lastError;
    float64 avgFrameInterval;
    float64 avgExposureTime;
    float64 avgGain;
    uint32 syncErrorCount;
    
    CameraDiagnostics() {
        totalFrames = 0;
        droppedFrames = 0;
        errorCount = 0;
        lastErrorTimestamp = 0;
        lastError = CameraStatus::OK;
        avgFrameInterval = 0;
        avgExposureTime = 0;
        avgGain = 0;
        syncErrorCount = 0;
    }
};

struct RollingShutterInfo {
    float64 rowReadoutTime;
    float64 lineDelay;
    uint32 totalRows;
    bool isRollingShutter;
    
    RollingShutterInfo() 
        : rowReadoutTime(0)
        , lineDelay(0)
        , totalRows(0)
        , isRollingShutter(true) {}
    
    float64 getTimestampForRow(uint32 row, uint64 frameStartTimestamp) const {
        return frameStartTimestamp + row * rowReadoutTime;
    }
    
    uint32 getRowForTimestamp(float64 deltaFromStart) const {
        if (rowReadoutTime > 0) {
            return static_cast<uint32>(deltaFromStart / rowReadoutTime);
        }
        return 0;
    }
};

}

#endif
