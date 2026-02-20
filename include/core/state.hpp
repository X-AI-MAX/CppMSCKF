#ifndef MSCKF_CORE_STATE_HPP
#define MSCKF_CORE_STATE_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../math/lie.hpp"
#include "../hal/imu_types.hpp"
#include "../hal/camera_types.hpp"

namespace msckf {

constexpr uint32 IMU_STATE_DIM = 15;
constexpr uint32 EXTRINSIC_STATE_DIM = 6;
constexpr uint32 CAMERA_STATE_DIM = 6;
constexpr uint32 FEATURE_STATE_DIM = 3;
constexpr uint32 MAX_CAMERA_FRAMES = 50;
constexpr uint32 MAX_FEATURES = 500;
constexpr uint32 MAX_STATE_DIM = IMU_STATE_DIM + EXTRINSIC_STATE_DIM + MAX_CAMERA_FRAMES * CAMERA_STATE_DIM;
constexpr uint32 NULLSPACE_DIM = 4;

struct IMUState {
    Vector3d position;
    Vector3d velocity;
    Quaterniond orientation;
    Vector3d gyroBias;
    Vector3d accelBias;
    uint64 timestamp;
    
    IMUState() 
        : position({0, 0, 0})
        , velocity({0, 0, 0})
        , orientation()
        , gyroBias({0, 0, 0})
        , accelBias({0, 0, 0})
        , timestamp(0) {}
    
    Vector3d gravity() const {
        return Vector3d({0, 0, -GRAVITY});
    }
};

struct CameraState {
    Quaterniond orientation;
    Vector3d position;
    uint64 timestamp;
    uint32 id;
    
    CameraState() 
        : orientation()
        , position({0, 0, 0})
        , timestamp(0)
        , id(0) {}
    
    SE3d pose() const {
        return SE3d(orientation, position);
    }
};

struct ExtrinsicState {
    Vector3d p_CB;
    Vector3d theta_CB;
    
    ExtrinsicState() 
        : p_CB({0, 0, 0})
        , theta_CB({0, 0, 0}) {}
    
    SE3d getTransform() const {
        return SE3d(SO3d::exp(theta_CB), p_CB);
    }
    
    Quaterniond getQuaternion() const {
        return SO3d::exp(theta_CB).quaternion();
    }
    
    Matrix3d getRotationMatrix() const {
        return SO3d::exp(theta_CB).matrix();
    }
};

struct Feature {
    uint32 id;
    float64 invDepth;
    Vector3d position;
    bool isTriangulated;
    uint32 firstFrameId;
    uint32 lastFrameId;
    uint32 trackCount;
    
    struct Observation {
        uint32 frameId;
        float64 u;
        float64 v;
    };
    
    static constexpr uint32 MAX_OBSERVATIONS = MAX_CAMERA_FRAMES;
    Observation observations[MAX_OBSERVATIONS];
    uint32 numObservations;
    
    Feature() 
        : id(0)
        , invDepth(0)
        , position({0, 0, 0})
        , isTriangulated(false)
        , firstFrameId(0)
        , lastFrameId(0)
        , trackCount(0)
        , numObservations(0) {}
    
    void addObservation(uint32 frameId, float64 u, float64 v) {
        if (numObservations < MAX_OBSERVATIONS) {
            observations[numObservations].frameId = frameId;
            observations[numObservations].u = u;
            observations[numObservations].v = v;
            numObservations++;
            lastFrameId = frameId;
            if (numObservations == 1) {
                firstFrameId = frameId;
            }
        }
    }
    
    bool hasObservation(uint32 frameId) const {
        for (uint32 i = 0; i < numObservations; ++i) {
            if (observations[i].frameId == frameId) return true;
        }
        return false;
    }
    
    bool getObservation(uint32 frameId, float64& u, float64& v) const {
        for (uint32 i = 0; i < numObservations; ++i) {
            if (observations[i].frameId == frameId) {
                u = observations[i].u;
                v = observations[i].v;
                return true;
            }
        }
        return false;
    }
};

struct ErrorState {
    Vector3d deltaPosition;
    Vector3d deltaVelocity;
    Vector3d deltaTheta;
    Vector3d deltaGyroBias;
    Vector3d deltaAccelBias;
    
    ErrorState() 
        : deltaPosition({0, 0, 0})
        , deltaVelocity({0, 0, 0})
        , deltaTheta({0, 0, 0})
        , deltaGyroBias({0, 0, 0})
        , deltaAccelBias({0, 0, 0}) {}
    
    static constexpr uint32 DIM = IMU_STATE_DIM;
    
    float64& operator[](uint32 i) {
        switch (i) {
            case 0: case 1: case 2: return deltaPosition[i];
            case 3: case 4: case 5: return deltaVelocity[i - 3];
            case 6: case 7: case 8: return deltaTheta[i - 6];
            case 9: case 10: case 11: return deltaGyroBias[i - 9];
            default: return deltaAccelBias[i - 12];
        }
    }
    
    const float64& operator[](uint32 i) const {
        switch (i) {
            case 0: case 1: case 2: return deltaPosition[i];
            case 3: case 4: case 5: return deltaVelocity[i - 3];
            case 6: case 7: case 8: return deltaTheta[i - 6];
            case 9: case 10: case 11: return deltaGyroBias[i - 9];
            default: return deltaAccelBias[i - 12];
        }
    }
};

struct MSCKFState {
    IMUState imuState;
    ExtrinsicState extrinsicState;
    CameraState cameraStates[MAX_CAMERA_FRAMES];
    uint32 numCameraStates;
    
    Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64> covariance;
    uint32 covarianceDim;
    
    Quaterniond q_IG;
    Vector3d p_I_G;
    
    Quaterniond q_CB;
    Vector3d p_C_B;
    bool extrinsicOnline;
    bool extrinsicConverged;
    uint64 extrinsicConvergeTime;
    
    uint64 timestamp;
    uint32 nextCameraId;
    
    MSCKFState() 
        : numCameraStates(0)
        , covarianceDim(IMU_STATE_DIM + EXTRINSIC_STATE_DIM)
        , q_IG()
        , p_I_G({0, 0, 0})
        , q_CB()
        , p_C_B({0, 0, 0})
        , extrinsicOnline(true)
        , extrinsicConverged(false)
        , extrinsicConvergeTime(0)
        , timestamp(0)
        , nextCameraId(0) {
        covariance = Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>::identity();
        for (uint32 i = 0; i < 3; ++i) {
            covariance(IMU_STATE_DIM + i, IMU_STATE_DIM + i) = 0.01;
            covariance(IMU_STATE_DIM + 3 + i, IMU_STATE_DIM + 3 + i) = 0.001;
        }
    }
    
    uint32 stateDim() const {
        return IMU_STATE_DIM + EXTRINSIC_STATE_DIM + numCameraStates * CAMERA_STATE_DIM;
    }
    
    uint32 imuStateStart() const { return 0; }
    uint32 extrinsicStateStart() const { return IMU_STATE_DIM; }
    uint32 cameraStateStart() const { return IMU_STATE_DIM + EXTRINSIC_STATE_DIM; }
    uint32 cameraStateIndex(uint32 camIdx) const { 
        return IMU_STATE_DIM + EXTRINSIC_STATE_DIM + camIdx * CAMERA_STATE_DIM; 
    }
    
    void injectErrorState(const ErrorState& delta) {
        imuState.position += delta.deltaPosition;
        imuState.velocity += delta.deltaVelocity;
        
        Quaterniond dq = Quaterniond::exp(delta.deltaTheta);
        imuState.orientation = dq * imuState.orientation;
        imuState.orientation.normalize();
        
        imuState.gyroBias += delta.deltaGyroBias;
        imuState.accelBias += delta.deltaAccelBias;
    }
    
    void injectExtrinsicErrorState(const Vector3d& delta_p, const Vector3d& delta_theta) {
        extrinsicState.p_CB += delta_p;
        extrinsicState.theta_CB += delta_theta;
        
        q_CB = extrinsicState.getQuaternion();
        p_C_B = extrinsicState.p_CB;
    }
    
    Matrix<3, 3, float64> getRotationJacobian() const {
        return imuState.orientation.toRotationMatrix();
    }
};

struct IMUNoiseParams {
    float64 gyroNoiseDensity;
    float64 gyroBiasRandomWalk;
    float64 accelNoiseDensity;
    float64 accelBiasRandomWalk;
    
    IMUNoiseParams() 
        : gyroNoiseDensity(1e-4)
        , gyroBiasRandomWalk(1e-6)
        , accelNoiseDensity(1e-3)
        , accelBiasRandomWalk(1e-5) {}
    
    float64 gyroNoiseVariance(float64 dt) const {
        return gyroNoiseDensity * gyroNoiseDensity / dt;
    }
    
    float64 accelNoiseVariance(float64 dt) const {
        return accelNoiseDensity * accelNoiseDensity / dt;
    }
    
    float64 gyroBiasVariance(float64 dt) const {
        return gyroBiasRandomWalk * gyroBiasRandomWalk * dt;
    }
    
    float64 accelBiasVariance(float64 dt) const {
        return accelBiasRandomWalk * accelBiasRandomWalk * dt;
    }
};

struct AdaptiveNoiseParams {
    float64 motionThreshold;
    float64 minScale;
    float64 maxScale;
    float64 currentScale;
    
    AdaptiveNoiseParams() 
        : motionThreshold(1.0)
        , minScale(0.1)
        , maxScale(10.0)
        , currentScale(1.0) {}
    
    float64 computeMotionFactor(const Vector3d& accel, const Vector3d& gyro) const {
        float64 accelNorm = (accel - Vector3d({0, 0, -GRAVITY})).norm();
        float64 gyroNorm = gyro.norm();
        
        float64 motionLevel = accelNorm / GRAVITY + gyroNorm / 10.0;
        return motionLevel / motionThreshold;
    }
    
    void updateScale(const Vector3d& accel, const Vector3d& gyro) {
        float64 factor = computeMotionFactor(accel, gyro);
        currentScale = clamp(factor, minScale, maxScale);
    }
};

}

#endif
