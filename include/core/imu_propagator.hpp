#ifndef MSCKF_CORE_IMU_PROPAGATOR_HPP
#define MSCKF_CORE_IMU_PROPAGATOR_HPP

#include "state.hpp"
#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../math/lie.hpp"
#include "../hal/imu_types.hpp"

namespace msckf {

class IMUPropagator {
public:
    struct Config {
        float64 gravityMagnitude;
        float64 maxImuDeltaT;
        uint32 maxImuBuffer;
        bool useAdaptiveNoise;
        bool enforcePositiveDefinite;
        float64 minEigenvalue;
        
        Config() 
            : gravityMagnitude(GRAVITY)
            , maxImuDeltaT(0.01)
            , maxImuBuffer(1000)
            , useAdaptiveNoise(true)
            , enforcePositiveDefinite(true)
            , minEigenvalue(1e-6) {}
    };

private:
    Config config_;
    IMUNoiseParams noiseParams_;
    AdaptiveNoiseParams adaptiveParams_;
    
    IMUData imuBuffer_[1000];
    uint32 bufferHead_;
    uint32 bufferTail_;
    uint32 bufferSize_;
    
    IMUState propagatedState_;
    Matrix<15, 15, float64> F_;
    Matrix<15, 12, float64> G_;
    Matrix<15, 15, float64> Phi_;
    Matrix<15, 15, float64> P_imu_;
    Matrix<21, 21, float64> P_imu_ext_;
    
    uint64 lastPropagationTime_;
    bool initialized_;
    
    uint32 consecutiveAbnormalCount_;
    static constexpr uint32 ABNORMAL_THRESHOLD = 10;

public:
    IMUPropagator() 
        : bufferHead_(0), bufferTail_(0), bufferSize_(0)
        , lastPropagationTime_(0), initialized_(false)
        , consecutiveAbnormalCount_(0) {}
    
    void init(const IMUNoiseParams& noiseParams, const Config& config = Config()) {
        noiseParams_ = noiseParams;
        config_ = config;
        initialized_ = true;
        
        F_ = Matrix<15, 15, float64>::identity();
        G_ = Matrix<15, 12, float64>();
        Phi_ = Matrix<15, 15, float64>::identity();
        P_imu_ = Matrix<15, 15, float64>::identity();
        P_imu_ext_ = Matrix<21, 21, float64>::identity();
    }
    
    void setState(const IMUState& state) {
        propagatedState_ = state;
        lastPropagationTime_ = state.timestamp;
    }
    
    const IMUState& getState() const {
        return propagatedState_;
    }
    
    void addImuData(const IMUData& data) {
        if (bufferSize_ < config_.maxImuBuffer) {
            imuBuffer_[bufferHead_] = data;
            bufferHead_ = (bufferHead_ + 1) % config_.maxImuBuffer;
            bufferSize_++;
        } else {
            imuBuffer_[bufferHead_] = data;
            bufferHead_ = (bufferHead_ + 1) % config_.maxImuBuffer;
            bufferTail_ = (bufferTail_ + 1) % config_.maxImuBuffer;
        }
    }
    
    bool propagateTo(uint64 timestamp, MSCKFState& state) {
        if (bufferSize_ == 0) return false;
        
        IMUData imuData[1000];
        uint32 count = 0;
        
        while (bufferSize_ > 0) {
            IMUData& data = imuBuffer_[bufferTail_];
            
            if (data.timestamp > timestamp) break;
            
            imuData[count++] = data;
            bufferTail_ = (bufferTail_ + 1) % config_.maxImuBuffer;
            bufferSize_--;
        }
        
        if (count == 0) return false;
        
        for (uint32 i = 0; i < count; ++i) {
            propagateOneImu(state, imuData[i]);
        }
        
        if (count > 0 && bufferSize_ > 0) {
            IMUData& nextImu = imuBuffer_[bufferTail_];
            if (nextImu.timestamp > timestamp) {
                interpolateAndPropagate(state, imuData[count - 1], nextImu, timestamp);
            }
        }
        
        return true;
    }
    
    void propagateOneImu(MSCKFState& state, const IMUData& imuData) {
        float64 dt = (imuData.timestamp - state.imuState.timestamp) * 1e-6;
        
        if (dt <= 0 || dt > config_.maxImuDeltaT) {
            state.imuState.timestamp = imuData.timestamp;
            return;
        }
        
        bool isAbnormal = checkImuAbnormality(imuData);
        
        if (config_.useAdaptiveNoise) {
            adaptiveParams_.updateScale(imuData.accel, imuData.gyro);
            if (isAbnormal) {
                adaptiveParams_.currentScale = min(adaptiveParams_.currentScale * 10.0, 100.0);
            }
        }
        
        Vector3d accel = imuData.accel - state.imuState.accelBias;
        Vector3d gyro = imuData.gyro - state.imuState.gyroBias;
        
        Matrix3d R = state.imuState.orientation.toRotationMatrix();
        Vector3d accelWorld = R * accel + state.imuState.gravity();
        
        Vector3d midGyro = gyro;
        Quaterniond dq = Quaterniond::exp(gyro * dt * 0.5);
        Quaterniond qMid = state.imuState.orientation * dq;
        Matrix3d RMid = qMid.toRotationMatrix();
        
        Vector3d midAccel = RMid * accel + state.imuState.gravity();
        
        Vector3d newVelocity = state.imuState.velocity + midAccel * dt;
        Vector3d newPosition = state.imuState.position + 
                               (state.imuState.velocity + newVelocity) * dt * 0.5;
        
        Quaterniond dqFull = Quaterniond::exp(gyro * dt);
        Quaterniond newOrientation = state.imuState.orientation * dqFull;
        newOrientation.normalize();
        
        computeStateTransitionMatrix(state.imuState, accel, gyro, dt);
        
        Matrix<12, 12, float64> Q = computeNoiseCovariance(dt);
        Matrix<15, 15, float64> Qd = G_ * Q * G_.transpose();
        
        P_imu_ = Phi_ * P_imu_ * Phi_.transpose() + Qd;
        P_imu_.symmetrize();
        
        state.imuState.position = newPosition;
        state.imuState.velocity = newVelocity;
        state.imuState.orientation = newOrientation;
        state.imuState.timestamp = imuData.timestamp;
        state.timestamp = imuData.timestamp;
        
        propagateCovariance(state, Phi_);
        
        if (config_.enforcePositiveDefinite) {
            enforceCovariancePositiveDefinite(state);
        }
    }
    
    void interpolateAndPropagate(MSCKFState& state, const IMUData& prev, 
                                 const IMUData& next, uint64 targetTime) {
        float64 t1 = static_cast<float64>(prev.timestamp);
        float64 t2 = static_cast<float64>(next.timestamp);
        float64 tt = static_cast<float64>(targetTime);
        
        float64 alpha = (tt - t1) / (t2 - t1);
        alpha = clamp(alpha, 0.0, 1.0);
        
        IMUData interpData;
        interpData.timestamp = targetTime;
        interpData.accel = prev.accel + (next.accel - prev.accel) * alpha;
        interpData.gyro = prev.gyro + (next.gyro - prev.gyro) * alpha;
        interpData.temperature = prev.temperature;
        
        propagateOneImu(state, interpData);
    }
    
    const Matrix<15, 15, float64>& getImuCovariance() const {
        return P_imu_;
    }
    
    void resetImuCovariance(const Matrix<15, 15, float64>& P) {
        P_imu_ = P;
    }
    
    float64 getAdaptiveScale() const {
        return adaptiveParams_.currentScale;
    }
    
    bool isAbnormalState() const {
        return consecutiveAbnormalCount_ > ABNORMAL_THRESHOLD;
    }
    
    void resetAbnormalState() {
        consecutiveAbnormalCount_ = 0;
    }

private:
    bool checkImuAbnormality(const IMUData& imuData) {
        float64 accelNorm = imuData.accel.norm();
        float64 gyroNorm = imuData.gyro.norm();
        
        bool isAbnormal = (accelNorm > 4.0 * GRAVITY || gyroNorm > 10.0);
        
        if (isAbnormal) {
            consecutiveAbnormalCount_++;
        } else {
            consecutiveAbnormalCount_ = 0;
        }
        
        return isAbnormal;
    }
    
    void computeStateTransitionMatrix(const IMUState& state, 
                                      const Vector3d& accel, 
                                      const Vector3d& gyro, 
                                      float64 dt) {
        F_ = Matrix<15, 15, float64>();
        
        Matrix3d R = state.orientation.toRotationMatrix();
        Matrix3d thetaHat = skewSymmetric(gyro);
        Matrix3d aHat = skewSymmetric(accel);
        
        for (uint32 i = 0; i < 3; ++i) {
            F_(i, i + 3) = dt;
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                F_(i, j + 6) = -0.5 * dt * dt * (R * aHat)(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                F_(i, j + 12) = -0.5 * dt * dt * R(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                F_(i + 3, j + 6) = -dt * (R * aHat)(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                F_(i + 3, j + 12) = -dt * R(i, j);
            }
        }
        
        Matrix3d Jr = SO3d::rightJacobian(gyro * dt);
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                F_(i + 6, j + 6) = Jr(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            F_(i + 6, i + 9) = -dt;
        }
        
        for (uint32 i = 0; i < 15; ++i) {
            F_(i, i) += 1.0;
        }
        
        Phi_ = F_;
        
        G_ = Matrix<15, 12, float64>();
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                G_(i, j) = 0.5 * dt * dt * R(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                G_(i + 3, j) = dt * R(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            G_(i + 6, i + 3) = -dt;
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            G_(i + 9, i + 6) = dt;
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            G_(i + 12, i + 9) = dt;
        }
    }
    
    Matrix<12, 12, float64> computeNoiseCovariance(float64 dt) {
        Matrix<12, 12, float64> Q = Matrix<12, 12, float64>();
        
        float64 scale = adaptiveParams_.currentScale;
        float64 na = noiseParams_.accelNoiseVariance(dt) * scale;
        float64 nw = noiseParams_.gyroNoiseVariance(dt) * scale;
        float64 nba = noiseParams_.accelBiasVariance(dt);
        float64 nbw = noiseParams_.gyroBiasVariance(dt);
        
        for (uint32 i = 0; i < 3; ++i) {
            Q(i, i) = na;
            Q(i + 3, i + 3) = nw;
            Q(i + 6, i + 6) = nba;
            Q(i + 9, i + 9) = nbw;
        }
        
        return Q;
    }
    
    void propagateCovariance(MSCKFState& state, const Matrix<15, 15, float64>& Phi) {
        uint32 dim = state.covarianceDim;
        
        Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64> P_new;
        
        for (uint32 i = 0; i < dim; ++i) {
            for (uint32 j = 0; j < dim; ++j) {
                float64 sum = 0;
                
                if (i < IMU_STATE_DIM && j < IMU_STATE_DIM) {
                    for (uint32 k = 0; k < IMU_STATE_DIM; ++k) {
                        for (uint32 l = 0; l < IMU_STATE_DIM; ++l) {
                            sum += Phi(i, k) * state.covariance(k, l) * Phi(j, l);
                        }
                    }
                    sum += P_imu_(i, j);
                } else if (i < IMU_STATE_DIM) {
                    for (uint32 k = 0; k < IMU_STATE_DIM; ++k) {
                        sum += Phi(i, k) * state.covariance(k, j);
                    }
                } else if (j < IMU_STATE_DIM) {
                    for (uint32 k = 0; k < IMU_STATE_DIM; ++k) {
                        sum += state.covariance(i, k) * Phi(j, k);
                    }
                } else {
                    sum = state.covariance(i, j);
                }
                
                P_new(i, j) = sum;
            }
        }
        
        state.covariance = P_new;
        state.covariance.symmetrize();
    }
    
    void enforceCovariancePositiveDefinite(MSCKFState& state) {
        uint32 dim = state.covarianceDim;
        
        state.covariance.symmetrize();
        
        float64 minDiag = state.covariance(0, 0);
        for (uint32 i = 1; i < dim; ++i) {
            if (state.covariance(i, i) < minDiag) {
                minDiag = state.covariance(i, i);
            }
        }
        
        if (minDiag < config_.minEigenvalue) {
            float64 correction = config_.minEigenvalue - minDiag;
            for (uint32 i = 0; i < dim; ++i) {
                state.covariance(i, i) += correction;
            }
        }
        
        for (uint32 i = 0; i < dim; ++i) {
            for (uint32 j = i + 1; j < dim; ++j) {
                float64 avg = (state.covariance(i, j) + state.covariance(j, i)) * 0.5;
                state.covariance(i, j) = avg;
                state.covariance(j, i) = avg;
            }
        }
    }
};

}

#endif
