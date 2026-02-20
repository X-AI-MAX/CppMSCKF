#ifndef MSCKF_CORE_MARGINALIZER_HPP
#define MSCKF_CORE_MARGINALIZER_HPP

#include "state.hpp"
#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"

namespace msckf {

class Marginalizer {
public:
    struct Config {
        uint32 maxCameraFrames;
        uint32 minCameraFrames;
        bool useFEJ;
        float64 priorNoise;
        
        Config() 
            : maxCameraFrames(30)
            , minCameraFrames(3)
            , useFEJ(true)
            , priorNoise(1e-6) {}
    };

private:
    Config config_;
    
    Quaterniond fejOrientation_[MAX_CAMERA_FRAMES];
    Vector3d fejPosition_[MAX_CAMERA_FRAMES];
    bool fejInitialized_[MAX_CAMERA_FRAMES];
    
    Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64> priorInformation_;
    Vector<MAX_STATE_DIM, float64> priorState_;
    bool hasPrior_;
    uint32 priorDim_;

public:
    Marginalizer() 
        : hasPrior_(false)
        , priorDim_(0) {
        for (uint32 i = 0; i < MAX_CAMERA_FRAMES; ++i) {
            fejInitialized_[i] = false;
        }
    }
    
    void init(const Config& config) {
        config_ = config;
        hasPrior_ = false;
        priorDim_ = 0;
    }
    
    void recordFEJPoint(MSCKFState& state, uint32 frameId) {
        if (!config_.useFEJ) return;
        
        if (frameId < state.numCameraStates && !fejInitialized_[frameId]) {
            fejOrientation_[frameId] = state.cameraStates[frameId].orientation;
            fejPosition_[frameId] = state.cameraStates[frameId].position;
            fejInitialized_[frameId] = true;
        }
    }
    
    void marginalizeOldestFrame(MSCKFState& state, Feature* features, uint32 numFeatures) {
        if (state.numCameraStates <= config_.minCameraFrames) {
            return;
        }
        
        uint32 removeIdx = 0;
        
        for (uint32 i = 0; i < numFeatures; ++i) {
            removeFeatureObservation(features[i], removeIdx);
        }
        
        applySchurComplement(state, removeIdx);
        
        removeCameraState(state, removeIdx);
        
        shiftFEJPoints(removeIdx);
    }
    
    void marginalizeOldestFrameWithPrior(MSCKFState& state, Feature* features, uint32 numFeatures) {
        if (state.numCameraStates <= config_.minCameraFrames) {
            return;
        }
        
        uint32 removeIdx = 0;
        
        for (uint32 i = 0; i < numFeatures; ++i) {
            removeFeatureObservation(features[i], removeIdx);
        }
        
        computePriorInformation(state, removeIdx);
        
        removeCameraState(state, removeIdx);
        
        shiftFEJPoints(removeIdx);
    }
    
    void applySchurComplement(MSCKFState& state, uint32 removeIdx) {
        uint32 stateDim = state.stateDim();
        uint32 removeStart = state.cameraStateIndex(removeIdx);
        uint32 removeEnd = removeStart + CAMERA_STATE_DIM;
        
        uint32 remainDim = stateDim - CAMERA_STATE_DIM;
        
        Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>& P = state.covariance;
        
        Matrix<6, 6, float64> P_marg;
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                P_marg(i, j) = P(removeStart + i, removeStart + j);
            }
        }
        
        Matrix<6, 6, float64> P_marg_inv = P_marg.inverse();
        
        Matrix<MAX_STATE_DIM, 6, float64> P_rm;
        Matrix<6, MAX_STATE_DIM, float64> P_mr;
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                P_rm(i, j) = P(i, removeStart + j);
                P_mr(j, i) = P(removeStart + j, i);
            }
        }
        
        Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64> P_new;
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                float64 schurCorrection = 0;
                for (uint32 k = 0; k < 6; ++k) {
                    for (uint32 l = 0; l < 6; ++l) {
                        schurCorrection += P_rm(i, k) * P_marg_inv(k, l) * P_mr(l, j);
                    }
                }
                P_new(i, j) = P(i, j) - schurCorrection;
            }
        }
        
        Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64> P_reduced;
        uint32 newI = 0;
        for (uint32 i = 0; i < stateDim; ++i) {
            if (i >= removeStart && i < removeEnd) continue;
            
            uint32 newJ = 0;
            for (uint32 j = 0; j < stateDim; ++j) {
                if (j >= removeStart && j < removeEnd) continue;
                
                P_reduced(newI, newJ) = P_new(i, j);
                newJ++;
            }
            newI++;
        }
        
        state.covariance = P_reduced;
        state.covarianceDim = remainDim;
        state.covariance.symmetrize();
    }
    
    void computePriorInformation(MSCKFState& state, uint32 removeIdx) {
        uint32 stateDim = state.stateDim();
        uint32 removeStart = state.cameraStateIndex(removeIdx);
        
        Matrix<6, 6, float64> P_marg;
        Matrix<MAX_STATE_DIM, 6, float64> P_rm;
        
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                P_marg(i, j) = state.covariance(removeStart + i, removeStart + j);
            }
        }
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                P_rm(i, j) = state.covariance(i, removeStart + j);
            }
        }
        
        Matrix<6, 6, float64> Lambda = P_marg.inverse();
        
        priorDim_ = stateDim - CAMERA_STATE_DIM;
        
        for (uint32 i = 0; i < priorDim_; ++i) {
            for (uint32 j = 0; j < priorDim_; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < 6; ++k) {
                    for (uint32 l = 0; l < 6; ++l) {
                        sum += P_rm(i, k) * Lambda(k, l) * P_rm(j, l);
                    }
                }
                priorInformation_(i, j) = sum;
            }
        }
        
        hasPrior_ = true;
    }
    
    void applyPrior(MSCKFState& state) {
        if (!hasPrior_) return;
        
        uint32 stateDim = state.stateDim();
        
        for (uint32 i = 0; i < stateDim && i < priorDim_; ++i) {
            for (uint32 j = 0; j < stateDim && j < priorDim_; ++j) {
                state.covariance(i, j) += priorInformation_(i, j) * config_.priorNoise;
            }
        }
        
        state.covariance.symmetrize();
    }
    
    void removeCameraState(MSCKFState& state, uint32 removeIdx) {
        if (removeIdx >= state.numCameraStates) return;
        
        for (uint32 i = removeIdx; i < state.numCameraStates - 1; ++i) {
            state.cameraStates[i] = state.cameraStates[i + 1];
            state.cameraStates[i].id = i;
        }
        
        state.numCameraStates--;
        state.covarianceDim = state.stateDim();
    }
    
    void addCameraState(MSCKFState& state, const CameraState& camState) {
        if (state.numCameraStates >= MAX_CAMERA_FRAMES) {
            return;
        }
        
        state.cameraStates[state.numCameraStates] = camState;
        state.cameraStates[state.numCameraStates].id = state.nextCameraId++;
        state.numCameraStates++;
        
        augmentCovariance(state);
        
        if (config_.useFEJ) {
            recordFEJPoint(state, state.numCameraStates - 1);
        }
    }
    
    void augmentCovariance(MSCKFState& state) {
        uint32 oldDim = state.covarianceDim;
        uint32 newDim = oldDim + CAMERA_STATE_DIM;
        
        Matrix3d R_GI = state.imuState.orientation.toRotationMatrix();
        Matrix3d R_IC = state.extrinsicState.getRotationMatrix();
        Vector3d p_IC = state.extrinsicState.p_CB;
        
        Matrix<6, 21, float64> J = Matrix<6, 21, float64>();
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                J(i, j) = R_GI(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                J(i, j + 3) = 0;
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                J(i + 3, j + 6) = R_GI(i, j);
            }
        }
        
        Matrix3d p_IC_hat = skewSymmetric(p_IC);
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                J(i, j + 6) = (R_GI * p_IC_hat)(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                J(i, IMU_STATE_DIM + j) = R_GI(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                J(i + 3, IMU_STATE_DIM + 3 + j) = R_GI(i, j);
            }
        }
        
        Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>& P = state.covariance;
        
        for (uint32 i = oldDim; i < newDim; ++i) {
            for (uint32 j = 0; j < oldDim; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < IMU_STATE_DIM + EXTRINSIC_STATE_DIM; ++k) {
                    sum += J(i - oldDim, k) * P(k, j);
                }
                P(i, j) = sum;
                P(j, i) = sum;
            }
        }
        
        for (uint32 i = oldDim; i < newDim; ++i) {
            for (uint32 j = oldDim; j < newDim; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < IMU_STATE_DIM + EXTRINSIC_STATE_DIM; ++k) {
                    for (uint32 l = 0; l < IMU_STATE_DIM + EXTRINSIC_STATE_DIM; ++l) {
                        sum += J(i - oldDim, k) * P(k, l) * J(j - oldDim, l);
                    }
                }
                P(i, j) = sum;
            }
        }
        
        state.covarianceDim = newDim;
        P.symmetrize();
    }
    
    const Quaterniond& getFEJOrientation(uint32 frameId) const {
        return fejOrientation_[frameId];
    }
    
    const Vector3d& getFEJPosition(uint32 frameId) const {
        return fejPosition_[frameId];
    }
    
    bool hasFEJPoint(uint32 frameId) const {
        return fejInitialized_[frameId];
    }
    
    bool hasPrior() const {
        return hasPrior_;
    }
    
    void clearPrior() {
        hasPrior_ = false;
        priorDim_ = 0;
    }

private:
    void removeFeatureObservation(Feature& feature, uint32 frameId) {
        uint32 writeIdx = 0;
        for (uint32 i = 0; i < feature.numObservations; ++i) {
            if (feature.observations[i].frameId != frameId) {
                feature.observations[writeIdx] = feature.observations[i];
                writeIdx++;
            }
        }
        feature.numObservations = writeIdx;
        
        if (feature.numObservations > 0) {
            feature.firstFrameId = feature.observations[0].frameId;
            feature.lastFrameId = feature.observations[feature.numObservations - 1].frameId;
        }
    }
    
    void shiftFEJPoints(uint32 removedIdx) {
        for (uint32 i = removedIdx; i < MAX_CAMERA_FRAMES - 1; ++i) {
            fejOrientation_[i] = fejOrientation_[i + 1];
            fejPosition_[i] = fejPosition_[i + 1];
            fejInitialized_[i] = fejInitialized_[i + 1];
        }
        fejInitialized_[MAX_CAMERA_FRAMES - 1] = false;
    }
};

}

#endif
