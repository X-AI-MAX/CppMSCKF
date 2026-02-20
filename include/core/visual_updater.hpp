#ifndef MSCKF_CORE_VISUAL_UPDATER_HPP
#define MSCKF_CORE_VISUAL_UPDATER_HPP

#include "state.hpp"
#include "marginalizer.hpp"
#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../math/lie.hpp"
#include "../hal/camera_types.hpp"
#include "../vision/image_processing.hpp"

namespace msckf {

constexpr uint32 MAX_OBS_DIM = MAX_FEATURES * 2 * MAX_CAMERA_FRAMES;
constexpr uint32 MAX_FEAT_DIM = MAX_FEATURES * 3;
constexpr uint32 TRIANGULATION_MAX_OBS = 2 * MAX_CAMERA_FRAMES;
constexpr uint32 CHOLESKY_MAX_DIM = MAX_OBS_DIM;

class VisualUpdater {
public:
    struct Config {
        float64 observationNoise;
        uint32 minTrackLength;
        float64 maxReprojectionError;
        float64 minTriangulationAngle;
        uint32 maxFeaturesPerUpdate;
        bool useSchurComplement;
        bool useNullspaceProjection;
        bool useFEJ;
        
        Config() 
            : observationNoise(1.0)
            , minTrackLength(3)
            , maxReprojectionError(3.0)
            , minTriangulationAngle(0.1)
            , maxFeaturesPerUpdate(100)
            , useSchurComplement(true)
            , useNullspaceProjection(true)
            , useFEJ(true) {}
    };

private:
    Config config_;
    CameraIntrinsics intrinsics_;
    CameraExtrinsics extrinsics_;
    ImageProcessor* imageProcessor_;
    const Marginalizer* marginalizer_;
    
    float64 H_x_data_[MAX_OBS_DIM * MAX_STATE_DIM];
    float64 H_f_data_[MAX_OBS_DIM * MAX_FEAT_DIM];
    float64 r_data_[MAX_OBS_DIM];
    float64 R_data_[MAX_OBS_DIM * MAX_OBS_DIM];
    
    float64 triangulation_A_[TRIANGULATION_MAX_OBS * 4];
    float64 schur_A_inv_[MAX_FEATURES * 9];
    float64 schur_H_x_tilde_[MAX_OBS_DIM * MAX_STATE_DIM];
    float64 schur_r_tilde_[MAX_OBS_DIM];
    float64 nullspace_H_proj_[MAX_OBS_DIM * MAX_STATE_DIM];
    float64 ekf_S_[MAX_OBS_DIM * MAX_OBS_DIM];
    float64 ekf_S_inv_[MAX_OBS_DIM * MAX_OBS_DIM];
    float64 ekf_K_[MAX_STATE_DIM * MAX_OBS_DIM];
    float64 ekf_I_KH_[MAX_STATE_DIM * MAX_STATE_DIM];
    float64 ekf_P_new_[MAX_STATE_DIM * MAX_STATE_DIM];
    float64 cholesky_L_[CHOLESKY_MAX_DIM * CHOLESKY_MAX_DIM];
    float64 cholesky_L_inv_[CHOLESKY_MAX_DIM * CHOLESKY_MAX_DIM];
    
    Matrix<MAX_STATE_DIM, NULLSPACE_DIM, float64> nullspaceBasis_;
    Matrix<NULLSPACE_DIM, NULLSPACE_DIM, float64> nullspaceProjection_;

public:
    VisualUpdater() 
        : imageProcessor_(nullptr)
        , marginalizer_(nullptr) {
        resetBuffers();
    }
    
    void init(const CameraIntrinsics& intrinsics, 
              const CameraExtrinsics& extrinsics,
              ImageProcessor* imgProcessor,
              const Marginalizer* marg = nullptr,
              const Config& config = Config()) {
        intrinsics_ = intrinsics;
        extrinsics_ = extrinsics;
        imageProcessor_ = imgProcessor;
        marginalizer_ = marg;
        config_ = config;
        
        resetBuffers();
        computeNullspaceBasis();
    }
    
    void setMarginalizer(const Marginalizer* marg) {
        marginalizer_ = marg;
    }
    
    void computeNullspaceBasis() {
        nullspaceBasis_ = Matrix<MAX_STATE_DIM, NULLSPACE_DIM, float64>();
        
        for (uint32 i = 0; i < 3; ++i) {
            nullspaceBasis_(i, i) = 1.0;
        }
        
        nullspaceProjection_ = Matrix<NULLSPACE_DIM, NULLSPACE_DIM, float64>();
        for (uint32 i = 0; i < NULLSPACE_DIM; ++i) {
            nullspaceProjection_(i, i) = 1.0;
        }
    }
    
    void updateNullspaceBasis(const MSCKFState& state) {
        uint32 stateDim = state.stateDim();
        
        nullspaceBasis_ = Matrix<MAX_STATE_DIM, NULLSPACE_DIM, float64>();
        
        for (uint32 i = 0; i < 3; ++i) {
            nullspaceBasis_(i, i) = 1.0;
        }
        
        Matrix3d R_GI = state.imuState.orientation.toRotationMatrix();
        Vector3d e3({0, 0, 1});
        Vector3d yawDir = R_GI * e3;
        
        for (uint32 i = 0; i < 3; ++i) {
            nullspaceBasis_(6 + i, 3) = yawDir[i];
        }
        
        Matrix<NULLSPACE_DIM, MAX_STATE_DIM, float64> Nt;
        for (uint32 i = 0; i < NULLSPACE_DIM; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                Nt(i, j) = nullspaceBasis_(j, i);
            }
        }
        
        for (uint32 i = 0; i < NULLSPACE_DIM; ++i) {
            for (uint32 j = 0; j < NULLSPACE_DIM; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < stateDim; ++k) {
                    sum += Nt(i, k) * nullspaceBasis_(k, j);
                }
                nullspaceProjection_(i, j) = sum;
            }
        }
        
        nullspaceProjection_ = nullspaceProjection_.inverse();
    }
    
    void update(MSCKFState& state, Feature* features, uint32 numFeatures) {
        Feature* validFeatures[MAX_FEATURES];
        uint32 numValid = 0;
        
        for (uint32 i = 0; i < numFeatures && numValid < config_.maxFeaturesPerUpdate; ++i) {
            if (features[i].numObservations >= config_.minTrackLength) {
                validFeatures[numValid++] = &features[i];
            }
        }
        
        if (numValid == 0) return;
        
        uint32 stateDim = state.stateDim();
        uint32 totalObsDim = 0;
        
        for (uint32 i = 0; i < numValid; ++i) {
            totalObsDim += 2 * (validFeatures[i]->numObservations - 1);
        }
        
        if (totalObsDim == 0 || totalObsDim > MAX_OBS_DIM) return;
        
        if (config_.useNullspaceProjection) {
            updateNullspaceBasis(state);
        }
        
        uint32 featureDim = 3 * numValid;
        if (featureDim > MAX_FEAT_DIM) return;
        
        for (uint32 i = 0; i < totalObsDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                H_x_data_[i * stateDim + j] = 0;
            }
            for (uint32 j = 0; j < featureDim; ++j) {
                H_f_data_[i * featureDim + j] = 0;
            }
            r_data_[i] = 0;
            R_data_[i * totalObsDim + i] = config_.observationNoise;
        }
        
        uint32 rowOffset = 0;
        uint32 featOffset = 0;
        
        for (uint32 i = 0; i < numValid; ++i) {
            Feature* feat = validFeatures[i];
            
            if (!triangulateFeature(state, feat)) {
                continue;
            }
            
            Matrix<2, 3, float64> H_fi[50];
            Matrix<2, 15, float64> H_imu[50];
            Matrix<2, 6, float64> H_cam[50];
            Matrix<2, 6, float64> H_ext[50];
            Vector2d ri[50];
            uint32 camIndices[50];
            uint32 numResiduals = 0;
            
            if (config_.useFEJ && marginalizer_) {
                computeFeatureJacobianFEJ(state, feat, *marginalizer_,
                                         H_fi, H_imu, H_cam, H_ext, ri, camIndices, numResiduals);
            } else {
                computeFeatureJacobian(state, feat, H_fi, H_imu, H_cam, H_ext, ri, camIndices, numResiduals);
            }
            
            if (numResiduals < 2) continue;
            
            for (uint32 j = 0; j < numResiduals; ++j) {
                uint32 camStartIdx = state.cameraStateIndex(camIndices[j]);
                
                for (uint32 k = 0; k < 2; ++k) {
                    for (uint32 l = 0; l < 15; ++l) {
                        H_x_data_[(rowOffset + j * 2 + k) * stateDim + l] = H_imu[j](k, l);
                    }
                    for (uint32 l = 0; l < 6; ++l) {
                        H_x_data_[(rowOffset + j * 2 + k) * stateDim + IMU_STATE_DIM + l] = H_ext[j](k, l);
                        H_x_data_[(rowOffset + j * 2 + k) * stateDim + camStartIdx + l] = H_cam[j](k, l);
                    }
                    for (uint32 l = 0; l < 3; ++l) {
                        H_f_data_[(rowOffset + j * 2 + k) * featureDim + featOffset + l] = H_fi[j](k, l);
                    }
                    r_data_[rowOffset + j * 2 + k] = ri[j][k];
                }
            }
            
            rowOffset += 2 * numResiduals;
            featOffset += 3;
        }
        
        if (rowOffset == 0) return;
        
        if (config_.useSchurComplement) {
            performSchurComplementUpdate(state, H_x_data_, H_f_data_, r_data_, 
                                         rowOffset, stateDim, featureDim);
        } else {
            performEKFUpdate(state, H_x_data_, r_data_, rowOffset, stateDim);
        }
    }
    
    bool triangulateFeature(const MSCKFState& state, Feature* feature) {
        if (feature->numObservations < 2) return false;
        
        uint32 numObs = feature->numObservations;
        if (numObs * 2 > TRIANGULATION_MAX_OBS) {
            numObs = TRIANGULATION_MAX_OBS / 2;
        }
        
        for (uint32 i = 0; i < numObs * 2 * 4; ++i) {
            triangulation_A_[i] = 0;
        }
        
        for (uint32 i = 0; i < numObs; ++i) {
            uint32 frameId = feature->observations[i].frameId;
            
            if (frameId >= state.numCameraStates) continue;
            
            const CameraState& camState = state.cameraStates[frameId];
            
            Matrix3d R_wc = camState.orientation.toRotationMatrix();
            Vector3d t_wc = camState.position;
            
            float64 u = feature->observations[i].u;
            float64 v = feature->observations[i].v;
            
            float64 x = (u - intrinsics_.cx) / intrinsics_.fx;
            float64 y = (v - intrinsics_.cy) / intrinsics_.fy;
            
            Vector3d f_norm({x, y, 1.0});
            f_norm = f_norm.normalized();
            
            Matrix3d R_cw = R_wc.transpose();
            Vector3d t_cw = -R_cw * t_wc;
            
            uint32 rowIdx = i * 2;
            
            triangulation_A_[rowIdx * 4 + 0] = f_norm[0] * R_cw(2, 0) - R_cw(0, 0);
            triangulation_A_[rowIdx * 4 + 1] = f_norm[0] * R_cw(2, 1) - R_cw(0, 1);
            triangulation_A_[rowIdx * 4 + 2] = f_norm[0] * R_cw(2, 2) - R_cw(0, 2);
            triangulation_A_[rowIdx * 4 + 3] = f_norm[0] * t_cw[2] - t_cw[0];
            
            triangulation_A_[(rowIdx + 1) * 4 + 0] = f_norm[1] * R_cw(2, 0) - R_cw(1, 0);
            triangulation_A_[(rowIdx + 1) * 4 + 1] = f_norm[1] * R_cw(2, 1) - R_cw(1, 1);
            triangulation_A_[(rowIdx + 1) * 4 + 2] = f_norm[1] * R_cw(2, 2) - R_cw(1, 2);
            triangulation_A_[(rowIdx + 1) * 4 + 3] = f_norm[1] * t_cw[2] - t_cw[1];
        }
        
        Matrix<4, 4, float64> AtA;
        for (uint32 i = 0; i < 4; ++i) {
            for (uint32 j = 0; j < 4; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < numObs * 2; ++k) {
                    sum += triangulation_A_[k * 4 + i] * triangulation_A_[k * 4 + j];
                }
                AtA(i, j) = sum;
            }
        }
        
        float64 eigenvalues[4];
        eigenvaluesQR(AtA, eigenvalues);
        
        float64 minEig = eigenvalues[0];
        for (uint32 i = 1; i < 4; ++i) {
            if (eigenvalues[i] < minEig) {
                minEig = eigenvalues[i];
            }
        }
        
        if (minEig > 0.1) {
            return false;
        }
        
        Matrix<4, 4, float64> AtA_inv = AtA.inverse();
        
        Vector4d X;
        for (uint32 i = 0; i < 4; ++i) {
            X[i] = AtA_inv(i, 3);
        }
        
        if (abs(X[3]) < 1e-6) {
            return false;
        }
        
        feature->position[0] = X[0] / X[3];
        feature->position[1] = X[1] / X[3];
        feature->position[2] = X[2] / X[3];
        
        if (feature->position[2] < 0.1) {
            return false;
        }
        
        feature->isTriangulated = true;
        return true;
    }
    
    void computeFeatureJacobian(const MSCKFState& state, Feature* feature,
                                Matrix<2, 3, float64>* H_f,
                                Matrix<2, 15, float64>* H_imu,
                                Matrix<2, 6, float64>* H_cam,
                                Matrix<2, 6, float64>* H_ext,
                                Vector2d* r,
                                uint32* camIndices,
                                uint32& numResiduals) {
        numResiduals = 0;
        
        Vector3d p_f = feature->position;
        uint32 anchorFrameId = feature->firstFrameId;
        
        for (uint32 i = 0; i < feature->numObservations; ++i) {
            uint32 frameId = feature->observations[i].frameId;
            
            if (frameId >= state.numCameraStates) continue;
            if (frameId == anchorFrameId) continue;
            
            const CameraState& camState = state.cameraStates[frameId];
            
            Matrix3d R_GI = state.imuState.orientation.toRotationMatrix();
            Matrix3d R_IC = state.extrinsicState.getRotationMatrix();
            Vector3d p_IC = state.extrinsicState.p_CB;
            
            Matrix3d R_GC = camState.orientation.toRotationMatrix();
            Vector3d p_GC = camState.position;
            
            Matrix3d R_CG = R_GC.transpose();
            Vector3d p_C = R_CG * (p_f - p_GC);
            
            if (p_C[2] < 0.1) continue;
            
            float64 invZ = 1.0 / p_C[2];
            float64 invZ2 = invZ * invZ;
            
            Matrix<2, 3, float64> J_proj;
            J_proj(0, 0) = intrinsics_.fx * invZ;
            J_proj(0, 1) = 0;
            J_proj(0, 2) = -intrinsics_.fx * p_C[0] * invZ2;
            J_proj(1, 0) = 0;
            J_proj(1, 1) = intrinsics_.fy * invZ;
            J_proj(1, 2) = -intrinsics_.fy * p_C[1] * invZ2;
            
            H_f[numResiduals] = J_proj * R_CG;
            
            Matrix3d p_f_hat = skewSymmetric(p_f);
            Matrix3d R_CG_p_f_hat = R_CG * p_f_hat;
            
            H_imu[numResiduals] = Matrix<2, 15, float64>();
            for (uint32 ri = 0; ri < 2; ++ri) {
                for (uint32 ci = 0; ci < 3; ++ci) {
                    H_imu[numResiduals](ri, ci) = -J_proj(ri, ci) * R_CG(ci, 0);
                    H_imu[numResiduals](ri, ci + 1) = -J_proj(ri, ci) * R_CG(ci, 1);
                    H_imu[numResiduals](ri, ci + 2) = -J_proj(ri, ci) * R_CG(ci, 2);
                }
                for (uint32 ci = 0; ci < 3; ++ci) {
                    H_imu[numResiduals](ri, 6 + ci) = J_proj(ri, 0) * R_CG_p_f_hat(0, ci) +
                                                       J_proj(ri, 1) * R_CG_p_f_hat(1, ci) +
                                                       J_proj(ri, 2) * R_CG_p_f_hat(2, ci);
                }
            }
            
            H_cam[numResiduals] = Matrix<2, 6, float64>();
            for (uint32 ri = 0; ri < 2; ++ri) {
                for (uint32 ci = 0; ci < 3; ++ci) {
                    H_cam[numResiduals](ri, ci) = -J_proj(ri, ci) * R_CG(ci, 0);
                    H_cam[numResiduals](ri, ci + 1) = -J_proj(ri, ci) * R_CG(ci, 1);
                    H_cam[numResiduals](ri, ci + 2) = -J_proj(ri, ci) * R_CG(ci, 2);
                }
                for (uint32 ci = 0; ci < 3; ++ci) {
                    H_cam[numResiduals](ri, 3 + ci) = J_proj(ri, 0) * R_CG_p_f_hat(0, ci) +
                                                       J_proj(ri, 1) * R_CG_p_f_hat(1, ci) +
                                                       J_proj(ri, 2) * R_CG_p_f_hat(2, ci);
                }
            }
            
            H_ext[numResiduals] = Matrix<2, 6, float64>();
            
            float64 u_proj = intrinsics_.fx * p_C[0] * invZ + intrinsics_.cx;
            float64 v_proj = intrinsics_.fy * p_C[1] * invZ + intrinsics_.cy;
            
            r[numResiduals][0] = feature->observations[i].u - u_proj;
            r[numResiduals][1] = feature->observations[i].v - v_proj;
            
            camIndices[numResiduals] = frameId;
            numResiduals++;
        }
    }
    
    void computeFeatureJacobianFEJ(const MSCKFState& state, Feature* feature,
                                   const Marginalizer& marg,
                                   Matrix<2, 3, float64>* H_f,
                                   Matrix<2, 15, float64>* H_imu,
                                   Matrix<2, 6, float64>* H_cam,
                                   Matrix<2, 6, float64>* H_ext,
                                   Vector2d* r,
                                   uint32* camIndices,
                                   uint32& numResiduals) {
        numResiduals = 0;
        
        Vector3d p_f = feature->position;
        uint32 anchorFrameId = feature->firstFrameId;
        
        Quaterniond q_wa0;
        Vector3d p_wa0;
        
        if (marg.hasFEJPoint(anchorFrameId)) {
            q_wa0 = marg.getFEJOrientation(anchorFrameId);
            p_wa0 = marg.getFEJPosition(anchorFrameId);
        } else if (anchorFrameId < state.numCameraStates) {
            q_wa0 = state.cameraStates[anchorFrameId].orientation;
            p_wa0 = state.cameraStates[anchorFrameId].position;
        } else {
            return;
        }
        
        Matrix3d R_wa0 = q_wa0.toRotationMatrix();
        
        for (uint32 i = 0; i < feature->numObservations; ++i) {
            uint32 frameId = feature->observations[i].frameId;
            
            if (frameId >= state.numCameraStates) continue;
            if (frameId == anchorFrameId) continue;
            
            const CameraState& camState = state.cameraStates[frameId];
            
            Matrix3d R_GC = camState.orientation.toRotationMatrix();
            Vector3d p_GC = camState.position;
            
            Matrix3d R_CG = R_GC.transpose();
            Vector3d p_C = R_CG * (p_f - p_GC);
            
            if (p_C[2] < 0.1) continue;
            
            float64 invZ = 1.0 / p_C[2];
            float64 invZ2 = invZ * invZ;
            
            Matrix<2, 3, float64> J_proj;
            J_proj(0, 0) = intrinsics_.fx * invZ;
            J_proj(0, 1) = 0;
            J_proj(0, 2) = -intrinsics_.fx * p_C[0] * invZ2;
            J_proj(1, 0) = 0;
            J_proj(1, 1) = intrinsics_.fy * invZ;
            J_proj(1, 2) = -intrinsics_.fy * p_C[1] * invZ2;
            
            H_f[numResiduals] = J_proj * R_CG;
            
            Matrix3d p_f_hat = skewSymmetric(p_f);
            Matrix3d R_CG_p_f_hat = R_CG * p_f_hat;
            
            H_imu[numResiduals] = Matrix<2, 15, float64>();
            for (uint32 ri = 0; ri < 2; ++ri) {
                for (uint32 ci = 0; ci < 3; ++ci) {
                    H_imu[numResiduals](ri, ci) = -J_proj(ri, ci) * R_CG(ci, 0);
                    H_imu[numResiduals](ri, ci + 1) = -J_proj(ri, ci) * R_CG(ci, 1);
                    H_imu[numResiduals](ri, ci + 2) = -J_proj(ri, ci) * R_CG(ci, 2);
                }
                for (uint32 ci = 0; ci < 3; ++ci) {
                    H_imu[numResiduals](ri, 6 + ci) = J_proj(ri, 0) * R_CG_p_f_hat(0, ci) +
                                                       J_proj(ri, 1) * R_CG_p_f_hat(1, ci) +
                                                       J_proj(ri, 2) * R_CG_p_f_hat(2, ci);
                }
            }
            
            H_cam[numResiduals] = Matrix<2, 6, float64>();
            for (uint32 ri = 0; ri < 2; ++ri) {
                for (uint32 ci = 0; ci < 3; ++ci) {
                    H_cam[numResiduals](ri, ci) = -J_proj(ri, ci) * R_CG(ci, 0);
                    H_cam[numResiduals](ri, ci + 1) = -J_proj(ri, ci) * R_CG(ci, 1);
                    H_cam[numResiduals](ri, ci + 2) = -J_proj(ri, ci) * R_CG(ci, 2);
                }
                for (uint32 ci = 0; ci < 3; ++ci) {
                    H_cam[numResiduals](ri, 3 + ci) = J_proj(ri, 0) * R_CG_p_f_hat(0, ci) +
                                                       J_proj(ri, 1) * R_CG_p_f_hat(1, ci) +
                                                       J_proj(ri, 2) * R_CG_p_f_hat(2, ci);
                }
            }
            
            H_ext[numResiduals] = Matrix<2, 6, float64>();
            
            float64 u_proj = intrinsics_.fx * p_C[0] * invZ + intrinsics_.cx;
            float64 v_proj = intrinsics_.fy * p_C[1] * invZ + intrinsics_.cy;
            
            r[numResiduals][0] = feature->observations[i].u - u_proj;
            r[numResiduals][1] = feature->observations[i].v - v_proj;
            
            camIndices[numResiduals] = frameId;
            numResiduals++;
        }
    }
    
    void performSchurComplementUpdate(MSCKFState& state, 
                                      float64* H_x, float64* H_f,
                                      float64* r, uint32 obsDim, 
                                      uint32 stateDim, uint32 featDim) {
        uint32 numFeatures = featDim / 3;
        float64 invObsNoise = 1.0 / config_.observationNoise;
        
        for (uint32 f = 0; f < numFeatures; ++f) {
            uint32 featStart = f * 3;
            
            Matrix<3, 3, float64> A_f;
            for (uint32 i = 0; i < 3; ++i) {
                for (uint32 j = 0; j < 3; ++j) {
                    A_f(i, j) = 0;
                }
            }
            
            for (uint32 obs = 0; obs < obsDim; ++obs) {
                for (uint32 i = 0; i < 3; ++i) {
                    for (uint32 j = 0; j < 3; ++j) {
                        A_f(i, j) += H_f[obs * featDim + featStart + i] * 
                                     invObsNoise * 
                                     H_f[obs * featDim + featStart + j];
                    }
                }
            }
            
            Matrix<3, 3, float64> A_inv = A_f.inverse();
            
            for (uint32 i = 0; i < 3; ++i) {
                for (uint32 j = 0; j < 3; ++j) {
                    schur_A_inv_[f * 9 + i * 3 + j] = A_inv(i, j);
                }
            }
        }
        
        for (uint32 i = 0; i < obsDim; ++i) {
            schur_r_tilde_[i] = r[i];
            for (uint32 j = 0; j < stateDim; ++j) {
                schur_H_x_tilde_[i * stateDim + j] = H_x[i * stateDim + j];
            }
        }
        
        for (uint32 f = 0; f < numFeatures; ++f) {
            uint32 featStart = f * 3;
            
            for (uint32 obs = 0; obs < obsDim; ++obs) {
                float64 Hf_row[3];
                for (uint32 i = 0; i < 3; ++i) {
                    Hf_row[i] = H_f[obs * featDim + featStart + i];
                }
                
                float64 HfAinv[3];
                for (uint32 i = 0; i < 3; ++i) {
                    HfAinv[i] = 0;
                    for (uint32 j = 0; j < 3; ++j) {
                        HfAinv[i] += Hf_row[j] * schur_A_inv_[f * 9 + j * 3 + i];
                    }
                }
                
                for (uint32 j = 0; j < stateDim; ++j) {
                    float64 correction = 0;
                    for (uint32 k = 0; k < 3; ++k) {
                        correction += HfAinv[k] * H_f[obs * featDim + featStart + k] * 
                                     invObsNoise * H_x[obs * stateDim + j];
                    }
                    schur_H_x_tilde_[obs * stateDim + j] -= correction;
                }
                
                float64 r_correction = 0;
                for (uint32 k = 0; k < 3; ++k) {
                    r_correction += HfAinv[k] * H_f[obs * featDim + featStart + k] * 
                                   invObsNoise * r[obs];
                }
                schur_r_tilde_[obs] -= r_correction;
            }
        }
        
        if (config_.useNullspaceProjection) {
            projectToNullspace(schur_H_x_tilde_, schur_r_tilde_, obsDim, stateDim);
        }
        
        performEKFUpdate(state, schur_H_x_tilde_, schur_r_tilde_, obsDim, stateDim);
    }
    
    void projectToNullspace(float64* H_x, float64* r, uint32 obsDim, uint32 stateDim) {
        Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64> I_NN;
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                float64 sum = (i == j) ? 1.0 : 0.0;
                for (uint32 k = 0; k < NULLSPACE_DIM; ++k) {
                    for (uint32 l = 0; l < NULLSPACE_DIM; ++l) {
                        sum -= nullspaceBasis_(i, k) * nullspaceProjection_(k, l) * nullspaceBasis_(j, l);
                    }
                }
                I_NN(i, j) = sum;
            }
        }
        
        for (uint32 i = 0; i < obsDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < stateDim; ++k) {
                    sum += H_x[i * stateDim + k] * I_NN(k, j);
                }
                nullspace_H_proj_[i * stateDim + j] = sum;
            }
        }
        
        for (uint32 i = 0; i < obsDim * stateDim; ++i) {
            H_x[i] = nullspace_H_proj_[i];
        }
    }
    
    void performEKFUpdate(MSCKFState& state, float64* H_x, float64* r,
                          uint32 obsDim, uint32 stateDim) {
        Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>& P = state.covariance;
        
        for (uint32 i = 0; i < obsDim; ++i) {
            for (uint32 j = 0; j < obsDim; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < stateDim; ++k) {
                    for (uint32 l = 0; l < stateDim; ++l) {
                        sum += H_x[i * stateDim + k] * P(k, l) * H_x[j * stateDim + l];
                    }
                }
                ekf_S_[i * obsDim + j] = sum + (i == j ? config_.observationNoise : 0);
            }
        }
        
        invertMatrixCholesky(ekf_S_, ekf_S_inv_, obsDim);
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < obsDim; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < stateDim; ++k) {
                    for (uint32 l = 0; l < obsDim; ++l) {
                        sum += P(i, k) * H_x[l * stateDim + k] * ekf_S_inv_[l * obsDim + j];
                    }
                }
                ekf_K_[i * obsDim + j] = sum;
            }
        }
        
        ErrorState delta;
        for (uint32 i = 0; i < IMU_STATE_DIM; ++i) {
            float64 sum = 0;
            for (uint32 k = 0; k < obsDim; ++k) {
                sum += ekf_K_[i * obsDim + k] * r[k];
            }
            delta[i] = sum;
        }
        
        Vector3d delta_p_ext({0, 0, 0});
        Vector3d delta_theta_ext({0, 0, 0});
        for (uint32 i = 0; i < 3; ++i) {
            float64 sum_p = 0, sum_theta = 0;
            for (uint32 k = 0; k < obsDim; ++k) {
                sum_p += ekf_K_[(IMU_STATE_DIM + i) * obsDim + k] * r[k];
                sum_theta += ekf_K_[(IMU_STATE_DIM + 3 + i) * obsDim + k] * r[k];
            }
            delta_p_ext[i] = sum_p;
            delta_theta_ext[i] = sum_theta;
        }
        
        state.injectErrorState(delta);
        state.injectExtrinsicErrorState(delta_p_ext, delta_theta_ext);
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                float64 sum = (i == j) ? 1.0 : 0.0;
                for (uint32 k = 0; k < obsDim; ++k) {
                    sum -= ekf_K_[i * obsDim + k] * H_x[k * stateDim + j];
                }
                ekf_I_KH_[i * stateDim + j] = sum;
            }
        }
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < stateDim; ++k) {
                    sum += ekf_I_KH_[i * stateDim + k] * P(k, j);
                }
                ekf_P_new_[i * stateDim + j] = sum;
            }
        }
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                state.covariance(i, j) = ekf_P_new_[i * stateDim + j];
            }
        }
        state.covariance.symmetrize();
    }
    
    void invertMatrixCholesky(const float64* A, float64* A_inv, uint32 n) {
        for (uint32 i = 0; i < n * n; ++i) {
            cholesky_L_[i] = 0;
            cholesky_L_inv_[i] = 0;
        }
        
        for (uint32 i = 0; i < n; ++i) {
            for (uint32 j = 0; j <= i; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < j; ++k) {
                    sum += cholesky_L_[i * n + k] * cholesky_L_[j * n + k];
                }
                if (i == j) {
                    float64 diag = A[i * n + i] - sum;
                    cholesky_L_[i * n + j] = sqrt(max(diag, 1e-10));
                } else {
                    cholesky_L_[i * n + j] = (A[i * n + j] - sum) / cholesky_L_[j * n + j];
                }
            }
        }
        
        for (uint32 i = 0; i < n; ++i) {
            cholesky_L_inv_[i * n + i] = 1.0 / cholesky_L_[i * n + i];
            for (uint32 j = i + 1; j < n; ++j) {
                float64 sum = 0;
                for (uint32 k = i; k < j; ++k) {
                    sum -= cholesky_L_[j * n + k] * cholesky_L_inv_[k * n + i];
                }
                cholesky_L_inv_[j * n + i] = sum / cholesky_L_[j * n + j];
            }
        }
        
        for (uint32 i = 0; i < n; ++i) {
            for (uint32 j = 0; j < n; ++j) {
                float64 sum = 0;
                for (uint32 k = max(i, j); k < n; ++k) {
                    sum += cholesky_L_inv_[k * n + i] * cholesky_L_inv_[k * n + j];
                }
                A_inv[i * n + j] = sum;
            }
        }
    }

private:
    void resetBuffers() {
        for (uint32 i = 0; i < MAX_OBS_DIM * MAX_STATE_DIM; ++i) {
            H_x_data_[i] = 0;
        }
        for (uint32 i = 0; i < MAX_OBS_DIM * MAX_FEAT_DIM; ++i) {
            H_f_data_[i] = 0;
        }
        for (uint32 i = 0; i < MAX_OBS_DIM; ++i) {
            r_data_[i] = 0;
        }
        for (uint32 i = 0; i < MAX_OBS_DIM * MAX_OBS_DIM; ++i) {
            R_data_[i] = 0;
        }
    }
};

}

#endif
