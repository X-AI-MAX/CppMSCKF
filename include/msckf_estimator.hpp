#ifndef MSCKF_MSCKF_ESTIMATOR_HPP
#define MSCKF_MSCKF_ESTIMATOR_HPP

#include "math/types.hpp"
#include "math/matrix.hpp"
#include "math/quaternion.hpp"
#include "math/lie.hpp"
#include "hal/imu_hal.hpp"
#include "hal/camera_hal.hpp"
#include "vision/image_processing.hpp"
#include "vision/corner_detector.hpp"
#include "vision/klt_tracker.hpp"
#include "core/state.hpp"
#include "core/imu_propagator.hpp"
#include "core/visual_updater.hpp"
#include "core/marginalizer.hpp"
#include "system/memory_pool.hpp"
#include "system/lockfree_queue.hpp"
#include "system/scheduler.hpp"

namespace msckf {

class MSCKFEstimator {
public:
    struct Config {
        IMUNoiseParams imuNoiseParams;
        CameraIntrinsics cameraIntrinsics;
        CameraExtrinsics cameraExtrinsics;
        
        uint32 maxCameraFrames;
        uint32 maxFeatures;
        uint32 minTrackLength;
        float64 observationNoise;
        float64 maxReprojectionError;
        
        bool useAdaptiveNoise;
        bool useFEJ;
        bool useRollingShutterCorrection;
        bool estimateExtrinsics;
        float64 extrinsicConvergenceThreshold;
        uint64 extrinsicConvergenceTime;
        
        uint32 visualUpdateRate;
        uint32 imuPropagationRate;
        
        float64 abnormalAccelThreshold;
        float64 abnormalGyroThreshold;
        uint32 abnormalCountThreshold;
        uint64 visualLossTimeThreshold;
        
        Config() {
            maxCameraFrames = 30;
            maxFeatures = 300;
            minTrackLength = 3;
            observationNoise = 1.0;
            maxReprojectionError = 3.0;
            useAdaptiveNoise = true;
            useFEJ = true;
            useRollingShutterCorrection = true;
            estimateExtrinsics = true;
            extrinsicConvergenceThreshold = 0.001;
            extrinsicConvergenceTime = 30000000;
            visualUpdateRate = 30;
            imuPropagationRate = 500;
            abnormalAccelThreshold = 4.0 * GRAVITY;
            abnormalGyroThreshold = 10.0;
            abnormalCountThreshold = 10;
            visualLossTimeThreshold = 5000000;
        }
    };

private:
    Config config_;
    
    MSCKFState state_;
    IMUPropagator imuPropagator_;
    VisualUpdater visualUpdater_;
    Marginalizer marginalizer_;
    
    ImageProcessor imageProcessor_;
    CornerDetector cornerDetector_;
    KLTTracker kltTracker_;
    
    Feature features_[500];
    uint32 numFeatures_;
    uint32 nextFeatureId_;
    
    Feature2D prevFeatures_[300];
    Feature2D currFeatures_[300];
    uint32 numPrevFeatures_;
    uint32 numCurrFeatures_;
    
    uint8* grayImage_;
    uint8* undistortedImage_;
    
    SPSCQueue<IMUData, 1000> imuQueue_;
    SPSCQueue<ImageData, 10> imageQueue_;
    
    bool initialized_;
    uint64 lastVisualUpdateTime_;
    uint32 consecutiveImuOnlyCount_;
    uint64 lastVisualFeatureTime_;
    bool imuOnlyMode_;
    
    uint32 consecutiveAbnormalCount_;
    uint64 abnormalStartTime_;
    float64 abnormalScaleFactor_;
    
    float64 prevExtrinsicNorm_;
    uint32 extrinsicConvergenceCount_;
    
    uint8 imageBuffer_[640 * 480 * 2];
    uint8 grayBuffer_[640 * 480];

public:
    MSCKFEstimator() 
        : numFeatures_(0)
        , nextFeatureId_(0)
        , numPrevFeatures_(0)
        , numCurrFeatures_(0)
        , grayImage_(nullptr)
        , undistortedImage_(nullptr)
        , initialized_(false)
        , lastVisualUpdateTime_(0)
        , consecutiveImuOnlyCount_(0)
        , lastVisualFeatureTime_(0)
        , imuOnlyMode_(false)
        , consecutiveAbnormalCount_(0)
        , abnormalStartTime_(0)
        , abnormalScaleFactor_(1.0)
        , prevExtrinsicNorm_(0)
        , extrinsicConvergenceCount_(0) {
    }
    
    ~MSCKFEstimator() {
        if (grayImage_) delete[] grayImage_;
        if (undistortedImage_) delete[] undistortedImage_;
    }
    
    void init(const Config& config) {
        config_ = config;
        
        imuPropagator_.init(config.imuNoiseParams);
        
        visualUpdater_.init(config.cameraIntrinsics, config.cameraExtrinsics, &imageProcessor_);
        
        Marginalizer::Config margConfig;
        margConfig.maxCameraFrames = config.maxCameraFrames;
        margConfig.useFEJ = config.useFEJ;
        marginalizer_.init(margConfig);
        
        visualUpdater_.setMarginalizer(&marginalizer_);
        
        imageProcessor_.init(config.cameraIntrinsics);
        
        CornerDetector::Config cornerConfig;
        cornerConfig.maxFeatures = config.maxFeatures;
        cornerDetector_.init(config.cameraIntrinsics.width, config.cameraIntrinsics.height, cornerConfig);
        
        KLTTracker::Config kltConfig;
        kltTracker_.init(config.cameraIntrinsics.width, config.cameraIntrinsics.height, kltConfig);
        
        uint32 imageSize = config.cameraIntrinsics.width * config.cameraIntrinsics.height;
        grayImage_ = new uint8[imageSize];
        undistortedImage_ = new uint8[imageSize];
        
        state_.extrinsicState.p_CB = config.cameraExtrinsics.t_cam_to_body;
        state_.extrinsicState.theta_CB = SO3d(config.cameraExtrinsics.q_cam_to_body).log();
        state_.q_CB = config.cameraExtrinsics.q_cam_to_body;
        state_.p_C_B = config.cameraExtrinsics.t_cam_to_body;
        state_.extrinsicOnline = config.estimateExtrinsics;
        state_.extrinsicConverged = false;
        
        prevExtrinsicNorm_ = state_.extrinsicState.p_CB.norm() + state_.extrinsicState.theta_CB.norm();
        
        initialized_ = true;
    }
    
    void processImu(const IMUData& imuData) {
        if (!initialized_) return;
        
        imuQueue_.push(imuData);
        
        imuPropagator_.addImuData(imuData);
    }
    
    void processImage(const ImageData& imageData) {
        if (!initialized_) return;
        
        imageQueue_.push(imageData);
    }
    
    void update() {
        processImuQueue();
        
        processImageQueue();
        
        checkVisualLoss();
    }
    
    void processImuQueue() {
        IMUData imuData;
        while (imuQueue_.pop(imuData)) {
            imuPropagator_.propagateTo(imuData.timestamp, state_);
            
            checkImuValidity(imuData);
        }
    }
    
    void processImageQueue() {
        ImageData imageData;
        while (imageQueue_.pop(imageData)) {
            processOneFrame(imageData);
        }
    }
    
    void processOneFrame(const ImageData& imageData) {
        imageProcessor_.bayerToGray(imageData.data, grayImage_, 
                                    imageData.width, imageData.height,
                                    ImageProcessor::BayerPattern::RGGB);
        
        imageProcessor_.undistortImage(grayImage_, undistortedImage_);
        
        trackFeatures();
        
        detectNewFeatures();
        
        if (shouldPerformVisualUpdate(imageData.timestamp)) {
            performVisualUpdate();
            lastVisualUpdateTime_ = imageData.timestamp;
        }
        
        addNewCameraFrame(imageData.timestamp);
        
        if (state_.numCameraStates >= config_.maxCameraFrames) {
            marginalizeOldestFrame();
        }
        
        checkExtrinsicConvergence(imageData.timestamp);
    }
    
    void trackFeatures() {
        if (numPrevFeatures_ == 0) {
            return;
        }
        
        kltTracker_.setImages(grayImage_, grayImage_);
        
        TrackResult results[300];
        kltTracker_.trackWithValidation(prevFeatures_, numPrevFeatures_, results);
        
        numCurrFeatures_ = 0;
        for (uint32 i = 0; i < numPrevFeatures_; ++i) {
            if (results[i].success) {
                currFeatures_[numCurrFeatures_] = results[i].currFeature;
                
                updateFeatureTrack(prevFeatures_[i].id, results[i].currFeature.x, 
                                  results[i].currFeature.y);
                
                numCurrFeatures_++;
            }
        }
        
        if (numCurrFeatures_ > 0) {
            lastVisualFeatureTime_ = state_.timestamp;
            if (imuOnlyMode_) {
                recoverFromImuOnlyMode();
            }
        }
    }
    
    void detectNewFeatures() {
        uint32 maxNewFeatures = config_.maxFeatures - numCurrFeatures_;
        if (maxNewFeatures == 0) return;
        
        cornerDetector_.createMask(currFeatures_, numCurrFeatures_);
        
        Feature2D newFeatures[200];
        uint32 numNew = 200;
        cornerDetector_.detectWithGrid(undistortedImage_, newFeatures, numNew);
        
        for (uint32 i = 0; i < numNew && numCurrFeatures_ < config_.maxFeatures; ++i) {
            currFeatures_[numCurrFeatures_] = newFeatures[i];
            currFeatures_[numCurrFeatures_].id = cornerDetector_.getNextFeatureId();
            
            addNewFeature(newFeatures[i].x, newFeatures[i].y, 
                         state_.numCameraStates, currFeatures_[numCurrFeatures_].id);
            
            numCurrFeatures_++;
        }
        
        for (uint32 i = 0; i < numCurrFeatures_; ++i) {
            prevFeatures_[i] = currFeatures_[i];
        }
        numPrevFeatures_ = numCurrFeatures_;
    }
    
    void performVisualUpdate() {
        Feature* validFeatures[300];
        uint32 numValid = 0;
        
        for (uint32 i = 0; i < numFeatures_; ++i) {
            if (features_[i].numObservations >= config_.minTrackLength) {
                validFeatures[numValid++] = &features_[i];
            }
        }
        
        if (numValid == 0) return;
        
        visualUpdater_.update(state_, features_, numFeatures_);
        
        removeLostFeatures();
    }
    
    void addNewCameraFrame(uint64 timestamp) {
        CameraState camState;
        
        Quaterniond q_IC = state_.extrinsicState.getQuaternion();
        Vector3d p_IC = state_.extrinsicState.p_CB;
        
        camState.orientation = state_.imuState.orientation * q_IC;
        camState.position = state_.imuState.position + 
                           state_.imuState.orientation.toRotationMatrix() * p_IC;
        camState.timestamp = timestamp;
        
        marginalizer_.addCameraState(state_, camState);
    }
    
    void marginalizeOldestFrame() {
        marginalizer_.marginalizeOldestFrame(state_, features_, numFeatures_);
        
        for (uint32 i = 0; i < numFeatures_; ++i) {
            for (uint32 j = 0; j < features_[i].numObservations; ++j) {
                if (features_[i].observations[j].frameId > 0) {
                    features_[i].observations[j].frameId--;
                }
            }
            if (features_[i].firstFrameId > 0) {
                features_[i].firstFrameId--;
            }
            if (features_[i].lastFrameId > 0) {
                features_[i].lastFrameId--;
            }
        }
    }
    
    void addNewFeature(float64 u, float64 v, uint32 frameId, uint32 featureId) {
        if (numFeatures_ >= 500) return;
        
        features_[numFeatures_].id = featureId;
        features_[numFeatures_].numObservations = 0;
        features_[numFeatures_].isTriangulated = false;
        features_[numFeatures_].trackCount = 0;
        features_[numFeatures_].addObservation(frameId, u, v);
        
        numFeatures_++;
    }
    
    void updateFeatureTrack(uint32 featureId, float64 u, float64 v) {
        for (uint32 i = 0; i < numFeatures_; ++i) {
            if (features_[i].id == featureId) {
                features_[i].addObservation(state_.numCameraStates, u, v);
                features_[i].trackCount++;
                break;
            }
        }
    }
    
    void removeLostFeatures() {
        uint32 writeIdx = 0;
        for (uint32 i = 0; i < numFeatures_; ++i) {
            bool hasCurrentObs = features_[i].hasObservation(state_.numCameraStates - 1);
            
            if (hasCurrentObs || features_[i].numObservations < config_.minTrackLength) {
                if (writeIdx != i) {
                    features_[writeIdx] = features_[i];
                }
                writeIdx++;
            }
        }
        numFeatures_ = writeIdx;
    }
    
    bool shouldPerformVisualUpdate(uint64 timestamp) {
        if (lastVisualUpdateTime_ == 0) return true;
        
        uint64 dt = timestamp - lastVisualUpdateTime_;
        uint64 interval = 1000000 / config_.visualUpdateRate;
        
        return dt >= interval;
    }
    
    void checkImuValidity(const IMUData& imuData) {
        float64 accelNorm = imuData.accel.norm();
        float64 gyroNorm = imuData.gyro.norm();
        
        bool isAbnormal = (accelNorm > config_.abnormalAccelThreshold || 
                          gyroNorm > config_.abnormalGyroThreshold);
        
        if (isAbnormal) {
            consecutiveAbnormalCount_++;
            
            if (consecutiveAbnormalCount_ == 1) {
                abnormalStartTime_ = imuData.timestamp;
            }
            
            abnormalScaleFactor_ = min(abnormalScaleFactor_ * 10.0, 100.0);
            
            if (consecutiveAbnormalCount_ >= config_.abnormalCountThreshold) {
                switchToImuOnlyMode();
            }
        } else {
            consecutiveAbnormalCount_ = 0;
            abnormalStartTime_ = 0;
            abnormalScaleFactor_ = max(abnormalScaleFactor_ * 0.9, 1.0);
        }
    }
    
    void switchToImuOnlyMode() {
        if (imuOnlyMode_) return;
        
        imuOnlyMode_ = true;
        
        for (uint32 i = 0; i < IMU_STATE_DIM; ++i) {
            for (uint32 j = 0; j < IMU_STATE_DIM; ++j) {
                state_.covariance(i, j) *= 10.0;
            }
        }
        
        for (uint32 i = IMU_STATE_DIM; i < state_.covarianceDim; ++i) {
            for (uint32 j = IMU_STATE_DIM; j < state_.covarianceDim; ++j) {
                state_.covariance(i, j) *= 10.0;
            }
        }
        
        state_.covariance.symmetrize();
        
        for (uint32 i = 0; i < numFeatures_; ++i) {
            features_[i].numObservations = 0;
            features_[i].isTriangulated = false;
        }
        numFeatures_ = 0;
        
        numPrevFeatures_ = 0;
        numCurrFeatures_ = 0;
    }
    
    void recoverFromImuOnlyMode() {
        if (!imuOnlyMode_) return;
        
        imuOnlyMode_ = false;
        
        for (uint32 i = 0; i < IMU_STATE_DIM; ++i) {
            for (uint32 j = 0; j < IMU_STATE_DIM; ++j) {
                state_.covariance(i, j) *= 0.5;
            }
        }
        
        state_.covariance.symmetrize();
        
        consecutiveAbnormalCount_ = 0;
        abnormalScaleFactor_ = 1.0;
    }
    
    void checkVisualLoss() {
        if (lastVisualFeatureTime_ == 0 || imuOnlyMode_) return;
        
        uint64 timeSinceLastVisual = state_.timestamp - lastVisualFeatureTime_;
        
        if (timeSinceLastVisual > config_.visualLossTimeThreshold) {
            switchToImuOnlyMode();
        }
    }
    
    void checkExtrinsicConvergence(uint64 currentTime) {
        if (!state_.extrinsicOnline || state_.extrinsicConverged) return;
        
        float64 currentNorm = state_.extrinsicState.p_CB.norm() + 
                              state_.extrinsicState.theta_CB.norm();
        
        float64 delta = abs(currentNorm - prevExtrinsicNorm_);
        
        if (delta < config_.extrinsicConvergenceThreshold) {
            extrinsicConvergenceCount_++;
            
            if (extrinsicConvergenceCount_ >= 30) {
                state_.extrinsicConverged = true;
                state_.extrinsicConvergeTime = currentTime;
            }
        } else {
            extrinsicConvergenceCount_ = 0;
        }
        
        prevExtrinsicNorm_ = currentNorm;
    }
    
    const MSCKFState& getState() const {
        return state_;
    }
    
    Vector3d getPosition() const {
        return state_.imuState.position;
    }
    
    Quaterniond getOrientation() const {
        return state_.imuState.orientation;
    }
    
    Vector3d getVelocity() const {
        return state_.imuState.velocity;
    }
    
    ExtrinsicState getExtrinsicState() const {
        return state_.extrinsicState;
    }
    
    bool isExtrinsicConverged() const {
        return state_.extrinsicConverged;
    }
    
    bool isImuOnlyMode() const {
        return imuOnlyMode_;
    }
    
    Matrix<6, 6, float64> getPoseCovariance() const {
        Matrix<6, 6, float64> poseCov;
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                poseCov(i, j) = state_.covariance(i, j);
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                poseCov(i + 3, j + 3) = state_.covariance(i + 6, j + 6);
            }
        }
        
        return poseCov;
    }
    
    bool isInitialized() const {
        return initialized_;
    }
    
    uint32 getNumFeatures() const {
        return numFeatures_;
    }
    
    uint32 getNumCameraFrames() const {
        return state_.numCameraStates;
    }
};

}

#endif
