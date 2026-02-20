#ifndef MSCKF_MSCKF_EXTENDED_HPP
#define MSCKF_MSCKF_EXTENDED_HPP

#include "msckf_estimator.hpp"
#include "collaborative/collaborative_msckf.hpp"
#include "semantic/semantic_msckf.hpp"
#include "core/incremental_schur.hpp"
#include "core/adaptive_kernel.hpp"
#include "core/lstm_predictor.hpp"
#include "utility/return_to_home.hpp"
#include "utility/battery_manager.hpp"
#include "utility/magnetic_rejection.hpp"
#include "utility/black_box.hpp"
#include "utility/auto_tuning.hpp"
#include "utility/zero_velocity_update.hpp"
#include "vision/multi_camera.hpp"

namespace msckf {

struct ExtendedConfig {
    MSCKFEstimator::Config msckfConfig;
    
    CollaborativeMSCKF::Config collaborativeConfig;
    SemanticMSCKF::Config semanticConfig;
    IncrementalSchurComplement::Config schurConfig;
    AdaptiveRobustKernel::Config kernelConfig;
    LSTMPredictor::Config lstmConfig;
    ReturnToHome::Config rthConfig;
    BatteryManager::Config batteryConfig;
    MagneticRejection::Config magneticConfig;
    BlackBox::Config blackboxConfig;
    AutoTuning::Config tuningConfig;
    ZeroVelocityUpdater::Config zuptConfig;
    ZeroVelocityDetector::Config zuptDetectorConfig;
    MultiCameraManager::Config multiCameraConfig;
    MultiCameraUpdater::Config multiCameraUpdaterConfig;
    
    bool enableCollaborative;
    bool enableSemantic;
    bool enableIncrementalSchur;
    bool enableAdaptiveKernel;
    bool enableLSTMPrediction;
    bool enableReturnToHome;
    bool enableBatteryManagement;
    bool enableMagneticRejection;
    bool enableBlackBox;
    bool enableAutoTuning;
    bool enableZeroVelocityUpdate;
    bool enableMultiCamera;
    bool enableNonHolonomicConstraint;
    
    ExtendedConfig() 
        : enableCollaborative(true)
        , enableSemantic(true)
        , enableIncrementalSchur(true)
        , enableAdaptiveKernel(true)
        , enableLSTMPrediction(true)
        , enableReturnToHome(true)
        , enableBatteryManagement(true)
        , enableMagneticRejection(true)
        , enableBlackBox(true)
        , enableAutoTuning(true)
        , enableZeroVelocityUpdate(true)
        , enableMultiCamera(false)
        , enableNonHolonomicConstraint(true) {}
};

class MSCKFExtended {
public:
    struct SystemStatus {
        uint64 timestamp;
        
        Vector3d position;
        Vector3d velocity;
        Quaterniond orientation;
        Vector3d gyroBias;
        Vector3d accelBias;
        
        Matrix<6, 6, float64> poseCovariance;
        
        uint32 numFeatures;
        uint32 numCameraFrames;
        uint32 numNeighbors;
        uint32 numSemanticFeatures;
        uint32 numStereoMatches;
        uint32 numCameras;
        
        float64 batteryPercentage;
        BatteryAlertLevel batteryAlert;
        
        MagneticState magneticState;
        float64 magneticWeight;
        
        ZUPTState zuptState;
        float64 zuptConfidence;
        uint32 zuptCount;
        
        bool isLocalized;
        bool isReturningHome;
        bool isLowPowerMode;
        bool isStationary;
        
        float64 adaptiveKernelThreshold;
        float64 predictionError;
        
        uint32 blackBoxEvents;
        float64 tuningScore;
    };

private:
    ExtendedConfig config_;
    MSCKFEstimator msckf_;
    
    CollaborativeMSCKF collaborative_;
    SemanticMSCKF semantic_;
    IncrementalSchurComplement incrementalSchur_;
    AdaptiveRobustKernel adaptiveKernel_;
    LSTMPredictor lstmPredictor_;
    IMUBiasPredictor biasPredictor_;
    ReturnToHome returnToHome_;
    BatteryManager batteryManager_;
    MagneticRejection magneticRejection_;
    BlackBox blackBox_;
    AutoTuning autoTuning_;
    ZeroVelocityUpdater zeroVelocityUpdater_;
    NonHolonomicConstraint nonHolonomicConstraint_;
    MultiCameraManager multiCameraManager_;
    MultiCameraUpdater multiCameraUpdater_;
    
    SystemStatus status_;
    
    bool initialized_;
    bool zeroVelocityDetected_;
    uint64 lastUpdateTime_;
    
    Vector3d homePosition_;
    bool homePositionSet_;
    
    uint8 droneId_;
    
    uint8* semanticImageBuffer_;
    uint32 semanticImageWidth_;
    uint32 semanticImageHeight_;

public:
    MSCKFExtended() 
        : initialized_(false)
        , zeroVelocityDetected_(false)
        , lastUpdateTime_(0)
        , homePositionSet_(false)
        , droneId_(0)
        , semanticImageBuffer_(nullptr)
        , semanticImageWidth_(0)
        , semanticImageHeight_(0) {
    }
    
    ~MSCKFExtended() {
        if (semanticImageBuffer_) delete[] semanticImageBuffer_;
    }
    
    void init(const ExtendedConfig& config, uint8 droneId = 0) {
        config_ = config;
        droneId_ = droneId;
        
        msckf_.init(config.msckfConfig);
        
        if (config_.enableCollaborative) {
            collaborative_.init(droneId, config.collaborativeConfig);
        }
        
        if (config_.enableSemantic) {
            semantic_.init(config.msckfConfig.cameraIntrinsics.width,
                          config.msckfConfig.cameraIntrinsics.height,
                          config.semanticConfig);
            
            semanticImageWidth_ = config.msckfConfig.cameraIntrinsics.width;
            semanticImageHeight_ = config.msckfConfig.cameraIntrinsics.height;
            semanticImageBuffer_ = new uint8[semanticImageWidth_ * semanticImageHeight_];
        }
        
        if (config_.enableIncrementalSchur) {
            incrementalSchur_.init(config.schurConfig);
        }
        
        if (config_.enableAdaptiveKernel) {
            adaptiveKernel_.init(config.kernelConfig);
        }
        
        if (config_.enableLSTMPrediction) {
            lstmPredictor_.init(config.lstmConfig);
            biasPredictor_.init();
        }
        
        if (config_.enableReturnToHome) {
            returnToHome_.init(config.rthConfig);
        }
        
        if (config_.enableBatteryManagement) {
            batteryManager_.init(config.batteryConfig);
        }
        
        if (config_.enableMagneticRejection) {
            magneticRejection_.init(config.magneticConfig);
        }
        
        if (config_.enableBlackBox) {
            blackBox_.init(config.blackboxConfig);
        }
        
        if (config_.enableAutoTuning) {
            autoTuning_.init(config.tuningConfig);
        }
        
        if (config_.enableZeroVelocityUpdate) {
            zeroVelocityUpdater_.init(config.zuptDetectorConfig, config.zuptConfig);
        }
        
        if (config_.enableNonHolonomicConstraint) {
            nonHolonomicConstraint_.init();
        }
        
        if (config_.enableMultiCamera) {
            multiCameraManager_.init(config.multiCameraConfig);
            multiCameraUpdater_.init(&multiCameraManager_, config.multiCameraUpdaterConfig);
        }
        
        initialized_ = true;
    }
    
    void processIMU(const IMUData& imuData) {
        msckf_.processImu(imuData);
        
        if (config_.enableLSTMPrediction) {
            lstmPredictor_.addImuData(imuData);
            biasPredictor_.update(imuData);
        }
        
        if (config_.enableBlackBox) {
            blackBox_.recordIMU(imuData);
        }
        
        if (config_.enableAutoTuning) {
            autoTuning_.addIMUSample(imuData);
        }
        
        if (config_.enableZeroVelocityUpdate) {
            zeroVelocityUpdater_.addIMUData(imuData);
        }
        
        lastUpdateTime_ = imuData.timestamp;
    }
    
    void processImage(const ImageData& imageData) {
        msckf_.processImage(imageData);
        
        if (config_.enableSemantic && semanticImageBuffer_) {
            processSemanticImage(imageData);
        }
        
        if (config_.enableReturnToHome && returnToHome_.isHomePositionSet()) {
            returnToHome_.update(msckf_.getState());
        }
    }
    
    void processMultiCameraImage(uint8 cameraId, const ImageData& imageData) {
        if (!config_.enableMultiCamera) return;
        
        multiCameraManager_.processFrame(cameraId, imageData.data, 
                                         imageData.width, imageData.height, 
                                         imageData.timestamp);
    }
    
    void addCamera(const CameraConfig& camConfig) {
        if (!config_.enableMultiCamera) return;
        
        multiCameraManager_.addCamera(camConfig);
    }
    
    void processSemanticImage(const ImageData& imageData) {
        convertToGrayscale(imageData.data, semanticImageBuffer_, 
                          imageData.width, imageData.height);
        
        semantic_.processImage(semanticImageBuffer_);
    }
    
    void convertToGrayscale(const uint8* input, uint8* output, 
                            uint32 width, uint32 height) {
        for (uint32 i = 0; i < width * height; ++i) {
            output[i] = input[i];
        }
    }
    
    void processNeighborData(const NeighborDrone& neighbor) {
        if (!config_.enableCollaborative) return;
        
        collaborative_.processNeighborData(neighbor);
    }
    
    void processMagneticField(const Vector3d& field, uint64 timestamp) {
        if (!config_.enableMagneticRejection) return;
        
        magneticRejection_.processMagneticField(field, timestamp);
    }
    
    void processBattery(const BatteryState& battery) {
        if (!config_.enableBatteryManagement) return;
        
        batteryManager_.update(battery);
        
        if (batteryManager_.isCritical()) {
            handleCriticalBattery();
        }
    }
    
    void update() {
        msckf_.update();
        
        updateStatus();
        
        if (config_.enableCollaborative) {
            collaborative_.updateCurrentPose(msckf_.getState());
            collaborative_.update();
        }
        
        if (config_.enableAdaptiveKernel) {
            adaptiveKernel_.updateThreshold();
        }
        
        if (config_.enableAutoTuning) {
            autoTuning_.update(lastUpdateTime_);
        }
        
        if (config_.enableBlackBox) {
            blackBox_.update(lastUpdateTime_);
        }
        
        if (config_.enableZeroVelocityUpdate) {
            MSCKFState& state = const_cast<MSCKFState&>(msckf_.getState());
            zeroVelocityUpdater_.applyZUPT(state);
            zeroVelocityUpdater_.update();
        }
        
        if (config_.enableNonHolonomicConstraint) {
            MSCKFState& state = const_cast<MSCKFState&>(msckf_.getState());
            nonHolonomicConstraint_.applyPlanarConstraint(state);
        }
        
        if (config_.enableMultiCamera) {
            multiCameraManager_.performCrossCameraTracking();
            MSCKFState& state = const_cast<MSCKFState&>(msckf_.getState());
            multiCameraUpdater_.update(state);
        }
    }
    
    void updateStatus() {
        const MSCKFState& state = msckf_.getState();
        
        status_.timestamp = state.timestamp;
        status_.position = state.imuState.position;
        status_.velocity = state.imuState.velocity;
        status_.orientation = state.imuState.orientation;
        status_.gyroBias = state.imuState.gyroBias;
        status_.accelBias = state.imuState.accelBias;
        
        status_.poseCovariance = msckf_.getPoseCovariance();
        
        status_.numFeatures = msckf_.getNumFeatures();
        status_.numCameraFrames = msckf_.getNumCameraFrames();
        
        if (config_.enableCollaborative) {
            status_.numNeighbors = collaborative_.getNumNeighbors();
        }
        
        if (config_.enableSemantic) {
            status_.numSemanticFeatures = semantic_.getNumSemanticFeatures();
        }
        
        if (config_.enableBatteryManagement) {
            status_.batteryPercentage = batteryManager_.getPercentage();
            status_.batteryAlert = batteryManager_.getAlertLevel();
            status_.isLowPowerMode = batteryManager_.isLowPowerMode();
        }
        
        if (config_.enableMagneticRejection) {
            status_.magneticState = magneticRejection_.getCurrentState();
            status_.magneticWeight = magneticRejection_.getMagneticWeight();
        }
        
        if (config_.enableReturnToHome) {
            status_.isReturningHome = returnToHome_.isNavigating() || 
                                      returnToHome_.isDescending();
        }
        
        if (config_.enableAdaptiveKernel) {
            status_.adaptiveKernelThreshold = adaptiveKernel_.getThreshold();
        }
        
        if (config_.enableLSTMPrediction) {
            status_.predictionError = lstmPredictor_.getAveragePredictionError();
        }
        
        if (config_.enableBlackBox) {
            status_.blackBoxEvents = blackBox_.getNumEvents();
        }
        
        if (config_.enableAutoTuning) {
            status_.tuningScore = autoTuning_.getBestScore();
        }
        
        if (config_.enableZeroVelocityUpdate) {
            status_.zuptState = zeroVelocityUpdater_.getCurrentState();
            status_.zuptConfidence = zeroVelocityUpdater_.getCurrentConfidence();
            status_.zuptCount = zeroVelocityUpdater_.getAppliedZUPTs();
            status_.isStationary = (status_.zuptState == ZUPTState::ZUPT_ACTIVE);
        }
        
        if (config_.enableMultiCamera) {
            status_.numCameras = multiCameraManager_.getNumCameras();
            status_.numStereoMatches = multiCameraManager_.getNumStereoMatches();
        }
        
        status_.isLocalized = !msckf_.isImuOnlyMode();
    }
    
    void handleCriticalBattery() {
        if (config_.enableReturnToHome && homePositionSet_) {
            returnToHome_.startReturnToHome();
        }
    }
    
    void setHomePosition() {
        homePosition_ = msckf_.getState().imuState.position;
        homePositionSet_ = true;
        
        if (config_.enableReturnToHome) {
            returnToHome_.setHomePosition(
                msckf_.getState().imuState.position,
                msckf_.getState().imuState.orientation
            );
        }
    }
    
    void startReturnToHome() {
        if (!config_.enableReturnToHome || !homePositionSet_) return;
        
        returnToHome_.startReturnToHome();
    }
    
    void cancelReturnToHome() {
        if (!config_.enableReturnToHome) return;
        
        returnToHome_.cancel();
    }
    
    void markBlackBoxEvent(const char* description) {
        if (!config_.enableBlackBox) return;
        
        blackBox_.triggerEvent(BlackBoxTrigger::MANUAL_MARK, 0, description);
    }
    
    void startIMUCalibration() {
        if (!config_.enableAutoTuning) return;
        
        autoTuning_.startIMUCalibration();
    }
    
    void stopIMUCalibration() {
        if (!config_.enableAutoTuning) return;
        
        autoTuning_.stopIMUCalibration();
    }
    
    void applyLearnedBiasPrediction() {
        if (!config_.enableLSTMPrediction) return;
        
        Vector3d predictedAccelBias = biasPredictor_.getPredictedAccelBias();
        Vector3d predictedGyroBias = biasPredictor_.getPredictedGyroBias();
        
        MSCKFState& state = const_cast<MSCKFState&>(msckf_.getState());
        
        float64 blendFactor = 0.1;
        state.imuState.accelBias = state.imuState.accelBias * (1 - blendFactor) + 
                                   predictedAccelBias * blendFactor;
        state.imuState.gyroBias = state.imuState.gyroBias * (1 - blendFactor) + 
                                  predictedGyroBias * blendFactor;
    }
    
    void filterDynamicFeatures() {
        if (!config_.enableSemantic) return;
        
        Feature* features = getFeaturesPtr();
        uint32 numFeatures = msckf_.getNumFeatures();
        
        semantic_.filterDynamicFeatures(features, numFeatures);
    }
    
    Feature* getFeaturesPtr() {
        return nullptr;
    }
    
    uint8* prepareBroadcastData(uint32& size) {
        if (!config_.enableCollaborative) {
            size = 0;
            return nullptr;
        }
        
        static uint8 broadcastBuffer_[1024];
        size = collaborative_.serializeBroadcastData(broadcastBuffer_, sizeof(broadcastBuffer_));
        
        return broadcastBuffer_;
    }
    
    bool processReceivedData(const uint8* data, uint32 size) {
        if (!config_.enableCollaborative) return false;
        
        NeighborDrone neighbor;
        if (collaborative_.deserializeNeighborData(data, size, neighbor)) {
            collaborative_.processNeighborData(neighbor);
            return true;
        }
        
        return false;
    }
    
    const SystemStatus& getStatus() const { return status_; }
    const MSCKFState& getState() const { return msckf_.getState(); }
    
    Vector3d getPosition() const { return msckf_.getPosition(); }
    Quaterniond getOrientation() const { return msckf_.getOrientation(); }
    Vector3d getVelocity() const { return msckf_.getVelocity(); }
    
    bool isInitialized() const { return initialized_; }
    bool isLocalized() const { return status_.isLocalized; }
    bool isReturningHome() const { return status_.isReturningHome; }
    bool isLowPowerMode() const { return status_.isLowPowerMode; }
    
    float64 getBatteryPercentage() const { return status_.batteryPercentage; }
    float64 getDistanceToHome() const { 
        return config_.enableReturnToHome ? returnToHome_.getDistanceToHome() : 0; 
    }
    
    Vector3d getHomePosition() const { return homePosition_; }
    bool isHomePositionSet() const { return homePositionSet_; }
    
    CollaborativeMSCKF& getCollaborative() { return collaborative_; }
    SemanticMSCKF& getSemantic() { return semantic_; }
    AdaptiveRobustKernel& getAdaptiveKernel() { return adaptiveKernel_; }
    LSTMPredictor& getLSTMPredictor() { return lstmPredictor_; }
    ReturnToHome& getReturnToHome() { return returnToHome_; }
    BatteryManager& getBatteryManager() { return batteryManager_; }
    MagneticRejection& getMagneticRejection() { return magneticRejection_; }
    BlackBox& getBlackBox() { return blackBox_; }
    AutoTuning& getAutoTuning() { return autoTuning_; }
    ZeroVelocityUpdater& getZeroVelocityUpdater() { return zeroVelocityUpdater_; }
    NonHolonomicConstraint& getNonHolonomicConstraint() { return nonHolonomicConstraint_; }
    MultiCameraManager& getMultiCameraManager() { return multiCameraManager_; }
    MultiCameraUpdater& getMultiCameraUpdater() { return multiCameraUpdater_; }
    
    void enableCollaborative(bool enable) { config_.enableCollaborative = enable; }
    void enableSemantic(bool enable) { config_.enableSemantic = enable; }
    void enableIncrementalSchur(bool enable) { config_.enableIncrementalSchur = enable; }
    void enableAdaptiveKernel(bool enable) { config_.enableAdaptiveKernel = enable; }
    void enableLSTMPrediction(bool enable) { config_.enableLSTMPrediction = enable; }
    void enableReturnToHome(bool enable) { config_.enableReturnToHome = enable; }
    void enableBatteryManagement(bool enable) { config_.enableBatteryManagement = enable; }
    void enableMagneticRejection(bool enable) { config_.enableMagneticRejection = enable; }
    void enableBlackBox(bool enable) { config_.enableBlackBox = enable; }
    void enableAutoTuning(bool enable) { config_.enableAutoTuning = enable; }
    void enableZeroVelocityUpdate(bool enable) { config_.enableZeroVelocityUpdate = enable; }
    void enableMultiCamera(bool enable) { config_.enableMultiCamera = enable; }
    void enableNonHolonomicConstraint(bool enable) { config_.enableNonHolonomicConstraint = enable; }
    
    bool isStationary() const { return status_.isStationary; }
    ZUPTState getZUPTState() const { return status_.zuptState; }
    uint32 getNumCameras() const { return status_.numCameras; }
    uint32 getNumStereoMatches() const { return status_.numStereoMatches; }
    
    void reset() {
        msckf_ = MSCKFEstimator();
        
        if (config_.enableCollaborative) collaborative_ = CollaborativeMSCKF();
        if (config_.enableSemantic) semantic_ = SemanticMSCKF();
        if (config_.enableIncrementalSchur) incrementalSchur_.reset();
        if (config_.enableAdaptiveKernel) adaptiveKernel_.reset();
        if (config_.enableLSTMPrediction) lstmPredictor_ = LSTMPredictor();
        if (config_.enableReturnToHome) returnToHome_.reset();
        if (config_.enableMagneticRejection) magneticRejection_.reset();
        if (config_.enableBlackBox) blackBox_.clear();
        if (config_.enableAutoTuning) autoTuning_.reset();
        if (config_.enableZeroVelocityUpdate) zeroVelocityUpdater_.reset();
        if (config_.enableNonHolonomicConstraint) nonHolonomicConstraint_.reset();
        if (config_.enableMultiCamera) multiCameraManager_.reset();
        
        initialized_ = false;
        homePositionSet_ = false;
    }
};

}

#endif
