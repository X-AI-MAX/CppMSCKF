#ifndef MSCKF_VISION_MULTI_CAMERA_HPP
#define MSCKF_VISION_MULTI_CAMERA_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../math/lie.hpp"
#include "../core/state.hpp"
#include "../hal/camera_types.hpp"
#include "../vision/corner_detector.hpp"
#include "../vision/klt_tracker.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 MAX_CAMERAS = 4;
constexpr uint32 MAX_MULTI_CAM_FEATURES = 1000;
constexpr uint32 MAX_STEREO_MATCHES = 500;

enum class CameraRole : uint8 {
    PRIMARY = 0,
    SECONDARY = 1,
    STEREO_LEFT = 2,
    STEREO_RIGHT = 3,
    DOWNWARD = 4,
    FORWARD = 5
};

struct CameraConfig {
    uint8 cameraId;
    CameraRole role;
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
    bool isEnabled;
    bool isStereo;
    uint8 stereoPairId;
    float64 baseline;
    
    CameraConfig() 
        : cameraId(0)
        , role(CameraRole::PRIMARY)
        , isEnabled(true)
        , isStereo(false)
        , stereoPairId(255)
        , baseline(0) {}
};

struct MultiCameraFeature {
    uint32 featureId;
    uint32 cameraId;
    Vector2d uv;
    float64 invDepth;
    Vector3d position;
    uint32 trackCount;
    bool isTriangulated;
    bool isStereoMatched;
    uint32 stereoMatchCameraId;
    Vector2d stereoUV;
    
    MultiCameraFeature() 
        : featureId(0)
        , cameraId(0)
        , uv({0, 0})
        , invDepth(0)
        , position({0, 0, 0})
        , trackCount(0)
        , isTriangulated(false)
        , isStereoMatched(false)
        , stereoMatchCameraId(255) {}
};

struct StereoMatch {
    uint32 featureId;
    uint8 leftCameraId;
    uint8 rightCameraId;
    Vector2d leftUV;
    Vector2d rightUV;
    float64 disparity;
    float64 depth;
    Vector3d position;
    float64 confidence;
    
    StereoMatch() 
        : featureId(0)
        , leftCameraId(0)
        , rightCameraId(1)
        , disparity(0)
        , depth(0)
        , position({0, 0, 0})
        , confidence(0) {}
};

struct CameraFrame {
    uint8 cameraId;
    uint64 timestamp;
    uint8* imageData;
    uint32 width;
    uint32 height;
    Feature2D features[300];
    uint32 numFeatures;
    bool isValid;
    
    CameraFrame() 
        : cameraId(0)
        , timestamp(0)
        , imageData(nullptr)
        , width(0)
        , height(0)
        , numFeatures(0)
        , isValid(false) {}
};

class MultiCameraManager {
public:
    struct Config {
        uint32 maxCameras;
        bool enableStereoMatching;
        bool enableCrossCameraTracking;
        float64 stereoMinDisparity;
        float64 stereoMaxDisparity;
        float64 stereoBaselineThreshold;
        uint32 stereoBlockSize;
        float64 crossCameraMinOverlap;
        bool enableTemporalStereo;
        
        Config() 
            : maxCameras(MAX_CAMERAS)
            , enableStereoMatching(true)
            , enableCrossCameraTracking(true)
            , stereoMinDisparity(1.0)
            , stereoMaxDisparity(100.0)
            , stereoBaselineThreshold(0.05)
            , stereoBlockSize(15)
            , crossCameraMinOverlap(0.3)
            , enableTemporalStereo(true) {}
    };

private:
    Config config_;
    CameraConfig cameras_[MAX_CAMERAS];
    uint32 numCameras_;
    
    CornerDetector cornerDetectors_[MAX_CAMERAS];
    KLTTracker kltTrackers_[MAX_CAMERAS];
    
    CameraFrame frames_[MAX_CAMERAS];
    uint64 lastFrameTime_[MAX_CAMERAS];
    
    MultiCameraFeature features_[MAX_MULTI_CAM_FEATURES];
    uint32 numFeatures_;
    uint32 nextFeatureId_;
    
    StereoMatch stereoMatches_[MAX_STEREO_MATCHES];
    uint32 numStereoMatches_;
    
    uint8* grayBuffers_[MAX_CAMERAS];
    uint8* undistortBuffers_[MAX_CAMERAS];
    
    Matrix<6, 6, float64> extrinsicCovariance_[MAX_CAMERAS];
    bool extrinsicOnline_[MAX_CAMERAS];
    bool extrinsicConverged_[MAX_CAMERAS];
    
    uint32 totalFrames_[MAX_CAMERAS];
    uint32 totalFeatures_[MAX_CAMERAS];
    uint32 totalStereoMatches_;

public:
    MultiCameraManager() 
        : numCameras_(0)
        , numFeatures_(0)
        , nextFeatureId_(0)
        , numStereoMatches_(0)
        , totalStereoMatches_(0) {
        
        memset(grayBuffers_, 0, sizeof(grayBuffers_));
        memset(undistortBuffers_, 0, sizeof(undistortBuffers_));
        memset(lastFrameTime_, 0, sizeof(lastFrameTime_));
        memset(totalFrames_, 0, sizeof(totalFrames_));
        memset(totalFeatures_, 0, sizeof(totalFeatures_));
    }
    
    ~MultiCameraManager() {
        for (uint32 i = 0; i < MAX_CAMERAS; ++i) {
            if (grayBuffers_[i]) delete[] grayBuffers_[i];
            if (undistortBuffers_[i]) delete[] undistortBuffers_[i];
        }
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        numCameras_ = 0;
        numFeatures_ = 0;
        nextFeatureId_ = 0;
        numStereoMatches_ = 0;
    }
    
    bool addCamera(const CameraConfig& camConfig) {
        if (numCameras_ >= config_.maxCameras) return false;
        
        uint32 idx = numCameras_;
        cameras_[idx] = camConfig;
        cameras_[idx].cameraId = idx;
        
        CornerDetector::Config cornerConfig;
        cornerConfig.maxFeatures = 150;
        cornerDetectors_[idx].init(camConfig.intrinsics.width, 
                                   camConfig.intrinsics.height, 
                                   cornerConfig);
        
        KLTTracker::Config kltConfig;
        kltTrackers_[idx].init(camConfig.intrinsics.width,
                               camConfig.intrinsics.height,
                               kltConfig);
        
        uint32 imageSize = camConfig.intrinsics.width * camConfig.intrinsics.height;
        grayBuffers_[idx] = new uint8[imageSize];
        undistortBuffers_[idx] = new uint8[imageSize];
        
        extrinsicCovariance_[idx] = Matrix<6, 6, float64>::identity();
        extrinsicOnline_[idx] = false;
        extrinsicConverged_[idx] = false;
        
        numCameras_++;
        return true;
    }
    
    void processFrame(uint8 cameraId, const uint8* imageData, 
                      uint32 width, uint32 height, uint64 timestamp) {
        if (cameraId >= numCameras_) return;
        if (!cameras_[cameraId].isEnabled) return;
        
        memcpy(grayBuffers_[cameraId], imageData, width * height);
        
        frames_[cameraId].cameraId = cameraId;
        frames_[cameraId].timestamp = timestamp;
        frames_[cameraId].width = width;
        frames_[cameraId].height = height;
        frames_[cameraId].imageData = grayBuffers_[cameraId];
        frames_[cameraId].isValid = true;
        
        detectFeatures(cameraId);
        
        if (cameraId > 0 && lastFrameTime_[0] == timestamp) {
            performStereoMatching(0, cameraId);
        }
        
        lastFrameTime_[cameraId] = timestamp;
        totalFrames_[cameraId]++;
    }
    
    void detectFeatures(uint8 cameraId) {
        if (cameraId >= numCameras_) return;
        
        CameraFrame& frame = frames_[cameraId];
        frame.numFeatures = 200;
        
        cornerDetectors_[cameraId].detectWithGrid(
            grayBuffers_[cameraId], 
            frame.features, 
            frame.numFeatures
        );
        
        for (uint32 i = 0; i < frame.numFeatures; ++i) {
            frame.features[i].id = nextFeatureId_++;
            frame.features[i].timestamp = frame.timestamp;
        }
        
        totalFeatures_[cameraId] += frame.numFeatures;
    }
    
    void performStereoMatching(uint8 leftCamId, uint8 rightCamId) {
        if (!config_.enableStereoMatching) return;
        if (leftCamId >= numCameras_ || rightCamId >= numCameras_) return;
        
        CameraFrame& leftFrame = frames_[leftCamId];
        CameraFrame& rightFrame = frames_[rightCamId];
        
        if (!leftFrame.isValid || !rightFrame.isValid) return;
        
        numStereoMatches_ = 0;
        
        const CameraIntrinsics& leftIntrinsics = cameras_[leftCamId].intrinsics;
        const CameraIntrinsics& rightIntrinsics = cameras_[rightCamId].intrinsics;
        
        float64 baseline = cameras_[rightCamId].extrinsics.t_cam_to_body[0] - 
                          cameras_[leftCamId].extrinsics.t_cam_to_body[0];
        
        if (abs(baseline) < config_.stereoBaselineThreshold) return;
        
        for (uint32 i = 0; i < leftFrame.numFeatures && numStereoMatches_ < MAX_STEREO_MATCHES; ++i) {
            Vector2d leftUV({leftFrame.features[i].x, leftFrame.features[i].y});
            
            Vector2d rightUV;
            float64 confidence;
            if (findStereoCorrespondence(leftCamId, rightCamId, leftUV, rightUV, confidence)) {
                float64 disparity = leftUV[0] - rightUV[0];
                
                if (disparity > config_.stereoMinDisparity && 
                    disparity < config_.stereoMaxDisparity) {
                    
                    StereoMatch& match = stereoMatches_[numStereoMatches_];
                    match.featureId = leftFrame.features[i].id;
                    match.leftCameraId = leftCamId;
                    match.rightCameraId = rightCamId;
                    match.leftUV = leftUV;
                    match.rightUV = rightUV;
                    match.disparity = disparity;
                    match.depth = (leftIntrinsics.fx * baseline) / disparity;
                    match.confidence = confidence;
                    
                    float64 x = (leftUV[0] - leftIntrinsics.cx) * match.depth / leftIntrinsics.fx;
                    float64 y = (leftUV[1] - leftIntrinsics.cy) * match.depth / leftIntrinsics.fy;
                    match.position = Vector3d({x, y, match.depth});
                    
                    numStereoMatches_++;
                    totalStereoMatches_++;
                }
            }
        }
    }
    
    bool findStereoCorrespondence(uint8 leftCamId, uint8 rightCamId,
                                   const Vector2d& leftUV, Vector2d& rightUV,
                                   float64& confidence) {
        int32 searchMin = static_cast<int32>(leftUV[0] - config_.stereoMaxDisparity);
        int32 searchMax = static_cast<int32>(leftUV[0] - config_.stereoMinDisparity);
        
        searchMin = max(searchMin, 0);
        searchMax = min(searchMax, static_cast<int32>(frames_[rightCamId].width - 1));
        
        if (searchMin >= searchMax) return false;
        
        int32 halfBlock = config_.stereoBlockSize / 2;
        int32 y = static_cast<int32>(leftUV[1]);
        
        if (y < halfBlock || y >= static_cast<int32>(frames_[rightCamId].height) - halfBlock) {
            return false;
        }
        
        float64 bestScore = 1e10;
        int32 bestX = -1;
        
        for (int32 x = searchMin; x <= searchMax; ++x) {
            float64 sad = 0;
            
            for (int32 dy = -halfBlock; dy <= halfBlock; ++dy) {
                for (int32 dx = -halfBlock; dx <= halfBlock; ++dx) {
                    int32 lx = static_cast<int32>(leftUV[0]) + dx;
                    int32 ly = y + dy;
                    int32 rx = x + dx;
                    int32 ry = y + dy;
                    
                    if (lx >= 0 && lx < static_cast<int32>(frames_[leftCamId].width) &&
                        ly >= 0 && ly < static_cast<int32>(frames_[leftCamId].height) &&
                        rx >= 0 && rx < static_cast<int32>(frames_[rightCamId].width) &&
                        ry >= 0 && ry < static_cast<int32>(frames_[rightCamId].height)) {
                        
                        sad += abs(static_cast<int32>(grayBuffers_[leftCamId][ly * frames_[leftCamId].width + lx]) -
                                  static_cast<int32>(grayBuffers_[rightCamId][ry * frames_[rightCamId].width + rx]));
                    }
                }
            }
            
            if (sad < bestScore) {
                bestScore = sad;
                bestX = x;
            }
        }
        
        if (bestX < 0) return false;
        
        rightUV[0] = static_cast<float64>(bestX);
        rightUV[1] = leftUV[1];
        
        confidence = 1.0 - bestScore / (config_.stereoBlockSize * config_.stereoBlockSize * 255.0);
        
        return confidence > 0.5;
    }
    
    void performCrossCameraTracking() {
        if (!config_.enableCrossCameraTracking || numCameras_ < 2) return;
        
        for (uint32 i = 0; i < numCameras_; ++i) {
            for (uint32 j = i + 1; j < numCameras_; ++j) {
                if (!frames_[i].isValid || !frames_[j].isValid) continue;
                
                float64 overlap = computeCameraOverlap(i, j);
                
                if (overlap > config_.crossCameraMinOverlap) {
                    matchCrossCameraFeatures(i, j);
                }
            }
        }
    }
    
    float64 computeCameraOverlap(uint8 cam1, uint8 cam2) {
        Vector3d forward1 = cameras_[cam1].extrinsics.q_cam_to_body * Vector3d({0, 0, 1});
        Vector3d forward2 = cameras_[cam2].extrinsics.q_cam_to_body * Vector3d({0, 0, 1});
        
        float64 dot = forward1.dot(forward2);
        
        float64 fov1 = 2 * atan(cameras_[cam1].intrinsics.width / 
                               (2 * cameras_[cam1].intrinsics.fx));
        float64 fov2 = 2 * atan(cameras_[cam2].intrinsics.width / 
                               (2 * cameras_[cam2].intrinsics.fx));
        
        float64 angleDiff = acos(clamp(dot, -1.0, 1.0));
        float64 avgFov = (fov1 + fov2) / 2;
        
        float64 overlap = max(0.0, (avgFov - angleDiff) / avgFov);
        
        return overlap;
    }
    
    void matchCrossCameraFeatures(uint8 cam1, uint8 cam2) {
        CameraFrame& frame1 = frames_[cam1];
        CameraFrame& frame2 = frames_[cam2];
        
        for (uint32 i = 0; i < frame1.numFeatures && numFeatures_ < MAX_MULTI_CAM_FEATURES; ++i) {
            for (uint32 j = 0; j < frame2.numFeatures; ++j) {
                if (frame1.features[i].id == frame2.features[j].id) {
                    MultiCameraFeature& feat = features_[numFeatures_++];
                    feat.featureId = frame1.features[i].id;
                    feat.cameraId = cam1;
                    feat.uv = Vector2d({frame1.features[i].x, frame1.features[i].y});
                    feat.isStereoMatched = true;
                    feat.stereoMatchCameraId = cam2;
                    feat.stereoUV = Vector2d({frame2.features[j].x, frame2.features[j].y});
                    break;
                }
            }
        }
    }
    
    void triangulateMultiView(const MSCKFState& state) {
        for (uint32 i = 0; i < numFeatures_; ++i) {
            MultiCameraFeature& feat = features_[i];
            
            if (feat.isTriangulated) continue;
            
            if (feat.isStereoMatched && feat.stereoMatchCameraId < numCameras_) {
                triangulateStereo(state, feat);
            }
        }
    }
    
    void triangulateStereo(const MSCKFState& state, MultiCameraFeature& feat) {
        uint8 cam1 = feat.cameraId;
        uint8 cam2 = feat.stereoMatchCameraId;
        
        Matrix3d R1 = state.imuState.orientation.toRotationMatrix() * 
                     cameras_[cam1].extrinsics.q_cam_to_body.toRotationMatrix();
        Vector3d t1 = state.imuState.position + 
                     state.imuState.orientation.toRotationMatrix() * 
                     cameras_[cam1].extrinsics.t_cam_to_body;
        
        Matrix3d R2 = state.imuState.orientation.toRotationMatrix() * 
                     cameras_[cam2].extrinsics.q_cam_to_body.toRotationMatrix();
        Vector3d t2 = state.imuState.position + 
                     state.imuState.orientation.toRotationMatrix() * 
                     cameras_[cam2].extrinsics.t_cam_to_body;
        
        float64 A[4];
        A[0] = feat.uv[0] * R1(2, 0) - R1(0, 0);
        A[1] = feat.uv[0] * R1(2, 1) - R1(0, 1);
        A[2] = feat.uv[0] * R1(2, 2) - R1(0, 2);
        A[3] = feat.uv[0] * t1[2] - t1[0];
        
        float64 B[4];
        B[0] = feat.uv[1] * R1(2, 0) - R1(1, 0);
        B[1] = feat.uv[1] * R1(2, 1) - R1(1, 1);
        B[2] = feat.uv[1] * R1(2, 2) - R1(1, 2);
        B[3] = feat.uv[1] * t1[2] - t1[1];
        
        float64 C[4];
        C[0] = feat.stereoUV[0] * R2(2, 0) - R2(0, 0);
        C[1] = feat.stereoUV[0] * R2(2, 1) - R2(0, 1);
        C[2] = feat.stereoUV[0] * R2(2, 2) - R2(0, 2);
        C[3] = feat.stereoUV[0] * t2[2] - t2[0];
        
        float64 D[4];
        D[0] = feat.stereoUV[1] * R2(2, 0) - R2(1, 0);
        D[1] = feat.stereoUV[1] * R2(2, 1) - R2(1, 1);
        D[2] = feat.stereoUV[1] * R2(2, 2) - R2(1, 2);
        D[3] = feat.stereoUV[1] * t2[2] - t2[1];
        
        Matrix<4, 4, float64> AtA;
        AtA(0, 0) = A[0]*A[0] + B[0]*B[0] + C[0]*C[0] + D[0]*D[0];
        AtA(0, 1) = A[0]*A[1] + B[0]*B[1] + C[0]*C[1] + D[0]*D[1];
        AtA(0, 2) = A[0]*A[2] + B[0]*B[2] + C[0]*C[2] + D[0]*D[2];
        AtA(0, 3) = A[0]*A[3] + B[0]*B[3] + C[0]*C[3] + D[0]*D[3];
        AtA(1, 0) = AtA(0, 1);
        AtA(1, 1) = A[1]*A[1] + B[1]*B[1] + C[1]*C[1] + D[1]*D[1];
        AtA(1, 2) = A[1]*A[2] + B[1]*B[2] + C[1]*C[2] + D[1]*D[2];
        AtA(1, 3) = A[1]*A[3] + B[1]*B[3] + C[1]*C[3] + D[1]*D[3];
        AtA(2, 0) = AtA(0, 2);
        AtA(2, 1) = AtA(1, 2);
        AtA(2, 2) = A[2]*A[2] + B[2]*B[2] + C[2]*C[2] + D[2]*D[2];
        AtA(2, 3) = A[2]*A[3] + B[2]*B[3] + C[2]*C[3] + D[2]*D[3];
        AtA(3, 0) = AtA(0, 3);
        AtA(3, 1) = AtA(1, 3);
        AtA(3, 2) = AtA(2, 3);
        AtA(3, 3) = A[3]*A[3] + B[3]*B[3] + C[3]*C[3] + D[3]*D[3];
        
        Matrix<4, 4, float64> AtA_inv = AtA.inverse();
        
        Vector4d X;
        X[0] = AtA_inv(0, 3);
        X[1] = AtA_inv(1, 3);
        X[2] = AtA_inv(2, 3);
        X[3] = AtA_inv(3, 3);
        
        if (abs(X[3]) > 1e-6) {
            feat.position[0] = X[0] / X[3];
            feat.position[1] = X[1] / X[3];
            feat.position[2] = X[2] / X[3];
            feat.isTriangulated = true;
            
            if (feat.position[2] > 0.1) {
                feat.invDepth = 1.0 / feat.position[2];
            }
        }
    }
    
    void updateExtrinsics(uint8 cameraId, const Vector3d& delta_p, const Vector3d& delta_theta) {
        if (cameraId >= numCameras_) return;
        
        cameras_[cameraId].extrinsics.t_cam_to_body += delta_p;
        
        Quaterniond dq = Quaterniond::exp(delta_theta);
        cameras_[cameraId].extrinsics.q_cam_to_body = dq * cameras_[cameraId].extrinsics.q_cam_to_body;
        cameras_[cameraId].extrinsics.q_cam_to_body.normalize();
    }
    
    void enableCamera(uint8 cameraId, bool enable) {
        if (cameraId < numCameras_) {
            cameras_[cameraId].isEnabled = enable;
        }
    }
    
    void setExtrinsicOnline(uint8 cameraId, bool online) {
        if (cameraId < numCameras_) {
            extrinsicOnline_[cameraId] = online;
        }
    }
    
    uint32 getNumCameras() const { return numCameras_; }
    uint32 getNumFeatures() const { return numFeatures_; }
    uint32 getNumStereoMatches() const { return numStereoMatches_; }
    uint32 getTotalStereoMatches() const { return totalStereoMatches_; }
    
    const CameraConfig* getCameras() const { return cameras_; }
    const CameraConfig& getCamera(uint8 id) const { return cameras_[id]; }
    const CameraFrame& getFrame(uint8 id) const { return frames_[id]; }
    const MultiCameraFeature* getFeatures() const { return features_; }
    const StereoMatch* getStereoMatches() const { return stereoMatches_; }
    
    uint32 getTotalFrames(uint8 cameraId) const { 
        return cameraId < numCameras_ ? totalFrames_[cameraId] : 0; 
    }
    uint32 getTotalFeatures(uint8 cameraId) const { 
        return cameraId < numCameras_ ? totalFeatures_[cameraId] : 0; 
    }
    
    void reset() {
        numFeatures_ = 0;
        numStereoMatches_ = 0;
        nextFeatureId_ = 0;
        
        for (uint32 i = 0; i < numCameras_; ++i) {
            frames_[i].isValid = false;
            frames_[i].numFeatures = 0;
        }
    }
};

class MultiCameraUpdater {
public:
    struct Config {
        float64 observationNoise;
        float64 maxReprojectionError;
        uint32 minTrackLength;
        bool useStereoConstraints;
        bool useMultiViewConstraints;
        float64 stereoWeight;
        
        Config() 
            : observationNoise(1.0)
            , maxReprojectionError(3.0)
            , minTrackLength(3)
            , useStereoConstraints(true)
            , useMultiViewConstraints(true)
            , stereoWeight(1.5) {}
    };

private:
    Config config_;
    MultiCameraManager* cameraManager_;
    
    float64* H_data_;
    float64* r_data_;
    uint32 maxObservations_;

public:
    MultiCameraUpdater() 
        : cameraManager_(nullptr)
        , H_data_(nullptr)
        , r_data_(nullptr)
        , maxObservations_(0) {}
    
    ~MultiCameraUpdater() {
        if (H_data_) delete[] H_data_;
        if (r_data_) delete[] r_data_;
    }
    
    void init(MultiCameraManager* manager, const Config& config = Config()) {
        cameraManager_ = manager;
        config_ = config;
        
        maxObservations_ = MAX_MULTI_CAM_FEATURES * 4;
        H_data_ = new float64[maxObservations_ * MAX_STATE_DIM];
        r_data_ = new float64[maxObservations_];
    }
    
    void update(MSCKFState& state) {
        if (!cameraManager_) return;
        
        cameraManager_->triangulateMultiView(state);
        
        uint32 obsCount = 0;
        
        if (config_.useStereoConstraints) {
            addStereoConstraints(state, obsCount);
        }
        
        if (config_.useMultiViewConstraints) {
            addMultiViewConstraints(state, obsCount);
        }
        
        if (obsCount > 0) {
            performUpdate(state, obsCount);
        }
    }
    
    void addStereoConstraints(MSCKFState& state, uint32& obsCount) {
        const StereoMatch* matches = cameraManager_->getStereoMatches();
        uint32 numMatches = cameraManager_->getNumStereoMatches();
        
        for (uint32 i = 0; i < numMatches && obsCount < maxObservations_ - 4; ++i) {
            const StereoMatch& match = matches[i];
            
            if (match.depth < 0.1 || match.depth > 100) continue;
            
            const CameraConfig& leftCam = cameraManager_->getCamera(match.leftCameraId);
            const CameraConfig& rightCam = cameraManager_->getCamera(match.rightCameraId);
            
            Matrix3d R_left = state.imuState.orientation.toRotationMatrix() * 
                             leftCam.extrinsics.q_cam_to_body.toRotationMatrix();
            Vector3d t_left = state.imuState.position + 
                             state.imuState.orientation.toRotationMatrix() * 
                             leftCam.extrinsics.t_cam_to_body;
            
            Vector3d p_cam = match.position;
            Vector3d p_world = R_left * p_cam + t_left;
            
            Vector3d p_left = R_left.transpose() * (p_world - t_left);
            
            float64 u_proj = leftCam.intrinsics.fx * p_left[0] / p_left[2] + leftCam.intrinsics.cx;
            float64 v_proj = leftCam.intrinsics.fy * p_left[1] / p_left[2] + leftCam.intrinsics.cy;
            
            r_data_[obsCount] = match.leftUV[0] - u_proj;
            r_data_[obsCount + 1] = match.leftUV[1] - v_proj;
            
            obsCount += 2;
        }
    }
    
    void addMultiViewConstraints(MSCKFState& state, uint32& obsCount) {
        const MultiCameraFeature* features = cameraManager_->getFeatures();
        uint32 numFeatures = cameraManager_->getNumFeatures();
        
        for (uint32 i = 0; i < numFeatures && obsCount < maxObservations_ - 4; ++i) {
            const MultiCameraFeature& feat = features[i];
            
            if (!feat.isTriangulated) continue;
            if (feat.trackCount < config_.minTrackLength) continue;
            
            const CameraConfig& cam = cameraManager_->getCamera(feat.cameraId);
            
            Matrix3d R_cam = state.imuState.orientation.toRotationMatrix() * 
                            cam.extrinsics.q_cam_to_body.toRotationMatrix();
            Vector3d t_cam = state.imuState.position + 
                            state.imuState.orientation.toRotationMatrix() * 
                            cam.extrinsics.t_cam_to_body;
            
            Vector3d p_cam_frame = R_cam.transpose() * (feat.position - t_cam);
            
            if (p_cam_frame[2] < 0.1) continue;
            
            float64 u_proj = cam.intrinsics.fx * p_cam_frame[0] / p_cam_frame[2] + cam.intrinsics.cx;
            float64 v_proj = cam.intrinsics.fy * p_cam_frame[1] / p_cam_frame[2] + cam.intrinsics.cy;
            
            r_data_[obsCount] = feat.uv[0] - u_proj;
            r_data_[obsCount + 1] = feat.uv[1] - v_proj;
            
            obsCount += 2;
        }
    }
    
    void performUpdate(MSCKFState& state, uint32 obsCount) {
        if (obsCount == 0) return;
        
        uint32 stateDim = state.stateDim();
        
        MatrixXd S(obsCount, obsCount);
        for (uint32 i = 0; i < obsCount; ++i) {
            for (uint32 j = 0; j < obsCount; ++j) {
                S(i, j) = 0;
                for (uint32 k = 0; k < stateDim; ++k) {
                    for (uint32 l = 0; l < stateDim; ++l) {
                        S(i, j) += H_data_[i * stateDim + k] * state.covariance(k, l) * 
                                  H_data_[j * stateDim + l];
                    }
                }
                if (i == j) S(i, j) += config_.observationNoise;
            }
        }
        
        VectorXd r(obsCount);
        for (uint32 i = 0; i < obsCount; ++i) {
            r[i] = r_data_[i];
        }
        
        VectorXd K = S.ldlt().solve(r);
        
        for (uint32 i = 0; i < 3; ++i) {
            float64 correction = 0;
            for (uint32 j = 0; j < obsCount; ++j) {
                correction += K[j] * H_data_[j * stateDim + i];
            }
            state.imuState.position[i] += correction * 0.01;
        }
    }
    
    void reset() {
    }
};

}

#endif
