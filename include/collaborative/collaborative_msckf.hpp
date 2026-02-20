#ifndef MSCKF_COLLABORATIVE_COLLABORATIVE_MSCKF_HPP
#define MSCKF_COLLABORATIVE_COLLABORATIVE_MSCKF_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../math/lie.hpp"
#include "../core/state.hpp"
#include "../system/communication.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 MAX_NEIGHBORS = 8;
constexpr uint32 MAX_SHARED_FEATURES = 100;
constexpr uint32 MAX_KEYFRAMES = 50;
constexpr uint32 DESCRIPTOR_SIZE = 32;

struct CompressedFeature {
    float64 u;
    float64 v;
    uint8 descriptor[DESCRIPTOR_SIZE];
    float64 invDepth;
    uint32 featureId;
    uint8 quality;
    
    CompressedFeature() 
        : u(0), v(0), invDepth(0), featureId(0), quality(0) {
        memset(descriptor, 0, DESCRIPTOR_SIZE);
    }
    
    uint32 serializedSize() const {
        return sizeof(float64) * 3 + DESCRIPTOR_SIZE + sizeof(uint32) + sizeof(uint8);
    }
    
    void serialize(uint8* buffer) const {
        uint32 idx = 0;
        memcpy(buffer + idx, &u, sizeof(float64)); idx += sizeof(float64);
        memcpy(buffer + idx, &v, sizeof(float64)); idx += sizeof(float64);
        memcpy(buffer + idx, &invDepth, sizeof(float64)); idx += sizeof(float64);
        memcpy(buffer + idx, descriptor, DESCRIPTOR_SIZE); idx += DESCRIPTOR_SIZE;
        memcpy(buffer + idx, &featureId, sizeof(uint32)); idx += sizeof(uint32);
        memcpy(buffer + idx, &quality, sizeof(uint8));
    }
    
    void deserialize(const uint8* buffer) {
        uint32 idx = 0;
        memcpy(&u, buffer + idx, sizeof(float64)); idx += sizeof(float64);
        memcpy(&v, buffer + idx, sizeof(float64)); idx += sizeof(float64);
        memcpy(&invDepth, buffer + idx, sizeof(float64)); idx += sizeof(float64);
        memcpy(descriptor, buffer + idx, DESCRIPTOR_SIZE); idx += DESCRIPTOR_SIZE;
        memcpy(&featureId, buffer + idx, sizeof(uint32)); idx += sizeof(uint32);
        memcpy(&quality, buffer + idx, sizeof(uint8));
    }
};

struct PoseWithCovariance {
    Vector3d position;
    Quaterniond orientation;
    Matrix<6, 6, float64> covariance;
    uint64 timestamp;
    
    PoseWithCovariance() 
        : position({0, 0, 0})
        , orientation()
        , timestamp(0) {
        covariance = Matrix<6, 6, float64>::identity();
    }
    
    uint32 serializedSize() const {
        return sizeof(float64) * 3 + sizeof(float64) * 4 + sizeof(float64) * 36 + sizeof(uint64);
    }
    
    void serialize(uint8* buffer) const {
        uint32 idx = 0;
        memcpy(buffer + idx, position.data(), sizeof(float64) * 3); idx += sizeof(float64) * 3;
        float64 quat[4] = {orientation.w, orientation.x, orientation.y, orientation.z};
        memcpy(buffer + idx, quat, sizeof(float64) * 4); idx += sizeof(float64) * 4;
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                memcpy(buffer + idx, &covariance(i, j), sizeof(float64));
                idx += sizeof(float64);
            }
        }
        memcpy(buffer + idx, &timestamp, sizeof(uint64));
    }
    
    void deserialize(const uint8* buffer) {
        uint32 idx = 0;
        memcpy(position.data(), buffer + idx, sizeof(float64) * 3); idx += sizeof(float64) * 3;
        float64 quat[4];
        memcpy(quat, buffer + idx, sizeof(float64) * 4); idx += sizeof(float64) * 4;
        orientation.w = quat[0]; orientation.x = quat[1]; orientation.y = quat[2]; orientation.z = quat[3];
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                memcpy(&covariance(i, j), buffer + idx, sizeof(float64));
                idx += sizeof(float64);
            }
        }
        memcpy(&timestamp, buffer + idx, sizeof(uint64));
    }
};

struct NeighborDrone {
    uint8 id;
    PoseWithCovariance pose;
    CompressedFeature features[MAX_SHARED_FEATURES];
    uint32 numFeatures;
    uint64 timestamp;
    float64 distance;
    float64 informationGain;
    bool isActive;
    uint64 lastUpdateTime;
    
    NeighborDrone() 
        : id(0)
        , numFeatures(0)
        , timestamp(0)
        , distance(0)
        , informationGain(0)
        , isActive(false)
        , lastUpdateTime(0) {}
};

struct KeyframeInfo {
    uint32 frameId;
    PoseWithCovariance pose;
    CompressedFeature features[MAX_SHARED_FEATURES];
    uint32 numFeatures;
    uint64 timestamp;
    bool isMarginalized;
    
    KeyframeInfo() 
        : frameId(0), numFeatures(0), timestamp(0), isMarginalized(false) {}
};

class CollaborativeMSCKF {
public:
    struct Config {
        uint32 maxNeighbors;
        float64 maxCommunicationRange;
        float64 minInformationGain;
        uint32 maxSharedFeatures;
        float64 ciFusionWeight;
        float64 admmRho;
        uint32 admmMaxIterations;
        float64 admmConvergenceThreshold;
        float64 neighborTimeout;
        bool enableDynamicTopology;
        bool enableConsensusOptimization;
        
        Config() 
            : maxNeighbors(MAX_NEIGHBORS)
            , maxCommunicationRange(100.0)
            , minInformationGain(0.1)
            , maxSharedFeatures(MAX_SHARED_FEATURES)
            , ciFusionWeight(0.5)
            , admmRho(1.0)
            , admmMaxIterations(10)
            , admmConvergenceThreshold(1e-4)
            , neighborTimeout(5000000)
            , enableDynamicTopology(true)
            , enableConsensusOptimization(true) {}
    };

private:
    Config config_;
    uint8 selfId_;
    
    NeighborDrone neighbors_[MAX_NEIGHBORS];
    uint32 numNeighbors_;
    
    KeyframeInfo keyframes_[MAX_KEYFRAMES];
    uint32 numKeyframes_;
    uint32 nextKeyframeId_;
    
    PoseWithCovariance currentPose_;
    Matrix<6, 6, float64> informationMatrix_;
    
    Feature* localFeatures_;
    uint32 numLocalFeatures_;
    
    uint64 lastBroadcastTime_;
    uint32 broadcastInterval_;
    
    float64* admmLambda_;
    float64* admmZ_;
    
    uint32 matchedFeaturePairs_[MAX_SHARED_FEATURES * 2];
    uint32 numMatchedPairs_;
    
    float64 totalInformationGain_;
    uint32 successfulFusions_;

public:
    CollaborativeMSCKF() 
        : selfId_(0)
        , numNeighbors_(0)
        , numKeyframes_(0)
        , nextKeyframeId_(0)
        , localFeatures_(nullptr)
        , numLocalFeatures_(0)
        , lastBroadcastTime_(0)
        , broadcastInterval_(100000)
        , admmLambda_(nullptr)
        , admmZ_(nullptr)
        , numMatchedPairs_(0)
        , totalInformationGain_(0)
        , successfulFusions_(0) {
        informationMatrix_ = Matrix<6, 6, float64>::identity();
    }
    
    ~CollaborativeMSCKF() {
        if (admmLambda_) delete[] admmLambda_;
        if (admmZ_) delete[] admmZ_;
    }
    
    void init(uint8 selfId, const Config& config = Config()) {
        selfId_ = selfId;
        config_ = config;
        
        admmLambda_ = new float64[6 * MAX_NEIGHBORS];
        admmZ_ = new float64[6 * MAX_NEIGHBORS];
        
        memset(admmLambda_, 0, sizeof(float64) * 6 * MAX_NEIGHBORS);
        memset(admmZ_, 0, sizeof(float64) * 6 * MAX_NEIGHBORS);
    }
    
    void setLocalFeatures(Feature* features, uint32 numFeatures) {
        localFeatures_ = features;
        numLocalFeatures_ = numFeatures;
    }
    
    void updateCurrentPose(const MSCKFState& state) {
        currentPose_.position = state.imuState.position;
        currentPose_.orientation = state.imuState.orientation;
        currentPose_.timestamp = state.timestamp;
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                currentPose_.covariance(i, j) = state.covariance(i, j);
                currentPose_.covariance(i + 3, j + 3) = state.covariance(i + 6, j + 6);
            }
        }
    }
    
    void processNeighborData(const NeighborDrone& neighbor) {
        uint32 neighborIdx = findOrCreateNeighbor(neighbor.id);
        
        if (neighborIdx >= MAX_NEIGHBORS) return;
        
        neighbors_[neighborIdx] = neighbor;
        neighbors_[neighborIdx].lastUpdateTime = currentPose_.timestamp;
        neighbors_[neighborIdx].isActive = true;
        
        computeDistance(neighbors_[neighborIdx]);
        computeInformationGain(neighbors_[neighborIdx]);
        
        if (config_.enableConsensusOptimization) {
            performConsensusOptimization(neighbors_[neighborIdx]);
        }
        
        performCovarianceIntersectionFusion(neighbors_[neighborIdx]);
    }
    
    void update() {
        removeInactiveNeighbors();
        
        if (config_.enableDynamicTopology) {
            selectBestNeighbors();
        }
        
        if (shouldBroadcast()) {
            prepareBroadcastData();
            lastBroadcastTime_ = currentPose_.timestamp;
        }
    }
    
    void addKeyframe(const MSCKFState& state, Feature* features, uint32 numFeatures) {
        if (numKeyframes_ >= MAX_KEYFRAMES) {
            marginalizeOldestKeyframe();
        }
        
        KeyframeInfo& kf = keyframes_[numKeyframes_];
        kf.frameId = nextKeyframeId_++;
        kf.pose.position = state.imuState.position;
        kf.pose.orientation = state.imuState.orientation;
        kf.pose.timestamp = state.timestamp;
        kf.numFeatures = min(numFeatures, config_.maxSharedFeatures);
        kf.isMarginalized = false;
        
        for (uint32 i = 0; i < kf.numFeatures; ++i) {
            compressFeature(features[i], kf.features[i]);
        }
        
        numKeyframes_++;
    }
    
    void marginalizeOldestKeyframe() {
        if (numKeyframes_ == 0) return;
        
        for (uint32 i = 0; i < numKeyframes_ - 1; ++i) {
            keyframes_[i] = keyframes_[i + 1];
        }
        numKeyframes_--;
    }
    
    void compressFeature(const Feature& feature, CompressedFeature& compressed) {
        compressed.u = feature.observations[feature.numObservations - 1].u;
        compressed.v = feature.observations[feature.numObservations - 1].v;
        compressed.invDepth = feature.invDepth;
        compressed.featureId = feature.id;
        compressed.quality = min(feature.trackCount, 255u);
        
        generateDescriptor(feature, compressed.descriptor);
    }
    
    void generateDescriptor(const Feature& feature, uint8* descriptor) {
        uint32 hash = feature.id;
        for (uint32 i = 0; i < feature.numObservations && i < 8; ++i) {
            uint32 uHash = static_cast<uint32>(feature.observations[i].u * 1000) & 0xFFFF;
            uint32 vHash = static_cast<uint32>(feature.observations[i].v * 1000) & 0xFFFF;
            hash ^= (uHash << 16) | vHash;
        }
        
        for (uint32 i = 0; i < DESCRIPTOR_SIZE; ++i) {
            descriptor[i] = (hash >> (i % 4 * 8)) & 0xFF;
        }
    }
    
    uint32 matchFeatures(const CompressedFeature* neighborFeatures, uint32 numNeighborFeatures) {
        numMatchedPairs_ = 0;
        
        for (uint32 i = 0; i < numNeighborFeatures && numMatchedPairs_ < MAX_SHARED_FEATURES; ++i) {
            for (uint32 j = 0; j < numLocalFeatures_; ++j) {
                if (localFeatures_[j].id == neighborFeatures[i].featureId) {
                    matchedFeaturePairs_[numMatchedPairs_ * 2] = j;
                    matchedFeaturePairs_[numMatchedPairs_ * 2 + 1] = i;
                    numMatchedPairs_++;
                    break;
                }
            }
        }
        
        return numMatchedPairs_;
    }
    
    void performCovarianceIntersectionFusion(NeighborDrone& neighbor) {
        if (numMatchedPairs_ < 3) return;
        
        Matrix<6, 6, float64> P1 = currentPose_.covariance;
        Matrix<6, 6, float64> P2 = neighbor.pose.covariance;
        
        float64 w = computeOptimalCIWeight(P1, P2);
        
        Matrix<6, 6, float64> P1_inv = P1.inverse();
        Matrix<6, 6, float64> P2_inv = P2.inverse();
        
        Matrix<6, 6, float64> P_ci_inv;
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                P_ci_inv(i, j) = w * P1_inv(i, j) + (1 - w) * P2_inv(i, j);
            }
        }
        
        Matrix<6, 6, float64> P_ci = P_ci_inv.inverse();
        
        Vector6d x1 = poseToVector(currentPose_);
        Vector6d x2 = poseToVector(neighbor.pose);
        
        Vector6d x_ci;
        for (uint32 i = 0; i < 6; ++i) {
            float64 sum = 0;
            for (uint32 j = 0; j < 6; ++j) {
                sum += P_ci(i, j) * (w * P1_inv(i, j) * x1[j] + (1 - w) * P2_inv(i, j) * x2[j]);
            }
            x_ci[i] = sum;
        }
        
        currentPose_.covariance = P_ci;
        vectorToPose(x_ci, currentPose_);
        
        successfulFusions_++;
    }
    
    float64 computeOptimalCIWeight(const Matrix<6, 6, float64>& P1, 
                                   const Matrix<6, 6, float64>& P2) {
        float64 bestW = 0.5;
        float64 minDet = 1e30;
        
        for (float64 w = 0.1; w <= 0.9; w += 0.1) {
            Matrix<6, 6, float64> P1_inv = P1.inverse();
            Matrix<6, 6, float64> P2_inv = P2.inverse();
            
            Matrix<6, 6, float64> P_ci_inv;
            for (uint32 i = 0; i < 6; ++i) {
                for (uint32 j = 0; j < 6; ++j) {
                    P_ci_inv(i, j) = w * P1_inv(i, j) + (1 - w) * P2_inv(i, j);
                }
            }
            
            float64 det = computeDeterminant6x6(P_ci_inv);
            if (det < minDet && det > 0) {
                minDet = det;
                bestW = w;
            }
        }
        
        return bestW;
    }
    
    float64 computeDeterminant6x6(const Matrix<6, 6, float64>& M) {
        float64 det = 1.0;
        float64 temp[6][6];
        
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                temp[i][j] = M(i, j);
            }
        }
        
        for (uint32 i = 0; i < 6; ++i) {
            uint32 maxRow = i;
            for (uint32 k = i + 1; k < 6; ++k) {
                if (abs(temp[k][i]) > abs(temp[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            if (maxRow != i) {
                for (uint32 k = 0; k < 6; ++k) {
                    float64 tmp = temp[i][k];
                    temp[i][k] = temp[maxRow][k];
                    temp[maxRow][k] = tmp;
                }
                det *= -1;
            }
            
            if (abs(temp[i][i]) < 1e-10) return 0;
            
            det *= temp[i][i];
            
            for (uint32 k = i + 1; k < 6; ++k) {
                float64 factor = temp[k][i] / temp[i][i];
                for (uint32 j = i; j < 6; ++j) {
                    temp[k][j] -= factor * temp[i][j];
                }
            }
        }
        
        return det;
    }
    
    void performConsensusOptimization(NeighborDrone& neighbor) {
        uint32 neighborIdx = findNeighborIndex(neighbor.id);
        if (neighborIdx >= MAX_NEIGHBORS) return;
        
        Vector6d x_local = poseToVector(currentPose_);
        Vector6d x_neighbor = poseToVector(neighbor.pose);
        
        float64* lambda = admmLambda_ + neighborIdx * 6;
        float64* z = admmZ_ + neighborIdx * 6;
        
        for (uint32 iter = 0; iter < config_.admmMaxIterations; ++iter) {
            Vector6d x_new;
            for (uint32 i = 0; i < 6; ++i) {
                x_new[i] = 0.5 * (x_local[i] + x_neighbor[i]) - lambda[i] / (2 * config_.admmRho);
            }
            
            for (uint32 i = 0; i < 6; ++i) {
                lambda[i] = lambda[i] + config_.admmRho * (x_new[i] - z[i]);
                z[i] = x_new[i];
            }
            
            float64 primalResidual = 0;
            for (uint32 i = 0; i < 6; ++i) {
                primalResidual += (x_new[i] - z[i]) * (x_new[i] - z[i]);
            }
            
            if (sqrt(primalResidual) < config_.admmConvergenceThreshold) {
                break;
            }
        }
    }
    
    void computeDistance(NeighborDrone& neighbor) {
        Vector3d diff = currentPose_.position - neighbor.pose.position;
        neighbor.distance = diff.norm();
    }
    
    void computeInformationGain(NeighborDrone& neighbor) {
        float64 distanceFactor = exp(-neighbor.distance / config_.maxCommunicationRange);
        
        uint32 commonFeatures = matchFeatures(neighbor.features, neighbor.numFeatures);
        float64 featureFactor = static_cast<float64>(commonFeatures) / config_.maxSharedFeatures;
        
        float64 covDet1 = computeDeterminant6x6(currentPose_.covariance);
        float64 covDet2 = computeDeterminant6x6(neighbor.pose.covariance);
        float64 uncertaintyFactor = sqrt(covDet2) / (sqrt(covDet1) + sqrt(covDet2) + 1e-10);
        
        neighbor.informationGain = distanceFactor * featureFactor * uncertaintyFactor;
    }
    
    void selectBestNeighbors() {
        if (numNeighbors_ <= config_.maxNeighbors) return;
        
        for (uint32 i = 0; i < numNeighbors_ - 1; ++i) {
            for (uint32 j = i + 1; j < numNeighbors_; ++j) {
                if (neighbors_[j].informationGain > neighbors_[i].informationGain) {
                    NeighborDrone temp = neighbors_[i];
                    neighbors_[i] = neighbors_[j];
                    neighbors_[j] = temp;
                }
            }
        }
        
        numNeighbors_ = config_.maxNeighbors;
    }
    
    void removeInactiveNeighbors() {
        uint32 writeIdx = 0;
        for (uint32 i = 0; i < numNeighbors_; ++i) {
            uint64 timeSinceUpdate = currentPose_.timestamp - neighbors_[i].lastUpdateTime;
            if (timeSinceUpdate < config_.neighborTimeout) {
                if (writeIdx != i) {
                    neighbors_[writeIdx] = neighbors_[i];
                }
                writeIdx++;
            }
        }
        numNeighbors_ = writeIdx;
    }
    
    bool shouldBroadcast() {
        return (currentPose_.timestamp - lastBroadcastTime_) >= broadcastInterval_;
    }
    
    void prepareBroadcastData() {
    }
    
    uint32 serializeBroadcastData(uint8* buffer, uint32 maxSize) {
        uint32 idx = 0;
        
        buffer[idx++] = selfId_;
        
        currentPose_.serialize(buffer + idx);
        idx += currentPose_.serializedSize();
        
        uint32 numFeaturesToShare = min(numLocalFeatures_, config_.maxSharedFeatures);
        memcpy(buffer + idx, &numFeaturesToShare, sizeof(uint32));
        idx += sizeof(uint32);
        
        for (uint32 i = 0; i < numFeaturesToShare && idx < maxSize; ++i) {
            CompressedFeature cf;
            compressFeature(localFeatures_[i], cf);
            cf.serialize(buffer + idx);
            idx += cf.serializedSize();
        }
        
        return idx;
    }
    
    bool deserializeNeighborData(const uint8* buffer, uint32 length, NeighborDrone& neighbor) {
        uint32 idx = 0;
        
        if (length < 1) return false;
        neighbor.id = buffer[idx++];
        
        neighbor.pose.deserialize(buffer + idx);
        idx += neighbor.pose.serializedSize();
        
        if (idx + sizeof(uint32) > length) return false;
        memcpy(&neighbor.numFeatures, buffer + idx, sizeof(uint32));
        idx += sizeof(uint32);
        
        for (uint32 i = 0; i < neighbor.numFeatures && idx < length; ++i) {
            neighbor.features[i].deserialize(buffer + idx);
            idx += neighbor.features[i].serializedSize();
        }
        
        neighbor.timestamp = currentPose_.timestamp;
        return true;
    }
    
    uint32 findOrCreateNeighbor(uint8 neighborId) {
        for (uint32 i = 0; i < numNeighbors_; ++i) {
            if (neighbors_[i].id == neighborId) {
                return i;
            }
        }
        
        if (numNeighbors_ < MAX_NEIGHBORS) {
            neighbors_[numNeighbors_].id = neighborId;
            return numNeighbors_++;
        }
        
        return MAX_NEIGHBORS;
    }
    
    uint32 findNeighborIndex(uint8 neighborId) {
        for (uint32 i = 0; i < numNeighbors_; ++i) {
            if (neighbors_[i].id == neighborId) {
                return i;
            }
        }
        return MAX_NEIGHBORS;
    }
    
    Vector6d poseToVector(const PoseWithCovariance& pose) {
        Vector6d v;
        v[0] = pose.position[0];
        v[1] = pose.position[1];
        v[2] = pose.position[2];
        
        Vector3d euler = pose.orientation.toEulerAngles();
        v[3] = euler[0];
        v[4] = euler[1];
        v[5] = euler[2];
        
        return v;
    }
    
    void vectorToPose(const Vector6d& v, PoseWithCovariance& pose) {
        pose.position[0] = v[0];
        pose.position[1] = v[1];
        pose.position[2] = v[2];
        
        pose.orientation = Quaterniond::fromEulerAngles(v[3], v[4], v[5]);
    }
    
    const PoseWithCovariance& getCurrentPose() const { return currentPose_; }
    uint32 getNumNeighbors() const { return numNeighbors_; }
    uint32 getNumKeyframes() const { return numKeyframes_; }
    float64 getTotalInformationGain() const { return totalInformationGain_; }
    uint32 getSuccessfulFusions() const { return successfulFusions_; }
    
    const NeighborDrone* getNeighbors() const { return neighbors_; }
    const KeyframeInfo* getKeyframes() const { return keyframes_; }
};

}

#endif
