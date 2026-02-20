#ifndef MSCKF_UTILITY_RETURN_TO_HOME_HPP
#define MSCKF_UTILITY_RETURN_TO_HOME_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../core/state.hpp"
#include "../vision/corner_detector.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 RTH_KEYFRAMES_MAX = 20;
constexpr uint32 RTH_FEATURES_PER_FRAME = 100;
constexpr uint32 RTH_DESCRIPTOR_SIZE = 32;
constexpr uint32 RTH_VOCABULARY_SIZE = 1000;

struct HomeKeyframe {
    uint32 frameId;
    Vector3d position;
    Quaterniond orientation;
    
    Feature2D features[RTH_FEATURES_PER_FRAME];
    uint8 descriptors[RTH_FEATURES_PER_FRAME * RTH_DESCRIPTOR_SIZE];
    uint32 numFeatures;
    
    uint64 timestamp;
    bool isValid;
    
    HomeKeyframe() 
        : frameId(0)
        , position({0, 0, 0})
        , orientation()
        , numFeatures(0)
        , timestamp(0)
        , isValid(false) {}
};

struct RelocalizationResult {
    bool success;
    Vector3d positionCorrection;
    Quaterniond orientationCorrection;
    float64 positionUncertainty;
    float64 orientationUncertainty;
    float64 matchScore;
    uint32 matchedFeatures;
    
    RelocalizationResult() 
        : success(false)
        , positionCorrection({0, 0, 0})
        , orientationCorrection()
        , positionUncertainty(0)
        , orientationUncertainty(0)
        , matchScore(0)
        , matchedFeatures(0) {}
};

class DBoW3Simplified {
public:
    struct Node {
        uint32 nodeId;
        uint32 parentId;
        uint32 children[10];
        uint32 numChildren;
        float64 weight;
        uint8 descriptor[RTH_DESCRIPTOR_SIZE];
        
        Node() : nodeId(0), parentId(0), numChildren(0), weight(1.0) {
            memset(children, 0, sizeof(children));
            memset(descriptor, 0, RTH_DESCRIPTOR_SIZE);
        }
    };

private:
    Node vocabulary_[RTH_VOCABULARY_SIZE];
    uint32 numNodes_;
    uint32 depth_;
    
    float64* queryVector_;
    float64* databaseVector_;
    uint32 vectorSize_;

public:
    DBoW3Simplified() 
        : numNodes_(0)
        , depth_(4)
        , queryVector_(nullptr)
        , databaseVector_(nullptr)
        , vectorSize_(0) {
    }
    
    ~DBoW3Simplified() {
        if (queryVector_) delete[] queryVector_;
        if (databaseVector_) delete[] databaseVector_;
    }
    
    void init() {
        vectorSize_ = RTH_VOCABULARY_SIZE;
        queryVector_ = new float64[vectorSize_];
        databaseVector_ = new float64[vectorSize_];
        
        buildVocabulary();
    }
    
    void buildVocabulary() {
        numNodes_ = 0;
        
        for (uint32 d = 0; d < depth_; ++d) {
            uint32 startNode = numNodes_;
            uint32 nodesAtDepth = 1;
            for (uint32 i = 0; i < d; ++i) {
                nodesAtDepth *= 4;
            }
            
            for (uint32 i = 0; i < nodesAtDepth && numNodes_ < RTH_VOCABULARY_SIZE; ++i) {
                Node& node = vocabulary_[numNodes_];
                node.nodeId = numNodes_;
                
                if (d > 0) {
                    node.parentId = startNode / 4 + i / 4;
                }
                
                for (uint32 j = 0; j < 4 && numNodes_ * 4 + j < RTH_VOCABULARY_SIZE; ++j) {
                    node.children[j] = numNodes_ * 4 + j;
                    node.numChildren++;
                }
                
                node.weight = 1.0 / (d + 1);
                
                for (uint32 k = 0; k < RTH_DESCRIPTOR_SIZE; ++k) {
                    node.descriptor[k] = static_cast<uint8>((numNodes_ + k) % 256);
                }
                
                numNodes_++;
            }
        }
    }
    
    void computeBowVector(const uint8* descriptors, uint32 numDescriptors, float64* bowVector) {
        memset(bowVector, 0, sizeof(float64) * vectorSize_);
        
        for (uint32 i = 0; i < numDescriptors; ++i) {
            const uint8* desc = descriptors + i * RTH_DESCRIPTOR_SIZE;
            uint32 nodeId = findClosestNode(desc);
            
            if (nodeId < numNodes_) {
                bowVector[nodeId] += vocabulary_[nodeId].weight;
            }
        }
        
        float64 norm = 0;
        for (uint32 i = 0; i < vectorSize_; ++i) {
            norm += bowVector[i] * bowVector[i];
        }
        norm = sqrt(norm);
        
        if (norm > 0) {
            for (uint32 i = 0; i < vectorSize_; ++i) {
                bowVector[i] /= norm;
            }
        }
    }
    
    uint32 findClosestNode(const uint8* descriptor) {
        uint32 bestNode = 0;
        float64 bestDist = 1e10;
        
        for (uint32 i = 0; i < numNodes_; ++i) {
            float64 dist = 0;
            for (uint32 j = 0; j < RTH_DESCRIPTOR_SIZE; ++j) {
                float64 diff = static_cast<float64>(descriptor[j]) - 
                              static_cast<float64>(vocabulary_[i].descriptor[j]);
                dist += diff * diff;
            }
            
            if (dist < bestDist) {
                bestDist = dist;
                bestNode = i;
            }
        }
        
        return bestNode;
    }
    
    float64 compareBowVectors(const float64* v1, const float64* v2) {
        float64 score = 0;
        
        for (uint32 i = 0; i < vectorSize_; ++i) {
            if (v1[i] > 0 && v2[i] > 0) {
                score += min(v1[i], v2[i]);
            }
        }
        
        return score;
    }
    
    float64 score(const uint8* queryDesc, uint32 numQuery,
                  const uint8* dbDesc, uint32 numDb) {
        computeBowVector(queryDesc, numQuery, queryVector_);
        computeBowVector(dbDesc, numDb, databaseVector_);
        
        return compareBowVectors(queryVector_, databaseVector_);
    }
};

class ReturnToHome {
public:
    struct Config {
        float64 homeRadius;
        float64 landingAccuracy;
        float64 maxSearchRadius;
        float64 minMatchScore;
        uint32 minMatchedFeatures;
        float64 maxPositionCorrection;
        float64 maxOrientationCorrection;
        float64 descentRate;
        float64 finalDescentHeight;
        bool enableVisualRelocalization;
        bool enableAutoLanding;
        
        Config() 
            : homeRadius(0.5)
            , landingAccuracy(0.3)
            , maxSearchRadius(5.0)
            , minMatchScore(0.3)
            , minMatchedFeatures(15)
            , maxPositionCorrection(2.0)
            , maxOrientationCorrection(0.5)
            , descentRate(0.5)
            , finalDescentHeight(0.5)
            , enableVisualRelocalization(true)
            , enableAutoLanding(true) {}
    };

private:
    Config config_;
    
    HomeKeyframe homeKeyframes_[RTH_KEYFRAMES_MAX];
    uint32 numHomeKeyframes_;
    
    Vector3d homePosition_;
    Quaterniond homeOrientation_;
    bool homePositionSet_;
    
    DBoW3Simplified bowDatabase_;
    
    RelocalizationResult lastRelocResult_;
    
    enum class RTHState : uint8 {
        IDLE,
        NAVIGATING,
        SEARCHING,
        RELOCALIZING,
        DESCENDING,
        LANDING,
        LANDED
    };
    
    RTHState currentState_;
    float64 currentDistance_;
    float64 currentHeight_;
    
    uint32 relocalizationAttempts_;
    uint32 maxRelocalizationAttempts_;
    
    Vector3d targetPosition_;
    
    uint64 lastUpdateTime_;

public:
    ReturnToHome() 
        : numHomeKeyframes_(0)
        , homePosition_({0, 0, 0})
        , homeOrientation_()
        , homePositionSet_(false)
        , currentState_(RTHState::IDLE)
        , currentDistance_(0)
        , currentHeight_(0)
        , relocalizationAttempts_(0)
        , maxRelocalizationAttempts_(10)
        , lastUpdateTime_(0) {
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        bowDatabase_.init();
        currentState_ = RTHState::IDLE;
        numHomeKeyframes_ = 0;
        homePositionSet_ = false;
    }
    
    void setHomePosition(const Vector3d& position, const Quaterniond& orientation) {
        homePosition_ = position;
        homeOrientation_ = orientation;
        homePositionSet_ = true;
    }
    
    void recordHomeKeyframe(const MSCKFState& state, 
                            const Feature2D* features, uint32 numFeatures,
                            const uint8* descriptors) {
        if (numHomeKeyframes_ >= RTH_KEYFRAMES_MAX) {
            for (uint32 i = 0; i < numHomeKeyframes_ - 1; ++i) {
                homeKeyframes_[i] = homeKeyframes_[i + 1];
            }
            numHomeKeyframes_--;
        }
        
        HomeKeyframe& kf = homeKeyframes_[numHomeKeyframes_];
        kf.frameId = numHomeKeyframes_;
        kf.position = state.imuState.position;
        kf.orientation = state.imuState.orientation;
        kf.timestamp = state.timestamp;
        kf.numFeatures = min(numFeatures, RTH_FEATURES_PER_FRAME);
        kf.isValid = true;
        
        for (uint32 i = 0; i < kf.numFeatures; ++i) {
            kf.features[i] = features[i];
        }
        
        if (descriptors) {
            memcpy(kf.descriptors, descriptors, kf.numFeatures * RTH_DESCRIPTOR_SIZE);
        }
        
        if (!homePositionSet_) {
            setHomePosition(state.imuState.position, state.imuState.orientation);
        }
        
        numHomeKeyframes_++;
    }
    
    void startReturnToHome() {
        if (!homePositionSet_) return;
        
        currentState_ = RTHState::NAVIGATING;
        relocalizationAttempts_ = 0;
    }
    
    void update(const MSCKFState& state) {
        if (!homePositionSet_) return;
        
        currentDistance_ = (state.imuState.position - homePosition_).norm();
        currentHeight_ = state.imuState.position[2];
        lastUpdateTime_ = state.timestamp;
        
        switch (currentState_) {
            case RTHState::IDLE:
                break;
                
            case RTHState::NAVIGATING:
                if (currentDistance_ < config_.maxSearchRadius) {
                    currentState_ = RTHState::SEARCHING;
                }
                break;
                
            case RTHState::SEARCHING:
                if (config_.enableVisualRelocalization) {
                    currentState_ = RTHState::RELOCALIZING;
                } else {
                    currentState_ = RTHState::DESCENDING;
                }
                break;
                
            case RTHState::RELOCALIZING:
                break;
                
            case RTHState::DESCENDING:
                if (currentHeight_ < config_.finalDescentHeight) {
                    currentState_ = RTHState::LANDING;
                }
                break;
                
            case RTHState::LANDING:
                if (currentHeight_ < 0.1) {
                    currentState_ = RTHState::LANDED;
                }
                break;
                
            case RTHState::LANDED:
                break;
        }
    }
    
    RelocalizationResult relocalize(const MSCKFState& state,
                                    const Feature2D* features, uint32 numFeatures,
                                    const uint8* descriptors) {
        RelocalizationResult result;
        
        if (numHomeKeyframes_ == 0 || numFeatures < config_.minMatchedFeatures) {
            return result;
        }
        
        float64 bestScore = 0;
        uint32 bestKeyframeIdx = 0;
        
        for (uint32 i = 0; i < numHomeKeyframes_; ++i) {
            float64 score = bowDatabase_.score(
                descriptors, numFeatures,
                homeKeyframes_[i].descriptors, homeKeyframes_[i].numFeatures
            );
            
            if (score > bestScore) {
                bestScore = score;
                bestKeyframeIdx = i;
            }
        }
        
        if (bestScore < config_.minMatchScore) {
            relocalizationAttempts_++;
            return result;
        }
        
        HomeKeyframe& bestKf = homeKeyframes_[bestKeyframeIdx];
        
        uint32 matchedIndices[100];
        uint32 matchedHomeIndices[100];
        uint32 numMatches = 0;
        
        matchFeatures(features, numFeatures, bestKf.features, bestKf.numFeatures,
                      matchedIndices, matchedHomeIndices, numMatches);
        
        if (numMatches < config_.minMatchedFeatures) {
            relocalizationAttempts_++;
            return result;
        }
        
        result = estimatePoseCorrection(state, bestKf, 
                                        features, bestKf.features,
                                        matchedIndices, matchedHomeIndices, numMatches);
        
        result.matchScore = bestScore;
        result.matchedFeatures = numMatches;
        
        if (result.success) {
            lastRelocResult_ = result;
            currentState_ = RTHState::DESCENDING;
        }
        
        return result;
    }
    
    void matchFeatures(const Feature2D* queryFeatures, uint32 numQuery,
                       const Feature2D* dbFeatures, uint32 numDb,
                       uint32* matchedIndices, uint32* matchedDbIndices,
                       uint32& numMatches) {
        numMatches = 0;
        
        for (uint32 i = 0; i < numQuery && numMatches < 100; ++i) {
            float64 bestDist = 1e10;
            uint32 bestIdx = 0;
            
            for (uint32 j = 0; j < numDb; ++j) {
                float64 du = queryFeatures[i].x - dbFeatures[j].x;
                float64 dv = queryFeatures[i].y - dbFeatures[j].y;
                float64 dist = sqrt(du * du + dv * dv);
                
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = j;
                }
            }
            
            if (bestDist < 50.0) {
                matchedIndices[numMatches] = i;
                matchedDbIndices[numMatches] = bestIdx;
                numMatches++;
            }
        }
    }
    
    RelocalizationResult estimatePoseCorrection(const MSCKFState& state,
                                                 const HomeKeyframe& homeKf,
                                                 const Feature2D* queryFeatures,
                                                 const Feature2D* homeFeatures,
                                                 const uint32* queryIndices,
                                                 const uint32* homeIndices,
                                                 uint32 numMatches) {
        RelocalizationResult result;
        
        if (numMatches < 5) return result;
        
        MatrixXd A(numMatches * 2, 6);
        VectorXd b(numMatches * 2);
        
        Matrix3d R_current = state.imuState.orientation.toRotationMatrix();
        Matrix3d R_home = homeKf.orientation.toRotationMatrix();
        
        for (uint32 i = 0; i < numMatches; ++i) {
            uint32 qi = queryIndices[i];
            uint32 hi = homeIndices[i];
            
            float64 u1 = queryFeatures[qi].x;
            float64 v1 = queryFeatures[qi].y;
            float64 u2 = homeFeatures[hi].x;
            float64 v2 = homeFeatures[hi].y;
            
            A(2 * i, 0) = 1; A(2 * i, 1) = 0; A(2 * i, 2) = 0;
            A(2 * i, 3) = 0; A(2 * i, 4) = -u1; A(2 * i, 5) = v1;
            
            A(2 * i + 1, 0) = 0; A(2 * i + 1, 1) = 1; A(2 * i + 1, 2) = 0;
            A(2 * i + 1, 3) = u1; A(2 * i + 1, 4) = 0; A(2 * i + 1, 5) = -v1;
            
            b(2 * i) = u2 - u1;
            b(2 * i + 1) = v2 - v1;
        }
        
        Vector6d x = solveLeastSquares(A, b, numMatches * 2, 6);
        
        result.positionCorrection[0] = x[0];
        result.positionCorrection[1] = x[1];
        result.positionCorrection[2] = x[2];
        
        float64 yawCorrection = x[3];
        result.orientationCorrection = Quaterniond::fromEulerAngles(yawCorrection, 0, 0);
        
        float64 posCorrectionNorm = result.positionCorrection.norm();
        float64 yawCorrectionAbs = abs(yawCorrection);
        
        if (posCorrectionNorm < config_.maxPositionCorrection &&
            yawCorrectionAbs < config_.maxOrientationCorrection) {
            result.success = true;
            result.positionUncertainty = posCorrectionNorm / numMatches;
            result.orientationUncertainty = yawCorrectionAbs / numMatches;
        }
        
        return result;
    }
    
    Vector6d solveLeastSquares(const MatrixXd& A, const VectorXd& b, 
                               uint32 rows, uint32 cols) {
        MatrixXd AtA(6, 6);
        VectorXd Atb(6);
        
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < rows; ++k) {
                    sum += A(k, i) * A(k, j);
                }
                AtA(i, j) = sum;
            }
            
            float64 sum = 0;
            for (uint32 k = 0; k < rows; ++k) {
                sum += A(k, i) * b(k);
            }
            Atb(i) = sum;
        }
        
        return AtA.inverse() * Atb;
    }
    
    Vector3d computeLandingCommand(const MSCKFState& state) {
        Vector3d command({0, 0, 0});
        
        if (currentState_ != RTHState::DESCENDING && 
            currentState_ != RTHState::LANDING) {
            return command;
        }
        
        Vector3d toHome = homePosition_ - state.imuState.position;
        toHome[2] = 0;
        
        float64 horizontalDist = toHome.norm();
        
        if (horizontalDist > config_.homeRadius) {
            command[0] = toHome[0] / horizontalDist * 0.5;
            command[1] = toHome[1] / horizontalDist * 0.5;
        }
        
        if (currentState_ == RTHState::DESCENDING) {
            command[2] = -config_.descentRate;
        } else if (currentState_ == RTHState::LANDING) {
            command[2] = -config_.descentRate * 0.5;
        }
        
        return command;
    }
    
    bool isHomePositionSet() const { return homePositionSet_; }
    Vector3d getHomePosition() const { return homePosition_; }
    Quaterniond getHomeOrientation() const { return homeOrientation_; }
    float64 getDistanceToHome() const { return currentDistance_; }
    uint32 getNumHomeKeyframes() const { return numHomeKeyframes_; }
    uint8 getCurrentState() const { return static_cast<uint8>(currentState_); }
    const RelocalizationResult& getLastRelocResult() const { return lastRelocResult_; }
    
    bool isNavigating() const { return currentState_ == RTHState::NAVIGATING; }
    bool isSearching() const { return currentState_ == RTHState::SEARCHING; }
    bool isRelocalizing() const { return currentState_ == RTHState::RELOCALIZING; }
    bool isDescending() const { return currentState_ == RTHState::DESCENDING; }
    bool isLanding() const { return currentState_ == RTHState::LANDING; }
    bool hasLanded() const { return currentState_ == RTHState::LANDED; }
    
    void cancel() {
        currentState_ = RTHState::IDLE;
    }
    
    void reset() {
        numHomeKeyframes_ = 0;
        homePositionSet_ = false;
        currentState_ = RTHState::IDLE;
        relocalizationAttempts_ = 0;
    }
};

}

#endif
