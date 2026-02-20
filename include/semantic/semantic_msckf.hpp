#ifndef MSCKF_SEMANTIC_SEMANTIC_MSCKF_HPP
#define MSCKF_SEMANTIC_SEMANTIC_MSCKF_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../math/quaternion.hpp"
#include "../core/state.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 SEMANTIC_NUM_CLASSES = 9;
constexpr uint32 SEMANTIC_FEATURES_MAX = 500;
constexpr uint32 SEMANTIC_MAP_POINTS = 1000;
constexpr uint32 CNN_INPUT_SIZE = 64;
constexpr uint32 CNN_FEATURE_CHANNELS = 32;

enum class SemanticClass : uint8 {
    BACKGROUND = 0,
    WALL = 1,
    FLOOR = 2,
    DOOR = 3,
    WINDOW = 4,
    PILLAR = 5,
    PERSON = 6,
    VEHICLE = 7,
    VEGETATION = 8
};

struct SemanticFeature {
    Vector2d uv;
    uint8 classId;
    float32 confidence;
    float64 invDepth;
    Vector3d position;
    uint32 featureId;
    uint32 trackCount;
    bool isValid;
    bool isDynamic;
    
    SemanticFeature() 
        : uv({0, 0})
        , classId(0)
        , confidence(0)
        , invDepth(0)
        , position({0, 0, 0})
        , featureId(0)
        , trackCount(0)
        , isValid(true)
        , isDynamic(false) {}
};

struct SemanticMapPoint {
    Vector3d position;
    uint8 classId;
    float32 confidence;
    uint32 observationCount;
    uint64 lastObservationTime;
    bool isLandmark;
    
    SemanticMapPoint() 
        : position({0, 0, 0})
        , classId(0)
        , confidence(0)
        , observationCount(0)
        , lastObservationTime(0)
        , isLandmark(false) {}
};

class LightweightCNN {
public:
    struct Config {
        uint32 inputSize;
        uint32 numClasses;
        uint32 featureChannels;
        bool useQuantization;
        
        Config() 
            : inputSize(CNN_INPUT_SIZE)
            , numClasses(SEMANTIC_NUM_CLASSES)
            , featureChannels(CNN_FEATURE_CHANNELS)
            , useQuantization(true) {}
    };

private:
    Config config_;
    
    int16_t encoderWeights1_[3 * 3 * 1 * 16];
    int16_t encoderWeights2_[3 * 3 * 16 * 32];
    int16_t encoderWeights3_[3 * 3 * 32 * 64];
    
    int16_t decoderWeights1_[3 * 3 * 64 * 32];
    int16_t decoderWeights2_[3 * 3 * 32 * 16];
    int16_t decoderWeights3_[3 * 3 * 16 * SEMANTIC_NUM_CLASSES];
    
    int32_t encoderBias1_[16];
    int32_t encoderBias2_[32];
    int32_t encoderBias3_[64];
    int32_t decoderBias1_[32];
    int32_t decoderBias2_[16];
    int32_t decoderBias3_[SEMANTIC_NUM_CLASSES];
    
    static constexpr uint32 MAX_INTERMEDIATE_SIZE = CNN_INPUT_SIZE * CNN_INPUT_SIZE * 64 * 4;
    float32 intermediateBuffer_[MAX_INTERMEDIATE_SIZE];
    uint8 outputMask_[CNN_INPUT_SIZE * CNN_INPUT_SIZE];
    
    bool initialized_;

public:
    LightweightCNN() 
        : initialized_(false) {
        memset(encoderWeights1_, 0, sizeof(encoderWeights1_));
        memset(encoderWeights2_, 0, sizeof(encoderWeights2_));
        memset(encoderWeights3_, 0, sizeof(encoderWeights3_));
        memset(decoderWeights1_, 0, sizeof(decoderWeights1_));
        memset(decoderWeights2_, 0, sizeof(decoderWeights2_));
        memset(decoderWeights3_, 0, sizeof(decoderWeights3_));
        memset(intermediateBuffer_, 0, sizeof(intermediateBuffer_));
        memset(outputMask_, 0, sizeof(outputMask_));
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        
        memset(intermediateBuffer_, 0, sizeof(intermediateBuffer_));
        memset(outputMask_, 0, sizeof(outputMask_));
        
        initializeWeights();
        initialized_ = true;
    }
    
    void initializeWeights() {
        for (uint32 i = 0; i < 3 * 3 * 1 * 16; ++i) {
            encoderWeights1_[i] = static_cast<int16_t>((rand() % 200 - 100) * 0.01);
        }
        for (uint32 i = 0; i < 3 * 3 * 16 * 32; ++i) {
            encoderWeights2_[i] = static_cast<int16_t>((rand() % 200 - 100) * 0.01);
        }
        for (uint32 i = 0; i < 3 * 3 * 32 * 64; ++i) {
            encoderWeights3_[i] = static_cast<int16_t>((rand() % 200 - 100) * 0.01);
        }
        
        for (uint32 i = 0; i < 3 * 3 * 64 * 32; ++i) {
            decoderWeights1_[i] = static_cast<int16_t>((rand() % 200 - 100) * 0.01);
        }
        for (uint32 i = 0; i < 3 * 3 * 32 * 16; ++i) {
            decoderWeights2_[i] = static_cast<int16_t>((rand() % 200 - 100) * 0.01);
        }
        for (uint32 i = 0; i < 3 * 3 * 16 * SEMANTIC_NUM_CLASSES; ++i) {
            decoderWeights3_[i] = static_cast<int16_t>((rand() % 200 - 100) * 0.01);
        }
    }
    
    void inference(const uint8* input, uint8* semanticMask, float32* confidence) {
        if (!initialized_) return;
        
        uint32 size = config_.inputSize;
        uint32 stride = 1;
        
        float32* enc1 = intermediateBuffer_;
        float32* enc2 = enc1 + size * size * 16;
        float32* enc3 = enc2 + size * size * 32;
        
        depthwiseConv2d(input, size, size, 1, encoderWeights1_, enc1, 16, stride);
        relu(enc1, size * size * 16);
        
        depthwiseConv2d(enc1, size, size, 16, encoderWeights2_, enc2, 32, stride);
        relu(enc2, size * size * 32);
        
        depthwiseConv2d(enc2, size, size, 32, encoderWeights3_, enc3, 64, stride);
        relu(enc3, size * size * 64);
        
        float32* dec1 = enc3 + size * size * 64;
        float32* dec2 = dec1 + size * size * 32;
        float32* dec3 = dec2 + size * size * 16;
        
        transposeConv2d(enc3, size, size, 64, decoderWeights1_, dec1, 32, stride);
        relu(dec1, size * size * 32);
        
        transposeConv2d(dec1, size, size, 32, decoderWeights2_, dec2, 16, stride);
        relu(dec2, size * size * 16);
        
        transposeConv2d(dec2, size, size, 16, decoderWeights3_, dec3, SEMANTIC_NUM_CLASSES, stride);
        
        softmax(dec3, size, size, SEMANTIC_NUM_CLASSES, semanticMask, confidence);
    }
    
    void depthwiseConv2d(const float32* input, uint32 inH, uint32 inW, uint32 inC,
                         const int16_t* weights, float32* output, 
                         uint32 outC, uint32 stride) {
        uint32 outH = inH;
        uint32 outW = inW;
        
        for (uint32 c = 0; c < outC; ++c) {
            for (uint32 h = 0; h < outH; ++h) {
                for (uint32 w = 0; w < outW; ++w) {
                    float32 sum = 0;
                    
                    for (uint32 kh = 0; kh < 3; ++kh) {
                        for (uint32 kw = 0; kw < 3; ++kw) {
                            int32 ih = h + kh - 1;
                            int32 iw = w + kw - 1;
                            
                            if (ih >= 0 && ih < static_cast<int32>(inH) &&
                                iw >= 0 && iw < static_cast<int32>(inW)) {
                                uint32 weightIdx = (kh * 3 + kw) * outC + c;
                                uint32 inputIdx = (ih * inW + iw) * inC + (c % inC);
                                sum += input[inputIdx] * weights[weightIdx] * 0.0001;
                            }
                        }
                    }
                    
                    output[(h * outW + w) * outC + c] = sum;
                }
            }
        }
    }
    
    void depthwiseConv2d(const uint8* input, uint32 inH, uint32 inW, uint32 inC,
                         const int16_t* weights, float32* output, 
                         uint32 outC, uint32 stride) {
        uint32 outH = inH;
        uint32 outW = inW;
        
        for (uint32 c = 0; c < outC; ++c) {
            for (uint32 h = 0; h < outH; ++h) {
                for (uint32 w = 0; w < outW; ++w) {
                    float32 sum = 0;
                    
                    for (uint32 kh = 0; kh < 3; ++kh) {
                        for (uint32 kw = 0; kw < 3; ++kw) {
                            int32 ih = h + kh - 1;
                            int32 iw = w + kw - 1;
                            
                            if (ih >= 0 && ih < static_cast<int32>(inH) &&
                                iw >= 0 && iw < static_cast<int32>(inW)) {
                                uint32 weightIdx = (kh * 3 + kw) * outC + c;
                                float32 inputVal = input[ih * inW + iw] / 255.0;
                                sum += inputVal * weights[weightIdx] * 0.0001;
                            }
                        }
                    }
                    
                    output[(h * outW + w) * outC + c] = sum;
                }
            }
        }
    }
    
    void transposeConv2d(const float32* input, uint32 inH, uint32 inW, uint32 inC,
                         const int16_t* weights, float32* output, 
                         uint32 outC, uint32 stride) {
        for (uint32 i = 0; i < inH * inW * outC; ++i) {
            output[i] = 0;
        }
        
        for (uint32 c = 0; c < outC; ++c) {
            for (uint32 h = 0; h < inH; ++h) {
                for (uint32 w = 0; w < inW; ++w) {
                    for (uint32 kh = 0; kh < 3; ++kh) {
                        for (uint32 kw = 0; kw < 3; ++kw) {
                            int32 oh = h + kh - 1;
                            int32 ow = w + kw - 1;
                            
                            if (oh >= 0 && oh < static_cast<int32>(inH) &&
                                ow >= 0 && ow < static_cast<int32>(inW)) {
                                uint32 weightIdx = (kh * 3 + kw) * inC * outC + c * inC;
                                uint32 inputIdx = (h * inW + w) * inC;
                                
                                float32 sum = 0;
                                for (uint32 ic = 0; ic < inC; ++ic) {
                                    sum += input[inputIdx + ic] * weights[weightIdx + ic] * 0.0001;
                                }
                                output[(oh * inW + ow) * outC + c] += sum;
                            }
                        }
                    }
                }
            }
        }
    }
    
    void relu(float32* data, uint32 size) {
        for (uint32 i = 0; i < size; ++i) {
            if (data[i] < 0) data[i] = 0;
        }
    }
    
    void softmax(const float32* input, uint32 h, uint32 w, uint32 c,
                 uint8* semanticMask, float32* confidence) {
        for (uint32 i = 0; i < h * w; ++i) {
            float32 maxVal = input[i * c];
            uint32 maxIdx = 0;
            
            for (uint32 j = 1; j < c; ++j) {
                if (input[i * c + j] > maxVal) {
                    maxVal = input[i * c + j];
                    maxIdx = j;
                }
            }
            
            float32 sum = 0;
            for (uint32 j = 0; j < c; ++j) {
                sum += exp(input[i * c + j] - maxVal);
            }
            
            semanticMask[i] = static_cast<uint8>(maxIdx);
            confidence[i] = exp(maxVal - maxVal) / (sum + 1e-10);
        }
    }
    
    void loadWeights(const int16_t* encoder1, const int16_t* encoder2, const int16_t* encoder3,
                     const int16_t* decoder1, const int16_t* decoder2, const int16_t* decoder3) {
        memcpy(encoderWeights1_, encoder1, sizeof(encoderWeights1_));
        memcpy(encoderWeights2_, encoder2, sizeof(encoderWeights2_));
        memcpy(encoderWeights3_, encoder3, sizeof(encoderWeights3_));
        memcpy(decoderWeights1_, decoder1, sizeof(decoderWeights1_));
        memcpy(decoderWeights2_, decoder2, sizeof(decoderWeights2_));
        memcpy(decoderWeights3_, decoder3, sizeof(decoderWeights3_));
    }
    
    uint32 getModelSize() const {
        return sizeof(encoderWeights1_) + sizeof(encoderWeights2_) + sizeof(encoderWeights3_) +
               sizeof(decoderWeights1_) + sizeof(decoderWeights2_) + sizeof(decoderWeights3_);
    }
};

class SemanticMSCKF {
public:
    struct Config {
        float32 minConfidence;
        float32 dynamicClassThreshold;
        bool enableDynamicRejection;
        bool enableSemanticConstraints;
        bool enableSemanticMap;
        uint32 minSemanticFeatures;
        float64 semanticWeight;
        
        Config() 
            : minConfidence(0.5)
            , dynamicClassThreshold(0.7)
            , enableDynamicRejection(true)
            , enableSemanticConstraints(true)
            , enableSemanticMap(true)
            , minSemanticFeatures(10)
            , semanticWeight(1.5) {}
    };

private:
    Config config_;
    LightweightCNN semanticNet_;
    
    SemanticFeature semanticFeatures_[SEMANTIC_FEATURES_MAX];
    uint32 numSemanticFeatures_;
    
    SemanticMapPoint semanticMap_[SEMANTIC_MAP_POINTS];
    uint32 numMapPoints_;
    
    static constexpr uint32 MAX_IMAGE_SIZE = 1280 * 800;
    uint8 semanticMask_[MAX_IMAGE_SIZE];
    float32 confidenceMap_[MAX_IMAGE_SIZE];
    uint32 imageWidth_;
    uint32 imageHeight_;
    
    uint32 nextFeatureId_;
    
    bool isDynamicClass_[SEMANTIC_NUM_CLASSES];
    bool isStructuralClass_[SEMANTIC_NUM_CLASSES];
    
    uint32 dynamicFeatureCount_;
    uint32 structuralFeatureCount_;

public:
    SemanticMSCKF() 
        : numSemanticFeatures_(0)
        , numMapPoints_(0)
        , imageWidth_(0)
        , imageHeight_(0)
        , nextFeatureId_(0)
        , dynamicFeatureCount_(0)
        , structuralFeatureCount_(0) {
        
        for (uint32 i = 0; i < MAX_IMAGE_SIZE; ++i) {
            semanticMask_[i] = 0;
            confidenceMap_[i] = 0;
        }
        
        isDynamicClass_[static_cast<uint8>(SemanticClass::PERSON)] = true;
        isDynamicClass_[static_cast<uint8>(SemanticClass::VEHICLE)] = true;
        
        isStructuralClass_[static_cast<uint8>(SemanticClass::WALL)] = true;
        isStructuralClass_[static_cast<uint8>(SemanticClass::DOOR)] = true;
        isStructuralClass_[static_cast<uint8>(SemanticClass::WINDOW)] = true;
        isStructuralClass_[static_cast<uint8>(SemanticClass::PILLAR)] = true;
    }
    
    void init(uint32 width, uint32 height, const Config& config = Config()) {
        config_ = config;
        imageWidth_ = min(width, static_cast<uint32>(1280));
        imageHeight_ = min(height, static_cast<uint32>(800));
        
        for (uint32 i = 0; i < imageWidth_ * imageHeight_; ++i) {
            semanticMask_[i] = 0;
            confidenceMap_[i] = 0;
        }
        
        LightweightCNN::Config cnnConfig;
        cnnConfig.inputSize = min(imageWidth_, imageHeight_);
        semanticNet_.init(cnnConfig);
    }
    
    void processImage(const uint8* image) {
        semanticNet_.inference(image, semanticMask_, confidenceMap_);
    }
    
    void updateWithSemantic(const std::vector<SemanticFeature>& semFeats) {
        for (const auto& feat : semFeats) {
            if (numSemanticFeatures_ >= SEMANTIC_FEATURES_MAX) break;
            
            semanticFeatures_[numSemanticFeatures_] = feat;
            semanticFeatures_[numSemanticFeatures_].featureId = nextFeatureId_++;
            
            if (isDynamicClass_[feat.classId] && feat.confidence > config_.dynamicClassThreshold) {
                semanticFeatures_[numSemanticFeatures_].isDynamic = true;
                dynamicFeatureCount_++;
            }
            
            if (isStructuralClass_[feat.classId]) {
                structuralFeatureCount_++;
            }
            
            numSemanticFeatures_++;
        }
    }
    
    void addSemanticFeature(float64 u, float64 v, float64 invDepth) {
        if (numSemanticFeatures_ >= SEMANTIC_FEATURES_MAX) return;
        
        uint32 px = static_cast<uint32>(u);
        uint32 py = static_cast<uint32>(v);
        
        if (px >= imageWidth_ || py >= imageHeight_) return;
        
        uint32 idx = py * imageWidth_ + px;
        
        SemanticFeature& feat = semanticFeatures_[numSemanticFeatures_];
        feat.uv[0] = u;
        feat.uv[1] = v;
        feat.classId = semanticMask_[idx];
        feat.confidence = confidenceMap_[idx];
        feat.invDepth = invDepth;
        feat.featureId = nextFeatureId_++;
        feat.trackCount = 0;
        feat.isValid = true;
        feat.isDynamic = false;
        
        if (isDynamicClass_[feat.classId] && feat.confidence > config_.dynamicClassThreshold) {
            feat.isDynamic = true;
            dynamicFeatureCount_++;
        }
        
        if (isStructuralClass_[feat.classId]) {
            structuralFeatureCount_++;
        }
        
        numSemanticFeatures_++;
    }
    
    bool isFeatureValidForOptimization(uint32 featureId) const {
        for (uint32 i = 0; i < numSemanticFeatures_; ++i) {
            if (semanticFeatures_[i].featureId == featureId) {
                if (semanticFeatures_[i].isDynamic) return false;
                if (semanticFeatures_[i].confidence < config_.minConfidence) return false;
                return semanticFeatures_[i].isValid;
            }
        }
        return true;
    }
    
    void filterDynamicFeatures(Feature* features, uint32& numFeatures) {
        uint32 writeIdx = 0;
        for (uint32 i = 0; i < numFeatures; ++i) {
            bool isDynamic = false;
            
            for (uint32 j = 0; j < numSemanticFeatures_; ++j) {
                float64 du = features[i].observations[features[i].numObservations - 1].u - 
                            semanticFeatures_[j].uv[0];
                float64 dv = features[i].observations[features[i].numObservations - 1].v - 
                            semanticFeatures_[j].uv[1];
                float64 dist = sqrt(du * du + dv * dv);
                
                if (dist < 5.0 && semanticFeatures_[j].isDynamic) {
                    isDynamic = true;
                    break;
                }
            }
            
            if (!isDynamic) {
                if (writeIdx != i) {
                    features[writeIdx] = features[i];
                }
                writeIdx++;
            }
        }
        numFeatures = writeIdx;
    }
    
    void applySemanticConstraints(MSCKFState& state, Feature* features, uint32 numFeatures) {
        if (!config_.enableSemanticConstraints) return;
        
        for (uint32 i = 0; i < numSemanticFeatures_; ++i) {
            SemanticFeature& semFeat = semanticFeatures_[i];
            
            if (!semFeat.isValid || semFeat.isDynamic) continue;
            
            if (isStructuralClass_[semFeat.classId] && semFeat.confidence > config_.minConfidence) {
                for (uint32 j = i + 1; j < numSemanticFeatures_; ++j) {
                    SemanticFeature& semFeat2 = semanticFeatures_[j];
                    
                    if (semFeat2.classId == semFeat.classId && 
                        semFeat2.confidence > config_.minConfidence) {
                        applyCoVisibilityConstraint(state, semFeat, semFeat2);
                    }
                }
            }
        }
    }
    
    void applyCoVisibilityConstraint(MSCKFState& state, 
                                     SemanticFeature& feat1, 
                                     SemanticFeature& feat2) {
        if (!feat1.isValid || !feat2.isValid) return;
        
        float64 dist = (feat1.position - feat2.position).norm();
        
        float64 expectedDist = estimateSemanticDistance(feat1.classId);
        
        float64 weight = config_.semanticWeight * min(feat1.confidence, feat2.confidence);
        
        if (dist > 0.1 && dist < 10.0) {
            float64 correction = weight * (expectedDist - dist) * 0.01;
            
            Vector3d direction = (feat2.position - feat1.position).normalized();
            
            feat1.position += direction * correction * 0.5;
            feat2.position -= direction * correction * 0.5;
        }
    }
    
    float64 estimateSemanticDistance(uint8 classId) const {
        switch (static_cast<SemanticClass>(classId)) {
            case SemanticClass::DOOR:
                return 0.9;
            case SemanticClass::WINDOW:
                return 1.5;
            case SemanticClass::PILLAR:
                return 0.4;
            case SemanticClass::WALL:
                return 3.0;
            default:
                return 1.0;
        }
    }
    
    void updateSemanticMap(const MSCKFState& state) {
        if (!config_.enableSemanticMap) return;
        
        for (uint32 i = 0; i < numSemanticFeatures_; ++i) {
            SemanticFeature& feat = semanticFeatures_[i];
            
            if (!feat.isValid || feat.isDynamic) continue;
            
            if (isStructuralClass_[feat.classId]) {
                Matrix3d R = state.imuState.orientation.toRotationMatrix();
                Vector3d p = state.imuState.position;
                
                Vector3d p_camera({feat.uv[0], feat.uv[1], 1.0 / feat.invDepth});
                feat.position = p + R * p_camera;
                
                bool foundMatch = false;
                for (uint32 j = 0; j < numMapPoints_; ++j) {
                    float64 dist = (semanticMap_[j].position - feat.position).norm();
                    if (dist < 0.5) {
                        semanticMap_[j].position = (semanticMap_[j].position * semanticMap_[j].observationCount + 
                                                   feat.position) / (semanticMap_[j].observationCount + 1);
                        semanticMap_[j].observationCount++;
                        semanticMap_[j].lastObservationTime = state.timestamp;
                        semanticMap_[j].confidence = max(semanticMap_[j].confidence, feat.confidence);
                        foundMatch = true;
                        break;
                    }
                }
                
                if (!foundMatch && numMapPoints_ < SEMANTIC_MAP_POINTS) {
                    SemanticMapPoint& mp = semanticMap_[numMapPoints_++];
                    mp.position = feat.position;
                    mp.classId = feat.classId;
                    mp.confidence = feat.confidence;
                    mp.observationCount = 1;
                    mp.lastObservationTime = state.timestamp;
                    mp.isLandmark = (feat.confidence > 0.8 && 
                                    feat.classId == static_cast<uint8>(SemanticClass::DOOR) ||
                                    feat.classId == static_cast<uint8>(SemanticClass::WINDOW));
                }
            }
        }
    }
    
    bool matchSemanticLandmark(const Vector3d& queryPos, uint8 classId, 
                               Vector3d& matchedPos, float64& matchScore) {
        matchScore = 0;
        
        for (uint32 i = 0; i < numMapPoints_; ++i) {
            if (semanticMap_[i].classId != classId) continue;
            if (!semanticMap_[i].isLandmark) continue;
            
            float64 dist = (semanticMap_[i].position - queryPos).norm();
            if (dist < 2.0 && semanticMap_[i].confidence > matchScore) {
                matchedPos = semanticMap_[i].position;
                matchScore = semanticMap_[i].confidence;
                return true;
            }
        }
        
        return false;
    }
    
    void getSemanticClassAt(float64 u, float64 v, uint8& classId, float32& confidence) const {
        uint32 px = static_cast<uint32>(u);
        uint32 py = static_cast<uint32>(v);
        
        if (px >= imageWidth_ || py >= imageHeight_) {
            classId = 0;
            confidence = 0;
            return;
        }
        
        uint32 idx = py * imageWidth_ + px;
        classId = semanticMask_[idx];
        confidence = confidenceMap_[idx];
    }
    
    void clearOldFeatures(uint64 currentTime, uint64 maxAge) {
        uint32 writeIdx = 0;
        for (uint32 i = 0; i < numSemanticFeatures_; ++i) {
            if (semanticFeatures_[i].trackCount < 10) {
                if (writeIdx != i) {
                    semanticFeatures_[writeIdx] = semanticFeatures_[i];
                }
                writeIdx++;
            }
        }
        numSemanticFeatures_ = writeIdx;
    }
    
    void incrementTrackCount(uint32 featureId) {
        for (uint32 i = 0; i < numSemanticFeatures_; ++i) {
            if (semanticFeatures_[i].featureId == featureId) {
                semanticFeatures_[i].trackCount++;
                break;
            }
        }
    }
    
    uint32 getNumSemanticFeatures() const { return numSemanticFeatures_; }
    uint32 getNumMapPoints() const { return numMapPoints_; }
    uint32 getDynamicFeatureCount() const { return dynamicFeatureCount_; }
    uint32 getStructuralFeatureCount() const { return structuralFeatureCount_; }
    const uint8* getSemanticMask() const { return semanticMask_; }
    const SemanticFeature* getSemanticFeatures() const { return semanticFeatures_; }
    const SemanticMapPoint* getSemanticMap() const { return semanticMap_; }
    
    LightweightCNN& getSemanticNet() { return semanticNet_; }
};

}

#endif
