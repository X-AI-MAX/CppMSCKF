#ifndef MSCKF_VISION_CORNER_DETECTOR_HPP
#define MSCKF_VISION_CORNER_DETECTOR_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include <cstring>

namespace msckf {

struct Feature2D {
    float64 x;
    float64 y;
    float64 score;
    uint32 id;
    uint64 timestamp;
    uint32 lifetime;
    
    Feature2D() : x(0), y(0), score(0), id(0), timestamp(0), lifetime(0) {}
    
    Feature2D(float64 x_, float64 y_, float64 score_, uint32 id_ = 0)
        : x(x_), y(y_), score(score_), id(id_), timestamp(0), lifetime(0) {}
    
    Vector2d point() const { return Vector2d({x, y}); }
};

class CornerDetector {
public:
    struct Config {
        uint32 blockSize;
        float64 qualityLevel;
        float64 minDistance;
        uint32 maxFeatures;
        uint32 gridSizeX;
        uint32 gridSizeY;
        float64 k;
        bool useHarris;
        
        Config() {
            blockSize = 3;
            qualityLevel = 0.01;
            minDistance = 10.0;
            maxFeatures = 150;
            gridSizeX = 10;
            gridSizeY = 8;
            k = 0.04;
            useHarris = false;
        }
    };

private:
    Config config_;
    uint32 width_;
    uint32 height_;
    
    float64* gradX_;
    float64* gradY_;
    float64* scoreMap_;
    uint8* mask_;
    
    uint32 nextFeatureId_;

public:
    CornerDetector() 
        : width_(0), height_(0)
        , gradX_(nullptr), gradY_(nullptr)
        , scoreMap_(nullptr), mask_(nullptr)
        , nextFeatureId_(0) {}
    
    ~CornerDetector() {
        releaseBuffers();
    }
    
    void init(uint32 width, uint32 height, const Config& config = Config()) {
        config_ = config;
        width_ = width;
        height_ = height;
        
        allocateBuffers();
    }
    
    void detect(const uint8* image, Feature2D* features, uint32& numFeatures) {
        computeGradients(image);
        
        computeScoreMap();
        
        applyNonMaxSuppression();
        
        selectBestFeatures(features, numFeatures);
    }
    
    void detectWithGrid(const uint8* image, Feature2D* features, uint32& numFeatures) {
        computeGradients(image);
        computeScoreMap();
        applyNonMaxSuppression();
        selectFeaturesByGrid(features, numFeatures);
    }
    
    void createMask(const Feature2D* existingFeatures, uint32 numExisting) {
        memset(mask_, 0, width_ * height_);
        
        uint32 radius = static_cast<uint32>(config_.minDistance);
        
        for (uint32 i = 0; i < numExisting; ++i) {
            int32 cx = static_cast<int32>(existingFeatures[i].x);
            int32 cy = static_cast<int32>(existingFeatures[i].y);
            
            for (int32 dy = -static_cast<int32>(radius); dy <= static_cast<int32>(radius); ++dy) {
                for (int32 dx = -static_cast<int32>(radius); dx <= static_cast<int32>(radius); ++dx) {
                    int32 x = cx + dx;
                    int32 y = cy + dy;
                    if (x >= 0 && x < static_cast<int32>(width_) && 
                        y >= 0 && y < static_cast<int32>(height_)) {
                        mask_[y * width_ + x] = 1;
                    }
                }
            }
        }
    }
    
    void setMaskRegion(uint32 x, uint32 y, uint32 w, uint32 h, bool block = true) {
        for (uint32 j = y; j < y + h && j < height_; ++j) {
            for (uint32 i = x; i < x + w && i < width_; ++i) {
                mask_[j * width_ + i] = block ? 1 : 0;
            }
        }
    }
    
    uint32 getNextFeatureId() {
        return nextFeatureId_++;
    }

private:
    void allocateBuffers() {
        uint32 size = width_ * height_;
        gradX_ = new float64[size];
        gradY_ = new float64[size];
        scoreMap_ = new float64[size];
        mask_ = new uint8[size];
        
        memset(mask_, 0, size);
    }
    
    void releaseBuffers() {
        if (gradX_) { delete[] gradX_; gradX_ = nullptr; }
        if (gradY_) { delete[] gradY_; gradY_ = nullptr; }
        if (scoreMap_) { delete[] scoreMap_; scoreMap_ = nullptr; }
        if (mask_) { delete[] mask_; mask_ = nullptr; }
    }
    
    void computeGradients(const uint8* image) {
        for (uint32 y = 1; y < height_ - 1; ++y) {
            for (uint32 x = 1; x < width_ - 1; ++x) {
                uint32 idx = y * width_ + x;
                gradX_[idx] = static_cast<float64>(image[idx + 1]) - static_cast<float64>(image[idx - 1]);
                gradY_[idx] = static_cast<float64>(image[idx + width_]) - static_cast<float64>(image[idx - width_]);
            }
        }
        
        for (uint32 x = 0; x < width_; ++x) {
            gradX_[x] = 0;
            gradY_[x] = 0;
            gradX_[(height_ - 1) * width_ + x] = 0;
            gradY_[(height_ - 1) * width_ + x] = 0;
        }
        for (uint32 y = 0; y < height_; ++y) {
            gradX_[y * width_] = 0;
            gradY_[y * width_] = 0;
            gradX_[y * width_ + width_ - 1] = 0;
            gradY_[y * width_ + width_ - 1] = 0;
        }
    }
    
    void computeScoreMap() {
        uint32 halfBlock = config_.blockSize / 2;
        
        for (uint32 y = halfBlock; y < height_ - halfBlock; ++y) {
            for (uint32 x = halfBlock; x < width_ - halfBlock; ++x) {
                float64 Ixx = 0, Iyy = 0, Ixy = 0;
                
                for (uint32 dy = -halfBlock; dy <= halfBlock; ++dy) {
                    for (uint32 dx = -halfBlock; dx <= halfBlock; ++dx) {
                        uint32 idx = (y + dy) * width_ + (x + dx);
                        float64 gx = gradX_[idx];
                        float64 gy = gradY_[idx];
                        
                        Ixx += gx * gx;
                        Iyy += gy * gy;
                        Ixy += gx * gy;
                    }
                }
                
                float64 score;
                if (config_.useHarris) {
                    float64 det = Ixx * Iyy - Ixy * Ixy;
                    float64 trace = Ixx + Iyy;
                    score = det - config_.k * trace * trace;
                } else {
                    float64 trace = Ixx + Iyy;
                    float64 det = Ixx * Iyy - Ixy * Ixy;
                    float64 lambda1 = trace / 2 + sqrt(max(trace * trace / 4 - det, 0.0));
                    float64 lambda2 = trace / 2 - sqrt(max(trace * trace / 4 - det, 0.0));
                    score = min(lambda1, lambda2);
                }
                
                scoreMap_[y * width_ + x] = score;
            }
        }
        
        for (uint32 y = 0; y < halfBlock; ++y) {
            for (uint32 x = 0; x < width_; ++x) {
                scoreMap_[y * width_ + x] = 0;
                scoreMap_[(height_ - 1 - y) * width_ + x] = 0;
            }
        }
        for (uint32 x = 0; x < halfBlock; ++x) {
            for (uint32 y = 0; y < height_; ++y) {
                scoreMap_[y * width_ + x] = 0;
                scoreMap_[y * width_ + width_ - 1 - x] = 0;
            }
        }
    }
    
    void applyNonMaxSuppression() {
        uint32 radius = static_cast<uint32>(config_.minDistance / 2);
        if (radius < 1) radius = 1;
        
        float64* tempScore = new float64[width_ * height_];
        memcpy(tempScore, scoreMap_, width_ * height_ * sizeof(float64));
        
        for (uint32 y = radius; y < height_ - radius; ++y) {
            for (uint32 x = radius; x < width_ - radius; ++x) {
                uint32 idx = y * width_ + x;
                float64 val = tempScore[idx];
                
                if (val < EPSILON_F64) continue;
                
                bool isMax = true;
                for (uint32 dy = -radius; dy <= radius && isMax; ++dy) {
                    for (uint32 dx = -radius; dx <= radius; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        if (tempScore[(y + dy) * width_ + (x + dx)] >= val) {
                            isMax = false;
                            break;
                        }
                    }
                }
                
                if (!isMax) {
                    scoreMap_[idx] = 0;
                }
            }
        }
        
        delete[] tempScore;
    }
    
    void selectBestFeatures(Feature2D* features, uint32& numFeatures) {
        float64 maxScore = 0;
        for (uint32 i = 0; i < width_ * height_; ++i) {
            if (scoreMap_[i] > maxScore) maxScore = scoreMap_[i];
        }
        
        float64 threshold = maxScore * config_.qualityLevel;
        
        uint32 count = 0;
        uint32 maxCount = min(numFeatures, config_.maxFeatures);
        
        while (count < maxCount) {
            float64 bestScore = 0;
            uint32 bestIdx = 0;
            bool found = false;
            
            for (uint32 i = 0; i < width_ * height_; ++i) {
                if (scoreMap_[i] > bestScore && mask_[i] == 0) {
                    bestScore = scoreMap_[i];
                    bestIdx = i;
                    found = true;
                }
            }
            
            if (!found || bestScore < threshold) break;
            
            uint32 x = bestIdx % width_;
            uint32 y = bestIdx / width_;
            
            features[count].x = static_cast<float64>(x);
            features[count].y = static_cast<float64>(y);
            features[count].score = bestScore;
            features[count].id = nextFeatureId_++;
            features[count].lifetime = 0;
            count++;
            
            uint32 radius = static_cast<uint32>(config_.minDistance);
            for (int32 dy = -static_cast<int32>(radius); dy <= static_cast<int32>(radius); ++dy) {
                for (int32 dx = -static_cast<int32>(radius); dx <= static_cast<int32>(radius); ++dx) {
                    int32 nx = static_cast<int32>(x) + dx;
                    int32 ny = static_cast<int32>(y) + dy;
                    if (nx >= 0 && nx < static_cast<int32>(width_) &&
                        ny >= 0 && ny < static_cast<int32>(height_)) {
                        scoreMap_[ny * width_ + nx] = 0;
                    }
                }
            }
        }
        
        numFeatures = count;
    }
    
    void selectFeaturesByGrid(Feature2D* features, uint32& numFeatures) {
        float64 maxScore = 0;
        for (uint32 i = 0; i < width_ * height_; ++i) {
            if (scoreMap_[i] > maxScore) maxScore = scoreMap_[i];
        }
        
        float64 threshold = maxScore * config_.qualityLevel;
        
        uint32 cellWidth = width_ / config_.gridSizeX;
        uint32 cellHeight = height_ / config_.gridSizeY;
        
        uint32 featuresPerCell = config_.maxFeatures / (config_.gridSizeX * config_.gridSizeY);
        if (featuresPerCell < 1) featuresPerCell = 1;
        
        uint32 count = 0;
        uint32 maxCount = min(numFeatures, config_.maxFeatures);
        
        for (uint32 gy = 0; gy < config_.gridSizeY && count < maxCount; ++gy) {
            for (uint32 gx = 0; gx < config_.gridSizeX && count < maxCount; ++gx) {
                uint32 startX = gx * cellWidth;
                uint32 startY = gy * cellHeight;
                uint32 endX = min(startX + cellWidth, width_);
                uint32 endY = min(startY + cellHeight, height_);
                
                for (uint32 f = 0; f < featuresPerCell && count < maxCount; ++f) {
                    float64 bestScore = 0;
                    uint32 bestX = 0, bestY = 0;
                    bool found = false;
                    
                    for (uint32 y = startY; y < endY; ++y) {
                        for (uint32 x = startX; x < endX; ++x) {
                            uint32 idx = y * width_ + x;
                            if (scoreMap_[idx] > bestScore && mask_[idx] == 0) {
                                bestScore = scoreMap_[idx];
                                bestX = x;
                                bestY = y;
                                found = true;
                            }
                        }
                    }
                    
                    if (!found || bestScore < threshold) break;
                    
                    features[count].x = static_cast<float64>(bestX);
                    features[count].y = static_cast<float64>(bestY);
                    features[count].score = bestScore;
                    features[count].id = nextFeatureId_++;
                    features[count].lifetime = 0;
                    count++;
                    
                    uint32 radius = static_cast<uint32>(config_.minDistance);
                    for (uint32 dy = 0; dy <= radius; ++dy) {
                        for (uint32 dx = 0; dx <= radius; ++dx) {
                            uint32 nx = bestX + dx;
                            uint32 ny = bestY + dy;
                            if (nx < width_ && ny < height_) {
                                scoreMap_[ny * width_ + nx] = 0;
                            }
                        }
                    }
                }
            }
        }
        
        numFeatures = count;
    }
};

}

#endif
