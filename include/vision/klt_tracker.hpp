#ifndef MSCKF_VISION_KLT_TRACKER_HPP
#define MSCKF_VISION_KLT_TRACKER_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "corner_detector.hpp"
#include <cstring>

namespace msckf {

struct TrackResult {
    Feature2D prevFeature;
    Feature2D currFeature;
    bool success;
    float64 error;
    float64 nccScore;
    
    TrackResult() : success(false), error(0), nccScore(0) {}
};

class KLTTracker {
public:
    struct Config {
        uint32 windowSize;
        uint32 maxIterations;
        float64 convergenceThreshold;
        float64 maxDisplacement;
        uint32 numPyramidLevels;
        float64 nccThreshold;
        float64 errorThreshold;
        bool useBidirectional;
        float64 bidirectionalThreshold;
        
        Config() {
            windowSize = 21;
            maxIterations = 30;
            convergenceThreshold = 0.01;
            maxDisplacement = 50.0;
            numPyramidLevels = 3;
            nccThreshold = 0.7;
            errorThreshold = 30.0;
            useBidirectional = true;
            bidirectionalThreshold = 1.0;
        }
    };

private:
    Config config_;
    uint32 width_;
    uint32 height_;
    
    uint8** pyramidPrev_;
    uint8** pyramidCurr_;
    float64** gradX_;
    float64** gradY_;
    uint32* pyramidWidth_;
    uint32* pyramidHeight_;
    
    float64* windowBuffer_;

public:
    KLTTracker()
        : width_(0), height_(0)
        , pyramidPrev_(nullptr), pyramidCurr_(nullptr)
        , gradX_(nullptr), gradY_(nullptr)
        , pyramidWidth_(nullptr), pyramidHeight_(nullptr)
        , windowBuffer_(nullptr) {}
    
    ~KLTTracker() {
        releaseBuffers();
    }
    
    void init(uint32 width, uint32 height, const Config& config = Config()) {
        config_ = config;
        width_ = width;
        height_ = height;
        
        allocateBuffers();
    }
    
    void setImages(const uint8* prevImage, const uint8* currImage) {
        buildPyramid(prevImage, pyramidPrev_, gradX_, gradY_);
        buildPyramid(currImage, pyramidCurr_, nullptr, nullptr);
    }
    
    void track(const Feature2D* prevFeatures, uint32 numPrev,
               Feature2D* currFeatures, bool* success, float64* errors) {
        for (uint32 i = 0; i < numPrev; ++i) {
            TrackResult result;
            trackSingle(prevFeatures[i], result);
            
            currFeatures[i] = result.currFeature;
            success[i] = result.success;
            errors[i] = result.error;
        }
    }
    
    void trackWithValidation(const Feature2D* prevFeatures, uint32 numPrev,
                             TrackResult* results) {
        for (uint32 i = 0; i < numPrev; ++i) {
            trackSingle(prevFeatures[i], results[i]);
            
            if (results[i].success && config_.useBidirectional) {
                validateBidirectional(results[i]);
            }
            
            if (results[i].success) {
                results[i].nccScore = computeNCC(
                    results[i].prevFeature.x, results[i].prevFeature.y,
                    results[i].currFeature.x, results[i].currFeature.y,
                    pyramidPrev_[0], pyramidCurr_[0], width_, height_
                );
                
                if (results[i].nccScore < config_.nccThreshold) {
                    results[i].success = false;
                }
            }
        }
    }
    
    float64 computeNCC(float64 x1, float64 y1, float64 x2, float64 y2,
                       const uint8* img1, const uint8* img2,
                       uint32 w, uint32 h) const {
        int32 halfWin = config_.windowSize / 2;
        
        float64 sum1 = 0, sum2 = 0;
        float64 sumSq1 = 0, sumSq2 = 0;
        float64 sumProd = 0;
        uint32 count = 0;
        
        for (int32 dy = -halfWin; dy <= halfWin; ++dy) {
            for (int32 dx = -halfWin; dx <= halfWin; ++dx) {
                int32 px1 = static_cast<int32>(x1) + dx;
                int32 py1 = static_cast<int32>(y1) + dy;
                int32 px2 = static_cast<int32>(x2) + dx;
                int32 py2 = static_cast<int32>(y2) + dy;
                
                if (px1 < 0 || px1 >= static_cast<int32>(w) ||
                    py1 < 0 || py1 >= static_cast<int32>(h) ||
                    px2 < 0 || px2 >= static_cast<int32>(w) ||
                    py2 < 0 || py2 >= static_cast<int32>(h)) {
                    continue;
                }
                
                float64 v1 = img1[py1 * w + px1];
                float64 v2 = img2[py2 * w + px2];
                
                sum1 += v1;
                sum2 += v2;
                sumSq1 += v1 * v1;
                sumSq2 += v2 * v2;
                sumProd += v1 * v2;
                count++;
            }
        }
        
        if (count < 10) return 0;
        
        float64 mean1 = sum1 / count;
        float64 mean2 = sum2 / count;
        float64 var1 = sumSq1 / count - mean1 * mean1;
        float64 var2 = sumSq2 / count - mean2 * mean2;
        
        if (var1 < 1.0 || var2 < 1.0) return 0;
        
        float64 ncc = (sumProd / count - mean1 * mean2) / sqrt(var1 * var2);
        return ncc;
    }

private:
    void allocateBuffers() {
        uint32 levels = config_.numPyramidLevels;
        
        pyramidPrev_ = new uint8*[levels];
        pyramidCurr_ = new uint8*[levels];
        gradX_ = new float64*[levels];
        gradY_ = new float64*[levels];
        pyramidWidth_ = new uint32[levels];
        pyramidHeight_ = new uint32[levels];
        
        for (uint32 i = 0; i < levels; ++i) {
            pyramidWidth_[i] = width_ >> i;
            pyramidHeight_[i] = height_ >> i;
            uint32 size = pyramidWidth_[i] * pyramidHeight_[i];
            
            pyramidPrev_[i] = new uint8[size];
            pyramidCurr_[i] = new uint8[size];
            gradX_[i] = new float64[size];
            gradY_[i] = new float64[size];
        }
        
        windowBuffer_ = new float64[config_.windowSize * config_.windowSize];
    }
    
    void releaseBuffers() {
        if (pyramidPrev_) {
            for (uint32 i = 0; i < config_.numPyramidLevels; ++i) {
                if (pyramidPrev_[i]) delete[] pyramidPrev_[i];
                if (pyramidCurr_[i]) delete[] pyramidCurr_[i];
                if (gradX_[i]) delete[] gradX_[i];
                if (gradY_[i]) delete[] gradY_[i];
            }
            delete[] pyramidPrev_;
            delete[] pyramidCurr_;
            delete[] gradX_;
            delete[] gradY_;
            delete[] pyramidWidth_;
            delete[] pyramidHeight_;
        }
        if (windowBuffer_) delete[] windowBuffer_;
    }
    
    void buildPyramid(const uint8* image, uint8** pyramid, 
                      float64** gradX, float64** gradY) {
        memcpy(pyramid[0], image, width_ * height_);
        
        if (gradX && gradY) {
            computeGradients(image, gradX[0], gradY[0], width_, height_);
        }
        
        for (uint32 level = 1; level < config_.numPyramidLevels; ++level) {
            uint32 srcW = pyramidWidth_[level - 1];
            uint32 srcH = pyramidHeight_[level - 1];
            uint32 dstW = pyramidWidth_[level];
            uint32 dstH = pyramidHeight_[level];
            
            for (uint32 y = 0; y < dstH; ++y) {
                for (uint32 x = 0; x < dstW; ++x) {
                    uint32 sx = x * 2;
                    uint32 sy = y * 2;
                    
                    float64 sum = 0;
                    sum += pyramid[level - 1][sy * srcW + sx];
                    sum += pyramid[level - 1][sy * srcW + sx + 1];
                    sum += pyramid[level - 1][(sy + 1) * srcW + sx];
                    sum += pyramid[level - 1][(sy + 1) * srcW + sx + 1];
                    
                    pyramid[level][y * dstW + x] = static_cast<uint8>(sum / 4.0);
                }
            }
            
            if (gradX && gradY) {
                computeGradients(pyramid[level], gradX[level], gradY[level], dstW, dstH);
            }
        }
    }
    
    void computeGradients(const uint8* image, float64* gradX, float64* gradY,
                          uint32 w, uint32 h) {
        for (uint32 y = 1; y < h - 1; ++y) {
            for (uint32 x = 1; x < w - 1; ++x) {
                uint32 idx = y * w + x;
                gradX[idx] = static_cast<float64>(image[idx + 1]) - 
                            static_cast<float64>(image[idx - 1]);
                gradY[idx] = static_cast<float64>(image[idx + w]) - 
                            static_cast<float64>(image[idx - w]);
            }
        }
    }
    
    void trackSingle(const Feature2D& prevFeature, TrackResult& result) {
        result.prevFeature = prevFeature;
        result.success = false;
        
        float64 x = prevFeature.x;
        float64 y = prevFeature.y;
        
        float64 dx = 0, dy = 0;
        
        for (int32 level = config_.numPyramidLevels - 1; level >= 0; --level) {
            float64 scale = static_cast<float64>(1 << level);
            float64 xLevel = x / scale;
            float64 yLevel = y / scale;
            float64 dxLevel = dx / scale;
            float64 dyLevel = dy / scale;
            
            if (!trackAtLevel(level, xLevel, yLevel, dxLevel, dyLevel)) {
                return;
            }
            
            dx = dxLevel * scale;
            dy = dyLevel * scale;
        }
        
        result.currFeature.x = x + dx;
        result.currFeature.y = y + dy;
        result.currFeature.id = prevFeature.id;
        result.currFeature.lifetime = prevFeature.lifetime + 1;
        result.error = sqrt(dx * dx + dy * dy);
        
        if (result.error > config_.maxDisplacement) {
            return;
        }
        
        int32 ix = static_cast<int32>(result.currFeature.x);
        int32 iy = static_cast<int32>(result.currFeature.y);
        if (ix < 0 || ix >= static_cast<int32>(width_) ||
            iy < 0 || iy >= static_cast<int32>(height_)) {
            return;
        }
        
        result.success = true;
    }
    
    bool trackAtLevel(uint32 level, float64 x, float64 y, 
                      float64& dx, float64& dy) {
        uint32 w = pyramidWidth_[level];
        uint32 h = pyramidHeight_[level];
        uint8* imgPrev = pyramidPrev_[level];
        uint8* imgCurr = pyramidCurr_[level];
        float64* gx = gradX_[level];
        float64* gy = gradY_[level];
        
        int32 halfWin = config_.windowSize / 2;
        
        float64 sumIx2 = 0, sumIy2 = 0, sumIxIy = 0;
        
        for (int32 dyWin = -halfWin; dyWin <= halfWin; ++dyWin) {
            for (int32 dxWin = -halfWin; dxWin <= halfWin; ++dxWin) {
                int32 px = static_cast<int32>(x) + dxWin;
                int32 py = static_cast<int32>(y) + dyWin;
                
                if (px < 0 || px >= static_cast<int32>(w) ||
                    py < 0 || py >= static_cast<int32>(h)) {
                    continue;
                }
                
                uint32 idx = py * w + px;
                sumIx2 += gx[idx] * gx[idx];
                sumIy2 += gy[idx] * gy[idx];
                sumIxIy += gx[idx] * gy[idx];
            }
        }
        
        float64 det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
        if (det < 1e-6) {
            return false;
        }
        
        for (uint32 iter = 0; iter < config_.maxIterations; ++iter) {
            float64 sumIxIt = 0, sumIyIt = 0;
            
            for (int32 dyWin = -halfWin; dyWin <= halfWin; ++dyWin) {
                for (int32 dxWin = -halfWin; dxWin <= halfWin; ++dxWin) {
                    int32 pxPrev = static_cast<int32>(x) + dxWin;
                    int32 pyPrev = static_cast<int32>(y) + dyWin;
                    
                    float64 pxCurr = x + dx + dxWin;
                    float64 pyCurr = y + dy + dyWin;
                    
                    if (pxPrev < 0 || pxPrev >= static_cast<int32>(w) ||
                        pyPrev < 0 || pyPrev >= static_cast<int32>(h) ||
                        pxCurr < 0 || pxCurr >= w ||
                        pyCurr < 0 || pyCurr >= h) {
                        continue;
                    }
                    
                    uint32 idxPrev = pyPrev * w + pxPrev;
                    float64 Iprev = imgPrev[idxPrev];
                    float64 Icurr = bilinearInterpolate(imgCurr, pxCurr, pyCurr, w);
                    
                    float64 It = Iprev - Icurr;
                    
                    sumIxIt += gx[idxPrev] * It;
                    sumIyIt += gy[idxPrev] * It;
                }
            }
            
            float64 deltaDx = (sumIy2 * sumIxIt - sumIxIy * sumIyIt) / det;
            float64 deltaDy = (sumIx2 * sumIyIt - sumIxIy * sumIxIt) / det;
            
            dx += deltaDx;
            dy += deltaDy;
            
            if (sqrt(deltaDx * deltaDx + deltaDy * deltaDy) < config_.convergenceThreshold) {
                break;
            }
        }
        
        return true;
    }
    
    void validateBidirectional(TrackResult& result) {
        float64 x = result.currFeature.x;
        float64 y = result.currFeature.y;
        
        float64 dx = 0, dy = 0;
        
        for (int32 level = config_.numPyramidLevels - 1; level >= 0; --level) {
            float64 scale = static_cast<float64>(1 << level);
            float64 xLevel = x / scale;
            float64 yLevel = y / scale;
            float64 dxLevel = dx / scale;
            float64 dyLevel = dy / scale;
            
            trackBackwardAtLevel(level, xLevel, yLevel, dxLevel, dyLevel);
            
            dx = dxLevel * scale;
            dy = dyLevel * scale;
        }
        
        float64 fwdX = result.currFeature.x;
        float64 fwdY = result.currFeature.y;
        float64 bwdX = x + dx;
        float64 bwdY = y + dy;
        
        float64 error = sqrt((fwdX - bwdX) * (fwdX - bwdX) + 
                            (fwdY - bwdY) * (fwdY - bwdY));
        
        if (error > config_.bidirectionalThreshold) {
            result.success = false;
        }
    }
    
    bool trackBackwardAtLevel(uint32 level, float64 x, float64 y,
                              float64& dx, float64& dy) {
        uint32 w = pyramidWidth_[level];
        uint32 h = pyramidHeight_[level];
        uint8* imgPrev = pyramidPrev_[level];
        uint8* imgCurr = pyramidCurr_[level];
        float64* gx = gradX_[level];
        float64* gy = gradY_[level];
        
        int32 halfWin = config_.windowSize / 2;
        
        float64 sumIx2 = 0, sumIy2 = 0, sumIxIy = 0;
        
        for (int32 dyWin = -halfWin; dyWin <= halfWin; ++dyWin) {
            for (int32 dxWin = -halfWin; dxWin <= halfWin; ++dxWin) {
                int32 px = static_cast<int32>(x) + dxWin;
                int32 py = static_cast<int32>(y) + dyWin;
                
                if (px < 0 || px >= static_cast<int32>(w) ||
                    py < 0 || py >= static_cast<int32>(h)) {
                    continue;
                }
                
                uint32 idx = py * w + px;
                sumIx2 += gx[idx] * gx[idx];
                sumIy2 += gy[idx] * gy[idx];
                sumIxIy += gx[idx] * gy[idx];
            }
        }
        
        float64 det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
        if (det < 1e-6) {
            return false;
        }
        
        for (uint32 iter = 0; iter < config_.maxIterations; ++iter) {
            float64 sumIxIt = 0, sumIyIt = 0;
            
            for (int32 dyWin = -halfWin; dyWin <= halfWin; ++dyWin) {
                for (int32 dxWin = -halfWin; dxWin <= halfWin; ++dxWin) {
                    int32 pxCurr = static_cast<int32>(x) + dxWin;
                    int32 pyCurr = static_cast<int32>(y) + dyWin;
                    
                    float64 pxPrev = x + dx + dxWin;
                    float64 pyPrev = y + dy + dyWin;
                    
                    if (pxCurr < 0 || pxCurr >= static_cast<int32>(w) ||
                        pyCurr < 0 || pyCurr >= static_cast<int32>(h) ||
                        pxPrev < 0 || pxPrev >= w ||
                        pyPrev < 0 || pyPrev >= h) {
                        continue;
                    }
                    
                    uint32 idxCurr = pyCurr * w + pxCurr;
                    float64 Icurr = imgCurr[idxCurr];
                    float64 Iprev = bilinearInterpolate(imgPrev, pxPrev, pyPrev, w);
                    
                    float64 It = Icurr - Iprev;
                    
                    sumIxIt += gx[idxCurr] * It;
                    sumIyIt += gy[idxCurr] * It;
                }
            }
            
            float64 deltaDx = (sumIy2 * sumIxIt - sumIxIy * sumIyIt) / det;
            float64 deltaDy = (sumIx2 * sumIyIt - sumIxIy * sumIxIt) / det;
            
            dx += deltaDx;
            dy += deltaDy;
            
            if (sqrt(deltaDx * deltaDx + deltaDy * deltaDy) < config_.convergenceThreshold) {
                break;
            }
        }
        
        return true;
    }
    
    float64 bilinearInterpolate(const uint8* img, float64 x, float64 y, uint32 w) const {
        int32 x0 = static_cast<int32>(x);
        int32 y0 = static_cast<int32>(y);
        int32 x1 = x0 + 1;
        int32 y1 = y0 + 1;
        
        float64 dx = x - x0;
        float64 dy = y - y0;
        
        float64 v00 = img[y0 * w + x0];
        float64 v01 = img[y0 * w + x1];
        float64 v10 = img[y1 * w + x0];
        float64 v11 = img[y1 * w + x1];
        
        return v00 * (1 - dx) * (1 - dy) +
               v01 * dx * (1 - dy) +
               v10 * (1 - dx) * dy +
               v11 * dx * dy;
    }
};

}

#endif
