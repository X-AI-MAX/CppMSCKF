#ifndef MSCKF_VISION_IMAGE_PROCESSING_HPP
#define MSCKF_VISION_IMAGE_PROCESSING_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../hal/camera_types.hpp"

namespace msckf {

class ImageProcessor {
public:
    enum class BayerPattern {
        RGGB = 0,
        GRBG = 1,
        GBRG = 2,
        BGGR = 3
    };

private:
    CameraIntrinsics intrinsics_;
    uint32 width_;
    uint32 height_;
    
    float64* lutUndistortX_;
    float64* lutUndistortY_;
    bool lutInitialized_;

public:
    ImageProcessor() 
        : width_(0), height_(0)
        , lutUndistortX_(nullptr)
        , lutUndistortY_(nullptr)
        , lutInitialized_(false) {}
    
    ~ImageProcessor() {
        if (lutUndistortX_) delete[] lutUndistortX_;
        if (lutUndistortY_) delete[] lutUndistortY_;
    }
    
    void init(const CameraIntrinsics& intrinsics) {
        intrinsics_ = intrinsics;
        width_ = intrinsics.width;
        height_ = intrinsics.height;
        
        buildUndistortionLUT();
    }
    
    void bayerToGray(const uint8* bayer, uint8* gray, 
                     uint32 width, uint32 height, BayerPattern pattern) {
        for (uint32 y = 1; y < height - 1; ++y) {
            for (uint32 x = 1; x < width - 1; ++x) {
                uint32 idx = y * width + x;
                uint8 r, g, b;
                
                extractRGB(bayer, x, y, width, pattern, r, g, b);
                
                gray[idx] = static_cast<uint8>(
                    0.299 * r + 0.587 * g + 0.114 * b
                );
            }
        }
        
        for (uint32 x = 0; x < width; ++x) {
            gray[x] = gray[width + x];
            gray[(height - 1) * width + x] = gray[(height - 2) * width + x];
        }
        for (uint32 y = 0; y < height; ++y) {
            gray[y * width] = gray[y * width + 1];
            gray[y * width + width - 1] = gray[y * width + width - 2];
        }
    }
    
    void bayerToGrayFast(const uint8* bayer, uint8* gray,
                         uint32 width, uint32 height, BayerPattern pattern) {
        for (uint32 y = 1; y < height - 1; y += 2) {
            for (uint32 x = 1; x < width - 1; x += 2) {
                uint32 idx = y * width + x;
                uint32 idxTL = idx;
                uint32 idxTR = idx + 1;
                uint32 idxBL = idx + width;
                uint32 idxBR = idx + width + 1;
                
                uint8 g00, g01, g10, g11;
                uint8 r0, r1, b0, b1;
                
                switch (pattern) {
                    case BayerPattern::RGGB:
                        r0 = bayer[idxTL];
                        g00 = bayer[idxTR]; g01 = bayer[idxBL];
                        g10 = bayer[idxTR]; g11 = bayer[idxBR];
                        b0 = bayer[idxBR];
                        gray[idxTL] = static_cast<uint8>(0.299 * r0 + 0.587 * g00 + 0.114 * b0);
                        gray[idxTR] = static_cast<uint8>(0.299 * r0 + 0.587 * g01 + 0.114 * b0);
                        gray[idxBL] = static_cast<uint8>(0.299 * r0 + 0.587 * g10 + 0.114 * b0);
                        gray[idxBR] = static_cast<uint8>(0.299 * r0 + 0.587 * g11 + 0.114 * b0);
                        break;
                    case BayerPattern::GRBG:
                        g00 = bayer[idxTL]; g01 = bayer[idxBL];
                        r0 = bayer[idxTR];
                        b0 = bayer[idxBR];
                        gray[idxTL] = static_cast<uint8>(0.299 * r0 + 0.587 * g00 + 0.114 * b0);
                        gray[idxTR] = static_cast<uint8>(0.299 * r0 + 0.587 * g00 + 0.114 * b0);
                        gray[idxBL] = static_cast<uint8>(0.299 * r0 + 0.587 * g01 + 0.114 * b0);
                        gray[idxBR] = static_cast<uint8>(0.299 * r0 + 0.587 * g01 + 0.114 * b0);
                        break;
                    case BayerPattern::GBRG:
                        g00 = bayer[idxTL]; g01 = bayer[idxTR];
                        b0 = bayer[idxBL];
                        r0 = bayer[idxBR];
                        gray[idxTL] = static_cast<uint8>(0.299 * r0 + 0.587 * g00 + 0.114 * b0);
                        gray[idxTR] = static_cast<uint8>(0.299 * r0 + 0.587 * g01 + 0.114 * b0);
                        gray[idxBL] = static_cast<uint8>(0.299 * r0 + 0.587 * g00 + 0.114 * b0);
                        gray[idxBR] = static_cast<uint8>(0.299 * r0 + 0.587 * g01 + 0.114 * b0);
                        break;
                    case BayerPattern::BGGR:
                        b0 = bayer[idxTL];
                        g00 = bayer[idxTR]; g01 = bayer[idxBL];
                        r0 = bayer[idxBR];
                        gray[idxTL] = static_cast<uint8>(0.299 * r0 + 0.587 * g00 + 0.114 * b0);
                        gray[idxTR] = static_cast<uint8>(0.299 * r0 + 0.587 * g00 + 0.114 * b0);
                        gray[idxBL] = static_cast<uint8>(0.299 * r0 + 0.587 * g01 + 0.114 * b0);
                        gray[idxBR] = static_cast<uint8>(0.299 * r0 + 0.587 * g01 + 0.114 * b0);
                        break;
                }
            }
        }
    }
    
    void undistortPoint(float64 u, float64 v, float64& u_undist, float64& v_undist) const {
        float64 x = (u - intrinsics_.cx) / intrinsics_.fx;
        float64 y = (v - intrinsics_.cy) / intrinsics_.fy;
        
        float64 x0 = x, y0 = y;
        
        for (uint32 iter = 0; iter < 10; ++iter) {
            float64 r2 = x * x + y * y;
            float64 r4 = r2 * r2;
            float64 r6 = r4 * r2;
            
            float64 radial = 1 + intrinsics_.k1 * r2 + intrinsics_.k2 * r4 + intrinsics_.k3 * r6;
            
            float64 xy = 2 * x * y;
            float64 dx = intrinsics_.p1 * xy + intrinsics_.p2 * (r2 + 2 * x * x);
            float64 dy = intrinsics_.p2 * xy + intrinsics_.p1 * (r2 + 2 * y * y);
            
            float64 x_dist = x * radial + dx;
            float64 y_dist = y * radial + dy;
            
            float64 err_x = x_dist - x0;
            float64 err_y = y_dist - y0;
            
            float64 j11 = radial + intrinsics_.k1 * 2 * x * x + intrinsics_.k2 * 4 * x * x * r2 
                        + intrinsics_.p1 * 2 * y + intrinsics_.p2 * (6 * x);
            float64 j12 = intrinsics_.k1 * 2 * x * y + intrinsics_.k2 * 4 * x * y * r2
                        + intrinsics_.p1 * 2 * x + intrinsics_.p2 * 2 * y;
            float64 j21 = intrinsics_.k1 * 2 * x * y + intrinsics_.k2 * 4 * x * y * r2
                        + intrinsics_.p2 * 2 * y + intrinsics_.p1 * 2 * x;
            float64 j22 = radial + intrinsics_.k1 * 2 * y * y + intrinsics_.k2 * 4 * y * y * r2
                        + intrinsics_.p2 * 2 * x + intrinsics_.p1 * (6 * y);
            
            float64 det = j11 * j22 - j12 * j21;
            if (abs(det) < 1e-10) break;
            
            x -= (j22 * err_x - j12 * err_y) / det;
            y -= (-j21 * err_x + j11 * err_y) / det;
            
            if (abs(err_x) < 1e-10 && abs(err_y) < 1e-10) break;
        }
        
        u_undist = x * intrinsics_.fx + intrinsics_.cx;
        v_undist = y * intrinsics_.fy + intrinsics_.cy;
    }
    
    void distortPoint(float64 x, float64 y, float64& u_dist, float64& v_dist) const {
        float64 r2 = x * x + y * y;
        float64 r4 = r2 * r2;
        float64 r6 = r4 * r2;
        
        float64 radial = 1 + intrinsics_.k1 * r2 + intrinsics_.k2 * r4 + intrinsics_.k3 * r6;
        
        float64 xy = 2 * x * y;
        float64 dx = intrinsics_.p1 * xy + intrinsics_.p2 * (r2 + 2 * x * x);
        float64 dy = intrinsics_.p2 * xy + intrinsics_.p1 * (r2 + 2 * y * y);
        
        float64 x_dist = x * radial + dx;
        float64 y_dist = y * radial + dy;
        
        u_dist = x_dist * intrinsics_.fx + intrinsics_.cx;
        v_dist = y_dist * intrinsics_.fy + intrinsics_.cy;
    }
    
    void undistortImage(const uint8* src, uint8* dst) const {
        if (!lutInitialized_) {
            for (uint32 i = 0; i < width_ * height_; ++i) {
                dst[i] = src[i];
            }
            return;
        }
        
        for (uint32 y = 0; y < height_; ++y) {
            for (uint32 x = 0; x < width_; ++x) {
                uint32 idx = y * width_ + x;
                float64 srcX = lutUndistortX_[idx];
                float64 srcY = lutUndistortY_[idx];
                
                dst[idx] = bilinearInterpolate(src, srcX, srcY);
            }
        }
    }
    
    void undistortImageFast(const uint8* src, uint8* dst) const {
        if (!lutInitialized_) {
            for (uint32 i = 0; i < width_ * height_; ++i) {
                dst[i] = src[i];
            }
            return;
        }
        
        for (uint32 y = 0; y < height_; ++y) {
            for (uint32 x = 0; x < width_; ++x) {
                uint32 idx = y * width_ + x;
                int32 srcX = static_cast<int32>(lutUndistortX_[idx] + 0.5);
                int32 srcY = static_cast<int32>(lutUndistortY_[idx] + 0.5);
                
                if (srcX >= 0 && srcX < static_cast<int32>(width_) &&
                    srcY >= 0 && srcY < static_cast<int32>(height_)) {
                    dst[idx] = src[srcY * width_ + srcX];
                } else {
                    dst[idx] = 0;
                }
            }
        }
    }
    
    Vector3d pixelToNormalized(float64 u, float64 v) const {
        float64 u_undist, v_undist;
        undistortPoint(u, v, u_undist, v_undist);
        
        float64 x = (u_undist - intrinsics_.cx) / intrinsics_.fx;
        float64 y = (v_undist - intrinsics_.cy) / intrinsics_.fy;
        
        return Vector3d({x, y, 1.0});
    }
    
    Vector2d normalizedToPixel(const Vector3d& n) const {
        float64 x = n[0] / n[2];
        float64 y = n[1] / n[2];
        
        float64 u, v;
        distortPoint(x, y, u, v);
        
        return Vector2d({u, v});
    }
    
    Vector2d normalizedToPixelNoDistortion(const Vector3d& n) const {
        float64 x = n[0] / n[2];
        float64 y = n[1] / n[2];
        
        float64 u = x * intrinsics_.fx + intrinsics_.cx;
        float64 v = y * intrinsics_.fy + intrinsics_.cy;
        
        return Vector2d({u, v});
    }
    
    void computeUndistortionJacobian(float64 u, float64 v,
                                     Matrix<2, 2, float64>& J) const {
        float64 x = (u - intrinsics_.cx) / intrinsics_.fx;
        float64 y = (v - intrinsics_.cy) / intrinsics_.fy;
        
        float64 r2 = x * x + y * y;
        float64 r4 = r2 * r2;
        float64 r6 = r4 * r2;
        
        float64 radial = 1 + intrinsics_.k1 * r2 + intrinsics_.k2 * r4 + intrinsics_.k3 * r6;
        float64 dradial = intrinsics_.k1 * 2 * r2 + intrinsics_.k2 * 4 * r4 + intrinsics_.k3 * 6 * r6;
        
        J(0, 0) = intrinsics_.fx * (radial + x * dradial * 2 * x / intrinsics_.fx);
        J(0, 1) = intrinsics_.fx * (x * dradial * 2 * y / intrinsics_.fy);
        J(1, 0) = intrinsics_.fy * (y * dradial * 2 * x / intrinsics_.fx);
        J(1, 1) = intrinsics_.fy * (radial + y * dradial * 2 * y / intrinsics_.fy);
    }

private:
    void extractRGB(const uint8* bayer, uint32 x, uint32 y, uint32 width,
                    BayerPattern pattern, uint8& r, uint8& g, uint8& b) {
        uint32 idx = y * width + x;
        uint32 evenRow = (y & 1);
        uint32 evenCol = (x & 1);
        
        switch (pattern) {
            case BayerPattern::RGGB:
                if (!evenRow && !evenCol) {
                    r = bayer[idx];
                    g = (bayer[idx - 1] + bayer[idx + 1] + bayer[idx - width] + bayer[idx + width]) >> 2;
                    b = (bayer[idx - width - 1] + bayer[idx - width + 1] + 
                         bayer[idx + width - 1] + bayer[idx + width + 1]) >> 2;
                } else if (!evenRow && evenCol) {
                    g = bayer[idx];
                    r = (bayer[idx - 1] + bayer[idx + 1]) >> 1;
                    b = (bayer[idx - width] + bayer[idx + width]) >> 1;
                } else if (evenRow && !evenCol) {
                    g = bayer[idx];
                    r = (bayer[idx - width] + bayer[idx + width]) >> 1;
                    b = (bayer[idx - 1] + bayer[idx + 1]) >> 1;
                } else {
                    b = bayer[idx];
                    g = (bayer[idx - 1] + bayer[idx + 1] + bayer[idx - width] + bayer[idx + width]) >> 2;
                    r = (bayer[idx - width - 1] + bayer[idx - width + 1] + 
                         bayer[idx + width - 1] + bayer[idx + width + 1]) >> 2;
                }
                break;
            case BayerPattern::GRBG:
                if (!evenRow && !evenCol) {
                    g = bayer[idx];
                    r = (bayer[idx + 1] + bayer[idx - 1]) >> 1;
                    b = (bayer[idx + width] + bayer[idx - width]) >> 1;
                } else if (!evenRow && evenCol) {
                    r = bayer[idx];
                    g = (bayer[idx - 1] + bayer[idx + 1] + bayer[idx - width] + bayer[idx + width]) >> 2;
                    b = (bayer[idx - width - 1] + bayer[idx - width + 1] + 
                         bayer[idx + width - 1] + bayer[idx + width + 1]) >> 2;
                } else if (evenRow && !evenCol) {
                    b = bayer[idx];
                    g = (bayer[idx - 1] + bayer[idx + 1] + bayer[idx - width] + bayer[idx + width]) >> 2;
                    r = (bayer[idx - width - 1] + bayer[idx - width + 1] + 
                         bayer[idx + width - 1] + bayer[idx + width + 1]) >> 2;
                } else {
                    g = bayer[idx];
                    r = (bayer[idx - width] + bayer[idx + width]) >> 1;
                    b = (bayer[idx - 1] + bayer[idx + 1]) >> 1;
                }
                break;
            case BayerPattern::GBRG:
                if (!evenRow && !evenCol) {
                    g = bayer[idx];
                    r = (bayer[idx + width] + bayer[idx - width]) >> 1;
                    b = (bayer[idx + 1] + bayer[idx - 1]) >> 1;
                } else if (!evenRow && evenCol) {
                    b = bayer[idx];
                    g = (bayer[idx - 1] + bayer[idx + 1] + bayer[idx - width] + bayer[idx + width]) >> 2;
                    r = (bayer[idx - width - 1] + bayer[idx - width + 1] + 
                         bayer[idx + width - 1] + bayer[idx + width + 1]) >> 2;
                } else if (evenRow && !evenCol) {
                    r = bayer[idx];
                    g = (bayer[idx - 1] + bayer[idx + 1] + bayer[idx - width] + bayer[idx + width]) >> 2;
                    b = (bayer[idx - width - 1] + bayer[idx - width + 1] + 
                         bayer[idx + width - 1] + bayer[idx + width + 1]) >> 2;
                } else {
                    g = bayer[idx];
                    r = (bayer[idx - 1] + bayer[idx + 1]) >> 1;
                    b = (bayer[idx - width] + bayer[idx + width]) >> 1;
                }
                break;
            case BayerPattern::BGGR:
                if (!evenRow && !evenCol) {
                    b = bayer[idx];
                    g = (bayer[idx - 1] + bayer[idx + 1] + bayer[idx - width] + bayer[idx + width]) >> 2;
                    r = (bayer[idx - width - 1] + bayer[idx - width + 1] + 
                         bayer[idx + width - 1] + bayer[idx + width + 1]) >> 2;
                } else if (!evenRow && evenCol) {
                    g = bayer[idx];
                    b = (bayer[idx - 1] + bayer[idx + 1]) >> 1;
                    r = (bayer[idx - width] + bayer[idx + width]) >> 1;
                } else if (evenRow && !evenCol) {
                    g = bayer[idx];
                    b = (bayer[idx - width] + bayer[idx + width]) >> 1;
                    r = (bayer[idx - 1] + bayer[idx + 1]) >> 1;
                } else {
                    r = bayer[idx];
                    g = (bayer[idx - 1] + bayer[idx + 1] + bayer[idx - width] + bayer[idx + width]) >> 2;
                    b = (bayer[idx - width - 1] + bayer[idx - width + 1] + 
                         bayer[idx + width - 1] + bayer[idx + width + 1]) >> 2;
                }
                break;
        }
    }
    
    void buildUndistortionLUT() {
        if (lutUndistortX_) delete[] lutUndistortX_;
        if (lutUndistortY_) delete[] lutUndistortY_;
        
        uint32 size = width_ * height_;
        lutUndistortX_ = new float64[size];
        lutUndistortY_ = new float64[size];
        
        for (uint32 y = 0; y < height_; ++y) {
            for (uint32 x = 0; x < width_; ++x) {
                uint32 idx = y * width_ + x;
                float64 u_undist, v_undist;
                undistortPoint(static_cast<float64>(x), static_cast<float64>(y), 
                              u_undist, v_undist);
                lutUndistortX_[idx] = u_undist;
                lutUndistortY_[idx] = v_undist;
            }
        }
        
        lutInitialized_ = true;
    }
    
    uint8 bilinearInterpolate(const uint8* img, float64 x, float64 y) const {
        int32 x0 = static_cast<int32>(x);
        int32 y0 = static_cast<int32>(y);
        int32 x1 = x0 + 1;
        int32 y1 = y0 + 1;
        
        if (x0 < 0 || x1 >= static_cast<int32>(width_) ||
            y0 < 0 || y1 >= static_cast<int32>(height_)) {
            return 0;
        }
        
        float64 dx = x - x0;
        float64 dy = y - y0;
        
        float64 v00 = img[y0 * width_ + x0];
        float64 v01 = img[y0 * width_ + x1];
        float64 v10 = img[y1 * width_ + x0];
        float64 v11 = img[y1 * width_ + x1];
        
        float64 v = v00 * (1 - dx) * (1 - dy) +
                   v01 * dx * (1 - dy) +
                   v10 * (1 - dx) * dy +
                   v11 * dx * dy;
        
        return static_cast<uint8>(clamp(v, 0.0, 255.0));
    }
};

}

#endif
