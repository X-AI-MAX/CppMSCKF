#ifndef MSCKF_MATH_FIXED_POINT_HPP
#define MSCKF_MATH_FIXED_POINT_HPP

#include "types.hpp"
#include <cstdint>

namespace msckf {

template<int32_t FractionalBits>
class FixedPoint {
public:
    static constexpr int32_t FB = FractionalBits;
    static constexpr int64_t ONE = (1LL << FB);
    static constexpr int64_t MASK = (ONE - 1);
    static constexpr int64_t MAX_VAL = (1LL << (31 - FB)) - 1;
    static constexpr int64_t MIN_VAL = -(1LL << (31 - FB));
    
    int32_t raw_;
    
    FixedPoint() : raw_(0) {}
    
    explicit FixedPoint(int32_t raw, bool) : raw_(raw) {}
    
    FixedPoint(float32 f) {
        raw_ = static_cast<int32_t>(f * ONE + (f >= 0 ? 0.5f : -0.5f));
    }
    
    FixedPoint(float64 d) {
        raw_ = static_cast<int32_t>(d * ONE + (d >= 0 ? 0.5 : -0.5));
    }
    
    static FixedPoint fromRaw(int32_t raw) {
        return FixedPoint(raw, true);
    }
    
    float32 toFloat32() const {
        return static_cast<float32>(raw_) / ONE;
    }
    
    float64 toFloat64() const {
        return static_cast<float64>(raw_) / ONE;
    }
    
    int32_t toInt32() const {
        return raw_ >> FB;
    }
    
    int32_t raw() const { return raw_; }
    
    FixedPoint operator+(const FixedPoint& rhs) const {
        return fromRaw(saturateAdd(raw_, rhs.raw_));
    }
    
    FixedPoint operator-(const FixedPoint& rhs) const {
        return fromRaw(saturateSub(raw_, rhs.raw_));
    }
    
    FixedPoint operator*(const FixedPoint& rhs) const {
        int64_t result = static_cast<int64_t>(raw_) * rhs.raw_;
        result = (result + (1LL << (FB - 1))) >> FB;
        return fromRaw(saturate(result));
    }
    
    FixedPoint operator/(const FixedPoint& rhs) const {
        if (rhs.raw_ == 0) {
            return fromRaw(raw_ >= 0 ? MAX_VAL : MIN_VAL);
        }
        int64_t temp = static_cast<int64_t>(raw_) << FB;
        int64_t result = temp / rhs.raw_;
        return fromRaw(saturate(result));
    }
    
    FixedPoint operator-() const {
        return fromRaw(-raw_);
    }
    
    FixedPoint& operator+=(const FixedPoint& rhs) {
        raw_ = saturateAdd(raw_, rhs.raw_);
        return *this;
    }
    
    FixedPoint& operator-=(const FixedPoint& rhs) {
        raw_ = saturateSub(raw_, rhs.raw_);
        return *this;
    }
    
    FixedPoint& operator*=(const FixedPoint& rhs) {
        int64_t result = static_cast<int64_t>(raw_) * rhs.raw_;
        result = (result + (1LL << (FB - 1))) >> FB;
        raw_ = saturate(result);
        return *this;
    }
    
    FixedPoint& operator/=(const FixedPoint& rhs) {
        if (rhs.raw_ == 0) {
            raw_ = raw_ >= 0 ? MAX_VAL : MIN_VAL;
        } else {
            int64_t temp = static_cast<int64_t>(raw_) << FB;
            raw_ = saturate(temp / rhs.raw_);
        }
        return *this;
    }
    
    bool operator==(const FixedPoint& rhs) const { return raw_ == rhs.raw_; }
    bool operator!=(const FixedPoint& rhs) const { return raw_ != rhs.raw_; }
    bool operator<(const FixedPoint& rhs) const { return raw_ < rhs.raw_; }
    bool operator<=(const FixedPoint& rhs) const { return raw_ <= rhs.raw_; }
    bool operator>(const FixedPoint& rhs) const { return raw_ > rhs.raw_; }
    bool operator>=(const FixedPoint& rhs) const { return raw_ >= rhs.raw_; }
    
    FixedPoint abs() const {
        return fromRaw(raw_ >= 0 ? raw_ : -raw_);
    }
    
    FixedPoint sqrt() const {
        if (raw_ <= 0) return FixedPoint();
        
        int32_t x = raw_;
        int32_t result = 0;
        int32_t bit = 1 << 30;
        
        while (bit > x) {
            bit >>= 2;
        }
        
        while (bit != 0) {
            if (x >= result + bit) {
                x -= result + bit;
                result = (result >> 1) + bit;
            } else {
                result >>= 1;
            }
            bit >>= 2;
        }
        
        return fromRaw(result << (FB / 2));
    }
    
    FixedPoint sin() const {
        FixedPoint x = *this;
        while (x.raw_ > FixedPoint(3.14159265f).raw_) {
            x -= FixedPoint(3.14159265f * 2);
        }
        while (x.raw_ < FixedPoint(-3.14159265f).raw_) {
            x += FixedPoint(3.14159265f * 2);
        }
        
        FixedPoint x2 = x * x;
        FixedPoint result = x;
        FixedPoint term = x;
        
        term = term * x2 / FixedPoint(6.0f);
        result = result - term;
        
        term = term * x2 / FixedPoint(20.0f);
        result = result + term;
        
        term = term * x2 / FixedPoint(42.0f);
        result = result - term;
        
        return result;
    }
    
    FixedPoint cos() const {
        return (*this + FixedPoint(1.57079632f)).sin();
    }

private:
    static int32_t saturate(int64_t val) {
        if (val > MAX_VAL) return MAX_VAL;
        if (val < MIN_VAL) return MIN_VAL;
        return static_cast<int32_t>(val);
    }
    
    static int32_t saturateAdd(int32_t a, int32_t b) {
        int64_t result = static_cast<int64_t>(a) + b;
        return saturate(result);
    }
    
    static int32_t saturateSub(int32_t a, int32_t b) {
        int64_t result = static_cast<int64_t>(a) - b;
        return saturate(result);
    }
};

using Q8_8 = FixedPoint<8>;
using Q16_16 = FixedPoint<16>;
using Q15_16 = FixedPoint<16>;

template<typename FP>
class FixedPointVector3 {
public:
    FP x, y, z;
    
    FixedPointVector3() : x(0), y(0), z(0) {}
    
    FixedPointVector3(FP x_, FP y_, FP z_) : x(x_), y(y_), z(z_) {}
    
    FixedPointVector3(float32 x_, float32 y_, float32 z_)
        : x(x_), y(y_), z(z_) {}
    
    FixedPointVector3 operator+(const FixedPointVector3& rhs) const {
        return FixedPointVector3(x + rhs.x, y + rhs.y, z + rhs.z);
    }
    
    FixedPointVector3 operator-(const FixedPointVector3& rhs) const {
        return FixedPointVector3(x - rhs.x, y - rhs.y, z - rhs.z);
    }
    
    FixedPointVector3 operator*(const FP& scalar) const {
        return FixedPointVector3(x * scalar, y * scalar, z * scalar);
    }
    
    FixedPointVector3 operator/(const FP& scalar) const {
        return FixedPointVector3(x / scalar, y / scalar, z / scalar);
    }
    
    FP dot(const FixedPointVector3& rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }
    
    FixedPointVector3 cross(const FixedPointVector3& rhs) const {
        return FixedPointVector3(
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        );
    }
    
    FP normSquared() const {
        return x * x + y * y + z * z;
    }
    
    FP norm() const {
        return normSquared().sqrt();
    }
    
    FixedPointVector3 normalized() const {
        FP n = norm();
        if (n.raw() == 0) return FixedPointVector3();
        return *this / n;
    }
};

template<typename FP>
class FixedPointMatrix3x3 {
public:
    FP m[3][3];
    
    FixedPointMatrix3x3() {
        for (int32_t i = 0; i < 3; ++i) {
            for (int32_t j = 0; j < 3; ++j) {
                m[i][j] = FP(0);
            }
        }
    }
    
    static FixedPointMatrix3x3 identity() {
        FixedPointMatrix3x3 result;
        result.m[0][0] = FP(1);
        result.m[1][1] = FP(1);
        result.m[2][2] = FP(1);
        return result;
    }
    
    FixedPointMatrix3x3 operator*(const FixedPointMatrix3x3& rhs) const {
        FixedPointMatrix3x3 result;
        for (int32_t i = 0; i < 3; ++i) {
            for (int32_t j = 0; j < 3; ++j) {
                FP sum(0);
                for (int32_t k = 0; k < 3; ++k) {
                    sum += m[i][k] * rhs.m[k][j];
                }
                result.m[i][j] = sum;
            }
        }
        return result;
    }
    
    FixedPointVector3<FP> operator*(const FixedPointVector3<FP>& v) const {
        return FixedPointVector3<FP>(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }
};

class FixedPointImageGradient {
public:
    using GradientType = Q8_8;
    
    static constexpr int32_t GRADIENT_SCALE = 256;
    
    struct GradientResult {
        GradientType gx;
        GradientType gy;
    };

private:
    int32_t width_;
    int32_t height_;
    GradientType* gradX_;
    GradientType* gradY_;

public:
    FixedPointImageGradient() 
        : width_(0), height_(0), gradX_(nullptr), gradY_(nullptr) {}
    
    ~FixedPointImageGradient() {
        release();
    }
    
    void init(int32_t width, int32_t height) {
        width_ = width;
        height_ = height;
        gradX_ = new GradientType[width * height];
        gradY_ = new GradientType[width * height];
    }
    
    void release() {
        if (gradX_) { delete[] gradX_; gradX_ = nullptr; }
        if (gradY_) { delete[] gradY_; gradY_ = nullptr; }
    }
    
    void computeGradients(const uint8_t* image) {
        for (int32_t y = 1; y < height_ - 1; ++y) {
            for (int32_t x = 1; x < width_ - 1; ++x) {
                int32_t idx = y * width_ + x;
                
                int32_t gx = static_cast<int32_t>(image[idx + 1]) - 
                            static_cast<int32_t>(image[idx - 1]);
                int32_t gy = static_cast<int32_t>(image[idx + width_]) - 
                            static_cast<int32_t>(image[idx - width_]);
                
                gradX_[idx] = GradientType::fromRaw(gx);
                gradY_[idx] = GradientType::fromRaw(gy);
            }
        }
        
        for (int32_t x = 0; x < width_; ++x) {
            gradX_[x] = GradientType(0);
            gradY_[x] = GradientType(0);
            gradX_[(height_ - 1) * width_ + x] = GradientType(0);
            gradY_[(height_ - 1) * width_ + x] = GradientType(0);
        }
        for (int32_t y = 0; y < height_; ++y) {
            gradX_[y * width_] = GradientType(0);
            gradY_[y * width_] = GradientType(0);
            gradX_[y * width_ + width_ - 1] = GradientType(0);
            gradY_[y * width_ + width_ - 1] = GradientType(0);
        }
    }
    
    GradientResult getGradient(int32_t x, int32_t y) const {
        GradientResult result;
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            int32_t idx = y * width_ + x;
            result.gx = gradX_[idx];
            result.gy = gradY_[idx];
        }
        return result;
    }
    
    const GradientType* getGradX() const { return gradX_; }
    const GradientType* getGradY() const { return gradY_; }
};

class FixedPointKLT {
public:
    using FixedType = Q16_16;
    using FixedVec2 = FixedPointVector3<FixedType>;
    
    struct Config {
        int32_t windowSize;
        int32_t maxIterations;
        FixedType convergenceThreshold;
        FixedType maxDisplacement;
        int32_t numPyramidLevels;
        
        Config() {
            windowSize = 21;
            maxIterations = 30;
            convergenceThreshold = FixedType(0.01f);
            maxDisplacement = FixedType(50.0f);
            numPyramidLevels = 3;
        }
    };

private:
    Config config_;
    int32_t width_;
    int32_t height_;
    
    uint8_t** pyramidPrev_;
    uint8_t** pyramidCurr_;
    FixedType** gradX_;
    FixedType** gradY_;
    int32_t* pyramidWidth_;
    int32_t* pyramidHeight_;

public:
    FixedPointKLT() 
        : width_(0), height_(0)
        , pyramidPrev_(nullptr), pyramidCurr_(nullptr)
        , gradX_(nullptr), gradY_(nullptr)
        , pyramidWidth_(nullptr), pyramidHeight_(nullptr) {}
    
    ~FixedPointKLT() {
        release();
    }
    
    void init(int32_t width, int32_t height, const Config& config = Config()) {
        config_ = config;
        width_ = width;
        height_ = height;
        allocateBuffers();
    }
    
    bool trackPoint(FixedType prevX, FixedType prevY,
                   FixedType& currX, FixedType& currY) {
        FixedType dx(0), dy(0);
        
        for (int32_t level = config_.numPyramidLevels - 1; level >= 0; --level) {
            int32_t scale = 1 << level;
            FixedType scaleFP(scale);
            
            FixedType xLevel = prevX / scaleFP;
            FixedType yLevel = prevY / scaleFP;
            FixedType dxLevel = dx / scaleFP;
            FixedType dyLevel = dy / scaleFP;
            
            if (!trackAtLevel(level, xLevel, yLevel, dxLevel, dyLevel)) {
                return false;
            }
            
            dx = dxLevel * scaleFP;
            dy = dyLevel * scaleFP;
        }
        
        currX = prevX + dx;
        currY = prevY + dy;
        
        FixedType displacement = (dx * dx + dy * dy).sqrt();
        return displacement < config_.maxDisplacement;
    }

private:
    void allocateBuffers() {
        int32_t levels = config_.numPyramidLevels;
        
        pyramidPrev_ = new uint8_t*[levels];
        pyramidCurr_ = new uint8_t*[levels];
        gradX_ = new FixedType*[levels];
        gradY_ = new FixedType*[levels];
        pyramidWidth_ = new int32_t[levels];
        pyramidHeight_ = new int32_t[levels];
        
        for (int32_t i = 0; i < levels; ++i) {
            pyramidWidth_[i] = width_ >> i;
            pyramidHeight_[i] = height_ >> i;
            int32_t size = pyramidWidth_[i] * pyramidHeight_[i];
            
            pyramidPrev_[i] = new uint8_t[size];
            pyramidCurr_[i] = new uint8_t[size];
            gradX_[i] = new FixedType[size];
            gradY_[i] = new FixedType[size];
        }
    }
    
    void release() {
        if (pyramidPrev_) {
            for (int32_t i = 0; i < config_.numPyramidLevels; ++i) {
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
    }
    
    bool trackAtLevel(int32_t level, FixedType x, FixedType y,
                     FixedType& dx, FixedType& dy) {
        int32_t w = pyramidWidth_[level];
        int32_t h = pyramidHeight_[level];
        uint8_t* imgPrev = pyramidPrev_[level];
        uint8_t* imgCurr = pyramidCurr_[level];
        FixedType* gx = gradX_[level];
        FixedType* gy = gradY_[level];
        
        int32_t halfWin = config_.windowSize / 2;
        
        FixedType sumIx2(0), sumIy2(0), sumIxIy(0);
        
        int32_t xInt = x.toInt32();
        int32_t yInt = y.toInt32();
        
        for (int32_t dyWin = -halfWin; dyWin <= halfWin; ++dyWin) {
            for (int32_t dxWin = -halfWin; dxWin <= halfWin; ++dxWin) {
                int32_t px = xInt + dxWin;
                int32_t py = yInt + dyWin;
                
                if (px < 0 || px >= w || py < 0 || py >= h) continue;
                
                int32_t idx = py * w + px;
                sumIx2 += gx[idx] * gx[idx];
                sumIy2 += gy[idx] * gy[idx];
                sumIxIy += gx[idx] * gy[idx];
            }
        }
        
        FixedType det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
        if (det < FixedType(0.0001f)) {
            return false;
        }
        
        for (int32_t iter = 0; iter < config_.maxIterations; ++iter) {
            FixedType sumIxIt(0), sumIyIt(0);
            
            FixedType currXFP = x + dx;
            FixedType currYFP = y + dy;
            int32_t currXInt = currXFP.toInt32();
            int32_t currYInt = currYFP.toInt32();
            
            for (int32_t dyWin = -halfWin; dyWin <= halfWin; ++dyWin) {
                for (int32_t dxWin = -halfWin; dxWin <= halfWin; ++dxWin) {
                    int32_t pxPrev = xInt + dxWin;
                    int32_t pyPrev = yInt + dyWin;
                    int32_t pxCurr = currXInt + dxWin;
                    int32_t pyCurr = currYInt + dyWin;
                    
                    if (pxPrev < 0 || pxPrev >= w || pyPrev < 0 || pyPrev >= h) continue;
                    if (pxCurr < 0 || pxCurr >= w || pyCurr < 0 || pyCurr >= h) continue;
                    
                    int32_t idxPrev = pyPrev * w + pxPrev;
                    int32_t idxCurr = pyCurr * w + pxCurr;
                    
                    FixedType Iprev(static_cast<float32>(imgPrev[idxPrev]));
                    FixedType Icurr(static_cast<float32>(imgCurr[idxCurr]));
                    
                    FixedType It = Iprev - Icurr;
                    
                    sumIxIt += gx[idxPrev] * It;
                    sumIyIt += gy[idxPrev] * It;
                }
            }
            
            FixedType deltaDx = (sumIy2 * sumIxIt - sumIxIy * sumIyIt) / det;
            FixedType deltaDy = (sumIx2 * sumIyIt - sumIxIy * sumIxIt) / det;
            
            dx += deltaDx;
            dy += deltaDy;
            
            FixedType delta = (deltaDx * deltaDx + deltaDy * deltaDy).sqrt();
            if (delta < config_.convergenceThreshold) {
                break;
            }
        }
        
        return true;
    }
};

class NCCLookupTable {
public:
    static constexpr int32_t TABLE_SIZE = 256;
    static constexpr int32_t INV_TABLE_SIZE_SHIFT = 8;

private:
    uint16_t sqrtTable_[TABLE_SIZE * TABLE_SIZE];
    uint16_t invSqrtTable_[TABLE_SIZE];
    int16_t cosTable_[TABLE_SIZE * 2];
    
    bool initialized_;

public:
    NCCLookupTable() : initialized_(false) {
        init();
    }
    
    void init() {
        for (int32_t i = 0; i < TABLE_SIZE * TABLE_SIZE; ++i) {
            float64 val = sqrt(static_cast<float64>(i));
            sqrtTable_[i] = static_cast<uint16_t>(val * 256.0 + 0.5);
        }
        
        for (int32_t i = 1; i < TABLE_SIZE; ++i) {
            float64 val = 1.0 / sqrt(static_cast<float64>(i) / TABLE_SIZE);
            invSqrtTable_[i] = static_cast<uint16_t>(val * 256.0 + 0.5);
        }
        invSqrtTable_[0] = 0;
        
        for (int32_t i = 0; i < TABLE_SIZE * 2; ++i) {
            float64 angle = static_cast<float64>(i - TABLE_SIZE) / TABLE_SIZE * PI;
            float64 val = cos(angle);
            cosTable_[i] = static_cast<int16_t>(val * 32767.0);
        }
        
        initialized_ = true;
    }
    
    uint16_t sqrtLookup(uint16_t value) const {
        if (value < TABLE_SIZE * TABLE_SIZE) {
            return sqrtTable_[value];
        }
        return static_cast<uint16_t>(sqrt(static_cast<float64>(value)) * 256.0);
    }
    
    uint16_t invSqrtLookup(uint8_t value) const {
        return invSqrtTable_[value];
    }
    
    int16_t cosLookup(int16_t normalizedAngle) const {
        int32_t idx = normalizedAngle + TABLE_SIZE;
        if (idx >= 0 && idx < TABLE_SIZE * 2) {
            return cosTable_[idx];
        }
        return 32767;
    }
    
    int32_t computeNCCFast(const uint8_t* img1, const uint8_t* img2,
                          int32_t x1, int32_t y1, int32_t x2, int32_t y2,
                          int32_t width, int32_t height, int32_t windowSize) const {
        int32_t halfWin = windowSize / 2;
        
        int32_t sum1 = 0, sum2 = 0;
        int32_t sumSq1 = 0, sumSq2 = 0;
        int32_t sumProd = 0;
        int32_t count = 0;
        
        for (int32_t dy = -halfWin; dy <= halfWin; ++dy) {
            for (int32_t dx = -halfWin; dx <= halfWin; ++dx) {
                int32_t px1 = x1 + dx;
                int32_t py1 = y1 + dy;
                int32_t px2 = x2 + dx;
                int32_t py2 = y2 + dy;
                
                if (px1 < 0 || px1 >= width || py1 < 0 || py1 >= height) continue;
                if (px2 < 0 || px2 >= width || py2 < 0 || py2 >= height) continue;
                
                int32_t idx1 = py1 * width + px1;
                int32_t idx2 = py2 * width + px2;
                
                int32_t v1 = img1[idx1];
                int32_t v2 = img2[idx2];
                
                sum1 += v1;
                sum2 += v2;
                sumSq1 += v1 * v1;
                sumSq2 += v2 * v2;
                sumProd += v1 * v2;
                count++;
            }
        }
        
        if (count < 10) return 0;
        
        int32_t mean1 = sum1 / count;
        int32_t mean2 = sum2 / count;
        
        int32_t var1 = sumSq1 / count - mean1 * mean1;
        int32_t var2 = sumSq2 / count - mean2 * mean2;
        
        if (var1 < 1 || var2 < 1) return 0;
        
        int32_t cov = sumProd / count - mean1 * mean2;
        
        uint16_t sqrtVar1 = sqrtLookup(static_cast<uint16_t>(var1));
        uint16_t sqrtVar2 = sqrtLookup(static_cast<uint16_t>(var2));
        
        if (sqrtVar1 == 0 || sqrtVar2 == 0) return 0;
        
        int32_t denom = (static_cast<int32_t>(sqrtVar1) * sqrtVar2) >> 8;
        if (denom == 0) return 0;
        
        int32_t ncc = (cov << 8) / denom;
        
        return clamp(ncc, -32768, 32767);
    }
    
    bool isInitialized() const { return initialized_; }
};

class DivisionLookupTable {
public:
    static constexpr int32_t TABLE_SIZE = 1024;
    static constexpr int32_t MAX_DIVISOR = 1024;

private:
    uint16_t invTable_[TABLE_SIZE + 1];
    bool initialized_;

public:
    DivisionLookupTable() : initialized_(false) {
        init();
    }
    
    void init() {
        for (int32_t i = 1; i <= TABLE_SIZE; ++i) {
            invTable_[i] = static_cast<uint16_t>((1 << 16) / i);
        }
        invTable_[0] = 65535;
        initialized_ = true;
    }
    
    int32_t divide(int32_t numerator, int32_t divisor) const {
        if (divisor <= 0) return 0;
        if (divisor > TABLE_SIZE) {
            return numerator / divisor;
        }
        
        int64_t result = static_cast<int64_t>(numerator) * invTable_[divisor];
        return static_cast<int32_t>(result >> 16);
    }
    
    uint16_t getInverse(uint16_t divisor) const {
        if (divisor > TABLE_SIZE) return 0;
        return invTable_[divisor];
    }
    
    bool isInitialized() const { return initialized_; }
};

extern NCCLookupTable g_nccLUT;
extern DivisionLookupTable g_divLUT;

inline void initGlobalLookupTables() {
    g_nccLUT.init();
    g_divLUT.init();
}

}

#endif
