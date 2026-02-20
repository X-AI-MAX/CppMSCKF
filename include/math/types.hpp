#ifndef MSCKF_MATH_TYPES_HPP
#define MSCKF_MATH_TYPES_HPP

#include <cstdint>
#include <cmath>

namespace msckf {

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

using float32 = float;
using float64 = double;

constexpr float32 EPSILON_F32 = 1e-6f;
constexpr float64 EPSILON_F64 = 1e-12;
constexpr float64 PI = 3.14159265358979323846;
constexpr float64 GRAVITY = 9.80665;

template<typename T>
inline constexpr T epsilon() { return T{}; }

template<>
inline constexpr float32 epsilon<float32>() { return EPSILON_F32; }

template<>
inline constexpr float64 epsilon<float64>() { return EPSILON_F64; }

template<typename T>
inline T abs(T x) { return x >= T(0) ? x : -x; }

template<typename T>
inline T sqrt(T x) { return std::sqrt(x); }

template<typename T>
inline T sin(T x) { return std::sin(x); }

template<typename T>
inline T cos(T x) { return std::cos(x); }

template<typename T>
inline T tan(T x) { return std::tan(x); }

template<typename T>
inline T asin(T x) { return std::asin(x); }

template<typename T>
inline T acos(T x) { return std::acos(x); }

template<typename T>
inline T atan2(T y, T x) { return std::atan2(y, x); }

template<typename T>
inline T exp(T x) { return std::exp(x); }

template<typename T>
inline T log(T x) { return std::log(x); }

template<typename T>
inline T pow(T base, T exp) { return std::pow(base, exp); }

template<typename T>
inline T floor(T x) { return std::floor(x); }

template<typename T>
inline T ceil(T x) { return std::ceil(x); }

template<typename T>
inline T min(T a, T b) { return a < b ? a : b; }

template<typename T>
inline T max(T a, T b) { return a > b ? a : b; }

template<typename T>
inline T clamp(T val, T lo, T hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

template<typename T>
inline T sign(T x) {
    if (x > T(0)) return T(1);
    if (x < T(0)) return T(-1);
    return T(0);
}

template<typename T>
inline bool nearZero(T x, T tol = epsilon<T>()) {
    return abs(x) < tol;
}

template<typename T>
inline T normalizeAngle(T angle) {
    while (angle > T(PI)) angle -= T(2 * PI);
    while (angle < T(-PI)) angle += T(2 * PI);
    return angle;
}

template<typename T>
inline T normalizeAngleFast(T angle) {
    T twoPi = T(2 * PI);
    angle = angle - twoPi * floor((angle + T(PI)) / twoPi);
    return angle;
}

template<typename T>
inline T angleDifference(T a, T b) {
    T diff = a - b;
    return normalizeAngle(diff);
}

template<typename T>
inline T angleLerp(T a, T b, T t) {
    T diff = angleDifference(b, a);
    return a + diff * t;
}

template<typename T>
inline bool angleInRange(T angle, T lo, T hi) {
    angle = normalizeAngle(angle);
    lo = normalizeAngle(lo);
    hi = normalizeAngle(hi);
    if (lo <= hi) {
        return angle >= lo && angle <= hi;
    } else {
        return angle >= lo || angle <= hi;
    }
}

}

#endif
