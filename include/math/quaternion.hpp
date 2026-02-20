#ifndef MSCKF_MATH_QUATERNION_HPP
#define MSCKF_MATH_QUATERNION_HPP

#include "matrix.hpp"

namespace msckf {

template<typename T = float64>
class Quaternion {
public:
    T w, x, y, z;

    Quaternion() : w(T(1)), x(T(0)), y(T(0)), z(T(0)) {}
    
    Quaternion(T w_, T x_, T y_, T z_) : w(w_), x(x_), y(y_), z(z_) {}
    
    Quaternion(const Vector<4, T>& v) : w(v[0]), x(v[1]), y(v[2]), z(v[3]) {}
    
    static Quaternion identity() {
        return Quaternion(T(1), T(0), T(0), T(0));
    }
    
    T& operator[](uint32 i) {
        switch (i) {
            case 0: return w;
            case 1: return x;
            case 2: return y;
            default: return z;
        }
    }
    
    const T& operator[](uint32 i) const {
        switch (i) {
            case 0: return w;
            case 1: return x;
            case 2: return y;
            default: return z;
        }
    }
    
    Vector<4, T> coeffs() const {
        return Vector<4, T>({w, x, y, z});
    }
    
    Vector<3, T> vec() const {
        return Vector<3, T>({x, y, z});
    }
    
    Quaternion operator+(const Quaternion& q) const {
        return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
    }
    
    Quaternion operator-(const Quaternion& q) const {
        return Quaternion(w - q.w, x - q.x, y - q.y, z - q.z);
    }
    
    Quaternion operator*(T scalar) const {
        return Quaternion(w * scalar, x * scalar, y * scalar, z * scalar);
    }
    
    Quaternion operator/(T scalar) const {
        T inv = T(1) / scalar;
        return Quaternion(w * inv, x * inv, y * inv, z * inv);
    }
    
    Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        );
    }
    
    Quaternion& operator*=(const Quaternion& q) {
        *this = *this * q;
        return *this;
    }
    
    Quaternion& operator*=(T scalar) {
        w *= scalar; x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }
    
    T dot(const Quaternion& q) const {
        return w * q.w + x * q.x + y * q.y + z * q.z;
    }
    
    T norm() const {
        return sqrt(w * w + x * x + y * y + z * z);
    }
    
    T squaredNorm() const {
        return w * w + x * x + y * y + z * z;
    }
    
    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }
    
    Quaternion inverse() const {
        T n2 = squaredNorm();
        if (nearZero(n2)) return identity();
        T invN2 = T(1) / n2;
        return Quaternion(w * invN2, -x * invN2, -y * invN2, -z * invN2);
    }
    
    Quaternion normalized() const {
        T n = norm();
        if (nearZero(n)) return identity();
        T inv = T(1) / n;
        return Quaternion(w * inv, x * inv, y * inv, z * inv);
    }
    
    void normalize() {
        T n = norm();
        if (!nearZero(n)) {
            T inv = T(1) / n;
            w *= inv; x *= inv; y *= inv; z *= inv;
        }
    }
    
    Vector<3, T> rotate(const Vector<3, T>& v) const {
        Quaternion qv(T(0), v[0], v[1], v[2]);
        Quaternion result = *this * qv * conjugate();
        return Vector<3, T>({result.x, result.y, result.z});
    }
    
    Matrix<3, 3, T> toRotationMatrix() const {
        T wx = w * x, wy = w * y, wz = w * z;
        T xx = x * x, xy = x * y, xz = x * z;
        T yy = y * y, yz = y * z, zz = z * z;
        
        Matrix<3, 3, T> R;
        R(0, 0) = T(1) - T(2) * (yy + zz);
        R(0, 1) = T(2) * (xy - wz);
        R(0, 2) = T(2) * (xz + wy);
        R(1, 0) = T(2) * (xy + wz);
        R(1, 1) = T(1) - T(2) * (xx + zz);
        R(1, 2) = T(2) * (yz - wx);
        R(2, 0) = T(2) * (xz - wy);
        R(2, 1) = T(2) * (yz + wx);
        R(2, 2) = T(1) - T(2) * (xx + yy);
        return R;
    }
    
    static Quaternion fromRotationMatrix(const Matrix<3, 3, T>& R) {
        T trace = R(0, 0) + R(1, 1) + R(2, 2);
        Quaternion q;
        
        if (trace > T(0)) {
            T s = sqrt(trace + T(1)) * T(2);
            q.w = T(0.25) * s;
            q.x = (R(2, 1) - R(1, 2)) / s;
            q.y = (R(0, 2) - R(2, 0)) / s;
            q.z = (R(1, 0) - R(0, 1)) / s;
        } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
            T s = sqrt(T(1) + R(0, 0) - R(1, 1) - R(2, 2)) * T(2);
            q.w = (R(2, 1) - R(1, 2)) / s;
            q.x = T(0.25) * s;
            q.y = (R(0, 1) + R(1, 0)) / s;
            q.z = (R(0, 2) + R(2, 0)) / s;
        } else if (R(1, 1) > R(2, 2)) {
            T s = sqrt(T(1) + R(1, 1) - R(0, 0) - R(2, 2)) * T(2);
            q.w = (R(0, 2) - R(2, 0)) / s;
            q.x = (R(0, 1) + R(1, 0)) / s;
            q.y = T(0.25) * s;
            q.z = (R(1, 2) + R(2, 1)) / s;
        } else {
            T s = sqrt(T(1) + R(2, 2) - R(0, 0) - R(1, 1)) * T(2);
            q.w = (R(1, 0) - R(0, 1)) / s;
            q.x = (R(0, 2) + R(2, 0)) / s;
            q.y = (R(1, 2) + R(2, 1)) / s;
            q.z = T(0.25) * s;
        }
        return q.normalized();
    }
    
    static Quaternion fromAngleAxis(T angle, const Vector<3, T>& axis) {
        T halfAngle = angle * T(0.5);
        T s = sin(halfAngle);
        Vector<3, T> n = axis.normalized();
        return Quaternion(cos(halfAngle), n[0] * s, n[1] * s, n[2] * s);
    }
    
    void toAngleAxis(T& angle, Vector<3, T>& axis) const {
        T n = sqrt(x * x + y * y + z * z);
        if (nearZero(n)) {
            angle = T(0);
            axis = Vector<3, T>({T(1), T(0), T(0)});
        } else {
            angle = T(2) * acos(clamp(w, T(-1), T(1)));
            T invN = T(1) / n;
            axis = Vector<3, T>({x * invN, y * invN, z * invN});
        }
    }
    
    static Quaternion exp(const Vector<3, T>& v) {
        T theta = v.norm();
        T halfTheta = theta * T(0.5);
        
        if (nearZero(theta)) {
            T w = T(1) - halfTheta * halfTheta * T(0.5);
            return Quaternion(w, v[0] * T(0.5), v[1] * T(0.5), v[2] * T(0.5));
        }
        
        T s = sin(halfTheta) / theta;
        return Quaternion(cos(halfTheta), v[0] * s, v[1] * s, v[2] * s);
    }
    
    Vector<3, T> log() const {
        T n = sqrt(x * x + y * y + z * z);
        
        if (nearZero(n)) {
            if (w > T(0)) {
                return Vector<3, T>({T(0), T(0), T(0)});
            } else {
                return Vector<3, T>({T(PI), T(0), T(0)});
            }
        }
        
        T theta = T(2) * atan2(n, w);
        T s = theta / n;
        return Vector<3, T>({x * s, y * s, z * s});
    }
    
    static Quaternion fromTwoVectors(const Vector<3, T>& from, const Vector<3, T>& to) {
        Vector<3, T> f = from.normalized();
        Vector<3, T> t = to.normalized();
        T dot = f.dot(t);
        
        if (dot > T(1) - epsilon<T>()) {
            return identity();
        }
        
        if (dot < T(-1) + epsilon<T>()) {
            Vector<3, T> ortho = (abs(f[0]) > T(0.5)) ? 
                Vector<3, T>({f[1], -f[0], T(0)}) : 
                Vector<3, T>({T(0), f[2], -f[1]});
            ortho.normalize();
            return fromAngleAxis(T(PI), ortho);
        }
        
        Vector<3, T> cross = f.cross(t);
        T s = sqrt((T(1) + dot) * T(2));
        T invS = T(1) / s;
        return Quaternion(s * T(0.5), cross[0] * invS, cross[1] * invS, cross[2] * invS).normalized();
    }
    
    Quaternion slerp(const Quaternion& q, T t) const {
        T d = dot(q);
        
        if (d < T(0)) {
            return slerp(-q, t);
        }
        
        d = clamp(d, T(-1), T(1));
        
        if (d > T(1) - epsilon<T>()) {
            return (*this + (q - *this) * t).normalized();
        }
        
        T theta0 = acos(d);
        T theta = theta0 * t;
        T sinTheta = sin(theta);
        T sinTheta0 = sin(theta0);
        
        T s0 = cos(theta) - d * sinTheta / sinTheta0;
        T s1 = sinTheta / sinTheta0;
        
        return (*this * s0 + q * s1).normalized();
    }
    
    Quaternion operator-() const {
        return Quaternion(-w, -x, -y, -z);
    }
    
    Vector<3, T> toEulerAngles() const {
        Vector<3, T> euler;
        
        T sinr_cosp = T(2) * (w * x + y * z);
        T cosr_cosp = T(1) - T(2) * (x * x + y * y);
        euler[0] = atan2(sinr_cosp, cosr_cosp);
        
        T sinp = T(2) * (w * y - z * x);
        if (abs(sinp) >= T(1)) {
            euler[1] = copysign(T(PI) / T(2), sinp);
        } else {
            euler[1] = asin(clamp(sinp, T(-1), T(1)));
        }
        
        T siny_cosp = T(2) * (w * z + x * y);
        T cosy_cosp = T(1) - T(2) * (y * y + z * z);
        euler[2] = atan2(siny_cosp, cosy_cosp);
        
        return euler;
    }
    
    static Quaternion fromEulerAngles(T roll, T pitch, T yaw) {
        T cr = cos(roll * T(0.5));
        T sr = sin(roll * T(0.5));
        T cp = cos(pitch * T(0.5));
        T sp = sin(pitch * T(0.5));
        T cy = cos(yaw * T(0.5));
        T sy = sin(yaw * T(0.5));
        
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }
    
    static Quaternion fromEulerAngles(const Vector<3, T>& euler) {
        return fromEulerAngles(euler[0], euler[1], euler[2]);
    }
    
    friend Quaternion operator*(T scalar, const Quaternion& q) {
        return q * scalar;
    }
};

using Quaternionf = Quaternion<float32>;
using Quaterniond = Quaternion<float64>;

template<typename T>
Quaternion<T> quaternionFromSmallAngle(const Vector<3, T>& dtheta) {
    T halfNorm = dtheta.norm() * T(0.5);
    T c = T(1) - halfNorm * halfNorm * T(0.5);
    T s = T(0.5) * (T(1) - halfNorm * halfNorm * T(6) / T(24));
    return Quaternion<T>(c, dtheta[0] * s, dtheta[1] * s, dtheta[2] * s).normalized();
}

template<typename T>
Vector<3, T> smallAngleFromQuaternion(const Quaternion<T>& q) {
    T n = sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    if (nearZero(n)) {
        return Vector<3, T>({T(2) * q.x, T(2) * q.y, T(2) * q.z});
    }
    T theta = T(2) * atan2(n, q.w);
    T s = theta / n;
    return Vector<3, T>({q.x * s, q.y * s, q.z * s});
}

}

#endif
