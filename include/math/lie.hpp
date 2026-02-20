#ifndef MSCKF_MATH_LIE_HPP
#define MSCKF_MATH_LIE_HPP

#include "matrix.hpp"
#include "quaternion.hpp"

namespace msckf {

template<typename T = float64>
class SO3 {
public:
    Quaternion<T> q_;

    SO3() : q_(Quaternion<T>::identity()) {}
    
    explicit SO3(const Quaternion<T>& q) : q_(q.normalized()) {}
    
    explicit SO3(const Matrix<3, 3, T>& R) : q_(Quaternion<T>::fromRotationMatrix(R)) {}
    
    static SO3 identity() { return SO3(); }
    
    static SO3 fromQuaternion(const Quaternion<T>& q) {
        return SO3(q);
    }
    
    static SO3 fromRotationMatrix(const Matrix<3, 3, T>& R) {
        return SO3(R);
    }
    
    static SO3 fromAngleAxis(T angle, const Vector<3, T>& axis) {
        return SO3(Quaternion<T>::fromAngleAxis(angle, axis));
    }
    
    static SO3 exp(const Vector<3, T>& theta) {
        T thetaNorm = theta.norm();
        
        if (nearZero(thetaNorm)) {
            Matrix<3, 3, T> R = Matrix<3, 3, T>::identity();
            R += skewSymmetric(theta);
            R += T(0.5) * skewSymmetric(theta) * skewSymmetric(theta);
            return SO3(R);
        }
        
        T thetaNorm2 = thetaNorm * thetaNorm;
        
        T A, B;
        if (thetaNorm < T(1e-4)) {
            A = T(1) - thetaNorm2 / T(6);
            B = T(0.5) - thetaNorm2 / T(24);
        } else {
            A = sin(thetaNorm) / thetaNorm;
            B = (T(1) - cos(thetaNorm)) / thetaNorm2;
        }
        
        Vector<3, T> a = theta.normalized();
        Matrix<3, 3, T> aHat = skewSymmetric(a);
        Matrix<3, 3, T> aHat2 = aHat * aHat;
        
        Matrix<3, 3, T> R = Matrix<3, 3, T>::identity() + 
                             A * skewSymmetric(theta) + 
                             B * aHat2 * thetaNorm2;
        
        return SO3(R);
    }
    
    Vector<3, T> log() const {
        return q_.log();
    }
    
    Matrix<3, 3, T> matrix() const {
        return q_.toRotationMatrix();
    }
    
    Quaternion<T> quaternion() const {
        return q_;
    }
    
    SO3 inverse() const {
        return SO3(q_.conjugate());
    }
    
    SO3 operator*(const SO3& other) const {
        return SO3(q_ * other.q_);
    }
    
    Vector<3, T> operator*(const Vector<3, T>& v) const {
        return q_.rotate(v);
    }
    
    static Matrix<3, 3, T> hat(const Vector<3, T>& v) {
        return skewSymmetric(v);
    }
    
    static Vector<3, T> vee(const Matrix<3, 3, T>& M) {
        return Vector<3, T>({M(2, 1), M(0, 2), M(1, 0)});
    }
    
    static Matrix<3, 3, T> leftJacobian(const Vector<3, T>& theta) {
        T thetaNorm = theta.norm();
        
        if (nearZero(thetaNorm)) {
            return Matrix<3, 3, T>::identity() + T(0.5) * skewSymmetric(theta);
        }
        
        Vector<3, T> a = theta.normalized();
        Matrix<3, 3, T> aHat = skewSymmetric(a);
        Matrix<3, 3, T> aHat2 = aHat * aHat;
        
        T theta2 = thetaNorm * thetaNorm;
        T theta3 = theta2 * thetaNorm;
        
        T A = (T(1) - cos(thetaNorm)) / theta2;
        T B = (thetaNorm - sin(thetaNorm)) / theta3;
        
        return Matrix<3, 3, T>::identity() + A * skewSymmetric(theta) + B * aHat2 * theta2;
    }
    
    static Matrix<3, 3, T> leftJacobianInverse(const Vector<3, T>& theta) {
        T thetaNorm = theta.norm();
        
        if (nearZero(thetaNorm)) {
            return Matrix<3, 3, T>::identity() - T(0.5) * skewSymmetric(theta);
        }
        
        Vector<3, T> a = theta.normalized();
        Matrix<3, 3, T> aHat = skewSymmetric(a);
        Matrix<3, 3, T> aHat2 = aHat * aHat;
        
        T theta2 = thetaNorm * thetaNorm;
        
        T halfTheta = thetaNorm * T(0.5);
        T cotHalf;
        if (abs(halfTheta) < T(1e-4)) {
            cotHalf = T(1) / halfTheta - halfTheta / T(3);
        } else if (abs(halfTheta - T(PI)) < T(0.1)) {
            T x = halfTheta - T(PI);
            cotHalf = -x / T(2) + x * x * x / T(24);
        } else {
            cotHalf = T(1) / tan(halfTheta);
        }
        
        T C = (T(1) - thetaNorm * cotHalf * T(0.5)) / theta2;
        
        return Matrix<3, 3, T>::identity() - 
               T(0.5) * skewSymmetric(theta) + 
               C * aHat2 * theta2;
    }
    
    static Matrix<3, 3, T> rightJacobian(const Vector<3, T>& theta) {
        return leftJacobian(-theta);
    }
    
    static Matrix<3, 3, T> rightJacobianInverse(const Vector<3, T>& theta) {
        return leftJacobianInverse(-theta);
    }
    
    Matrix<3, 3, T> adjoint() const {
        return q_.toRotationMatrix();
    }
    
    Vector<3, T> manifoldMinus(const SO3& other) const {
        return (other.inverse() * *this).log();
    }
    
    SO3 manifoldPlus(const Vector<3, T>& delta) const {
        return SO3::exp(delta) * (*this);
    }
    
    SO3 manifoldPlusRight(const Vector<3, T>& delta) const {
        return (*this) * SO3::exp(delta);
    }
};

template<typename T = float64>
class SE3 {
public:
    SO3<T> R_;
    Vector<3, T> t_;

    SE3() : R_(), t_({T(0), T(0), T(0)}) {}
    
    SE3(const SO3<T>& R, const Vector<3, T>& t) : R_(R), t_(t) {}
    
    SE3(const Matrix<3, 3, T>& R, const Vector<3, T>& t) : R_(R), t_(t) {}
    
    SE3(const Quaternion<T>& q, const Vector<3, T>& t) : R_(q), t_(t) {}
    
    static SE3 identity() { return SE3(); }
    
    static SE3 fromRotationTranslation(const Matrix<3, 3, T>& R, const Vector<3, T>& t) {
        return SE3(R, t);
    }
    
    static SE3 fromQuaternionTranslation(const Quaternion<T>& q, const Vector<3, T>& t) {
        return SE3(q, t);
    }
    
    static SE3 exp(const Vector<6, T>& xi) {
        Vector<3, T> rho({xi[0], xi[1], xi[2]});
        Vector<3, T> theta({xi[3], xi[4], xi[5]});
        
        T thetaNorm = theta.norm();
        
        if (nearZero(thetaNorm)) {
            return SE3(SO3<T>::identity(), rho);
        }
        
        SO3<T> R = SO3<T>::exp(theta);
        Matrix<3, 3, T> J = SO3<T>::leftJacobian(theta);
        
        return SE3(R, J * rho);
    }
    
    Vector<6, T> log() const {
        Vector<3, T> theta = R_.log();
        T thetaNorm = theta.norm();
        
        Vector<3, T> rho;
        if (nearZero(thetaNorm)) {
            rho = t_;
        } else {
            Matrix<3, 3, T> Jinv = SO3<T>::leftJacobianInverse(theta);
            rho = Jinv * t_;
        }
        
        return Vector<6, T>({rho[0], rho[1], rho[2], theta[0], theta[1], theta[2]});
    }
    
    Matrix<4, 4, T> matrix() const {
        Matrix<4, 4, T> Tmat = Matrix<4, 4, T>::identity();
        Matrix<3, 3, T> R = R_.matrix();
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                Tmat(i, j) = R(i, j);
            }
            Tmat(i, 3) = t_[i];
        }
        return Tmat;
    }
    
    Matrix<3, 4, T> matrix3x4() const {
        Matrix<3, 4, T> Tmat;
        Matrix<3, 3, T> R = R_.matrix();
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                Tmat(i, j) = R(i, j);
            }
            Tmat(i, 3) = t_[i];
        }
        return Tmat;
    }
    
    SE3 inverse() const {
        SO3<T> Rinv = R_.inverse();
        return SE3(Rinv, Rinv * (-t_));
    }
    
    SE3 operator*(const SE3& other) const {
        return SE3(R_ * other.R_, R_ * other.t_ + t_);
    }
    
    Vector<3, T> operator*(const Vector<3, T>& p) const {
        return R_ * p + t_;
    }
    
    Vector<4, T> operator*(const Vector<4, T>& p) const {
        Vector<3, T> result = R_ * Vector<3, T>({p[0], p[1], p[2]}) + t_;
        return Vector<4, T>({result[0], result[1], result[2], p[3]});
    }
    
    static Matrix<6, 6, T> hat(const Vector<6, T>& xi) {
        Matrix<6, 6, T> Xi;
        Matrix<3, 3, T> omegaHat = skewSymmetric(Vector<3, T>({xi[3], xi[4], xi[5]}));
        Matrix<3, 3, T> vHat = skewSymmetric(Vector<3, T>({xi[0], xi[1], xi[2]}));
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                Xi(i, j) = omegaHat(i, j);
                Xi(i, j + 3) = vHat(i, j);
                Xi(i + 3, j) = T(0);
                Xi(i + 3, j + 3) = omegaHat(i, j);
            }
        }
        return Xi;
    }
    
    static Vector<6, T> vee(const Matrix<6, 6, T>& Xi) {
        return Vector<6, T>({
            Xi(2, 1), Xi(0, 2), Xi(1, 0),
            Xi(2, 4), Xi(0, 5), Xi(1, 3)
        });
    }
    
    Matrix<6, 6, T> adjoint() const {
        Matrix<6, 6, T> Ad;
        Matrix<3, 3, T> R = R_.matrix();
        Matrix<3, 3, T> tHat = skewSymmetric(t_);
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                Ad(i, j) = R(i, j);
                Ad(i, j + 3) = (tHat * R)(i, j);
                Ad(i + 3, j) = T(0);
                Ad(i + 3, j + 3) = R(i, j);
            }
        }
        return Ad;
    }
    
    static Matrix<6, 6, T> leftJacobian(const Vector<6, T>& xi) {
        Vector<3, T> rho({xi[0], xi[1], xi[2]});
        Vector<3, T> theta({xi[3], xi[4], xi[5]});
        
        T thetaNorm = theta.norm();
        
        Matrix<6, 6, T> J = Matrix<6, 6, T>::identity();
        
        if (nearZero(thetaNorm)) {
            Matrix<3, 3, T> rhoHat = skewSymmetric(rho);
            for (uint32 i = 0; i < 3; ++i) {
                for (uint32 j = 0; j < 3; ++j) {
                    J(i, j + 3) = T(0.5) * rhoHat(i, j);
                }
            }
            return J;
        }
        
        Matrix<3, 3, T> Jl = SO3<T>::leftJacobian(theta);
        Matrix<3, 3, T> JlInv = SO3<T>::leftJacobianInverse(theta);
        
        Vector<3, T> a = theta.normalized();
        Matrix<3, 3, T> aHat = skewSymmetric(a);
        Matrix<3, 3, T> aHat2 = aHat * aHat;
        
        T theta2 = thetaNorm * thetaNorm;
        T theta3 = theta2 * thetaNorm;
        T theta4 = theta3 * thetaNorm;
        T theta5 = theta4 * thetaNorm;
        
        T sinT = sin(thetaNorm);
        T cosT = cos(thetaNorm);
        
        T A = (thetaNorm - sinT) / theta3;
        T B = (T(1) - T(0.5) * theta2 - cosT) / theta4;
        T C = (T(0.5) * theta3 + thetaNorm * cosT - sinT) / theta5;
        
        Matrix<3, 3, T> rhoHat = skewSymmetric(rho);
        Matrix<3, 3, T> Q = T(0.5) * rhoHat + 
                           A * (aHat * rhoHat + rhoHat * aHat - T(3) * aHat * rhoHat * aHat) * theta2 +
                           B * (aHat2 * rhoHat + rhoHat * aHat2 - T(3) * aHat * rhoHat * aHat2) * theta2;
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                J(i, j) = Jl(i, j);
                J(i, j + 3) = Q(i, j);
                J(i + 3, j) = T(0);
                J(i + 3, j + 3) = Jl(i, j);
            }
        }
        
        return J;
    }
    
    static Matrix<6, 6, T> leftJacobianInverse(const Vector<6, T>& xi) {
        Vector<3, T> rho({xi[0], xi[1], xi[2]});
        Vector<3, T> theta({xi[3], xi[4], xi[5]});
        
        T thetaNorm = theta.norm();
        
        Matrix<6, 6, T> Jinv = Matrix<6, 6, T>::identity();
        
        if (nearZero(thetaNorm)) {
            Matrix<3, 3, T> rhoHat = skewSymmetric(rho);
            for (uint32 i = 0; i < 3; ++i) {
                for (uint32 j = 0; j < 3; ++j) {
                    Jinv(i, j + 3) = -T(0.5) * rhoHat(i, j);
                }
            }
            return Jinv;
        }
        
        Matrix<3, 3, T> JlInv = SO3<T>::leftJacobianInverse(theta);
        Matrix<6, 6, T> Jl = leftJacobian(xi);
        
        for (uint32 i = 0; i < 6; ++i) {
            for (uint32 j = 0; j < 6; ++j) {
                Jinv(i, j) = Jl(i, j);
            }
        }
        
        return Jinv;
    }
    
    Vector<6, T> manifoldMinus(const SE3& other) const {
        return (other.inverse() * *this).log();
    }
    
    SE3 manifoldPlus(const Vector<6, T>& delta) const {
        return SE3::exp(delta) * (*this);
    }
    
    SE3 manifoldPlusRight(const Vector<6, T>& delta) const {
        return (*this) * SE3::exp(delta);
    }
};

using SO3f = SO3<float32>;
using SO3d = SO3<float64>;
using SE3f = SE3<float32>;
using SE3d = SE3<float64>;

}

#endif
