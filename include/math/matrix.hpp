#ifndef MSCKF_MATH_MATRIX_HPP
#define MSCKF_MATH_MATRIX_HPP

#include "types.hpp"
#include <cstring>

namespace msckf {

template<uint32 R, uint32 C, typename T = float64>
class Matrix {
public:
    static constexpr uint32 Rows = R;
    static constexpr uint32 Cols = C;
    static constexpr uint32 Size = R * C;

private:
    alignas(16) T data_[Size];

public:
    Matrix() {
        memset(data_, 0, Size * sizeof(T));
    }

    Matrix(const Matrix& other) {
        memcpy(data_, other.data_, Size * sizeof(T));
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            memcpy(data_, other.data_, Size * sizeof(T));
        }
        return *this;
    }

    explicit Matrix(T scalar) {
        for (uint32 i = 0; i < Size; ++i) {
            data_[i] = scalar;
        }
    }

    T& operator()(uint32 i, uint32 j) {
        return data_[i * Cols + j];
    }

    const T& operator()(uint32 i, uint32 j) const {
        return data_[i * Cols + j];
    }

    T& operator[](uint32 idx) {
        return data_[idx];
    }

    const T& operator[](uint32 idx) const {
        return data_[idx];
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

    Matrix operator+(const Matrix& other) const {
        Matrix result;
        for (uint32 i = 0; i < Size; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        Matrix result;
        for (uint32 i = 0; i < Size; ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    Matrix operator-() const {
        Matrix result;
        for (uint32 i = 0; i < Size; ++i) {
            result.data_[i] = -data_[i];
        }
        return result;
    }

    Matrix operator*(T scalar) const {
        Matrix result;
        for (uint32 i = 0; i < Size; ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    Matrix operator/(T scalar) const {
        Matrix result;
        T inv = T(1) / scalar;
        for (uint32 i = 0; i < Size; ++i) {
            result.data_[i] = data_[i] * inv;
        }
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        for (uint32 i = 0; i < Size; ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        for (uint32 i = 0; i < Size; ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Matrix& operator*=(T scalar) {
        for (uint32 i = 0; i < Size; ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    Matrix& operator/=(T scalar) {
        T inv = T(1) / scalar;
        for (uint32 i = 0; i < Size; ++i) {
            data_[i] *= inv;
        }
        return *this;
    }

    template<uint32 K>
    Matrix<R, K, T> operator*(const Matrix<C, K, T>& other) const {
        Matrix<R, K, T> result;
        for (uint32 i = 0; i < R; ++i) {
            for (uint32 k = 0; k < K; ++k) {
                T sum = T(0);
                for (uint32 j = 0; j < C; ++j) {
                    sum += data_[i * Cols + j] * other(j, k);
                }
                result(i, k) = sum;
            }
        }
        return result;
    }

    Matrix<C, R, T> transpose() const {
        Matrix<C, R, T> result;
        for (uint32 i = 0; i < R; ++i) {
            for (uint32 j = 0; j < C; ++j) {
                result(j, i) = data_[i * Cols + j];
            }
        }
        return result;
    }

    static Matrix identity() {
        static_assert(R == C, "Identity only for square matrices");
        Matrix result;
        for (uint32 i = 0; i < R; ++i) {
            result(i, i) = T(1);
        }
        return result;
    }

    static Matrix zeros() {
        return Matrix();
    }

    T trace() const {
        static_assert(R == C, "Trace only for square matrices");
        T sum = T(0);
        for (uint32 i = 0; i < R; ++i) {
            sum += data_[i * Cols + i];
        }
        return sum;
    }

    T determinant() const;

    Matrix inverse() const;

    void symmetrize() {
        static_assert(R == C, "Symmetrize only for square matrices");
        for (uint32 i = 0; i < R; ++i) {
            for (uint32 j = i + 1; j < C; ++j) {
                T avg = (data_[i * Cols + j] + data_[j * Cols + i]) * T(0.5);
                data_[i * Cols + j] = avg;
                data_[j * Cols + i] = avg;
            }
        }
    }

    T norm() const {
        T sum = T(0);
        for (uint32 i = 0; i < Size; ++i) {
            sum += data_[i] * data_[i];
        }
        return sqrt(sum);
    }

    T normInf() const {
        T maxVal = T(0);
        for (uint32 i = 0; i < Size; ++i) {
            T val = abs(data_[i]);
            if (val > maxVal) maxVal = val;
        }
        return maxVal;
    }

    void setBlock(uint32 startRow, uint32 startCol, 
                  const Matrix<R == 1 ? 1 : R, C == 1 ? 1 : C, T>& block) {
        constexpr uint32 BlockR = R == 1 ? 1 : R;
        constexpr uint32 BlockC = C == 1 ? 1 : C;
        for (uint32 i = 0; i < BlockR; ++i) {
            for (uint32 j = 0; j < BlockC; ++j) {
                data_[(startRow + i) * Cols + (startCol + j)] = block(i, j);
            }
        }
    }

    template<uint32 BR, uint32 BC>
    Matrix<BR, BC, T> block(uint32 startRow, uint32 startCol) const {
        Matrix<BR, BC, T> result;
        for (uint32 i = 0; i < BR; ++i) {
            for (uint32 j = 0; j < BC; ++j) {
                result(i, j) = data_[(startRow + i) * Cols + (startCol + j)];
            }
        }
        return result;
    }
};

template<uint32 N, typename T>
Matrix<N, N, T> choleskyDecomposition(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> L;
    for (uint32 i = 0; i < N; ++i) {
        for (uint32 j = 0; j <= i; ++j) {
            T sum = T(0);
            for (uint32 k = 0; k < j; ++k) {
                sum += L(i, k) * L(j, k);
            }
            if (i == j) {
                T diag = A(i, i) - sum;
                L(i, j) = sqrt(max(diag, epsilon<T>()));
            } else {
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }
    }
    return L;
}

template<uint32 N, typename T>
Matrix<N, N, T> choleskyInverse(const Matrix<N, N, T>& L) {
    Matrix<N, N, T> Linv;
    for (uint32 i = 0; i < N; ++i) {
        Linv(i, i) = T(1) / L(i, i);
        for (uint32 j = i + 1; j < N; ++j) {
            T sum = T(0);
            for (uint32 k = i; k < j; ++k) {
                sum -= L(j, k) * Linv(k, i);
            }
            Linv(j, i) = sum / L(j, j);
        }
    }
    return Linv;
}

template<uint32 N, typename T>
Matrix<N, N, T> luInverse(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> L, U;
    for (uint32 i = 0; i < N; ++i) {
        for (uint32 j = 0; j < N; ++j) {
            if (j < i) {
                L(i, j) = A(i, j);
                U(i, j) = T(0);
            } else if (j == i) {
                L(i, j) = T(1);
                U(i, j) = A(i, j);
            } else {
                L(i, j) = T(0);
                U(i, j) = A(i, j);
            }
        }
    }
    for (uint32 k = 0; k < N - 1; ++k) {
        for (uint32 i = k + 1; i < N; ++i) {
            if (nearZero(U(k, k))) continue;
            T factor = U(k, i) / U(k, k);
            L(k, i) = factor;
            U(k, i) = T(0);
            for (uint32 j = k + 1; j < N; ++j) {
                U(j, i) -= factor * U(j, k);
            }
        }
    }
    Matrix<N, N, T> inv;
    for (uint32 col = 0; col < N; ++col) {
        T y[N];
        for (uint32 i = 0; i < N; ++i) {
            T sum = (col == i) ? T(1) : T(0);
            for (uint32 j = 0; j < i; ++j) {
                sum -= L(i, j) * y[j];
            }
            y[i] = sum / L(i, i);
        }
        T x[N];
        for (int32 i = N - 1; i >= 0; --i) {
            T sum = y[i];
            for (uint32 j = i + 1; j < N; ++j) {
                sum -= U(i, j) * x[j];
            }
            x[i] = sum / U(i, i);
        }
        for (uint32 i = 0; i < N; ++i) {
            inv(i, col) = x[i];
        }
    }
    return inv;
}

template<uint32 N, typename T>
void qrDecomposition(const Matrix<N, N, T>& A, Matrix<N, N, T>& Q, Matrix<N, N, T>& R) {
    Q = Matrix<N, N, T>::identity();
    R = A;
    for (uint32 k = 0; k < N - 1; ++k) {
        T norm = T(0);
        for (uint32 i = k; i < N; ++i) {
            norm += R(i, k) * R(i, k);
        }
        norm = sqrt(norm);
        if (nearZero(norm)) continue;
        T sign = R(k, k) >= T(0) ? T(1) : T(-1);
        T r = norm * sign + R(k, k);
        T s = T(1) / sqrt(T(2) * norm * (norm + abs(R(k, k))));
        T v[N];
        for (uint32 i = 0; i < N; ++i) {
            v[i] = (i < k) ? T(0) : ((i == k) ? r : R(i, k));
        }
        for (uint32 j = k; j < N; ++j) {
            T dot = T(0);
            for (uint32 i = k; i < N; ++i) {
                dot += v[i] * R(i, j);
            }
            dot *= s;
            for (uint32 i = k; i < N; ++i) {
                R(i, j) -= v[i] * dot;
            }
        }
        for (uint32 j = 0; j < N; ++j) {
            T dot = T(0);
            for (uint32 i = k; i < N; ++i) {
                dot += v[i] * Q(i, j);
            }
            dot *= s;
            for (uint32 i = k; i < N; ++i) {
                Q(i, j) -= v[i] * dot;
            }
        }
    }
    Q = Q.transpose();
}

template<uint32 N, typename T>
void eigenvaluesQR(const Matrix<N, N, T>& A, T eigenvalues[N], 
                   uint32 maxIter = 100, T tol = epsilon<T>()) {
    Matrix<N, N, T> Ak = A;
    for (uint32 iter = 0; iter < maxIter; ++iter) {
        Matrix<N, N, T> Q, R;
        qrDecomposition(Ak, Q, R);
        Ak = R * Q;
        T offDiag = T(0);
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                if (i != j) offDiag += abs(Ak(i, j));
            }
        }
        if (offDiag < tol) break;
    }
    for (uint32 i = 0; i < N; ++i) {
        eigenvalues[i] = Ak(i, i);
    }
}

template<uint32 N, typename T>
void ensurePositiveDefinite(Matrix<N, N, T>& P, T minEig = epsilon<T>()) {
    T eigenvalues[N];
    eigenvaluesQR(P, eigenvalues);
    T minVal = eigenvalues[0];
    for (uint32 i = 1; i < N; ++i) {
        if (eigenvalues[i] < minVal) minVal = eigenvalues[i];
    }
    if (minVal < minEig) {
        T correction = minEig - minVal;
        for (uint32 i = 0; i < N; ++i) {
            P(i, i) += correction;
        }
    }
}

template<uint32 R, uint32 C, typename T>
Matrix<R, C, T> operator*(T scalar, const Matrix<R, C, T>& mat) {
    return mat * scalar;
}

template<uint32 N, typename T>
class Vector : public Matrix<N, 1, T> {
public:
    using Base = Matrix<N, 1, T>;
    
    Vector() : Base() {}
    
    Vector(const Base& mat) : Base(mat) {}
    
    Vector(std::initializer_list<T> init) : Base() {
        uint32 i = 0;
        for (auto val : init) {
            if (i < N) (*this)[i++] = val;
        }
    }
    
    T& operator[](uint32 i) { return Base::data()[i]; }
    const T& operator[](uint32 i) const { return Base::data()[i]; }
    
    T dot(const Vector& other) const {
        T sum = T(0);
        for (uint32 i = 0; i < N; ++i) {
            sum += (*this)[i] * other[i];
        }
        return sum;
    }
    
    T norm() const {
        return sqrt(dot(*this));
    }
    
    T squaredNorm() const {
        return dot(*this);
    }
    
    Vector normalized() const {
        T n = norm();
        if (nearZero(n)) return Vector();
        return *this / n;
    }
    
    void normalize() {
        T n = norm();
        if (!nearZero(n)) {
            T inv = T(1) / n;
            for (uint32 i = 0; i < N; ++i) {
                (*this)[i] *= inv;
            }
        }
    }
    
    Vector cross(const Vector& other) const {
        static_assert(N == 3, "Cross product only for 3D vectors");
        Vector result;
        result[0] = (*this)[1] * other[2] - (*this)[2] * other[1];
        result[1] = (*this)[2] * other[0] - (*this)[0] * other[2];
        result[2] = (*this)[0] * other[1] - (*this)[1] * other[0];
        return result;
    }
    
    static Vector Zero() { return Vector(); }
    
    static Vector UnitX() { 
        static_assert(N >= 1, "Vector too small");
        Vector v; v[0] = T(1); return v; 
    }
    
    static Vector UnitY() { 
        static_assert(N >= 2, "Vector too small");
        Vector v; v[1] = T(1); return v; 
    }
    
    static Vector UnitZ() { 
        static_assert(N >= 3, "Vector too small");
        Vector v; v[2] = T(1); return v; 
    }
};

using Vector2f = Vector<2, float32>;
using Vector3f = Vector<3, float32>;
using Vector4f = Vector<4, float32>;
using Vector2d = Vector<2, float64>;
using Vector3d = Vector<3, float64>;
using Vector4d = Vector<4, float64>;

using Matrix2f = Matrix<2, 2, float32>;
using Matrix3f = Matrix<3, 3, float32>;
using Matrix4f = Matrix<4, 4, float32>;
using Matrix2d = Matrix<2, 2, float64>;
using Matrix3d = Matrix<3, 3, float64>;
using Matrix4d = Matrix<4, 4, float64>;

template<uint32 R, uint32 C>
using MatrixXf = Matrix<R, C, float32>;
template<uint32 R, uint32 C>
using MatrixXd = Matrix<R, C, float64>;

template<uint32 N, typename T>
T Matrix<N, N, T>::determinant() const {
    if constexpr (N == 1) {
        return data_[0];
    } else if constexpr (N == 2) {
        return data_[0] * data_[3] - data_[1] * data_[2];
    } else if constexpr (N == 3) {
        return data_[0] * (data_[4] * data_[8] - data_[5] * data_[7])
             - data_[1] * (data_[3] * data_[8] - data_[5] * data_[6])
             + data_[2] * (data_[3] * data_[7] - data_[4] * data_[6]);
    } else {
        Matrix<N, N, T> L, U;
        T det = T(1);
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                if (j < i) {
                    L(i, j) = data_[i * N + j];
                    U(i, j) = T(0);
                } else if (j == i) {
                    L(i, j) = T(1);
                    U(i, j) = data_[i * N + j];
                } else {
                    L(i, j) = T(0);
                    U(i, j) = data_[i * N + j];
                }
            }
        }
        for (uint32 k = 0; k < N - 1; ++k) {
            if (nearZero(U(k, k))) return T(0);
            for (uint32 i = k + 1; i < N; ++i) {
                T factor = U(k, i) / U(k, k);
                L(k, i) = factor;
                U(k, i) = T(0);
                for (uint32 j = k + 1; j < N; ++j) {
                    U(j, i) -= factor * U(j, k);
                }
            }
        }
        for (uint32 i = 0; i < N; ++i) {
            det *= U(i, i);
        }
        return det;
    }
}

template<uint32 N, typename T>
Matrix<N, N, T> Matrix<N, N, T>::inverse() const {
    if constexpr (N == 1) {
        Matrix<1, 1, T> inv;
        inv(0, 0) = T(1) / data_[0];
        return inv;
    } else if constexpr (N == 2) {
        T det = determinant();
        if (nearZero(det)) return Matrix<N, N, T>::identity();
        T invDet = T(1) / det;
        Matrix<2, 2, T> inv;
        inv(0, 0) = data_[3] * invDet;
        inv(0, 1) = -data_[1] * invDet;
        inv(1, 0) = -data_[2] * invDet;
        inv(1, 1) = data_[0] * invDet;
        return inv;
    } else if constexpr (N == 3) {
        T det = determinant();
        if (nearZero(det)) return Matrix<N, N, T>::identity();
        T invDet = T(1) / det;
        Matrix<3, 3, T> inv;
        inv(0, 0) = (data_[4] * data_[8] - data_[5] * data_[7]) * invDet;
        inv(0, 1) = (data_[2] * data_[7] - data_[1] * data_[8]) * invDet;
        inv(0, 2) = (data_[1] * data_[5] - data_[2] * data_[4]) * invDet;
        inv(1, 0) = (data_[5] * data_[6] - data_[3] * data_[8]) * invDet;
        inv(1, 1) = (data_[0] * data_[8] - data_[2] * data_[6]) * invDet;
        inv(1, 2) = (data_[2] * data_[3] - data_[0] * data_[5]) * invDet;
        inv(2, 0) = (data_[3] * data_[7] - data_[4] * data_[6]) * invDet;
        inv(2, 1) = (data_[1] * data_[6] - data_[0] * data_[7]) * invDet;
        inv(2, 2) = (data_[0] * data_[4] - data_[1] * data_[3]) * invDet;
        return inv;
    } else {
        return luInverse(*this);
    }
}

template<uint32 N, typename T>
Matrix<N, N, T> outerProduct(const Vector<N, T>& a, const Vector<N, T>& b) {
    Matrix<N, N, T> result;
    for (uint32 i = 0; i < N; ++i) {
        for (uint32 j = 0; j < N; ++j) {
            result(i, j) = a[i] * b[j];
        }
    }
    return result;
}

template<typename T>
Matrix<3, 3, T> skewSymmetric(const Vector<3, T>& v) {
    Matrix<3, 3, T> S;
    S(0, 0) = T(0);      S(0, 1) = -v[2];    S(0, 2) = v[1];
    S(1, 0) = v[2];      S(1, 1) = T(0);     S(1, 2) = -v[0];
    S(2, 0) = -v[1];     S(2, 1) = v[0];     S(2, 2) = T(0);
    return S;
}

template<uint32 N, typename T>
Matrix<N, N, T> makePositiveDefinite(const Matrix<N, N, T>& P, T minEig = T(1e-6)) {
    Matrix<N, N, T> result = P;
    result.symmetrize();
    ensurePositiveDefinite(result, minEig);
    return result;
}

}

#endif
