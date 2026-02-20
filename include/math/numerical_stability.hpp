#ifndef MSCKF_MATH_NUMERICAL_STABILITY_HPP
#define MSCKF_MATH_NUMERICAL_STABILITY_HPP

#include "types.hpp"
#include "matrix.hpp"
#include <cstring>

namespace msckf {

class KahanAccumulator {
public:
    float64 sum_;
    float64 compensation_;
    
    KahanAccumulator() : sum_(0), compensation_(0) {}
    
    void reset() {
        sum_ = 0;
        compensation_ = 0;
    }
    
    void add(float64 value) {
        float64 y = value - compensation_;
        float64 t = sum_ + y;
        compensation_ = (t - sum_) - y;
        sum_ = t;
    }
    
    float64 getSum() const {
        return sum_;
    }
    
    KahanAccumulator& operator+=(float64 value) {
        add(value);
        return *this;
    }
    
    KahanAccumulator& operator=(float64 value) {
        sum_ = value;
        compensation_ = 0;
        return *this;
    }
};

template<uint32 N>
class KahanVectorAccumulator {
private:
    float64 sums_[N];
    float64 compensations_[N];

public:
    KahanVectorAccumulator() {
        reset();
    }
    
    void reset() {
        for (uint32 i = 0; i < N; ++i) {
            sums_[i] = 0;
            compensations_[i] = 0;
        }
    }
    
    void add(const float64* values) {
        for (uint32 i = 0; i < N; ++i) {
            float64 y = values[i] - compensations_[i];
            float64 t = sums_[i] + y;
            compensations_[i] = (t - sums_[i]) - y;
            sums_[i] = t;
        }
    }
    
    void add(uint32 i, float64 value) {
        if (i < N) {
            float64 y = value - compensations_[i];
            float64 t = sums_[i] + y;
            compensations_[i] = (t - sums_[i]) - y;
            sums_[i] = t;
        }
    }
    
    void getSum(float64* result) const {
        for (uint32 i = 0; i < N; ++i) {
            result[i] = sums_[i];
        }
    }
    
    float64 getSum(uint32 i) const {
        return (i < N) ? sums_[i] : 0;
    }
    
    uint32 dimension() const { return N; }
};

template<uint32 N>
class KahanMatrixAccumulator {
private:
    Matrix<N, N, float64> sums_;
    Matrix<N, N, float64> compensations_;

public:
    KahanMatrixAccumulator() {
        reset();
    }
    
    void reset() {
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                sums_(i, j) = 0;
                compensations_(i, j) = 0;
            }
        }
    }
    
    void add(uint32 i, uint32 j, float64 value) {
        float64 y = value - compensations_(i, j);
        float64 t = sums_(i, j) + y;
        compensations_(i, j) = (t - sums_(i, j)) - y;
        sums_(i, j) = t;
    }
    
    void add(const Matrix<N, N, float64>& mat) {
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                add(i, j, mat(i, j));
            }
        }
    }
    
    const Matrix<N, N, float64>& getSum() const {
        return sums_;
    }
    
    float64 getSum(uint32 i, uint32 j) const {
        return sums_(i, j);
    }
};

template<uint32 N>
class UDDecomposition {
public:
    Matrix<N, N, float64> U_;
    Vector<N, float64> d_;
    
    UDDecomposition() {
        for (uint32 i = 0; i < N; ++i) {
            d_[i] = 0;
            for (uint32 j = 0; j < N; ++j) {
                U_(i, j) = 0;
            }
        }
    }
    
    static UDDecomposition<N> decompose(const Matrix<N, N, float64>& P) {
        UDDecomposition<N> result;
        
        for (uint32 j = N; j >= 1; --j) {
            float64 alpha = 0;
            
            if (j > 1) {
                for (uint32 k = 1; k <= j - 1; ++k) {
                    KahanAccumulator acc;
                    for (uint32 i = 1; i <= k; ++i) {
                        acc.add(P(i-1, j-1) * result.U_(i-1, k-1));
                    }
                    result.U_(k-1, j-1) = acc.getSum();
                }
                
                KahanAccumulator acc;
                for (uint32 k = 1; k <= j - 1; ++k) {
                    acc.add(result.U_(k-1, j-1) * result.U_(k-1, j-1) * result.d_[k-1]);
                }
                alpha = P(j-1, j-1) - acc.getSum();
            } else {
                alpha = P(0, 0);
            }
            
            if (alpha <= 0) {
                alpha = 1e-10;
            }
            result.d_[j-1] = alpha;
            
            if (j > 1) {
                for (uint32 k = 1; k <= j - 1; ++k) {
                    result.U_(k-1, j-1) /= alpha;
                }
            }
        }
        
        for (uint32 i = 0; i < N; ++i) {
            result.U_(i, i) = 1.0;
        }
        
        return result;
    }
    
    Matrix<N, N, float64> reconstruct() const {
        Matrix<N, N, float64> P;
        
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                KahanAccumulator acc;
                for (uint32 k = 0; k < N; ++k) {
                    acc.add(U_(i, k) * d_[k] * U_(j, k));
                }
                P(i, j) = acc.getSum();
            }
        }
        
        return P;
    }
    
    Matrix<N, N, float64> inverse() const {
        Matrix<N, N, float64> Uinv;
        Vector<N, float64> dinv;
        
        for (uint32 i = 0; i < N; ++i) {
            dinv[i] = safeDivide(1.0, d_[i], 1e10);
        }
        
        for (uint32 i = 0; i < N; ++i) {
            Uinv(i, i) = 1.0;
            for (uint32 j = i + 1; j < N; ++j) {
                KahanAccumulator acc;
                for (uint32 k = i; k < j; ++k) {
                    acc.add(-U_(i, k) * Uinv(k, j));
                }
                Uinv(i, j) = acc.getSum();
            }
        }
        
        Matrix<N, N, float64> Pinv;
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                KahanAccumulator acc;
                for (uint32 k = 0; k < N; ++k) {
                    acc.add(Uinv(i, k) * dinv[k] * Uinv(j, k));
                }
                Pinv(i, j) = acc.getSum();
            }
        }
        
        return Pinv;
    }
    
    bool isPositiveDefinite() const {
        for (uint32 i = 0; i < N; ++i) {
            if (d_[i] <= 0) {
                return false;
            }
        }
        return true;
    }
};

template<uint32 N>
class SquareRootCovariance {
public:
    Matrix<N, N, float64> S_;
    
    SquareRootCovariance() {
        S_ = Matrix<N, N, float64>::identity();
    }
    
    explicit SquareRootCovariance(const Matrix<N, N, float64>& P) {
        computeSqrt(P);
    }
    
    void computeSqrt(const Matrix<N, N, float64>& P) {
        for (uint32 j = 0; j < N; ++j) {
            KahanAccumulator acc;
            for (uint32 k = 0; k < j; ++k) {
                acc.add(S_(j, k) * S_(j, k));
            }
            
            float64 diag = P(j, j) - acc.getSum();
            if (diag < 1e-10) {
                diag = 1e-10;
            }
            S_(j, j) = sqrt(diag);
            
            float64 invSjj = 1.0 / S_(j, j);
            
            for (uint32 i = j + 1; i < N; ++i) {
                acc.reset();
                for (uint32 k = 0; k < j; ++k) {
                    acc.add(S_(i, k) * S_(j, k));
                }
                S_(i, j) = (P(i, j) - acc.getSum()) * invSjj;
                S_(j, i) = 0;
            }
        }
    }
    
    Matrix<N, N, float64> reconstruct() const {
        Matrix<N, N, float64> P;
        
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                KahanAccumulator acc;
                for (uint32 k = 0; k < N; ++k) {
                    acc.add(S_(i, k) * S_(j, k));
                }
                P(i, j) = acc.getSum();
            }
        }
        
        return P;
    }
    
    void propagate(const Matrix<N, N, float64>& Phi, const Matrix<N, N, float64>& Q_sqrt) {
        Matrix<N, N, float64> S_new;
        
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                KahanAccumulator acc;
                for (uint32 k = 0; k < N; ++k) {
                    acc.add(Phi(i, k) * S_(k, j));
                }
                S_new(i, j) = acc.getSum();
            }
        }
        
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                S_new(i, j) += Q_sqrt(i, j);
            }
        }
        
        qrDecompose(S_new);
    }
    
    void update(const Matrix<N, 1, float64>& H, float64 R, float64 innovation) {
        Vector<N, float64> Phi;
        
        for (uint32 i = 0; i < N; ++i) {
            KahanAccumulator acc;
            for (uint32 k = 0; k < N; ++k) {
                acc.add(S_(k, i) * H(k, 0));
            }
            Phi[i] = acc.getSum();
        }
        
        float64 alpha = sqrt(R);
        float64 beta = R;
        
        for (uint32 j = 0; j < N; ++j) {
            float64 gamma = Phi[j] / alpha;
            
            float64 newAlpha = sqrt(alpha * alpha + Phi[j] * Phi[j]);
            
            for (uint32 i = 0; i < N; ++i) {
                float64 s = S_(i, j);
                S_(i, j) = (s * alpha + Phi[j] * (gamma * s - 
                            (i < j ? 0 : (i == j ? Phi[j] : Phi[i])))) / newAlpha;
            }
            
            alpha = newAlpha;
        }
    }
    
private:
    void qrDecompose(Matrix<N, N, float64>& A) {
        for (uint32 k = 0; k < N; ++k) {
            float64 norm = 0;
            for (uint32 i = k; i < N; ++i) {
                norm += A(i, k) * A(i, k);
            }
            norm = sqrt(norm);
            
            if (norm < 1e-10) continue;
            
            float64 s = -sign(A(k, k)) * norm;
            float64 u1 = A(k, k) - s;
            A(k, k) = s;
            
            float64 beta = -u1 * s;
            if (abs(beta) < 1e-10) continue;
            
            for (uint32 j = k + 1; j < N; ++j) {
                KahanAccumulator acc;
                acc.add(u1 * A(k, j));
                for (uint32 i = k + 1; i < N; ++i) {
                    acc.add(A(i, k) * A(i, j));
                }
                float64 gamma = acc.getSum() / beta;
                
                A(k, j) -= u1 * gamma;
                for (uint32 i = k + 1; i < N; ++i) {
                    A(i, j) -= A(i, k) * gamma;
                }
            }
        }
        
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = i + 1; j < N; ++j) {
                S_(i, j) = 0;
            }
        }
    }
};

class NumericalStabilityChecker {
public:
    struct Config {
        float64 maxConditionNumber;
        float64 minEigenvalue;
        float64 maxEigenvalue;
        float64 maxMatrixNorm;
        float64 nanCheckThreshold;
        float64 infCheckThreshold;
        
        Config() 
            : maxConditionNumber(1e12)
            , minEigenvalue(1e-15)
            , maxEigenvalue(1e15)
            , maxMatrixNorm(1e10)
            , nanCheckThreshold(1e-100)
            , infCheckThreshold(1e100) {}
    };

private:
    Config config_;
    
    uint32 nanCount_;
    uint32 infCount_;
    uint32 conditionNumberViolations_;
    uint32 eigenvalueViolations_;

public:
    NumericalStabilityChecker() 
        : nanCount_(0)
        , infCount_(0)
        , conditionNumberViolations_(0)
        , eigenvalueViolations_(0) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    bool checkValue(float64 value) const {
        if (value != value) {
            return false;
        }
        if (value > config_.infCheckThreshold || value < -config_.infCheckThreshold) {
            return false;
        }
        return true;
    }
    
    template<uint32 N>
    bool checkMatrix(const Matrix<N, N, float64>& M) {
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = 0; j < N; ++j) {
                if (!checkValue(M(i, j))) {
                    nanCount_++;
                    return false;
                }
            }
        }
        return true;
    }
    
    template<uint32 N>
    float64 computeConditionNumber(const Matrix<N, N, float64>& M) {
        float64 minDiag = abs(M(0, 0));
        float64 maxDiag = abs(M(0, 0));
        
        for (uint32 i = 1; i < N; ++i) {
            float64 diag = abs(M(i, i));
            if (diag < minDiag) minDiag = diag;
            if (diag > maxDiag) maxDiag = diag;
        }
        
        if (minDiag < config_.minEigenvalue) {
            eigenvalueViolations_++;
            return config_.maxConditionNumber;
        }
        
        return maxDiag / minDiag;
    }
    
    template<uint32 N>
    bool isConditionNumberValid(const Matrix<N, N, float64>& M) {
        float64 condNum = computeConditionNumber(M);
        
        if (condNum > config_.maxConditionNumber) {
            conditionNumberViolations_++;
            return false;
        }
        return true;
    }
    
    template<uint32 N>
    void stabilizeMatrix(Matrix<N, N, float64>& M, float64 epsilon = 1e-10) {
        for (uint32 i = 0; i < N; ++i) {
            if (abs(M(i, i)) < epsilon) {
                M(i, i) = epsilon;
            }
        }
    }
    
    template<uint32 N>
    void symmetrizeMatrix(Matrix<N, N, float64>& M) {
        for (uint32 i = 0; i < N; ++i) {
            for (uint32 j = i + 1; j < N; ++j) {
                float64 avg = (M(i, j) + M(j, i)) * 0.5;
                M(i, j) = avg;
                M(j, i) = avg;
            }
        }
    }
    
    uint32 getNanCount() const { return nanCount_; }
    uint32 getInfCount() const { return infCount_; }
    uint32 getConditionNumberViolations() const { return conditionNumberViolations_; }
    uint32 getEigenvalueViolations() const { return eigenvalueViolations_; }
    
    void reset() {
        nanCount_ = 0;
        infCount_ = 0;
        conditionNumberViolations_ = 0;
        eigenvalueViolations_ = 0;
    }
};

inline float64 robustNorm(const Vector3d& v, float64 epsilon = 1e-6) {
    float64 sum = 0;
    for (uint32 i = 0; i < 3; ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum + epsilon * epsilon);
}

inline Vector3d robustNormalize(const Vector3d& v, float64 epsilon = 1e-6) {
    float64 norm = robustNorm(v, epsilon);
    return v / norm;
}

inline float64 huberLoss(float64 r, float64 threshold = 1.0) {
    float64 absR = abs(r);
    if (absR <= threshold) {
        return 0.5 * r * r;
    } else {
        return threshold * (absR - 0.5 * threshold);
    }
}

inline float64 huberDerivative(float64 r, float64 threshold = 1.0) {
    float64 absR = abs(r);
    if (absR <= threshold) {
        return r;
    } else {
        return threshold * sign(r);
    }
}

inline float64 cauchyLoss(float64 r, float64 threshold = 1.0) {
    float64 x = r / threshold;
    return 0.5 * threshold * threshold * log(1.0 + x * x);
}

inline float64 cauchyDerivative(float64 r, float64 threshold = 1.0) {
    float64 x = r / threshold;
    return r / (1.0 + x * x);
}

inline float64 tukeyLoss(float64 r, float64 threshold = 4.6851) {
    float64 x = r / threshold;
    if (abs(x) > 1.0) {
        return threshold * threshold / 6.0;
    }
    float64 x2 = x * x;
    return threshold * threshold / 6.0 * (1.0 - pow(1.0 - x2, 3));
}

inline float64 tukeyDerivative(float64 r, float64 threshold = 4.6851) {
    float64 x = r / threshold;
    if (abs(x) > 1.0) {
        return 0;
    }
    return r * pow(1.0 - x * x, 2);
}

}

#endif
