#ifndef MSCKF_CORE_INCREMENTAL_SCHUR_HPP
#define MSCKF_CORE_INCREMENTAL_SCHUR_HPP

#include "../math/types.hpp"
#include "../math/matrix.hpp"
#include "../core/state.hpp"
#include <cstring>

namespace msckf {

constexpr uint32 MAX_INCREMENTAL_FEATURES = 500;
constexpr uint32 SCHUR_BLOCK_SIZE = 3;

struct FeatureBlock {
    Matrix<3, 3, float64> A;
    Matrix<MAX_STATE_DIM, 3, float64> B;
    Matrix<3, MAX_STATE_DIM, float64> Bt;
    float64 r[3];
    bool isActive;
    uint32 featureId;
    uint32 observationCount;
    
    FeatureBlock() 
        : isActive(false)
        , featureId(0)
        , observationCount(0) {
        A = Matrix<3, 3, float64>();
        B = Matrix<MAX_STATE_DIM, 3, float64>();
        Bt = Matrix<3, MAX_STATE_DIM, float64>();
        r[0] = r[1] = r[2] = 0;
    }
};

class IncrementalSchurComplement {
public:
    struct Config {
        uint32 maxFeatures;
        float64 regularizationFactor;
        float64 minEigenvalue;
        bool useWoodbury;
        bool useSparseCholesky;
        uint32 updateThreshold;
        
        Config() 
            : maxFeatures(MAX_INCREMENTAL_FEATURES)
            , regularizationFactor(1e-6)
            , minEigenvalue(1e-8)
            , useWoodbury(true)
            , useSparseCholesky(true)
            , updateThreshold(5) {}
    };

private:
    Config config_;
    
    FeatureBlock featureBlocks_[MAX_INCREMENTAL_FEATURES];
    uint32 numActiveBlocks_;
    uint32 nextBlockIdx_;
    
    Matrix<3, 3, float64> A_inv_[MAX_INCREMENTAL_FEATURES];
    bool A_inv_valid_[MAX_INCREMENTAL_FEATURES];
    
    Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64> H_tilde_;
    Vector<MAX_STATE_DIM, float64> r_tilde_;
    uint32 stateDim_;
    
    float64 cholL_[MAX_STATE_DIM * MAX_STATE_DIM];
    uint32 cholSize_;
    
    uint32 updateCounter_;
    bool needsFullRecompute_;

public:
    IncrementalSchurComplement() 
        : numActiveBlocks_(0)
        , nextBlockIdx_(0)
        , stateDim_(0)
        , cholSize_(0)
        , updateCounter_(0)
        , needsFullRecompute_(true) {
        memset(A_inv_valid_, 0, sizeof(A_inv_valid_));
        for (uint32 i = 0; i < MAX_STATE_DIM * MAX_STATE_DIM; ++i) {
            cholL_[i] = 0;
        }
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        numActiveBlocks_ = 0;
        nextBlockIdx_ = 0;
        updateCounter_ = 0;
        needsFullRecompute_ = true;
        cholSize_ = 0;
        
        memset(A_inv_valid_, 0, sizeof(A_inv_valid_));
    }
    
    void setStateDimension(uint32 dim) {
        stateDim_ = dim;
        needsFullRecompute_ = true;
    }
    
    void addFeature(uint32 featureId, const Matrix<3, 3, float64>& A_ii,
                    const Matrix<MAX_STATE_DIM, 3, float64>& B_i,
                    const float64* residual) {
        if (numActiveBlocks_ >= config_.maxFeatures) {
            removeOldestFeature();
        }
        
        uint32 blockIdx = findOrCreateBlock(featureId);
        
        FeatureBlock& block = featureBlocks_[blockIdx];
        block.A = A_ii;
        block.B = B_i;
        block.featureId = featureId;
        block.isActive = true;
        block.observationCount++;
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < stateDim_; ++j) {
                block.Bt(i, j) = B_i(j, i);
            }
            block.r[i] = residual[i];
        }
        
        if (config_.useWoodbury) {
            updateAInverseWoodbury(blockIdx, A_ii);
        } else {
            A_inv_[blockIdx] = A_ii.inverse();
            A_inv_valid_[blockIdx] = true;
        }
        
        updateCounter_++;
        if (updateCounter_ >= config_.updateThreshold) {
            needsFullRecompute_ = true;
            updateCounter_ = 0;
        }
    }
    
    void removeFeature(uint32 featureId) {
        uint32 blockIdx = findBlock(featureId);
        if (blockIdx >= MAX_INCREMENTAL_FEATURES) return;
        
        featureBlocks_[blockIdx].isActive = false;
        A_inv_valid_[blockIdx] = false;
        
        if (config_.useSparseCholesky) {
            downdateCholesky(blockIdx);
        }
        
        numActiveBlocks_--;
        needsFullRecompute_ = true;
    }
    
    void removeOldestFeature() {
        uint32 oldestIdx = 0;
        uint32 maxObs = 0;
        
        for (uint32 i = 0; i < MAX_INCREMENTAL_FEATURES; ++i) {
            if (featureBlocks_[i].isActive && 
                featureBlocks_[i].observationCount > maxObs) {
                maxObs = featureBlocks_[i].observationCount;
                oldestIdx = i;
            }
        }
        
        removeFeature(featureBlocks_[oldestIdx].featureId);
    }
    
    void updateAInverseWoodbury(uint32 blockIdx, const Matrix<3, 3, float64>& A_new) {
        if (!A_inv_valid_[blockIdx]) {
            A_inv_[blockIdx] = A_new.inverse();
            A_inv_valid_[blockIdx] = true;
            return;
        }
        
        Matrix<3, 3, float64> A_old_inv = A_inv_[blockIdx];
        Matrix<3, 3, float64> A_old;
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                A_old(i, j) = 0;
                for (uint32 k = 0; k < 3; ++k) {
                    A_old(i, j) += A_old_inv(i, k) * A_old_inv(j, k);
                }
            }
        }
        
        Matrix<3, 3, float64> delta_A;
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                delta_A(i, j) = A_new(i, j) - A_old(i, j);
            }
        }
        
        Matrix<3, 3, float64> I = Matrix<3, 3, float64>::identity();
        Matrix<3, 3, float64> A_old_inv_delta = A_old_inv * delta_A;
        Matrix<3, 3, float64> denominator = I + A_old_inv_delta;
        
        float64 det = denominator(0, 0) * (denominator(1, 1) * denominator(2, 2) - denominator(1, 2) * denominator(2, 1)) -
                      denominator(0, 1) * (denominator(1, 0) * denominator(2, 2) - denominator(1, 2) * denominator(2, 0)) +
                      denominator(0, 2) * (denominator(1, 0) * denominator(2, 1) - denominator(1, 1) * denominator(2, 0));
        
        if (abs(det) < config_.minEigenvalue) {
            A_inv_[blockIdx] = A_new.inverse();
        } else {
            Matrix<3, 3, float64> denominator_inv = denominator.inverse();
            A_inv_[blockIdx] = A_old_inv - A_old_inv_delta * denominator_inv * A_old_inv;
        }
        
        regularizeMatrix(A_inv_[blockIdx]);
    }
    
    void regularizeMatrix(Matrix<3, 3, float64>& M) {
        for (uint32 i = 0; i < 3; ++i) {
            M(i, i) += config_.regularizationFactor;
        }
    }
    
    void computeSchurComplement(MSCKFState& state, 
                                float64* H_x, float64* r,
                                uint32 obsDim, uint32 stateDim) {
        if (needsFullRecompute_) {
            fullSchurComplement(state, H_x, r, obsDim, stateDim);
        } else {
            incrementalSchurComplement(state, H_x, r, obsDim, stateDim);
        }
    }
    
    void fullSchurComplement(MSCKFState& state,
                             float64* H_x, float64* r,
                             uint32 obsDim, uint32 stateDim) {
        H_tilde_ = Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>();
        r_tilde_ = Vector<MAX_STATE_DIM, float64>();
        
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < obsDim; ++k) {
                    sum += H_x[k * stateDim + i] * H_x[k * stateDim + j];
                }
                H_tilde_(i, j) = sum;
            }
            
            float64 sum = 0;
            for (uint32 k = 0; k < obsDim; ++k) {
                sum += H_x[k * stateDim + i] * r[k];
            }
            r_tilde_[i] = sum;
        }
        
        for (uint32 f = 0; f < MAX_INCREMENTAL_FEATURES; ++f) {
            if (!featureBlocks_[f].isActive) continue;
            
            FeatureBlock& block = featureBlocks_[f];
            Matrix<3, 3, float64>& A_inv = A_inv_[f];
            
            Matrix<MAX_STATE_DIM, 3, float64> B_Ainv;
            for (uint32 i = 0; i < stateDim; ++i) {
                for (uint32 j = 0; j < 3; ++j) {
                    float64 sum = 0;
                    for (uint32 k = 0; k < 3; ++k) {
                        sum += block.B(i, k) * A_inv(k, j);
                    }
                    B_Ainv(i, j) = sum;
                }
            }
            
            for (uint32 i = 0; i < stateDim; ++i) {
                for (uint32 j = 0; j < stateDim; ++j) {
                    float64 sum = 0;
                    for (uint32 k = 0; k < 3; ++k) {
                        sum += B_Ainv(i, k) * block.Bt(k, j);
                    }
                    H_tilde_(i, j) -= sum;
                }
            }
            
            Vector3d r_f({block.r[0], block.r[1], block.r[2]});
            Vector3d Ainv_r = A_inv * r_f;
            
            for (uint32 i = 0; i < stateDim; ++i) {
                float64 sum = 0;
                for (uint32 k = 0; k < 3; ++k) {
                    sum += block.B(i, k) * Ainv_r[k];
                }
                r_tilde_[i] -= sum;
            }
        }
        
        needsFullRecompute_ = false;
    }
    
    void incrementalSchurComplement(MSCKFState& state,
                                    float64* H_x, float64* r,
                                    uint32 obsDim, uint32 stateDim) {
        for (uint32 f = 0; f < MAX_INCREMENTAL_FEATURES; ++f) {
            if (!featureBlocks_[f].isActive || !A_inv_valid_[f]) continue;
            
            FeatureBlock& block = featureBlocks_[f];
            
            if (block.observationCount == 1) {
                Matrix<3, 3, float64>& A_inv = A_inv_[f];
                
                for (uint32 i = 0; i < stateDim; ++i) {
                    for (uint32 j = 0; j < stateDim; ++j) {
                        float64 correction = 0;
                        for (uint32 k = 0; k < 3; ++k) {
                            for (uint32 l = 0; l < 3; ++l) {
                                correction += block.B(i, k) * A_inv(k, l) * block.Bt(l, j);
                            }
                        }
                        H_tilde_(i, j) -= correction;
                    }
                }
                
                Vector3d r_f({block.r[0], block.r[1], block.r[2]});
                Vector3d Ainv_r = A_inv * r_f;
                
                for (uint32 i = 0; i < stateDim; ++i) {
                    float64 correction = 0;
                    for (uint32 k = 0; k < 3; ++k) {
                        correction += block.B(i, k) * Ainv_r[k];
                    }
                    r_tilde_[i] -= correction;
                }
            }
        }
    }
    
    void downdateCholesky(uint32 blockIdx) {
        if (!config_.useSparseCholesky || cholL_ == nullptr) return;
        
        FeatureBlock& block = featureBlocks_[blockIdx];
        
        Matrix<3, 3, float64>& A = block.A;
        
        float64 L[3][3];
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j < 3; ++j) {
                L[i][j] = 0;
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = 0; j <= i; ++j) {
                float64 sum = 0;
                for (uint32 k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                if (i == j) {
                    float64 diag = A(i, i) - sum;
                    L[i][j] = sqrt(max(diag, config_.minEigenvalue));
                } else {
                    L[i][j] = (A(i, j) - sum) / L[j][j];
                }
            }
        }
        
        for (uint32 i = 0; i < 3; ++i) {
            for (uint32 j = i; j < 3; ++j) {
                float64 diag = L[j][i];
                for (uint32 k = i; k < j; ++k) {
                    L[j][k] -= diag * L[i][k] / L[i][i];
                }
                L[j][i] /= L[i][i];
            }
        }
    }
    
    uint32 findOrCreateBlock(uint32 featureId) {
        uint32 idx = findBlock(featureId);
        if (idx < MAX_INCREMENTAL_FEATURES) return idx;
        
        for (uint32 i = 0; i < MAX_INCREMENTAL_FEATURES; ++i) {
            if (!featureBlocks_[i].isActive) {
                featureBlocks_[i].featureId = featureId;
                featureBlocks_[i].observationCount = 0;
                return i;
            }
        }
        
        return 0;
    }
    
    uint32 findBlock(uint32 featureId) {
        for (uint32 i = 0; i < MAX_INCREMENTAL_FEATURES; ++i) {
            if (featureBlocks_[i].isActive && 
                featureBlocks_[i].featureId == featureId) {
                return i;
            }
        }
        return MAX_INCREMENTAL_FEATURES;
    }
    
    void getSchurComplementResult(float64* H_out, float64* r_out, uint32 stateDim) {
        for (uint32 i = 0; i < stateDim; ++i) {
            for (uint32 j = 0; j < stateDim; ++j) {
                H_out[i * stateDim + j] = H_tilde_(i, j);
            }
            r_out[i] = r_tilde_[i];
        }
    }
    
    void reset() {
        numActiveBlocks_ = 0;
        nextBlockIdx_ = 0;
        updateCounter_ = 0;
        needsFullRecompute_ = true;
        
        for (uint32 i = 0; i < MAX_INCREMENTAL_FEATURES; ++i) {
            featureBlocks_[i].isActive = false;
            A_inv_valid_[i] = false;
        }
        
        H_tilde_ = Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>();
        r_tilde_ = Vector<MAX_STATE_DIM, float64>();
    }
    
    uint32 getNumActiveBlocks() const { return numActiveBlocks_; }
    bool needsRecompute() const { return needsFullRecompute_; }
    const Matrix<MAX_STATE_DIM, MAX_STATE_DIM, float64>& getH_tilde() const { return H_tilde_; }
    const Vector<MAX_STATE_DIM, float64>& getR_tilde() const { return r_tilde_; }
};

class SparseCholeskySolver {
public:
    struct Config {
        uint32 maxMatrixSize;
        float64 dropTolerance;
        uint32 maxFillIn;
        bool useAMD;
        
        Config() 
            : maxMatrixSize(MAX_STATE_DIM)
            , dropTolerance(1e-10)
            , maxFillIn(10)
            , useAMD(true) {}
    };

private:
    Config config_;
    
    static constexpr uint32 MAX_NNZ = MAX_STATE_DIM * MAX_STATE_DIM;
    float64 L_values_[MAX_NNZ];
    uint32 L_rowPtr_[MAX_STATE_DIM + 1];
    uint32 L_colIdx_[MAX_NNZ];
    uint32 L_nnz_;
    
    uint32 permutation_[MAX_STATE_DIM];
    uint32 invPermutation_[MAX_STATE_DIM];
    
    float64 workspace_[MAX_STATE_DIM * 3];
    uint32 degree_[MAX_STATE_DIM];
    bool eliminated_[MAX_STATE_DIM];

public:
    SparseCholeskySolver() 
        : L_nnz_(0) {
        for (uint32 i = 0; i < MAX_STATE_DIM; ++i) {
            permutation_[i] = i;
            invPermutation_[i] = i;
        }
    }
    
    void init(const Config& config = Config()) {
        config_ = config;
        L_nnz_ = 0;
        for (uint32 i = 0; i < MAX_STATE_DIM; ++i) {
            permutation_[i] = i;
            invPermutation_[i] = i;
        }
    }
    
    void analyzePattern(const float64* A, uint32 n) {
        if (config_.useAMD) {
            approximateMinimumDegree(A, n);
        }
    }
    
    void approximateMinimumDegree(const float64* A, uint32 n) {
        for (uint32 i = 0; i < n; ++i) {
            degree_[i] = 0;
            eliminated_[i] = false;
            
            for (uint32 j = 0; j < n; ++j) {
                if (A[i * n + j] != 0) degree_[i]++;
            }
        }
        
        for (uint32 iter = 0; iter < n; ++iter) {
            uint32 minDegree = n + 1;
            uint32 minIdx = 0;
            
            for (uint32 i = 0; i < n; ++i) {
                if (!eliminated_[i] && degree_[i] < minDegree) {
                    minDegree = degree_[i];
                    minIdx = i;
                }
            }
            
            permutation_[iter] = minIdx;
            invPermutation_[minIdx] = iter;
            eliminated_[minIdx] = true;
            
            for (uint32 i = 0; i < n; ++i) {
                if (!eliminated_[i] && A[minIdx * n + i] != 0) {
                    degree_[i]++;
                }
            }
        }
    }
    
    bool factorize(const float64* A, uint32 n) {
        float64* L = workspace_;
        
        for (uint32 i = 0; i < n * n; ++i) {
            L[i] = 0;
        }
        
        for (uint32 j = 0; j < n; ++j) {
            uint32 pj = permutation_[j];
            
            for (uint32 i = j; i < n; ++i) {
                uint32 pi = permutation_[i];
                L[i * n + j] = A[pi * n + pj];
            }
            
            for (uint32 k = 0; k < j; ++k) {
                float64 Ljk = L[j * n + k];
                if (abs(Ljk) < config_.dropTolerance) continue;
                
                for (uint32 i = j; i < n; ++i) {
                    L[i * n + j] -= L[i * n + k] * Ljk;
                }
            }
            
            float64 diag = L[j * n + j];
            if (diag <= 0) {
                diag = config_.dropTolerance;
            }
            float64 invDiag = 1.0 / sqrt(diag);
            
            for (uint32 i = j; i < n; ++i) {
                L[i * n + j] *= invDiag;
            }
        }
        
        L_nnz_ = 0;
        L_rowPtr_[0] = 0;
        
        for (uint32 j = 0; j < n; ++j) {
            for (uint32 i = j; i < n; ++i) {
                if (abs(L[i * n + j]) > config_.dropTolerance) {
                    L_values_[L_nnz_] = L[i * n + j];
                    L_colIdx_[L_nnz_] = i;
                    L_nnz_++;
                }
            }
            L_rowPtr_[j + 1] = L_nnz_;
        }
        
        return true;
    }
    
    void solve(const float64* b, float64* x, uint32 n) {
        float64* y = workspace_;
        
        for (uint32 i = 0; i < n; ++i) {
            float64 sum = b[permutation_[i]];
            uint32 rowStart = L_rowPtr_[i];
            uint32 rowEnd = L_rowPtr_[i + 1];
            
            for (uint32 k = rowStart; k < rowEnd; ++k) {
                if (L_colIdx_[k] < i) {
                    sum -= L_values_[k] * y[L_colIdx_[k]];
                }
            }
            
            float64 diag = 1.0;
            for (uint32 k = rowStart; k < rowEnd; ++k) {
                if (L_colIdx_[k] == i) {
                    diag = L_values_[k];
                    break;
                }
            }
            y[i] = sum / diag;
        }
        
        for (int32 i = n - 1; i >= 0; --i) {
            float64 sum = y[i];
            uint32 rowStart = L_rowPtr_[i];
            uint32 rowEnd = L_rowPtr_[i + 1];
            
            for (uint32 k = rowStart; k < rowEnd; ++k) {
                if (L_colIdx_[k] > static_cast<uint32>(i)) {
                    sum -= L_values_[k] * x[invPermutation_[L_colIdx_[k]]];
                }
            }
            
            float64 diag = 1.0;
            for (uint32 k = rowStart; k < rowEnd; ++k) {
                if (L_colIdx_[k] == static_cast<uint32>(i)) {
                    diag = L_values_[k];
                    break;
                }
            }
            x[invPermutation_[i]] = sum / diag;
        }
    }
    
    void solveL(const float64* b, float64* y, uint32 n) {
        for (uint32 i = 0; i < n; ++i) {
            float64 sum = b[permutation_[i]];
            uint32 rowStart = L_rowPtr_[i];
            uint32 rowEnd = L_rowPtr_[i + 1];
            
            for (uint32 k = rowStart; k < rowEnd; ++k) {
                if (L_colIdx_[k] < i) {
                    sum -= L_values_[k] * y[L_colIdx_[k]];
                }
            }
            
            float64 diag = 1.0;
            for (uint32 k = rowStart; k < rowEnd; ++k) {
                if (L_colIdx_[k] == i) {
                    diag = L_values_[k];
                    break;
                }
            }
            y[i] = sum / diag;
        }
    }
    
    void solveLt(const float64* y, float64* x, uint32 n) {
        for (int32 i = n - 1; i >= 0; --i) {
            float64 sum = y[i];
            uint32 rowStart = L_rowPtr_[i];
            uint32 rowEnd = L_rowPtr_[i + 1];
            
            for (uint32 k = rowStart; k < rowEnd; ++k) {
                if (L_colIdx_[k] > static_cast<uint32>(i)) {
                    sum -= L_values_[k] * x[invPermutation_[L_colIdx_[k]]];
                }
            }
            
            float64 diag = 1.0;
            for (uint32 k = rowStart; k < rowEnd; ++k) {
                if (L_colIdx_[k] == static_cast<uint32>(i)) {
                    diag = L_values_[k];
                    break;
                }
            }
            x[invPermutation_[i]] = sum / diag;
        }
    }
    
    uint32 getNonZeros() const { return L_nnz_; }
    float64 getFillRatio() const {
        uint32 n = config_.maxMatrixSize;
        return static_cast<float64>(L_nnz_) / (n * n);
    }
};

}

#endif
