#ifndef MSCKF_CORE_CONSISTENCY_CHECKER_HPP
#define MSCKF_CORE_CONSISTENCY_CHECKER_HPP

#include "state.hpp"
#include "../math/types.hpp"
#include "../math/matrix.hpp"

namespace msckf {

class ConsistencyChecker {
public:
    struct Config {
        float64 conditionNumberThreshold;
        float64 velocityJumpThreshold;
        float64 attitudeJumpThreshold;
        float64 positionDriftThreshold;
        uint32 checkInterval;
        bool enableLogging;
        
        Config() 
            : conditionNumberThreshold(1e6)
            , velocityJumpThreshold(3.0)
            , attitudeJumpThreshold(0.1)
            , positionDriftThreshold(0.5)
            , checkInterval(10)
            , enableLogging(true) {}
    };
    
    struct DiagnosticInfo {
        float64 conditionNumber;
        float64 minEigenvalue;
        float64 maxEigenvalue;
        float64 velocityJump;
        float64 attitudeJump;
        float64 positionDrift;
        uint32 nullspaceDimension;
        bool isPositiveDefinite;
        bool isConsistent;
        uint64 timestamp;
        
        DiagnosticInfo() 
            : conditionNumber(0)
            , minEigenvalue(0)
            , maxEigenvalue(0)
            , velocityJump(0)
            , attitudeJump(0)
            , positionDrift(0)
            , nullspaceDimension(0)
            , isPositiveDefinite(true)
            , isConsistent(true)
            , timestamp(0) {}
    };

private:
    Config config_;
    DiagnosticInfo lastDiagnostic_;
    
    MSCKFState prevState_;
    bool hasPrevState_;
    uint32 checkCounter_;
    
    float64 totalPositionDrift_;
    uint64 driftStartTime_;
    bool isDriftTracking_;

public:
    ConsistencyChecker() 
        : hasPrevState_(false)
        , checkCounter_(0)
        , totalPositionDrift_(0)
        , driftStartTime_(0)
        , isDriftTracking_(false) {}
    
    void init(const Config& config) {
        config_ = config;
    }
    
    bool check(const MSCKFState& state, DiagnosticInfo& info) {
        checkCounter_++;
        
        if (checkCounter_ % config_.checkInterval != 0) {
            info = lastDiagnostic_;
            return info.isConsistent;
        }
        
        info.timestamp = state.timestamp;
        
        checkCovarianceConditionNumber(state, info);
        
        checkPositiveDefiniteness(state, info);
        
        if (hasPrevState_) {
            checkStateConsistency(state, prevState_, info);
        }
        
        checkNullspaceDimension(state, info);
        
        info.isConsistent = info.isPositiveDefinite && 
                           (info.conditionNumber < config_.conditionNumberThreshold) &&
                           (info.velocityJump < config_.velocityJumpThreshold) &&
                           (info.attitudeJump < config_.attitudeJumpThreshold);
        
        prevState_ = state;
        hasPrevState_ = true;
        lastDiagnostic_ = info;
        
        return info.isConsistent;
    }
    
    void startDriftTracking(uint64 startTime) {
        driftStartTime_ = startTime;
        isDriftTracking_ = true;
        totalPositionDrift_ = 0;
    }
    
    void updateDriftTracking(const MSCKFState& state, float64 dt) {
        if (!isDriftTracking_) return;
        
        totalPositionDrift_ += state.imuState.velocity.norm() * dt;
    }
    
    float64 getDriftRate(uint64 currentTime) const {
        if (!isDriftTracking_) return 0;
        
        float64 elapsed = (currentTime - driftStartTime_) * 1e-6;
        if (elapsed < 1.0) return 0;
        
        return totalPositionDrift_ / elapsed;
    }
    
    const DiagnosticInfo& getLastDiagnostic() const {
        return lastDiagnostic_;
    }
    
    void reset() {
        hasPrevState_ = false;
        checkCounter_ = 0;
        totalPositionDrift_ = 0;
        isDriftTracking_ = false;
    }

private:
    void checkCovarianceConditionNumber(const MSCKFState& state, DiagnosticInfo& info) {
        uint32 dim = state.covarianceDim;
        
        float64 minDiag = state.covariance(0, 0);
        float64 maxDiag = state.covariance(0, 0);
        
        for (uint32 i = 1; i < dim; ++i) {
            float64 diag = state.covariance(i, i);
            if (diag < minDiag) minDiag = diag;
            if (diag > maxDiag) maxDiag = diag;
        }
        
        info.minEigenvalue = minDiag;
        info.maxEigenvalue = maxDiag;
        
        if (minDiag > 1e-15) {
            info.conditionNumber = maxDiag / minDiag;
        } else {
            info.conditionNumber = 1e15;
        }
    }
    
    void checkPositiveDefiniteness(const MSCKFState& state, DiagnosticInfo& info) {
        uint32 dim = state.covarianceDim;
        
        info.isPositiveDefinite = true;
        
        for (uint32 i = 0; i < dim; ++i) {
            if (state.covariance(i, i) <= 0) {
                info.isPositiveDefinite = false;
                return;
            }
        }
        
        for (uint32 i = 0; i < dim; ++i) {
            for (uint32 j = i + 1; j < dim; ++j) {
                float64 diag_i = state.covariance(i, i);
                float64 diag_j = state.covariance(j, j);
                float64 offDiag = state.covariance(i, j);
                
                if (offDiag * offDiag >= diag_i * diag_j) {
                    info.isPositiveDefinite = false;
                    return;
                }
            }
        }
    }
    
    void checkStateConsistency(const MSCKFState& curr, const MSCKFState& prev, 
                               DiagnosticInfo& info) {
        float64 dt = (curr.timestamp - prev.timestamp) * 1e-6;
        
        if (dt < 1e-6) {
            info.velocityJump = 0;
            info.attitudeJump = 0;
            return;
        }
        
        Vector3d deltaV = curr.imuState.velocity - prev.imuState.velocity;
        info.velocityJump = deltaV.norm() / dt;
        
        Quaterniond dq = curr.imuState.orientation * prev.imuState.orientation.conjugate();
        Vector3d dtheta = dq.log();
        info.attitudeJump = dtheta.norm() / dt;
        
        Vector3d deltaP = curr.imuState.position - prev.imuState.position;
        info.positionDrift = deltaP.norm() / dt;
    }
    
    void checkNullspaceDimension(const MSCKFState& state, DiagnosticInfo& info) {
        info.nullspaceDimension = NULLSPACE_DIM;
    }
};

}

#endif
