#ifndef MSCKF_TESTS_TEST_SAFETY_HPP
#define MSCKF_TESTS_TEST_SAFETY_HPP

#include "test_framework.hpp"
#include "../include/core/safety_guard.hpp"
#include "../include/core/consistency_checker.hpp"
#include "../include/core/state.hpp"
#include "../include/math/quaternion.hpp"
#include "../include/math/matrix.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_safety_guard_default_constructor() {
    TEST_BEGIN("Safety guard default constructor");
    
    SafetyGuard guard;
    
    TEST_ASSERT_TRUE(guard.isInitialized());
    
    TEST_END();
}

bool test_safety_guard_init() {
    TEST_BEGIN("Safety guard init");
    
    SafetyGuard guard;
    SafetyGuard::Config config;
    config.maxPositionError = 10.0;
    config.maxVelocityError = 5.0;
    config.maxAttitudeError = 0.5;
    config.maxGyroBias = 0.1;
    config.maxAccelBias = 0.5;
    
    guard.init(config);
    
    TEST_ASSERT_TRUE(guard.isInitialized());
    
    TEST_END();
}

bool test_safety_guard_check_state_valid() {
    TEST_BEGIN("Safety guard check state valid");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.imuState.position = Vector3d({0, 0, 0});
    state.imuState.velocity = Vector3d({0, 0, 0});
    state.imuState.orientation = Quaterniond();
    
    SafetyStatus status = guard.checkState(state);
    
    TEST_ASSERT_TRUE(status == SafetyStatus::OK || status == SafetyStatus::WARNING);
    
    TEST_END();
}

bool test_safety_guard_check_large_position() {
    TEST_BEGIN("Safety guard check large position");
    
    SafetyGuard guard;
    SafetyGuard::Config config;
    config.maxPositionError = 1.0;
    guard.init(config);
    
    MSCKFState state;
    state.imuState.position = Vector3d({1000, 1000, 1000});
    
    SafetyStatus status = guard.checkState(state);
    
    TEST_ASSERT_TRUE(status != SafetyStatus::OK);
    
    TEST_END();
}

bool test_safety_guard_check_large_velocity() {
    TEST_BEGIN("Safety guard check large velocity");
    
    SafetyGuard guard;
    SafetyGuard::Config config;
    config.maxVelocityError = 1.0;
    guard.init(config);
    
    MSCKFState state;
    state.imuState.velocity = Vector3d({100, 100, 100});
    
    SafetyStatus status = guard.checkState(state);
    
    TEST_ASSERT_TRUE(status != SafetyStatus::OK);
    
    TEST_END();
}

bool test_safety_guard_check_large_attitude() {
    TEST_BEGIN("Safety guard check large attitude");
    
    SafetyGuard guard;
    SafetyGuard::Config config;
    config.maxAttitudeError = 0.1;
    guard.init(config);
    
    MSCKFState state;
    state.imuState.orientation = Quaterniond::fromEulerAngles(1.0, 1.0, 1.0);
    
    SafetyStatus status = guard.checkState(state);
    
    TEST_ASSERT_TRUE(status != SafetyStatus::OK);
    
    TEST_END();
}

bool test_safety_guard_check_nan_position() {
    TEST_BEGIN("Safety guard check nan position");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.imuState.position[0] = std::nan("");
    
    SafetyStatus status = guard.checkState(state);
    
    TEST_ASSERT_TRUE(status == SafetyStatus::CRITICAL);
    
    TEST_END();
}

bool test_safety_guard_check_nan_velocity() {
    TEST_BEGIN("Safety guard check nan velocity");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.imuState.velocity[1] = std::nan("");
    
    SafetyStatus status = guard.checkState(state);
    
    TEST_ASSERT_TRUE(status == SafetyStatus::CRITICAL);
    
    TEST_END();
}

bool test_safety_guard_check_nan_orientation() {
    TEST_BEGIN("Safety guard check nan orientation");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.imuState.orientation.w = std::nan("");
    
    SafetyStatus status = guard.checkState(state);
    
    TEST_ASSERT_TRUE(status == SafetyStatus::CRITICAL);
    
    TEST_END();
}

bool test_safety_guard_check_covariance_valid() {
    TEST_BEGIN("Safety guard check covariance valid");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    for (uint32 i = 0; i < state.covarianceDim; ++i) {
        state.covariance(i, i) = 1.0;
    }
    
    SafetyStatus status = guard.checkCovariance(state);
    
    TEST_ASSERT_TRUE(status == SafetyStatus::OK || status == SafetyStatus::WARNING);
    
    TEST_END();
}

bool test_safety_guard_check_covariance_negative_diagonal() {
    TEST_BEGIN("Safety guard check covariance negative diagonal");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.covariance(0, 0) = -1.0;
    
    SafetyStatus status = guard.checkCovariance(state);
    
    TEST_ASSERT_TRUE(status != SafetyStatus::OK);
    
    TEST_END();
}

bool test_safety_guard_check_covariance_nan() {
    TEST_BEGIN("Safety guard check covariance nan");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.covariance(0, 0) = std::nan("");
    
    SafetyStatus status = guard.checkCovariance(state);
    
    TEST_ASSERT_TRUE(status == SafetyStatus::CRITICAL);
    
    TEST_END();
}

bool test_safety_guard_enforce_positive_definite() {
    TEST_BEGIN("Safety guard enforce positive definite");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.covariance(0, 0) = -1.0;
    state.covariance(1, 1) = 0.0;
    state.covariance(2, 2) = -0.5;
    
    guard.enforceCovariancePositiveDefinite(state);
    
    TEST_ASSERT_TRUE(state.covariance(0, 0) > 0);
    TEST_ASSERT_TRUE(state.covariance(1, 1) > 0);
    TEST_ASSERT_TRUE(state.covariance(2, 2) > 0);
    
    TEST_END();
}

bool test_safety_guard_normalize_quaternion() {
    TEST_BEGIN("Safety guard normalize quaternion");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.imuState.orientation = Quaterniond(2.0, 2.0, 2.0, 2.0);
    
    guard.normalizeQuaternion(state);
    
    float64 norm = state.imuState.orientation.norm();
    TEST_ASSERT_NEAR(1.0, norm, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_safety_guard_reset() {
    TEST_BEGIN("Safety guard reset");
    
    SafetyGuard guard;
    guard.init(SafetyGuard::Config());
    
    MSCKFState state;
    state.imuState.position[0] = std::nan("");
    guard.checkState(state);
    
    guard.reset();
    
    TEST_ASSERT_TRUE(guard.isInitialized());
    
    TEST_END();
}

bool test_consistency_checker_default_constructor() {
    TEST_BEGIN("Consistency checker default constructor");
    
    ConsistencyChecker checker;
    
    TEST_ASSERT_TRUE(checker.isInitialized());
    
    TEST_END();
}

bool test_consistency_checker_init() {
    TEST_BEGIN("Consistency checker init");
    
    ConsistencyChecker checker;
    ConsistencyChecker::Config config;
    config.windowSize = 100;
    config.positionThreshold = 5.0;
    config.rotationThreshold = 0.1;
    
    checker.init(config);
    
    TEST_ASSERT_TRUE(checker.isInitialized());
    
    TEST_END();
}

bool test_consistency_checker_add_measurement() {
    TEST_BEGIN("Consistency checker add measurement");
    
    ConsistencyChecker checker;
    checker.init(ConsistencyChecker::Config());
    
    MSCKFState state;
    state.imuState.position = Vector3d({0, 0, 0});
    
    checker.addMeasurement(state);
    
    TEST_ASSERT_TRUE(checker.getMeasurementCount() > 0);
    
    TEST_END();
}

bool test_consistency_checker_check_consistency() {
    TEST_BEGIN("Consistency checker check consistency");
    
    ConsistencyChecker checker;
    checker.init(ConsistencyChecker::Config());
    
    MSCKFState state;
    for (uint32 i = 0; i < 10; ++i) {
        state.imuState.position = Vector3d({static_cast<float64>(i), 0, 0});
        checker.addMeasurement(state);
    }
    
    bool consistent = checker.checkConsistency();
    
    TEST_ASSERT_TRUE(consistent);
    
    TEST_END();
}

bool test_consistency_checker_nees_computation() {
    TEST_BEGIN("Consistency checker NEES computation");
    
    ConsistencyChecker checker;
    checker.init(ConsistencyChecker::Config());
    
    MSCKFState state;
    state.imuState.position = Vector3d({0, 0, 0});
    state.covariance(0, 0) = 1.0;
    state.covariance(1, 1) = 1.0;
    state.covariance(2, 2) = 1.0;
    
    checker.addMeasurement(state);
    
    float64 nees = checker.computeNEES();
    
    TEST_ASSERT_TRUE(nees >= 0);
    
    TEST_END();
}

bool test_consistency_checker_reset() {
    TEST_BEGIN("Consistency checker reset");
    
    ConsistencyChecker checker;
    checker.init(ConsistencyChecker::Config());
    
    MSCKFState state;
    checker.addMeasurement(state);
    
    checker.reset();
    
    TEST_ASSERT_EQ(0u, checker.getMeasurementCount());
    
    TEST_END();
}

bool test_consistency_checker_window_size() {
    TEST_BEGIN("Consistency checker window size");
    
    ConsistencyChecker checker;
    ConsistencyChecker::Config config;
    config.windowSize = 5;
    checker.init(config);
    
    MSCKFState state;
    for (uint32 i = 0; i < 10; ++i) {
        checker.addMeasurement(state);
    }
    
    TEST_ASSERT_TRUE(checker.getMeasurementCount() <= 5);
    
    TEST_END();
}

REGISTER_TEST("Safety guard default constructor", test_safety_guard_default_constructor, "Safety");
REGISTER_TEST("Safety guard init", test_safety_guard_init, "Safety");
REGISTER_TEST("Safety guard check state valid", test_safety_guard_check_state_valid, "Safety");
REGISTER_TEST("Safety guard check large position", test_safety_guard_check_large_position, "Safety");
REGISTER_TEST("Safety guard check large velocity", test_safety_guard_check_large_velocity, "Safety");
REGISTER_TEST("Safety guard check large attitude", test_safety_guard_check_large_attitude, "Safety");
REGISTER_TEST("Safety guard check nan position", test_safety_guard_check_nan_position, "Safety");
REGISTER_TEST("Safety guard check nan velocity", test_safety_guard_check_nan_velocity, "Safety");
REGISTER_TEST("Safety guard check nan orientation", test_safety_guard_check_nan_orientation, "Safety");
REGISTER_TEST("Safety guard check covariance valid", test_safety_guard_check_covariance_valid, "Safety");
REGISTER_TEST("Safety guard check covariance negative diagonal", test_safety_guard_check_covariance_negative_diagonal, "Safety");
REGISTER_TEST("Safety guard check covariance nan", test_safety_guard_check_covariance_nan, "Safety");
REGISTER_TEST("Safety guard enforce positive definite", test_safety_guard_enforce_positive_definite, "Safety");
REGISTER_TEST("Safety guard normalize quaternion", test_safety_guard_normalize_quaternion, "Safety");
REGISTER_TEST("Safety guard reset", test_safety_guard_reset, "Safety");
REGISTER_TEST("Consistency checker default constructor", test_consistency_checker_default_constructor, "Safety");
REGISTER_TEST("Consistency checker init", test_consistency_checker_init, "Safety");
REGISTER_TEST("Consistency checker add measurement", test_consistency_checker_add_measurement, "Safety");
REGISTER_TEST("Consistency checker check consistency", test_consistency_checker_check_consistency, "Safety");
REGISTER_TEST("Consistency checker NEES computation", test_consistency_checker_nees_computation, "Safety");
REGISTER_TEST("Consistency checker reset", test_consistency_checker_reset, "Safety");
REGISTER_TEST("Consistency checker window size", test_consistency_checker_window_size, "Safety");

}
}

#endif
