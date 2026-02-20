#ifndef MSCKF_TESTS_TEST_IMU_PROPAGATOR_HPP
#define MSCKF_TESTS_TEST_IMU_PROPAGATOR_HPP

#include "test_framework.hpp"
#include "../include/core/imu_propagator.hpp"
#include "../include/core/state.hpp"
#include "../include/hal/imu_types.hpp"
#include "../include/math/quaternion.hpp"
#include "../include/math/matrix.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_imu_propagator_default_constructor() {
    TEST_BEGIN("IMU propagator default constructor");
    
    IMUPropagator propagator;
    
    TEST_ASSERT_TRUE(propagator.isInitialized());
    
    TEST_END();
}

bool test_imu_propagator_init() {
    TEST_BEGIN("IMU propagator init");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    config.gyroNoiseDensity = 1e-4;
    config.accelNoiseDensity = 1e-3;
    config.gyroBiasRandomWalk = 1e-5;
    config.accelBiasRandomWalk = 1e-4;
    
    propagator.init(config);
    
    TEST_ASSERT_TRUE(propagator.isInitialized());
    
    TEST_END();
}

bool test_imu_propagator_static_propagation() {
    TEST_BEGIN("IMU propagator static propagation");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    propagator.init(config);
    
    MSCKFState state;
    state.imuState.position = Vector3d({0, 0, 0});
    state.imuState.velocity = Vector3d({0, 0, 0});
    state.imuState.orientation = Quaterniond();
    state.timestamp = 0;
    
    IMUData imuData;
    imuData.accel = Vector3d({0, 0, 9.81});
    imuData.gyro = Vector3d({0, 0, 0});
    imuData.timestamp = 10000;
    
    propagator.propagate(state, imuData);
    
    TEST_ASSERT_NEAR(0.0, state.imuState.position[0], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, state.imuState.position[1], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, state.imuState.position[2], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_imu_propagator_free_fall() {
    TEST_BEGIN("IMU propagator free fall");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    propagator.init(config);
    
    MSCKFState state;
    state.imuState.position = Vector3d({0, 0, 100});
    state.imuState.velocity = Vector3d({0, 0, 0});
    state.imuState.orientation = Quaterniond();
    state.timestamp = 0;
    
    IMUData imuData;
    imuData.accel = Vector3d({0, 0, 0});
    imuData.gyro = Vector3d({0, 0, 0});
    imuData.timestamp = 100000;
    
    propagator.propagate(state, imuData);
    
    float64 dt = 0.1;
    float64 expectedVel = -9.81 * dt;
    float64 expectedPos = 100 + 0.5 * (-9.81) * dt * dt;
    
    TEST_ASSERT_NEAR(expectedVel, state.imuState.velocity[2], 0.1);
    TEST_ASSERT_NEAR(expectedPos, state.imuState.position[2], 0.5);
    
    TEST_END();
}

bool test_imu_propagator_constant_velocity() {
    TEST_BEGIN("IMU propagator constant velocity");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    propagator.init(config);
    
    MSCKFState state;
    state.imuState.position = Vector3d({0, 0, 0});
    state.imuState.velocity = Vector3d({10, 0, 0});
    state.imuState.orientation = Quaterniond();
    state.timestamp = 0;
    
    IMUData imuData;
    imuData.accel = Vector3d({0, 0, 9.81});
    imuData.gyro = Vector3d({0, 0, 0});
    imuData.timestamp = 100000;
    
    propagator.propagate(state, imuData);
    
    float64 dt = 0.1;
    float64 expectedPos = 10 * dt;
    
    TEST_ASSERT_NEAR(expectedPos, state.imuState.position[0], 0.1);
    TEST_ASSERT_NEAR(10.0, state.imuState.velocity[0], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_imu_propagator_rotation() {
    TEST_BEGIN("IMU propagator rotation");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    propagator.init(config);
    
    MSCKFState state;
    state.imuState.orientation = Quaterniond();
    state.timestamp = 0;
    
    IMUData imuData;
    imuData.accel = Vector3d({0, 0, 9.81});
    imuData.gyro = Vector3d({0, 0, 1.0});
    imuData.timestamp = 100000;
    
    propagator.propagate(state, imuData);
    
    float64 dt = 0.1;
    float64 expectedAngle = 1.0 * dt;
    
    Vector3d euler = state.imuState.orientation.toEulerAngles();
    TEST_ASSERT_NEAR(expectedAngle, euler[2], 0.01);
    
    TEST_END();
}

bool test_imu_propagator_covariance_growth() {
    TEST_BEGIN("IMU propagator covariance growth");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    config.gyroNoiseDensity = 1e-4;
    config.accelNoiseDensity = 1e-3;
    propagator.init(config);
    
    MSCKFState state;
    state.timestamp = 0;
    
    float64 initialCov = state.covariance(0, 0);
    
    IMUData imuData;
    imuData.accel = Vector3d({0, 0, 9.81});
    imuData.gyro = Vector3d({0, 0, 0});
    imuData.timestamp = 100000;
    
    propagator.propagate(state, imuData);
    
    TEST_ASSERT_TRUE(state.covariance(0, 0) >= initialCov);
    
    TEST_END();
}

bool test_imu_propagator_bias_propagation() {
    TEST_BEGIN("IMU propagator bias propagation");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    propagator.init(config);
    
    MSCKFState state;
    state.imuState.gyroBias = Vector3d({0.01, 0.02, 0.03});
    state.imuState.accelBias = Vector3d({0.1, 0.2, 0.3});
    state.timestamp = 0;
    
    IMUData imuData;
    imuData.accel = Vector3d({0, 0, 9.81});
    imuData.gyro = Vector3d({0, 0, 0});
    imuData.timestamp = 100000;
    
    propagator.propagate(state, imuData);
    
    TEST_ASSERT_NEAR(0.01, state.imuState.gyroBias[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.02, state.imuState.gyroBias[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.03, state.imuState.gyroBias[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_imu_propagator_timestamp_update() {
    TEST_BEGIN("IMU propagator timestamp update");
    
    IMUPropagator propagator;
    propagator.init(IMUPropagator::Config());
    
    MSCKFState state;
    state.timestamp = 0;
    
    IMUData imuData;
    imuData.timestamp = 12345;
    
    propagator.propagate(state, imuData);
    
    TEST_ASSERT_EQ(12345u, state.timestamp);
    
    TEST_END();
}

bool test_imu_propagator_multiple_propagations() {
    TEST_BEGIN("IMU propagator multiple propagations");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    propagator.init(config);
    
    MSCKFState state;
    state.imuState.position = Vector3d({0, 0, 0});
    state.imuState.velocity = Vector3d({1, 0, 0});
    state.timestamp = 0;
    
    for (uint32 i = 0; i < 10; ++i) {
        IMUData imuData;
        imuData.accel = Vector3d({0, 0, 9.81});
        imuData.gyro = Vector3d({0, 0, 0});
        imuData.timestamp = (i + 1) * 10000;
        
        propagator.propagate(state, imuData);
    }
    
    float64 totalTime = 0.1;
    float64 expectedPos = 1.0 * totalTime;
    
    TEST_ASSERT_NEAR(expectedPos, state.imuState.position[0], 0.05);
    TEST_ASSERT_NEAR(1.0, state.imuState.velocity[0], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_imu_propagator_zero_dt() {
    TEST_BEGIN("IMU propagator zero dt");
    
    IMUPropagator propagator;
    propagator.init(IMUPropagator::Config());
    
    MSCKFState state;
    state.imuState.position = Vector3d({1, 2, 3});
    state.timestamp = 10000;
    
    IMUData imuData;
    imuData.timestamp = 10000;
    
    propagator.propagate(state, imuData);
    
    TEST_ASSERT_NEAR(1.0, state.imuState.position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, state.imuState.position[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, state.imuState.position[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_imu_propagator_large_acceleration() {
    TEST_BEGIN("IMU propagator large acceleration");
    
    IMUPropagator propagator;
    IMUPropagator::Config config;
    config.gravity = 9.81;
    propagator.init(config);
    
    MSCKFState state;
    state.imuState.orientation = Quaterniond();
    state.timestamp = 0;
    
    IMUData imuData;
    imuData.accel = Vector3d({100, 0, 9.81});
    imuData.gyro = Vector3d({0, 0, 0});
    imuData.timestamp = 10000;
    
    propagator.propagate(state, imuData);
    
    TEST_ASSERT_TRUE(state.imuState.velocity[0] > 0);
    
    TEST_END();
}

bool test_imu_propagator_negative_dt() {
    TEST_BEGIN("IMU propagator negative dt");
    
    IMUPropagator propagator;
    propagator.init(IMUPropagator::Config());
    
    MSCKFState state;
    state.imuState.position = Vector3d({1, 2, 3});
    state.timestamp = 20000;
    
    IMUData imuData;
    imuData.timestamp = 10000;
    
    propagator.propagate(state, imuData);
    
    TEST_ASSERT_NEAR(1.0, state.imuState.position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, state.imuState.position[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, state.imuState.position[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_imu_propagator_quaternion_normalization() {
    TEST_BEGIN("IMU propagator quaternion normalization");
    
    IMUPropagator propagator;
    propagator.init(IMUPropagator::Config());
    
    MSCKFState state;
    state.timestamp = 0;
    
    for (uint32 i = 0; i < 100; ++i) {
        IMUData imuData;
        imuData.accel = Vector3d({0, 0, 9.81});
        imuData.gyro = Vector3d({0.1, 0.2, 0.3});
        imuData.timestamp = (i + 1) * 10000;
        
        propagator.propagate(state, imuData);
    }
    
    float64 norm = state.imuState.orientation.norm();
    TEST_ASSERT_NEAR(1.0, norm, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

REGISTER_TEST("IMU propagator default constructor", test_imu_propagator_default_constructor, "IMUPropagator");
REGISTER_TEST("IMU propagator init", test_imu_propagator_init, "IMUPropagator");
REGISTER_TEST("IMU propagator static propagation", test_imu_propagator_static_propagation, "IMUPropagator");
REGISTER_TEST("IMU propagator free fall", test_imu_propagator_free_fall, "IMUPropagator");
REGISTER_TEST("IMU propagator constant velocity", test_imu_propagator_constant_velocity, "IMUPropagator");
REGISTER_TEST("IMU propagator rotation", test_imu_propagator_rotation, "IMUPropagator");
REGISTER_TEST("IMU propagator covariance growth", test_imu_propagator_covariance_growth, "IMUPropagator");
REGISTER_TEST("IMU propagator bias propagation", test_imu_propagator_bias_propagation, "IMUPropagator");
REGISTER_TEST("IMU propagator timestamp update", test_imu_propagator_timestamp_update, "IMUPropagator");
REGISTER_TEST("IMU propagator multiple propagations", test_imu_propagator_multiple_propagations, "IMUPropagator");
REGISTER_TEST("IMU propagator zero dt", test_imu_propagator_zero_dt, "IMUPropagator");
REGISTER_TEST("IMU propagator large acceleration", test_imu_propagator_large_acceleration, "IMUPropagator");
REGISTER_TEST("IMU propagator negative dt", test_imu_propagator_negative_dt, "IMUPropagator");
REGISTER_TEST("IMU propagator quaternion normalization", test_imu_propagator_quaternion_normalization, "IMUPropagator");

}
}

#endif
