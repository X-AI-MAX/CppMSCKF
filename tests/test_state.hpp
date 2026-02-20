#ifndef MSCKF_TESTS_TEST_STATE_HPP
#define MSCKF_TESTS_TEST_STATE_HPP

#include "test_framework.hpp"
#include "../include/core/state.hpp"
#include "../include/math/quaternion.hpp"
#include "../include/math/matrix.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_imu_state_default_constructor() {
    TEST_BEGIN("IMU state default constructor");
    
    IMUState state;
    
    TEST_ASSERT_NEAR(0.0, state.position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.position[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.position[2], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.velocity[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.velocity[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.velocity[2], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(1.0, state.orientation.w, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.orientation.x, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.orientation.y, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.orientation.z, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_imu_state_set_position() {
    TEST_BEGIN("IMU state set position");
    
    IMUState state;
    state.position = Vector3d({1.0, 2.0, 3.0});
    
    TEST_ASSERT_NEAR(1.0, state.position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, state.position[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, state.position[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_imu_state_set_velocity() {
    TEST_BEGIN("IMU state set velocity");
    
    IMUState state;
    state.velocity = Vector3d({4.0, 5.0, 6.0});
    
    TEST_ASSERT_NEAR(4.0, state.velocity[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(5.0, state.velocity[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(6.0, state.velocity[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_imu_state_set_orientation() {
    TEST_BEGIN("IMU state set orientation");
    
    IMUState state;
    state.orientation = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), PI / 4);
    
    float64 halfAngle = PI / 8;
    TEST_ASSERT_NEAR(cos(halfAngle), state.orientation.w, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, state.orientation.x, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.orientation.y, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(sin(halfAngle), state.orientation.z, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_imu_state_set_bias() {
    TEST_BEGIN("IMU state set bias");
    
    IMUState state;
    state.gyroBias = Vector3d({0.01, 0.02, 0.03});
    state.accelBias = Vector3d({0.1, 0.2, 0.3});
    
    TEST_ASSERT_NEAR(0.01, state.gyroBias[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.02, state.gyroBias[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.03, state.gyroBias[2], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.1, state.accelBias[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.2, state.accelBias[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.3, state.accelBias[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_camera_state_default_constructor() {
    TEST_BEGIN("Camera state default constructor");
    
    CameraState state;
    
    TEST_ASSERT_NEAR(0.0, state.position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.position[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.position[2], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(1.0, state.orientation.w, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.orientation.x, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.orientation.y, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.orientation.z, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_extrinsic_state_default_constructor() {
    TEST_BEGIN("Extrinsic state default constructor");
    
    ExtrinsicState state;
    
    TEST_ASSERT_NEAR(0.0, state.p_CB[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.p_CB[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.p_CB[2], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(1.0, state.q_CB.w, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.q_CB.x, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.q_CB.y, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.q_CB.z, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_extrinsic_state_get_rotation_matrix() {
    TEST_BEGIN("Extrinsic state get rotation matrix");
    
    ExtrinsicState state;
    state.q_CB = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), PI / 2);
    
    Matrix3d R = state.getRotationMatrix();
    
    TEST_ASSERT_NEAR(0.0, R(0, 0), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(-1.0, R(0, 1), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(1.0, R(1, 0), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, R(1, 1), TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_msckf_state_default_constructor() {
    TEST_BEGIN("MSCKF state default constructor");
    
    MSCKFState state;
    
    TEST_ASSERT_EQ(0u, state.numCameraStates);
    TEST_ASSERT_EQ(IMU_STATE_DIM + EXTRINSIC_DIM, state.covarianceDim);
    TEST_ASSERT_EQ(0u, state.timestamp);
    
    TEST_END();
}

bool test_msckf_state_add_camera_state() {
    TEST_BEGIN("MSCKF state add camera state");
    
    MSCKFState state;
    
    CameraState camState;
    camState.position = Vector3d({1.0, 2.0, 3.0});
    camState.orientation = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), PI / 4);
    
    state.addCameraState(camState);
    
    TEST_ASSERT_EQ(1u, state.numCameraStates);
    TEST_ASSERT_NEAR(1.0, state.cameraStates[0].position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, state.cameraStates[0].position[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, state.cameraStates[0].position[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_msckf_state_add_multiple_camera_states() {
    TEST_BEGIN("MSCKF state add multiple camera states");
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 5; ++i) {
        CameraState camState;
        camState.position = Vector3d({static_cast<float64>(i), 0, 0});
        state.addCameraState(camState);
    }
    
    TEST_ASSERT_EQ(5u, state.numCameraStates);
    
    for (uint32 i = 0; i < 5; ++i) {
        TEST_ASSERT_NEAR(static_cast<float64>(i), state.cameraStates[i].position[0], TEST_TOLERANCE);
    }
    
    TEST_END();
}

bool test_msckf_state_remove_camera_state() {
    TEST_BEGIN("MSCKF state remove camera state");
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 5; ++i) {
        CameraState camState;
        camState.position = Vector3d({static_cast<float64>(i), 0, 0});
        state.addCameraState(camState);
    }
    
    state.removeCameraState(0);
    
    TEST_ASSERT_EQ(4u, state.numCameraStates);
    TEST_ASSERT_NEAR(1.0, state.cameraStates[0].position[0], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_msckf_state_remove_middle_camera_state() {
    TEST_BEGIN("MSCKF state remove middle camera state");
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 5; ++i) {
        CameraState camState;
        camState.position = Vector3d({static_cast<float64>(i), 0, 0});
        state.addCameraState(camState);
    }
    
    state.removeCameraState(2);
    
    TEST_ASSERT_EQ(4u, state.numCameraStates);
    TEST_ASSERT_NEAR(0.0, state.cameraStates[0].position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(1.0, state.cameraStates[1].position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, state.cameraStates[2].position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(4.0, state.cameraStates[3].position[0], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_msckf_state_state_dim() {
    TEST_BEGIN("MSCKF state state dim");
    
    MSCKFState state;
    
    uint32 dim0 = state.stateDim();
    TEST_ASSERT_EQ(IMU_STATE_DIM + EXTRINSIC_DIM, dim0);
    
    CameraState camState;
    state.addCameraState(camState);
    
    uint32 dim1 = state.stateDim();
    TEST_ASSERT_EQ(IMU_STATE_DIM + EXTRINSIC_DIM + CAMERA_STATE_DIM, dim1);
    
    TEST_END();
}

bool test_msckf_state_camera_state_index() {
    TEST_BEGIN("MSCKF state camera state index");
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 3; ++i) {
        CameraState camState;
        state.addCameraState(camState);
    }
    
    uint32 idx0 = state.cameraStateIndex(0);
    uint32 idx1 = state.cameraStateIndex(1);
    uint32 idx2 = state.cameraStateIndex(2);
    
    TEST_ASSERT_EQ(IMU_STATE_DIM + EXTRINSIC_DIM, idx0);
    TEST_ASSERT_EQ(IMU_STATE_DIM + EXTRINSIC_DIM + CAMERA_STATE_DIM, idx1);
    TEST_ASSERT_EQ(IMU_STATE_DIM + EXTRINSIC_DIM + 2 * CAMERA_STATE_DIM, idx2);
    
    TEST_END();
}

bool test_msckf_state_covariance_dimension() {
    TEST_BEGIN("MSCKF state covariance dimension");
    
    MSCKFState state;
    
    TEST_ASSERT_EQ(IMU_STATE_DIM + EXTRINSIC_DIM, state.covarianceDim);
    
    CameraState camState;
    state.addCameraState(camState);
    
    TEST_ASSERT_EQ(IMU_STATE_DIM + EXTRINSIC_DIM + CAMERA_STATE_DIM, state.covarianceDim);
    
    TEST_END();
}

bool test_msckf_state_inject_error_state() {
    TEST_BEGIN("MSCKF state inject error state");
    
    MSCKFState state;
    state.imuState.position = Vector3d({1.0, 2.0, 3.0});
    
    ErrorState delta;
    delta[0] = 0.1;
    delta[1] = 0.2;
    delta[2] = 0.3;
    
    state.injectErrorState(delta);
    
    TEST_ASSERT_NEAR(1.1, state.imuState.position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.2, state.imuState.position[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.3, state.imuState.position[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_msckf_state_inject_error_state_velocity() {
    TEST_BEGIN("MSCKF state inject error state velocity");
    
    MSCKFState state;
    state.imuState.velocity = Vector3d({1.0, 2.0, 3.0});
    
    ErrorState delta;
    delta[3] = 0.1;
    delta[4] = 0.2;
    delta[5] = 0.3;
    
    state.injectErrorState(delta);
    
    TEST_ASSERT_NEAR(1.1, state.imuState.velocity[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.2, state.imuState.velocity[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.3, state.imuState.velocity[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_msckf_state_inject_error_state_orientation() {
    TEST_BEGIN("MSCKF state inject error state orientation");
    
    MSCKFState state;
    state.imuState.orientation = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), 0.1);
    
    ErrorState delta;
    delta[6] = 0.0;
    delta[7] = 0.0;
    delta[8] = 0.05;
    
    state.injectErrorState(delta);
    
    Vector3d euler = state.imuState.orientation.toEulerAngles();
    TEST_ASSERT_NEAR(0.15, euler[2], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_msckf_state_inject_error_state_bias() {
    TEST_BEGIN("MSCKF state inject error state bias");
    
    MSCKFState state;
    state.imuState.gyroBias = Vector3d({0.01, 0.02, 0.03});
    state.imuState.accelBias = Vector3d({0.1, 0.2, 0.3});
    
    ErrorState delta;
    delta[9] = 0.001;
    delta[10] = 0.002;
    delta[11] = 0.003;
    delta[12] = 0.01;
    delta[13] = 0.02;
    delta[14] = 0.03;
    
    state.injectErrorState(delta);
    
    TEST_ASSERT_NEAR(0.011, state.imuState.gyroBias[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.022, state.imuState.gyroBias[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.033, state.imuState.gyroBias[2], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.11, state.imuState.accelBias[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.22, state.imuState.accelBias[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.33, state.imuState.accelBias[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_msckf_state_covariance_symmetry() {
    TEST_BEGIN("MSCKF state covariance symmetry");
    
    MSCKFState state;
    
    for (uint32 i = 0; i < state.covarianceDim; ++i) {
        for (uint32 j = 0; j < state.covarianceDim; ++j) {
            state.covariance(i, j) = static_cast<float64>(i * state.covarianceDim + j);
        }
    }
    
    state.covariance.symmetrize();
    
    for (uint32 i = 0; i < state.covarianceDim; ++i) {
        for (uint32 j = 0; j < state.covarianceDim; ++j) {
            TEST_ASSERT_NEAR(state.covariance(i, j), state.covariance(j, i), TEST_TOLERANCE);
        }
    }
    
    TEST_END();
}

bool test_msckf_state_max_camera_states_limit() {
    TEST_BEGIN("MSCKF state max camera states limit");
    
    MSCKFState state;
    
    for (uint32 i = 0; i < MAX_CAMERA_FRAMES + 5; ++i) {
        CameraState camState;
        state.addCameraState(camState);
    }
    
    TEST_ASSERT_TRUE(state.numCameraStates <= MAX_CAMERA_FRAMES);
    
    TEST_END();
}

bool test_msckf_state_reset() {
    TEST_BEGIN("MSCKF state reset");
    
    MSCKFState state;
    
    state.imuState.position = Vector3d({1.0, 2.0, 3.0});
    state.imuState.velocity = Vector3d({4.0, 5.0, 6.0});
    state.timestamp = 1000000;
    
    for (uint32 i = 0; i < 3; ++i) {
        CameraState camState;
        state.addCameraState(camState);
    }
    
    state.reset();
    
    TEST_ASSERT_NEAR(0.0, state.imuState.position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, state.imuState.velocity[0], TEST_TOLERANCE);
    TEST_ASSERT_EQ(0u, state.numCameraStates);
    TEST_ASSERT_EQ(0u, state.timestamp);
    
    TEST_END();
}

REGISTER_TEST("IMU state default constructor", test_imu_state_default_constructor, "State");
REGISTER_TEST("IMU state set position", test_imu_state_set_position, "State");
REGISTER_TEST("IMU state set velocity", test_imu_state_set_velocity, "State");
REGISTER_TEST("IMU state set orientation", test_imu_state_set_orientation, "State");
REGISTER_TEST("IMU state set bias", test_imu_state_set_bias, "State");
REGISTER_TEST("Camera state default constructor", test_camera_state_default_constructor, "State");
REGISTER_TEST("Extrinsic state default constructor", test_extrinsic_state_default_constructor, "State");
REGISTER_TEST("Extrinsic state get rotation matrix", test_extrinsic_state_get_rotation_matrix, "State");
REGISTER_TEST("MSCKF state default constructor", test_msckf_state_default_constructor, "State");
REGISTER_TEST("MSCKF state add camera state", test_msckf_state_add_camera_state, "State");
REGISTER_TEST("MSCKF state add multiple camera states", test_msckf_state_add_multiple_camera_states, "State");
REGISTER_TEST("MSCKF state remove camera state", test_msckf_state_remove_camera_state, "State");
REGISTER_TEST("MSCKF state remove middle camera state", test_msckf_state_remove_middle_camera_state, "State");
REGISTER_TEST("MSCKF state state dim", test_msckf_state_state_dim, "State");
REGISTER_TEST("MSCKF state camera state index", test_msckf_state_camera_state_index, "State");
REGISTER_TEST("MSCKF state covariance dimension", test_msckf_state_covariance_dimension, "State");
REGISTER_TEST("MSCKF state inject error state", test_msckf_state_inject_error_state, "State");
REGISTER_TEST("MSCKF state inject error state velocity", test_msckf_state_inject_error_state_velocity, "State");
REGISTER_TEST("MSCKF state inject error state orientation", test_msckf_state_inject_error_state_orientation, "State");
REGISTER_TEST("MSCKF state inject error state bias", test_msckf_state_inject_error_state_bias, "State");
REGISTER_TEST("MSCKF state covariance symmetry", test_msckf_state_covariance_symmetry, "State");
REGISTER_TEST("MSCKF state max camera states limit", test_msckf_state_max_camera_states_limit, "State");
REGISTER_TEST("MSCKF state reset", test_msckf_state_reset, "State");

}
}

#endif
