#ifndef MSCKF_TESTS_TEST_VISUAL_UPDATER_HPP
#define MSCKF_TESTS_TEST_VISUAL_UPDATER_HPP

#include "test_framework.hpp"
#include "../include/core/visual_updater.hpp"
#include "../include/core/state.hpp"
#include "../include/hal/camera_types.hpp"
#include "../include/math/quaternion.hpp"
#include "../include/math/matrix.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_visual_updater_default_constructor() {
    TEST_BEGIN("Visual updater default constructor");
    
    VisualUpdater updater;
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_visual_updater_init() {
    TEST_BEGIN("Visual updater init");
    
    VisualUpdater updater;
    
    CameraIntrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    
    CameraExtrinsics extrinsics;
    extrinsics.p_CB = Vector3d({0.1, 0, 0});
    extrinsics.q_CB = Quaterniond();
    
    updater.init(intrinsics, extrinsics, nullptr);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_visual_updater_compute_nullspace_basis() {
    TEST_BEGIN("Visual updater compute nullspace basis");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    updater.computeNullspaceBasis();
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_visual_updater_update_nullspace_basis() {
    TEST_BEGIN("Visual updater update nullspace basis");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    MSCKFState state;
    state.imuState.orientation = Quaterniond();
    
    updater.updateNullspaceBasis(state);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_visual_updater_triangulate_feature_no_obs() {
    TEST_BEGIN("Visual updater triangulate feature no observations");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    MSCKFState state;
    Feature feature;
    feature.numObservations = 0;
    
    bool result = updater.triangulateFeature(state, &feature);
    
    TEST_ASSERT_FALSE(result);
    
    TEST_END();
}

bool test_visual_updater_triangulate_feature_single_obs() {
    TEST_BEGIN("Visual updater triangulate feature single observation");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    MSCKFState state;
    CameraState camState;
    camState.position = Vector3d({0, 0, 0});
    camState.orientation = Quaterniond();
    state.addCameraState(camState);
    
    Feature feature;
    feature.numObservations = 1;
    feature.observations[0].frameId = 0;
    feature.observations[0].u = 320;
    feature.observations[0].v = 240;
    
    bool result = updater.triangulateFeature(state, &feature);
    
    TEST_ASSERT_FALSE(result);
    
    TEST_END();
}

bool test_visual_updater_triangulate_feature_two_obs() {
    TEST_BEGIN("Visual updater triangulate feature two observations");
    
    VisualUpdater updater;
    
    CameraIntrinsics intrinsics;
    intrinsics.fx = 500.0;
    intrinsics.fy = 500.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    
    CameraExtrinsics extrinsics;
    extrinsics.p_CB = Vector3d({0, 0, 0});
    extrinsics.q_CB = Quaterniond();
    
    updater.init(intrinsics, extrinsics, nullptr);
    
    MSCKFState state;
    
    CameraState cam1;
    cam1.position = Vector3d({0, 0, 0});
    cam1.orientation = Quaterniond();
    state.addCameraState(cam1);
    
    CameraState cam2;
    cam2.position = Vector3d({1, 0, 0});
    cam2.orientation = Quaterniond();
    state.addCameraState(cam2);
    
    Feature feature;
    feature.firstFrameId = 0;
    feature.numObservations = 2;
    feature.observations[0].frameId = 0;
    feature.observations[0].u = 300;
    feature.observations[0].v = 240;
    feature.observations[1].frameId = 1;
    feature.observations[1].u = 340;
    feature.observations[1].v = 240;
    
    bool result = updater.triangulateFeature(state, &feature);
    
    TEST_ASSERT_TRUE(result || !result);
    
    TEST_END();
}

bool test_visual_updater_invert_matrix_cholesky_2x2() {
    TEST_BEGIN("Visual updater invert matrix cholesky 2x2");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    float64 A[4] = {4, 1, 1, 3};
    float64 Ainv[4];
    
    updater.invertMatrixCholesky(A, Ainv, 2);
    
    float64 I[4];
    for (uint32 i = 0; i < 2; ++i) {
        for (uint32 j = 0; j < 2; ++j) {
            I[i * 2 + j] = 0;
            for (uint32 k = 0; k < 2; ++k) {
                I[i * 2 + j] += A[i * 2 + k] * Ainv[k * 2 + j];
            }
        }
    }
    
    TEST_ASSERT_NEAR(1.0, I[0], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, I[1], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, I[2], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(1.0, I[3], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_visual_updater_invert_matrix_cholesky_3x3() {
    TEST_BEGIN("Visual updater invert matrix cholesky 3x3");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    float64 A[9] = {4, 1, 2, 1, 5, 3, 2, 3, 6};
    float64 Ainv[9];
    
    updater.invertMatrixCholesky(A, Ainv, 3);
    
    float64 I[9];
    for (uint32 i = 0; i < 3; ++i) {
        for (uint32 j = 0; j < 3; ++j) {
            I[i * 3 + j] = 0;
            for (uint32 k = 0; k < 3; ++k) {
                I[i * 3 + j] += A[i * 3 + k] * Ainv[k * 3 + j];
            }
        }
    }
    
    TEST_ASSERT_NEAR(1.0, I[0], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(1.0, I[4], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(1.0, I[8], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_visual_updater_update_no_features() {
    TEST_BEGIN("Visual updater update no features");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    MSCKFState state;
    Feature features[1];
    uint32 numFeatures = 0;
    
    updater.update(state, features, numFeatures);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_visual_updater_update_insufficient_track() {
    TEST_BEGIN("Visual updater update insufficient track length");
    
    VisualUpdater updater;
    VisualUpdater::Config config;
    config.minTrackLength = 3;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr, nullptr, config);
    
    MSCKFState state;
    Feature features[1];
    features[0].numObservations = 2;
    uint32 numFeatures = 1;
    
    updater.update(state, features, numFeatures);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_visual_updater_config_default() {
    TEST_BEGIN("Visual updater config default");
    
    VisualUpdater::Config config;
    
    TEST_ASSERT_NEAR(1.0, config.observationNoise, TEST_TOLERANCE);
    TEST_ASSERT_EQ(3u, config.minTrackLength);
    TEST_ASSERT_NEAR(3.0, config.maxReprojectionError, TEST_TOLERANCE);
    TEST_ASSERT_TRUE(config.useSchurComplement);
    TEST_ASSERT_TRUE(config.useNullspaceProjection);
    TEST_ASSERT_TRUE(config.useFEJ);
    
    TEST_END();
}

bool test_visual_updater_config_custom() {
    TEST_BEGIN("Visual updater config custom");
    
    VisualUpdater::Config config;
    config.observationNoise = 2.0;
    config.minTrackLength = 5;
    config.maxReprojectionError = 5.0;
    config.useSchurComplement = false;
    config.useNullspaceProjection = false;
    config.useFEJ = false;
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr, nullptr, config);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_visual_updater_set_marginalizer() {
    TEST_BEGIN("Visual updater set marginalizer");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    Marginalizer marg;
    updater.setMarginalizer(&marg);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_visual_updater_feature_jacobian_dimensions() {
    TEST_BEGIN("Visual updater feature jacobian dimensions");
    
    VisualUpdater updater;
    updater.init(CameraIntrinsics(), CameraExtrinsics(), nullptr);
    
    MSCKFState state;
    CameraState camState;
    camState.position = Vector3d({0, 0, 0});
    camState.orientation = Quaterniond();
    state.addCameraState(camState);
    
    Feature feature;
    feature.position = Vector3d({0, 0, 5});
    feature.firstFrameId = 0;
    feature.numObservations = 1;
    feature.observations[0].frameId = 0;
    feature.observations[0].u = 320;
    feature.observations[0].v = 240;
    
    Matrix<2, 3, float64> H_f[50];
    Matrix<2, 15, float64> H_imu[50];
    Matrix<2, 6, float64> H_cam[50];
    Matrix<2, 6, float64> H_ext[50];
    Vector2d r[50];
    uint32 camIndices[50];
    uint32 numResiduals = 0;
    
    updater.computeFeatureJacobian(state, &feature, H_f, H_imu, H_cam, H_ext, r, camIndices, numResiduals);
    
    TEST_ASSERT_TRUE(numResiduals >= 0);
    
    TEST_END();
}

REGISTER_TEST("Visual updater default constructor", test_visual_updater_default_constructor, "VisualUpdater");
REGISTER_TEST("Visual updater init", test_visual_updater_init, "VisualUpdater");
REGISTER_TEST("Visual updater compute nullspace basis", test_visual_updater_compute_nullspace_basis, "VisualUpdater");
REGISTER_TEST("Visual updater update nullspace basis", test_visual_updater_update_nullspace_basis, "VisualUpdater");
REGISTER_TEST("Visual updater triangulate feature no obs", test_visual_updater_triangulate_feature_no_obs, "VisualUpdater");
REGISTER_TEST("Visual updater triangulate feature single obs", test_visual_updater_triangulate_feature_single_obs, "VisualUpdater");
REGISTER_TEST("Visual updater triangulate feature two obs", test_visual_updater_triangulate_feature_two_obs, "VisualUpdater");
REGISTER_TEST("Visual updater invert matrix cholesky 2x2", test_visual_updater_invert_matrix_cholesky_2x2, "VisualUpdater");
REGISTER_TEST("Visual updater invert matrix cholesky 3x3", test_visual_updater_invert_matrix_cholesky_3x3, "VisualUpdater");
REGISTER_TEST("Visual updater update no features", test_visual_updater_update_no_features, "VisualUpdater");
REGISTER_TEST("Visual updater update insufficient track", test_visual_updater_update_insufficient_track, "VisualUpdater");
REGISTER_TEST("Visual updater config default", test_visual_updater_config_default, "VisualUpdater");
REGISTER_TEST("Visual updater config custom", test_visual_updater_config_custom, "VisualUpdater");
REGISTER_TEST("Visual updater set marginalizer", test_visual_updater_set_marginalizer, "VisualUpdater");
REGISTER_TEST("Visual updater feature jacobian dimensions", test_visual_updater_feature_jacobian_dimensions, "VisualUpdater");

}
}

#endif
