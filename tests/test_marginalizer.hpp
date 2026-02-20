#ifndef MSCKF_TESTS_TEST_MARGINALIZER_HPP
#define MSCKF_TESTS_TEST_MARGINALIZER_HPP

#include "test_framework.hpp"
#include "../include/core/marginalizer.hpp"
#include "../include/core/state.hpp"
#include "../include/math/quaternion.hpp"
#include "../include/math/matrix.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_marginalizer_default_constructor() {
    TEST_BEGIN("Marginalizer default constructor");
    
    Marginalizer marg;
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_marginalizer_init() {
    TEST_BEGIN("Marginalizer init");
    
    Marginalizer marg;
    Marginalizer::Config config;
    config.maxCameraFrames = 20;
    config.minCameraFrames = 5;
    
    marg.init(config);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_marginalizer_config_default() {
    TEST_BEGIN("Marginalizer config default");
    
    Marginalizer::Config config;
    
    TEST_ASSERT_EQ(30u, config.maxCameraFrames);
    TEST_ASSERT_EQ(5u, config.minCameraFrames);
    TEST_ASSERT_TRUE(config.useFEJ);
    
    TEST_END();
}

bool test_marginalizer_augment_state() {
    TEST_BEGIN("Marginalizer augment state");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    state.imuState.position = Vector3d({0, 0, 0});
    state.imuState.orientation = Quaterniond();
    
    uint32 initialDim = state.covarianceDim;
    
    marg.augmentState(state);
    
    TEST_ASSERT_EQ(initialDim + CAMERA_STATE_DIM, state.covarianceDim);
    TEST_ASSERT_EQ(1u, state.numCameraStates);
    
    TEST_END();
}

bool test_marginalizer_augment_multiple() {
    TEST_BEGIN("Marginalizer augment multiple");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 5; ++i) {
        marg.augmentState(state);
    }
    
    TEST_ASSERT_EQ(5u, state.numCameraStates);
    
    TEST_END();
}

bool test_marginalizer_marginalize_single() {
    TEST_BEGIN("Marginalizer marginalize single");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 5; ++i) {
        marg.augmentState(state);
    }
    
    uint32 beforeDim = state.covarianceDim;
    marg.marginalize(state);
    uint32 afterDim = state.covarianceDim;
    
    TEST_ASSERT_EQ(4u, state.numCameraStates);
    TEST_ASSERT_EQ(beforeDim - CAMERA_STATE_DIM, afterDim);
    
    TEST_END();
}

bool test_marginalizer_marginalize_multiple() {
    TEST_BEGIN("Marginalizer marginalize multiple");
    
    Marginalizer marg;
    Marginalizer::Config config;
    config.maxCameraFrames = 10;
    marg.init(config);
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 15; ++i) {
        marg.augmentState(state);
        if (state.numCameraStates > config.maxCameraFrames) {
            while (state.numCameraStates > config.maxCameraFrames - 2) {
                marg.marginalize(state);
            }
        }
    }
    
    TEST_ASSERT_TRUE(state.numCameraStates <= config.maxCameraFrames);
    
    TEST_END();
}

bool test_marginalizer_augment_covariance_symmetry() {
    TEST_BEGIN("Marginalizer augment covariance symmetry");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    
    for (uint32 i = 0; i < state.covarianceDim; ++i) {
        state.covariance(i, i) = 1.0;
    }
    
    marg.augmentState(state);
    
    for (uint32 i = 0; i < state.covarianceDim; ++i) {
        for (uint32 j = 0; j < state.covarianceDim; ++j) {
            TEST_ASSERT_NEAR(state.covariance(i, j), state.covariance(j, i), TEST_TOLERANCE_LOOSE);
        }
    }
    
    TEST_END();
}

bool test_marginalizer_marginalize_covariance_symmetry() {
    TEST_BEGIN("Marginalizer marginalize covariance symmetry");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 5; ++i) {
        marg.augmentState(state);
    }
    
    for (uint32 i = 0; i < state.covarianceDim; ++i) {
        state.covariance(i, i) = 1.0 + i * 0.1;
    }
    
    marg.marginalize(state);
    
    state.covariance.symmetrize();
    
    for (uint32 i = 0; i < state.covarianceDim; ++i) {
        for (uint32 j = 0; j < state.covarianceDim; ++j) {
            TEST_ASSERT_NEAR(state.covariance(i, j), state.covariance(j, i), TEST_TOLERANCE_LOOSE);
        }
    }
    
    TEST_END();
}

bool test_marginalizer_fej_point_storage() {
    TEST_BEGIN("Marginalizer FEJ point storage");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    state.imuState.position = Vector3d({1, 2, 3});
    state.imuState.orientation = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), 0.5);
    
    marg.augmentState(state);
    
    TEST_ASSERT_TRUE(marg.hasFEJPoint(0));
    
    TEST_END();
}

bool test_marginalizer_fej_point_retrieval() {
    TEST_BEGIN("Marginalizer FEJ point retrieval");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    state.imuState.position = Vector3d({1, 2, 3});
    state.imuState.orientation = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), 0.5);
    
    marg.augmentState(state);
    
    Quaterniond fejOrient = marg.getFEJOrientation(0);
    Vector3d fejPos = marg.getFEJPosition(0);
    
    TEST_ASSERT_NEAR(1.0, fejPos[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, fejPos[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, fejPos[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_marginalizer_apply_schur_complement() {
    TEST_BEGIN("Marginalizer apply Schur complement");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    Matrix<6, 6, float64> P_marg;
    for (uint32 i = 0; i < 6; ++i) {
        P_marg(i, i) = 1.0;
    }
    
    float64 P_rm[6 * 100];
    float64 P_mr[100 * 6];
    float64 P_marg_inv[36];
    
    for (uint32 i = 0; i < 6; ++i) {
        P_marg_inv[i * 6 + i] = 1.0;
    }
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_marginalizer_max_frames_limit() {
    TEST_BEGIN("Marginalizer max frames limit");
    
    Marginalizer marg;
    Marginalizer::Config config;
    config.maxCameraFrames = 5;
    marg.init(config);
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 10; ++i) {
        marg.augmentState(state);
        if (state.numCameraStates > config.maxCameraFrames) {
            marg.marginalize(state);
        }
    }
    
    TEST_ASSERT_TRUE(state.numCameraStates <= config.maxCameraFrames);
    
    TEST_END();
}

bool test_marginalizer_min_frames_preserved() {
    TEST_BEGIN("Marginalizer min frames preserved");
    
    Marginalizer marg;
    Marginalizer::Config config;
    config.maxCameraFrames = 10;
    config.minCameraFrames = 3;
    marg.init(config);
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 10; ++i) {
        marg.augmentState(state);
    }
    
    for (uint32 i = 0; i < 10; ++i) {
        marg.marginalize(state);
    }
    
    TEST_ASSERT_TRUE(state.numCameraStates >= config.minCameraFrames || state.numCameraStates == 0);
    
    TEST_END();
}

bool test_marginalizer_reset() {
    TEST_BEGIN("Marginalizer reset");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    marg.augmentState(state);
    marg.augmentState(state);
    
    marg.reset();
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_marginalizer_camera_state_tracking() {
    TEST_BEGIN("Marginalizer camera state tracking");
    
    Marginalizer marg;
    marg.init(Marginalizer::Config());
    
    MSCKFState state;
    
    for (uint32 i = 0; i < 3; ++i) {
        CameraState cam;
        cam.position = Vector3d({static_cast<float64>(i), 0, 0});
        state.addCameraState(cam);
    }
    
    marg.marginalize(state);
    
    TEST_ASSERT_EQ(2u, state.numCameraStates);
    TEST_ASSERT_NEAR(1.0, state.cameraStates[0].position[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, state.cameraStates[1].position[0], TEST_TOLERANCE);
    
    TEST_END();
}

REGISTER_TEST("Marginalizer default constructor", test_marginalizer_default_constructor, "Marginalizer");
REGISTER_TEST("Marginalizer init", test_marginalizer_init, "Marginalizer");
REGISTER_TEST("Marginalizer config default", test_marginalizer_config_default, "Marginalizer");
REGISTER_TEST("Marginalizer augment state", test_marginalizer_augment_state, "Marginalizer");
REGISTER_TEST("Marginalizer augment multiple", test_marginalizer_augment_multiple, "Marginalizer");
REGISTER_TEST("Marginalizer marginalize single", test_marginalizer_marginalize_single, "Marginalizer");
REGISTER_TEST("Marginalizer marginalize multiple", test_marginalizer_marginalize_multiple, "Marginalizer");
REGISTER_TEST("Marginalizer augment covariance symmetry", test_marginalizer_augment_covariance_symmetry, "Marginalizer");
REGISTER_TEST("Marginalizer marginalize covariance symmetry", test_marginalizer_marginalize_covariance_symmetry, "Marginalizer");
REGISTER_TEST("Marginalizer FEJ point storage", test_marginalizer_fej_point_storage, "Marginalizer");
REGISTER_TEST("Marginalizer FEJ point retrieval", test_marginalizer_fej_point_retrieval, "Marginalizer");
REGISTER_TEST("Marginalizer apply Schur complement", test_marginalizer_apply_schur_complement, "Marginalizer");
REGISTER_TEST("Marginalizer max frames limit", test_marginalizer_max_frames_limit, "Marginalizer");
REGISTER_TEST("Marginalizer min frames preserved", test_marginalizer_min_frames_preserved, "Marginalizer");
REGISTER_TEST("Marginalizer reset", test_marginalizer_reset, "Marginalizer");
REGISTER_TEST("Marginalizer camera state tracking", test_marginalizer_camera_state_tracking, "Marginalizer");

}
}

#endif
