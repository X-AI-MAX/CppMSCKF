#ifndef MSCKF_TESTS_TEST_EXTREME_SCENARIO_HPP
#define MSCKF_TESTS_TEST_EXTREME_SCENARIO_HPP

#include "test_framework.hpp"
#include "../include/core/extreme_scenario_handler.hpp"
#include "../include/core/state.hpp"
#include "../include/math/quaternion.hpp"
#include "../include/math/matrix.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_pure_rotation_handler_default_constructor() {
    TEST_BEGIN("Pure rotation handler default constructor");
    
    PureRotationHandler handler;
    
    TEST_ASSERT_FALSE(handler.isInPureRotationMode());
    
    TEST_END();
}

bool test_pure_rotation_handler_init() {
    TEST_BEGIN("Pure rotation handler init");
    
    PureRotationHandler handler;
    PureRotationHandler::Config config;
    config.translationThreshold = 0.05;
    config.minObservationTime = 0.5;
    
    handler.init(config);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_pure_rotation_handler_enter_mode() {
    TEST_BEGIN("Pure rotation handler enter mode");
    
    PureRotationHandler handler;
    handler.init(PureRotationHandler::Config());
    
    handler.enterPureRotationMode(1000000);
    
    TEST_ASSERT_TRUE(handler.isInPureRotationMode());
    
    TEST_END();
}

bool test_pure_rotation_handler_exit_mode() {
    TEST_BEGIN("Pure rotation handler exit mode");
    
    PureRotationHandler handler;
    handler.init(PureRotationHandler::Config());
    
    handler.enterPureRotationMode(1000000);
    handler.exitPureRotationMode();
    
    TEST_ASSERT_FALSE(handler.isInPureRotationMode());
    
    TEST_END();
}

bool test_pure_rotation_handler_check_small_translation() {
    TEST_BEGIN("Pure rotation handler check small translation");
    
    PureRotationHandler handler;
    PureRotationHandler::Config config;
    config.translationThreshold = 0.1;
    config.minObservationsForSwitch = 5;
    handler.init(config);
    
    MSCKFState state;
    Vector3d deltaTranslation({0.01, 0.01, 0.01});
    
    for (uint32 i = 0; i < 10; ++i) {
        handler.checkAndUpdate(state, deltaTranslation);
    }
    
    TEST_ASSERT_TRUE(handler.isInPureRotationMode());
    
    TEST_END();
}

bool test_pure_rotation_handler_check_large_translation() {
    TEST_BEGIN("Pure rotation handler check large translation");
    
    PureRotationHandler handler;
    PureRotationHandler::Config config;
    config.translationThreshold = 0.1;
    handler.init(config);
    
    MSCKFState state;
    Vector3d deltaTranslation({1.0, 1.0, 1.0});
    
    handler.checkAndUpdate(state, deltaTranslation);
    
    TEST_ASSERT_FALSE(handler.isInPureRotationMode());
    
    TEST_END();
}

bool test_pure_rotation_handler_get_fixed_inverse_depth() {
    TEST_BEGIN("Pure rotation handler get fixed inverse depth");
    
    PureRotationHandler handler;
    PureRotationHandler::Config config;
    config.fixedInverseDepth = 0.5;
    handler.init(config);
    
    float64 invDepth = handler.getFixedInverseDepth();
    
    TEST_ASSERT_NEAR(0.5, invDepth, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_dynamic_object_filter_default_constructor() {
    TEST_BEGIN("Dynamic object filter default constructor");
    
    DynamicObjectFilter filter;
    
    TEST_ASSERT_NEAR(1.0, filter.getDynamicInlierRatio(), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_dynamic_object_filter_init() {
    TEST_BEGIN("Dynamic object filter init");
    
    DynamicObjectFilter filter;
    DynamicObjectFilter::Config config;
    config.ransacInlierRatioThreshold = 0.7;
    config.multiFrameConsistencyThreshold = 2.0;
    
    filter.init(config);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_dynamic_object_filter_reset() {
    TEST_BEGIN("Dynamic object filter reset");
    
    DynamicObjectFilter filter;
    filter.init(DynamicObjectFilter::Config());
    
    filter.reset();
    
    TEST_ASSERT_NEAR(1.0, filter.getDynamicInlierRatio(), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_dynamic_object_filter_get_adaptive_ransac_threshold() {
    TEST_BEGIN("Dynamic object filter get adaptive RANSAC threshold");
    
    DynamicObjectFilter filter;
    DynamicObjectFilter::Config config;
    config.epipolarThreshold = 2.0;
    filter.init(config);
    
    float64 threshold = filter.getAdaptiveRansacThreshold();
    
    TEST_ASSERT_TRUE(threshold > 0);
    
    TEST_END();
}

bool test_shock_recovery_handler_default_constructor() {
    TEST_BEGIN("Shock recovery handler default constructor");
    
    ShockRecoveryHandler handler;
    
    TEST_ASSERT_FALSE(handler.isInShock());
    TEST_ASSERT_FALSE(handler.isInRecovery());
    
    TEST_END();
}

bool test_shock_recovery_handler_init() {
    TEST_BEGIN("Shock recovery handler init");
    
    ShockRecoveryHandler handler;
    ShockRecoveryHandler::Config config;
    config.shockAccelThreshold = 100.0;
    config.shockGyroThreshold = 30.0;
    
    handler.init(config);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_shock_recovery_handler_detect_shock_accel() {
    TEST_BEGIN("Shock recovery handler detect shock accel");
    
    ShockRecoveryHandler handler;
    ShockRecoveryHandler::Config config;
    config.shockAccelThreshold = 50.0;
    config.shockCountThreshold = 3;
    handler.init(config);
    
    Vector3d largeAccel({100.0, 0, 0});
    Vector3d gyro({0, 0, 0});
    
    for (uint32 i = 0; i < 5; ++i) {
        handler.detectShock(largeAccel, gyro, i * 10000);
    }
    
    TEST_ASSERT_TRUE(handler.isInShock());
    
    TEST_END();
}

bool test_shock_recovery_handler_detect_shock_gyro() {
    TEST_BEGIN("Shock recovery handler detect shock gyro");
    
    ShockRecoveryHandler handler;
    ShockRecoveryHandler::Config config;
    config.shockGyroThreshold = 20.0;
    config.shockCountThreshold = 3;
    handler.init(config);
    
    Vector3d accel({0, 0, 9.81});
    Vector3d largeGyro({0, 0, 50.0});
    
    for (uint32 i = 0; i < 5; ++i) {
        handler.detectShock(accel, largeGyro, i * 10000);
    }
    
    TEST_ASSERT_TRUE(handler.isInShock());
    
    TEST_END();
}

bool test_shock_recovery_handler_no_shock() {
    TEST_BEGIN("Shock recovery handler no shock");
    
    ShockRecoveryHandler handler;
    ShockRecoveryHandler::Config config;
    config.shockAccelThreshold = 100.0;
    config.shockGyroThreshold = 30.0;
    handler.init(config);
    
    Vector3d normalAccel({0, 0, 9.81});
    Vector3d normalGyro({0.01, 0.01, 0.01});
    
    handler.detectShock(normalAccel, normalGyro, 10000);
    
    TEST_ASSERT_FALSE(handler.isInShock());
    
    TEST_END();
}

bool test_shock_recovery_handler_reset() {
    TEST_BEGIN("Shock recovery handler reset");
    
    ShockRecoveryHandler handler;
    handler.init(ShockRecoveryHandler::Config());
    
    Vector3d largeAccel({100.0, 0, 0});
    Vector3d gyro({0, 0, 0});
    
    for (uint32 i = 0; i < 5; ++i) {
        handler.detectShock(largeAccel, gyro, i * 10000);
    }
    
    handler.reset();
    
    TEST_ASSERT_FALSE(handler.isInShock());
    TEST_ASSERT_FALSE(handler.isInRecovery());
    
    TEST_END();
}

bool test_shock_recovery_handler_total_events() {
    TEST_BEGIN("Shock recovery handler total events");
    
    ShockRecoveryHandler handler;
    ShockRecoveryHandler::Config config;
    config.shockAccelThreshold = 50.0;
    config.shockCountThreshold = 2;
    handler.init(config);
    
    Vector3d largeAccel({100.0, 0, 0});
    Vector3d gyro({0, 0, 0});
    
    for (uint32 i = 0; i < 3; ++i) {
        handler.detectShock(largeAccel, gyro, i * 10000);
    }
    
    TEST_ASSERT_TRUE(handler.getTotalShockEvents() > 0);
    
    TEST_END();
}

bool test_lighting_adaptation_handler_default_constructor() {
    TEST_BEGIN("Lighting adaptation handler default constructor");
    
    LightingAdaptationHandler handler;
    
    TEST_ASSERT_FALSE(handler.hasLightingChanged());
    
    TEST_END();
}

bool test_lighting_adaptation_handler_init() {
    TEST_BEGIN("Lighting adaptation handler init");
    
    LightingAdaptationHandler handler;
    LightingAdaptationHandler::Config config;
    config.exposureTarget = 128.0;
    config.adaptationRate = 0.1;
    
    handler.init(config);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_lighting_adaptation_handler_analyze_image() {
    TEST_BEGIN("Lighting adaptation handler analyze image");
    
    LightingAdaptationHandler handler;
    handler.init(LightingAdaptationHandler::Config());
    
    uint8 image[100 * 100];
    for (uint32 i = 0; i < 100 * 100; ++i) {
        image[i] = 128;
    }
    
    handler.analyzeImage(image, 100, 100);
    
    TEST_ASSERT_NEAR(128.0, handler.getAvgBrightness(), 5.0);
    
    TEST_END();
}

bool test_lighting_adaptation_handler_exposure_correction() {
    TEST_BEGIN("Lighting adaptation handler exposure correction");
    
    LightingAdaptationHandler handler;
    LightingAdaptationHandler::Config config;
    config.exposureTarget = 128.0;
    handler.init(config);
    
    uint8 darkImage[100 * 100];
    for (uint32 i = 0; i < 100 * 100; ++i) {
        darkImage[i] = 50;
    }
    
    handler.analyzeImage(darkImage, 100, 100);
    float64 correction = handler.computeExposureCorrection();
    
    TEST_ASSERT_TRUE(correction > 1.0);
    
    TEST_END();
}

bool test_extreme_scenario_manager_default_constructor() {
    TEST_BEGIN("Extreme scenario manager default constructor");
    
    ExtremeScenarioManager manager;
    
    TEST_ASSERT_TRUE(manager.getOverallStatus() == SafetyStatus::OK);
    
    TEST_END();
}

bool test_extreme_scenario_manager_init() {
    TEST_BEGIN("Extreme scenario manager init");
    
    ExtremeScenarioManager manager;
    ExtremeScenarioManager::Config config;
    config.enablePureRotationHandling = true;
    config.enableDynamicObjectFiltering = true;
    config.enableShockRecovery = true;
    
    manager.init(config);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_extreme_scenario_manager_process_imu() {
    TEST_BEGIN("Extreme scenario manager process IMU");
    
    ExtremeScenarioManager manager;
    manager.init(ExtremeScenarioManager::Config());
    
    Vector3d accel({0, 0, 9.81});
    Vector3d gyro({0.01, 0.01, 0.01});
    Vector3d deltaTrans({0.01, 0.01, 0.01});
    
    manager.processImuData(accel, gyro, deltaTrans, 10000);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_extreme_scenario_manager_reset() {
    TEST_BEGIN("Extreme scenario manager reset");
    
    ExtremeScenarioManager manager;
    manager.init(ExtremeScenarioManager::Config());
    
    manager.reset();
    
    TEST_ASSERT_TRUE(manager.getOverallStatus() == SafetyStatus::OK);
    TEST_ASSERT_EQ(0u, manager.getActiveScenarioCount());
    
    TEST_END();
}

REGISTER_TEST("Pure rotation handler default constructor", test_pure_rotation_handler_default_constructor, "ExtremeScenario");
REGISTER_TEST("Pure rotation handler init", test_pure_rotation_handler_init, "ExtremeScenario");
REGISTER_TEST("Pure rotation handler enter mode", test_pure_rotation_handler_enter_mode, "ExtremeScenario");
REGISTER_TEST("Pure rotation handler exit mode", test_pure_rotation_handler_exit_mode, "ExtremeScenario");
REGISTER_TEST("Pure rotation handler check small translation", test_pure_rotation_handler_check_small_translation, "ExtremeScenario");
REGISTER_TEST("Pure rotation handler check large translation", test_pure_rotation_handler_check_large_translation, "ExtremeScenario");
REGISTER_TEST("Pure rotation handler get fixed inverse depth", test_pure_rotation_handler_get_fixed_inverse_depth, "ExtremeScenario");
REGISTER_TEST("Dynamic object filter default constructor", test_dynamic_object_filter_default_constructor, "ExtremeScenario");
REGISTER_TEST("Dynamic object filter init", test_dynamic_object_filter_init, "ExtremeScenario");
REGISTER_TEST("Dynamic object filter reset", test_dynamic_object_filter_reset, "ExtremeScenario");
REGISTER_TEST("Dynamic object filter get adaptive RANSAC threshold", test_dynamic_object_filter_get_adaptive_ransac_threshold, "ExtremeScenario");
REGISTER_TEST("Shock recovery handler default constructor", test_shock_recovery_handler_default_constructor, "ExtremeScenario");
REGISTER_TEST("Shock recovery handler init", test_shock_recovery_handler_init, "ExtremeScenario");
REGISTER_TEST("Shock recovery handler detect shock accel", test_shock_recovery_handler_detect_shock_accel, "ExtremeScenario");
REGISTER_TEST("Shock recovery handler detect shock gyro", test_shock_recovery_handler_detect_shock_gyro, "ExtremeScenario");
REGISTER_TEST("Shock recovery handler no shock", test_shock_recovery_handler_no_shock, "ExtremeScenario");
REGISTER_TEST("Shock recovery handler reset", test_shock_recovery_handler_reset, "ExtremeScenario");
REGISTER_TEST("Shock recovery handler total events", test_shock_recovery_handler_total_events, "ExtremeScenario");
REGISTER_TEST("Lighting adaptation handler default constructor", test_lighting_adaptation_handler_default_constructor, "ExtremeScenario");
REGISTER_TEST("Lighting adaptation handler init", test_lighting_adaptation_handler_init, "ExtremeScenario");
REGISTER_TEST("Lighting adaptation handler analyze image", test_lighting_adaptation_handler_analyze_image, "ExtremeScenario");
REGISTER_TEST("Lighting adaptation handler exposure correction", test_lighting_adaptation_handler_exposure_correction, "ExtremeScenario");
REGISTER_TEST("Extreme scenario manager default constructor", test_extreme_scenario_manager_default_constructor, "ExtremeScenario");
REGISTER_TEST("Extreme scenario manager init", test_extreme_scenario_manager_init, "ExtremeScenario");
REGISTER_TEST("Extreme scenario manager process IMU", test_extreme_scenario_manager_process_imu, "ExtremeScenario");
REGISTER_TEST("Extreme scenario manager reset", test_extreme_scenario_manager_reset, "ExtremeScenario");

}
}

#endif
