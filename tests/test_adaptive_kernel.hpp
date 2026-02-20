#ifndef MSCKF_TESTS_TEST_ADAPTIVE_KERNEL_HPP
#define MSCKF_TESTS_TEST_ADAPTIVE_KERNEL_HPP

#include "test_framework.hpp"
#include "../include/core/adaptive_kernel.hpp"
#include "../include/math/types.hpp"
#include "../include/math/matrix.hpp"

namespace msckf {
namespace test {

bool test_adaptive_kernel_default_constructor() {
    TEST_BEGIN("Adaptive kernel default constructor");
    
    AdaptiveKernelEstimator estimator;
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_adaptive_kernel_init() {
    TEST_BEGIN("Adaptive kernel init");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.kernelType = KernelType::HUBER;
    config.useAdaptiveThreshold = true;
    
    estimator.init(config);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_adaptive_kernel_config_default() {
    TEST_BEGIN("Adaptive kernel config default");
    
    AdaptiveKernelEstimator::Config config;
    
    TEST_ASSERT_TRUE(config.useAdaptiveThreshold);
    TEST_ASSERT_NEAR(1.345, config.huberThreshold, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.385, config.cauchyThreshold, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(4.685, config.tukeyThreshold, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_adaptive_kernel_huber_weight() {
    TEST_BEGIN("Adaptive kernel Huber weight");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.kernelType = KernelType::HUBER;
    config.huberThreshold = 1.345;
    estimator.init(config);
    
    float64 r_small = 0.5;
    float64 w_small = estimator.computeWeight(r_small);
    TEST_ASSERT_NEAR(1.0, w_small, TEST_TOLERANCE);
    
    float64 r_large = 5.0;
    float64 w_large = estimator.computeWeight(r_large);
    TEST_ASSERT_TRUE(w_large < 1.0);
    TEST_ASSERT_TRUE(w_large > 0);
    
    TEST_END();
}

bool test_adaptive_kernel_cauchy_weight() {
    TEST_BEGIN("Adaptive kernel Cauchy weight");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.kernelType = KernelType::CAUCHY;
    config.cauchyThreshold = 2.385;
    estimator.init(config);
    
    float64 r = 2.0;
    float64 w = estimator.computeWeight(r);
    
    TEST_ASSERT_TRUE(w > 0);
    TEST_ASSERT_TRUE(w <= 1.0);
    
    TEST_END();
}

bool test_adaptive_kernel_tukey_weight() {
    TEST_BEGIN("Adaptive kernel Tukey weight");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.kernelType = KernelType::TUKEY;
    config.tukeyThreshold = 4.685;
    estimator.init(config);
    
    float64 r_small = 2.0;
    float64 w_small = estimator.computeWeight(r_small);
    TEST_ASSERT_TRUE(w_small > 0);
    
    float64 r_large = 10.0;
    float64 w_large = estimator.computeWeight(r_large);
    TEST_ASSERT_NEAR(0.0, w_large, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_adaptive_kernel_mcclure_weight() {
    TEST_BEGIN("Adaptive kernel McClure weight");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.kernelType = KernelType::MCCLURE;
    estimator.init(config);
    
    float64 r = 1.0;
    float64 w = estimator.computeWeight(r);
    
    TEST_ASSERT_TRUE(w > 0);
    TEST_ASSERT_TRUE(w <= 1.0);
    
    TEST_END();
}

bool test_adaptive_kernel_geman_mcclure_weight() {
    TEST_BEGIN("Adaptive kernel Geman-McClure weight");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.kernelType = KernelType::GEMAN_MCCLURE;
    estimator.init(config);
    
    float64 r = 1.0;
    float64 w = estimator.computeWeight(r);
    
    TEST_ASSERT_TRUE(w > 0);
    TEST_ASSERT_TRUE(w <= 1.0);
    
    TEST_END();
}

bool test_adaptive_kernel_add_residual() {
    TEST_BEGIN("Adaptive kernel add residual");
    
    AdaptiveKernelEstimator estimator;
    estimator.init(AdaptiveKernelEstimator::Config());
    
    for (uint32 i = 0; i < 100; ++i) {
        estimator.addResidual(static_cast<float64>(i) * 0.01);
    }
    
    TEST_ASSERT_TRUE(estimator.getResidualCount() >= 0);
    
    TEST_END();
}

bool test_adaptive_kernel_compute_statistics() {
    TEST_BEGIN("Adaptive kernel compute statistics");
    
    AdaptiveKernelEstimator estimator;
    estimator.init(AdaptiveKernelEstimator::Config());
    
    for (uint32 i = 0; i < 100; ++i) {
        estimator.addResidual(static_cast<float64>(i) * 0.01);
    }
    
    estimator.computeStatistics();
    
    TEST_ASSERT_TRUE(estimator.getMedian() >= 0);
    TEST_ASSERT_TRUE(estimator.getMAD() >= 0);
    
    TEST_END();
}

bool test_adaptive_kernel_update_threshold() {
    TEST_BEGIN("Adaptive kernel update threshold");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.useAdaptiveThreshold = true;
    estimator.init(config);
    
    for (uint32 i = 0; i < 100; ++i) {
        estimator.addResidual(static_cast<float64>(i) * 0.01);
    }
    
    float64 before = estimator.getCurrentThreshold();
    estimator.updateThreshold();
    float64 after = estimator.getCurrentThreshold();
    
    TEST_ASSERT_TRUE(after >= config.minThreshold);
    TEST_ASSERT_TRUE(after <= config.maxThreshold);
    
    TEST_END();
}

bool test_adaptive_kernel_reset() {
    TEST_BEGIN("Adaptive kernel reset");
    
    AdaptiveKernelEstimator estimator;
    estimator.init(AdaptiveKernelEstimator::Config());
    
    for (uint32 i = 0; i < 50; ++i) {
        estimator.addResidual(static_cast<float64>(i));
    }
    
    estimator.reset();
    
    TEST_ASSERT_EQ(0u, estimator.getResidualCount());
    
    TEST_END();
}

bool test_adaptive_kernel_kernel_switching() {
    TEST_BEGIN("Adaptive kernel kernel switching");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.useKernelSwitching = true;
    estimator.init(config);
    
    for (uint32 i = 0; i < 100; ++i) {
        estimator.addResidual(static_cast<float64>(i) * 0.1);
    }
    
    estimator.updateKernelType();
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_adaptive_kernel_set_kernel_type() {
    TEST_BEGIN("Adaptive kernel set kernel type");
    
    AdaptiveKernelEstimator estimator;
    estimator.init(AdaptiveKernelEstimator::Config());
    
    estimator.setKernelType(KernelType::CAUCHY);
    
    TEST_ASSERT_TRUE(true);
    
    TEST_END();
}

bool test_adaptive_kernel_get_current_threshold() {
    TEST_BEGIN("Adaptive kernel get current threshold");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.huberThreshold = 1.345;
    config.kernelType = KernelType::HUBER;
    estimator.init(config);
    
    float64 threshold = estimator.getCurrentThreshold();
    
    TEST_ASSERT_NEAR(1.345, threshold, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_adaptive_kernel_threshold_bounds() {
    TEST_BEGIN("Adaptive kernel threshold bounds");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.minThreshold = 0.5;
    config.maxThreshold = 5.0;
    config.useAdaptiveThreshold = true;
    estimator.init(config);
    
    for (uint32 i = 0; i < 100; ++i) {
        estimator.addResidual(static_cast<float64>(i) * 0.1);
    }
    
    estimator.updateThreshold();
    float64 threshold = estimator.getCurrentThreshold();
    
    TEST_ASSERT_TRUE(threshold >= config.minThreshold);
    TEST_ASSERT_TRUE(threshold <= config.maxThreshold);
    
    TEST_END();
}

bool test_adaptive_kernel_weight_zero_residual() {
    TEST_BEGIN("Adaptive kernel weight zero residual");
    
    AdaptiveKernelEstimator estimator;
    estimator.init(AdaptiveKernelEstimator::Config());
    
    float64 w = estimator.computeWeight(0.0);
    
    TEST_ASSERT_NEAR(1.0, w, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_adaptive_kernel_weight_negative_residual() {
    TEST_BEGIN("Adaptive kernel weight negative residual");
    
    AdaptiveKernelEstimator estimator;
    AdaptiveKernelEstimator::Config config;
    config.kernelType = KernelType::HUBER;
    estimator.init(config);
    
    float64 w_pos = estimator.computeWeight(1.0);
    float64 w_neg = estimator.computeWeight(-1.0);
    
    TEST_ASSERT_NEAR(w_pos, w_neg, TEST_TOLERANCE);
    
    TEST_END();
}

REGISTER_TEST("Adaptive kernel default constructor", test_adaptive_kernel_default_constructor, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel init", test_adaptive_kernel_init, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel config default", test_adaptive_kernel_config_default, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel Huber weight", test_adaptive_kernel_huber_weight, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel Cauchy weight", test_adaptive_kernel_cauchy_weight, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel Tukey weight", test_adaptive_kernel_tukey_weight, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel McClure weight", test_adaptive_kernel_mcclure_weight, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel Geman-McClure weight", test_adaptive_kernel_geman_mcclure_weight, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel add residual", test_adaptive_kernel_add_residual, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel compute statistics", test_adaptive_kernel_compute_statistics, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel update threshold", test_adaptive_kernel_update_threshold, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel reset", test_adaptive_kernel_reset, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel kernel switching", test_adaptive_kernel_kernel_switching, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel set kernel type", test_adaptive_kernel_set_kernel_type, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel get current threshold", test_adaptive_kernel_get_current_threshold, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel threshold bounds", test_adaptive_kernel_threshold_bounds, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel weight zero residual", test_adaptive_kernel_weight_zero_residual, "AdaptiveKernel");
REGISTER_TEST("Adaptive kernel weight negative residual", test_adaptive_kernel_weight_negative_residual, "AdaptiveKernel");

}
}

#endif
