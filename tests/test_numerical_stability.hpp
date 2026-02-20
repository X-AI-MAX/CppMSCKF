#ifndef MSCKF_TESTS_TEST_NUMERICAL_STABILITY_HPP
#define MSCKF_TESTS_TEST_NUMERICAL_STABILITY_HPP

#include "test_framework.hpp"
#include "../include/math/numerical_stability.hpp"
#include "../include/math/matrix.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_safe_sqrt_positive() {
    TEST_BEGIN("Safe sqrt positive");
    
    float64 result = safeSqrt(4.0);
    
    TEST_ASSERT_NEAR(2.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_safe_sqrt_zero() {
    TEST_BEGIN("Safe sqrt zero");
    
    float64 result = safeSqrt(0.0);
    
    TEST_ASSERT_NEAR(0.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_safe_sqrt_negative() {
    TEST_BEGIN("Safe sqrt negative");
    
    float64 result = safeSqrt(-1.0);
    
    TEST_ASSERT_NEAR(0.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_safe_sqrt_small_positive() {
    TEST_BEGIN("Safe sqrt small positive");
    
    float64 result = safeSqrt(1e-20);
    
    TEST_ASSERT_TRUE(result >= 0);
    TEST_ASSERT_TRUE(result < 1e-8);
    
    TEST_END();
}

bool test_safe_division_normal() {
    TEST_BEGIN("Safe division normal");
    
    float64 result = safeDivide(10.0, 2.0, 0.0);
    
    TEST_ASSERT_NEAR(5.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_safe_division_zero() {
    TEST_BEGIN("Safe division zero");
    
    float64 result = safeDivide(10.0, 0.0, -1.0);
    
    TEST_ASSERT_NEAR(-1.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_safe_division_small_denominator() {
    TEST_BEGIN("Safe division small denominator");
    
    float64 result = safeDivide(10.0, 1e-15, 0.0);
    
    TEST_ASSERT_NEAR(0.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_safe_inverse_normal() {
    TEST_BEGIN("Safe inverse normal");
    
    float64 result = safeInverse(2.0);
    
    TEST_ASSERT_NEAR(0.5, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_safe_inverse_zero() {
    TEST_BEGIN("Safe inverse zero");
    
    float64 result = safeInverse(0.0);
    
    TEST_ASSERT_NEAR(0.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_safe_inverse_small() {
    TEST_BEGIN("Safe inverse small");
    
    float64 result = safeInverse(1e-15);
    
    TEST_ASSERT_NEAR(0.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_clamp_within_range() {
    TEST_BEGIN("Clamp within range");
    
    float64 result = clamp(5.0, 0.0, 10.0);
    
    TEST_ASSERT_NEAR(5.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_clamp_below_min() {
    TEST_BEGIN("Clamp below min");
    
    float64 result = clamp(-5.0, 0.0, 10.0);
    
    TEST_ASSERT_NEAR(0.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_clamp_above_max() {
    TEST_BEGIN("Clamp above max");
    
    float64 result = clamp(15.0, 0.0, 10.0);
    
    TEST_ASSERT_NEAR(10.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_clamp_min_equals_max() {
    TEST_BEGIN("Clamp min equals max");
    
    float64 result = clamp(5.0, 5.0, 5.0);
    
    TEST_ASSERT_NEAR(5.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_near_zero_true() {
    TEST_BEGIN("Near zero true");
    
    bool result = nearZero(1e-10);
    
    TEST_ASSERT_TRUE(result);
    
    TEST_END();
}

bool test_near_zero_false() {
    TEST_BEGIN("Near zero false");
    
    bool result = nearZero(1e-5);
    
    TEST_ASSERT_FALSE(result);
    
    TEST_END();
}

bool test_near_zero_negative() {
    TEST_BEGIN("Near zero negative");
    
    bool result = nearZero(-1e-10);
    
    TEST_ASSERT_TRUE(result);
    
    TEST_END();
}

bool test_near_zero_exact_zero() {
    TEST_BEGIN("Near zero exact zero");
    
    bool result = nearZero(0.0);
    
    TEST_ASSERT_TRUE(result);
    
    TEST_END();
}

bool test_is_finite_normal() {
    TEST_BEGIN("Is finite normal");
    
    bool result = isFinite(1.0);
    
    TEST_ASSERT_TRUE(result);
    
    TEST_END();
}

bool test_is_finite_nan() {
    TEST_BEGIN("Is finite nan");
    
    float64 nan_val = std::nan("");
    bool result = isFinite(nan_val);
    
    TEST_ASSERT_FALSE(result);
    
    TEST_END();
}

bool test_is_finite_inf() {
    TEST_BEGIN("Is finite inf");
    
    float64 inf_val = 1.0 / 0.0;
    bool result = isFinite(inf_val);
    
    TEST_ASSERT_FALSE(result);
    
    TEST_END();
}

bool test_is_finite_negative_inf() {
    TEST_BEGIN("Is finite negative inf");
    
    float64 inf_val = -1.0 / 0.0;
    bool result = isFinite(inf_val);
    
    TEST_ASSERT_FALSE(result);
    
    TEST_END();
}

bool test_ensure_positive_definite_already_pd() {
    TEST_BEGIN("Ensure positive definite already PD");
    
    Matrix3d P = Matrix3d::identity();
    P(0, 0) = 1.0;
    P(1, 1) = 2.0;
    P(2, 2) = 3.0;
    
    ensurePositiveDefinite(P, 3);
    
    TEST_ASSERT_NEAR(1.0, P(0, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, P(1, 1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, P(2, 2), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_ensure_positive_definite_negative_diagonal() {
    TEST_BEGIN("Ensure positive definite negative diagonal");
    
    Matrix3d P = Matrix3d::identity();
    P(0, 0) = -1.0;
    P(1, 1) = 2.0;
    P(2, 2) = 3.0;
    
    ensurePositiveDefinite(P, 3);
    
    TEST_ASSERT_TRUE(P(0, 0) > 0);
    TEST_ASSERT_TRUE(P(1, 1) > 0);
    TEST_ASSERT_TRUE(P(2, 2) > 0);
    
    TEST_END();
}

bool test_ensure_positive_definite_zero_diagonal() {
    TEST_BEGIN("Ensure positive definite zero diagonal");
    
    Matrix3d P = Matrix3d::identity();
    P(0, 0) = 0.0;
    P(1, 1) = 0.0;
    P(2, 2) = 0.0;
    
    ensurePositiveDefinite(P, 3);
    
    TEST_ASSERT_TRUE(P(0, 0) > 0);
    TEST_ASSERT_TRUE(P(1, 1) > 0);
    TEST_ASSERT_TRUE(P(2, 2) > 0);
    
    TEST_END();
}

bool test_symmetrize_matrix() {
    TEST_BEGIN("Symmetrize matrix");
    
    Matrix3d M;
    M(0, 0) = 1; M(0, 1) = 2; M(0, 2) = 3;
    M(1, 0) = 4; M(1, 1) = 5; M(1, 2) = 6;
    M(2, 0) = 7; M(2, 1) = 8; M(2, 2) = 9;
    
    M.symmetrize();
    
    TEST_ASSERT_NEAR(M(0, 1), M(1, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(M(0, 2), M(2, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(M(1, 2), M(2, 1), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_kahan_accumulator_basic() {
    TEST_BEGIN("Kahan accumulator basic");
    
    KahanVectorAccumulator<3> acc;
    
    float64 values[3] = {1.0, 2.0, 3.0};
    acc.add(values);
    
    TEST_ASSERT_NEAR(1.0, acc.getSum(0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, acc.getSum(1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, acc.getSum(2), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_kahan_accumulator_multiple() {
    TEST_BEGIN("Kahan accumulator multiple");
    
    KahanVectorAccumulator<3> acc;
    
    float64 values1[3] = {1.0, 2.0, 3.0};
    float64 values2[3] = {4.0, 5.0, 6.0};
    acc.add(values1);
    acc.add(values2);
    
    TEST_ASSERT_NEAR(5.0, acc.getSum(0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(7.0, acc.getSum(1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(9.0, acc.getSum(2), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_kahan_accumulator_reset() {
    TEST_BEGIN("Kahan accumulator reset");
    
    KahanVectorAccumulator<3> acc;
    
    float64 values[3] = {1.0, 2.0, 3.0};
    acc.add(values);
    acc.reset();
    
    TEST_ASSERT_NEAR(0.0, acc.getSum(0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, acc.getSum(1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, acc.getSum(2), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_kahan_accumulator_precision() {
    TEST_BEGIN("Kahan accumulator precision");
    
    KahanVectorAccumulator<1> acc;
    
    float64 small = 1e-10;
    float64 large = 1e10;
    
    for (uint32 i = 0; i < 1000; ++i) {
        acc.add(0, small);
    }
    
    float64 expected = small * 1000;
    float64 result = acc.getSum(0);
    
    float64 relative_error = abs(result - expected) / expected;
    TEST_ASSERT_TRUE(relative_error < 1e-6);
    
    TEST_END();
}

bool test_kahan_accumulator_single_element() {
    TEST_BEGIN("Kahan accumulator single element");
    
    KahanVectorAccumulator<1> acc;
    
    acc.add(0, 5.0);
    acc.add(0, 3.0);
    acc.add(0, 2.0);
    
    TEST_ASSERT_NEAR(10.0, acc.getSum(0), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_kahan_accumulator_get_sum_array() {
    TEST_BEGIN("Kahan accumulator get sum array");
    
    KahanVectorAccumulator<3> acc;
    
    float64 values[3] = {1.0, 2.0, 3.0};
    acc.add(values);
    
    float64 result[3];
    acc.getSum(result);
    
    TEST_ASSERT_NEAR(1.0, result[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2.0, result[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3.0, result[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_kahan_accumulator_dimension() {
    TEST_BEGIN("Kahan accumulator dimension");
    
    KahanVectorAccumulator<10> acc;
    
    TEST_ASSERT_EQ(10u, acc.dimension());
    
    TEST_END();
}

bool test_kahan_accumulator_out_of_bounds() {
    TEST_BEGIN("Kahan accumulator out of bounds");
    
    KahanVectorAccumulator<3> acc;
    
    float64 result = acc.getSum(100);
    
    TEST_ASSERT_NEAR(0.0, result, TEST_TOLERANCE);
    
    TEST_END();
}

REGISTER_TEST("Safe sqrt positive", test_safe_sqrt_positive, "NumericalStability");
REGISTER_TEST("Safe sqrt zero", test_safe_sqrt_zero, "NumericalStability");
REGISTER_TEST("Safe sqrt negative", test_safe_sqrt_negative, "NumericalStability");
REGISTER_TEST("Safe sqrt small positive", test_safe_sqrt_small_positive, "NumericalStability");
REGISTER_TEST("Safe division normal", test_safe_division_normal, "NumericalStability");
REGISTER_TEST("Safe division zero", test_safe_division_zero, "NumericalStability");
REGISTER_TEST("Safe division small denominator", test_safe_division_small_denominator, "NumericalStability");
REGISTER_TEST("Safe inverse normal", test_safe_inverse_normal, "NumericalStability");
REGISTER_TEST("Safe inverse zero", test_safe_inverse_zero, "NumericalStability");
REGISTER_TEST("Safe inverse small", test_safe_inverse_small, "NumericalStability");
REGISTER_TEST("Clamp within range", test_clamp_within_range, "NumericalStability");
REGISTER_TEST("Clamp below min", test_clamp_below_min, "NumericalStability");
REGISTER_TEST("Clamp above max", test_clamp_above_max, "NumericalStability");
REGISTER_TEST("Clamp min equals max", test_clamp_min_equals_max, "NumericalStability");
REGISTER_TEST("Near zero true", test_near_zero_true, "NumericalStability");
REGISTER_TEST("Near zero false", test_near_zero_false, "NumericalStability");
REGISTER_TEST("Near zero negative", test_near_zero_negative, "NumericalStability");
REGISTER_TEST("Near zero exact zero", test_near_zero_exact_zero, "NumericalStability");
REGISTER_TEST("Is finite normal", test_is_finite_normal, "NumericalStability");
REGISTER_TEST("Is finite nan", test_is_finite_nan, "NumericalStability");
REGISTER_TEST("Is finite inf", test_is_finite_inf, "NumericalStability");
REGISTER_TEST("Is finite negative inf", test_is_finite_negative_inf, "NumericalStability");
REGISTER_TEST("Ensure positive definite already PD", test_ensure_positive_definite_already_pd, "NumericalStability");
REGISTER_TEST("Ensure positive definite negative diagonal", test_ensure_positive_definite_negative_diagonal, "NumericalStability");
REGISTER_TEST("Ensure positive definite zero diagonal", test_ensure_positive_definite_zero_diagonal, "NumericalStability");
REGISTER_TEST("Symmetrize matrix", test_symmetrize_matrix, "NumericalStability");
REGISTER_TEST("Kahan accumulator basic", test_kahan_accumulator_basic, "NumericalStability");
REGISTER_TEST("Kahan accumulator multiple", test_kahan_accumulator_multiple, "NumericalStability");
REGISTER_TEST("Kahan accumulator reset", test_kahan_accumulator_reset, "NumericalStability");
REGISTER_TEST("Kahan accumulator precision", test_kahan_accumulator_precision, "NumericalStability");
REGISTER_TEST("Kahan accumulator single element", test_kahan_accumulator_single_element, "NumericalStability");
REGISTER_TEST("Kahan accumulator get sum array", test_kahan_accumulator_get_sum_array, "NumericalStability");
REGISTER_TEST("Kahan accumulator dimension", test_kahan_accumulator_dimension, "NumericalStability");
REGISTER_TEST("Kahan accumulator out of bounds", test_kahan_accumulator_out_of_bounds, "NumericalStability");

}
}

#endif
