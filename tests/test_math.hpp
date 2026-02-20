#ifndef MSCKF_TESTS_TEST_MATH_HPP
#define MSCKF_TESTS_TEST_MATH_HPP

#include "test_framework.hpp"
#include "../include/math/quaternion.hpp"
#include "../include/math/lie.hpp"
#include "../include/math/matrix.hpp"
#include "../include/math/types.hpp"

namespace msckf {
namespace test {

bool test_quaternion_default_constructor() {
    TEST_BEGIN("Quaternion default constructor");
    
    Quaterniond q;
    
    TEST_ASSERT_NEAR(1.0, q.w, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, q.x, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, q.y, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, q.z, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_quaternion_custom_constructor() {
    TEST_BEGIN("Quaternion custom constructor");
    
    Quaterniond q(0.7071, 0.7071, 0.0, 0.0);
    
    TEST_ASSERT_NEAR(0.7071, q.w, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.7071, q.x, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, q.y, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, q.z, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_quaternion_norm() {
    TEST_BEGIN("Quaternion norm");
    
    Quaterniond q(1.0, 2.0, 3.0, 4.0);
    float64 norm = q.norm();
    
    TEST_ASSERT_NEAR(sqrt(30.0), norm, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_quaternion_normalize() {
    TEST_BEGIN("Quaternion normalize");
    
    Quaterniond q(1.0, 2.0, 3.0, 4.0);
    q.normalize();
    
    float64 norm = q.norm();
    TEST_ASSERT_NEAR(1.0, norm, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_quaternion_normalized() {
    TEST_BEGIN("Quaternion normalized");
    
    Quaterniond q(1.0, 2.0, 3.0, 4.0);
    Quaterniond qn = q.normalized();
    
    float64 norm = qn.norm();
    TEST_ASSERT_NEAR(1.0, norm, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_quaternion_conjugate() {
    TEST_BEGIN("Quaternion conjugate");
    
    Quaterniond q(1.0, 2.0, 3.0, 4.0);
    Quaterniond qc = q.conjugate();
    
    TEST_ASSERT_NEAR(1.0, qc.w, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(-2.0, qc.x, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(-3.0, qc.y, TEST_TOLERANCE);
    TEST_ASSERT_NEAR(-4.0, qc.z, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_quaternion_inverse() {
    TEST_BEGIN("Quaternion inverse");
    
    Quaterniond q(1.0, 2.0, 3.0, 4.0);
    q.normalize();
    Quaterniond qi = q.inverse();
    
    Quaterniond identity = q * qi;
    
    TEST_ASSERT_NEAR(1.0, identity.w, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, identity.x, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, identity.y, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, identity.z, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_multiplication() {
    TEST_BEGIN("Quaternion multiplication");
    
    Quaterniond q1(1.0, 0.0, 0.0, 0.0);
    Quaterniond q2(0.7071, 0.7071, 0.0, 0.0);
    
    Quaterniond q3 = q1 * q2;
    
    TEST_ASSERT_NEAR(q2.w, q3.w, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(q2.x, q3.x, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(q2.y, q3.y, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(q2.z, q3.z, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_multiplication_associativity() {
    TEST_BEGIN("Quaternion multiplication associativity");
    
    Quaterniond q1(0.5, 0.5, 0.5, 0.5);
    Quaterniond q2(0.7071, 0.0, 0.7071, 0.0);
    Quaterniond q3(0.0, 0.0, 0.7071, 0.7071);
    
    q1.normalize();
    q2.normalize();
    q3.normalize();
    
    Quaterniond r1 = (q1 * q2) * q3;
    Quaterniond r2 = q1 * (q2 * q3);
    
    TEST_ASSERT_QUATERNION_NEAR(r1, r2, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_rotation_matrix() {
    TEST_BEGIN("Quaternion to rotation matrix");
    
    Quaterniond q;
    q = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), PI / 4);
    
    Matrix3d R = q.toRotationMatrix();
    
    float64 c = cos(PI / 4);
    float64 s = sin(PI / 4);
    
    TEST_ASSERT_NEAR(c, R(0, 0), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(-s, R(0, 1), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0, R(0, 2), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(s, R(1, 0), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(c, R(1, 1), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0, R(1, 2), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0, R(2, 0), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0, R(2, 1), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(1, R(2, 2), TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_from_axis_angle() {
    TEST_BEGIN("Quaternion from axis angle");
    
    Vector3d axis({0, 0, 1});
    float64 angle = PI / 2;
    
    Quaterniond q = Quaterniond::fromAxisAngle(axis, angle);
    
    float64 halfAngle = angle / 2;
    TEST_ASSERT_NEAR(cos(halfAngle), q.w, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0, q.x, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0, q.y, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(sin(halfAngle), q.z, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_rotate_vector() {
    TEST_BEGIN("Quaternion rotate vector");
    
    Vector3d v({1, 0, 0});
    Quaterniond q = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), PI / 2);
    
    Vector3d vr = q.rotate(v);
    
    TEST_ASSERT_NEAR(0, vr[0], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(1, vr[1], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0, vr[2], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_to_euler_angles() {
    TEST_BEGIN("Quaternion to Euler angles");
    
    float64 roll = 0.1;
    float64 pitch = 0.2;
    float64 yaw = 0.3;
    
    Quaterniond q = Quaterniond::fromEulerAngles(roll, pitch, yaw);
    Vector3d euler = q.toEulerAngles();
    
    TEST_ASSERT_NEAR(roll, euler[0], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(pitch, euler[1], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(yaw, euler[2], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_from_euler_angles() {
    TEST_BEGIN("Quaternion from Euler angles");
    
    Quaterniond q = Quaterniond::fromEulerAngles(0.0, 0.0, PI / 2);
    
    Vector3d v({1, 0, 0});
    Vector3d vr = q.rotate(v);
    
    TEST_ASSERT_NEAR(0, vr[0], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(1, vr[1], TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0, vr[2], TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_small_angle_rotation() {
    TEST_BEGIN("Quaternion small angle rotation");
    
    float64 smallAngle = 1e-8;
    Quaterniond q = Quaterniond::fromAxisAngle(Vector3d({0, 0, 1}), smallAngle);
    
    TEST_ASSERT_NEAR(1.0, q.w, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, q.x, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(0.0, q.y, TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(smallAngle / 2, q.z, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_exp_log_roundtrip() {
    TEST_BEGIN("Quaternion exp log roundtrip");
    
    Vector3d theta({0.1, 0.2, 0.3});
    
    Quaterniond q = Quaterniond::exp(theta);
    Vector3d theta2 = q.log();
    
    TEST_ASSERT_NEAR_VEC(theta, theta2, 3, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_quaternion_exp_small_angle() {
    TEST_BEGIN("Quaternion exp small angle");
    
    Vector3d theta({1e-10, 1e-10, 1e-10});
    
    Quaterniond q = Quaterniond::exp(theta);
    
    TEST_ASSERT_NEAR(1.0, q.w, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

REGISTER_TEST("Quaternion default constructor", test_quaternion_default_constructor, "Quaternion");
REGISTER_TEST("Quaternion custom constructor", test_quaternion_custom_constructor, "Quaternion");
REGISTER_TEST("Quaternion norm", test_quaternion_norm, "Quaternion");
REGISTER_TEST("Quaternion normalize", test_quaternion_normalize, "Quaternion");
REGISTER_TEST("Quaternion normalized", test_quaternion_normalized, "Quaternion");
REGISTER_TEST("Quaternion conjugate", test_quaternion_conjugate, "Quaternion");
REGISTER_TEST("Quaternion inverse", test_quaternion_inverse, "Quaternion");
REGISTER_TEST("Quaternion multiplication", test_quaternion_multiplication, "Quaternion");
REGISTER_TEST("Quaternion multiplication associativity", test_quaternion_multiplication_associativity, "Quaternion");
REGISTER_TEST("Quaternion to rotation matrix", test_quaternion_rotation_matrix, "Quaternion");
REGISTER_TEST("Quaternion from axis angle", test_quaternion_from_axis_angle, "Quaternion");
REGISTER_TEST("Quaternion rotate vector", test_quaternion_rotate_vector, "Quaternion");
REGISTER_TEST("Quaternion to Euler angles", test_quaternion_to_euler_angles, "Quaternion");
REGISTER_TEST("Quaternion from Euler angles", test_quaternion_from_euler_angles, "Quaternion");
REGISTER_TEST("Quaternion small angle rotation", test_quaternion_small_angle_rotation, "Quaternion");
REGISTER_TEST("Quaternion exp log roundtrip", test_quaternion_exp_log_roundtrip, "Quaternion");
REGISTER_TEST("Quaternion exp small angle", test_quaternion_exp_small_angle, "Quaternion");

bool test_so3_exp_identity() {
    TEST_BEGIN("SO3 exp identity");
    
    Vector3d theta({0, 0, 0});
    SO3<double> R = SO3<double>::exp(theta);
    
    Matrix3d Rmat = R.matrix();
    Matrix3d I = Matrix3d::identity();
    
    TEST_ASSERT_MATRIX_NEAR(I, Rmat, 3, 3, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_so3_exp_small_angle() {
    TEST_BEGIN("SO3 exp small angle");
    
    Vector3d theta({1e-8, 1e-8, 1e-8});
    SO3<double> R = SO3<double>::exp(theta);
    
    Matrix3d Rmat = R.matrix();
    Matrix3d I = Matrix3d::identity();
    
    TEST_ASSERT_MATRIX_NEAR(I, Rmat, 3, 3, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_so3_exp_rotation() {
    TEST_BEGIN("SO3 exp rotation");
    
    Vector3d theta({0, 0, PI / 4});
    SO3<double> R = SO3<double>::exp(theta);
    
    Matrix3d Rmat = R.matrix();
    
    float64 c = cos(PI / 4);
    float64 s = sin(PI / 4);
    
    TEST_ASSERT_NEAR(c, Rmat(0, 0), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(-s, Rmat(0, 1), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(s, Rmat(1, 0), TEST_TOLERANCE_LOOSE);
    TEST_ASSERT_NEAR(c, Rmat(1, 1), TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_so3_log_identity() {
    TEST_BEGIN("SO3 log identity");
    
    SO3<double> R(Matrix3d::identity());
    Vector3d theta = R.log();
    
    TEST_ASSERT_NEAR(0, theta[0], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0, theta[1], TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0, theta[2], TEST_TOLERANCE);
    
    TEST_END();
}

bool test_so3_exp_log_roundtrip() {
    TEST_BEGIN("SO3 exp log roundtrip");
    
    Vector3d theta({0.1, 0.2, 0.3});
    
    SO3<double> R = SO3<double>::exp(theta);
    Vector3d theta2 = R.log();
    
    TEST_ASSERT_NEAR_VEC(theta, theta2, 3, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_so3_left_jacobian_identity() {
    TEST_BEGIN("SO3 left jacobian identity");
    
    Vector3d theta({0, 0, 0});
    Matrix3d J = SO3<double>::leftJacobian(theta);
    
    Matrix3d I = Matrix3d::identity();
    TEST_ASSERT_MATRIX_NEAR(I, J, 3, 3, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_so3_left_jacobian_inverse_identity() {
    TEST_BEGIN("SO3 left jacobian inverse identity");
    
    Vector3d theta({0, 0, 0});
    Matrix3d Jinv = SO3<double>::leftJacobianInverse(theta);
    
    Matrix3d I = Matrix3d::identity();
    TEST_ASSERT_MATRIX_NEAR(I, Jinv, 3, 3, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_so3_left_jacobian_roundtrip() {
    TEST_BEGIN("SO3 left jacobian roundtrip");
    
    Vector3d theta({0.1, 0.2, 0.3});
    
    Matrix3d J = SO3<double>::leftJacobian(theta);
    Matrix3d Jinv = SO3<double>::leftJacobianInverse(theta);
    
    Matrix3d I = J * Jinv;
    Matrix3d I2 = Matrix3d::identity();
    
    TEST_ASSERT_MATRIX_NEAR(I2, I, 3, 3, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_so3_left_jacobian_near_pi() {
    TEST_BEGIN("SO3 left jacobian near pi");
    
    Vector3d theta({PI - 0.1, 0, 0});
    
    Matrix3d Jinv = SO3<double>::leftJacobianInverse(theta);
    
    for (uint32 i = 0; i < 3; ++i) {
        for (uint32 j = 0; j < 3; ++j) {
            TEST_ASSERT_FALSE(std::isnan(Jinv(i, j)));
            TEST_ASSERT_FALSE(std::isinf(Jinv(i, j)));
        }
    }
    
    TEST_END();
}

REGISTER_TEST("SO3 exp identity", test_so3_exp_identity, "SO3");
REGISTER_TEST("SO3 exp small angle", test_so3_exp_small_angle, "SO3");
REGISTER_TEST("SO3 exp rotation", test_so3_exp_rotation, "SO3");
REGISTER_TEST("SO3 log identity", test_so3_log_identity, "SO3");
REGISTER_TEST("SO3 exp log roundtrip", test_so3_exp_log_roundtrip, "SO3");
REGISTER_TEST("SO3 left jacobian identity", test_so3_left_jacobian_identity, "SO3");
REGISTER_TEST("SO3 left jacobian inverse identity", test_so3_left_jacobian_inverse_identity, "SO3");
REGISTER_TEST("SO3 left jacobian roundtrip", test_so3_left_jacobian_roundtrip, "SO3");
REGISTER_TEST("SO3 left jacobian near pi", test_so3_left_jacobian_near_pi, "SO3");

bool test_matrix_identity() {
    TEST_BEGIN("Matrix identity");
    
    Matrix3d I = Matrix3d::identity();
    
    TEST_ASSERT_NEAR(1.0, I(0, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(1.0, I(1, 1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(1.0, I(2, 2), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, I(0, 1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0.0, I(0, 2), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_matrix_zeros() {
    TEST_BEGIN("Matrix zeros");
    
    Matrix3d Z = Matrix3d();
    
    for (uint32 i = 0; i < 3; ++i) {
        for (uint32 j = 0; j < 3; ++j) {
            TEST_ASSERT_NEAR(0.0, Z(i, j), TEST_TOLERANCE);
        }
    }
    
    TEST_END();
}

bool test_matrix_addition() {
    TEST_BEGIN("Matrix addition");
    
    Matrix3d A, B;
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;
    
    B(0, 0) = 9; B(0, 1) = 8; B(0, 2) = 7;
    B(1, 0) = 6; B(1, 1) = 5; B(1, 2) = 4;
    B(2, 0) = 3; B(2, 1) = 2; B(2, 2) = 1;
    
    Matrix3d C = A + B;
    
    for (uint32 i = 0; i < 3; ++i) {
        for (uint32 j = 0; j < 3; ++j) {
            TEST_ASSERT_NEAR(10.0, C(i, j), TEST_TOLERANCE);
        }
    }
    
    TEST_END();
}

bool test_matrix_multiplication() {
    TEST_BEGIN("Matrix multiplication");
    
    Matrix3d A = Matrix3d::identity();
    Matrix3d B;
    B(0, 0) = 1; B(0, 1) = 2; B(0, 2) = 3;
    B(1, 0) = 4; B(1, 1) = 5; B(1, 2) = 6;
    B(2, 0) = 7; B(2, 1) = 8; B(2, 2) = 9;
    
    Matrix3d C = A * B;
    
    TEST_ASSERT_MATRIX_NEAR(B, C, 3, 3, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_matrix_transpose() {
    TEST_BEGIN("Matrix transpose");
    
    Matrix3d A;
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;
    
    Matrix3d At = A.transpose();
    
    TEST_ASSERT_NEAR(A(0, 0), At(0, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(A(0, 1), At(1, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(A(0, 2), At(2, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(A(1, 0), At(0, 1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(A(1, 1), At(1, 1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(A(1, 2), At(2, 1), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_matrix_inverse() {
    TEST_BEGIN("Matrix inverse");
    
    Matrix3d A;
    A(0, 0) = 4; A(0, 1) = 7; A(0, 2) = 2;
    A(1, 0) = 3; A(1, 1) = 6; A(1, 2) = 1;
    A(2, 0) = 2; A(2, 1) = 5; A(2, 2) = 3;
    
    Matrix3d Ainv = A.inverse();
    Matrix3d I = A * Ainv;
    
    Matrix3d Iexpected = Matrix3d::identity();
    TEST_ASSERT_MATRIX_NEAR(Iexpected, I, 3, 3, TEST_TOLERANCE_LOOSE);
    
    TEST_END();
}

bool test_matrix_determinant() {
    TEST_BEGIN("Matrix determinant");
    
    Matrix3d A = Matrix3d::identity();
    
    float64 det = 0;
    det += A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1));
    det -= A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0));
    det += A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
    
    TEST_ASSERT_NEAR(1.0, det, TEST_TOLERANCE);
    
    TEST_END();
}

bool test_skew_symmetric() {
    TEST_BEGIN("Skew symmetric");
    
    Vector3d v({1, 2, 3});
    Matrix3d S = skewSymmetric(v);
    
    TEST_ASSERT_NEAR(0, S(0, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(-3, S(0, 1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(2, S(0, 2), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(3, S(1, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0, S(1, 1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(-1, S(1, 2), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(-2, S(2, 0), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(1, S(2, 1), TEST_TOLERANCE);
    TEST_ASSERT_NEAR(0, S(2, 2), TEST_TOLERANCE);
    
    TEST_END();
}

bool test_skew_symmetric_cross_product() {
    TEST_BEGIN("Skew symmetric cross product");
    
    Vector3d a({1, 2, 3});
    Vector3d b({4, 5, 6});
    
    Matrix3d Sa = skewSymmetric(a);
    Vector3d cross1 = Sa * b;
    
    Vector3d cross2;
    cross2[0] = a[1] * b[2] - a[2] * b[1];
    cross2[1] = a[2] * b[0] - a[0] * b[2];
    cross2[2] = a[0] * b[1] - a[1] * b[0];
    
    TEST_ASSERT_NEAR_VEC(cross1, cross2, 3, TEST_TOLERANCE);
    
    TEST_END();
}

REGISTER_TEST("Matrix identity", test_matrix_identity, "Matrix");
REGISTER_TEST("Matrix zeros", test_matrix_zeros, "Matrix");
REGISTER_TEST("Matrix addition", test_matrix_addition, "Matrix");
REGISTER_TEST("Matrix multiplication", test_matrix_multiplication, "Matrix");
REGISTER_TEST("Matrix transpose", test_matrix_transpose, "Matrix");
REGISTER_TEST("Matrix inverse", test_matrix_inverse, "Matrix");
REGISTER_TEST("Matrix determinant", test_matrix_determinant, "Matrix");
REGISTER_TEST("Skew symmetric", test_skew_symmetric, "Matrix");
REGISTER_TEST("Skew symmetric cross product", test_skew_symmetric_cross_product, "Matrix");

}
}

#endif
