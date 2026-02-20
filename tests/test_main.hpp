#ifndef MSCKF_TESTS_TEST_MAIN_HPP
#define MSCKF_TESTS_TEST_MAIN_HPP

#include "test_framework.hpp"
#include "test_math.hpp"
#include "test_numerical_stability.hpp"
#include "test_state.hpp"
#include "test_imu_propagator.hpp"
#include "test_visual_updater.hpp"
#include "test_marginalizer.hpp"
#include "test_safety.hpp"
#include "test_lockfree.hpp"
#include "test_adaptive_kernel.hpp"
#include "test_memory_pool.hpp"
#include "test_extreme_scenario.hpp"

namespace msckf {
namespace test {

void runAllTests() {
    g_testRunner.runAll();
}

int getTestResult() {
    return g_testRunner.allPassed() ? 0 : 1;
}

}
}

#endif
