#ifndef MSCKF_TESTS_TEST_FRAMEWORK_HPP
#define MSCKF_TESTS_TEST_FRAMEWORK_HPP

#include <cstring>
#include <cmath>
#include <cstdio>

namespace msckf {
namespace test {

constexpr float64 TEST_TOLERANCE = 1e-10;
constexpr float64 TEST_TOLERANCE_LOOSE = 1e-6;

struct TestResult {
    uint32 passed;
    uint32 failed;
    uint32 skipped;
    char currentTest[256];
    char lastError[512];
};

static TestResult g_testResult = {0, 0, 0, "", ""};

#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            snprintf(g_testResult.lastError, sizeof(g_testResult.lastError), \
                     "Assertion failed: %s at line %d in %s", #condition, __LINE__, __FILE__); \
            g_testResult.failed++; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_TRUE(condition) TEST_ASSERT(condition)

#define TEST_ASSERT_FALSE(condition) TEST_ASSERT(!(condition))

#define TEST_ASSERT_EQ(expected, actual) \
    do { \
        if ((expected) != (actual)) { \
            snprintf(g_testResult.lastError, sizeof(g_testResult.lastError), \
                     "Expected %s but got %s at line %d", #expected, #actual, __LINE__); \
            g_testResult.failed++; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(expected, actual, tolerance) \
    do { \
        float64 diff = abs((expected) - (actual)); \
        if (diff > (tolerance)) { \
            snprintf(g_testResult.lastError, sizeof(g_testResult.lastError), \
                     "Expected %.10f but got %.10f (diff=%.10f) at line %d", \
                     static_cast<float64>(expected), static_cast<float64>(actual), diff, __LINE__); \
            g_testResult.failed++; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR_VEC(expected, actual, size, tolerance) \
    do { \
        for (uint32 i = 0; i < (size); ++i) { \
            float64 diff = abs((expected)[i] - (actual)[i]); \
            if (diff > (tolerance)) { \
                snprintf(g_testResult.lastError, sizeof(g_testResult.lastError), \
                         "Vector mismatch at index %u: expected %.10f but got %.10f at line %d", \
                         i, static_cast<float64>((expected)[i]), static_cast<float64>((actual)[i]), __LINE__); \
                g_testResult.failed++; \
                return false; \
            } \
        } \
    } while(0)

#define TEST_ASSERT_MATRIX_NEAR(expected, actual, rows, cols, tolerance) \
    do { \
        for (uint32 i = 0; i < (rows); ++i) { \
            for (uint32 j = 0; j < (cols); ++j) { \
                float64 diff = abs((expected)(i, j) - (actual)(i, j)); \
                if (diff > (tolerance)) { \
                    snprintf(g_testResult.lastError, sizeof(g_testResult.lastError), \
                             "Matrix mismatch at (%u,%u): expected %.10f but got %.10f at line %d", \
                             i, j, static_cast<float64>((expected)(i, j)), static_cast<float64>((actual)(i, j)), __LINE__); \
                    g_testResult.failed++; \
                    return false; \
                } \
            } \
        } \
    } while(0)

#define TEST_ASSERT_QUATERNION_NEAR(q1, q2, tolerance) \
    do { \
        float64 dot = abs(q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z); \
        if (dot < 1.0 - (tolerance)) { \
            snprintf(g_testResult.lastError, sizeof(g_testResult.lastError), \
                     "Quaternion mismatch: q1=(%.6f,%.6f,%.6f,%.6f) q2=(%.6f,%.6f,%.6f,%.6f) at line %d", \
                     q1.w, q1.x, q1.y, q1.z, q2.w, q2.x, q2.y, q2.z, __LINE__); \
            g_testResult.failed++; \
            return false; \
        } \
    } while(0)

#define TEST_BEGIN(name) \
    snprintf(g_testResult.currentTest, sizeof(g_testResult.currentTest), "%s", name); \
    printf("  [TEST] %s... ", name)

#define TEST_END() \
    do { \
        g_testResult.passed++; \
        printf("PASSED\n"); \
        return true; \
    } while(0)

#define TEST_SKIP(reason) \
    do { \
        g_testResult.skipped++; \
        printf("SKIPPED (%s)\n", reason); \
        return true; \
    } while(0)

typedef bool (*TestFunction)();

struct TestCase {
    const char* name;
    TestFunction func;
    const char* category;
};

class TestRunner {
private:
    static constexpr uint32 MAX_TESTS = 500;
    TestCase tests_[MAX_TESTS];
    uint32 numTests_;
    uint32 currentCategory_;
    const char* categories_[50];
    
    uint32 totalPassed_;
    uint32 totalFailed_;
    uint32 totalSkipped_;

public:
    TestRunner() 
        : numTests_(0)
        , currentCategory_(0)
        , totalPassed_(0)
        , totalFailed_(0)
        , totalSkipped_(0) {}
    
    void registerTest(const char* name, TestFunction func, const char* category) {
        if (numTests_ < MAX_TESTS) {
            tests_[numTests_].name = name;
            tests_[numTests_].func = func;
            tests_[numTests_].category = category;
            numTests_++;
        }
    }
    
    void runAll() {
        printf("\n");
        printf("========================================\n");
        printf("   MSCKF Unit Test Suite\n");
        printf("========================================\n\n");
        
        const char* currentCategory = "";
        
        for (uint32 i = 0; i < numTests_; ++i) {
            if (strcmp(tests_[i].category, currentCategory) != 0) {
                currentCategory = tests_[i].category;
                printf("\n[%s]\n", currentCategory);
            }
            
            g_testResult = TestResult();
            bool result = tests_[i].func();
            
            if (result) {
                totalPassed_++;
            } else {
                totalFailed_++;
                printf("    ERROR: %s\n", g_testResult.lastError);
            }
        }
        
        printf("\n========================================\n");
        printf("   Test Results Summary\n");
        printf("========================================\n");
        printf("  Passed:  %u\n", totalPassed_);
        printf("  Failed:  %u\n", totalFailed_);
        printf("  Skipped: %u\n", totalSkipped_);
        printf("  Total:   %u\n", totalPassed_ + totalFailed_ + totalSkipped_);
        printf("  Coverage: %.1f%%\n", 
               100.0 * totalPassed_ / (totalPassed_ + totalFailed_ + totalSkipped_));
        printf("========================================\n\n");
        
        if (totalFailed_ > 0) {
            printf("!!! SOME TESTS FAILED !!!\n\n");
        } else {
            printf("ALL TESTS PASSED\n\n");
        }
    }
    
    void runCategory(const char* category) {
        printf("\n[%s Tests]\n", category);
        
        for (uint32 i = 0; i < numTests_; ++i) {
            if (strcmp(tests_[i].category, category) == 0) {
                g_testResult = TestResult();
                bool result = tests_[i].func();
                
                if (result) {
                    totalPassed_++;
                } else {
                    totalFailed_++;
                    printf("    ERROR: %s\n", g_testResult.lastError);
                }
            }
        }
    }
    
    uint32 getPassed() const { return totalPassed_; }
    uint32 getFailed() const { return totalFailed_; }
    uint32 getSkipped() const { return totalSkipped_; }
    bool allPassed() const { return totalFailed_ == 0; }
};

static TestRunner g_testRunner;

class TestRegistrar {
public:
    TestRegistrar(const char* name, TestFunction func, const char* category) {
        g_testRunner.registerTest(name, func, category);
    }
};

#define REGISTER_TEST(name, func, category) \
    static TestRegistrar registrar_##func(name, func, category)

}
}

#endif
