cmake_minimum_required(VERSION 3.16)

if(NOT TARGET msckf)
    project(MSCKF_Tests CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(TEST_SOURCES
    main.cpp
)

set(TEST_HEADERS
    test_framework.hpp
    test_math.hpp
    test_numerical_stability.hpp
    test_state.hpp
    test_imu_propagator.hpp
    test_visual_updater.hpp
    test_marginalizer.hpp
    test_safety.hpp
    test_lockfree.hpp
    test_adaptive_kernel.hpp
    test_memory_pool.hpp
    test_extreme_scenario.hpp
    test_main.hpp
)

add_executable(msckf_tests ${TEST_SOURCES} ${TEST_HEADERS})

if(TARGET msckf)
    target_link_libraries(msckf_tests PRIVATE msckf)
else()
    set(MSCKF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_include_directories(msckf_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MSCKF_INCLUDE_DIR}/include
    )
endif()

if(MSVC)
    target_compile_options(msckf_tests PRIVATE /W4 /WX-)
    target_compile_definitions(msckf_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(msckf_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(Threads QUIET)
if(Threads_FOUND)
    target_link_libraries(msckf_tests PRIVATE Threads::Threads)
endif()

if(ENABLE_COVERAGE AND NOT MSVC)
    target_compile_options(msckf_tests PRIVATE --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(msckf_tests PRIVATE --coverage)
endif()

add_test(NAME MSCKF_UnitTests COMMAND msckf_tests)

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS msckf_tests
    COMMENT "Running MSCKF unit tests"
)

if(NOT MSVC AND NOT CMAKE_CROSSCOMPILING)
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)
    
    if(LCOV_PATH AND GENHTML_PATH)
        add_custom_target(coverage
            COMMAND ${LCOV_PATH} --zerocounters --directory ${CMAKE_BINARY_DIR}
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND ${LCOV_PATH} --capture --directory ${CMAKE_BINARY_DIR} --output-file coverage.info
            COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND ${GENHTML_PATH} coverage.info --output-directory coverage_report
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS msckf_tests
            COMMENT "Generating code coverage report"
        )
    endif()
endif()

message(STATUS "MSCKF Tests Configuration:")
message(STATUS "  Test executable: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/msckf_tests")
message(STATUS "  Coverage enabled: ${ENABLE_COVERAGE}")
