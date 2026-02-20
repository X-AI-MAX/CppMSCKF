cmake_minimum_required(VERSION 3.16)

project(MSCKF 
    VERSION 1.0.0
    DESCRIPTION "Multi-State Constraint Kalman Filter for Embedded Systems"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MSCKF_BUILD_TESTS "Build unit tests" ON)
option(MSCKF_BUILD_EXAMPLES "Build example applications" OFF)
option(MSCKF_ENABLE_COLLABORATIVE "Enable multi-UAV collaborative module" ON)
option(MSCKF_ENABLE_SEMANTIC "Enable semantic assistance module" ON)
option(MSCKF_ENABLE_LSTM "Enable LSTM predictor module" ON)
option(MSCKF_ENABLE_FIXED_POINT "Enable fixed-point math support" ON)
option(MSCKF_STRICT_EMBEDDED "Strict embedded mode (no dynamic allocation)" OFF)

if(MSCKF_STRICT_EMBEDDED)
    add_compile_definitions(MSCKF_STRICT_EMBEDDED)
    set(MSCKF_ENABLE_COLLABORATIVE OFF CACHE BOOL "Disabled in strict embedded mode" FORCE)
    set(MSCKF_ENABLE_SEMANTIC OFF CACHE BOOL "Disabled in strict embedded mode" FORCE)
    set(MSCKF_ENABLE_LSTM OFF CACHE BOOL "Disabled in strict embedded mode" FORCE)
endif()

add_library(msckf INTERFACE)

target_include_directories(msckf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(msckf INTERFACE cxx_std_17)

if(MSVC)
    target_compile_options(msckf INTERFACE /W4 /WX-)
    target_compile_definitions(msckf INTERFACE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(msckf INTERFACE -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(msckf INTERFACE -Wno-unused-parameter -Wno-unused-variable)
    endif()
endif()

target_compile_definitions(msckf INTERFACE
    MSCKF_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    MSCKF_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    MSCKF_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

set(MSCK_HEADERS
    include/math/types.hpp
    include/math/matrix.hpp
    include/math/quaternion.hpp
    include/math/lie.hpp
    include/math/numerical_stability.hpp
    include/math/fixed_point.hpp
    include/core/state.hpp
    include/core/imu_propagator.hpp
    include/core/visual_updater.hpp
    include/core/marginalizer.hpp
    include/core/safety_guard.hpp
    include/core/consistency_checker.hpp
    include/core/adaptive_kernel.hpp
    include/core/incremental_schur.hpp
    include/core/extreme_scenario_handler.hpp
    include/hal/imu_types.hpp
    include/hal/imu_hal.hpp
    include/hal/camera_types.hpp
    include/hal/camera_hal.hpp
    include/hal/power_management.hpp
    include/vision/image_processing.hpp
    include/vision/corner_detector.hpp
    include/vision/klt_tracker.hpp
    include/vision/multi_camera.hpp
    include/system/memory_pool.hpp
    include/system/memory_safety.hpp
    include/system/lockfree_queue.hpp
    include/system/scheduler.hpp
    include/system/performance_monitor.hpp
    include/system/communication.hpp
    include/system/fault_injection.hpp
    include/system/code_optimization.hpp
    include/utility/black_box.hpp
    include/utility/battery_manager.hpp
    include/utility/return_to_home.hpp
    include/utility/zero_velocity_update.hpp
    include/utility/magnetic_rejection.hpp
    include/utility/auto_tuning.hpp
    include/audit/final_audit_report.hpp
    include/audit/system_documentation.hpp
    include/msckf_estimator.hpp
    include/msckf_extended.hpp
)

if(MSCKF_ENABLE_COLLABORATIVE)
    list(APPEND MSCK_HEADERS include/collaborative/collaborative_msckf.hpp)
    target_compile_definitions(msckf INTERFACE MSCKF_ENABLE_COLLABORATIVE)
endif()

if(MSCKF_ENABLE_SEMANTIC)
    list(APPEND MSCK_HEADERS include/semantic/semantic_msckf.hpp)
    target_compile_definitions(msckf INTERFACE MSCKF_ENABLE_SEMANTIC)
endif()

if(MSCKF_ENABLE_LSTM)
    list(APPEND MSCK_HEADERS include/core/lstm_predictor.hpp)
    target_compile_definitions(msckf INTERFACE MSCKF_ENABLE_LSTM)
endif()

if(MSCKF_ENABLE_FIXED_POINT)
    target_compile_definitions(msckf INTERFACE MSCKF_ENABLE_FIXED_POINT)
endif()

target_sources(msckf INTERFACE ${MSCK_HEADERS})

if(MSCKF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MSCKF_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)

install(TARGETS msckf
    EXPORT msckf-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT msckf-targets
    FILE MSCKFTargets.cmake
    NAMESPACE MSCKF::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSCKF
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MSCKFConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MSCKFConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSCKF
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MSCKFConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MSCKFConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MSCKFConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSCKF
)

message(STATUS "")
message(STATUS "MSCKF Configuration:")
message(STATUS "  Version:                ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:           ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Tests:            ${MSCKF_BUILD_TESTS}")
message(STATUS "  Build Examples:         ${MSCKF_BUILD_EXAMPLES}")
message(STATUS "  Strict Embedded Mode:   ${MSCKF_STRICT_EMBEDDED}")
message(STATUS "  Collaborative Module:   ${MSCKF_ENABLE_COLLABORATIVE}")
message(STATUS "  Semantic Module:        ${MSCKF_ENABLE_SEMANTIC}")
message(STATUS "  LSTM Predictor:         ${MSCKF_ENABLE_LSTM}")
message(STATUS "  Fixed-Point Math:       ${MSCKF_ENABLE_FIXED_POINT}")
message(STATUS "")
