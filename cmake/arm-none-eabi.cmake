set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ARM)

set(TOOLCHAIN_PREFIX arm-none-eabi-)

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_AR ${TOOLCHAIN_PREFIX}ar)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}objdump)
set(CMAKE_SIZE ${TOOLCHAIN_PREFIX}size)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CPU_PARAMETERS
    -mcpu=cortex-m7
    -mthumb
    -mfpu=fpv5-d16
    -mfloat-abi=hard
)

set(CMAKE_C_FLAGS_DEBUG "-O0 -g3" CACHE INTERNAL "")
set(CMAKE_C_FLAGS_RELEASE "-O3 -g0" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g0" CACHE INTERNAL "")

set(CMAKE_C_FLAGS "${CPU_PARAMETERS} -Wall -Wextra -Wpedantic -Wno-unused-parameter -fdata-sections -ffunction-sections" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS "${CPU_PARAMETERS} -Wall -Wextra -Wpedantic -Wno-unused-parameter -fdata-sections -ffunction-sections -fno-exceptions -fno-rtti" CACHE INTERNAL "")
set(CMAKE_ASM_FLAGS "${CPU_PARAMETERS} -x assembler-with-cpp" CACHE INTERNAL "")

set(CMAKE_EXE_LINKER_FLAGS "-specs=nano.specs -specs=nosys.specs -Wl,--gc-sections -Wl,--print-memory-usage" CACHE INTERNAL "")

set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_C_COMPILER> <CMAKE_C_LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> <CMAKE_CXX_LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

set(MSCKF_STRICT_EMBEDDED ON CACHE BOOL "Strict embedded mode for ARM cross-compilation" FORCE)

message(STATUS "ARM Cross-Compilation Toolchain:")
message(STATUS "  Processor: Cortex-M7")
message(STATUS "  FPU: fpv5-d16")
message(STATUS "  Float ABI: hard")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
