cmake_minimum_required(VERSION 3.17)

project(limap)

# limap.__version__. _FULL keeps the pre-release suffix that
# SKBUILD_PROJECT_VERSION drops (2.0.0rc1 -> 2.0.0).
if(NOT DEFINED VERSION_INFO)
    if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
        set(VERSION_INFO "${SKBUILD_PROJECT_VERSION_FULL}")
    else()
        set(VERSION_INFO "dev")
    endif()
endif()

################################################################################
# Include CMake dependencies
################################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    message(FATAL_ERROR "GCC version needs to be at least 9.1")
endif()

# MSVC does not define M_PI and friends without this.
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

################################################################################
# Options
#
# Build with tests and benchmarks (from the build/ directory):
#   cmake -DLIMAP_TESTS_ENABLED=ON -DLIMAP_BENCHMARK_ENABLED=ON ..
################################################################################
option(FETCH_POSELIB "Whether to use PoseLib with FetchContent or with self-installed software" ON)
option(FETCH_COLMAP "Whether to use COLMAP with FetchContent or with self-installed software" ON)
option(FETCH_JLINKAGE "Whether to use JLinkage with FetchContent or with self-installed software" ON)
option(CUDA_ENABLED "Whether to use CUDA (only for the third-party COLMAP)" OFF)
option(LIMAP_TESTS_ENABLED "Build test executables" OFF)
option(LIMAP_BENCHMARK_ENABLED "Build benchmark executables" OFF)

include(cmake/FindDependencies.cmake)

if(LIMAP_TESTS_ENABLED)
    enable_testing()
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

# Add third-party dependencies (FetchContent + submodules)
add_subdirectory(src/thirdparty)

# Target name indirection (must come after thirdparty so targets exist)
include(cmake/InitVariables.cmake)

link_directories(${LIMAP_LINK_DIRS})

# Add sources
add_subdirectory(src/limap)

# Python bindings (only when building via scikit-build)
if(DEFINED SKBUILD_PROJECT_NAME)
    add_subdirectory(src/pylimap)
endif()
