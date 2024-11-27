# cmake version to be used
cmake_minimum_required( VERSION 3.17 )

# project name
project(LIMAP)

################################################################################
# Include CMake dependencies
################################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    message(FATAL_ERROR "GCC version needs to be at least 9.1")
endif()
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

################################################################################
# Options
################################################################################
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)
option(INTERPOLATION_ENABLED "Whether to enable interpolation-based pixel-perfect optimization" OFF)
option(FETCH_POSELIB "Whether to use PoseLib with FetchContent or with self-installed software" ON)
option(FETCH_COLMAP "Whether to use COLMAP with FetchContent or with self-installed software" ON)
option(CUDA_ENABLED "Whether to use CUDA (only for the third-party COLMAP)" OFF)
option(FETCH_JLINKAGE "Whether to use JLinkage with FetchContent or with self-installed software" ON)

include(cmake/FindDependencies.cmake)
include(cmake/InitVariables.cmake)

# include directories
include_directories(
    third-party
    limap
    ${PROJECT_SOURCE_DIR}
    ${LIMAP_INCLUDE_DIRS}
)
link_directories(${LIMAP_LINK_DIRS})

# Add sources
add_subdirectory(third-party)
include_directories(${RANSACLIB_INCLUDE_DIRS})
add_subdirectory(limap)

# Install find_package scripts for dependencies.
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    DESTINATION share/colmap
    FILES_MATCHING PATTERN "Find*.cmake")

################################################################################
# Generate source groups for Visual Studio, XCode, etc.
################################################################################
LIMAP_ADD_SOURCE_DIR(limap/base BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/ceresbase BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/solvers BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/vplib BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/pointsfm BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/undistortion BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/fitting BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/estimators BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/triangulation BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/merging BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/features BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/evaluation BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/structures BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/optimize BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/optimize/line_refinement BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/optimize/hybrid_bundle_adjustment BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/optimize/global_pl_association BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/optimize/line_localization BASE_SRCS *.h *.cc)

