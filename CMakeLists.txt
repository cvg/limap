# cmake version to be used
cmake_minimum_required( VERSION 3.17 )

# project name
project(LIMAP)

################################################################################
# Include CMake dependencies
################################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -std=c++14")

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

################################################################################
# Options
################################################################################
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)

################################################################################
# Find packages
################################################################################
# find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED) #not-required
find_package(Eigen3 3.3 REQUIRED)
find_package(COLMAP REQUIRED)

################################################################################
# Compiler specific configuration
################################################################################

if(OPENMP_ENABLED)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

################################################################################
# Add sources
################################################################################

set(LIMAP_INCLUDE_DIRS
  ${HDF5_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}
  ${FREEIMAGE_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${COLMAP_INCLUDE_DIRS}
)

set(LIMAP_LINK_DIRS
  ${COLMAP_LINK_DIRS}
)

set(LIMAP_EXTERNAL_LIBRARIES
  ${CERES_LIBRARIES}
  ${FREEIMAGE_LIBRARIES}
  ${COLMAP_LIBRARIES}
  ${HDF5_C_LIBRARIES}
  ${Boost_LIBRARIES}
)

if(OPENMP_FOUND)
    list(APPEND LIMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif()

set(LIMAP_INTERNAL_LIBRARIES
  HighFive
  pybind11::module
  pybind11::embed
  JLinkage
  igl::core
)

include_directories(
    third-party
    limap
    ${LIMAP_INCLUDE_DIRS}
)

link_directories(${LIMAP_LINK_DIRS})

add_subdirectory(third-party)
include_directories(${JLINKAGE_INCLUDE_DIRS})
include_directories(${RANSACLIB_INCLUDE_DIRS})
add_subdirectory(limap)

################################################################################
# Generate source groups for Visual Studio, XCode, etc.
################################################################################
LIMAP_ADD_SOURCE_DIR(limap/vpdetection BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/pointsfm BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/triangulation BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/merging BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/undistortion BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/features BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/refinement BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/lineKA BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/lineBA BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/evaluation BASE_SRCS *.h *.cc)
LIMAP_ADD_SOURCE_DIR(limap/fitting BASE_SRCS *.h *.cc)

