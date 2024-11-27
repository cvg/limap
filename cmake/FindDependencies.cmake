################################################################################
# Find packages
################################################################################
find_package(Eigen3 3.4 REQUIRED)
find_package(FreeImage REQUIRED)
find_package(Glog REQUIRED)
if(DEFINED glog_VERSION_MAJOR)
  # Older versions of glog don't export version variables.
  add_definitions("-DGLOG_VERSION_MAJOR=${glog_VERSION_MAJOR}")
  add_definitions("-DGLOG_VERSION_MINOR=${glog_VERSION_MINOR}")
endif()
find_package(Boost REQUIRED COMPONENTS
             graph
             program_options
             system)

# Ceres
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)
if(${CERES_VERSION} VERSION_LESS "2.2.0")
    # ceres 2.2.0 changes the interface of local parameterization
    add_definitions("-DCERES_PARAMETERIZATION_ENABLED")
endif()
if(INTERPOLATION_ENABLED)
    message(STATUS "Enabling pixelwise optimization with ceres interpolation. This should be disabled for clang.")
    add_definitions("-DINTERPOLATION_ENABLED")
else()
    message(STATUS "Disabling pixelwise optimization with ceres interpolation.")
endif()

# OpenMP
if(OPENMP_ENABLED)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Enabling OpenMP support")
    add_definitions("-DOPENMP_ENABLED")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# PoseLib
include(FetchContent)
FetchContent_Declare(PoseLib
    GIT_REPOSITORY    https://github.com/PoseLib/PoseLib.git
    GIT_TAG           a84c545a9895e46d12a3f5ccde2581c25e6a6953
    EXCLUDE_FROM_ALL
)
message(STATUS "Configuring PoseLib...")
if (FETCH_POSELIB) 
    FetchContent_MakeAvailable(PoseLib)
else()
    find_package(PoseLib REQUIRED)
endif()
message(STATUS "Configuring PoseLib... done")

# COLMAP
FetchContent_Declare(COLMAP
    GIT_REPOSITORY    https://github.com/colmap/colmap.git
    GIT_TAG           63b2cc000de32dc697f45ef1576dec7e67abddbc 
    EXCLUDE_FROM_ALL
)
message(STATUS "Configuring COLMAP...")
if (FETCH_COLMAP) 
    FetchContent_MakeAvailable(COLMAP)
else()
    find_package(COLMAP REQUIRED)
endif()
message(STATUS "Configuring COLMAP... done")

# JLinkage
FetchContent_Declare(JLinkage
    GIT_REPOSITORY    https://github.com/B1ueber2y/JLinkage.git
    GIT_TAG           452d67eda005db01a02071a5af8f0eced0a02079
    EXCLUDE_FROM_ALL
)
message(STATUS "Configuring JLinkage...")
if (FETCH_JLINKAGE)
    FetchContent_MakeAvailable(JLinkage)
else()
    find_package(JLinkage REQUIRED)
endif()
message(STATUS "Configuring JLinkage... done")
