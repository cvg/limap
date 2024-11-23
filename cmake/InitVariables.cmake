# External libraries
set(LIMAP_EXTERNAL_LIBRARIES
  ${CERES_LIBRARIES}
  ${FREEIMAGE_LIBRARIES}
  ${HDF5_C_LIBRARIES}
  ${Boost_LIBRARIES}
)

# OpenMP
if(OPENMP_FOUND)
    list(APPEND LIMAP_EXTERNAL_LIBRARIES ${OpenMP_libomp_LIBRARY})
endif()

# colmap
if(NOT FETCH_COLMAP)
    list(APPEND LIMAP_EXTERNAL_LIBRARIES colmap::colmap)
else()
    list(APPEND LIMAP_EXTERNAL_LIBRARIES colmap)
endif()

# PoseLib
if(NOT FETCH_POSELIB)
    list(APPEND LIMAP_EXTERNAL_LIBRARIES PoseLib::PoseLib)
else()
    list(APPEND LIMAP_EXTERNAL_LIBRARIES PoseLib)
endif()

# Internal libraries
set(LIMAP_INTERNAL_LIBRARIES
  HighFive
  pybind11::module
  JLinkage
  igl::core
)

# Directories to include from dependencies
set(LIMAP_INCLUDE_DIRS
  ${HDF5_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${FREEIMAGE_INCLUDE_DIRS}
  ${COLMAP_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
)
