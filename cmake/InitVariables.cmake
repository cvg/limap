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

if(NOT FETCH_JLINKAGE)
    list(APPEND LIMAP_EXTERNAL_LIBRARIES JLinkage::JLinkage)
else()
    list(APPEND LIMAP_EXTERNAL_LIBRARIES JLinkage)
endif()

# Internal libraries
set(LIMAP_INTERNAL_LIBRARIES
  HighFive
  pybind11::module
  igl::core
)

# Init variables for directories
set(LIMAP_INCLUDE_DIRS
  ${HDF5_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${FREEIMAGE_INCLUDE_DIRS}
  ${COLMAP_INCLUDE_DIRS}
)

set(LIMAP_LINK_DIRS
  ${Boost_LIBRARY_DIRS}
)
