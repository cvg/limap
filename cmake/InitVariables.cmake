# Target name indirection for dependencies that can be fetched or system-installed.
# FetchContent creates non-namespaced targets, find_package creates namespaced ones.

# PoseLib
if(NOT FETCH_POSELIB)
    set(LIMAP_POSELIB_TARGET PoseLib::PoseLib)
else()
    set(LIMAP_POSELIB_TARGET PoseLib)
endif()

# COLMAP
if(NOT FETCH_COLMAP)
    set(LIMAP_COLMAP_TARGET colmap::colmap)
else()
    set(LIMAP_COLMAP_TARGET colmap)
endif()

# JLinkage
if(NOT FETCH_JLINKAGE)
    set(LIMAP_JLINKAGE_TARGET JLinkage::JLinkage)
else()
    set(LIMAP_JLINKAGE_TARGET JLinkage)
endif()

# Include/link directories
set(LIMAP_INCLUDE_DIRS
    ${HDF5_INCLUDE_DIRS}
    ${FREEIMAGE_INCLUDE_DIRS}
    ${COLMAP_INCLUDE_DIRS}
)

set(LIMAP_LINK_DIRS
    ${Boost_LIBRARY_DIRS}
)
