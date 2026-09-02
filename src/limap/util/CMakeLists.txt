set(FOLDER_NAME "util")

LIMAP_ADD_LIBRARY(
    NAME limap_util
    SRCS
        types.h
        eigen_types.h
        kd_tree.h kd_tree.cc
    PUBLIC_LINK_LIBS
        ${LIMAP_COLMAP_TARGET}
        Eigen3::Eigen
)
