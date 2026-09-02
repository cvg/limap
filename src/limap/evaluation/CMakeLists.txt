set(FOLDER_NAME "evaluation")

LIMAP_ADD_LIBRARY(
    NAME limap_evaluation
    SRCS
        base_evaluator.h base_evaluator.cc
        point_cloud_evaluator.h point_cloud_evaluator.cc
        mesh_evaluator.h mesh_evaluator.cc
        refline_evaluator.h refline_evaluator.cc
    PUBLIC_LINK_LIBS
        limap_geometry
        limap_util
        igl::core
    PRIVATE_LINK_LIBS
        OpenMP::OpenMP_CXX
)
