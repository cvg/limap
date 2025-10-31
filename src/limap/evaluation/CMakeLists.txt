set(FOLDER_NAME "evaluation")

LIMAP_ADD_SOURCES(
    bindings.cc
    base_evaluator.h base_evaluator.cc
    point_cloud_evaluator.h point_cloud_evaluator.cc
    mesh_evaluator.h mesh_evaluator.cc
    refline_evaluator.h refline_evaluator.cc
)

