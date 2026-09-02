set(FOLDER_NAME "sfm")

LIMAP_ADD_LIBRARY(
    NAME limap_sfm
    SRCS
        fitted_line_merging.h fitted_line_merging.cc
        line_triangulation_utils.h line_triangulation_utils.cc
        global_line_triangulation.h global_line_triangulation.cc
        global_group_triangulation.h global_group_triangulation.cc
        global_structure_triangulation.h global_structure_triangulation.cc
        group_verification.h group_verification.cc
        incremental_line_triangulator.h incremental_line_triangulator.cc
        incremental_group_triangulator.h incremental_group_triangulator.cc
        incremental_structure_triangulator.h incremental_structure_triangulator.cc
        structure_observation_manager.h structure_observation_manager.cc
        structure_incremental_mapper.h structure_incremental_mapper.cc
        structure_incremental_pipeline.h structure_incremental_pipeline.cc
    PUBLIC_LINK_LIBS
        limap_estimators
        limap_scene
        limap_geometry
        limap_util
        ${LIMAP_COLMAP_TARGET}
    PRIVATE_LINK_LIBS
        ${CERES_LIBRARIES}
)
