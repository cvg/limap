set(FOLDER_NAME "scene")

LIMAP_ADD_LIBRARY(
    NAME limap_scene
    SRCS
        colmap_mvs_model.h colmap_mvs_model.cc
        group2d.h group2d.cc
        group3d.h group3d.cc
        group3d_with_active_labels.h
        wireframe.h wireframe.cc
        structure2d.h structure2d.cc
        structure2d_serialization.h structure2d_serialization.cc
        structure_correspondence_graph.h structure_correspondence_graph.cc
        line3d_with_active_labels.h
        structure_reconstruction.h structure_reconstruction.cc
        structure_reconstruction_io_binary.h structure_reconstruction_io_binary.cc
        structure_reconstruction_io_text.h structure_reconstruction_io_text.cc
        structure_database.h structure_database.cc
        structure_database_sqlite.h structure_database_sqlite.cc
        structure_database_cache.h structure_database_cache.cc
        holistic_reconstruction.h holistic_reconstruction.cc
        holistic_reconstruction_manager.h holistic_reconstruction_manager.cc
    PUBLIC_LINK_LIBS
        limap_geometry
        limap_util
        ${LIMAP_COLMAP_TARGET}
)
