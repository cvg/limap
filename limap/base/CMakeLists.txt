set(FOLDER_NAME "base")

LIMAP_ADD_SOURCES(
    graph.h graph.cc
    camera.h camera.cc
    camera_models.h
    transforms.h transforms.cc
    camera_view.h camera_view.cc
    image_collection.h image_collection.cc
    pose.h pose.cc

    pointtrack.h pointtrack.cc
    linebase.h linebase.cc
    linetrack.h linetrack.cc
    line_dists.h line_dists.cc
    line_linker.h line_linker.cc
    infinite_line.h infinite_line.cc

    bindings.cc
)

