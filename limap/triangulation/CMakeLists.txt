set(FOLDER_NAME "triangulation")

LIMAP_ADD_SOURCES(
    bindings.cc
    functions.h functions.cc
    triangulator.h triangulator.cc
    build_initial_graph.h build_initial_graph.cc
)

