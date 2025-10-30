set(FOLDER_NAME "triangulation")

LIMAP_ADD_SOURCES(
    bindings.cc
    functions.h functions.cc
    base_line_triangulator.h base_line_triangulator.cc
    global_line_triangulator.h global_line_triangulator.cc
)

