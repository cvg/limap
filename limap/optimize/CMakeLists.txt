set(FOLDER_NAME "optimize")

add_subdirectory(line_refinement)
add_subdirectory(line_bundle_adjustment)

LIMAP_ADD_SOURCES(
    bindings.cc
)

