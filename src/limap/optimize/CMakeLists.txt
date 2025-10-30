set(FOLDER_NAME "optimize")

add_subdirectory(line_refinement)
add_subdirectory(hybrid_bundle_adjustment)
add_subdirectory(global_pl_association)
add_subdirectory(hybrid_localization)

LIMAP_ADD_SOURCES(
    bindings.cc
)

