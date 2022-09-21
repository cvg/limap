set(FOLDER_NAME "line_refinement")

LIMAP_ADD_SOURCES(
    bindings.cc
    cost_functions.h pixel_cost_functions.h
    refine.h refine.cc
    refinement_config.h
)

