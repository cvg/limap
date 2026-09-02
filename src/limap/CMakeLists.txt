add_subdirectory(util)
add_subdirectory(geometry)
add_subdirectory(evaluation)
add_subdirectory(scene)
add_subdirectory(image)
add_subdirectory(estimators)
add_subdirectory(sfm)

# Umbrella interface library for backward compatibility (pybind11, etc.)
add_library(limap INTERFACE)
target_link_libraries(limap INTERFACE
    limap_util
    limap_geometry
    limap_evaluation
    limap_scene
    limap_image
    limap_estimators
    limap_sfm
)
