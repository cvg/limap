set(FOLDER_NAME "estimators")

add_subdirectory(absolute_pose)

LIMAP_ADD_SOURCES(
    bindings.cc
)
