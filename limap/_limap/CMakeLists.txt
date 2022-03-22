set(FOLDER_NAME "pylimap")

LIMAP_ADD_SOURCES(
    helpers.h
)

LIMAP_ADD_PYMODULE(_limap
    bindings.cc)

target_link_libraries(_limap PRIVATE
 ${LIMAP_INTERNAL_LIBRARIES} ${LIMAP_EXTERNAL_LIBRARIES})


