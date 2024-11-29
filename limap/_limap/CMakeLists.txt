set(FOLDER_NAME "pylimap")

LIMAP_ADD_SOURCES(
    helpers.h
)

LIMAP_ADD_PYMODULE(_limap
    bindings.cc)

target_compile_definitions(_limap PRIVATE VERSION_INFO="${VERSION_INFO}")
target_link_libraries(_limap PRIVATE
 ${LIMAP_INTERNAL_LIBRARIES} ${LIMAP_EXTERNAL_LIBRARIES})
install(TARGETS _limap LIBRARY DESTINATION limap)
