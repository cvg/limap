# Find Python and pybind11
if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python REQUIRED COMPONENTS Interpreter ${DEV_MODULE})
find_package(pybind11 3.0.1 REQUIRED)

file(GLOB_RECURSE PYLIMAP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

pybind11_add_module(_limap ${PYLIMAP_SOURCES})
target_include_directories(_limap PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${LIMAP_INCLUDE_DIRS}
)
target_link_libraries(_limap PRIVATE limap)
target_compile_definitions(_limap PRIVATE VERSION_INFO="${VERSION_INFO}")

install(TARGETS _limap LIBRARY DESTINATION limap)
