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

# In a wheel, glog/gflags/ceres are linked statically into this module, and
# pycolmap's extension carries its own static copies. Keep ours private, so the
# two do not resolve into a single gflags registry -- glog would then register
# its flags twice and abort the import with "flag 'timestamp_in_logfile_name'
# was defined more than once ... linked both statically and dynamically".
if(APPLE)
    # dyld coalesces exported weak definitions across loaded images, so gflags'
    # registry would be shared with pycolmap's copy. Export only the module init.
    target_link_options(_limap PRIVATE "LINKER:-exported_symbol,_PyInit__limap")
elseif(UNIX)
    target_link_options(_limap PRIVATE "LINKER:--exclude-libs,ALL")
endif()

install(TARGETS _limap LIBRARY DESTINATION limap)
