# This macro was borrowed from the pixel-perfect-sfm project.

# This macro will search for source files in a given directory, will add them
# to a source group (folder within a project), and will then return paths to
# each of the found files. The usage of the macro is as follows:
# LIMAP_ADD_SOURCE_DIR(
#     <source directory to search>
#     <output variable with found source files>
#     <search expressions such as *.h *.cc>)
macro(LIMAP_ADD_SOURCE_DIR SRC_DIR SRC_VAR)
    # Create the list of expressions to be used in the search.
    set(GLOB_EXPRESSIONS "")
    foreach(ARG ${ARGN})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/${ARG})
    endforeach()
    # Perform the search for the source files.
    file(GLOB ${SRC_VAR} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         ${GLOB_EXPRESSIONS})
    # Create the source group.
    string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})
    source_group(${GROUP_NAME} FILES ${${SRC_VAR}})
    # Clean-up.
    unset(GLOB_EXPRESSIONS)
    unset(ARG)
    unset(GROUP_NAME)
endmacro(LIMAP_ADD_SOURCE_DIR)

# Macro to add source files to COLMAP library.
macro(LIMAP_ADD_SOURCES)
    set(SOURCE_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else()
            list(APPEND SOURCE_FILES
                 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
    endforeach()
    set(LIMAP_SOURCES ${LIMAP_SOURCES} ${SOURCE_FILES} PARENT_SCOPE)
endmacro(LIMAP_ADD_SOURCES)

# Replacement for the normal add_library() command. The syntax remains the same
# in that the first argument is the target name, and the following arguments
# are the source files to use when building the target.
macro(LIMAP_ADD_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${LIMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/limap/)
endmacro(LIMAP_ADD_LIBRARY)
macro(LIMAP_ADD_STATIC_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} STATIC ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${LIMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION lib/limap)
endmacro(LIMAP_ADD_STATIC_LIBRARY)

# Replacement for the normal add_executable() command. The syntax remains the
# same in that the first argument is the target name, and the following
# arguments are the source files to use when building the target.
macro(LIMAP_ADD_EXECUTABLE TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_executable(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${LIMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(${TARGET_NAME} limap)
    install(TARGETS ${TARGET_NAME} DESTINATION bin/)
endmacro(LIMAP_ADD_EXECUTABLE)


macro(LIMAP_ADD_PYMODULE TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    pybind11_add_module(${TARGET_NAME} ${ARGN})
    # set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
    #     ${LIMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(${TARGET_NAME} PRIVATE limap)
    install(TARGETS ${TARGET_NAME} DESTINATION bin/)
endmacro(LIMAP_ADD_PYMODULE)

