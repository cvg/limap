# Helper macros for the limap project.
# Modeled after COLMAP's CMakeHelper.cmake with named-argument macros.

# Enable solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_TARGETS_ROOT_FOLDER "cmake")
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER
             ${CMAKE_TARGETS_ROOT_FOLDER})
set(LIMAP_TARGETS_ROOT_FOLDER "limap_targets")

# Search for source files in a given directory, add them to a source group,
# and return paths to each found file.
macro(LIMAP_ADD_SOURCE_DIR SRC_DIR SRC_VAR)
    set(GLOB_EXPRESSIONS "")
    foreach(ARG ${ARGN})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/${ARG})
    endforeach()
    file(GLOB ${SRC_VAR} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         ${GLOB_EXPRESSIONS})
    string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})
    source_group(${GROUP_NAME} FILES ${${SRC_VAR}})
    unset(GLOB_EXPRESSIONS)
    unset(ARG)
    unset(GROUP_NAME)
endmacro(LIMAP_ADD_SOURCE_DIR)

# Add a library target with named arguments.
# Usage:
#   LIMAP_ADD_LIBRARY(
#       NAME limap_util
#       SRCS types.h kd_tree.h kd_tree.cc
#       PUBLIC_LINK_LIBS Eigen3::Eigen
#       PRIVATE_LINK_LIBS ${OpenCV_LIBRARIES}
#   )
macro(LIMAP_ADD_LIBRARY)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs NAME SRCS PRIVATE_LINK_LIBS PUBLIC_LINK_LIBS)
    cmake_parse_arguments(LIMAP_ADD_LIBRARY
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_library(${LIMAP_ADD_LIBRARY_NAME} STATIC ${LIMAP_ADD_LIBRARY_SRCS})
    set_target_properties(${LIMAP_ADD_LIBRARY_NAME} PROPERTIES FOLDER
        ${LIMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_include_directories(${LIMAP_ADD_LIBRARY_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${LIMAP_INCLUDE_DIRS})
    target_link_libraries(${LIMAP_ADD_LIBRARY_NAME}
        PRIVATE ${LIMAP_ADD_LIBRARY_PRIVATE_LINK_LIBS}
        PUBLIC ${LIMAP_ADD_LIBRARY_PUBLIC_LINK_LIBS})
    install(TARGETS ${LIMAP_ADD_LIBRARY_NAME} DESTINATION lib/limap)
endmacro(LIMAP_ADD_LIBRARY)

# Add a test executable with named arguments.
# Usage:
#   LIMAP_ADD_TEST(
#       NAME my_test
#       SRCS my_test.cc
#       LINK_LIBS limap_util
#   )
macro(LIMAP_ADD_TEST)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs NAME SRCS LINK_LIBS)
    cmake_parse_arguments(LIMAP_ADD_TEST
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(LIMAP_TESTS_ENABLED)
        set(LIMAP_ADD_TEST_TARGET "limap_${FOLDER_NAME}_${LIMAP_ADD_TEST_NAME}")
        add_executable(${LIMAP_ADD_TEST_TARGET} ${LIMAP_ADD_TEST_SRCS})
        set_target_properties(${LIMAP_ADD_TEST_TARGET} PROPERTIES
            FOLDER ${LIMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME}
            OUTPUT_NAME "${LIMAP_ADD_TEST_NAME}")
        target_link_libraries(${LIMAP_ADD_TEST_TARGET}
            ${LIMAP_ADD_TEST_LINK_LIBS}
            GTest::gtest_main)
        target_include_directories(${LIMAP_ADD_TEST_TARGET} PRIVATE
            ${PROJECT_SOURCE_DIR}/src)
        add_test(NAME "${FOLDER_NAME}/${LIMAP_ADD_TEST_NAME}"
            COMMAND $<TARGET_FILE:${LIMAP_ADD_TEST_TARGET}>)
    endif()
endmacro(LIMAP_ADD_TEST)

# Add an executable target.
macro(LIMAP_ADD_EXECUTABLE TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
        ${LIMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    install(TARGETS ${TARGET_NAME} DESTINATION bin/)
endmacro(LIMAP_ADD_EXECUTABLE)

# Add a pybind11 Python module linked against the limap library.
macro(LIMAP_ADD_PYMODULE TARGET_NAME)
    pybind11_add_module(${TARGET_NAME} ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE limap)
    install(TARGETS ${TARGET_NAME} DESTINATION bin/)
endmacro(LIMAP_ADD_PYMODULE)
