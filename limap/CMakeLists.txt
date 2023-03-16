add_subdirectory(base)
add_subdirectory(ceresbase)
add_subdirectory(solvers)
add_subdirectory(vplib)
add_subdirectory(util)
add_subdirectory(pointsfm)
add_subdirectory(triangulation)
add_subdirectory(merging)
add_subdirectory(undistortion)
add_subdirectory(evaluation)
add_subdirectory(fitting)
add_subdirectory(estimators)
add_subdirectory(optimize)
add_subdirectory(structures)
if(INTERPOLATION_ENABLED)
    add_subdirectory(features)
endif()

add_subdirectory(_limap)

LIMAP_ADD_STATIC_LIBRARY(limap ${LIMAP_SOURCES})
target_link_libraries(limap
    ${LIMAP_INTERNAL_LIBRARIES} ${LIMAP_EXTERNAL_LIBRARIES})

