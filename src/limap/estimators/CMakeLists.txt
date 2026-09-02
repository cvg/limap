set(FOLDER_NAME "estimators")

LIMAP_ADD_LIBRARY(
    NAME limap_estimators
    SRCS
        # absolute_pose
        absolute_pose/point_line_absolute_pose.h
        absolute_pose/point_line_absolute_pose.cc
        # bundle_adjustment
        bundle_adjustment/custom_tolerance_callbacks.h
        bundle_adjustment/cost_functions.h
        bundle_adjustment/analytical_line_cost_functions.h
        bundle_adjustment/analytical_group_cost_functions.h
        bundle_adjustment/analytical_wireframe_cost_functions.h
        bundle_adjustment/group_cost_function_utils.h
        bundle_adjustment/group_cost_functions.h
        bundle_adjustment/wireframe_cost_functions.h
        bundle_adjustment/bundle_adjustment.h bundle_adjustment/bundle_adjustment.cc
        bundle_adjustment/group_bundle_adjustment.h bundle_adjustment/group_bundle_adjustment.cc
        bundle_adjustment/wireframe_bundle_adjustment.h bundle_adjustment/wireframe_bundle_adjustment.cc
        bundle_adjustment/structure_bundle_adjustment.h bundle_adjustment/structure_bundle_adjustment.cc
        # group3d
        group3d/group3d_estimator.h group3d/group3d_estimator.cc
        group3d/cone_estimator.h group3d/cone_estimator.cc
        group3d/cuboid_estimator.h group3d/cuboid_estimator.cc
        group3d/cylinder_estimator.h group3d/cylinder_estimator.cc
        group3d/ellipsoid_estimator.h group3d/ellipsoid_estimator.cc
        group3d/plane_estimator.h group3d/plane_estimator.cc
        group3d/sphere_estimator.h group3d/sphere_estimator.cc
        group3d/vp_estimator.h group3d/vp_estimator.cc
        # line3d
        line3d/line3d_estimator.h line3d/line3d_estimator.cc
        # triangulation
        triangulation/functions.h triangulation/functions.cc
        triangulation/triangulate_line_with_one_point.h
        triangulation/triangulate_line_with_one_point.cc
    PUBLIC_LINK_LIBS
        limap_scene
        limap_geometry
        limap_util
        ${LIMAP_COLMAP_TARGET}
        ${CERES_LIBRARIES}
        ${LIMAP_POSELIB_TARGET}
)

if(LIMAP_TESTS_ENABLED)
    LIMAP_ADD_TEST(
        NAME test_analytical_line_jacobians
        SRCS bundle_adjustment/test_analytical_line_jacobians.cc
        LINK_LIBS limap_estimators
    )

    LIMAP_ADD_TEST(
        NAME test_analytical_group_jacobians
        SRCS bundle_adjustment/test_analytical_group_jacobians.cc
        LINK_LIBS limap_estimators
    )

    LIMAP_ADD_TEST(
        NAME test_analytical_wireframe_jacobians
        SRCS bundle_adjustment/test_analytical_wireframe_jacobians.cc
        LINK_LIBS limap_estimators
    )

    LIMAP_ADD_TEST(
        NAME test_triangulation_covariance
        SRCS triangulation/test_triangulation_covariance.cc
        LINK_LIBS limap_estimators
    )

    LIMAP_ADD_TEST(
        NAME test_custom_tolerance_callbacks
        SRCS bundle_adjustment/test_custom_tolerance_callbacks.cc
        LINK_LIBS limap_estimators
    )

    LIMAP_ADD_TEST(
        NAME test_synthetic_structured_dataset
        SRCS
            bundle_adjustment/synthetic_structured_dataset.cc
            bundle_adjustment/test_synthetic_structured_dataset.cc
        LINK_LIBS limap_estimators
    )

    LIMAP_ADD_TEST(
        NAME test_bundle_adjustment
        SRCS
            bundle_adjustment/synthetic_structured_dataset.cc
            bundle_adjustment/test_bundle_adjustment.cc
        LINK_LIBS limap_estimators
    )
endif()

if(LIMAP_BENCHMARK_ENABLED)
    add_executable(benchmark_bundle_adjustment
        bundle_adjustment/synthetic_structured_dataset.cc
        bundle_adjustment/benchmark_bundle_adjustment.cc)
    target_link_libraries(benchmark_bundle_adjustment limap_estimators benchmark::benchmark)
    target_include_directories(benchmark_bundle_adjustment PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
endif()
