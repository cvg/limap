set(FOLDER_NAME "geometry")

LIMAP_ADD_LIBRARY(
    NAME limap_geometry
    SRCS
        camera_utils.h camera_utils.cc
        image_utils.h image_utils.cc
        camera_models.h
        groups.h groups.cc
        line2d.h line2d.cc
        line3d.h line3d.cc
        inf_line2d.h inf_line2d.cc
        inf_line3d.h inf_line3d.cc
        minimal_inf_line3d.h minimal_inf_line3d.cc
        line_metrics.h line_metrics.cc
        line_linker.h line_linker.cc
        ceres_angle_utils.h
        ceres_point_functions.h
        ceres_line_functions.h
        line_jacobians.h
        line_pixel_uncertainty.h line_pixel_uncertainty.cc
        groups/cone.h groups/cone.cc
        groups/cuboid.h groups/cuboid.cc
        groups/cylinder.h groups/cylinder.cc
        groups/ellipsoid.h groups/ellipsoid.cc
        groups/plane.h groups/plane.cc
        groups/projection.h
        groups/sphere.h groups/sphere.cc
        groups/vp.h groups/vp.cc
    PUBLIC_LINK_LIBS
        limap_util
        ${LIMAP_COLMAP_TARGET}
        ${CERES_LIBRARIES}
)

if(LIMAP_TESTS_ENABLED)
    LIMAP_ADD_TEST(
        NAME test_line_pixel_uncertainty
        SRCS test_line_pixel_uncertainty.cc
        LINK_LIBS limap_geometry
    )
endif()
