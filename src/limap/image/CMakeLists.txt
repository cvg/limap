set(FOLDER_NAME "image")

LIMAP_ADD_LIBRARY(
    NAME limap_image
    SRCS
        groups/group_voting.h groups/group_voting.cc
        groups/vplib/base_vp_detector.h groups/vplib/base_vp_detector.cc
        groups/vplib/convert_to_groups.h groups/vplib/convert_to_groups.cc
        groups/vplib/vpbase.h groups/vplib/vpbase.cc
        groups/vplib/JLinkage/JLinkage.h groups/vplib/JLinkage/JLinkage.cc
    PUBLIC_LINK_LIBS
        limap_scene
        limap_geometry
        limap_util
        ${LIMAP_COLMAP_TARGET}
    PRIVATE_LINK_LIBS
        ${LIMAP_JLINKAGE_TARGET}
)
