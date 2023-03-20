set(FOLDER_NAME "vplib")

add_subdirectory(JLinkage)

LIMAP_ADD_SOURCES(
    vpbase.h
    base_vp_detector.h base_vp_detector.cc

    vptrack.h vptrack.cc
    global_vptrack_constructor.h global_vptrack_constructor.cc

    bindings.cc
)

