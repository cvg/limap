set(FOLDER_NAME "vplib")

LIMAP_ADD_SOURCES(
    bindings.cc
    vpbase.h
    base_vp_detector.h base_vp_detector.cc
    jlinkage.h jlinkage.cc

    vptrack.h vptrack.cc
    global_vptrack_constructor.h global_vptrack_constructor.cc
)

