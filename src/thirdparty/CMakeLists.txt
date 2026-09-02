################################################################################
# Third-party dependencies: FetchContent + submodules
################################################################################
include(FetchContent)

################################################################################
# PoseLib
################################################################################
message(STATUS "Configuring PoseLib...")
if(FETCH_POSELIB)
    # PoseLib compiles with -Werror -Wfatal-errors by default. GCC 14 reports a
    # false-positive -Warray-bounds in Eigen's SSE loads under -O3, which breaks
    # the manylinux wheel build. Do not fail our build on a dependency's warnings.
    set(WERROR OFF CACHE BOOL "Disable -Werror in PoseLib" FORCE)
    FetchContent_Declare(PoseLib
        GIT_REPOSITORY    https://github.com/PoseLib/PoseLib.git
        GIT_TAG           1758bd6d20f607c276388d28b5ba88f3651e85e9
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(PoseLib)
else()
    find_package(PoseLib REQUIRED)
endif()
message(STATUS "Configuring PoseLib... done")

################################################################################
# COLMAP
################################################################################
message(STATUS "Configuring COLMAP...")
if(FETCH_COLMAP)
    set(LSD_ENABLED OFF CACHE BOOL "Disable LSD in COLMAP" FORCE)
    set(GUI_ENABLED OFF CACHE BOOL "Disable GUI in COLMAP" FORCE)
    set(CGAL_ENABLED OFF CACHE BOOL "Disable CGAL in COLMAP" FORCE)
    # limap's detectors come through hloc on the Python side; COLMAP's ONNX
    # runtime is pure build time and wheel size here. pycolmap's own wheel
    # scripts disable it for the same reason.
    set(ONNX_ENABLED OFF CACHE BOOL "Disable ONNX in COLMAP" FORCE)
    set(TESTS_ENABLED OFF CACHE BOOL "Disable tests in COLMAP" FORCE)
    # Corresponding to COLMAP 4.2.0 release.
    FetchContent_Declare(COLMAP
        GIT_REPOSITORY    https://github.com/colmap/colmap.git
        GIT_TAG           83583625d3ceb12dd8d6a42cc657696701adcee8
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(COLMAP)
else()
    find_package(COLMAP REQUIRED)
endif()
message(STATUS "Configuring COLMAP... done")

################################################################################
# JLinkage
################################################################################
message(STATUS "Configuring JLinkage...")
if(FETCH_JLINKAGE)
    FetchContent_Declare(JLinkage
        GIT_REPOSITORY    https://github.com/B1ueber2y/JLinkage.git
        GIT_TAG           75dadd555f81b1cf1b0f016d8cac76f3b554ba9b
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(JLinkage)
else()
    find_package(JLinkage REQUIRED)
endif()
message(STATUS "Configuring JLinkage... done")

################################################################################
# Google Benchmark (for benchmarks only)
################################################################################
if(LIMAP_BENCHMARK_ENABLED)
    FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
        EXCLUDE_FROM_ALL)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

################################################################################
# libigl
################################################################################
message(STATUS "Configuring libigl...")
set(LIBIGL_INSTALL OFF CACHE BOOL "Disable libigl install" FORCE)
FetchContent_Declare(libigl
    GIT_REPOSITORY    https://github.com/libigl/libigl.git
    GIT_TAG           cab03717c7f6f3657bed6438601abf43d84a208f
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(libigl)
message(STATUS "Configuring libigl... done")
