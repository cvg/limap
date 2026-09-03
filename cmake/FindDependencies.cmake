################################################################################
# Find system packages
################################################################################
find_package(OpenMP REQUIRED COMPONENTS C CXX)

find_package(Glog REQUIRED)
if(DEFINED glog_VERSION_MAJOR)
  # Older versions of glog don't export version variables.
  add_definitions("-DGLOG_VERSION_MAJOR=${glog_VERSION_MAJOR}")
  add_definitions("-DGLOG_VERSION_MINOR=${glog_VERSION_MINOR}")
endif()

# Ceres
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)

# Boost
find_package(Boost REQUIRED COMPONENTS
             graph
             program_options
             OPTIONAL_COMPONENTS
             system)

# GTest (for tests only)
if(LIMAP_TESTS_ENABLED)
    find_package(GTest REQUIRED)
endif()
