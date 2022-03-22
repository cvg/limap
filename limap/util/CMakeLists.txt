set(FOLDER_NAME "util")

LIMAP_ADD_SOURCES(
    types.h
    simple_logger.h simple_logger.cc
    log_exceptions.h

    nanoflann.hpp
    kd_tree.h kd_tree.cpp
)

