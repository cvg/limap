#ifndef LIMAP_UNDISTORTION_UNDISTORT_H_
#define LIMAP_UNDISTORTION_UNDISTORT_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include "base/camera.h"
#include "util/types.h"

namespace py = pybind11;

namespace limap {

namespace undistortion {

PinholeCamera Undistort(const std::string& imname_in, const PinholeCamera& camera, const std::string& imname_out);

} // namespace undistortion

} // namespace limap

#endif

