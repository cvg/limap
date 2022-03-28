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

Camera UndistortCamera(const std::string& imname_in, const Camera& camera, const std::string& imname_out);

CameraView UndistortCameraView(const std::string& imname_in, const CameraView& view, const std::string& imname_out);

} // namespace undistortion

} // namespace limap

#endif

