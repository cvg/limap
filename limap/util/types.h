#pragma once

#include <Eigen/Core>
#include <third-party/half.h>
#include <colmap/util/types.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "util/log_exceptions.h"

namespace py = pybind11;

namespace limap {

using V2F = Eigen::Vector2f;
using V3F = Eigen::Vector3f;
using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using V4D = Eigen::Vector4d;

using M2F = Eigen::Matrix2f;
using M3F = Eigen::Matrix3f;
using M2D = Eigen::Matrix2d;
using M3D = Eigen::Matrix3d;

const double EPS = 1e-12;

}

