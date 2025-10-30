#pragma once

#include <Eigen/Core>
#include <colmap/util/types.h>
#include <third-party/half.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <colmap/util/logging.h>

namespace py = pybind11;

namespace limap {

using Node2d = std::pair<uint16_t, uint16_t>; // (img_id, feature_id), change to
                                              // int if there exists id > 65535

using V2F = Eigen::Vector2f;
using V3F = Eigen::Vector3f;
using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using V4D = Eigen::Vector4d;

using M2F = Eigen::Matrix2f;
using M3F = Eigen::Matrix3f;
using M4F = Eigen::Matrix4f;
using M2D = Eigen::Matrix2d;
using M3D = Eigen::Matrix3d;
using M4D = Eigen::Matrix4d;
using M6D = Eigen::Matrix<double, 6, 6>;
using M8D = Eigen::Matrix<double, 8, 8>;

const double EPS = 1e-12;

inline V3D homogeneous(const V2D &v2d) { return V3D(v2d(0), v2d(1), 1.0); }
inline V4D homogeneous(const V3D &v3d) {
  return V4D(v3d(0), v3d(1), v3d(2), 1.0);
}
inline V2D dehomogeneous(const V3D &v3d) {
  return V2D(v3d(0), v3d(1)) / (v3d(2) + EPS);
}
inline V3D dehomogeneous(const V4D &v4d) {
  return V3D(v4d(0), v4d(1), v4d(2)) / (v4d(3) + EPS);
}

} // namespace limap
