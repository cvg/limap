#pragma once

// We use Eigen from COLMAP for now to avoid conflict dependencies.
#include <colmap/geometry/rigid3.h>
#include <colmap/util/eigen_alignment.h>

namespace limap {

using V2F = Eigen::Vector2f;
using V3F = Eigen::Vector3f;
using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using V4D = Eigen::Vector4d;
using V6D = Eigen::Matrix<double, 6, 1>;

using M2F = Eigen::Matrix2f;
using M3F = Eigen::Matrix3f;
using M4F = Eigen::Matrix4f;
using M2D = Eigen::Matrix2d;
using M3D = Eigen::Matrix3d;
using M4D = Eigen::Matrix4d;
using M6D = Eigen::Matrix<double, 6, 6>;
using M8D = Eigen::Matrix<double, 8, 8>;

} // namespace limap
