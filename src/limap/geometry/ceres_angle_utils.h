#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace limap {

// Jet-safe clamp to [0, 1] for numerical safety (unit-vector dot products
// can slightly exceed 1.0 due to floating-point error).
template <typename T> inline T ClampToOne(const T &x) {
  return x > T(1) ? T(1) : x;
}

// All Angle* functions below assume unit-length input directions.
// This is guaranteed by construction:
//   - Line directions: d = R(uvec)*e1, always unit
//   - VP params: constrained by SphereManifold<3>
//   - Plane normals: constrained by ProductManifold<SphereManifold<3>, ...>

template <typename T> inline T AngleSine2D(const T dir1[2], const T dir2[2]) {
  // |sin(angle)| = |a x b| for unit vectors
  const T s = ceres::abs(dir1[0] * dir2[1] - dir1[1] * dir2[0]);
  return ClampToOne(s);
}

template <typename T> inline T AngleCosine2D(const T dir1[2], const T dir2[2]) {
  // |cos(angle)| = |a . b| for unit vectors
  const T c = ceres::abs(dir1[0] * dir2[0] + dir1[1] * dir2[1]);
  return ClampToOne(c);
}

template <typename T>
inline T AngleBetween2D(const T dir1[2], const T dir2[2]) {
  const T c = AngleCosine2D<T>(dir1, dir2);
  return ceres::acos(c);
}

template <typename T> inline T AngleSine3D(const T dir1[3], const T dir2[3]) {
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> a(dir1), b(dir2);
  // |sin(angle)| = ||a x b|| for unit vectors
  const T s = (a.cross(b)).norm();
  return ClampToOne(s);
}

template <typename T> inline T AngleCosine3D(const T dir1[3], const T dir2[3]) {
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> a(dir1), b(dir2);
  // |cos(angle)| = |a . b| for unit vectors
  const T c = ceres::abs(a.dot(b));
  return ClampToOne(c);
}

template <typename T>
inline T AngleBetween3D(const T dir1[3], const T dir2[3]) {
  const T c = AngleCosine3D<T>(dir1, dir2);
  return ceres::acos(c);
}

} // namespace limap
