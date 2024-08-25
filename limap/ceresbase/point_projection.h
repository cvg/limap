#ifndef LIMAP_CERESBASE_POINT_PROJECTION_H
#define LIMAP_CERESBASE_POINT_PROJECTION_H

// Modified from the pixel-perfect-sfm project

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace limap {

template <typename T>
void CamFromImg(const T *kvec, const T x, const T y, T *u, T *v) {
  const T f1 = kvec[0];
  const T f2 = kvec[1];
  const T c1 = kvec[2];
  const T c2 = kvec[3];

  *u = (x - c1) / f1;
  *v = (y - c2) / f2;
}

template <typename T>
void ImgFromCam(const T *kvec, const T u, const T v, T *x, T *y) {
  const T f1 = kvec[0];
  const T f2 = kvec[1];
  const T c1 = kvec[2];
  const T c2 = kvec[3];

  *x = f1 * u + c1;
  *y = f2 * v + c2;
}

template <typename T>
void PixelToWorld(const T *kvec, const T *qvec, const T *tvec, const T x,
                  const T y, const T *depth, T *xyz) {
  T local_xyz[3];
  CamFromImg(kvec, x, y, &local_xyz[0], &local_xyz[1]);
  local_xyz[2] = T(1.0);
  for (int i = 0; i < 3; i++) {
    local_xyz[i] = local_xyz[i] * depth[0] - tvec[i];
  }

  Eigen::Quaternion<T> q(qvec[0], qvec[1], qvec[2], qvec[3]);
  Eigen::Map<Eigen::Matrix<T, 3, 1>> map(xyz);
  map = q.conjugate() * Eigen::Map<const Eigen::Matrix<T, 3, 1>>(local_xyz);
}

template <typename T>
inline void WorldToPixel(const T *kvec, const T *qvec, const T *tvec,
                         const T *xyz, T *xy) {
  T projection[3];
  ceres::QuaternionRotatePoint(qvec, xyz, projection);
  projection[0] += tvec[0];
  projection[1] += tvec[1];
  projection[2] += tvec[2];

  projection[0] /= projection[2]; // u
  projection[1] /= projection[2]; // v
  ImgFromCam(kvec, projection[0], projection[1], &xy[0], &xy[1]);
}

template <typename T> inline bool IsInsideZeroL(const T &value, double L) {
  return (value > 0.0 && value < L);
}

} // namespace limap

#endif
