#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace limap {

// [Note] All the kvec here is a 4-dim array: [fx, fy, cx, cy]

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

  Eigen::Map<Eigen::Quaternion<T>> q(qvec); // xyzw
  Eigen::Map<Eigen::Matrix<T, 3, 1>> map(xyz);
  map = q.conjugate() * Eigen::Map<const Eigen::Matrix<T, 3, 1>>(local_xyz);
}

template <typename T>
inline void WorldToPixel(const T *kvec, const T *qvec, const T *tvec,
                         const T *xyz, T *xy) {
  Eigen::Map<const Eigen::Quaternion<T>> q(qvec); // xyzw
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pw(xyz);

  const Eigen::Matrix<T, 3, 1> Pc = q * Pw + t;

  const T u = Pc(0) / Pc(2);
  const T v = Pc(1) / Pc(2);

  ImgFromCam(kvec, u, v, &xy[0], &xy[1]);
}

template <typename T> inline bool IsInsideZeroL(const T &value, double L) {
  return (value > 0.0 && value < L);
}

} // namespace limap
