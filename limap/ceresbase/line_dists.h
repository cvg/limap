#ifndef LIMAP_CERESBASE_LINE_DISTS_H
#define LIMAP_CERESBASE_LINE_DISTS_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace limap {

template <typename T>
T CeresComputeDist2D_sine(const T dir1[2], const T dir2[2]) {
  T dir1_norm = ceres::sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] + EPS);
  T dir2_norm = ceres::sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] + EPS);
  T sine = (dir1[0] * dir2[1] - dir1[1] * dir2[0]) / (dir1_norm * dir2_norm);
  sine = ceres::abs(sine);
  if (sine > T(1.0))
    sine = T(1.0);
  return sine;
}

template <typename T>
T CeresComputeDist2D_cosine(const T dir1[2], const T dir2[2]) {
  T dir1_norm = ceres::sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] + EPS);
  T dir2_norm = ceres::sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] + EPS);
  T cosine = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) / (dir1_norm * dir2_norm);
  cosine = ceres::abs(cosine);
  if (cosine > T(1.0))
    cosine = T(1.0);
  return cosine;
}

template <typename T>
T CeresComputeDist2D_angle(const T dir1[2], const T dir2[2]) {
  T cosine = CeresComputeDist2D_cosine<T>(dir1, dir2);
  T angle = ceres::acos(cosine);
  if (ceres::IsNaN(angle) || ceres::IsInfinite(angle))
    angle = T(0.0);
  return angle;
}

template <typename T>
T CeresComputeDist3D_sine(const T dir1[3], const T dir2[3]) {
  T dir1_norm = ceres::sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] +
                            dir1[2] * dir1[2] + EPS);
  T dir2_norm = ceres::sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] +
                            dir2[2] * dir2[2] + EPS);
  T dir1_normalized[3], dir2_normalized[3];
  for (size_t i = 0; i < 3; ++i) {
    dir1_normalized[i] = dir1[i] / dir1_norm;
    dir2_normalized[i] = dir2[i] / dir2_norm;
  }
  T res[3];
  ceres::CrossProduct(dir1_normalized, dir2_normalized, res);
  T sine =
      ceres::sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + EPS);
  if (sine > T(1.0))
    sine = T(1.0);
  return sine;
}

template <typename T>
T CeresComputeDist3D_cosine(const T dir1[3], const T dir2[3]) {
  T dir1_norm = ceres::sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] +
                            dir1[2] * dir1[2] + EPS);
  T dir2_norm = ceres::sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] +
                            dir2[2] * dir2[2] + EPS);
  T dir1_normalized[3], dir2_normalized[3];
  for (size_t i = 0; i < 3; ++i) {
    dir1_normalized[i] = dir1[i] / dir1_norm;
    dir2_normalized[i] = dir2[i] / dir2_norm;
  }
  T cosine = T(0.0);
  for (size_t i = 0; i < 3; ++i) {
    cosine += dir1_normalized[i] * dir2_normalized[i];
  }
  cosine = ceres::abs(cosine);
  if (cosine > T(1.0))
    cosine = T(1.0);
  return cosine;
}

template <typename T>
T CeresComputeDist3D_angle(const T dir1[3], const T dir2[3]) {
  T cosine = CeresComputeDist3D_cosine<T>(dir1, dir2);
  T angle = ceres::acos(cosine);
  if (ceres::IsNaN(angle) || ceres::IsInfinite(angle))
    angle = T(0.0);
  return angle;
}

} // namespace limap

#endif
