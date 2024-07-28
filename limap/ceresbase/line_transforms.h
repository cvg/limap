#ifndef LIMAP_CERESBASE_LINE_TRANSFORMS_H
#define LIMAP_CERESBASE_LINE_TRANSFORMS_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace limap {

template <typename T>
void MinimalPluckerToPlucker(const T uvec[4], const T wvec[2], T d[3], T m[3]) {
  // [LINK]
  // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
  // [LINK] https://hal.archives-ouvertes.fr/hal-00092589/document
  // Refer to "base/infinite_line.h"
  T rotmat[3 * 3];
  ceres::QuaternionToRotation(uvec, rotmat);
  T w1, w2;
  w1 = ceres::abs(wvec[0]);
  w2 = ceres::abs(wvec[1]);

  // direc = a = Q.col(0) * w1
  // b = Q.col(1) * w2
  d[0] = rotmat[0];
  d[1] = rotmat[3];
  d[2] = rotmat[6];
  T b_norm = w2 / (w1 + EPS);
  m[0] = rotmat[1] * b_norm;
  m[1] = rotmat[4] * b_norm;
  m[2] = rotmat[7] * b_norm;
}

template <typename T>
void PluckerToMatrix(const T d[3], const T m[3], T L[4 * 4]) {
  // Plucker matrix from geometric form
  // [LINK] https://en.wikipedia.org/wiki/Pl%C3%BCcker_matrix
  // Refer to "base/infinite_line.h"
  L[0] = T(0.0);
  L[1] = -m[2];
  L[2] = m[1];
  L[3] = d[0];
  L[4] = m[2];
  L[5] = T(0.0);
  L[6] = -m[0];
  L[7] = d[1];
  L[8] = -m[1];
  L[9] = m[0];
  L[10] = T(0.0);
  L[11] = d[2];
  L[12] = -d[0];
  L[13] = -d[1];
  L[14] = -d[2];
  L[15] = T(0.0);
}

template <typename T>
bool Ceres_IntersectLineCoordinates(const T coor1[3], const T coor2[3],
                                    T xy[2]) {
  T p_homo[3];
  ceres::CrossProduct(coor1, coor2, p_homo);
  T norm = ceres::sqrt(p_homo[0] * p_homo[0] + p_homo[1] * p_homo[1] +
                       p_homo[2] * p_homo[2]);
  p_homo[0] /= norm;
  p_homo[1] /= norm;
  p_homo[2] /= norm;
  T eps(EPS);
  if (ceres::abs(p_homo[2]) < eps)
    return false;
  else {
    xy[0] = p_homo[0] / p_homo[2];
    xy[1] = p_homo[1] / p_homo[2];
  }
  return true;
}

} // namespace limap

#endif
