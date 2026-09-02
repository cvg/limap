#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "limap/geometry/ceres_point_functions.h"

namespace limap {

// [Note] All the kvec here is a 4-dim array: [fx, fy, cx, cy]

template <typename T>
void MinimalPluckerToPlucker(const T params[6], T d[3], T m[3]) {
  // [LINK]
  // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
  // [LINK] https://hal.archives-ouvertes.fr/hal-00092589/document
  // Map quaternion in Eigen's xyzw layout
  Eigen::Map<const Eigen::Quaternion<T>> u_xyzw(params);

  // Rotation matrix
  const Eigen::Matrix<T, 3, 3> R = u_xyzw.toRotationMatrix();

  // SO(2) weights — no abs() needed because construction guarantees w1 > 0
  // and SphereManifold preserves this during optimization.
  const T w1 = params[4];
  const T w2 = params[5];

  // d = Q.col(0)
  d[0] = R(0, 0);
  d[1] = R(1, 0);
  d[2] = R(2, 0);

  // m ∥ Q.col(1), scale = w2 / w1
  const T b_norm = w2 / w1;
  m[0] = R(0, 1) * b_norm;
  m[1] = R(1, 1) * b_norm;
  m[2] = R(2, 1) * b_norm;
}

template <typename T>
void PluckerToMatrix(const T d[3], const T m[3], T L[16]) {
  const T mx = m[0], my = m[1], mz = m[2];
  const T dx = d[0], dy = d[1], dz = d[2];

  Eigen::Map<Eigen::Matrix<T, 4, 4, Eigen::RowMajor>> M(L);
  M << T(0), -mz, my, dx, mz, T(0), -mx, dy, -my, mx, T(0), dz, -dx, -dy, -dz,
      T(0);
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
  if (ceres::abs(p_homo[2]) < T(1e-12))
    return false;
  else {
    xy[0] = p_homo[0] / p_homo[2];
    xy[1] = p_homo[1] / p_homo[2];
  }
  return true;
}

template <typename T>
void Line_ImgFromCam(const T *kvec, const T *mvec, T *coor) {
  // [m]x
  Eigen::Matrix<T, 3, 3> mskew;
  mskew << T(0), -mvec[2], mvec[1], mvec[2], T(0), -mvec[0], -mvec[1], mvec[0],
      T(0);

  // K
  Eigen::Matrix<T, 3, 3> K;
  K << kvec[0], T(0), kvec[2], T(0), kvec[1], kvec[3], T(0), T(0), T(1);

  // K * [m]x * K^T
  const Eigen::Matrix<T, 3, 3> coor_skew = K * mskew * K.transpose();

  // vee(coor_skew): (2,1), (0,2), (1,0)
  Eigen::Map<Eigen::Matrix<T, 3, 1>> l(coor);
  l << coor_skew(2, 1), coor_skew(0, 2), coor_skew(1, 0);

  // normalize
  T n = l.norm();
  if (n > T(0)) {
    l /= n;
  }
}

template <typename T>
void Line_WorldToPixel(const T *kvec, const T *qvec, const T *tvec,
                       const T *dvec, const T *mvec, T *coor) {
  // Transform Plucker line (d, m) from world to camera frame:
  //   m_cam = R*m + t × (R*d)

  Eigen::Map<const Eigen::Quaternion<T>> q(qvec); // xyzw
  const Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> d(dvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> m(mvec);

  const Eigen::Matrix<T, 3, 1> Rd = R * d;
  const Eigen::Matrix<T, 3, 1> m_cam = R * m + t.cross(Rd);

  T mvec_transformed[3] = {m_cam(0), m_cam(1), m_cam(2)};
  Line_ImgFromCam<T>(kvec, mvec_transformed, coor);
}

// line projection (camera frame) in normal coordinate
template <typename T>
void Line_ImgFromCam_Regular(const T *kvec, const T *p3d, const T *dir3d,
                             T *p2d, T *dir2d) {
  T u = p3d[0] / p3d[2];
  T v = p3d[1] / p3d[2];
  ImgFromCam<T>(kvec, u, v, &p2d[0], &p2d[1]);

  // compute vanishing point
  T u_vp = dir3d[0] / dir3d[2];
  T v_vp = dir3d[1] / dir3d[2];
  T xy_vp[2];
  ImgFromCam<T>(kvec, u_vp, v_vp, &xy_vp[0], &xy_vp[1]);

  // compute 2d direction
  dir2d[0] = xy_vp[0] - p2d[0];
  dir2d[1] = xy_vp[1] - p2d[1];
  T norm = ceres::sqrt(dir2d[0] * dir2d[0] + dir2d[1] * dir2d[1]);
  dir2d[0] /= norm;
  dir2d[1] /= norm;
}

// line projection in normal coordinate
template <typename T>
void Line_WorldToPixel_Regular(const T *kvec, const T *qvec, const T *tvec,
                               const T *p3d, const T *dir3d, T *p2d, T *dir2d) {
  // Map inputs directly (quaternion stored as xyzw)
  Eigen::Map<const Eigen::Quaternion<T>> q(qvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pw(p3d);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> Dw(dir3d);

  // Rotate and translate (no normalization needed — handled by manifold)
  Eigen::Matrix<T, 3, 1> Pc = q * Pw + t; // world → camera
  Eigen::Matrix<T, 3, 1> Dc = q * Dw;     // direction → camera

  // Pass to projection
  T p_projection[3] = {Pc(0), Pc(1), Pc(2)};
  T dir_projection[3] = {Dc(0), Dc(1), Dc(2)};

  Line_ImgFromCam_Regular<T>(kvec, p_projection, dir_projection, p2d, dir2d);
}

// get direction from vp
template <typename T>
void GetDirectionFromVP(const T vp[3], const T kvec[4], T direc[3]) {
  direc[0] = vp[0] / kvec[0] - kvec[2] / kvec[0] * vp[2];
  direc[1] = vp[1] / kvec[1] - kvec[3] / kvec[1] * vp[2];
  direc[2] = vp[2];
  T norm = ceres::sqrt(direc[0] * direc[0] + direc[1] * direc[1] +
                       direc[2] * direc[2]);
  direc[0] /= norm;
  direc[1] /= norm;
  direc[2] /= norm;
}

// epipolar geometry
template <typename T>
void GetEpipolarLineCoordinate(const T kvec_ref[4], const T qvec_ref[4],
                               const T tvec_ref[3], const T kvec_tgt[4],
                               const T qvec_tgt[4], const T tvec_tgt[3],
                               const T xy[2], T epiline_coord[3]) {
  // Map poses (xyzw)
  Eigen::Map<const Eigen::Quaternion<T>> q_ref(qvec_ref);
  Eigen::Map<const Eigen::Quaternion<T>> q_tgt(qvec_tgt);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> Tref(tvec_ref);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> Ttgt(tvec_tgt);

  // Relative pose: R = R_tgt * R_ref^T, t = Ttgt - R * Tref
  const Eigen::Matrix<T, 3, 3> Rref = q_ref.toRotationMatrix();
  const Eigen::Matrix<T, 3, 3> Rtgt = q_tgt.toRotationMatrix();
  const Eigen::Matrix<T, 3, 3> R = Rtgt * Rref.transpose();
  const Eigen::Matrix<T, 3, 1> t = Ttgt - R * Tref;

  // Essential: E = [t]_x R
  Eigen::Matrix<T, 3, 3> tskew;
  tskew << T(0), -t(2), t(1), t(2), T(0), -t(0), -t(1), t(0), T(0);
  const Eigen::Matrix<T, 3, 3> E = tskew * R;

  // K^{-1} for ref and tgt
  Eigen::Matrix<T, 3, 3> Kinv_ref;
  Kinv_ref.setZero();
  Kinv_ref(0, 0) = T(1) / kvec_ref[0];
  Kinv_ref(1, 1) = T(1) / kvec_ref[1];
  Kinv_ref(0, 2) = -kvec_ref[2] / kvec_ref[0];
  Kinv_ref(1, 2) = -kvec_ref[3] / kvec_ref[1];
  Kinv_ref(2, 2) = T(1);

  Eigen::Matrix<T, 3, 3> Kinv_tgt;
  Kinv_tgt.setZero();
  Kinv_tgt(0, 0) = T(1) / kvec_tgt[0];
  Kinv_tgt(1, 1) = T(1) / kvec_tgt[1];
  Kinv_tgt(0, 2) = -kvec_tgt[2] / kvec_tgt[0];
  Kinv_tgt(1, 2) = -kvec_tgt[3] / kvec_tgt[1];
  Kinv_tgt(2, 2) = T(1);

  // Fundamental: F = K_tgt^{-T} E K_ref^{-1}
  const Eigen::Matrix<T, 3, 3> F = Kinv_tgt.transpose() * E * Kinv_ref;

  // Epipolar line in target: l' = F * [u v 1]^T, then normalize
  Eigen::Matrix<T, 3, 1> p_homo;
  p_homo << xy[0], xy[1], T(1);

  Eigen::Map<Eigen::Matrix<T, 3, 1>> map(epiline_coord);
  map = (F * p_homo).normalized();
}

template <typename T>
bool Ceres_GetIntersection2dFromInfiniteLine3d(
    const T uvec[4], const T wvec[2], // MinimalLine
    const T kvec[4], const T qvec[4],
    const T tvec[3], // intrinsics and extrinsics
    const T coor[3], // InfiniteLine2d sample
    T xy[2]) {
  T dvec[3], mvec[3];
  MinimalPluckerToPlucker<T>(uvec, wvec, dvec, mvec);
  T coor_proj[3];
  Line_WorldToPixel<T>(kvec, qvec, tvec, dvec, mvec, coor_proj);
  return Ceres_IntersectLineCoordinates(coor_proj, coor, xy);
}

} // namespace limap
