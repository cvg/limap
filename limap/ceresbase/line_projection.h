#ifndef LIMAP_CERESBASE_LINE_PROJECTION_H
#define LIMAP_CERESBASE_LINE_PROJECTION_H

#include "ceresbase/line_transforms.h"
#include "ceresbase/point_projection.h"
#include "util/types.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// [Note] All the kvec here is a 4-dim array: [fx, fy, cx, cy]

namespace limap {

// line projection (camera frame) in plucker coordinate
template <typename T>
void Line_ImgFromCam(const T *kvec, const T *mvec, T *coor) {
  // K * [m]x * K.transpose()
  Eigen::Matrix<T, 3, 3> mskew;
  mskew(0, 0) = T(0.0);
  mskew(0, 1) = -mvec[2];
  mskew(0, 2) = mvec[1];
  mskew(1, 0) = mvec[2];
  mskew(1, 1) = T(0.0);
  mskew(1, 2) = -mvec[0];
  mskew(2, 0) = -mvec[1];
  mskew(2, 1) = mvec[0];
  mskew(2, 2) = T(0.0);

  Eigen::Matrix<T, 3, 3> K;
  K(0, 0) = kvec[0];
  K(0, 1) = T(0.0);
  K(0, 2) = kvec[2];
  K(1, 0) = T(0.0);
  K(1, 1) = kvec[1];
  K(1, 2) = kvec[3];
  K(2, 0) = T(0.0);
  K(2, 1) = T(0.0);
  K(2, 2) = T(1.0);
  Eigen::Matrix<T, 3, 3> coor_skew = K * mskew * K.transpose();

  coor[0] = coor_skew(2, 1);
  coor[1] = coor_skew(0, 2);
  coor[2] = coor_skew(1, 0);
  T coor_norm = ceres::sqrt(coor[0] * coor[0] + coor[1] * coor[1] +
                            coor[2] * coor[2] + EPS);
  coor[0] /= coor_norm;
  coor[1] /= coor_norm;
  coor[2] /= coor_norm;
}

template <typename T>
void Line_WorldToPixel(const T *kvec, const T *qvec, const T *tvec,
                       const T *dvec, const T *mvec, T *coor) {
  // R * [m]x * R.transpose() - t * (R * d).transpose() + (R * d) *
  // t.transpose();
  T R_pt[3 * 3];
  ceres::QuaternionToRotation(qvec, R_pt);
  Eigen::Map<Eigen::Matrix<T, 3, 3, Eigen::RowMajor>> R(R_pt);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tvec);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> d(dvec);

  Eigen::Matrix<T, 3, 3> mskew;
  mskew(0, 0) = T(0.0);
  mskew(0, 1) = -mvec[2];
  mskew(0, 2) = mvec[1];
  mskew(1, 0) = mvec[2];
  mskew(1, 1) = T(0.0);
  mskew(1, 2) = -mvec[0];
  mskew(2, 0) = -mvec[1];
  mskew(2, 1) = mvec[0];
  mskew(2, 2) = T(0.0);

  Eigen::Matrix<T, 3, 3> mskew_transformed = R * mskew * R.transpose() -
                                             t * (R * d).transpose() +
                                             (R * d) * t.transpose();
  T mvec_transformed[3];
  mvec_transformed[0] = mskew_transformed(2, 1);
  mvec_transformed[1] = mskew_transformed(0, 2);
  mvec_transformed[2] = mskew_transformed(1, 0);
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
  // transform point
  T p_projection[3];
  ceres::QuaternionRotatePoint(qvec, p3d, p_projection);
  p_projection[0] += tvec[0];
  p_projection[1] += tvec[1];
  p_projection[2] += tvec[2];

  // transform direction
  T dir_projection[3];
  ceres::QuaternionRotatePoint(qvec, dir3d, dir_projection);

  // world to image
  Line_ImgFromCam_Regular<T>(kvec, p_projection, dir_projection, p2d, dir2d);
}

// get direction from vp
template <typename T>
void GetDirectionFromVP(const T vp[3], const T kvec[4], T direc[3]) {
  direc[0] = vp[0] / kvec[0] - kvec[2] / kvec[0] * vp[2];
  direc[1] = vp[1] / kvec[1] - kvec[3] / kvec[1] * vp[2];
  direc[2] = vp[2];
  T norm = ceres::sqrt(direc[0] * direc[0] + direc[1] * direc[1] +
                       direc[2] * direc[2] + EPS);
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
  T Rref_pt[3 * 3], Rtgt_pt[3 * 3];
  ceres::QuaternionToRotation(qvec_ref, Rref_pt);
  Eigen::Map<Eigen::Matrix<T, 3, 3, Eigen::RowMajor>> Rref(Rref_pt);
  ceres::QuaternionToRotation(qvec_tgt, Rtgt_pt);
  Eigen::Map<Eigen::Matrix<T, 3, 3, Eigen::RowMajor>> Rtgt(Rtgt_pt);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> Tref(tvec_ref);
  Eigen::Map<const Eigen::Matrix<T, 3, 1>> Ttgt(tvec_tgt);

  // compose relative pose
  T relR_pt[9], relT_pt[3];
  Eigen::Map<Eigen::Matrix<T, 3, 3>> relR(relR_pt);
  Eigen::Map<Eigen::Matrix<T, 3, 1>> relT(relT_pt);
  relR = Rtgt * Rref.transpose();
  relT = Ttgt - relR * Tref;

  // compose essential matrix
  Eigen::Matrix<T, 3, 3> tskew;
  tskew(0, 0) = T(0.0);
  tskew(0, 1) = -relT[2];
  tskew(0, 2) = relT[1];
  tskew(1, 0) = relT[2];
  tskew(1, 1) = T(0.0);
  tskew(1, 2) = -relT[0];
  tskew(2, 0) = -relT[1];
  tskew(2, 1) = relT[0];
  tskew(2, 2) = T(0.0);
  Eigen::Matrix<T, 3, 3> E = tskew * relR;

  // compose fundamental matrix
  Eigen::Matrix<T, 3, 3> Kinv_ref;
  Kinv_ref(0, 0) = T(1.0) / kvec_ref[0];
  Kinv_ref(0, 1) = T(0.0);
  Kinv_ref(0, 2) = -kvec_ref[2] / kvec_ref[0];
  Kinv_ref(1, 0) = T(0.0);
  Kinv_ref(1, 1) = T(1.0) / kvec_ref[1];
  Kinv_ref(1, 2) = -kvec_ref[3] / kvec_ref[1];
  Kinv_ref(2, 0) = T(0.0);
  Kinv_ref(2, 1) = T(0.0);
  Kinv_ref(2, 2) = T(1.0);
  Eigen::Matrix<T, 3, 3> Kinv_tgt;
  Kinv_tgt(0, 0) = T(1.0) / kvec_tgt[0];
  Kinv_tgt(0, 1) = T(0.0);
  Kinv_tgt(0, 2) = -kvec_tgt[2] / kvec_tgt[0];
  Kinv_tgt(1, 0) = T(0.0);
  Kinv_tgt(1, 1) = T(1.0) / kvec_tgt[1];
  Kinv_tgt(1, 2) = -kvec_tgt[3] / kvec_tgt[1];
  Kinv_tgt(2, 0) = T(0.0);
  Kinv_tgt(2, 1) = T(0.0);
  Kinv_tgt(2, 2) = T(1.0);
  Eigen::Matrix<T, 3, 3> F = Kinv_tgt.transpose() * E * Kinv_ref;

  // compute epipolar line
  Eigen::Matrix<T, 3, 1> p_homo;
  p_homo(0) = xy[0];
  p_homo(1) = xy[1];
  p_homo(2) = T(1.0);
  Eigen::Map<Eigen::Matrix<T, 3, 1>> map(epiline_coord);
  map = (F * p_homo).normalized();
}

template <typename T>
bool Ceres_GetIntersection2dFromInfiniteLine3d(
    const T uvec[4], const T wvec[2],                  // MinimalLine
    const T kvec[4], const T qvec[4], const T tvec[3], // CameraView
    const T coor[3],                                   // InfiniteLine2d sample
    T xy[2]) {
  T dvec[3], mvec[3];
  MinimalPluckerToPlucker<T>(uvec, wvec, dvec, mvec);
  T coor_proj[3];
  Line_WorldToPixel<T>(kvec, qvec, tvec, dvec, mvec, coor_proj);
  return Ceres_IntersectLineCoordinates(coor_proj, coor, xy);
}

} // namespace limap

#endif
