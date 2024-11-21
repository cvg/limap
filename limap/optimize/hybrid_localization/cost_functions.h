#ifndef LIMAP_OPTIMIZE_HYBRID_LOCALIZATION_COST_FUNCTIONS_H_
#define LIMAP_OPTIMIZE_HYBRID_LOCALIZATION_COST_FUNCTIONS_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "base/camera.h"
#include "base/linebase.h"
#include "util/types.h"

#include "base/line_dists.h"
#include "ceresbase/line_dists.h"
#include "ceresbase/point_projection.h"
#include <ceres/ceres.h>

#include "optimize/hybrid_localization/hybrid_localization_config.h"

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace hybrid_localization {

////////////////////////////////////////////////////////////
// 2D Weights and Line Dists
////////////////////////////////////////////////////////////
template <typename T>
T Ceres_Compute2DWeight(const LineLocCostFunctionWeight &weight_type,
                        const T dir2d[2], const T p1[2], const T p2[2],
                        const double alpha = 10) {
  const T alpha_t = T(alpha);
  T direc[2];
  direc[0] = p2[0] - p1[0];
  direc[1] = p2[1] - p1[1];

  T weight;
  switch (weight_type) {
  case ENoneWeight:
    weight = T(1.0);
    break;
  case ECosineWeight:
    weight = ceres::exp(alpha_t *
                        (T(1.0) - CeresComputeDist2D_cosine(dir2d, direc)));
    break;
  case ELine3dppWeight:
    weight = ceres::exp(alpha_t *
                        ceres::acos(CeresComputeDist2D_cosine(dir2d, direc)));
    break;
  case ELengthWeight:
    weight = ceres::sqrt(direc[0] * direc[0] + direc[1] * direc[1] + 1e-8);
    break;
  case EInvLengthWeight:
    weight =
        1.0 / ceres::sqrt(direc[0] * direc[0] + direc[1] * direc[1] + 1e-8);
    break;
  default:
    throw std::runtime_error("Unsupported 2D Line Cost Function Weight!");
  }
  return weight;
}

template <typename T>
void Ceres_2DMidpointDist2(const T s2d1[2], const T e2d1[2], const T s2d2[2],
                           const T e2d2[2], T *res) {
  T m1[2] = {0.5 * (s2d1[0] + e2d1[0]), 0.5 * (s2d1[1] + e2d1[1])};
  T m2[2] = {0.5 * (s2d2[0] + e2d2[0]), 0.5 * (s2d2[1] + e2d2[1])};
  T md[2] = {m1[0] - m2[0], m1[1] - m2[1]};
  res[0] = md[0];
  res[1] = md[1];
}

// Only use this for pose refinement, not compatible with RANSAC scoring
template <typename T>
void Ceres_2DMidpointAngleDist3(const T s2d1[2], const T e2d1[2],
                                const T s2d2[2], const T e2d2[2], T *res) {
  Ceres_2DMidpointDist2(s2d1, e2d1, s2d2, e2d2, res);

  T dir1[2] = {e2d1[0] - s2d1[0], e2d1[1] - s2d1[1]};
  T norm1 = ceres::sqrt(dir1[0] * dir1[0] + dir1[1] * dir1[1] + 1e-8);
  dir1[0] /= norm1;
  dir1[1] /= norm1;

  T dir2[2] = {e2d2[0] - s2d2[0], e2d2[1] - s2d2[1]};
  T norm2 = ceres::sqrt(dir2[0] * dir2[0] + dir2[1] * dir2[1] + 1e-8);
  dir2[0] /= norm2;
  dir2[1] /= norm2;

  res[2] = norm1 * CeresComputeDist2D_sine(dir1, dir2);
}

template <typename T>
void Ceres_2DPerpendicularDist4(const T p2d[2], const T dir2d[2], const T p1[2],
                                const T p2[2], T *res) {
  T disp1[2], disp2[2];
  disp1[0] = p1[0] - p2d[0];
  disp1[1] = p1[1] - p2d[1];
  disp2[0] = p2[0] - p2d[0];
  disp2[1] = p2[1] - p2d[1];
  T sine1 = CeresComputeDist2D_sine(dir2d, disp1);
  T sine2 = CeresComputeDist2D_sine(dir2d, disp2);
  res[0] = disp1[0] * sine1;
  res[1] = disp1[1] * sine1;
  res[2] = disp2[0] * sine2;
  res[3] = disp2[1] * sine2;
}

template <typename T>
void Ceres_2DPerpendicularDist2(const T p2d[2], const T dir2d[2], const T p1[2],
                                const T p2[2], T *res) {
  T r[4];
  Ceres_2DPerpendicularDist4(p2d, dir2d, p1, p2, r);

  res[0] = ceres::sqrt(r[0] * r[0] + r[1] * r[1] + 1e-8);
  res[1] = ceres::sqrt(r[2] * r[2] + r[3] * r[3] + 1e-8);
}

////////////////////////////////////////////////////////////
// 3D Line Dists
////////////////////////////////////////////////////////////
template <typename T>
T Ceres_3DLineLineDist(const T *kvec, const T *qvec, const T *tvec,
                       const T p3d[3], const T dir3d[3], const T p2d[2]) {
  T res;
  T pa[3], pb[3];
  T depth[2] = {T(1.0), T(2.0)};
  PixelToWorld(kvec, qvec, tvec, p2d[0], p2d[1], &depth[0], pa);
  PixelToWorld(kvec, qvec, tvec, p2d[0], p2d[1], &depth[1], pb);

  T dir[3];
  for (int i = 0; i < 3; i++) {
    dir[i] = pb[i] - pa[i];
  }

  T n[3];
  ceres::CrossProduct(dir, dir3d, n);

  T n_squared_norm;
  n_squared_norm = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
  T d[3], cross[3];
  for (int i = 0; i < 3; i++) {
    d[i] = p3d[i] - pa[i];
  }

  if (n_squared_norm <= 1e-8) {
    ceres::CrossProduct(dir, d, cross);
    res = ceres::sqrt(
        ceres::DotProduct(cross, cross) / ceres::DotProduct(dir, dir) + 1e-8);
  } else {
    T dot = ceres::DotProduct(n, d);
    res = dot / ceres::sqrt(n_squared_norm);
  }
  return res;
}

template <typename T>
void Ceres_3DLineLineDist2(const T *kvec, const T *qvec, const T *tvec,
                           const T p3d[3], const T dir3d[3], const T p1[2],
                           const T p2[2], T *res) {
  res[0] = Ceres_3DLineLineDist(kvec, qvec, tvec, p3d, dir3d, p1);
  res[1] = Ceres_3DLineLineDist(kvec, qvec, tvec, p3d, dir3d, p2);
}

template <typename T>
void Ceres_3DPlaneLineDist2(const T *kvec, const T *qvec, const T *tvec,
                            const T s3d[3], const T e3d[3], const T p1[2],
                            const T p2[2], T *res) {
  T p1a[3], p1b[3], p2a[3], p2b[3];
  T depth[2] = {T(1.0), T(2.0)};
  PixelToWorld(kvec, qvec, tvec, p1[0], p1[1], &depth[0], p1a);
  PixelToWorld(kvec, qvec, tvec, p1[0], p1[1], &depth[1], p1b);
  PixelToWorld(kvec, qvec, tvec, p2[0], p2[1], &depth[0], p2a);
  PixelToWorld(kvec, qvec, tvec, p2[0], p2[1], &depth[1], p2b);

  T dir1[3], dir2[3];
  for (int i = 0; i < 3; i++) {
    dir1[i] = p1b[i] - p1a[i];
    dir2[i] = p2b[i] - p2a[i];
  }

  T n[3];
  ceres::CrossProduct(dir1, dir2, n);

  T n_squared_norm;
  n_squared_norm = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
  T ds[3], de[3];
  for (int i = 0; i < 3; i++) {
    ds[i] = s3d[i] - p1a[i];
    de[i] = e3d[i] - p1a[i];
  }
  res[0] = ceres::DotProduct(n, ds) / ceres::sqrt(n_squared_norm);
  res[1] = ceres::DotProduct(n, de) / ceres::sqrt(n_squared_norm);
}

static const int getResidualNum(const LineLocCostFunction &costfunc) {
  if (costfunc == E2DPerpendicularDist4)
    return 4;
  if (costfunc == E2DMidpointAngleDist3)
    return 3;
  return 2;
}

struct ReprojectionLineFunctor {
public:
  ReprojectionLineFunctor(const LineLocCostFunction &costfunc,
                          const LineLocCostFunctionWeight &weight,
                          const Line3d &line3d, const Line2d &line2d,
                          const double alpha = 10.0)
      : costfunc_(costfunc), weight_(weight), line3d_(line3d), line2d_(line2d),
        alpha_(alpha) {}

  static ceres::CostFunction *Create(const LineLocCostFunction &costfunc,
                                     const LineLocCostFunctionWeight &weight,
                                     const Line3d &line3d, const Line2d &line2d,
                                     const double alpha = 10.0) {
    const int num_res = getResidualNum(costfunc);
    if (num_res == 2)
      return new ceres::AutoDiffCostFunction<ReprojectionLineFunctor, 2, 4, 4,
                                             3>(
          new ReprojectionLineFunctor(costfunc, weight, line3d, line2d, alpha));
    else if (num_res == 3)
      return new ceres::AutoDiffCostFunction<ReprojectionLineFunctor, 3, 4, 4,
                                             3>(
          new ReprojectionLineFunctor(costfunc, weight, line3d, line2d, alpha));
    else if (num_res == 4)
      return new ceres::AutoDiffCostFunction<ReprojectionLineFunctor, 4, 4, 4,
                                             3>(
          new ReprojectionLineFunctor(costfunc, weight, line3d, line2d, alpha));
    else
      return new ceres::AutoDiffCostFunction<ReprojectionLineFunctor, 1, 4, 4,
                                             3>(
          new ReprojectionLineFunctor(costfunc, weight, line3d, line2d, alpha));
  }

  template <typename T>
  bool operator()(const T *const kvec, const T *const qvec, const T *const tvec,
                  T *residuals) const {
    // reproject to 2d
    T s3d[3] = {T(line3d_.start(0)), T(line3d_.start(1)), T(line3d_.start(2))};
    T e3d[3] = {T(line3d_.end(0)), T(line3d_.end(1)), T(line3d_.end(2))};

    T dir3d[3] = {e3d[0] - s3d[0], e3d[1] - s3d[1], e3d[2] - s3d[2]};
    T dir3d_norm = ceres::sqrt(ceres::DotProduct(dir3d, dir3d) + 1e-8);
    dir3d[0] /= dir3d_norm;
    dir3d[1] /= dir3d_norm;
    dir3d[2] /= dir3d_norm;

    T s2d_reproj[2], e2d_reproj[2];
    WorldToPixel<T>(kvec, qvec, tvec, s3d, s2d_reproj);
    WorldToPixel<T>(kvec, qvec, tvec, e3d, e2d_reproj);

    T s2d[2] = {T(line2d_.start(0)), T(line2d_.start(1))};
    T e2d[2] = {T(line2d_.end(0)), T(line2d_.end(1))};

    T dir2d[2] = {e2d_reproj[0] - s2d_reproj[0], e2d_reproj[1] - s2d_reproj[1]};
    T dir_norm = ceres::sqrt(dir2d[0] * dir2d[0] + dir2d[1] * dir2d[1] + 1e-8);
    dir2d[0] /= dir_norm;
    dir2d[1] /= dir_norm;

    switch (costfunc_) {
    case E2DMidpointDist2:
      Ceres_2DMidpointDist2(s2d_reproj, e2d_reproj, s2d, e2d, residuals);
      break;
    case E2DMidpointAngleDist3:
      Ceres_2DMidpointAngleDist3(s2d_reproj, e2d_reproj, s2d, e2d, residuals);
      break;
    case E2DPerpendicularDist2:
      Ceres_2DPerpendicularDist2(s2d_reproj, dir2d, s2d, e2d, residuals);
      break;
    case E2DPerpendicularDist4:
      Ceres_2DPerpendicularDist4(s2d_reproj, dir2d, s2d, e2d, residuals);
      break;
    case E3DLineLineDist2:
      Ceres_3DLineLineDist2(kvec, qvec, tvec, s3d, dir3d, s2d, e2d, residuals);
      break;
    case E3DPlaneLineDist2:
      Ceres_3DPlaneLineDist2(kvec, qvec, tvec, s3d, e3d, s2d, e2d, residuals);
      break;
    }

    T weight = Ceres_Compute2DWeight(weight_, dir2d, s2d, e2d);
    for (int i = 0; i < getResidualNum(costfunc_); i++)
      residuals[i] *= weight;
    return true;
  }

protected:
  LineLocCostFunction costfunc_;
  LineLocCostFunctionWeight weight_;
  Line3d line3d_;
  Line2d line2d_;
  double alpha_; // for weighting angle
};

struct ReprojectionPointFunctor {
public:
  ReprojectionPointFunctor(const V3D &p3d, const V2D &p2d,
                           bool use_3d_dist = false)
      : p3d_(p3d), p2d_(p2d), use_3d_dist_(use_3d_dist) {}

  static ceres::CostFunction *Create(const V3D &p3d, const V2D &p2d,
                                     bool use_3d_dist = false) {
    return new ceres::AutoDiffCostFunction<ReprojectionPointFunctor, 2, 4, 4,
                                           3>(
        new ReprojectionPointFunctor(p3d, p2d, use_3d_dist));
  }

  template <typename T>
  bool operator()(const T *const kvec, const T *const qvec, const T *const tvec,
                  T *residuals) const {
    T p2d[2] = {T(p2d_[0]), T(p2d_[1])};
    T p3d[3] = {T(p3d_[0]), T(p3d_[1]), T(p3d_[2])};
    if (use_3d_dist_) {
      T pa[3], pb[3];
      T depth[2] = {T(1.0), T(2.0)};
      PixelToWorld(kvec, qvec, tvec, p2d[0], p2d[1], &depth[0], pa);
      PixelToWorld(kvec, qvec, tvec, p2d[0], p2d[1], &depth[1], pb);

      T dir[3];
      for (int i = 0; i < 3; i++)
        dir[i] = pb[i] - pa[i];
      T dir_norm = ceres::sqrt(ceres::DotProduct(dir, dir) + 1e-8);
      dir[0] /= dir_norm;
      dir[1] /= dir_norm;
      dir[2] /= dir_norm;

      T d_p3dpa[3] = {p3d[0] - pa[0], p3d[1] - pa[1], p3d[2] - pa[2]};
      residuals[0] =
          ceres::sqrt(ceres::DotProduct(d_p3dpa, d_p3dpa) -
                      ceres::pow(ceres::DotProduct(d_p3dpa, dir), 2) + 1e-8);
      residuals[1] = T(0);
    } else {
      T p2d_reproj[2];
      WorldToPixel<T>(kvec, qvec, tvec, p3d, p2d_reproj);

      residuals[0] = p2d_reproj[0] - p2d_[0];
      residuals[1] = p2d_reproj[1] - p2d_[1];
    }
    return true;
  }

protected:
  V3D p3d_;
  V2D p2d_;
  bool use_3d_dist_;
};

} // namespace hybrid_localization

} // namespace optimize

} // namespace limap

#endif
