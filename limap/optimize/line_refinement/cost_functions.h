#ifndef LIMAP_OPTIMIZE_LINE_REFINEMENT_COST_FUNCTIONS_H_
#define LIMAP_OPTIMIZE_LINE_REFINEMENT_COST_FUNCTIONS_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <ceres/ceres.h>
#include <colmap/sensor/models.h>

#include "base/camera_models.h"
#include "base/infinite_line.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "util/types.h"

#include "ceresbase/line_dists.h"
#include "ceresbase/line_projection.h"
#include "ceresbase/line_transforms.h"

#ifdef INTERPOLATION_ENABLED
#include "optimize/line_refinement/pixel_cost_functions.h"
#endif // INTERPOLATION_ENABLED

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace line_refinement {

////////////////////////////////////////////////////////////
// VP Constraints
////////////////////////////////////////////////////////////
template <typename CameraModel> struct VPConstraintsFunctor {
public:
  VPConstraintsFunctor(const V3D &VP, const double *params = NULL,
                       const double *qvec = NULL)
      : VP_(VP), params_(params), qvec_(qvec) {}

  static ceres::CostFunction *Create(const V3D &VP, const double *params = NULL,
                                     const double *qvec = NULL) {
    if (!params && !qvec)
      return new ceres::AutoDiffCostFunction<VPConstraintsFunctor, 1, 4, 2,
                                             CameraModel::num_params, 4>(
          new VPConstraintsFunctor(VP, NULL, NULL));
    else
      return new ceres::AutoDiffCostFunction<VPConstraintsFunctor, 1, 4, 2>(
          new VPConstraintsFunctor(VP, params, qvec));
  }

  template <typename T>
  bool operator()(const T *const uvec, const T *const wvec,
                  T *residuals) const {
    CHECK_NOTNULL(params_);
    CHECK_NOTNULL(qvec_);

    const int num_params = CameraModel::num_params;
    T params[num_params];
    for (size_t i = 0; i < num_params; ++i) {
      params[i] = T(params_[i]);
    }
    T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
    return (*this)(uvec, wvec, params, qvec, residuals);
  }

  template <typename T>
  bool operator()(const T *const uvec, const T *const wvec,
                  const T *const params, const T *const qvec,
                  T *residuals) const {
    T kvec[4];
    ParamsToKvec(CameraModel::model_id, params, kvec);
    T dir3d[3], m[3];
    MinimalPluckerToPlucker<T>(uvec, wvec, dir3d, m);
    T dir3d_rotated[3];
    ceres::QuaternionRotatePoint(qvec, dir3d, dir3d_rotated);

    const V3D &vp = VP_;
    T vpvec[3] = {T(vp[0]), T(vp[1]), T(vp[2])};
    T direc[3];
    GetDirectionFromVP<T>(vpvec, kvec, direc);
    residuals[0] = CeresComputeDist3D_sine(dir3d_rotated, direc);
    return true;
  }

protected:
  V3D VP_;
  const double *params_;
  const double *qvec_;
};

////////////////////////////////////////////////////////////
// Geometric Refinement
////////////////////////////////////////////////////////////

template <typename T>
void Ceres_PerpendicularDist2D(const T coor[3], const T p1[2], const T p2[2],
                               T *res) {
  T direc_norm = ceres::sqrt(coor[0] * coor[0] + coor[1] * coor[1] + EPS);
  T dist1 = (p1[0] * coor[0] + p1[1] * coor[1] + coor[2]) / direc_norm;
  T dist2 = (p2[0] * coor[0] + p2[1] * coor[1] + coor[2]) / direc_norm;
  res[0] = dist1;
  res[1] = dist2;
}

template <typename T>
void Ceres_CosineWeightedPerpendicularDist2D_1D(const T coor[3], const T p1[2],
                                                const T p2[2], T *res,
                                                const double alpha = 10.0) {
  T direc_norm = ceres::sqrt(coor[0] * coor[0] + coor[1] * coor[1] + EPS);
  // 2D direction of the projection
  T dir2d[2];
  dir2d[0] = -coor[1] / direc_norm;
  dir2d[1] = coor[0] / direc_norm;
  // 2D direction of the 2D line segment
  T direc[2];
  direc[0] = p2[0] - p1[0];
  direc[1] = p2[1] - p1[1];
  // compute weight
  T cosine = CeresComputeDist2D_cosine(dir2d, direc);
  const T alpha_t = T(alpha);
  T weight = ceres::exp(alpha_t * (T(1.0) - cosine));
  // compute raw distance and multiply it by the weight
  Ceres_PerpendicularDist2D(coor, p1, p2, res);
  res[0] *= weight;
  res[1] *= weight;
}

template <typename CameraModel> struct GeometricRefinementFunctor {
public:
  GeometricRefinementFunctor(const Line2d &line2d, const double *params = NULL,
                             const double *qvec = NULL,
                             const double *tvec = NULL,
                             const double alpha = 10.0)
      : line2d_(line2d), params_(params), qvec_(qvec), tvec_(tvec),
        alpha_(alpha) {}

  static ceres::CostFunction *Create(const Line2d &line2d,
                                     const double *params = NULL,
                                     const double *qvec = NULL,
                                     const double *tvec = NULL,
                                     const double alpha = 10.0) {
    if (!params && !qvec && !tvec)
      return new ceres::AutoDiffCostFunction<GeometricRefinementFunctor, 2, 4,
                                             2, CameraModel::num_params, 4, 3>(
          new GeometricRefinementFunctor(line2d, NULL, NULL, NULL, alpha));
    else
      return new ceres::AutoDiffCostFunction<GeometricRefinementFunctor, 2, 4,
                                             2>(
          new GeometricRefinementFunctor(line2d, params, qvec, tvec, alpha));
  }

  template <typename T>
  bool operator()(const T *const uvec, const T *const wvec,
                  T *residuals) const {
    CHECK_NOTNULL(params_);
    CHECK_NOTNULL(qvec_);
    CHECK_NOTNULL(tvec_);

    const int num_params = CameraModel::num_params;
    T params[num_params];
    for (size_t i = 0; i < num_params; ++i) {
      params[i] = T(params_[i]);
    }
    T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
    T tvec[3] = {T(tvec_[0]), T(tvec_[1]), T(tvec_[2])};
    return (*this)(uvec, wvec, params, qvec, tvec, residuals);
  }

  template <typename T>
  bool operator()(const T *const uvec, const T *const wvec,
                  const T *const params, const T *const qvec,
                  const T *const tvec, T *residuals) const {
    T kvec[4];
    ParamsToKvec(CameraModel::model_id, params, kvec);
    T dvec[3], mvec[3];
    MinimalPluckerToPlucker<T>(uvec, wvec, dvec, mvec);
    T coor[3];
    Line_WorldToPixel<T>(kvec, qvec, tvec, dvec, mvec, coor);

    const Line2d &line = line2d_;
    T p1[2] = {T(line.start(0)), T(line.start(1))};
    T p2[2] = {T(line.end(0)), T(line.end(1))};
    Ceres_CosineWeightedPerpendicularDist2D_1D(coor, p1, p2, residuals, alpha_);
    return true;
  }

protected:
  Line2d line2d_;
  const double *params_;
  const double *qvec_;
  const double *tvec_;
  double alpha_; // for weighting angle
};

} // namespace line_refinement

} // namespace optimize

} // namespace limap

#endif
