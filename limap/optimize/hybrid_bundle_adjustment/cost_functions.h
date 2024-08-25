#ifndef LIMAP_HYBRIDBA_COST_FUNCTIONS_H_
#define LIMAP_HYBRIDBA_COST_FUNCTIONS_H_

#include "_limap/helpers.h"
#include <ceres/ceres.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "base/camera_models.h"
#include "base/infinite_line.h"
#include "util/types.h"

#include "ceresbase/line_dists.h"
#include "ceresbase/line_projection.h"
#include "ceresbase/point_projection.h"

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace hybrid_bundle_adjustment {

////////////////////////////////////////////////////////////
// Point Geometric Refinement
////////////////////////////////////////////////////////////

template <typename CameraModel> struct PointGeometricRefinementFunctor {
public:
  PointGeometricRefinementFunctor(const V2D &p2d, const double *params = NULL,
                                  const double *qvec = NULL,
                                  const double *tvec = NULL)
      : p2d_(p2d), params_(params), qvec_(qvec), tvec_(tvec) {}

  static ceres::CostFunction *Create(const V2D &p2d,
                                     const double *params = NULL,
                                     const double *qvec = NULL,
                                     const double *tvec = NULL) {
    if (!params && !qvec && !tvec)
      return new ceres::AutoDiffCostFunction<PointGeometricRefinementFunctor, 2,
                                             3, CameraModel::num_params, 4, 3>(
          new PointGeometricRefinementFunctor(p2d, NULL, NULL, NULL));
    else
      return new ceres::AutoDiffCostFunction<PointGeometricRefinementFunctor, 2,
                                             3>(
          new PointGeometricRefinementFunctor(p2d, params, qvec, tvec));
  }

  template <typename T>
  bool operator()(const T *const point_vec, T *residuals) const {
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
    return (*this)(point_vec, params, qvec, tvec, residuals);
  }

  template <typename T>
  bool operator()(const T *const point_vec, const T *const params,
                  const T *const qvec, const T *const tvec,
                  T *residuals) const {
    T kvec[4];
    ParamsToKvec(CameraModel::model_id, params, kvec);
    T point2d[2] = {T(p2d_[0]), T(p2d_[1])};

    T proj_point[2];
    WorldToPixel(kvec, qvec, tvec, point_vec, proj_point);
    residuals[0] = proj_point[0] - point2d[0];
    residuals[1] = proj_point[1] - point2d[1];
    return true;
  }

protected:
  V2D p2d_;
  const double *params_;
  const double *qvec_;
  const double *tvec_;
};

} // namespace hybrid_bundle_adjustment

} // namespace optimize

} // namespace limap

#endif
