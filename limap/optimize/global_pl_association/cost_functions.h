#ifndef LIMAP_PL_TRIANGULATION_COST_FUNCTIONS_H_
#define LIMAP_PL_TRIANGULATION_COST_FUNCTIONS_H_

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

namespace global_pl_association {

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

////////////////////////////////////////////////////////////
// Point-line association on 3D
////////////////////////////////////////////////////////////

template <typename T>
T Ceres_PerpendicularDist3D(const T p3d[3], const T dir3d[3], const T p[3]) {
  T disp[3];
  for (size_t i = 0; i < 3; ++i) {
    disp[i] = p[i] - p3d[i];
  }
  T sine = CeresComputeDist3D_sine(dir3d, disp);
  // T dist = ceres::sqrt(disp[0] * disp[0] + disp[1] * disp[1] + disp[2] *
  // disp[2] + EPS) * sine;
  T dist = ceres::sqrt(disp[0] * disp[0] + disp[1] * disp[1] +
                       disp[2] * disp[2] + EPS);
  return dist;
}

struct PointLineAssociation3dFunctor {
public:
  PointLineAssociation3dFunctor() {}

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<PointLineAssociation3dFunctor, 3, 3,
                                           4, 2>(
        new PointLineAssociation3dFunctor());
  }

  template <typename T>
  bool operator()(const T *const point_vec, const T *const uvec,
                  const T *const wvec, T *residuals) const {
    T dir3d[3], b[3];
    MinimalPluckerToPlucker<T>(uvec, wvec, dir3d, b);

    // Reference: page 4 in the following link:
    // [LINK]:
    // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
    T b_point[3];
    ceres::CrossProduct(dir3d, point_vec, b_point);
    for (size_t i = 0; i < 3; ++i) {
      b_point[i] = b_point[i] + b[i];
    }
    T disp[3];
    ceres::CrossProduct(dir3d, b_point, disp);
    residuals[0] = disp[0] + EPS;
    residuals[1] = disp[1] + EPS;
    residuals[2] = disp[2] + EPS;
    return true;
  }
};

////////////////////////////////////////////////////////////
// VP-line association on 3D
////////////////////////////////////////////////////////////

struct VPLineAssociation3dFunctor {
public:
  VPLineAssociation3dFunctor() {}

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<VPLineAssociation3dFunctor, 1, 3, 4,
                                           2>(new VPLineAssociation3dFunctor());
  }

  template <typename T>
  bool operator()(const T *const vpvec, const T *const uvec,
                  const T *const wvec, T *residuals) const {
    T dir3d[3], m[3];
    MinimalPluckerToPlucker<T>(uvec, wvec, dir3d, m);
    residuals[0] = CeresComputeDist3D_sine(dir3d, vpvec);
    return true;
  }
};

////////////////////////////////////////////////////////////
// VP-VP orthogonality on 3D
////////////////////////////////////////////////////////////

struct VPOrthogonalityFunctor {
public:
  VPOrthogonalityFunctor() {}

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<VPOrthogonalityFunctor, 1, 3, 3>(
        new VPOrthogonalityFunctor());
  }

  template <typename T>
  bool operator()(const T *const vp1, const T *const vp2, T *residuals) const {
    residuals[0] = CeresComputeDist3D_cosine(vp1, vp2);
    return true;
  }
};

////////////////////////////////////////////////////////////
// VP-VP collinearity on 3D
////////////////////////////////////////////////////////////

struct VPCollinearityFunctor {
public:
  VPCollinearityFunctor() {}

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<VPCollinearityFunctor, 1, 3, 3>(
        new VPCollinearityFunctor());
  }

  template <typename T>
  bool operator()(const T *const vp1, const T *const vp2, T *residuals) const {
    residuals[0] = CeresComputeDist3D_sine(vp1, vp2);
    return true;
  }
};

} // namespace global_pl_association

} // namespace optimize

} // namespace limap

#endif
