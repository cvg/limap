#ifndef LIMAP_CERESBASE_PARAMETERIZATION_H
#define LIMAP_CERESBASE_PARAMETERIZATION_H

// Inspired by COLMAP: https://colmap.github.io/

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace limap {

inline void SetQuaternionManifold(ceres::Problem *problem, double *qvec) {
#ifdef CERES_PARAMETERIZATION_ENABLED
  problem->SetParameterization(qvec, new ceres::QuaternionParameterization);
#else
  problem->SetManifold(qvec, new ceres::QuaternionManifold);
#endif
}

inline void SetSubsetManifold(int size, const std::vector<int> &constant_params,
                              ceres::Problem *problem, double *params) {
#ifdef CERES_PARAMETERIZATION_ENABLED
  problem->SetParameterization(
      params, new ceres::SubsetParameterization(size, constant_params));
#else
  problem->SetManifold(params,
                       new ceres::SubsetManifold(size, constant_params));
#endif
}

template <int size>
inline void SetSphereManifold(ceres::Problem *problem, double *params) {
#ifdef CERES_PARAMETERIZATION_ENABLED
  problem->SetParameterization(
      params, new ceres::HomogeneousVectorParameterization(size));
#else
  problem->SetManifold(params, new ceres::SphereManifold<size>);
#endif
}

} // namespace limap

#endif
