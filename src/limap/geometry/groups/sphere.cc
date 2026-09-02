#include "limap/geometry/groups/sphere.h"

#include <ceres/manifold.h>
#include <colmap/util/logging.h>

#include <cmath>
#include <stdexcept>

#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/geometry/groups/projection.h"

namespace limap {

std::vector<double> SphereGroup::GetDefaultParams2D() const {
  // Default 2D: unit circle at origin (cx, cy, log_r)
  // log(1) = 0, so unit circle has log_r = 0
  return {0.0, 0.0, 0.0};
}

std::vector<double> SphereGroup::GetDefaultParams3D() const {
  // Default 3D: unit sphere at origin (cx, cy, cz, log_r)
  // log(1) = 0, so unit sphere has log_r = 0
  return {0.0, 0.0, 0.0, 0.0};
}

void SphereGroup::NormalizeParams3D(double *params) const {
  // No-op: log parameterization naturally handles positivity
  // Any real log_r maps to positive r = exp(log_r)
  (void)params;
}

bool SphereGroup::CheckParams3D(const double *params, double tol) const {
  // Always valid: any real log_r maps to positive r = exp(log_r)
  // Just check for NaN/Inf
  (void)tol;
  return std::isfinite(params[0]) && std::isfinite(params[1]) &&
         std::isfinite(params[2]) && std::isfinite(params[3]);
}

V3D SphereGroup::ProjectPoint(const V3D &point, const double *params) const {
  V3D projected;
  SphereProjection::ProjectPoint(point.data(), params, projected.data());
  return projected;
}

//============================================================================
// SphereGroup 3D Cost Factory Methods
//============================================================================

ceres::CostFunction *SphereGroup::CreatePointCost3D() const {
  return estimators::PointToSphere3DCostFunctor::Create();
}

ceres::Manifold *SphereGroup::CreateManifold3D() const {
  // EuclideanManifold<4> for [cx, cy, cz, r]
  // Note: This doesn't enforce r > 0, but NormalizeParams3D handles that
  return new ceres::EuclideanManifold<4>();
}

//============================================================================
// SphereGroup 2D Point Cost Factory Methods
//============================================================================

ceres::CostFunction *
SphereGroup::CreatePointCost2D(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToSphere2DCostFunctor>(camera_model_id);
}

ceres::CostFunction *SphereGroup::CreatePointCost2DConstantPose(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_world) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToSphere2DConstantPoseCostFunctor>(camera_model_id,
                                                          cam_from_world);
}

ceres::CostFunction *
SphereGroup::CreatePointCost2DRig(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToSphere2DRigCostFunctor>(camera_model_id);
}

ceres::CostFunction *SphereGroup::CreatePointCost2DConstantRig(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_rig) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToSphere2DConstantRigCostFunctor>(camera_model_id,
                                                         cam_from_rig);
}

} // namespace limap
