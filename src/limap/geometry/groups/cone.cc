#include "limap/geometry/groups/cone.h"

#include <ceres/manifold.h>
#include <ceres/product_manifold.h>
#include <colmap/util/logging.h>

#include <cmath>

#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/geometry/groups/projection.h"

namespace limap {

std::vector<double> ConeGroup::GetDefaultParams2D() const {
  // 2D projection not supported for cones
  return {};
}

std::vector<double> ConeGroup::GetDefaultParams3D() const {
  // Default 3D: 45° cone at origin along +z
  // Params: [apex(3), direction(3), log_tan_alpha(1)]
  return {0.0, 0.0, 0.0, // apex (origin)
          0.0, 0.0, 1.0, // direction (+z)
          0.0};          // log_tan_alpha (tan(45°)=1, log(1)=0)
}

void ConeGroup::NormalizeParams3D(double *params) const {
  // Normalize direction vector to unit length
  double dx = params[3], dy = params[4], dz = params[5];
  double dnorm = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (dnorm > 1e-12) {
    params[3] /= dnorm;
    params[4] /= dnorm;
    params[5] /= dnorm;
  }
}

bool ConeGroup::CheckParams3D(const double *params, double tol) const {
  // Check direction is unit
  double dx = params[3], dy = params[4], dz = params[5];
  double dnorm_sq = dx * dx + dy * dy + dz * dz;
  if (std::abs(dnorm_sq - 1.0) > tol) {
    return false;
  }

  // Check all params are finite
  for (int i = 0; i < 7; ++i) {
    if (!std::isfinite(params[i])) {
      return false;
    }
  }

  return true;
}

V3D ConeGroup::ProjectPoint(const V3D &point, const double *params) const {
  V3D projected;
  ConeProjection::ProjectPoint(point.data(), params, projected.data());
  return projected;
}

//============================================================================
// ConeGroup 3D Cost Factory Methods
//============================================================================

ceres::CostFunction *ConeGroup::CreatePointCost3D() const {
  return estimators::PointToCone3DCostFunctor::Create();
}

ceres::Manifold *ConeGroup::CreateManifold3D() const {
  // ProductManifold<EuclideanManifold<3>, SphereManifold<3>,
  //                 EuclideanManifold<1>>
  // DOF: 3 + 2 + 1 = 6
  return new ceres::ProductManifold<ceres::EuclideanManifold<3>,
                                    ceres::SphereManifold<3>,
                                    ceres::EuclideanManifold<1>>();
}

//============================================================================
// ConeGroup 2D Point Cost Factory Methods
//============================================================================

ceres::CostFunction *
ConeGroup::CreatePointCost2D(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCone2DCostFunctor>(camera_model_id);
}

ceres::CostFunction *ConeGroup::CreatePointCost2DConstantPose(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_world) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCone2DConstantPoseCostFunctor>(camera_model_id,
                                                        cam_from_world);
}

ceres::CostFunction *
ConeGroup::CreatePointCost2DRig(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCone2DRigCostFunctor>(camera_model_id);
}

ceres::CostFunction *ConeGroup::CreatePointCost2DConstantRig(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_rig) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCone2DConstantRigCostFunctor>(camera_model_id,
                                                       cam_from_rig);
}

} // namespace limap
