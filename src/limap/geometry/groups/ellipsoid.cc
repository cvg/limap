#include "limap/geometry/groups/ellipsoid.h"

#include <ceres/manifold.h>
#include <ceres/product_manifold.h>
#include <colmap/util/logging.h>

#include <cmath>

#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/geometry/groups/projection.h"

namespace limap {

std::vector<double> EllipsoidGroup::GetDefaultParams2D() const {
  // 2D projection not supported for ellipsoids
  return {};
}

std::vector<double> EllipsoidGroup::GetDefaultParams3D() const {
  // Default 3D: unit sphere at origin
  // Params: [quat(4), center(3), log_scales(3)]
  // - quat = identity quaternion (x, y, z, w) = (0, 0, 0, 1)
  // - center = (0, 0, 0)
  // - log_scales = (0, 0, 0) meaning scales = (1, 1, 1)
  return {0.0, 0.0, 0.0, 1.0, // quat (identity)
          0.0, 0.0, 0.0,      // center (origin)
          0.0, 0.0, 0.0};     // log_scales (unit sphere)
}

void EllipsoidGroup::NormalizeParams3D(double *params) const {
  // Normalize quaternion to unit length
  double qx = params[0], qy = params[1], qz = params[2], qw = params[3];
  double qnorm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (qnorm > 1e-12) {
    params[0] /= qnorm;
    params[1] /= qnorm;
    params[2] /= qnorm;
    params[3] /= qnorm;
  }

  // Center (params[4:7]) and log_scales (params[7:10]) don't need normalization
}

bool EllipsoidGroup::CheckParams3D(const double *params, double tol) const {
  // Check quaternion is unit
  double qx = params[0], qy = params[1], qz = params[2], qw = params[3];
  double qnorm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
  if (std::abs(qnorm_sq - 1.0) > tol) {
    return false;
  }

  // Check all params are finite
  for (int i = 0; i < 10; ++i) {
    if (!std::isfinite(params[i])) {
      return false;
    }
  }

  return true;
}

V3D EllipsoidGroup::ProjectPoint(const V3D &point, const double *params) const {
  V3D projected;
  EllipsoidProjection::ProjectPoint(point.data(), params, projected.data());
  return projected;
}

//============================================================================
// EllipsoidGroup 3D Cost Factory Methods
//============================================================================

ceres::CostFunction *EllipsoidGroup::CreatePointCost3D() const {
  return estimators::PointToEllipsoid3DCostFunctor::Create();
}

ceres::Manifold *EllipsoidGroup::CreateManifold3D() const {
  // ProductManifold<EigenQuaternionManifold, EuclideanManifold<3>,
  //                 EuclideanManifold<3>>
  // DOF: 3 + 3 + 3 = 9
  return new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                    ceres::EuclideanManifold<3>,
                                    ceres::EuclideanManifold<3>>();
}

//============================================================================
// EllipsoidGroup 2D Point Cost Factory Methods
//============================================================================

ceres::CostFunction *
EllipsoidGroup::CreatePointCost2D(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToEllipsoid2DCostFunctor>(camera_model_id);
}

ceres::CostFunction *EllipsoidGroup::CreatePointCost2DConstantPose(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_world) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToEllipsoid2DConstantPoseCostFunctor>(camera_model_id,
                                                             cam_from_world);
}

ceres::CostFunction *EllipsoidGroup::CreatePointCost2DRig(
    colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToEllipsoid2DRigCostFunctor>(camera_model_id);
}

ceres::CostFunction *EllipsoidGroup::CreatePointCost2DConstantRig(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_rig) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToEllipsoid2DConstantRigCostFunctor>(camera_model_id,
                                                            cam_from_rig);
}

} // namespace limap
