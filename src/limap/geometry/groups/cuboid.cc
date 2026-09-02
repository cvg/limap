#include "limap/geometry/groups/cuboid.h"

#include <ceres/manifold.h>
#include <ceres/product_manifold.h>
#include <colmap/util/logging.h>

#include <cmath>

#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/geometry/groups/projection.h"

namespace limap {

std::vector<double> CuboidGroup::GetDefaultParams2D() const {
  // 2D projection not supported for cuboids
  return {};
}

std::vector<double> CuboidGroup::GetDefaultParams3D() const {
  // Default: identity quat + unit cube centered at origin
  // Face offsets: +x at 0.5, -x at 0.5, +y at 0.5, -y at 0.5, +z at 0.5, -z
  // at 0.5
  return {0.0, 0.0, 0.0, 1.0,            // quat (identity)
          0.5, 0.5, 0.5, 0.5, 0.5, 0.5}; // d (unit cube)
}

void CuboidGroup::NormalizeParams3D(double *params) const {
  // Normalize quaternion to unit length
  double qx = params[0], qy = params[1], qz = params[2], qw = params[3];
  double qnorm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (qnorm > 1e-12) {
    params[0] /= qnorm;
    params[1] /= qnorm;
    params[2] /= qnorm;
    params[3] /= qnorm;
  }

  // d values (params[4:10]) are unconstrained reals — no normalization needed
}

bool CuboidGroup::CheckParams3D(const double *params, double tol) const {
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

V3D CuboidGroup::ProjectPoint(const V3D &point, const double *params) const {
  V3D projected;
  CuboidProjection::ProjectPoint(point.data(), params, projected.data());
  return projected;
}

Line3d CuboidGroup::ProjectLine(const Line3d &line,
                                const double *params) const {
  V3D proj_start = ProjectPoint(line.start, params);
  V3D proj_end = ProjectPoint(line.end, params);
  return Line3d(proj_start, proj_end);
}

//============================================================================
// CuboidGroup 3D Cost Factory Methods
//============================================================================

ceres::CostFunction *CuboidGroup::CreatePointCost3D() const {
  return estimators::PointToCuboid3DCostFunctor::Create();
}

ceres::Manifold *CuboidGroup::CreateManifold3D() const {
  // ProductManifold<EigenQuaternionManifold, EuclideanManifold<6>>
  // DOF: 3 + 6 = 9
  return new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                    ceres::EuclideanManifold<6>>();
}

//============================================================================
// CuboidGroup 2D Point Cost Factory Methods
//============================================================================

ceres::CostFunction *
CuboidGroup::CreatePointCost2D(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCuboid2DCostFunctor>(camera_model_id);
}

ceres::CostFunction *CuboidGroup::CreatePointCost2DConstantPose(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_world) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCuboid2DConstantPoseCostFunctor>(camera_model_id,
                                                          cam_from_world);
}

ceres::CostFunction *
CuboidGroup::CreatePointCost2DRig(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCuboid2DRigCostFunctor>(camera_model_id);
}

ceres::CostFunction *CuboidGroup::CreatePointCost2DConstantRig(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_rig) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCuboid2DConstantRigCostFunctor>(camera_model_id,
                                                         cam_from_rig);
}

//============================================================================
// CuboidGroup 2D Line Cost Factory Methods
//============================================================================

ceres::CostFunction *
CuboidGroup::CreateLineCost2D(colmap::CameraModelId camera_model_id,
                              const Line2d &observed_2d) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::LineToCuboid2DCostFunctor>(camera_model_id, observed_2d);
}

ceres::CostFunction *CuboidGroup::CreateLineCost2DConstantPose(
    colmap::CameraModelId camera_model_id, const Line2d &observed_2d,
    const colmap::Rigid3d &cam_from_world) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::LineToCuboid2DConstantPoseCostFunctor>(
      camera_model_id, observed_2d, cam_from_world);
}

ceres::CostFunction *
CuboidGroup::CreateLineCost2DRig(colmap::CameraModelId camera_model_id,
                                 const Line2d &observed_2d) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::LineToCuboid2DRigCostFunctor>(camera_model_id, observed_2d);
}

ceres::CostFunction *CuboidGroup::CreateLineCost2DConstantRig(
    colmap::CameraModelId camera_model_id, const Line2d &observed_2d,
    const colmap::Rigid3d &cam_from_rig) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::LineToCuboid2DConstantRigCostFunctor>(
      camera_model_id, observed_2d, cam_from_rig);
}

} // namespace limap
