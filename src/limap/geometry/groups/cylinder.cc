#include "limap/geometry/groups/cylinder.h"

#include <ceres/manifold.h>
#include <ceres/product_manifold.h>
#include <ceres/sphere_manifold.h>
#include <colmap/util/logging.h>

#include <cmath>

#include "limap/estimators/bundle_adjustment/group_cost_functions.h"
#include "limap/geometry/groups/projection.h"
#include "limap/geometry/minimal_inf_line3d.h"

namespace limap {

std::vector<double> CylinderGroup::GetDefaultParams2D() const {
  // 2D projection not supported for cylinders
  return {};
}

std::vector<double> CylinderGroup::GetDefaultParams3D() const {
  // Default 3D: unit cylinder with axis along Z, passing through origin
  // Params: [quat(4), wvec(2), log_r(1)]
  // - quat = identity (axis along Z)
  // - wvec = [1, 0] (axis passes through origin, moment = 0)
  // - log_r = 0 (unit radius)

  // Identity quaternion (w, x, y, z) in Eigen storage order (x, y, z, w)
  // For axis along Z: rotation matrix col(0) should be [0, 0, 1]
  // This requires a 90-degree rotation around Y axis
  // R = Ry(90) gives col(0) = [0, 0, 1]
  // Quaternion for this: (cos(45), 0, sin(45), 0) = (sqrt(2)/2, 0, sqrt(2)/2,
  // 0)
  double sqrt2_2 = std::sqrt(2.0) / 2.0;
  return {0.0, sqrt2_2, 0.0, sqrt2_2, // quat (x, y, z, w) - 90 deg around Y
          1.0, 0.0,                   // wvec (axis through origin)
          0.0};                       // log_r = 0 (unit radius)
}

void CylinderGroup::NormalizeParams3D(double *params) const {
  // Normalize quaternion to unit length
  double qx = params[0], qy = params[1], qz = params[2], qw = params[3];
  double qnorm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (qnorm > 1e-12) {
    params[0] /= qnorm;
    params[1] /= qnorm;
    params[2] /= qnorm;
    params[3] /= qnorm;
  }

  // Normalize wvec to unit length (SO(2))
  double w0 = params[4], w1 = params[5];
  double wnorm = std::sqrt(w0 * w0 + w1 * w1);
  if (wnorm > 1e-12) {
    params[4] /= wnorm;
    params[5] /= wnorm;
  }

  // log_r (params[6]) doesn't need normalization
}

bool CylinderGroup::CheckParams3D(const double *params, double tol) const {
  // Check quaternion is unit
  double qx = params[0], qy = params[1], qz = params[2], qw = params[3];
  double qnorm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
  if (std::abs(qnorm_sq - 1.0) > tol) {
    return false;
  }

  // Check wvec is unit
  double w0 = params[4], w1 = params[5];
  double wnorm_sq = w0 * w0 + w1 * w1;
  if (std::abs(wnorm_sq - 1.0) > tol) {
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

V3D CylinderGroup::ProjectPoint(const V3D &point, const double *params) const {
  V3D projected;
  CylinderProjection::ProjectPoint(point.data(), params, projected.data());
  return projected;
}

//============================================================================
// CylinderGroup 3D Cost Factory Methods
//============================================================================

ceres::CostFunction *CylinderGroup::CreatePointCost3D() const {
  return estimators::PointToCylinder3DCostFunctor::Create();
}

ceres::Manifold *CylinderGroup::CreateManifold3D() const {
  // ProductManifold<EigenQuaternionManifold, SphereManifold<2>,
  // EuclideanManifold<1>> DOF: 3 + 1 + 1 = 5
  return new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                    ceres::SphereManifold<2>,
                                    ceres::EuclideanManifold<1>>();
}

//============================================================================
// CylinderGroup 2D Point Cost Factory Methods
//============================================================================

ceres::CostFunction *
CylinderGroup::CreatePointCost2D(colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCylinder2DCostFunctor>(camera_model_id);
}

ceres::CostFunction *CylinderGroup::CreatePointCost2DConstantPose(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_world) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCylinder2DConstantPoseCostFunctor>(camera_model_id,
                                                            cam_from_world);
}

ceres::CostFunction *CylinderGroup::CreatePointCost2DRig(
    colmap::CameraModelId camera_model_id) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCylinder2DRigCostFunctor>(camera_model_id);
}

ceres::CostFunction *CylinderGroup::CreatePointCost2DConstantRig(
    colmap::CameraModelId camera_model_id,
    const colmap::Rigid3d &cam_from_rig) const {
  return estimators::CreateSurfaceCameraCostFunction2D<
      estimators::PointToCylinder2DConstantRigCostFunctor>(camera_model_id,
                                                           cam_from_rig);
}

} // namespace limap
